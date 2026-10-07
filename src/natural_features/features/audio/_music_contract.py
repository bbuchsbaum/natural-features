"""Native music feature bookkeeping, independent of optional model libraries."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from natural_features.core.feature_types import FeatureSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.core.timebase import SupportSpec, TimebaseSpec
from natural_features.features.common import extractor_metadata


def mono_audio(stimulus: AudioStimulus, *, channel_policy: str = "mean") -> np.ndarray:
    """Validate finite floating audio; channel averaging is an explicit policy."""
    if channel_policy not in {"mean", "mono"}:
        raise ValueError("channel_policy must be 'mean' or 'mono'")
    sr = stimulus.sr_hz
    if isinstance(sr, bool) or not np.isfinite(sr) or int(sr) != sr or sr <= 0:
        raise ValueError("sample rate must be a positive integer")
    if not np.isfinite(stimulus.start_offset_s):
        raise ValueError("start_offset_s must be finite")
    x = np.asarray(stimulus.samples)
    if not np.issubdtype(x.dtype, np.floating):
        raise ValueError("audio must contain floating-point samples; scale integer PCM explicitly")
    if not np.all(np.isfinite(x)) or x.size == 0:
        raise ValueError("audio must contain finite samples and at least one channel")
    if x.ndim == 2:
        if channel_policy == "mono" and x.shape[1] != 1:
            raise ValueError("channel_policy='mono' requires one channel")
        x = x.mean(axis=1, dtype=np.float64)
    x = x.astype(np.float32)
    if not np.all(np.isfinite(x)):
        raise ValueError("audio exceeds finite float32 range")
    return x


def positive_seconds(value: float, name: str) -> float:
    if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and > 0")
    return float(value)


def computation_context(
    stimulus: AudioStimulus,
    *,
    policy: str,
    bounds_s: np.ndarray | list[list[float]],
    preprocessing: dict[str, Any],
) -> dict[str, Any]:
    """Clock-tagged source dependencies, never implicitly relabeled by in_clock.

    Bounds are either one interval shared by all output rows or one per row.
    They describe real input dependencies; observation support is separate.
    """
    if policy not in {"local", "past_only", "bidirectional"}:
        raise ValueError("unknown computation context policy")
    bounds = np.asarray(bounds_s, dtype=np.float64)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("context bounds must have shape (1 or n_time, 2)")
    if not np.all(np.isfinite(bounds)) or np.any(bounds[:, 1] < bounds[:, 0]):
        raise ValueError("context bounds must be finite ordered intervals")
    start = float(stimulus.start_offset_s)
    end = start + len(stimulus.samples) / float(stimulus.sr_hz)
    if np.any(bounds[:, 0] < start - 1e-10) or np.any(bounds[:, 1] > end + 1e-10):
        raise ValueError("context bounds exceed supplied audio")
    return {
        "schema": "MusicComputationContext/v1",
        "clock": str(stimulus.clock),
        "policy": policy,
        "input_bounds_s": [start, end],
        "dependency_bounds_s": bounds.tolist(),
        "preprocessing": dict(preprocessing),
    }


def asset_digest(path: str | Path) -> str:
    """Hash a checkpoint file or model snapshot, including config and source.

    Hugging Face snapshot symlinks are followed. Local edits to weights or code
    change identity; the pathname is not treated as a revision.
    """
    root = Path(path)
    if root.is_file():
        files = [root]
    elif root.is_dir():
        files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in {
            ".json", ".py", ".bin", ".safetensors", ".pt", ".pth", ".ckpt",
        })
    else:
        raise FileNotFoundError(f"model asset does not exist: {root}")
    if not files:
        raise ValueError("model snapshot has no configuration, code or weight assets")
    digest = hashlib.sha256()
    for file in files:
        name = file.relative_to(root).as_posix() if root.is_dir() else "checkpoint"
        digest.update(name.encode() + b"\0")
        file_digest = hashlib.sha256()
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                file_digest.update(block)
        digest.update(file_digest.digest())
    return "sha256:" + digest.hexdigest()


def music_series(
    stimulus: AudioStimulus,
    *,
    values: np.ndarray,
    times_s: np.ndarray,
    bounds_s: np.ndarray,
    names: list[str] | None = None,
    extractor: str,
    params: dict[str, Any],
    context: dict[str, Any],
    hop_s: float,
    dims: tuple[str, ...] = ("time", "feature"),
    coords: dict[str, list[Any]] | None = None,
    model_revision: str = "none",
    extra: dict[str, Any] | None = None,
    alignment: str = "center",
) -> FeatureSeries:
    """Construct a native series without conflating support and dependencies."""
    values = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError("music feature values must be finite; use explicit validity diagnostics")
    n = len(times_s)
    dependency = np.asarray(context["dependency_bounds_s"]).reshape(-1, 2)
    if len(dependency) not in {1, n}:
        raise ValueError("context must have one interval or one interval per output row")
    if context["policy"] == "past_only" and np.any(dependency[:, 1] > times_s + 1e-10):
        raise ValueError("past-only context extends beyond an output timestamp")
    return FeatureSeries(
        values=values,
        times_s=times_s,
        dims=dims,
        coords=coords if coords is not None else {"feature": names or []},
        metadata=extractor_metadata(
            extractor, params=params, code_version="music-v1", model_revision=model_revision,
            extra={"computation_context": context, **(extra or {})},
        ),
        timebase=TimebaseSpec(
            kind="audio_hop", reference=stimulus.clock, hop_s=hop_s,
            sampling_rate_hz=1.0 / hop_s, alignment=alignment,
            support=SupportSpec(kind="interval", anchor=alignment),
        ),
        time_bounds_s=bounds_s,
        temporal_context=stimulus.temporal_context,
    )
