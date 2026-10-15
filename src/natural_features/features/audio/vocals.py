"""Optional, explicit source separation and monophonic source-pitch primitives.

A model's ``vocals`` output is an estimated source, not a vocal-presence label.
CREPE periodicity is not a calibrated probability of singing. Callers must
qualify domain support and learn any decision thresholds independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from natural_features.core.feature_types import FeatureSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.core.timebase import TimebaseSpec
from natural_features.features.common import extractor_metadata


@dataclass(frozen=True)
class SeparatedAudio:
    """Same-clock estimated waveforms and explicit model/input provenance."""

    sources: dict[str, AudioStimulus]
    metadata: dict[str, Any]


def _input(stimulus: AudioStimulus, rate: int) -> np.ndarray:
    if stimulus.sr_hz != rate:
        raise ValueError(f"Explicitly resample to {rate} Hz before extraction")
    x = np.asarray(stimulus.samples, dtype=np.float32)
    if not np.isfinite(stimulus.start_offset_s) or not np.all(np.isfinite(x)):
        raise ValueError("Audio samples and offset must be finite")
    if x.ndim == 2 and x.shape[1] not in (1, 2):
        raise ValueError("Require mono or stereo audio")
    return x


def _digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def separate_vocals(
    stimulus: AudioStimulus,
    *,
    model_repo: str | Path,
    model_name: str = "htdemucs",
    segment_s: float = 7.8,
    overlap: float = 0.25,
    device: str = "cpu",
) -> SeparatedAudio:
    """Separate with local Demucs weights; never download or silently resample.

    Preserves the sample count, offset and named clock. Inference uses the
    complete input with overlapping segments and zero random shifts. Every
    output can depend on future samples through the separator; its waveform
    sampling interval must not be interpreted as the model's context width.
    ``model_repo`` must be a dedicated directory of pinned .th/.yaml files.
    """
    x = _input(stimulus, 44100)
    if not np.isfinite(segment_s) or segment_s <= 0:
        raise ValueError("segment_s must be positive and finite")
    if not np.isfinite(overlap) or not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0,1)")
    if not model_name or any(c in model_name for c in ("/", "\\", ":")):
        raise ValueError("model_name must be a local Demucs signature or bag name")
    repo = Path(model_repo)
    files = sorted(p for p in repo.iterdir() if p.suffix in {".th", ".yaml"})
    if not any(p.suffix == ".th" for p in files):
        raise ValueError("model_repo must contain prefetched Demucs .th weights")
    try:
        import torch
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
    except ImportError as exc:
        raise RuntimeError("Install natural-features[vocals] for Demucs") from exc
    model = get_model(model_name, repo=repo).to(device).eval()
    if model.samplerate != 44100 or model.audio_channels != 2:
        raise ValueError("This adapter requires a 44.1kHz stereo Demucs model")
    if "vocals" not in model.sources:
        raise ValueError("The selected model has no vocals source")
    stereo = x[:, None] if x.ndim == 1 else x
    stereo = np.repeat(stereo, 2, axis=1) if stereo.shape[1] == 1 else stereo
    wav = torch.from_numpy(stereo.copy().T)
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std()
    # Dual-mono cancellation can give a zero normalization scale for nonzero
    # antiphase inputs. Reject it instead of discarding the input silently.
    if not torch.isfinite(std) or float(std) <= 1e-8:
        if np.max(np.abs(stereo)) > 1e-8:
            raise ValueError("Degenerate Demucs normalization; inspect stereo phase")
        out = np.zeros((len(model.sources), 2, len(x)), dtype=np.float32)
    else:
        with torch.inference_mode():
            out = (
                apply_model(
                    model,
                    ((wav - mean) / std)[None],
                    device=device,
                    shifts=0,
                    split=True,
                    overlap=overlap,
                    segment=segment_s,
                    progress=False,
                    num_workers=0,
                )[0].cpu()
                * std
                + mean
            )
        out = out.numpy()
    if out.shape != (len(model.sources), 2, len(x)) or not np.all(np.isfinite(out)):
        raise RuntimeError("Demucs returned unexpected shape or nonfinite samples")
    weights = {p.name: _digest(p) for p in files}
    md = extractor_metadata(
        "audio.vocal_separation",
        model_revision=sha256(repr(weights).encode()).hexdigest(),
        params={
            "model_name": model_name,
            "segment_s": segment_s,
            "overlap": overlap,
            "shifts": 0,
            "device": device,
            "normalization": "demucs_reference_mean_std",
        },
        extra={
            "backend": "demucs",
            "backend_version": version("demucs"),
            "model_files_sha256": weights,
            "input_sha256": sha256(x.tobytes()).hexdigest(),
            "source_scope": "estimated_stems_not_vocal_presence_labels",
            "future_access": True,
            "context": "overlapping_full_input_segments",
            "sr_hz": 44100,
            "input_samples": len(x),
            "sources": list(model.sources),
        },
    )
    return SeparatedAudio(
        {
            name: AudioStimulus(
                samples=out[i].T,
                sr_hz=44100,
                start_offset_s=stimulus.start_offset_s,
                source=stimulus.source,
                clock=stimulus.clock,
                temporal_context=stimulus.temporal_context,
            )
            for i, name in enumerate(model.sources)
        },
        md,
    )


def source_pitch_crepe(
    stimulus: AudioStimulus,
    *,
    source_role: str,
    source_provenance: dict[str, Any],
    model: str = "full",
    hop_s: float = 0.02,
    fmin: float = 50.0,
    fmax: float = 1000.0,
    batch_size: int = 64,
    device: str = "cpu",
) -> FeatureSeries:
    """Raw CREPE source pitch, periodicity and RMS on unpadded native frames.

    Requires explicit 16kHz mono audio and source provenance. Uses independent
    weighted-argmax decoding, avoiding batch-dependent Viterbi transitions.
    No voiced/vocal threshold is silently applied. Each 1024-sample row has
    64ms support; source preprocessing can have wider, noncausal context.
    """
    x = _input(stimulus, 16000)
    if x.ndim == 2:
        if x.shape[1] != 1:
            raise ValueError("Explicitly downmix before source pitch extraction")
        x = x[:, 0]
    if (
        not source_role
        or not isinstance(source_provenance, dict)
        or not source_provenance
    ):
        raise ValueError("Explicit source_role and nonempty source_provenance required")
    if model not in {"full", "tiny"}:
        raise ValueError("CREPE model must be full or tiny")
    if not np.isfinite(hop_s) or hop_s <= 0 or batch_size < 1:
        raise ValueError("Require positive finite hop_s and batch_size")
    if not (np.isfinite(fmin) and np.isfinite(fmax) and 32 <= fmin < fmax <= 2006):
        raise ValueError("Require 32 <= fmin < fmax <= 2006 Hz")
    if len(x) < 1024:
        raise ValueError("Require at least one full 1024-sample CREPE frame")
    hop = max(1, round(hop_s * 16000))
    try:
        import torch
        import torchcrepe
    except ImportError as exc:
        raise RuntimeError("Install natural-features[vocals] for torchcrepe") from exc
    with torch.inference_mode():
        pitch, periodicity = torchcrepe.predict(
            torch.from_numpy(x.copy())[None],
            16000,
            hop_length=hop,
            fmin=fmin,
            fmax=fmax,
            model=model,
            decoder=torchcrepe.decode.weighted_argmax,
            return_periodicity=True,
            batch_size=batch_size,
            device=device,
            pad=False,
        )
    starts = np.arange(1 + (len(x) - 1024) // hop, dtype=np.int64) * hop
    p, q = pitch.cpu().numpy().reshape(-1), periodicity.cpu().numpy().reshape(-1)
    if (
        len(p) != len(starts)
        or len(q) != len(starts)
        or not np.all(np.isfinite([p, q]))
    ):
        raise RuntimeError("CREPE output does not match its native frame geometry")
    energy = np.array(
        [np.sqrt(np.mean(x[i : i + 1024].astype(np.float64) ** 2)) for i in starts]
    )
    weights = Path(torchcrepe.__file__).parent / "assets" / f"{model}.pth"
    weight_hash = _digest(weights)
    bounds = stimulus.start_offset_s + np.column_stack([starts, starts + 1024]) / 16000
    md = extractor_metadata(
        "audio.source_pitch_crepe",
        model_revision=weight_hash,
        params={
            "model": model,
            "hop_samples": hop,
            "fmin": fmin,
            "fmax": fmax,
            "decoder": "weighted_argmax",
            "padding": False,
            "device": device,
        },
        extra={
            "backend": "torchcrepe",
            "backend_version": version("torchcrepe"),
            "weight_sha256": weight_hash,
            "source_role": source_role,
            "source_provenance": source_provenance,
            "input_sha256": sha256(x.tobytes()).hexdigest(),
            "pitch_scope": "monophonic_source_not_vocal_detection",
            "periodicity_scope": "uncalibrated_network_confidence",
            "voicing_threshold": None,
            "silence_gate": None,
            "context": "64ms_pitch_frames_plus_upstream_source_context",
            "source_context_inherited": True,
        },
    )
    return FeatureSeries(
        values=np.column_stack([p, q, energy]).astype(np.float32),
        times_s=bounds.mean(axis=1),
        coords={"feature": ["f0_hz", "periodicity", "rms"]},
        metadata=md,
        timebase=TimebaseSpec(
            kind="audio_hop",
            reference=stimulus.clock,
            hop_s=hop / 16000,
            window_s=1024 / 16000,
        ),
        time_bounds_s=bounds,
        temporal_context=stimulus.temporal_context,
    )
