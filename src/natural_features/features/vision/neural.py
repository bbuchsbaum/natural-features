"""Vision neural embedding extractors with deterministic fallbacks."""

from __future__ import annotations

import numpy as np

from natural_features.core.execution import add_execution_provenance, resolve_execution_mode
from natural_features.core.feature_types import FeatureSeries
from natural_features.core.timebase import TimebaseSpec
from natural_features.features.common import extractor_metadata
from natural_features.features.vision.common import VisualStimulus, ensure_frames, frame_sampling_rate_hz, frame_times_s
from natural_features.features.vision.lowlevel import _edge_energy, _saturation, _to_gray
from natural_features.util.hashing import stable_hash


def _frame_descriptors(stimulus: VisualStimulus) -> np.ndarray:
    frames = ensure_frames(stimulus).astype(np.float32)
    gray = _to_gray(frames)
    flat = gray.reshape(gray.shape[0], -1)
    q = np.quantile(flat, [0.1, 0.5, 0.9], axis=1).T
    desc = np.column_stack(
        [
            flat.mean(axis=1),
            flat.std(axis=1),
            q,
            _saturation(frames),
            _edge_energy(gray),
        ]
    )
    return desc.astype(np.float32)


def _fallback_embedding(
    stimulus: VisualStimulus,
    *,
    extractor_name: str,
    dim: int,
    stride_frames: int,
    execution_mode: str,
    reason: str,
    params: dict[str, object],
) -> FeatureSeries:
    stride = max(1, int(stride_frames))
    desc = _frame_descriptors(stimulus)[::stride]
    seed = int(stable_hash({"extractor": extractor_name, "dim": dim}, length=8), 16) % (2**32)
    rng = np.random.default_rng(seed)
    proj = rng.normal(0.0, 1.0 / np.sqrt(desc.shape[1]), size=(desc.shape[1], dim)).astype(np.float32)
    vals = np.tanh(desc @ proj).astype(np.float32)
    md = add_execution_provenance(
        extractor_metadata(extractor_name, params=params, extra={"backend": "fallback_projection"}),
        execution_mode=execution_mode,
        fallback_used=True,
        fallback_reason=reason,
    )
    return FeatureSeries(
        values=vals,
        times_s=frame_times_s(stimulus)[::stride],
        dims=("time", "feature"),
        coords={"feature": [f"dim_{i}" for i in range(dim)]},
        metadata=md,
        timebase=TimebaseSpec(kind="frames", sampling_rate_hz=frame_sampling_rate_hz(stimulus, stride_frames=stride)),
    )


def vision_clip_embeddings(
    stimulus: VisualStimulus,
    *,
    model: str = "openai/clip-vit-base-patch32",
    stride_frames: int = 1,
    dim: int = 64,
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
) -> FeatureSeries:
    mode, strict = resolve_execution_mode(execution_mode=execution_mode, strict_dependency=strict_dependency)
    try:
        import torch  # type: ignore  # noqa: F401
        from transformers import CLIPModel, CLIPProcessor  # type: ignore  # noqa: F401
    except Exception as exc:
        if strict:
            raise RuntimeError("transformers+torch are required for strict CLIP extraction.") from exc
        return _fallback_embedding(
            stimulus,
            extractor_name="vision.clip",
            dim=dim,
            stride_frames=stride_frames,
            execution_mode=mode,
            reason="transformers/torch unavailable",
            params={"model": model, "stride_frames": stride_frames, "dim": dim},
        )
    if strict:
        raise RuntimeError("CLIP backend is not implemented yet.")
    return _fallback_embedding(
        stimulus,
        extractor_name="vision.clip",
        dim=dim,
        stride_frames=stride_frames,
        execution_mode=mode,
        reason="CLIP backend not implemented",
        params={"model": model, "stride_frames": stride_frames, "dim": dim},
    )


def vision_dino_embeddings(
    stimulus: VisualStimulus,
    *,
    model: str = "facebook/dinov2-base",
    stride_frames: int = 1,
    layers: list[int] | None = None,
    dim: int = 64,
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
) -> FeatureSeries:
    mode, strict = resolve_execution_mode(execution_mode=execution_mode, strict_dependency=strict_dependency)
    layers = layers or [6, 12]
    try:
        import torch  # type: ignore  # noqa: F401
        from transformers import AutoImageProcessor, AutoModel  # type: ignore  # noqa: F401
    except Exception as exc:
        if strict:
            raise RuntimeError("transformers+torch are required for strict DINO extraction.") from exc
        return _fallback_embedding(
            stimulus,
            extractor_name="vision.dino",
            dim=dim * len(layers),
            stride_frames=stride_frames,
            execution_mode=mode,
            reason="transformers/torch unavailable",
            params={"model": model, "stride_frames": stride_frames, "layers": layers, "dim": dim},
        )
    if strict:
        raise RuntimeError("DINO backend is not implemented yet.")
    return _fallback_embedding(
        stimulus,
        extractor_name="vision.dino",
        dim=dim * len(layers),
        stride_frames=stride_frames,
        execution_mode=mode,
        reason="DINO backend not implemented",
        params={"model": model, "stride_frames": stride_frames, "layers": layers, "dim": dim},
    )
