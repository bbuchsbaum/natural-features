"""Semantic view event extractors."""

from __future__ import annotations

import numpy as np

from natural_features.core.execution import add_execution_provenance, resolve_execution_mode
from natural_features.core.feature_types import EventSeries
from natural_features.core.stimulus import VideoStimulus
from natural_features.core.timebase import TimebaseSpec
from natural_features.features.common import extractor_metadata
from natural_features.features.vision.common import ensure_frames, frame_duration_s, frame_times_s
from natural_features.features.vision.lowlevel import _edge_energy, _saturation, _to_gray


def _semantic_labels(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = _to_gray(frames)
    flat = gray.reshape(gray.shape[0], -1)
    luminance = flat.mean(axis=1)
    contrast = flat.std(axis=1)
    saturation = _saturation(frames)
    edge = _edge_energy(gray)
    labels = []
    confidence = []
    lum_mid = float(np.median(luminance)) if luminance.size else 0.0
    contrast_mid = float(np.median(contrast)) if contrast.size else 0.0
    for lum, con, sat, edg in zip(luminance, contrast, saturation, edge, strict=False):
        if con > contrast_mid * 1.25 and edg > np.median(edge):
            labels.append("structured_scene")
            confidence.append(float(np.clip(con / (contrast_mid + 1e-6), 0.0, 1.0)))
        elif sat > 0.35:
            labels.append("colorful_scene")
            confidence.append(float(np.clip(sat, 0.0, 1.0)))
        elif lum >= lum_mid:
            labels.append("bright_scene")
            confidence.append(float(np.clip(lum / (lum_mid + 1e-6), 0.0, 1.0)))
        else:
            labels.append("dark_scene")
            confidence.append(float(np.clip(1.0 - lum / (lum_mid + 1e-6), 0.0, 1.0)))
    return np.asarray(labels, dtype=object), np.asarray(confidence, dtype=np.float32)


def vision_semantic_views(
    stimulus: VideoStimulus,
    *,
    stride_frames: int = 1,
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
) -> EventSeries:
    """Return frame-aligned semantic view labels."""

    mode, strict = resolve_execution_mode(execution_mode=execution_mode, strict_dependency=strict_dependency)
    if strict:
        raise RuntimeError("strict semantic view extraction requires a configured external vision-language backend.")
    stride = max(1, int(stride_frames))
    frames = ensure_frames(stimulus)[::stride].astype(np.float32)
    labels, confidence = _semantic_labels(frames)
    onset = frame_times_s(stimulus)[::stride]
    dur = frame_duration_s(stimulus, stride_frames=stride) or 0.0
    md = add_execution_provenance(
        extractor_metadata("vision.semantic_views", params={"stride_frames": stride}, extra={"backend": "scene_proxy"}),
        execution_mode=mode,
        fallback_used=True,
        fallback_reason="external semantic view backend unavailable",
    )
    return EventSeries(
        onset_s=onset,
        offset_s=onset + float(dur),
        label=labels,
        confidence=confidence,
        extra={"view_type": labels.copy()},
        metadata=md,
        timebase=TimebaseSpec(kind="frames", sampling_rate_hz=stimulus.fps / stride),
    )
