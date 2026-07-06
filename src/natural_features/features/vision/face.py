"""Face-detection style visual features."""

from __future__ import annotations

import numpy as np

from natural_features.core.execution import add_execution_provenance, resolve_execution_mode
from natural_features.core.feature_types import FeatureSeries
from natural_features.core.timebase import TimebaseSpec
from natural_features.features.common import extractor_metadata
from natural_features.features.vision.common import VisualStimulus, ensure_frames, frame_sampling_rate_hz, frame_times_s
from natural_features.features.vision.lowlevel import _saturation, _to_gray


def _fallback_face_proxy(stimulus: VisualStimulus, *, execution_mode: str, reason: str) -> FeatureSeries:
    frames = ensure_frames(stimulus).astype(np.float32)
    gray = _to_gray(frames)
    sat = _saturation(frames)
    luminance = gray.reshape(gray.shape[0], -1).mean(axis=1)
    contrast = gray.reshape(gray.shape[0], -1).std(axis=1)
    presence = np.clip((sat * 1.5) + (contrast > 0.05).astype(np.float32) * 0.1, 0.0, 1.0)
    count = (presence > 0.25).astype(np.float32)
    center_x = np.full_like(presence, 0.5, dtype=np.float32)
    center_y = np.full_like(presence, 0.5, dtype=np.float32)
    vals = np.column_stack([presence, count, luminance, center_x, center_y]).astype(np.float32)
    md = add_execution_provenance(
        extractor_metadata(
            "vision.face",
            params={},
            extra={"backend": "fallback_proxy"},
        ),
        execution_mode=execution_mode,
        fallback_used=True,
        fallback_reason=reason,
    )
    return FeatureSeries(
        values=vals,
        times_s=frame_times_s(stimulus),
        dims=("time", "feature"),
        coords={"feature": ["face_presence", "face_count", "face_luminance", "face_center_x", "face_center_y"]},
        metadata=md,
        timebase=TimebaseSpec(kind="frames", sampling_rate_hz=frame_sampling_rate_hz(stimulus)),
    )


def face_detection(
    stimulus: VisualStimulus,
    *,
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
) -> FeatureSeries:
    mode, strict = resolve_execution_mode(execution_mode=execution_mode, strict_dependency=strict_dependency)
    try:
        import mediapipe  # type: ignore  # noqa: F401
    except Exception as exc:
        if strict:
            raise RuntimeError("mediapipe is required for strict face detection.") from exc
        return _fallback_face_proxy(stimulus, execution_mode=mode, reason="mediapipe unavailable")

    if strict:
        raise RuntimeError("mediapipe face detection backend is not implemented yet.")
    return _fallback_face_proxy(stimulus, execution_mode=mode, reason="mediapipe backend not implemented")
