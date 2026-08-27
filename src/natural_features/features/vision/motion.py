"""Gradient-change and optical-flow extractors."""

from __future__ import annotations

import numpy as np

from natural_features.core.backend_errors import BackendDependencyError, BackendInferenceError
from natural_features.core.execution import add_execution_provenance, resolve_execution_mode
from natural_features.core.feature_types import FeatureSeries
from natural_features.core.stimulus import VideoStimulus
from natural_features.core.timebase import TimebaseSpec
from natural_features.features.common import extractor_metadata
from natural_features.features.vision.lowlevel import _to_gray


def gradient_motion(stimulus: VideoStimulus) -> FeatureSeries:
    """Return a spatiotemporal gradient-change statistic."""
    gray = _to_gray(stimulus.frames.astype(np.float32))
    dt = np.diff(gray, axis=0)
    dx = np.gradient(gray[1:], axis=2)
    dy = np.gradient(gray[1:], axis=1)
    mag = np.sqrt(dt * dt + dx * dx + dy * dy)
    mean_mag = mag.reshape(mag.shape[0], -1).mean(axis=1)
    p95_mag = np.percentile(mag.reshape(mag.shape[0], -1), 95, axis=1)
    vals = np.column_stack([np.r_[0.0, mean_mag], np.r_[0.0, p95_mag]]).astype(np.float32)
    metadata = extractor_metadata(
        "vision.motion.gradient_change",
        params={"method": "spatiotemporal_gradient_magnitude"},
    )
    return FeatureSeries(
        values=vals,
        times_s=stimulus.frame_times_s,
        dims=("time", "feature"),
        coords={"feature": ["gradient_motion_mean", "gradient_motion_p95"]},
        metadata=metadata,
        timebase=TimebaseSpec(kind="frames", sampling_rate_hz=stimulus.fps),
    )


def optical_flow_mag(stimulus: VideoStimulus) -> FeatureSeries:
    """Compatibility alias for :func:`gradient_motion`."""

    return gradient_motion(stimulus)


def optical_flow(
    stimulus: VideoStimulus,
    *,
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
) -> FeatureSeries:
    """Return compact dense optical-flow summaries."""

    mode, _strict = resolve_execution_mode(execution_mode=execution_mode, strict_dependency=strict_dependency)
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise BackendDependencyError("OpenCV optical flow", "opencv-python is required") from exc
    gray = _to_gray(stimulus.frames.astype(np.float32))
    if gray.size and np.nanmax(gray) <= 1.0:
        gray = gray * 255.0
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    vals = np.zeros((gray.shape[0], 4), dtype=np.float32)
    try:
        for i in range(1, gray.shape[0]):
            flow = cv2.calcOpticalFlowFarneback(gray[i - 1], gray[i], None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            vals[i] = np.asarray(
                [flow[..., 0].mean(), flow[..., 1].mean(), mag.mean(), np.percentile(mag, 95)],
                dtype=np.float32,
            )
    except Exception as exc:
        raise BackendInferenceError("OpenCV optical flow", "Farneback evaluation failed") from exc
    md = add_execution_provenance(
        extractor_metadata("vision.optical_flow", params={}, extra={"backend": "opencv_farneback"}),
        execution_mode=mode,
        fallback_used=False,
    )
    return FeatureSeries(
        values=vals.astype(np.float32),
        times_s=stimulus.frame_times_s,
        dims=("time", "feature"),
        coords={"feature": ["flow_x_mean", "flow_y_mean", "flow_mag_mean", "flow_mag_p95"]},
        metadata=md,
        timebase=TimebaseSpec(kind="frames", sampling_rate_hz=stimulus.fps),
    )
