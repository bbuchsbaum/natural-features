"""Motion proxy extractors."""

from __future__ import annotations

import numpy as np

from natural_features.core.execution import add_execution_provenance, resolve_execution_mode
from natural_features.core.feature_types import FeatureSeries
from natural_features.core.stimulus import VideoStimulus
from natural_features.core.timebase import TimebaseSpec
from natural_features.features.common import extractor_metadata
from natural_features.features.vision.lowlevel import _to_gray


def optical_flow_mag(stimulus: VideoStimulus) -> FeatureSeries:
    gray = _to_gray(stimulus.frames.astype(np.float32))
    dt = np.diff(gray, axis=0)
    dx = np.gradient(gray[1:], axis=2)
    dy = np.gradient(gray[1:], axis=1)
    mag = np.sqrt(dt * dt + dx * dx + dy * dy)
    mean_mag = mag.reshape(mag.shape[0], -1).mean(axis=1)
    p95_mag = np.percentile(mag.reshape(mag.shape[0], -1), 95, axis=1)
    vals = np.column_stack([np.r_[0.0, mean_mag], np.r_[0.0, p95_mag]]).astype(np.float32)
    metadata = extractor_metadata("vision.motion.optical_flow_mag", params={"method": "gradient_proxy"})
    return FeatureSeries(
        values=vals,
        times_s=stimulus.frame_times_s,
        dims=("time", "feature"),
        coords={"feature": ["flow_mag_mean", "flow_mag_p95"]},
        metadata=metadata,
        timebase=TimebaseSpec(kind="frames", sampling_rate_hz=stimulus.fps),
    )


def _gradient_flow_summary(stimulus: VideoStimulus) -> np.ndarray:
    gray = _to_gray(stimulus.frames.astype(np.float32))
    if gray.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)
    dt = np.diff(gray, axis=0)
    dx = np.gradient(gray[1:], axis=2)
    dy = np.gradient(gray[1:], axis=1)
    mag = np.sqrt(dt * dt + dx * dx + dy * dy)
    flat = mag.reshape(mag.shape[0], -1)
    horiz = dx.reshape(dx.shape[0], -1).mean(axis=1)
    vert = dy.reshape(dy.shape[0], -1).mean(axis=1)
    vals = np.column_stack(
        [
            np.r_[0.0, horiz],
            np.r_[0.0, vert],
            np.r_[0.0, flat.mean(axis=1)],
            np.r_[0.0, np.percentile(flat, 95, axis=1)],
        ]
    )
    return vals.astype(np.float32)


def optical_flow(
    stimulus: VideoStimulus,
    *,
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
) -> FeatureSeries:
    """Return compact dense optical-flow summaries."""

    mode, strict = resolve_execution_mode(execution_mode=execution_mode, strict_dependency=strict_dependency)
    backend = "gradient_proxy"
    fallback_used = True
    fallback_reason: str | None = "opencv unavailable"
    try:
        import cv2  # type: ignore
    except Exception as exc:
        if strict:
            raise RuntimeError("opencv-python is required for strict optical flow extraction.") from exc
        vals = _gradient_flow_summary(stimulus)
    else:
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
            backend = "opencv_farneback"
            fallback_used = False
            fallback_reason = None
        except Exception as exc:
            if strict:
                raise RuntimeError("OpenCV optical flow failed in strict mode.") from exc
            vals = _gradient_flow_summary(stimulus)
            fallback_reason = f"OpenCV optical flow failed: {type(exc).__name__}"
    md = add_execution_provenance(
        extractor_metadata("vision.optical_flow", params={}, extra={"backend": backend}),
        execution_mode=mode,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
    )
    return FeatureSeries(
        values=vals.astype(np.float32),
        times_s=stimulus.frame_times_s,
        dims=("time", "feature"),
        coords={"feature": ["flow_x_mean", "flow_y_mean", "flow_mag_mean", "flow_mag_p95"]},
        metadata=md,
        timebase=TimebaseSpec(kind="frames", sampling_rate_hz=stimulus.fps),
    )
