"""Motion proxy extractors."""

from __future__ import annotations

import numpy as np

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

