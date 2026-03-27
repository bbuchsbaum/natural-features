"""Temporal dynamics vision baselines."""

from __future__ import annotations

import numpy as np

from natural_features.core.feature_types import FeatureSeries
from natural_features.core.stimulus import VideoStimulus
from natural_features.core.timebase import TimebaseSpec
from natural_features.features.common import extractor_metadata
from natural_features.features.vision.lowlevel import _to_gray


def frame_diffs(stimulus: VideoStimulus) -> FeatureSeries:
    gray = _to_gray(stimulus.frames.astype(np.float32))
    diffs = np.abs(np.diff(gray, axis=0))
    mean_abs = diffs.reshape(diffs.shape[0], -1).mean(axis=1)
    p95_abs = np.percentile(diffs.reshape(diffs.shape[0], -1), 95, axis=1)
    vals = np.column_stack([np.r_[0.0, mean_abs], np.r_[0.0, p95_abs]]).astype(np.float32)
    metadata = extractor_metadata("vision.dynamics.frame_diffs", params={})
    return FeatureSeries(
        values=vals,
        times_s=stimulus.frame_times_s,
        dims=("time", "feature"),
        coords={"feature": ["mean_abs_diff", "p95_abs_diff"]},
        metadata=metadata,
        timebase=TimebaseSpec(kind="frames", sampling_rate_hz=stimulus.fps),
    )

