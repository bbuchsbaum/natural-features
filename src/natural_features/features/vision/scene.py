"""Scene cut detection baseline."""

from __future__ import annotations

import numpy as np

from natural_features.core.feature_types import EventSeries
from natural_features.core.stimulus import VideoStimulus
from natural_features.features.common import extractor_metadata
from natural_features.features.vision.lowlevel import _to_gray


def scene_cuts(
    stimulus: VideoStimulus,
    *,
    threshold_z: float = 2.5,
) -> EventSeries:
    gray = _to_gray(stimulus.frames.astype(np.float32))
    diffs = np.abs(np.diff(gray, axis=0)).reshape(gray.shape[0] - 1, -1).mean(axis=1)
    mu, sigma = float(diffs.mean()), float(diffs.std() + 1e-8)
    z = (diffs - mu) / sigma
    hit_idx = np.where(z >= threshold_z)[0] + 1
    onset = stimulus.frame_times_s[hit_idx]
    offset = onset.copy()
    confidence = np.clip(z[hit_idx - 1], a_min=0.0, a_max=None)
    metadata = extractor_metadata("meta.scene_cuts", params={"threshold_z": threshold_z})
    return EventSeries(
        onset_s=onset,
        offset_s=offset,
        label=np.array(["cut"] * len(onset), dtype=object),
        confidence=confidence.astype(np.float32),
        metadata=metadata,
    )

