"""Visual social/affect proxy features."""

from __future__ import annotations

import numpy as np

from natural_features.core.feature_types import FeatureSeries
from natural_features.core.stimulus import VideoStimulus
from natural_features.core.timebase import TimebaseSpec
from natural_features.features.common import extractor_metadata
from natural_features.features.vision.lowlevel import _to_gray
from natural_features.features.vision.motion import optical_flow_mag


def social_visual_proxies(stimulus: VideoStimulus) -> FeatureSeries:
    frames = stimulus.frames.astype(np.float32)
    gray = _to_gray(frames)
    luminance = gray.reshape(gray.shape[0], -1).mean(axis=1)
    if frames.ndim == 4 and frames.shape[-1] >= 3:
        rgb = frames[..., :3]
        maxc = rgb.max(axis=-1)
        minc = rgb.min(axis=-1)
        sat = ((maxc - minc) / np.maximum(maxc, 1e-6)).reshape(maxc.shape[0], -1).mean(axis=1)
    else:
        sat = np.zeros_like(luminance)
    motion = optical_flow_mag(stimulus).values[:, 0]
    face_presence_proxy = np.clip(sat * 1.5, 0.0, 1.0)
    social_intensity_proxy = np.clip(face_presence_proxy * (motion / (motion.max() + 1e-8)), 0.0, 1.0)
    vals = np.column_stack([luminance, face_presence_proxy, social_intensity_proxy]).astype(np.float32)
    md = extractor_metadata("affect.visual.social_proxies", params={})
    return FeatureSeries(
        values=vals,
        times_s=stimulus.frame_times_s,
        dims=("time", "feature"),
        coords={"feature": ["luminance", "face_presence_proxy", "social_intensity_proxy"]},
        metadata=md,
        timebase=TimebaseSpec(kind="frames", sampling_rate_hz=stimulus.fps),
    )

