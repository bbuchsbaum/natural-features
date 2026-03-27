"""Low-level vision baselines."""

from __future__ import annotations

import numpy as np

from natural_features.core.feature_types import FeatureSeries
from natural_features.core.stimulus import VideoStimulus
from natural_features.core.timebase import TimebaseSpec
from natural_features.features.common import extractor_metadata


def _to_gray(frames: np.ndarray) -> np.ndarray:
    if frames.ndim == 3:
        return frames.astype(np.float32)
    if frames.shape[-1] == 1:
        return frames[..., 0].astype(np.float32)
    rgb = frames[..., :3].astype(np.float32)
    return 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]


def _saturation(frames: np.ndarray) -> np.ndarray:
    if frames.ndim == 3:
        return np.zeros(frames.shape[0], dtype=np.float32)
    rgb = frames[..., :3].astype(np.float32)
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    denom = np.maximum(maxc, 1e-6)
    sat = (maxc - minc) / denom
    return sat.reshape(sat.shape[0], -1).mean(axis=1)


def _edge_energy(gray_frames: np.ndarray) -> np.ndarray:
    gx = np.gradient(gray_frames, axis=2)
    gy = np.gradient(gray_frames, axis=1)
    mag = np.sqrt(gx * gx + gy * gy)
    return mag.reshape(mag.shape[0], -1).mean(axis=1)


def visual_energy(
    stimulus: VideoStimulus,
    *,
    include_deltas: bool = True,
) -> FeatureSeries:
    frames = stimulus.frames.astype(np.float32)
    gray = _to_gray(frames)

    luminance = gray.reshape(gray.shape[0], -1).mean(axis=1)
    contrast = gray.reshape(gray.shape[0], -1).std(axis=1)
    saturation = _saturation(frames)
    edge = _edge_energy(gray)

    base = np.column_stack([luminance, contrast, saturation, edge]).astype(np.float32)
    names = ["luminance", "contrast", "saturation", "edge_energy"]
    if include_deltas:
        delta = np.vstack([np.zeros((1, base.shape[1]), dtype=np.float32), np.diff(base, axis=0)])
        base = np.concatenate([base, delta], axis=1)
        names.extend([f"delta_{n}" for n in ["luminance", "contrast", "saturation", "edge_energy"]])

    params = {"include_deltas": include_deltas}
    metadata = extractor_metadata("vision.lowlevel.visual_energy", params=params)
    return FeatureSeries(
        values=base,
        times_s=stimulus.frame_times_s,
        dims=("time", "feature"),
        coords={"feature": names},
        metadata=metadata,
        timebase=TimebaseSpec(kind="frames", sampling_rate_hz=stimulus.fps),
    )

