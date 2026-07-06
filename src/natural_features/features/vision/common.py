"""Shared visual stimulus helpers."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np

from natural_features.core.stimulus import ImageStimulus, VideoStimulus

VisualStimulus: TypeAlias = ImageStimulus | VideoStimulus


def ensure_frames(stimulus: VisualStimulus) -> np.ndarray:
    if isinstance(stimulus, ImageStimulus):
        return stimulus.as_frames()
    if isinstance(stimulus, VideoStimulus):
        return stimulus.frames
    raise TypeError("stimulus must be an ImageStimulus or VideoStimulus")


def frame_times_s(stimulus: VisualStimulus) -> np.ndarray:
    return stimulus.frame_times_s


def frame_sampling_rate_hz(stimulus: VisualStimulus, *, stride_frames: int = 1) -> float | None:
    stride = max(1, int(stride_frames))
    if isinstance(stimulus, ImageStimulus):
        return None
    return float(stimulus.fps) / stride


def frame_duration_s(stimulus: VisualStimulus, *, stride_frames: int = 1) -> float | None:
    stride = max(1, int(stride_frames))
    if isinstance(stimulus, ImageStimulus):
        return stimulus.duration_s
    return stride / float(stimulus.fps)
