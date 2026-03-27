"""Timebase utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

TimebaseKind = Literal["frames", "audio_hop", "windows", "events", "tokens"]


@dataclass(frozen=True)
class TimebaseSpec:
    kind: TimebaseKind
    reference: str = "stimulus_start"
    sampling_rate_hz: float | None = None
    hop_s: float | None = None
    window_s: float | None = None
    stride_s: float | None = None
    alignment: str | None = None


def times_from_rate(
    n_samples: int,
    sampling_rate_hz: float,
    *,
    start_offset_s: float = 0.0,
) -> np.ndarray:
    if n_samples < 0:
        raise ValueError("n_samples must be >= 0")
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be > 0")
    idx = np.arange(n_samples, dtype=np.float64)
    return start_offset_s + (idx / float(sampling_rate_hz))


def times_from_hop(
    n_steps: int,
    hop_s: float,
    *,
    start_offset_s: float = 0.0,
    center: bool = False,
    window_s: float | None = None,
) -> np.ndarray:
    if n_steps < 0:
        raise ValueError("n_steps must be >= 0")
    if hop_s <= 0:
        raise ValueError("hop_s must be > 0")
    shift = 0.0
    if center and window_s is not None:
        if window_s <= 0:
            raise ValueError("window_s must be > 0 when center=True")
        shift = window_s / 2.0
    return start_offset_s + shift + (np.arange(n_steps, dtype=np.float64) * hop_s)


def is_monotonic_non_decreasing(times_s: np.ndarray) -> bool:
    if times_s.ndim != 1:
        return False
    if len(times_s) < 2:
        return True
    return bool(np.all(np.diff(times_s) >= 0))

