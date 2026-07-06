"""Spatial DCT vision features."""

from __future__ import annotations

import math

import numpy as np

from natural_features.core.feature_types import FeatureSeries
from natural_features.core.timebase import TimebaseSpec
from natural_features.features.common import extractor_metadata
from natural_features.features.vision.common import VisualStimulus, ensure_frames, frame_sampling_rate_hz, frame_times_s
from natural_features.features.vision.lowlevel import _to_gray


def _dct_basis(n: int, k: int) -> np.ndarray:
    x = np.arange(n, dtype=np.float32)[None, :]
    u = np.arange(k, dtype=np.float32)[:, None]
    basis = np.sqrt(2.0 / n) * np.cos((math.pi / n) * (x + 0.5) * u)
    basis[0, :] *= 1.0 / np.sqrt(2.0)
    return basis.astype(np.float32)


def _lowfreq_pairs(k: int, side: int) -> list[tuple[int, int]]:
    pairs = [(u, v) for u in range(side) for v in range(side)]
    pairs.sort(key=lambda uv: (uv[0] + uv[1], uv[0]))
    return pairs[:k]


def _sample_frame_grid(gray: np.ndarray, size: int) -> np.ndarray:
    y = np.linspace(0, gray.shape[1] - 1, size).round().astype(int)
    x = np.linspace(0, gray.shape[2] - 1, size).round().astype(int)
    return gray[:, y][:, :, x].astype(np.float32)


def vision_dct_features(
    stimulus: VisualStimulus,
    *,
    k: int = 64,
    color: str = "gray",
    size: int = 32,
) -> FeatureSeries:
    """Return low-frequency spatial DCT coefficients for each frame."""

    if k <= 0:
        raise ValueError("k must be > 0")
    if color != "gray":
        raise ValueError("Only color='gray' is currently supported")
    frames = ensure_frames(stimulus).astype(np.float32)
    gray = _to_gray(frames)
    small = _sample_frame_grid(gray, max(4, int(size)))
    side = small.shape[1]
    basis = _dct_basis(side, side)
    pairs = _lowfreq_pairs(k, side)
    values = np.zeros((small.shape[0], len(pairs)), dtype=np.float32)
    for i, frame in enumerate(small):
        coeff = basis @ frame @ basis.T
        values[i] = np.asarray([coeff[u, v] for u, v in pairs], dtype=np.float32)
    md = extractor_metadata(
        "vision.dct",
        params={"k": k, "color": color, "size": size},
        extra={"backend": "numpy_dct"},
    )
    return FeatureSeries(
        values=values,
        times_s=frame_times_s(stimulus),
        dims=("time", "feature"),
        coords={"feature": [f"dct_{u}_{v}" for u, v in pairs]},
        metadata=md,
        timebase=TimebaseSpec(kind="frames", sampling_rate_hz=frame_sampling_rate_hz(stimulus)),
    )
