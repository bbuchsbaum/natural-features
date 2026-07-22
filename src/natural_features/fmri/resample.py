"""Resampling utilities."""

from __future__ import annotations

import numpy as np

from natural_features.core.feature_types import FeatureSeries
from natural_features.core.timebase import TimebaseSpec
from natural_features.features.common import extractor_metadata


def build_tr_grid(duration_s: float, tr_s: float, *, start_s: float = 0.0) -> np.ndarray:
    if duration_s <= 0:
        raise ValueError("duration_s must be > 0")
    if tr_s <= 0:
        raise ValueError("tr_s must be > 0")
    if start_s >= duration_s:
        return np.array([], dtype=np.float64)
    return np.arange(start_s, duration_s, tr_s, dtype=np.float64)


def _resample_mean(times: np.ndarray, values: np.ndarray, tr_grid: np.ndarray, tr_s: float) -> np.ndarray:
    out = np.zeros((len(tr_grid), values.shape[1]), dtype=np.float64)
    for i, t in enumerate(tr_grid):
        lo = t - 0.5 * tr_s
        hi = t + 0.5 * tr_s
        m = (times >= lo) & (times < hi)
        if np.any(m):
            out[i] = values[m].mean(axis=0)
        elif i > 0:
            out[i] = out[i - 1]
        else:
            out[i] = values[0]
    return out


def resample_feature_series(
    feature: FeatureSeries,
    tr_s: float,
    *,
    method: str = "mean",
    duration_s: float | None = None,
    time_grid_s: np.ndarray | None = None,
) -> FeatureSeries:
    if feature.values.ndim != 2:
        raise ValueError("resample_feature_series currently supports 2-D FeatureSeries only")
    if tr_s <= 0:
        raise ValueError("tr_s must be > 0")
    if time_grid_s is None:
        if duration_s is None:
            duration_s = float(feature.times_s[-1])
        tr_grid = build_tr_grid(duration_s=duration_s, tr_s=tr_s, start_s=float(feature.times_s[0]))
    else:
        tr_grid = np.asarray(time_grid_s, dtype=np.float64)
    if method == "mean":
        vals = _resample_mean(feature.times_s, feature.values, tr_grid, tr_s)
    elif method == "nearest":
        idx_right = np.searchsorted(feature.times_s, tr_grid, side="left")
        idx_left = np.clip(idx_right - 1, 0, len(feature.times_s) - 1)
        idx_right = np.clip(idx_right, 0, len(feature.times_s) - 1)
        left_dist = np.abs(tr_grid - feature.times_s[idx_left])
        right_dist = np.abs(feature.times_s[idx_right] - tr_grid)
        idx = np.where(right_dist < left_dist, idx_right, idx_left)
        vals = feature.values[idx]
    elif method == "linear":
        vals = np.vstack([np.interp(tr_grid, feature.times_s, feature.values[:, j]) for j in range(feature.values.shape[1])]).T
    else:
        raise ValueError(f"Unsupported resample method: {method}")
    metadata = dict(feature.metadata)
    metadata.update(
        extractor_metadata(
            "fmri.resample_to_tr",
            params={
                "tr_s": tr_s,
                "method": method,
                "explicit_grid": time_grid_s is not None,
            },
        )
    )
    return FeatureSeries(
        values=vals.astype(np.float32),
        times_s=tr_grid,
        dims=feature.dims,
        coords=feature.coords,
        metadata=metadata,
        timebase=TimebaseSpec(
            kind="windows",
            reference=feature.clock,
            stride_s=tr_s,
            window_s=tr_s,
            alignment="center",
        ),
        temporal_context=feature.temporal_context,
    )
