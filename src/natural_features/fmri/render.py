"""Event rendering utilities."""

from __future__ import annotations

import numpy as np

from natural_features.core.feature_types import EventSeries, FeatureSeries
from natural_features.core.timebase import TimebaseSpec
from natural_features.features.common import extractor_metadata


def _grid_edges(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(grid) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    if len(grid) == 1:
        return np.array([grid[0] - 0.5], dtype=np.float64), np.array([grid[0] + 0.5], dtype=np.float64)
    if np.any(np.diff(grid) <= 0):
        raise ValueError("time_grid_s must be strictly increasing")
    mids = 0.5 * (grid[:-1] + grid[1:])
    left = np.empty_like(grid)
    right = np.empty_like(grid)
    left[1:] = mids
    right[:-1] = mids
    left[0] = grid[0] - (mids[0] - grid[0])
    right[-1] = grid[-1] + (grid[-1] - mids[-1])
    return left, right


def render_events(
    events: EventSeries,
    time_grid_s: np.ndarray,
    *,
    mode: str = "impulse",
    value: str = "count",
) -> FeatureSeries:
    grid = np.asarray(time_grid_s, dtype=np.float64)
    out = np.zeros((len(grid), 1), dtype=np.float32)
    if len(events) == 0:
        metadata = extractor_metadata("fmri.render_events", params={"mode": mode, "value": value})
        return FeatureSeries(
            values=out,
            times_s=grid,
            dims=("time", "feature"),
            coords={"feature": [f"events_{mode}_{value}"]},
            metadata=metadata,
            timebase=TimebaseSpec(kind="windows"),
        )

    left_edges, right_edges = _grid_edges(grid)
    for i in range(len(grid)):
        lo = left_edges[i]
        hi = right_edges[i]
        if mode == "impulse":
            m = (events.onset_s >= lo) & (events.onset_s < hi)
            if value == "count":
                out[i, 0] = float(np.sum(m))
            elif value == "confidence":
                if events.confidence is None:
                    raise ValueError("value='confidence' requires EventSeries.confidence")
                out[i, 0] = float(np.sum(events.confidence[m]))
            else:
                raise ValueError(f"Unsupported value for impulse mode: {value}")
        elif mode == "boxcar":
            overlaps = np.maximum(
                0.0,
                np.minimum(events.offset_s, hi) - np.maximum(events.onset_s, lo),
            )
            if value == "duration":
                out[i, 0] = float(np.sum(overlaps))
            elif value == "count":
                out[i, 0] = float(np.sum(overlaps > 0))
            elif value == "confidence":
                if events.confidence is None:
                    raise ValueError("value='confidence' requires EventSeries.confidence")
                out[i, 0] = float(np.sum((overlaps > 0) * events.confidence))
            else:
                raise ValueError(f"Unsupported value for boxcar mode: {value}")
        else:
            raise ValueError(f"Unsupported mode: {mode}")
    metadata = extractor_metadata("fmri.render_events", params={"mode": mode, "value": value})
    return FeatureSeries(
        values=out,
        times_s=grid,
        dims=("time", "feature"),
        coords={"feature": [f"events_{mode}_{value}"]},
        metadata=metadata,
        timebase=TimebaseSpec(kind="windows"),
    )
