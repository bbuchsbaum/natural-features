"""Compatibility adapters for fmrimod interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from natural_features.core.feature_types import EventSeries, FeatureSeries
from natural_features.core.timebase import TimebaseSpec
from natural_features.features.common import extractor_metadata


def _require_fmrimod() -> Any:
    try:
        import fmrimod  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "fmrimod is required for this adapter. Install or add it to PYTHONPATH."
        ) from exc
    return fmrimod


@lru_cache(maxsize=1)
def has_fmrimod() -> bool:
    try:
        _require_fmrimod()
        return True
    except RuntimeError:
        return False


def map_hrf_name(name: str) -> str:
    mapping = {
        "glover": "spmg1",
        "spm": "spm",
    }
    return mapping.get(name, name)


@dataclass(frozen=True)
class SamplingFrameAdapter:
    sampling_frame: Any
    tr_s: float
    n_scans: int

    def grid(self) -> np.ndarray:
        return np.asarray(self.sampling_frame.sample_times(global_time=True), dtype=np.float64)


def to_sampling_frame(
    *,
    tr_s: float,
    n_scans: int | None = None,
    duration_s: float | None = None,
    precision: float = 0.1,
    start_time: float | None = None,
) -> SamplingFrameAdapter:
    if tr_s <= 0:
        raise ValueError("tr_s must be > 0")
    if n_scans is None:
        if duration_s is None or duration_s <= 0:
            raise ValueError("Provide n_scans or duration_s")
        n_scans = int(np.floor(duration_s / tr_s))
    fmrimod = _require_fmrimod()
    sf = fmrimod.sampling.SamplingFrame(n_scans=n_scans, tr=tr_s, start_time=start_time, precision=precision)
    return SamplingFrameAdapter(sampling_frame=sf, tr_s=tr_s, n_scans=n_scans)


def _event_values(events: EventSeries, *, value_mode: str) -> np.ndarray:
    if value_mode == "count":
        return np.ones(len(events), dtype=np.float64)
    if value_mode == "confidence":
        if events.confidence is None:
            raise ValueError("value_mode='confidence' requires EventSeries.confidence")
        return np.asarray(events.confidence, dtype=np.float64)
    if value_mode == "duration":
        return np.asarray(events.offset_s - events.onset_s, dtype=np.float64)
    raise ValueError(f"Unsupported value_mode: {value_mode}")


def event_series_to_fmrimod_event_variable(
    events: EventSeries,
    *,
    name: str = "event",
    value_mode: str = "count",
    center: bool = False,
    scale: bool = False,
) -> Any:
    fmrimod = _require_fmrimod()
    values = _event_values(events, value_mode=value_mode)
    durations = np.asarray(events.offset_s - events.onset_s, dtype=np.float64)
    return fmrimod.events.EventVariable(
        name=name,
        onsets=np.asarray(events.onset_s, dtype=np.float64),
        values=values,
        durations=durations,
        center=center,
        scale=scale,
    )


def render_events_with_fmrimod(
    events: EventSeries,
    *,
    tr_s: float,
    n_scans: int | None = None,
    duration_s: float | None = None,
    hrf: str = "spmg1",
    value_mode: str = "count",
    precision: float = 0.1,
) -> FeatureSeries:
    _require_fmrimod()
    from fmrimod.regressor import regressor  # type: ignore

    sf = to_sampling_frame(tr_s=tr_s, n_scans=n_scans, duration_s=duration_s, precision=precision)
    grid = sf.grid()
    amplitude = _event_values(events, value_mode=value_mode)
    durations = np.asarray(events.offset_s - events.onset_s, dtype=np.float64)
    reg = regressor(
        onsets=np.asarray(events.onset_s, dtype=np.float64),
        duration=durations,
        amplitude=amplitude,
        hrf=map_hrf_name(hrf),
    )
    pred = np.asarray(reg.evaluate(grid, precision=precision), dtype=np.float32).reshape(-1, 1)
    metadata = extractor_metadata(
        "fmri.compat.render_events_with_fmrimod",
        params={"tr_s": tr_s, "n_scans": n_scans, "duration_s": duration_s, "hrf": hrf, "value_mode": value_mode},
    )
    return FeatureSeries(
        values=pred,
        times_s=grid,
        dims=("time", "feature"),
        coords={"feature": [f"events_fmrimod_{value_mode}_{hrf}"]},
        metadata=metadata,
        timebase=TimebaseSpec(
            kind="windows",
            reference="experiment",
            stride_s=tr_s,
            window_s=tr_s,
            alignment="center",
        ),
    )
