"""Canonical feature object contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .timebase import (
    ClockRef,
    TemporalContext,
    TimebaseSpec,
    is_monotonic_non_decreasing,
    validate_time_bounds,
)

REQUIRED_METADATA_FIELDS = {
    "extractor_id",
    "params_hash",
}


def _ensure_1d_float_array(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


@dataclass(frozen=True)
class FeatureSeries:
    values: np.ndarray
    times_s: np.ndarray
    dims: tuple[str, ...] = ("time", "feature")
    coords: dict[str, list[Any]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema: str = "FeatureSeries/v1"
    timebase: TimebaseSpec = field(default_factory=lambda: TimebaseSpec(kind="frames"))
    time_bounds_s: np.ndarray | None = None
    temporal_context: TemporalContext = field(default_factory=TemporalContext)

    def __post_init__(self) -> None:
        if not isinstance(self.timebase, TimebaseSpec):
            raise TypeError("timebase must be a TimebaseSpec")
        values = np.asarray(self.values)
        times_s = _ensure_1d_float_array(self.times_s, "times_s")
        if values.ndim < 2:
            raise ValueError("FeatureSeries.values must have at least 2 dims (time + features)")
        if values.shape[0] != times_s.shape[0]:
            raise ValueError("values first dimension must match times_s length")
        if self.dims[0] != "time":
            raise ValueError("dims[0] must be 'time'")
        if len(self.dims) != values.ndim:
            raise ValueError("len(dims) must match values.ndim")
        if not is_monotonic_non_decreasing(times_s):
            raise ValueError("times_s must be monotonic non-decreasing")
        missing = sorted(REQUIRED_METADATA_FIELDS - set(self.metadata.keys()))
        if missing:
            raise ValueError(f"metadata missing required fields: {missing}")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "times_s", times_s)
        if self.time_bounds_s is not None:
            object.__setattr__(
                self,
                "time_bounds_s",
                validate_time_bounds(self.time_bounds_s, len(times_s)),
            )
        elif self.timebase.support.kind == "interval":
            raise ValueError("interval support requires time_bounds_s")
        if not isinstance(self.temporal_context, TemporalContext):
            object.__setattr__(
                self,
                "temporal_context",
                TemporalContext.from_dict(self.temporal_context),
            )

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.values.shape)

    @property
    def clock(self) -> ClockRef:
        return self.timebase.clock

    @property
    def temporal_bounds_s(self) -> np.ndarray:
        return self.timebase.support.bounds(
            self.times_s,
            explicit_bounds_s=self.time_bounds_s,
        )


@dataclass(frozen=True)
class EventSeries:
    onset_s: np.ndarray
    offset_s: np.ndarray
    label: np.ndarray | None = None
    confidence: np.ndarray | None = None
    extra: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema: str = "EventSeries/v1"
    timebase: TimebaseSpec = field(default_factory=lambda: TimebaseSpec(kind="events"))
    temporal_context: TemporalContext = field(default_factory=TemporalContext)

    def __post_init__(self) -> None:
        if not isinstance(self.timebase, TimebaseSpec):
            raise TypeError("timebase must be a TimebaseSpec")
        if self.timebase.support.kind != "interval":
            raise ValueError("EventSeries timebase must declare interval support")
        onset_s = _ensure_1d_float_array(self.onset_s, "onset_s")
        offset_s = _ensure_1d_float_array(self.offset_s, "offset_s")
        if onset_s.shape != offset_s.shape:
            raise ValueError("onset_s and offset_s must have same shape")
        if np.any(offset_s < onset_s):
            raise ValueError("offset_s must be >= onset_s for all events")
        if len(onset_s) > 1 and not is_monotonic_non_decreasing(onset_s):
            raise ValueError("onset_s must be monotonic non-decreasing")
        if self.label is not None and len(self.label) != len(onset_s):
            raise ValueError("label length must match onset_s length")
        if self.confidence is not None and len(self.confidence) != len(onset_s):
            raise ValueError("confidence length must match onset_s length")
        missing = sorted(REQUIRED_METADATA_FIELDS - set(self.metadata.keys()))
        if missing:
            raise ValueError(f"metadata missing required fields: {missing}")
        object.__setattr__(self, "onset_s", onset_s)
        object.__setattr__(self, "offset_s", offset_s)
        if not isinstance(self.temporal_context, TemporalContext):
            object.__setattr__(
                self,
                "temporal_context",
                TemporalContext.from_dict(self.temporal_context),
            )

    def __len__(self) -> int:
        return int(self.onset_s.shape[0])

    @property
    def clock(self) -> ClockRef:
        return self.timebase.clock

    @property
    def temporal_bounds_s(self) -> np.ndarray:
        return np.column_stack([self.onset_s, self.offset_s])


@dataclass(frozen=True)
class TrackSeries:
    times_s: np.ndarray
    track_id: np.ndarray
    values: np.ndarray
    dims: tuple[str, ...] = ("time", "track", "feature")
    coords: dict[str, list[Any]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema: str = "TrackSeries/v1"
    timebase: TimebaseSpec = field(default_factory=lambda: TimebaseSpec(kind="frames"))
    time_bounds_s: np.ndarray | None = None
    temporal_context: TemporalContext = field(default_factory=TemporalContext)

    def __post_init__(self) -> None:
        if not isinstance(self.timebase, TimebaseSpec):
            raise TypeError("timebase must be a TimebaseSpec")
        times_s = _ensure_1d_float_array(self.times_s, "times_s")
        values = np.asarray(self.values)
        track_id = np.asarray(self.track_id)
        if values.ndim < 2:
            raise ValueError("TrackSeries.values must have at least 2 dims")
        if values.shape[0] != times_s.shape[0]:
            raise ValueError("values first dimension must match times_s length")
        if values.shape[1] != track_id.shape[0]:
            raise ValueError("values second dimension must match track_id length")
        if not is_monotonic_non_decreasing(times_s):
            raise ValueError("times_s must be monotonic non-decreasing")
        missing = sorted(REQUIRED_METADATA_FIELDS - set(self.metadata.keys()))
        if missing:
            raise ValueError(f"metadata missing required fields: {missing}")
        object.__setattr__(self, "times_s", times_s)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "track_id", track_id)
        if self.time_bounds_s is not None:
            object.__setattr__(
                self,
                "time_bounds_s",
                validate_time_bounds(self.time_bounds_s, len(times_s)),
            )
        elif self.timebase.support.kind == "interval":
            raise ValueError("interval support requires time_bounds_s")
        if not isinstance(self.temporal_context, TemporalContext):
            object.__setattr__(
                self,
                "temporal_context",
                TemporalContext.from_dict(self.temporal_context),
            )

    @property
    def clock(self) -> ClockRef:
        return self.timebase.clock

    @property
    def temporal_bounds_s(self) -> np.ndarray:
        return self.timebase.support.bounds(
            self.times_s,
            explicit_bounds_s=self.time_bounds_s,
        )
