"""Generic timeline and alignment contracts for temporal feature objects."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

import numpy as np

from .feature_types import EventSeries, FeatureSeries, TrackSeries
from .frame_timeline import FrameTimeline
from .stimulus import VideoStimulus

AlignmentPolicy = str

_VALID_POLICIES = {"start", "center", "nearest", "overlap"}


def _as_1d_float(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    return arr


def _default_index(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.int64)


def _infer_offsets_from_times(times_s: np.ndarray, *, duration_s: float | None = None) -> np.ndarray:
    if len(times_s) == 0:
        return np.asarray([], dtype=np.float64)
    if duration_s is not None:
        duration = float(duration_s)
        if not np.isfinite(duration) or duration < 0:
            raise ValueError("duration_s must be a finite non-negative value")
        return times_s + duration
    if len(times_s) == 1:
        return times_s.astype(np.float64, copy=True)
    offsets = np.empty_like(times_s, dtype=np.float64)
    offsets[:-1] = times_s[1:]
    diffs = np.diff(times_s)
    positive = diffs[diffs > 0]
    step = float(np.median(positive)) if positive.size else 0.0
    offsets[-1] = times_s[-1] + step
    offsets = np.maximum(offsets, times_s)
    return offsets


def _temporal_intervals(obj: Any, *, point_samples: bool = True) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(obj, EventSeries):
        return obj.onset_s, obj.offset_s
    if isinstance(obj, FeatureSeries):
        onset = obj.times_s
        offset = onset if point_samples else _infer_offsets_from_times(onset)
        return onset, offset
    if isinstance(obj, TrackSeries):
        onset = obj.times_s
        offset = onset if point_samples else _infer_offsets_from_times(onset)
        return onset, offset
    raise TypeError("Expected FeatureSeries, EventSeries, or TrackSeries")


def _source_kind(obj: Any) -> str:
    if isinstance(obj, FeatureSeries):
        return "features"
    if isinstance(obj, EventSeries):
        return "events"
    if isinstance(obj, TrackSeries):
        return "tracks"
    return type(obj).__name__


def _source_schema(obj: Any) -> str:
    return str(getattr(obj, "schema", type(obj).__name__))


def _sanitize_prefix(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return token or "target"


@dataclass(frozen=True)
class Timeline:
    """A named interval timeline used as an alignment target."""

    name: str
    onset_s: np.ndarray
    offset_s: np.ndarray
    index: np.ndarray | None = None
    kind: str = "intervals"
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        onset_s = _as_1d_float(self.onset_s, "onset_s")
        offset_s = _as_1d_float(self.offset_s, "offset_s")
        if onset_s.shape != offset_s.shape:
            raise ValueError("onset_s and offset_s must have the same length")
        if not np.all(np.isfinite(onset_s)) or not np.all(np.isfinite(offset_s)):
            raise ValueError("onset_s and offset_s must contain only finite values")
        if np.any(offset_s < onset_s):
            raise ValueError("offset_s must be >= onset_s")
        if len(onset_s) > 1 and np.any(np.diff(onset_s) < 0):
            raise ValueError("onset_s must be monotonic non-decreasing")
        index = _default_index(len(onset_s)) if self.index is None else np.asarray(self.index)
        if index.ndim != 1 or len(index) != len(onset_s):
            raise ValueError("index must be 1-D with the same length as onset_s")
        object.__setattr__(self, "onset_s", onset_s)
        object.__setattr__(self, "offset_s", offset_s)
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_points(
        cls,
        name: str,
        times_s: Any,
        *,
        duration_s: float | None = None,
        index: Any | None = None,
        kind: str = "samples",
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "Timeline":
        times = _as_1d_float(times_s, "times_s")
        return cls(
            name=name,
            onset_s=times,
            offset_s=_infer_offsets_from_times(times, duration_s=duration_s),
            index=_default_index(len(times)) if index is None else np.asarray(index),
            kind=kind,
            source=source,
            metadata=metadata or {},
        )

    @classmethod
    def from_feature(cls, name: str, obj: FeatureSeries | EventSeries | TrackSeries) -> "Timeline":
        if isinstance(obj, EventSeries):
            index = (
                np.asarray(obj.extra["object_id"], dtype=object)
                if "object_id" in obj.extra and len(obj.extra["object_id"]) == len(obj)
                else _default_index(len(obj))
            )
            return cls(
                name=name,
                onset_s=obj.onset_s,
                offset_s=obj.offset_s,
                index=index,
                kind=obj.timebase.kind,
                metadata={"schema": obj.schema, **obj.metadata},
            )
        onset, offset = _temporal_intervals(obj, point_samples=False)
        return cls(
            name=name,
            onset_s=onset,
            offset_s=offset,
            index=_default_index(len(onset)),
            kind=obj.timebase.kind,
            metadata={"schema": obj.schema, **obj.metadata},
        )

    @classmethod
    def from_frame_timeline(cls, frame_timeline: FrameTimeline, *, name: str = "video_frames") -> "Timeline":
        return cls(
            name=name,
            onset_s=frame_timeline.onset_s,
            offset_s=frame_timeline.offset_s,
            index=frame_timeline.frame_index,
            kind="frames",
            source=frame_timeline.source,
            metadata={"fps": frame_timeline.fps},
        )

    @classmethod
    def from_video_stimulus(cls, video: VideoStimulus, *, name: str = "video_frames") -> "Timeline":
        return cls.from_frame_timeline(FrameTimeline.from_video_stimulus(video), name=name)

    @property
    def centers_s(self) -> np.ndarray:
        return self.onset_s + (self.offset_s - self.onset_s) / 2.0

    def _clip_position(self, position: np.ndarray) -> np.ndarray:
        if len(self.onset_s) == 0:
            return position.astype(np.int64)
        return np.clip(position, 0, len(self.onset_s) - 1).astype(np.int64)

    def _position_containing_time(self, times_s: np.ndarray) -> np.ndarray:
        position = np.searchsorted(self.offset_s, times_s, side="right")
        return self._clip_position(position)

    def _nearest_position(self, times_s: np.ndarray) -> np.ndarray:
        centers = self.centers_s
        position = np.searchsorted(centers, times_s, side="left")
        right = self._clip_position(position)
        left = self._clip_position(position - 1)
        use_left = np.abs(times_s - centers[left]) <= np.abs(times_s - centers[right])
        return np.where(use_left, left, right).astype(np.int64)

    def map_intervals(
        self,
        onset_s: Any,
        offset_s: Any | None = None,
        *,
        policy: AlignmentPolicy = "overlap",
    ) -> dict[str, np.ndarray]:
        """Map source intervals to this timeline."""

        policy = str(policy).strip().lower()
        if policy not in _VALID_POLICIES:
            raise ValueError(f"policy must be one of {sorted(_VALID_POLICIES)}, got {policy!r}")
        onset = _as_1d_float(onset_s, "onset_s")
        offset = onset if offset_s is None else _as_1d_float(offset_s, "offset_s")
        if onset.shape != offset.shape:
            raise ValueError("onset_s and offset_s must have the same length")
        if not np.all(np.isfinite(onset)) or not np.all(np.isfinite(offset)):
            raise ValueError("onset_s and offset_s must contain only finite values")
        if np.any(offset < onset):
            raise ValueError("offset_s must be >= onset_s")
        n = len(onset)
        if len(self.onset_s) == 0:
            empty_int = np.full(n, -1, dtype=np.int64)
            empty_obj = np.full(n, None, dtype=object)
            empty_float = np.full(n, np.nan, dtype=np.float64)
            return {
                "target_position_start": empty_int,
                "target_position_end": empty_int,
                "target_index_start": empty_obj,
                "target_index_end": empty_obj,
                "target_time_s": empty_float,
                "target_end_time_s": empty_float,
            }

        if policy == "start":
            start_pos = self._position_containing_time(onset)
            end_pos = start_pos.copy()
        elif policy in {"center", "nearest"}:
            source_center = onset + (offset - onset) / 2.0
            start_pos = self._nearest_position(source_center)
            end_pos = start_pos.copy()
        else:
            start_pos = self._position_containing_time(onset)
            end_pos = np.searchsorted(self.onset_s, offset, side="left") - 1
            zero_duration = offset <= onset
            end_pos = np.where(zero_duration, start_pos, end_pos)
            end_pos = np.maximum(end_pos, start_pos)
            end_pos = self._clip_position(end_pos)

        assert self.index is not None
        return {
            "target_position_start": start_pos,
            "target_position_end": end_pos,
            "target_index_start": self.index[start_pos],
            "target_index_end": self.index[end_pos],
            "target_time_s": self.onset_s[start_pos],
            "target_end_time_s": self.offset_s[end_pos],
        }

    def to_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        assert self.index is not None
        for position, (idx, onset, offset) in enumerate(zip(self.index, self.onset_s, self.offset_s, strict=True)):
            rows.append(
                {
                    "timeline": self.name,
                    "position": int(position),
                    "index": idx.item() if isinstance(idx, np.generic) else idx,
                    "onset_s": float(onset),
                    "offset_s": float(offset),
                    "duration_s": float(offset - onset),
                    "kind": self.kind,
                    "source": self.source,
                }
            )
        return rows


@dataclass(frozen=True)
class FeatureAlignment:
    """Alignment of one temporal feature output to a target timeline."""

    source_name: str
    source: FeatureSeries | EventSeries | TrackSeries
    source_kind: str
    source_schema: str
    source_onset_s: np.ndarray
    source_offset_s: np.ndarray
    target: Timeline
    mapping: dict[str, np.ndarray]
    policy: AlignmentPolicy = "overlap"

    def __post_init__(self) -> None:
        onset = _as_1d_float(self.source_onset_s, "source_onset_s")
        offset = _as_1d_float(self.source_offset_s, "source_offset_s")
        if onset.shape != offset.shape:
            raise ValueError("source_onset_s and source_offset_s must have the same length")
        for key, value in self.mapping.items():
            arr = np.asarray(value)
            if len(arr) != len(onset):
                raise ValueError(f"mapping column {key!r} must have length {len(onset)}")
        object.__setattr__(self, "source_onset_s", onset)
        object.__setattr__(self, "source_offset_s", offset)

    def to_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row_i, (onset, offset) in enumerate(zip(self.source_onset_s, self.source_offset_s, strict=True)):
            row: dict[str, Any] = {
                "source_name": self.source_name,
                "source_kind": self.source_kind,
                "source_schema": self.source_schema,
                "source_row": int(row_i),
                "source_onset_s": float(onset),
                "source_offset_s": float(offset),
                "source_duration_s": float(offset - onset),
                "target_name": self.target.name,
                "target_kind": self.target.kind,
                "policy": self.policy,
            }
            for key, values in self.mapping.items():
                value = np.asarray(values)[row_i]
                row[key] = value.item() if isinstance(value, np.generic) else value
            rows.append(row)
        return rows

    def to_dataframe(self) -> Any:
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:
            raise RuntimeError("pandas is required for alignment dataframe output") from exc
        return pd.DataFrame(self.to_rows())

    def annotated_events(self, *, prefix: str | None = None) -> EventSeries:
        if not isinstance(self.source, EventSeries):
            raise TypeError("annotated_events is only defined for EventSeries sources")
        prefix = _sanitize_prefix(prefix or self.target.name)
        extra = dict(self.source.extra)
        extra.update(
            {
                f"{prefix}_position_start": self.mapping["target_position_start"],
                f"{prefix}_position_end": self.mapping["target_position_end"],
                f"{prefix}_index_start": self.mapping["target_index_start"],
                f"{prefix}_index_end": self.mapping["target_index_end"],
                f"{prefix}_time_s": self.mapping["target_time_s"],
                f"{prefix}_end_time_s": self.mapping["target_end_time_s"],
            }
        )
        return EventSeries(
            onset_s=self.source.onset_s,
            offset_s=self.source.offset_s,
            label=self.source.label,
            confidence=self.source.confidence,
            extra=extra,
            metadata=dict(self.source.metadata),
            schema=self.source.schema,
            timebase=self.source.timebase,
        )


def align_feature_to_timeline(
    source_name: str,
    source: FeatureSeries | EventSeries | TrackSeries,
    target: Timeline,
    *,
    policy: AlignmentPolicy = "overlap",
) -> FeatureAlignment:
    """Align a typed feature object to a target timeline."""

    onset, offset = _temporal_intervals(source, point_samples=True)
    mapping = target.map_intervals(onset, offset, policy=policy)
    return FeatureAlignment(
        source_name=source_name,
        source=source,
        source_kind=_source_kind(source),
        source_schema=_source_schema(source),
        source_onset_s=onset,
        source_offset_s=offset,
        target=target,
        mapping=mapping,
        policy=policy,
    )


__all__ = [
    "AlignmentPolicy",
    "FeatureAlignment",
    "Timeline",
    "align_feature_to_timeline",
]
