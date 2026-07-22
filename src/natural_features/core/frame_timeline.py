"""Frame-index mapping for events aligned to video time."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .feature_types import EventSeries
from .stimulus import VideoStimulus
from .timebase import ClockRef, STIMULUS_CLOCK, TemporalContext

FramePolicy = str

_VALID_POLICIES = {"start", "center", "nearest", "overlap"}


def _as_1d_float(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    return arr


@dataclass(frozen=True)
class FrameTimeline:
    """A non-lossy map between continuous seconds and discrete video frames."""

    frame_index: np.ndarray
    onset_s: np.ndarray
    offset_s: np.ndarray
    source: str | None = None
    fps: float | None = None
    reference: ClockRef | str = STIMULUS_CLOCK
    temporal_context: TemporalContext = TemporalContext()

    def __post_init__(self) -> None:
        frame_index = np.asarray(self.frame_index, dtype=np.int64)
        onset_s = _as_1d_float(self.onset_s, "onset_s")
        offset_s = _as_1d_float(self.offset_s, "offset_s")
        if frame_index.ndim != 1:
            raise ValueError("frame_index must be 1-D")
        if frame_index.shape != onset_s.shape or onset_s.shape != offset_s.shape:
            raise ValueError(
                "frame_index, onset_s, and offset_s must have the same length"
            )
        if np.any(offset_s < onset_s):
            raise ValueError("frame offsets must be >= frame onsets")
        if len(onset_s) > 1 and np.any(np.diff(onset_s) < 0):
            raise ValueError("frame onsets must be monotonic non-decreasing")
        if self.fps is not None and (
            not np.isfinite(float(self.fps)) or float(self.fps) <= 0
        ):
            raise ValueError("fps must be a positive finite value when provided")
        object.__setattr__(self, "frame_index", frame_index)
        object.__setattr__(self, "onset_s", onset_s)
        object.__setattr__(self, "offset_s", offset_s)
        object.__setattr__(self, "reference", ClockRef(self.reference))
        if not isinstance(self.temporal_context, TemporalContext):
            object.__setattr__(self, "temporal_context", TemporalContext.from_dict(self.temporal_context))
        if self.fps is not None:
            object.__setattr__(self, "fps", float(self.fps))

    @classmethod
    def from_fps(
        cls,
        *,
        duration_s: float,
        fps: float,
        start_s: float = 0.0,
        first_frame_index: int = 0,
        source: str | None = None,
        reference: ClockRef | str = STIMULUS_CLOCK,
        temporal_context: TemporalContext = TemporalContext(),
    ) -> "FrameTimeline":
        """Build a regular timeline covering a clip duration at a known FPS."""

        duration = float(duration_s)
        rate = float(fps)
        start = float(start_s)
        if not np.isfinite(duration) or duration < 0:
            raise ValueError("duration_s must be a finite non-negative value")
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError("fps must be a positive finite value")
        if not np.isfinite(start):
            raise ValueError("start_s must be finite")
        n_frames = int(np.ceil(duration * rate))
        if duration > 0 and n_frames == 0:
            n_frames = 1
        first = int(first_frame_index)
        frame_index = first + np.arange(n_frames, dtype=np.int64)
        onset_s = start + np.arange(n_frames, dtype=np.float64) / rate
        offset_s = onset_s + (1.0 / rate)
        return cls(
            frame_index=frame_index,
            onset_s=onset_s,
            offset_s=offset_s,
            source=source,
            fps=rate,
            reference=reference,
            temporal_context=temporal_context,
        )

    @classmethod
    def from_video_stimulus(
        cls,
        video: VideoStimulus,
        *,
        source: str | None = None,
    ) -> "FrameTimeline":
        """Build a timeline from a materialized ``VideoStimulus``."""

        onset_s = video.frame_times_s
        offset_s = onset_s + (1.0 / float(video.fps))
        return cls(
            frame_index=np.arange(len(video.frames), dtype=np.int64),
            onset_s=onset_s,
            offset_s=offset_s,
            source=source if source is not None else video.source,
            fps=float(video.fps),
            reference=video.clock,
            temporal_context=video.temporal_context,
        )

    @property
    def centers_s(self) -> np.ndarray:
        return self.onset_s + (self.offset_s - self.onset_s) / 2.0

    def _clip_index(self, idx: np.ndarray) -> np.ndarray:
        if len(self.frame_index) == 0:
            return idx.astype(np.int64)
        return np.clip(idx, 0, len(self.frame_index) - 1).astype(np.int64)

    def _frame_containing_time(self, times_s: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.offset_s, times_s, side="right")
        return self._clip_index(idx)

    def _nearest_frame(self, times_s: np.ndarray) -> np.ndarray:
        centers = self.centers_s
        idx = np.searchsorted(centers, times_s, side="left")
        right = self._clip_index(idx)
        left = self._clip_index(idx - 1)
        use_left = np.abs(times_s - centers[left]) <= np.abs(times_s - centers[right])
        return np.where(use_left, left, right).astype(np.int64)

    def map_events(
        self, events: EventSeries, *, policy: FramePolicy = "overlap"
    ) -> dict[str, np.ndarray]:
        """Map event intervals to frame indices using an explicit policy."""

        policy = str(policy).strip().lower()
        if policy not in _VALID_POLICIES:
            raise ValueError(
                f"policy must be one of {sorted(_VALID_POLICIES)}, got {policy!r}"
            )
        context = events.temporal_context.merged(self.temporal_context)
        event_onset = events.onset_s
        event_offset = events.offset_s
        if events.clock != self.reference:
            mapping = context.resolve(events.clock, self.reference)
            event_onset = np.asarray(mapping.apply(event_onset), dtype=np.float64)
            event_offset = np.asarray(mapping.apply(event_offset), dtype=np.float64)
        n = len(events)
        if len(self.frame_index) == 0:
            empty_int = np.full(n, -1, dtype=np.int64)
            empty_float = np.full(n, np.nan, dtype=np.float64)
            return {
                "frame_start": empty_int,
                "frame_end": empty_int,
                "frame_time_s": empty_float,
                "frame_end_time_s": empty_float,
            }

        if policy == "start":
            start_idx = self._frame_containing_time(event_onset)
            end_idx = start_idx.copy()
        elif policy in {"center", "nearest"}:
            event_center = event_onset + (event_offset - event_onset) / 2.0
            start_idx = self._nearest_frame(event_center)
            end_idx = start_idx.copy()
        else:
            start_idx = self._frame_containing_time(event_onset)
            # End is inclusive. Events ending exactly at a frame boundary belong
            # to the preceding frame for overlap accounting.
            end_idx = np.searchsorted(self.onset_s, event_offset, side="left") - 1
            zero_duration = event_offset <= event_onset
            end_idx = np.where(zero_duration, start_idx, end_idx)
            end_idx = np.maximum(end_idx, start_idx)
            end_idx = self._clip_index(end_idx)

        return {
            "frame_start": self.frame_index[start_idx],
            "frame_end": self.frame_index[end_idx],
            "frame_time_s": self.onset_s[start_idx],
            "frame_end_time_s": self.offset_s[end_idx],
        }

    def annotate_events(
        self, events: EventSeries, *, policy: FramePolicy = "overlap"
    ) -> EventSeries:
        """Return a copy of ``events`` with frame index/time columns in ``extra``."""

        extra = dict(events.extra)
        extra.update(self.map_events(events, policy=policy))
        return EventSeries(
            onset_s=events.onset_s,
            offset_s=events.offset_s,
            label=events.label,
            confidence=events.confidence,
            extra=extra,
            metadata=dict(events.metadata),
            schema=events.schema,
            timebase=events.timebase,
            temporal_context=events.temporal_context.merged(self.temporal_context),
        )

    def to_rows(self) -> list[dict[str, float | int | str | None]]:
        """Return a dependency-light table representation."""

        rows: list[dict[str, float | int | str | None]] = []
        for idx, onset, offset in zip(
            self.frame_index, self.onset_s, self.offset_s, strict=True
        ):
            rows.append(
                {
                    "frame_index": int(idx),
                    "onset_s": float(onset),
                    "offset_s": float(offset),
                    "duration_s": float(offset - onset),
                    "source": self.source,
                    "fps": None if self.fps is None else float(self.fps),
                    "time_reference": str(self.reference),
                }
            )
        return rows


__all__ = ["FrameTimeline", "FramePolicy"]
