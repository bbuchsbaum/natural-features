from __future__ import annotations

import numpy as np

from natural_features.core.feature_types import EventSeries
from natural_features.core.frame_timeline import FrameTimeline
from natural_features.features.common import extractor_metadata


def _events() -> EventSeries:
    return EventSeries(
        onset_s=np.array([0.05, 0.21, 0.40], dtype=np.float64),
        offset_s=np.array([0.16, 0.21, 0.62], dtype=np.float64),
        label=np.array(["a", "b", "c"], dtype=object),
        metadata=extractor_metadata("test.events"),
    )


def test_frame_timeline_maps_events_with_explicit_policies() -> None:
    timeline = FrameTimeline.from_fps(duration_s=1.0, fps=10.0)
    events = _events()

    overlap = timeline.map_events(events, policy="overlap")
    assert overlap["frame_start"].tolist() == [0, 2, 4]
    assert overlap["frame_end"].tolist() == [1, 2, 6]

    start = timeline.map_events(events, policy="start")
    assert start["frame_start"].tolist() == [0, 2, 4]
    assert start["frame_end"].tolist() == [0, 2, 4]

    center = timeline.map_events(events, policy="center")
    assert center["frame_start"].tolist() == [1, 2, 5]
    assert center["frame_end"].tolist() == [1, 2, 5]


def test_frame_timeline_preserves_source_frame_indices_for_trimmed_clips() -> None:
    timeline = FrameTimeline.from_fps(
        duration_s=0.5,
        fps=10.0,
        start_s=2.0,
        first_frame_index=20,
        source="movie.mp4",
    )
    event = EventSeries(
        onset_s=np.array([2.05]),
        offset_s=np.array([2.19]),
        label=np.array(["hello"], dtype=object),
        metadata=extractor_metadata("test.word"),
    )

    annotated = timeline.annotate_events(event, policy="overlap")

    assert annotated.extra["frame_start"].tolist() == [20]
    assert annotated.extra["frame_end"].tolist() == [21]
    np.testing.assert_allclose(annotated.extra["frame_time_s"], [2.0])
    np.testing.assert_allclose(annotated.extra["frame_end_time_s"], [2.2])
