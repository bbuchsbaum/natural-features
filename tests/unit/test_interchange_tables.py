from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pandas")

from natural_features.core.feature_types import EventSeries, FeatureSeries, TrackSeries
from natural_features.core.interchange import (
    as_event_table,
    as_object_table,
    ensure_object_ids,
    merge_feature_tables,
    object_events,
)
from natural_features.features.common import extractor_metadata
from natural_features.core.timebase import ClockMap, TemporalContext, TimebaseSpec


def test_object_events_create_canonical_geometry_table() -> None:
    md = extractor_metadata("vision.faces", params={"threshold": 0.5}, extra={"source": "movie.mp4"})
    faces = object_events(
        onset_s=[1.0, 1.0],
        offset_s=[1.5, 1.5],
        object_type="face",
        label=["face", "face"],
        confidence=[0.9, 0.8],
        x=[0.1, 0.6],
        y=[0.2, 0.2],
        width=[0.2, 0.15],
        height=[0.3, 0.25],
        coordinate_space="relative",
        metadata=md,
    )
    table = as_object_table(faces)
    assert list(table.columns[:16]) == [
        "source_id",
        "time_s",
        "onset_s",
        "offset_s",
        "duration_s",
        "object_id",
        "track_id",
        "object_type",
        "label",
        "confidence",
        "x",
        "y",
        "width",
        "height",
        "area",
        "coordinate_space",
    ]
    assert list(table["source_id"]) == ["movie.mp4", "movie.mp4"]
    assert list(table["object_id"]) == ["face_0001", "face_0002"]
    np.testing.assert_allclose(table["area"].to_numpy(), np.array([0.06, 0.0375]))


def test_ensure_object_ids_preserves_existing_and_fills_missing() -> None:
    words = EventSeries(
        onset_s=np.array([0.0, 0.5, 1.0]),
        offset_s=np.array([0.4, 0.9, 1.4]),
        label=np.array(["the", "cat", "sat"], dtype=object),
        extra={"object_type": np.array(["word", "word", "word"], dtype=object), "object_id": np.array(["word_custom", None, ""], dtype=object)},
        metadata=extractor_metadata("speech.words"),
    )
    out = ensure_object_ids(words)
    assert list(out.extra["object_id"]) == ["word_custom", "word_0002", "word_0003"]


def test_as_object_table_sparse_empty_and_track_inputs() -> None:
    scenes = object_events(onset_s=[0.0], offset_s=[2.0], object_type="scene", label=["intro"])
    table = as_object_table(scenes, include_metadata=False)
    assert list(table["object_id"]) == ["scene_0001"]
    assert np.isnan(table["confidence"].iloc[0])

    empty = object_events(onset_s=[], offset_s=[], object_type="ocr")
    empty_table = as_object_table(empty)
    assert len(empty_table) == 0
    assert "object_id" in empty_table.columns

    tracks = TrackSeries(
        times_s=np.array([0.0, 1.0]),
        track_id=np.array(["speaker_0", "speaker_1"], dtype=object),
        values=np.ones((2, 2, 1), dtype=np.float32),
        coords={"feature": ["activity"]},
        metadata=extractor_metadata("speaker.tracks", extra={"source": "dialog.wav"}),
    )
    track_table = as_object_table(tracks)
    assert len(track_table) == 4
    assert sorted(track_table["object_id"].unique()) == ["speaker_0", "speaker_1"]
    assert set(track_table["object_type"]) == {"track"}


def test_merge_feature_tables_long_and_wide_contracts() -> None:
    fs = FeatureSeries(
        values=np.array([[1.0], [2.0]], dtype=np.float32),
        times_s=np.array([0.0, 1.0]),
        coords={"feature": ["energy"]},
        metadata=extractor_metadata("merge.feature"),
    )
    events = object_events(onset_s=[0.2, 0.2], offset_s=[0.4, 0.4], object_type="word", label=["hello", "world"])
    long = merge_feature_tables({"acoustic": fs, "words": events}, format="long", include_metadata=False)
    assert {"feature_id", "output_name", "output_type"} <= set(long.columns)
    assert {"features", "objects"} <= set(long["output_type"])
    assert "object_id" in long.columns

    plain_events = EventSeries(
        onset_s=np.array([0.0]),
        offset_s=np.array([0.5]),
        label=np.array(["plain"], dtype=object),
        metadata=extractor_metadata("plain.events"),
    )
    plain = merge_feature_tables({"events": plain_events}, format="long", include_objects=False, include_metadata=False)
    assert "onset_s" in plain.columns
    assert "object_id" not in plain.columns

    wide = merge_feature_tables({"acoustic": fs}, format="wide")
    assert "acoustic__energy" in wide.columns
    with pytest.raises(ValueError, match="Wide format only supports FeatureSeries"):
        merge_feature_tables({"events": events}, format="wide")


def test_as_event_table_keeps_plain_events_plain() -> None:
    events = EventSeries(
        onset_s=np.array([0.0, 1.0]),
        offset_s=np.array([0.5, 1.5]),
        label=np.array(["a", "b"], dtype=object),
        metadata=extractor_metadata("events.test"),
    )
    table = as_event_table(events, include_metadata=False)
    assert list(table.columns) == [
        "onset_s",
        "offset_s",
        "duration_s",
        "time_reference",
        "label",
        "confidence",
    ]


def test_tables_expose_temporal_reference_and_cross_clock_wide_merge_is_explicit() -> None:
    left = FeatureSeries(
        values=np.array([[1.0], [2.0]], dtype=np.float32),
        times_s=np.array([30.0, 31.0]),
        coords={"feature": ["left"]},
        metadata=extractor_metadata("left"),
        timebase=TimebaseSpec(kind="samples", reference="stimulus"),
    )
    right = FeatureSeries(
        values=np.array([[3.0], [4.0]], dtype=np.float32),
        times_s=np.array([7.0, 8.0]),
        coords={"feature": ["right"]},
        metadata=extractor_metadata("right"),
        timebase=TimebaseSpec(kind="samples", reference="scan:run-01"),
    )
    context = TemporalContext((ClockMap("stimulus", "scan:run-01", offset_s=-23.0),))

    long = merge_feature_tables({"left": left, "right": right}, format="long")
    assert set(long["time_reference"]) == {"stimulus", "scan:run-01"}
    with pytest.raises(ValueError, match="different clocks"):
        merge_feature_tables({"left": left, "right": right}, format="wide")
    with pytest.raises(ValueError, match="outer_exact"):
        merge_feature_tables(
            {"left": left, "right": right},
            format="wide",
            temporal_context=context,
            target_clock="scan:run-01",
            join_policy="linear",
        )
    wide = merge_feature_tables(
        {"left": left, "right": right},
        format="wide",
        temporal_context=context,
        target_clock="scan:run-01",
        join_policy="outer_exact",
    )
    assert list(wide["time_s"]) == [7.0, 8.0]
    assert set(wide["time_reference"]) == {"scan:run-01"}
    np.testing.assert_array_equal(wide["left__left"], [1.0, 2.0])
    np.testing.assert_array_equal(wide["right__right"], [3.0, 4.0])
