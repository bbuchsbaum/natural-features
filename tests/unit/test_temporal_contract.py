from __future__ import annotations

import inspect

import numpy as np
import pytest

from natural_features.core.feature_bundle import FeatureBundle, temporal_object_in_clock
from natural_features.core.feature_types import EventSeries, FeatureSeries
from natural_features.core.frame_timeline import FrameTimeline
from natural_features.core.timebase import (
    ClockMap,
    ClockRef,
    SupportSpec,
    TemporalContext,
    TimebaseSpec,
)


def _metadata(name: str) -> dict[str, str]:
    return {"extractor_id": name, "params_hash": "fixed"}


def test_clock_map_convention_inverse_and_composition() -> None:
    stimulus_to_scan = ClockMap("stimulus", "scan:run-01", offset_s=-23.0)
    scan_to_experiment = ClockMap("scan:run-01", "experiment", offset_s=100.0)

    assert stimulus_to_scan.apply(30.0) == 7.0
    assert stimulus_to_scan.inverse().apply(7.0) == 30.0
    composed = stimulus_to_scan.then(scan_to_experiment)
    assert composed.source == ClockRef("stimulus")
    assert composed.target == ClockRef("experiment")
    assert composed.apply(30.0) == 107.0


def test_temporal_context_resolves_paths_and_rejects_inconsistency() -> None:
    context = TemporalContext(
        (
            ClockMap("stimulus", "scan", offset_s=-23.0),
            ClockMap("scan", "experiment", offset_s=100.0),
        )
    )
    np.testing.assert_allclose(
        context.convert([30.0, 31.0], source="stimulus", target="experiment"),
        [107.0, 108.0],
    )
    with pytest.raises(ValueError, match="inconsistent clock mappings"):
        TemporalContext(
            (
                ClockMap("stimulus", "scan", offset_s=-23.0),
                ClockMap("scan", "experiment", offset_s=100.0),
                ClockMap("stimulus", "experiment", offset_s=78.0),
            )
        )
    with pytest.raises(KeyError, match="no clock mapping"):
        context.resolve("stimulus", "unrelated")

    duplicate = context.merged(context)
    inverse = TemporalContext(
        (
            ClockMap("experiment", "scan", offset_s=-100.0),
            ClockMap("scan", "stimulus", offset_s=23.0),
        )
    )
    assert duplicate == context
    assert inverse.digest == context.digest


def test_support_spec_preserves_points_and_constructs_window_bounds() -> None:
    times = np.array([0.1, 0.2, 0.3])
    points = SupportSpec(kind="point")
    np.testing.assert_array_equal(points.bounds(times), np.column_stack([times, times]))

    centered = SupportSpec(kind="window", anchor="center", width_s=0.1)
    np.testing.assert_allclose(
        centered.bounds(times),
        np.array([[0.05, 0.15], [0.15, 0.25], [0.25, 0.35]]),
    )
    interval = SupportSpec(kind="interval", anchor="onset")
    with pytest.raises(ValueError, match="requires per-row"):
        interval.bounds(times)
    with pytest.raises(ValueError, match="positive finite"):
        centered.scaled(0.0)


def test_temporal_coordinates_reject_nonfinite_singletons() -> None:
    with pytest.raises(ValueError, match="finite"):
        FeatureSeries(
            values=np.ones((1, 1), dtype=np.float32),
            times_s=np.array([np.nan]),
            metadata=_metadata("nonfinite"),
        )
    with pytest.raises(ValueError, match="finite"):
        EventSeries(
            onset_s=np.array([np.nan]),
            offset_s=np.array([1.0]),
            metadata=_metadata("nonfinite-event"),
        )


def test_feature_bundle_converts_clocks_without_resampling_or_copying_values() -> None:
    context = TemporalContext((ClockMap("stimulus", "scan:run-01", offset_s=-23.0),))
    fast_times = np.arange(29.8, 30.3, 0.1)
    slow_times = np.arange(29.0, 31.0, 0.5)
    fast_values = np.arange(len(fast_times), dtype=np.float32).reshape(-1, 1)
    slow_values = np.arange(len(slow_times), dtype=np.float32).reshape(-1, 1)
    fast = FeatureSeries(
        values=fast_values,
        times_s=fast_times,
        metadata=_metadata("fast"),
        timebase=TimebaseSpec(
            kind="audio_hop",
            reference="stimulus",
            hop_s=0.1,
            window_s=0.1,
            alignment="center",
        ),
    )
    slow = FeatureSeries(
        values=slow_values,
        times_s=slow_times,
        metadata=_metadata("slow"),
        timebase=TimebaseSpec(kind="windows", reference="stimulus", stride_s=0.5),
    )
    events = EventSeries(
        onset_s=np.array([29.7, 30.4]),
        offset_s=np.array([29.9, 30.9]),
        metadata=_metadata("events"),
        timebase=TimebaseSpec(kind="events", reference="stimulus"),
    )
    bundle = FeatureBundle(
        {"fast": fast, "slow": slow, "events": events},
        temporal_context=context,
    )

    fast_scan = bundle.in_clock("fast", "scan:run-01")
    slow_scan = bundle.in_clock("slow", "scan:run-01")
    events_scan = bundle.in_clock("events", "scan:run-01")

    assert len(fast_scan.times_s) == len(fast_times)
    assert len(slow_scan.times_s) == len(slow_times)
    assert len(events_scan.onset_s) == len(events.onset_s)
    assert fast_scan.values is fast.values
    assert slow_scan.values is slow.values
    np.testing.assert_allclose(fast_scan.times_s, fast_times - 23.0)
    np.testing.assert_allclose(events_scan.onset_s, events.onset_s - 23.0)
    assert np.isclose(context.convert(30.0, source="stimulus", target="scan:run-01"), 7.0)

    payload = bundle.temporal_payload("fast", target_clock="scan:run-01")
    assert payload.clock == "scan:run-01"
    assert payload.values is fast.values
    assert payload.time_bounds_s is not None
    payload_dict = payload.to_dict()
    assert payload_dict["clock"] == "scan:run-01"
    assert payload_dict["values"] is fast.values
    assert "tr_s" not in inspect.signature(bundle.temporal_payload).parameters
    assert "hrf" not in inspect.signature(bundle.temporal_payload).parameters


def test_per_row_bounds_transform_with_clock_scale() -> None:
    feature = FeatureSeries(
        values=np.ones((2, 1), dtype=np.float32),
        times_s=np.array([1.0, 3.0]),
        time_bounds_s=np.array([[0.5, 1.5], [2.0, 4.0]]),
        metadata=_metadata("bounded"),
        timebase=TimebaseSpec(
            kind="irregular",
            reference="media",
            support=SupportSpec(kind="interval", anchor="center"),
        ),
    )
    context = TemporalContext((ClockMap("media", "scanner", scale=2.0, offset_s=5.0),))
    observed = temporal_object_in_clock(feature, "scanner", context=context)
    np.testing.assert_allclose(observed.times_s, [7.0, 11.0])
    np.testing.assert_allclose(observed.time_bounds_s, [[6.0, 8.0], [9.0, 13.0]])
    np.testing.assert_array_equal(observed.values, feature.values)


def test_frame_timeline_maps_events_across_explicit_clocks() -> None:
    context = TemporalContext(
        (ClockMap("stimulus", "scan:run-01", offset_s=-23.0),)
    )
    events = EventSeries(
        onset_s=np.array([30.0]),
        offset_s=np.array([30.4]),
        metadata=_metadata("word"),
        timebase=TimebaseSpec(kind="events", reference="stimulus"),
        temporal_context=context,
    )
    frames = FrameTimeline.from_fps(
        duration_s=2.0,
        fps=2.0,
        start_s=7.0,
        reference="scan:run-01",
        temporal_context=context,
    )

    mapped = frames.map_events(events)

    assert mapped["frame_start"].tolist() == [0]
    assert mapped["frame_end"].tolist() == [0]


def test_timebase_serialization_is_explicit_and_legacy_strings_remain_valid() -> None:
    timebase = TimebaseSpec(
        kind="audio_hop",
        reference="stimulus_start",
        hop_s=0.1,
        window_s=0.2,
        alignment="center",
    )
    assert timebase.reference == "stimulus_start"
    assert isinstance(timebase.reference, ClockRef)
    restored = TimebaseSpec.from_dict(timebase.to_dict(), default_kind="frames")
    assert restored == timebase
    assert restored.support == SupportSpec(kind="window", anchor="center", width_s=0.2)
