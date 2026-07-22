from __future__ import annotations

import numpy as np
import pytest

from natural_features.core.feature_types import EventSeries, FeatureSeries, TrackSeries
from natural_features.core.timebase import ClockMap, SupportSpec, TemporalContext, TimebaseSpec
from natural_features.storage.catalog import Catalog
from natural_features.storage.readers import read_event_series, read_feature_series, read_track_series
from natural_features.storage.writers import write_event_series, write_feature_series, write_track_series


def _metadata(name: str) -> dict[str, str]:
    return {"extractor_id": name, "params_hash": "fixed"}


def _context() -> TemporalContext:
    return TemporalContext((ClockMap("stimulus", "scan:run-01", offset_s=-23.0),))


def _assert_contract_equal(observed: object, expected: object) -> None:
    assert observed.schema == expected.schema
    assert observed.timebase == expected.timebase
    assert observed.temporal_context == expected.temporal_context
    assert observed.clock == expected.clock


def test_feature_npz_round_trip_preserves_full_temporal_contract(tmp_path) -> None:
    feature = FeatureSeries(
        values=np.arange(6, dtype=np.float32).reshape(3, 2),
        times_s=np.array([29.9, 30.0, 30.1]),
        time_bounds_s=np.array([[29.85, 29.95], [29.95, 30.05], [30.05, 30.15]]),
        metadata=_metadata("fast"),
        schema="FeatureSeries/temporal-test",
        timebase=TimebaseSpec(
            kind="audio_hop",
            reference="stimulus",
            hop_s=0.1,
            support=SupportSpec(kind="interval", anchor="center"),
        ),
        temporal_context=_context(),
    )
    path = write_feature_series(feature, tmp_path, fmt="npz")
    observed = read_feature_series(path)
    _assert_contract_equal(observed, feature)
    np.testing.assert_array_equal(observed.values, feature.values)
    np.testing.assert_array_equal(observed.times_s, feature.times_s)
    np.testing.assert_array_equal(observed.time_bounds_s, feature.time_bounds_s)


def test_feature_zarr_round_trip_preserves_full_temporal_contract(tmp_path) -> None:
    pytest.importorskip("zarr")
    feature = FeatureSeries(
        values=np.arange(6, dtype=np.float32).reshape(3, 2),
        times_s=np.array([29.9, 30.0, 30.1]),
        metadata=_metadata("fast-zarr"),
        schema="FeatureSeries/temporal-test",
        timebase=TimebaseSpec(
            kind="audio_hop",
            reference="stimulus",
            hop_s=0.1,
            window_s=0.1,
        ),
        temporal_context=_context(),
    )

    path = write_feature_series(feature, tmp_path, fmt="zarr")
    observed = read_feature_series(path)

    _assert_contract_equal(observed, feature)
    np.testing.assert_array_equal(observed.values, feature.values)
    np.testing.assert_array_equal(observed.temporal_bounds_s, feature.temporal_bounds_s)


def test_event_parquet_round_trip_preserves_full_temporal_contract(tmp_path) -> None:
    pytest.importorskip("pyarrow")
    events = EventSeries(
        onset_s=np.array([29.7, 30.4]),
        offset_s=np.array([29.9, 30.9]),
        label=np.array(["word", "word"]),
        confidence=np.array([0.9, 0.8]),
        metadata=_metadata("events"),
        schema="EventSeries/temporal-test",
        timebase=TimebaseSpec(kind="events", reference="stimulus"),
        temporal_context=_context(),
    )
    path = write_event_series(events, tmp_path, fmt="parquet")
    observed = read_event_series(path)
    _assert_contract_equal(observed, events)
    np.testing.assert_array_equal(observed.onset_s, events.onset_s)
    np.testing.assert_array_equal(observed.offset_s, events.offset_s)
    np.testing.assert_array_equal(observed.label, events.label)


def test_event_npz_round_trip_preserves_object_columns_without_pickle(tmp_path) -> None:
    events = EventSeries(
        onset_s=np.array([1.0, 2.0]),
        offset_s=np.array([1.2, 2.4]),
        label=np.array(["alpha", "beta"], dtype=object),
        extra={"speaker": np.array([None, "speaker-a"], dtype=object)},
        metadata=_metadata("object-events"),
        timebase=TimebaseSpec(kind="events", reference="scan:run-01"),
        temporal_context=_context(),
    )

    path = write_event_series(events, tmp_path, fmt="npz")
    observed = read_event_series(path)

    _assert_contract_equal(observed, events)
    np.testing.assert_array_equal(observed.label, events.label)
    np.testing.assert_array_equal(observed.extra["speaker"], events.extra["speaker"])


def test_track_zarr_round_trip_preserves_full_temporal_contract(tmp_path) -> None:
    pytest.importorskip("zarr")
    tracks = TrackSeries(
        times_s=np.array([29.5, 30.0]),
        track_id=np.array(["face-a", "face-b"], dtype=object),
        values=np.arange(8, dtype=np.float32).reshape(2, 2, 2),
        dims=("time", "track", "feature"),
        metadata=_metadata("tracks"),
        schema="TrackSeries/temporal-test",
        timebase=TimebaseSpec(
            kind="frames",
            reference="stimulus",
            window_s=0.5,
            alignment="center",
        ),
        temporal_context=_context(),
    )
    path = write_track_series(tracks, tmp_path, fmt="zarr")
    observed = read_track_series(path)
    _assert_contract_equal(observed, tracks)
    np.testing.assert_array_equal(observed.values, tracks.values)
    np.testing.assert_array_equal(observed.times_s, tracks.times_s)
    np.testing.assert_array_equal(observed.track_id, tracks.track_id)
    np.testing.assert_array_equal(observed.temporal_bounds_s, tracks.temporal_bounds_s)


def test_track_npz_round_trip_preserves_full_temporal_contract(tmp_path) -> None:
    tracks = TrackSeries(
        times_s=np.array([29.5, 30.0]),
        track_id=np.array(["face-a", "face-b"], dtype=object),
        values=np.arange(8, dtype=np.float32).reshape(2, 2, 2),
        metadata=_metadata("tracks-npz"),
        schema="TrackSeries/temporal-test",
        timebase=TimebaseSpec(
            kind="frames",
            reference="stimulus",
            window_s=0.5,
        ),
        temporal_context=_context(),
    )

    path = write_track_series(tracks, tmp_path, fmt="npz")
    observed = read_track_series(path)

    _assert_contract_equal(observed, tracks)
    np.testing.assert_array_equal(observed.values, tracks.values)
    np.testing.assert_array_equal(observed.track_id, tracks.track_id)
    np.testing.assert_array_equal(observed.temporal_bounds_s, tracks.temporal_bounds_s)


def test_catalog_identity_includes_coordinates_clock_and_context(tmp_path) -> None:
    values = np.ones((2, 1), dtype=np.float32)
    base = FeatureSeries(
        values=values,
        times_s=np.array([0.0, 0.1]),
        metadata=_metadata("same"),
        timebase=TimebaseSpec(kind="samples", reference="stimulus"),
        temporal_context=_context(),
    )
    shifted_clock = FeatureSeries(
        values=values,
        times_s=np.array([0.0, 0.1]),
        metadata=_metadata("same"),
        timebase=TimebaseSpec(kind="samples", reference="scan:run-01"),
        temporal_context=_context(),
    )
    shifted_coordinates = FeatureSeries(
        values=values,
        times_s=np.array([1.0, 1.1]),
        metadata=_metadata("same"),
        timebase=base.timebase,
        temporal_context=_context(),
    )
    catalog = Catalog(tmp_path / "catalog")
    records = [
        catalog.put(
            obj,
            run_id="run",
            stage_id="same-stage",
            code_version="dev",
            preferred_format="npz",
        )
        for obj in (base, shifted_clock, shifted_coordinates)
    ]
    assert len({record.artifact_id for record in records}) == 3
    for record in records:
        assert record.timebase["temporal_digest"]
        assert record.timebase["temporal_context"] == _context().to_dict()


def test_catalog_hashes_zarr_directory_payload(tmp_path) -> None:
    pytest.importorskip("zarr")
    feature = FeatureSeries(
        values=np.ones((2, 1), dtype=np.float32),
        times_s=np.array([0.0, 0.5]),
        metadata=_metadata("zarr-catalog"),
        timebase=TimebaseSpec(
            kind="windows",
            reference="scan:run-01",
            window_s=0.5,
        ),
        temporal_context=_context(),
    )
    catalog = Catalog(tmp_path / "catalog-zarr")

    record = catalog.put(
        feature,
        run_id="run",
        stage_id="zarr-stage",
        code_version="dev",
        preferred_format="zarr",
    )

    payload = catalog.artifacts_dir / record.artifact_id / "feature_series.zarr"
    assert payload.is_dir()
    metadata = catalog._read_metadata_payload(record.artifact_id)
    assert metadata["payload"]["bytes"] > 0
    assert len(metadata["payload"]["sha256"]) == 64

    manifest = catalog.export_manifest(tmp_path / "zarr-manifest.json")
    imported = Catalog(tmp_path / "imported-zarr")
    assert imported.import_manifest(
        manifest,
        source_root=catalog.root,
        copy_artifacts=True,
    ) == 1
    imported_payload = imported.root / record.path
    assert imported_payload.is_dir()
    assert Catalog._path_sha256(imported_payload) == metadata["payload"]["sha256"]
