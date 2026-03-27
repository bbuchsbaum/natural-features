from __future__ import annotations

import numpy as np
import pytest

from natural_features.core.feature_types import EventSeries, FeatureSeries, TrackSeries


def _meta() -> dict[str, str]:
    return {"extractor_id": "x", "params_hash": "y"}


def test_feature_series_validates_shape_and_times() -> None:
    fs = FeatureSeries(
        values=np.ones((5, 3), dtype=np.float32),
        times_s=np.linspace(0.0, 0.4, 5),
        metadata=_meta(),
    )
    assert fs.shape == (5, 3)


def test_feature_series_rejects_non_monotonic_time() -> None:
    with pytest.raises(ValueError, match="monotonic"):
        FeatureSeries(
            values=np.ones((3, 2), dtype=np.float32),
            times_s=np.array([0.0, 0.2, 0.1]),
            metadata=_meta(),
        )


def test_event_series_rejects_negative_duration() -> None:
    with pytest.raises(ValueError, match="offset_s must be >="):
        EventSeries(
            onset_s=np.array([0.2]),
            offset_s=np.array([0.1]),
            metadata=_meta(),
        )


def test_track_series_validates_track_axis() -> None:
    ts = TrackSeries(
        times_s=np.array([0.0, 0.1, 0.2]),
        track_id=np.array(["t1", "t2"]),
        values=np.ones((3, 2, 4), dtype=np.float32),
        metadata=_meta(),
    )
    assert ts.values.shape == (3, 2, 4)
