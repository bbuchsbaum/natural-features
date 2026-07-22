from __future__ import annotations

import numpy as np
import pytest

from natural_features.core.feature_types import EventSeries, FeatureSeries
from natural_features.fmri.design import add_lags, concat_feature_series
from natural_features.fmri.hrf import hrf_convolve, hrf_kernel
from natural_features.fmri.render import render_events
from natural_features.fmri.resample import build_tr_grid, resample_feature_series


def _feature(n_t: int = 20, n_f: int = 3) -> FeatureSeries:
    t = np.arange(n_t, dtype=np.float32) * 0.5
    x = np.stack([np.sin(t), np.cos(t), np.sin(0.5 * t)], axis=1)[:, :n_f].astype(np.float32)
    return FeatureSeries(values=x, times_s=t, metadata={"extractor_id": "a", "params_hash": "b"})


def test_resample_and_hrf() -> None:
    f = _feature()
    r = resample_feature_series(f, tr_s=1.0, method="mean")
    assert r.values.shape[0] <= f.values.shape[0]
    h = hrf_kernel(1.0, kind="glover")
    assert h.ndim == 1
    assert np.isclose(np.max(np.abs(h)), 1.0, atol=1e-6)
    c = hrf_convolve(r, tr_s=1.0, kind="spm")
    assert c.values.shape == r.values.shape


def test_render_events_and_lags_concat() -> None:
    f = _feature(n_t=10, n_f=2)
    grid = f.times_s
    events = EventSeries(
        onset_s=np.array([1.0, 2.0, 2.5]),
        offset_s=np.array([1.0, 3.0, 2.5]),
        confidence=np.array([0.5, 0.8, 0.9]),
        metadata={"extractor_id": "e", "params_hash": "p"},
    )
    e = render_events(events, grid, mode="boxcar", value="duration")
    windowed = resample_feature_series(
        f,
        tr_s=0.5,
        method="nearest",
        time_grid_s=grid,
    )
    lagged = add_lags(windowed, [0, 1, 2])
    dm = concat_feature_series([lagged, e], standardize=True, add_intercept=True)
    assert dm.values.shape[0] == len(grid)
    assert dm.values.shape[1] == lagged.values.shape[1] + 1 + 1


def test_build_tr_grid_half_open_interval() -> None:
    grid = build_tr_grid(duration_s=10.0, tr_s=2.0, start_s=0.0)
    np.testing.assert_allclose(grid, np.array([0.0, 2.0, 4.0, 6.0, 8.0]))


def test_nearest_resample_selects_true_nearest_point() -> None:
    fs = FeatureSeries(
        values=np.array([[10.0], [20.0]], dtype=np.float32),
        times_s=np.array([0.0, 2.0], dtype=np.float64),
        metadata={"extractor_id": "a", "params_hash": "b"},
    )
    out = resample_feature_series(
        fs,
        tr_s=1.0,
        method="nearest",
        time_grid_s=np.array([0.9, 1.1], dtype=np.float64),
    )
    np.testing.assert_allclose(out.values[:, 0], np.array([10.0, 20.0], dtype=np.float32))


def test_render_events_respects_non_uniform_grid() -> None:
    events = EventSeries(
        onset_s=np.array([2.1], dtype=np.float64),
        offset_s=np.array([2.1], dtype=np.float64),
        metadata={"extractor_id": "e", "params_hash": "p"},
    )
    grid = np.array([0.0, 1.0, 3.0], dtype=np.float64)
    out = render_events(events, grid, mode="impulse", value="count")
    np.testing.assert_allclose(out.values[:, 0], np.array([0.0, 0.0, 1.0], dtype=np.float32))


def test_render_events_confidence_requires_confidence_column() -> None:
    events = EventSeries(
        onset_s=np.array([0.1], dtype=np.float64),
        offset_s=np.array([0.2], dtype=np.float64),
        metadata={"extractor_id": "e", "params_hash": "p"},
    )
    with np.testing.assert_raises(ValueError):
        _ = render_events(events, np.array([0.0, 1.0]), mode="impulse", value="confidence")


def test_add_lags_is_causal_for_short_and_overlong_lags() -> None:
    feature = FeatureSeries(
        values=np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
        times_s=np.array([0.0, 1.0, 2.0]),
        coords={"feature": ["signal"]},
        metadata={"extractor_id": "analytic", "params_hash": "fixed"},
    )

    lagged = add_lags(feature, [5, 1, 0, 3, 1])

    assert lagged.coords["feature"] == [
        "signal_lag0",
        "signal_lag1",
        "signal_lag3",
        "signal_lag5",
    ]
    np.testing.assert_array_equal(
        lagged.values,
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [2.0, 1.0, 0.0, 0.0],
                [3.0, 2.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        err_msg="Lagging must be causal, zero padded, and length preserving",
    )
    with pytest.raises(ValueError, match="lags cannot be empty"):
        add_lags(feature, [])
    with pytest.raises(ValueError, match="Negative lags"):
        add_lags(feature, [-1])


def test_design_assembly_rejects_incompatible_domains() -> None:
    times = np.array([0.0, 1.0])
    tensor = FeatureSeries(
        values=np.ones((2, 2, 2), dtype=np.float32),
        times_s=times,
        dims=("time", "layer", "feature"),
        metadata={"extractor_id": "tensor", "params_hash": "fixed"},
    )
    with pytest.raises(ValueError, match="2-D"):
        add_lags(tensor, [0])
    with pytest.raises(ValueError, match="cannot be empty"):
        concat_feature_series([])

    left = FeatureSeries(
        values=np.ones((2, 1), dtype=np.float32),
        times_s=times,
        metadata={"extractor_id": "left", "params_hash": "fixed"},
    )
    right = FeatureSeries(
        values=np.ones((2, 1), dtype=np.float32),
        times_s=times + 0.25,
        metadata={"extractor_id": "right", "params_hash": "fixed"},
    )
    with pytest.raises(ValueError, match="same time grid"):
        concat_feature_series([left, right])


def test_concat_standardizes_varying_columns_and_annihilates_constants() -> None:
    times = np.arange(4, dtype=np.float64)
    varying = FeatureSeries(
        values=np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32),
        times_s=times,
        coords={"feature": ["varying"]},
        metadata={"extractor_id": "varying", "params_hash": "fixed"},
    )
    constant = FeatureSeries(
        values=np.full((4, 1), 7.0, dtype=np.float32),
        times_s=times,
        coords={"feature": ["constant"]},
        metadata={"extractor_id": "constant", "params_hash": "fixed"},
    )

    design = concat_feature_series([varying, constant])

    np.testing.assert_allclose(design.values[:, 0].mean(), 0.0, atol=1e-7)
    np.testing.assert_allclose(design.values[:, 0].std(), 1.0, rtol=1e-6)
    np.testing.assert_array_equal(design.values[:, 1], np.zeros(4))
    np.testing.assert_array_equal(design.values[:, 2], np.ones(4))
    assert design.coords["feature"] == ["varying", "constant", "intercept"]


def test_linear_resampling_is_exact_for_affine_signals_and_time_translation() -> None:
    times = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    values = np.column_stack([3.0 * times - 2.0, -0.5 * times + 4.0]).astype(np.float32)
    feature = FeatureSeries(
        values=values,
        times_s=times,
        coords={"feature": ["up", "down"]},
        metadata={"extractor_id": "affine", "params_hash": "fixed"},
    )
    grid = np.array([0.0, 0.25, 0.75, 1.5, 2.0], dtype=np.float64)

    observed = resample_feature_series(feature, tr_s=0.5, method="linear", time_grid_s=grid)
    expected = np.column_stack([3.0 * grid - 2.0, -0.5 * grid + 4.0])
    np.testing.assert_allclose(
        observed.values,
        expected,
        rtol=1e-6,
        atol=1e-6,
        err_msg="Linear interpolation must reproduce affine signals exactly",
    )

    shifted = FeatureSeries(
        values=values,
        times_s=times + 11.0,
        coords=feature.coords,
        metadata=feature.metadata,
    )
    shifted_observed = resample_feature_series(
        shifted,
        tr_s=0.5,
        method="linear",
        time_grid_s=grid + 11.0,
    )
    np.testing.assert_allclose(
        shifted_observed.values,
        observed.values,
        rtol=0.0,
        atol=0.0,
        err_msg="Resampling must be equivariant to a common time-origin shift",
    )


def test_event_rendering_conserves_count_confidence_and_interval_measure() -> None:
    events = EventSeries(
        onset_s=np.array([0.25, 1.25, 2.25]),
        offset_s=np.array([0.75, 2.75, 2.50]),
        confidence=np.array([0.2, 0.5, 0.8]),
        metadata={"extractor_id": "events", "params_hash": "fixed"},
    )
    grid = np.array([0.5, 1.5, 2.5], dtype=np.float64)

    impulse_count = render_events(events, grid, mode="impulse", value="count")
    impulse_confidence = render_events(events, grid, mode="impulse", value="confidence")
    boxcar_duration = render_events(events, grid, mode="boxcar", value="duration")

    np.testing.assert_allclose(impulse_count.values.sum(), len(events), atol=0.0)
    np.testing.assert_allclose(
        impulse_confidence.values.sum(), events.confidence.sum(), rtol=1e-6
    )
    np.testing.assert_allclose(
        boxcar_duration.values.sum(),
        np.sum(events.offset_s - events.onset_s),
        rtol=1e-6,
        atol=1e-7,
        err_msg="Boxcar rendering must conserve event measure over a covering grid",
    )


def test_boxcar_count_and_confidence_use_positive_measure_overlap() -> None:
    events = EventSeries(
        onset_s=np.array([0.25, 1.50]),
        offset_s=np.array([1.25, 1.75]),
        confidence=np.array([0.2, 0.7]),
        metadata={"extractor_id": "events", "params_hash": "fixed"},
    )
    grid = np.array([0.5, 1.5])

    counts = render_events(events, grid, mode="boxcar", value="count")
    confidence = render_events(events, grid, mode="boxcar", value="confidence")

    np.testing.assert_array_equal(
        counts.values[:, 0],
        [1.0, 2.0],
        err_msg="An interval contributes once to every cell with positive-measure overlap",
    )
    np.testing.assert_allclose(
        confidence.values[:, 0],
        [0.2, 0.9],
        rtol=1e-6,
        err_msg="Boxcar confidence must sum confidence over overlapping events",
    )


def test_event_rendering_validates_requests_even_when_events_are_empty() -> None:
    empty = EventSeries(
        onset_s=np.array([], dtype=np.float64),
        offset_s=np.array([], dtype=np.float64),
        metadata={"extractor_id": "empty", "params_hash": "fixed"},
    )

    with pytest.raises(ValueError, match="Unsupported mode"):
        render_events(empty, np.array([0.0]), mode="unknown")
    with pytest.raises(ValueError, match="Unsupported value"):
        render_events(empty, np.array([0.0]), mode="boxcar", value="unknown")
    with pytest.raises(ValueError, match="confidence"):
        render_events(empty, np.array([0.0]), value="confidence")
    with pytest.raises(ValueError, match="strictly increasing"):
        render_events(empty, np.array([0.0, 0.0]))
    with pytest.raises(ValueError, match="1-D"):
        render_events(empty, np.array([[0.0, 1.0]]))
    with pytest.raises(ValueError, match="finite"):
        render_events(empty, np.array([0.0, np.nan]))
