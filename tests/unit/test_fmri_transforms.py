from __future__ import annotations

import numpy as np

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
    lagged = add_lags(f, [0, 1, 2])
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
