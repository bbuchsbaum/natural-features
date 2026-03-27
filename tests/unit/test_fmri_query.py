from __future__ import annotations

import numpy as np

from natural_features.core.feature_types import FeatureSeries
from natural_features.features.common import extractor_metadata
from natural_features.fmri.query import (
    build_experiment_grid,
    query_feature_window,
    query_feature_window_tr,
    query_feature_zoo_window_tr,
)


def _feature(duration_s: float = 140.0, step_s: float = 0.1) -> FeatureSeries:
    times = np.arange(0.0, duration_s, step_s, dtype=np.float64)
    vals = np.stack([np.sin(times), np.cos(times)], axis=1).astype(np.float32)
    return FeatureSeries(
        values=vals,
        times_s=times,
        metadata=extractor_metadata("test.feature"),
        coords={"feature": ["sin", "cos"]},
    )


def test_build_experiment_grid_sequential_runs() -> None:
    g = build_experiment_grid(tr_s=1.5, n_trs_by_run=[100, 120], start_s=0.0, run_gap_s=5.0)
    assert len(g.runs) == 2
    assert g.runs[0].run_index == 1
    assert g.runs[1].run_index == 2
    assert np.isclose(g.runs[1].start_s, (1.5 * 100) + 5.0)


def test_query_feature_window_raw_run_relative() -> None:
    fs = _feature()
    g = build_experiment_grid(tr_s=1.5, n_trs_by_run=[100], start_s=0.0)
    out = query_feature_window(
        fs,
        g,
        run_index=1,
        t_start_s=1.3,
        t_end_s=36.7,
        relative_to_run=True,
        output_time="run_relative",
    )
    assert out.values.shape[0] > 0
    assert out.times_s.min() >= 1.3
    assert out.times_s.max() < 36.7


def test_query_feature_window_with_feature_t0_offset() -> None:
    fs = _feature()
    g = build_experiment_grid(
        tr_s=1.0,
        n_trs_by_run=[100],
        start_s=0.0,
        feature_t0_s=22.3,  # feature t=0 occurs at scan t=22.3
    )
    out = query_feature_window(
        fs,
        g,
        run_index=1,
        t_start_s=24.0,
        t_end_s=26.0,
        relative_to_run=True,
        output_time="feature",
    )
    # query [24,26) scan -> feature [1.7,3.7)
    assert out.values.shape[0] > 0
    assert out.times_s.min() >= 1.7 - 1e-9
    assert out.times_s.max() < 3.7 + 1e-9


def test_query_feature_window_tr_on_run_grid() -> None:
    fs = _feature()
    g = build_experiment_grid(tr_s=2.0, n_trs_by_run=[80], start_s=0.0)
    out = query_feature_window_tr(
        fs,
        g,
        run_index=1,
        t_start_s=1.3,
        t_end_s=36.7,
        relative_to_run=True,
        method="mean",
        output_time="run_relative",
    )
    assert out.values.shape[0] > 0
    # run-grid times in this window should be 2,4,...,36 (run-relative)
    assert np.isclose(out.times_s[0], 2.0)
    assert np.isclose(out.times_s[-1], 36.0)


def test_query_feature_window_tr_with_feature_t0_offset() -> None:
    times = np.arange(0.0, 50.0, 0.1, dtype=np.float64)
    vals = times.reshape(-1, 1).astype(np.float32)  # value == feature time
    fs = FeatureSeries(
        values=vals,
        times_s=times,
        metadata=extractor_metadata("test.linear_time"),
        coords={"feature": ["t"]},
    )
    g = build_experiment_grid(
        tr_s=2.0,
        n_trs_by_run=[40],
        start_s=0.0,
        feature_t0_s=22.3,
    )
    out = query_feature_window_tr(
        fs,
        g,
        run_index=1,
        t_start_s=24.3,
        t_end_s=30.3,
        relative_to_run=True,
        method="linear",
        output_time="run_relative",
    )
    # run grid points in [24.3,30.3): 26,28,30
    np.testing.assert_allclose(out.times_s, np.array([26.0, 28.0, 30.0]), atol=1e-6)
    # mapped feature-grid points: 3.7, 5.7, 7.7
    np.testing.assert_allclose(out.values[:, 0], np.array([3.7, 5.7, 7.7]), atol=1e-5)


def test_query_feature_zoo_window_tr_multiple_spaces() -> None:
    fs = _feature()
    zoo = {"f1": fs, "f2": fs}
    g = build_experiment_grid(tr_s=1.0, n_trs_by_run=[50], start_s=0.0)
    out = query_feature_zoo_window_tr(
        zoo,
        g,
        run_index=1,
        t_start_s=3.0,
        t_end_s=10.0,
        relative_to_run=True,
    )
    assert set(out.keys()) == {"f1", "f2"}
    assert out["f1"].values.shape[0] == out["f2"].values.shape[0]
    assert np.allclose(out["f1"].times_s, out["f2"].times_s)
