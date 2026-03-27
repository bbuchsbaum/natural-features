"""Run-aware querying helpers for feature spaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from natural_features.core.feature_types import FeatureSeries
from natural_features.core.timebase import TimebaseSpec
from natural_features.features.common import extractor_metadata

from .resample import resample_feature_series


@dataclass(frozen=True)
class RunGrid:
    run_index: int
    tr_s: float
    n_trs: int
    start_s: float
    feature_t0_s: float = 0.0

    @property
    def end_s(self) -> float:
        return float(self.start_s + (self.tr_s * self.n_trs))

    @property
    def times_s(self) -> np.ndarray:
        return self.start_s + (np.arange(self.n_trs, dtype=np.float64) * self.tr_s)


@dataclass(frozen=True)
class ExperimentGrid:
    runs: tuple[RunGrid, ...]

    def get_run(self, run_index: int) -> RunGrid:
        for run in self.runs:
            if run.run_index == int(run_index):
                return run
        raise KeyError(f"Unknown run_index={run_index}")


def build_experiment_grid(
    *,
    tr_s: float,
    n_trs_by_run: Iterable[int],
    run_starts_s: Iterable[float] | None = None,
    feature_t0_s: float = 0.0,
    feature_t0_by_run: Iterable[float] | None = None,
    start_s: float = 0.0,
    run_gap_s: float = 0.0,
) -> ExperimentGrid:
    """Build simple experiment grid with per-run TR clocks."""

    if tr_s <= 0:
        raise ValueError("tr_s must be > 0")
    n_trs = [int(x) for x in n_trs_by_run]
    if not n_trs or any(x <= 0 for x in n_trs):
        raise ValueError("n_trs_by_run must contain positive integers")

    if run_starts_s is not None:
        starts = [float(x) for x in run_starts_s]
        if len(starts) != len(n_trs):
            raise ValueError("run_starts_s length must match n_trs_by_run length")
    else:
        starts = []
        cur = float(start_s)
        for n in n_trs:
            starts.append(cur)
            cur = cur + (tr_s * n) + float(run_gap_s)

    if feature_t0_by_run is not None:
        t0s = [float(x) for x in feature_t0_by_run]
        if len(t0s) != len(n_trs):
            raise ValueError("feature_t0_by_run length must match n_trs_by_run length")
    else:
        t0s = [float(feature_t0_s)] * len(n_trs)

    runs = tuple(
        RunGrid(
            run_index=i + 1,
            tr_s=tr_s,
            n_trs=n_trs[i],
            start_s=starts[i],
            feature_t0_s=t0s[i],
        )
        for i in range(len(n_trs))
    )
    return ExperimentGrid(runs=runs)


def _resolve_abs_window(
    run: RunGrid,
    *,
    t_start_s: float,
    t_end_s: float,
    relative_to_run: bool,
) -> tuple[float, float]:
    if t_end_s <= t_start_s:
        raise ValueError("t_end_s must be > t_start_s")
    if relative_to_run:
        abs_start = run.start_s + float(t_start_s)
        abs_end = run.start_s + float(t_end_s)
    else:
        abs_start = float(t_start_s)
        abs_end = float(t_end_s)
    return abs_start, abs_end


def query_feature_window(
    feature: FeatureSeries,
    grid: ExperimentGrid,
    *,
    run_index: int,
    t_start_s: float,
    t_end_s: float,
    relative_to_run: bool = True,
    output_time: str = "absolute",
) -> FeatureSeries:
    """Slice a raw feature window for one run.

    `output_time`:
    - `"absolute"`: keep stimulus/global time.
    - `"run_relative"`: subtract run start from output times.
    """

    run = grid.get_run(run_index)
    abs_start, abs_end = _resolve_abs_window(
        run,
        t_start_s=t_start_s,
        t_end_s=t_end_s,
        relative_to_run=relative_to_run,
    )
    feature_start = abs_start - run.feature_t0_s
    feature_end = abs_end - run.feature_t0_s
    m = (feature.times_s >= feature_start) & (feature.times_s < feature_end)
    feature_times = feature.times_s[m]
    query_abs_times = feature_times + run.feature_t0_s
    if output_time == "feature":
        out_times = feature_times
    elif output_time == "absolute":
        out_times = query_abs_times
    elif output_time == "run_relative":
        out_times = query_abs_times - run.start_s
    else:
        raise ValueError("output_time must be one of {'absolute','run_relative','feature'}")
    vals = feature.values[m]
    md = dict(feature.metadata)
    md.update(
        extractor_metadata(
            "fmri.query.window",
            params={
                "run_index": run_index,
                "t_start_s": float(t_start_s),
                "t_end_s": float(t_end_s),
                "relative_to_run": bool(relative_to_run),
                "feature_t0_s": float(run.feature_t0_s),
                "output_time": output_time,
            },
        )
    )
    return FeatureSeries(
        values=vals,
        times_s=out_times,
        dims=feature.dims,
        coords=feature.coords,
        metadata=md,
        timebase=feature.timebase,
    )


def query_feature_window_tr(
    feature: FeatureSeries,
    grid: ExperimentGrid,
    *,
    run_index: int,
    t_start_s: float,
    t_end_s: float,
    relative_to_run: bool = True,
    method: str = "mean",
    output_time: str = "absolute",
) -> FeatureSeries:
    """Slice a feature window and sample on the run TR grid."""

    run = grid.get_run(run_index)
    abs_start, abs_end = _resolve_abs_window(
        run,
        t_start_s=t_start_s,
        t_end_s=t_end_s,
        relative_to_run=relative_to_run,
    )
    query_grid = run.times_s
    query_grid = query_grid[(query_grid >= abs_start) & (query_grid < abs_end)]
    feature_grid = query_grid - run.feature_t0_s
    sampled = resample_feature_series(feature, tr_s=run.tr_s, method=method, time_grid_s=feature_grid)
    if output_time == "feature":
        out_times = feature_grid
    elif output_time == "run_relative":
        out_times = query_grid - run.start_s
    elif output_time == "absolute":
        out_times = query_grid
    else:
        raise ValueError("output_time must be one of {'absolute','run_relative','feature'}")
    md = dict(sampled.metadata)
    md.update(
        extractor_metadata(
            "fmri.query.window_tr",
            params={
                "run_index": run_index,
                "t_start_s": float(t_start_s),
                "t_end_s": float(t_end_s),
                "relative_to_run": bool(relative_to_run),
                "feature_t0_s": float(run.feature_t0_s),
                "method": method,
                "output_time": output_time,
            },
        )
    )
    return FeatureSeries(
        values=sampled.values,
        times_s=out_times,
        dims=sampled.dims,
        coords=sampled.coords,
        metadata=md,
        timebase=TimebaseSpec(kind="windows", stride_s=run.tr_s, window_s=run.tr_s, alignment="center"),
    )


def query_feature_zoo_window_tr(
    zoo: dict[str, FeatureSeries],
    grid: ExperimentGrid,
    *,
    run_index: int,
    t_start_s: float,
    t_end_s: float,
    relative_to_run: bool = True,
    method: str = "mean",
    output_time: str = "absolute",
) -> dict[str, FeatureSeries]:
    """Apply run-aware TR query to every feature in a collection."""

    return {
        name: query_feature_window_tr(
            fs,
            grid,
            run_index=run_index,
            t_start_s=t_start_s,
            t_end_s=t_end_s,
            relative_to_run=relative_to_run,
            method=method,
            output_time=output_time,
        )
        for name, fs in zoo.items()
    }
