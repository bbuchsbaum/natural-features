"""Run-aware querying helpers for feature spaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from natural_features.core.feature_bundle import temporal_object_in_clock
from natural_features.core.feature_types import FeatureSeries
from natural_features.core.timebase import (
    ClockMap,
    ClockRef,
    STIMULUS_CLOCK,
    TemporalContext,
    TimebaseSpec,
)
from natural_features.features.common import extractor_metadata

from .resample import resample_feature_series


@dataclass(frozen=True)
class RunGrid:
    run_index: int
    tr_s: float
    n_trs: int
    start_s: float
    feature_t0_s: float = 0.0
    feature_clock: ClockRef | str = STIMULUS_CLOCK
    experiment_clock: ClockRef | str = "experiment"
    feature_to_experiment: ClockMap | None = None

    def __post_init__(self) -> None:
        tr_s = float(self.tr_s)
        n_trs = int(self.n_trs)
        start_s = float(self.start_s)
        feature_t0_s = float(self.feature_t0_s)
        if not np.isfinite(tr_s) or tr_s <= 0:
            raise ValueError("tr_s must be a positive finite value")
        if n_trs <= 0 or n_trs != self.n_trs:
            raise ValueError("n_trs must be a positive integer")
        if not np.isfinite(start_s):
            raise ValueError("start_s must be finite")
        if not np.isfinite(feature_t0_s):
            raise ValueError("feature_t0_s must be finite")
        feature_clock = ClockRef(self.feature_clock)
        experiment_clock = ClockRef(self.experiment_clock)
        mapping = self.feature_to_experiment
        if mapping is None:
            mapping = ClockMap(
                feature_clock,
                experiment_clock,
                offset_s=feature_t0_s,
            )
        else:
            if mapping.source != feature_clock or mapping.target != experiment_clock:
                raise ValueError("feature_to_experiment endpoints must match the declared clocks")
            if feature_t0_s != 0.0 and (
                not np.isclose(mapping.scale, 1.0)
                or not np.isclose(mapping.offset_s, feature_t0_s)
            ):
                raise ValueError("feature_t0_s conflicts with feature_to_experiment")
            object.__setattr__(self, "feature_t0_s", float(mapping.offset_s))
        object.__setattr__(self, "tr_s", tr_s)
        object.__setattr__(self, "n_trs", n_trs)
        object.__setattr__(self, "start_s", start_s)
        object.__setattr__(self, "feature_clock", feature_clock)
        object.__setattr__(self, "experiment_clock", experiment_clock)
        object.__setattr__(self, "feature_to_experiment", mapping)

    @property
    def end_s(self) -> float:
        return float(self.start_s + (self.tr_s * self.n_trs))

    @property
    def times_s(self) -> np.ndarray:
        return self.start_s + (np.arange(self.n_trs, dtype=np.float64) * self.tr_s)

    @property
    def run_clock(self) -> ClockRef:
        return ClockRef(f"scan:run-{self.run_index:02d}")

    @property
    def temporal_context(self) -> TemporalContext:
        assert self.feature_to_experiment is not None
        return TemporalContext(
            (
                self.feature_to_experiment,
                ClockMap(self.experiment_clock, self.run_clock, offset_s=-float(self.start_s)),
            )
        )


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
    feature_clock: ClockRef | str = STIMULUS_CLOCK,
    experiment_clock: ClockRef | str = "experiment",
    feature_to_experiment_by_run: Iterable[ClockMap] | None = None,
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

    if feature_to_experiment_by_run is not None:
        clock_maps = list(feature_to_experiment_by_run)
        if len(clock_maps) != len(n_trs):
            raise ValueError("feature_to_experiment_by_run length must match n_trs_by_run length")
    else:
        clock_maps = [None] * len(n_trs)

    runs = tuple(
        RunGrid(
            run_index=i + 1,
            tr_s=tr_s,
            n_trs=n_trs[i],
            start_s=starts[i],
            feature_t0_s=t0s[i],
            feature_clock=feature_clock,
            experiment_clock=experiment_clock,
            feature_to_experiment=clock_maps[i],
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


def _feature_to_experiment_map(feature: FeatureSeries, run: RunGrid) -> tuple[ClockMap, TemporalContext]:
    context = feature.temporal_context.merged(run.temporal_context)
    return context.resolve(feature.clock, run.experiment_clock), context


def _output_reference(run: RunGrid, feature: FeatureSeries, output_time: str) -> ClockRef:
    if output_time == "feature":
        return feature.clock
    if output_time == "absolute":
        return ClockRef(run.experiment_clock)
    if output_time == "run_relative":
        return run.run_clock
    raise ValueError("output_time must be one of {'absolute','run_relative','feature'}")


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
    - `"absolute"`: express output on the experiment clock.
    - `"run_relative"`: subtract run start from output times.
    - `"feature"`: preserve the feature object's native clock.
    """

    run = grid.get_run(run_index)
    abs_start, abs_end = _resolve_abs_window(
        run,
        t_start_s=t_start_s,
        t_end_s=t_end_s,
        relative_to_run=relative_to_run,
    )
    feature_to_experiment, context = _feature_to_experiment_map(feature, run)
    experiment_to_feature = feature_to_experiment.inverse()
    feature_start = float(experiment_to_feature.apply(abs_start))
    feature_end = float(experiment_to_feature.apply(abs_end))
    m = (feature.times_s >= feature_start) & (feature.times_s < feature_end)
    feature_times = feature.times_s[m]
    native_bounds = feature.time_bounds_s[m] if feature.time_bounds_s is not None else None
    output_reference = _output_reference(run, feature, output_time)
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
    native = FeatureSeries(
        values=vals,
        times_s=feature_times,
        dims=feature.dims,
        coords=feature.coords,
        metadata=md,
        timebase=feature.timebase,
        time_bounds_s=native_bounds,
        temporal_context=context,
    )
    transformed = temporal_object_in_clock(
        native,
        output_reference,
        context=context,
    )
    assert isinstance(transformed, FeatureSeries)
    return transformed


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
    feature_to_experiment, context = _feature_to_experiment_map(feature, run)
    feature_grid = np.asarray(feature_to_experiment.inverse().apply(query_grid), dtype=np.float64)
    feature_tr_s = run.tr_s / feature_to_experiment.scale
    sampled = resample_feature_series(
        feature,
        tr_s=feature_tr_s,
        method=method,
        time_grid_s=feature_grid,
    )
    if output_time == "feature":
        out_times = feature_grid
    elif output_time == "run_relative":
        out_times = query_grid - run.start_s
    elif output_time == "absolute":
        out_times = query_grid
    else:
        _output_reference(run, feature, output_time)
        raise AssertionError("unreachable")
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
    output_stride = run.tr_s if output_time != "feature" else feature_tr_s
    return FeatureSeries(
        values=sampled.values,
        times_s=out_times,
        dims=sampled.dims,
        coords=sampled.coords,
        metadata=md,
        timebase=TimebaseSpec(
            kind="windows",
            reference=_output_reference(run, feature, output_time),
            stride_s=output_stride,
            window_s=output_stride,
            alignment="center",
        ),
        temporal_context=context,
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
