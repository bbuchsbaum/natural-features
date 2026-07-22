# fMRI Query API Reference

Stable top-level symbols:

- `RunGrid`
- `ExperimentGrid`
- `build_experiment_grid(...)`
- `query_feature_window(...)`
- `query_feature_window_tr(...)`
- `query_feature_zoo_window_tr(...)`

Available from:

```python
from natural_features import (
    RunGrid,
    ExperimentGrid,
    build_experiment_grid,
    query_feature_window,
    query_feature_window_tr,
    query_feature_zoo_window_tr,
)
```

## `build_experiment_grid(...)`

Builds a run-aware TR grid.

Key args:

- `tr_s`: TR in seconds
- `n_trs_by_run`: list of run lengths
- `run_starts_s` or (`start_s` + `run_gap_s`)
- `feature_to_experiment_by_run` for explicit affine clock mappings
- compatibility `feature_t0_s` or `feature_t0_by_run` shorthands

Returns:

- `ExperimentGrid` containing `RunGrid` entries with:
  - `run_index`, `tr_s`, `n_trs`, `start_s`, named clocks, and `feature_to_experiment`

## `query_feature_window(...)`

Slices raw feature times for one run/window.

Args:

- `feature`: `FeatureSeries`
- `grid`: `ExperimentGrid`
- `run_index`: target run
- `t_start_s`, `t_end_s`
- `relative_to_run`: interpret start/end relative to run start
- `output_time`: `"absolute" | "run_relative" | "feature"`

Returns:

- `FeatureSeries` with:
  - `times_s`: according to `output_time`
  - `values`: subset of rows in requested interval
  - metadata includes extractor provenance (`fmri.query.window`)

## `query_feature_window_tr(...)`

Run-aware window query sampled onto TR grid.

Additional args:

- `method`: resampling method (`mean`, `nearest`, `linear`)

Returns:

- `FeatureSeries` with:
  - one row per selected TR in the window
  - `timebase.kind="windows"` and TR stride/window metadata
  - metadata includes extractor provenance (`fmri.query.window_tr`)

## `query_feature_zoo_window_tr(...)`

Applies `query_feature_window_tr(...)` to a dict of feature spaces.

Args:

- `zoo`: `dict[str, FeatureSeries]`
- same run/window args as above

Returns:

- `dict[str, FeatureSeries]` keyed by feature-space name.
