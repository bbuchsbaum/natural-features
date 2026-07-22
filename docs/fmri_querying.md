# Run-Aware fMRI Querying

For raw feature matrices, use extractors/workflows directly.

For compatibility run-aware querying on an fMRI grid (run index + TR + nTRs),
use the stable top-level API (`natural_features`) or `natural_features.fmri.query`.
New pipelines should preserve native feature grids in `FeatureBundle` and let
downstream modeling libraries own TR resampling, HRF convolution, and design.

## Example

```python
from natural_features import (
    ClockMap,
    build_experiment_grid,
    query_feature_window,
    query_feature_window_tr,
    query_feature_zoo_window_tr,
)

# Suppose this is a FeatureSeries from any extractor:
# fs.times_s in absolute stimulus seconds.
fs = ...

# Two runs: run1 has 300 TRs, run2 has 280 TRs, TR=1.5s, 10s gap.
grid = build_experiment_grid(
    tr_s=1.5,
    n_trs_by_run=[300, 280],
    start_s=0.0,
    run_gap_s=10.0,
    feature_to_experiment_by_run=[
        ClockMap("stimulus", "experiment"),
        ClockMap("stimulus", "experiment"),
    ],
)

# Raw window in run 1, relative to run start:
raw = query_feature_window(
    fs,
    grid,
    run_index=1,
    t_start_s=1.3,
    t_end_s=36.7,
    relative_to_run=True,
    output_time="run_relative",
)

# Same window, sampled at run TR grid:
tr = query_feature_window_tr(
    fs,
    grid,
    run_index=1,
    t_start_s=1.3,
    t_end_s=36.7,
    relative_to_run=True,
    method="mean",          # or "nearest", "linear"
    output_time="run_relative",
)

# Query multiple feature spaces at once:
zoo = {"audio_rms": fs1, "mfcc": fs2, "vision": fs3}
zoo_tr = query_feature_zoo_window_tr(
    zoo,
    grid,
    run_index=1,
    t_start_s=1.3,
    t_end_s=36.7,
    relative_to_run=True,
)
```

## Stimulus/Scan Time Mapping

If feature time `t=30` is scan time `7`, provide the affine mapping explicitly:

```python
grid = build_experiment_grid(
    tr_s=1.5,
    n_trs_by_run=[300, 280],
    start_s=0.0,
    feature_to_experiment_by_run=[
        ClockMap("stimulus", "experiment", offset_s=-23.0),
        ClockMap("stimulus", "experiment", offset_s=-23.0),
    ],
)
```

Then query in run/scan time as usual; mapping to feature time is handled internally.

`feature_t0_s` and `feature_t0_by_run` remain supported as compatibility
shorthands for unit-scale mappings.

## Notes

- `query_feature_window(...)` returns raw timepoints in the requested interval.
- `query_feature_window_tr(...)` restricts to the run TR grid and resamples.
- `output_time`:
  - `"absolute"`: scan/run absolute times
  - `"run_relative"`: relative to run start
  - `"feature"`: feature timeline times
