# Run-Aware fMRI Querying

For raw feature matrices, use extractors/workflows directly.

For convenient run-aware querying on an fMRI grid (run index + TR + nTRs),
use the stable top-level API (`natural_features`) or `natural_features.fmri.query`.

## Example

```python
from natural_features import (
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

## Stimulus/Scan Time Offset

If feature time `t=0` is not scan/run `t=0` (e.g. stimulus starts at scan
time `22.3s`), set `feature_t0_s` (or per-run `feature_t0_by_run`) in the grid:

```python
grid = build_experiment_grid(
    tr_s=1.5,
    n_trs_by_run=[300, 280],
    start_s=0.0,
    feature_t0_s=22.3,  # feature t=0 corresponds to scan t=22.3
)
```

Then query in run/scan time as usual; mapping to feature time is handled internally.

## Notes

- `query_feature_window(...)` returns raw timepoints in the requested interval.
- `query_feature_window_tr(...)` restricts to the run TR grid and resamples.
- `output_time`:
  - `"absolute"`: scan/run absolute times
  - `"run_relative"`: relative to run start
  - `"feature"`: feature timeline times
