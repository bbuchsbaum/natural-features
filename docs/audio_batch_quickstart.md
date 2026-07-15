# Audio Batch Quickstart

For short clips (2-6s), use the high-level workflow API:

```python
from natural_features.workflows.audio_batch import extract_audio_dir

result = extract_audio_dir(
    "data/audio_clips",
    pattern="*.wav",
    resolution_s=1.0,  # 0.5, 1.0, 2.0...
    selected_features=["rms", "mfcc", "spectral_stats", "vad"],
    feature_params={
        "mfcc": {"n_mfcc": 13, "n_mels": 40, "include_deltas": True},
        "vad": {"threshold": 0.5},
    },
    as_dataframe=True,
    collapse="mean+sd",  # also: "mean", "min", "max", or list like ["mean", "max"]
)

# Per-file outputs
for file_id, fr in result.files.items():
    X = fr.matrix               # numpy matrix (time x features)
    names = fr.feature_names    # column names
    df = fr.dataframe           # per-file dataframe

# Combined dataframe across all files
long_df = result.long_dataframe

# One row per file with collapsed stats over time
collapsed_df = result.collapsed_dataframe
```

Notes:

- Resolution is implemented via shared resampling grid.
- Feature names are prefixed by feature family (`rms.*`, `mfcc.*`, etc.).
- Optional named methods such as openSMILE fail fast by default. Set
  `execution_mode="fallback"` only when you deliberately want a documented
  proxy during pipeline development.
- Use `as_dataframe=False` for matrix-only workflows.
- `collapse` aggregates each feature across time per file.
