# Public API Compatibility Policy

This package distinguishes between a **stable public API** and **experimental/internal surfaces**.

## Stable API

The stable contract is defined by `natural_features.public_api` and re-exported from `natural_features`.

Stable symbols are enumerated by:

- `natural_features.STABLE_EXPORTS`
- `natural_features.__all__`

A breaking change to these stable symbols requires incrementing:

- `natural_features.API_COMPAT_VERSION`

### Current stable symbols

- `FeatureSeries`
- `EventSeries`
- `TrackSeries`
- `ClockRef`
- `ClockMap`
- `SupportSpec`
- `TemporalContext`
- `TimebaseSpec`
- `FeatureBundle`
- `TemporalPayload`
- `temporal_object_in_clock`
- `FrameTimeline`
- `Timeline`
- `FeatureAlignment`
- `ExtractFeaturesResult`
- `AlignedFeatureSet`
- `VideoTextResult`
- `RunGrid`
- `ExperimentGrid`
- `build_experiment_grid`
- `query_feature_window`
- `query_feature_window_tr`
- `query_feature_zoo_window_tr`
- `extract_acoustic_phonetics`
- `available_features`
- `feature_catalog`
- `plan_features`
- `extract_features`
- `extract_audio_files`
- `extract_audio_dir`
- `extract_multiscale_language`
- `extract_video_text`

## Experimental / Internal API

Everything outside the stable symbols above is considered non-stable and may change without compatibility guarantees.

Documented experimental namespaces:

- `natural_features.features`
- `natural_features.flow`
- `natural_features.core.recipe`
- `natural_features.core.registry`

## Compatibility promise

For a fixed `API_COMPAT_VERSION`:

- Stable symbol names remain available.
- Stable symbols preserve their intended high-level behavior and return contracts.
- Any unavoidable behavior shift must be documented in release notes.

When breaking stable contracts:

1. Increment `API_COMPAT_VERSION`.
2. Add migration notes.
3. Update tests locking public API exports.
