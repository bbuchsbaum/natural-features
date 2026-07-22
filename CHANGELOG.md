# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Explicit temporal contracts: `ClockRef`, affine `ClockMap`, `SupportSpec`,
  `TemporalContext`, and `TimebaseSpec` serialization.
- `FeatureBundle` and dependency-light `TemporalPayload` handoff objects that
  preserve heterogeneous native sampling grids without resampling.
- Clock-aware timeline alignment, cross-clock table safeguards, temporal
  storage round trips, and temporal digests in catalog artifact identity.
- Native-time specification, cookbook, and downstream modeling boundary docs.
- Public API compatibility contract (`natural_features.public_api`) and policy documentation.
- Unified strict/fallback execution mode with provenance metadata.
- Hardened recipe/schema validation with output contract checks.
- Manifest v2 provenance exports with payload integrity hashes.
- Local semantic fallback provider (`local_bow`) and provider quality upgrades.
- Tier-A golden regression fixture and generator workflow.
- CLI ergonomics: `nf validate`, `nf preset-list`, `nf preset-show`.
- Onboarding and docs index.
- Release discipline docs and release-check script.

### Changed
- Stimuli, feature objects, extraction workflows, and NPZ/Zarr/Parquet storage
  now preserve explicit clock, support, row bounds, and temporal context.
- `RunGrid` and fMRI query compatibility helpers use explicit clock mappings
  internally; `feature_t0_s` remains a compatibility shorthand.
- Multiscale provider fallback now prefers `local_bow` over random local hash fallback.
- API compatibility contract advanced to version 2: named feature methods and
  stable workflows now fail fast when an optional backend is unavailable.
- Deterministic proxies require an explicit `execution_mode="fallback"` (or
  legacy `strict_dependency=False`) request.
- Explicit alignment passthrough via `backend="none"` is a method choice and is
  no longer reported as fallback execution.

### Migration Notes
- Existing constructors remain valid and default to the `"stimulus"` clock.
  New pipelines should set `timebase.reference`, retain native grids in a
  `FeatureBundle`, and express stimulus/scan offsets with `ClockMap`.
- `feature_t0_s` and `feature_t0_by_run` remain supported, but new fMRI-facing
  code should pass `feature_to_experiment_by_run` mappings or consume a
  `TemporalPayload` downstream.
- The `features.hrf` recipe route is deprecated. Keep native features in this
  package and perform HRF convolution, interpolation to TRs, lags, and design
  construction in `fmrimod`, `fmrireg`, `fmridesign`, or another modeling layer.
- Code that deliberately uses deterministic proxies must now pass
  `execution_mode="fallback"`. Omit the setting to require the named method.
- Legacy `strict_dependency=False` remains supported as an explicit fallback
  request; migrate saved recipes to `execution_mode="fallback"` when practical.
- If you consume exported manifests, prefer `manifest_version=2` fields (`payload_sha256`, `payload_bytes`).
