# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Music feature family: `audio.music.chroma`, `audio.music.tonnetz`,
  `audio.music.onset_strength`, `audio.music.tempogram`, `audio.music.rhythm` and
  `audio.music.tonality`, covering pitch-class content, tonal centroid, key/mode
  (Krumhansl-Schmuckler, including the full 24-key correlation profile), and the
  onset/tempo family. Pure numpy, no new runtime dependencies.
- `audio.modulation.mps`: spectrotemporal modulation power spectrum on a
  log-frequency cochleagram, with a signed spectral-modulation axis in cycles per
  octave and a temporal axis in Hz. `modulation_power_spectrum()` is exposed as a
  numeric kernel so it can be validated against synthetic ripples of known
  `(Omega, omega)`.
- Cookbook: "Build a music feature hierarchy", covering the frame-rate versus
  window sampling conventions and a ladder of integration windows.

### Fixed
- `audio.clap` and `audio.ast` under transformers 5.x. `get_audio_features` now
  returns a `ModelOutput` rather than a tensor, so `_numpy` unwraps
  `audio_embeds`/`pooler_output`/`last_hidden_state` before casting, and the
  processor keyword renamed from `audios` to `audio`.
- A sample-rate mismatch on a whole-clip audio model now raises a message naming
  the required rate and pointing at `audio.resample`, instead of surfacing as
  "audio projection failed". CLAP requires 48 kHz and AST 16 kHz; nothing is
  resampled implicitly.
- Cookbook and example for extracting `audio.rms` and sampling it onto a scan
  TR grid, including a 2 s TR with stimulus onset at 0.67 s.
- Cookbook and example for convolving native `audio.rms` with fmrimod's SPMG1
  HRF to produce a scan-grid BOLD regressor.
- Explicit temporal contracts: `ClockRef`, affine `ClockMap`, `SupportSpec`,
  `TemporalContext`, and `TimebaseSpec` serialization.
- `FeatureBundle` and dependency-light `TemporalPayload` handoff objects that
  preserve heterogeneous native sampling grids without resampling.
- Clock-aware timeline alignment, cross-clock table safeguards, temporal
  storage round trips, and temporal digests in catalog artifact identity.
- Native-time specification, cookbook, and downstream modeling boundary docs.
- Public API compatibility contract (`natural_features.public_api`) and policy documentation.
- Strict execution policy with normalized method/backend provenance metadata.
- Hardened recipe/schema validation with output contract checks.
- Manifest v2 provenance exports with payload integrity hashes.
- Explicit local bag-of-words provider (`local_bow`) and provider quality upgrades.
- Tier-A golden regression fixture and generator workflow.
- CLI ergonomics: `nf validate`, `nf preset-list`, `nf preset-show`.
- Onboarding and docs index.
- Release discipline docs and release-check script.

### Changed
- Stimuli, feature objects, extraction workflows, and NPZ/Zarr/Parquet storage
  now preserve explicit clock, support, row bounds, and temporal context.
- `RunGrid` and fMRI query compatibility helpers use explicit clock mappings
  internally; `feature_t0_s` remains a compatibility shorthand.
- `in_clock` is the public name for rewriting a feature's times into another
  clock without resampling. `temporal_object_in_clock` remains as an alias.
- API compatibility contract advanced to version 3: named feature methods are
  strict-only. Proxy and surrogate quantities must have their own explicit
  extractor or provider names and cannot be selected as fallback execution.
- Language surprisal now computes summed causal-language-model subword negative
  log probability in nats instead of token-length/character-diversity scores.
- Catalogue execution modes enumerate only `strict`, fallback tags were removed,
  and unknown parameter schema types now fail catalogue validation.
- CLIP, DINO, CLAP, and AST outputs preserve the model's native embedding
  width. The legacy `dim` argument is now an assertion and never truncates or
  zero-pads representations.
- Explicit alignment passthrough via `backend="none"` is a method choice and is
  no longer reported as fallback execution.
- Artifact IDs now include the semantic values of feature, event, and track
  payloads; equal metadata and timing can no longer alias distinct arrays.
- Strict-only CI commands no longer request the removed fallback mode. The
  dependency-free Tier A alignment baseline names `backend="none"` explicitly,
  and its quality gate fails closed on failed cases or missing metrics.

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
- Remove `execution_mode="fallback"` and `strict_dependency=False` from saved
  recipes. Choose an explicitly named proxy feature when one exists; otherwise
  install/configure the named method's backend.
- Consumers of `language.surprisal` must install the `transformers` and `torch`
  dependencies and make the selected causal language model available.
- If you consume exported manifests, prefer `manifest_version=2` fields (`payload_sha256`, `payload_bytes`).
