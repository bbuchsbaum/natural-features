# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
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
- Multiscale provider fallback now prefers `local_bow` over random local hash fallback.
- API compatibility contract advanced to version 2: named feature methods and
  stable workflows now fail fast when an optional backend is unavailable.
- Deterministic proxies require an explicit `execution_mode="fallback"` (or
  legacy `strict_dependency=False`) request.
- Explicit alignment passthrough via `backend="none"` is a method choice and is
  no longer reported as fallback execution.

### Migration Notes
- Code that deliberately uses deterministic proxies must now pass
  `execution_mode="fallback"`. Omit the setting to require the named method.
- Legacy `strict_dependency=False` remains supported as an explicit fallback
  request; migrate saved recipes to `execution_mode="fallback"` when practical.
- If you consume exported manifests, prefer `manifest_version=2` fields (`payload_sha256`, `payload_bytes`).
