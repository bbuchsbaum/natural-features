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

### Migration Notes
- If you depend on missing OpenAI credentials causing hard errors, set `execution_mode="strict"`.
- If you consume exported manifests, prefer `manifest_version=2` fields (`payload_sha256`, `payload_bytes`).
