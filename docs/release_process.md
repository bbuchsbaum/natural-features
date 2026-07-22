# Release Process

## Versioning policy

- Package version: semantic versioning (`MAJOR.MINOR.PATCH`).
- Public API contract version: `natural_features.API_COMPAT_VERSION`.
- Catalog manifest version: `manifest_version` in exported manifest JSON.

Breaking stable API changes require:

1. incrementing `API_COMPAT_VERSION`,
2. documenting migration notes,
3. updating `CHANGELOG.md` under `Unreleased`.

## Pre-release checklist

1. Run tests:
   - `uv run pytest -q`
   - `uv run --extra storage pytest -q tests/unit/test_temporal_storage_roundtrip.py`
2. Validate deterministic fixtures:
   - `make validate-tier-a`
   - `uv run pytest -q tests/unit/test_golden_regressions.py`
3. Run release gates:
   - `make release-check`
   - `make parity-check`
   - CI-only parity mode, when `~/code/natfeatures` is unavailable:
     - `uv run python scripts/release_check.py --no-r-compare`
   - Optional hard benchmark gate:
     - `nf speech-benchmark --manifest tests/benchmarks/manifests/tier_a_alignment_manifest.json --json > /tmp/alignment_report.json`
     - `NF_ALIGNMENT_BENCHMARK_REPORT=/tmp/alignment_report.json make release-check`
   - For temporal-contract changes, run the native-grid acceptance fixture:
     - `uv run pytest -q tests/integration/test_native_temporal_handoff.py`
   - When the sibling `fmrimod` checkout is available, run the cross-library
     payload test in its dependency environment.
4. Update docs/changelog:
   - `CHANGELOG.md`
   - `docs/public_api_policy.md` (if relevant)
   - migration notes for breaking changes
5. Build executable documentation:
   - `make -C docs quarto`

## Migration notes template

Use this section in changelog entries when compatibility changes:

- **What changed**
- **Who is impacted**
- **How to migrate** (code snippets/CLI changes)
- **Version boundary** (`from -> to`)

## Manifest compatibility

- `manifest_version=1` and `manifest_version=2` are import-compatible.
- New manifest versions must include explicit upgrade notes in `CHANGELOG.md`.
