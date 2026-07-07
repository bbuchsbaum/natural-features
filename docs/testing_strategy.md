# Testing Strategy

## Tiers

- Tier A (`tests/stimuli/tier_a/`): tiny deterministic synthetic fixtures committed in-repo.
- Tier B (`tests/stimuli/tier_b/`): medium real-world fixtures fetched on demand from upstream sources with checksum verification.
- Tier C: large benchmarking corpora, local-only and not required for CI.

## Pytest markers

- `smoke`: fast deterministic tests for every run.
- `media`: media fixture tests (Tier A/B).
- `nightly`: heavier tests intended for scheduled workflows.
- `external`: tests depending on optional external dependencies/tools.

## Make targets

```bash
make test-smoke
make test-media
make test-nightly
make test-external
```

Tier utility targets:

```bash
make generate-tier-a
make validate-tier-a
make fetch-tier-b
make prepare-tier-b
make generate-golden
```

## Tier A commands

```bash
python scripts/generate_tier_a_stimuli.py
python scripts/validate_tier_a_stimuli.py
pytest -m "smoke or media"
```

## Tier B commands

```bash
python scripts/fetch_tier_b_stimuli.py --allow-missing-sha
python scripts/prepare_tier_b_clips.py
```

## External dataset contracts (local-only)

For local validation against the SNL sentence dataset:

- default root: `data/snl_2023_task` (repo-local)
- override root with `NF_SNL_DATA_ROOT`
- enable tests with `NF_ENABLE_EXTERNAL_DATA=1`

Run:

```bash
NF_ENABLE_EXTERNAL_DATA=1 pytest -q -m external
```

Backend runtime validation (when optional stacks are installed):

```bash
nf speech-validate-backends --json
nf speech-doctor --json
```

Real optional backend contract tests are opt-in and never download models during pytest. Set one or more local model variables, then run the external backend test module:

```bash
NF_TEST_AST_MODEL=/path-or-local-id/to/ast \
NF_TEST_CLAP_MODEL=/path-or-local-id/to/clap \
NF_TEST_HUBERT_MODEL=/path-or-local-id/to/hubert \
NF_TEST_WAVLM_MODEL=/path-or-local-id/to/wavlm \
NF_TEST_SPEECH_EMOTION_MODEL=/path-or-local-id/to/emotion \
NF_TEST_NEURAL_VAD_MODEL=silero_vad \
NF_TEST_CLIP_MODEL=/path-or-local-id/to/clip \
NF_TEST_DINO_MODEL=/path-or-local-id/to/dino \
NF_TEST_SPACY_MODEL=en_core_web_sm \
NF_TEST_ENABLE_TESSERACT_OCR=1 \
pytest -q tests/external/test_real_optional_backends.py
```

Corpus benchmark (manifest-driven):

```bash
nf speech-benchmark --manifest tests/benchmarks/manifests/tier_a_alignment_manifest.json --json
```

To import from Dropbox source into repo-local data:

```bash
make import-snl
```

These tests verify:

- `sentence_stimuli.csv` contract consistency,
- WAV existence and duration tags (`_8`, `_12`, `_16`),
- sentence mapping consistency from `memory_run_*_version_*.csv` (authoritative text source),
- end-to-end feature extraction sanity on real audio.

## Golden regressions

Tier A golden references are tracked in:

- `tests/fixtures/golden_reference_v1.json`

Regenerate after intentional numerical/contract changes:

```bash
make generate-golden
```

Then run:

```bash
pytest -q tests/unit/test_golden_regressions.py
```

Recommended policy:

- Keep `enabled: false` in Tier B manifest by default.
- Fill SHA256 in the manifest before enabling entries in CI.
- Keep upstream URLs and license notes in manifest for auditability.

## CI workflow

`.github/workflows/tests.yml` runs:

- `smoke` on push/PR/schedule/manual dispatch.
- `media` on schedule/manual dispatch.
- `nightly` on schedule.
