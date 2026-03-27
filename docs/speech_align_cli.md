# `nf speech-align` Quick Guide

Single-command ASR + alignment path:

```bash
nf speech-align \
  --audio-wav input.wav \
  --chunked \
  --chunk-window-s 30 \
  --chunk-overlap-s 1 \
  --align-backend auto \
  --execution-mode fallback \
  --ctm-out aligned.ctm \
  --textgrid-out aligned.TextGrid \
  --out-json aligned_summary.json \
  --json
```

## Key Options

- `--execution-mode fallback|strict`
  - `fallback`: graceful degradations with explicit `fallback_used`.
  - `strict`: dependency/runtime failures are raised.
- `--chunked`: enables long-audio chunk/stitch path.
- `--align-backend auto|whisperx|mfa|gentle|none`
  - `auto` probes known backends and picks the first available.
  - `none` forces passthrough alignment.
- MFA runtime options (for `--align-backend mfa`):
  - `--mfa-dictionary /path/to/dictionary.dict`
  - `--mfa-acoustic-model /path/to/acoustic_model.zip`
  - `--mfa-timeout-s 300`
  - `--mfa-tmp-dir /tmp/nf_mfa`
  - `--mfa-extra-arg <arg>` (repeatable)

## Outputs

- `CTM` (`--ctm-out`)
- `TextGrid` (`--textgrid-out`)
- JSON summary (`--out-json`) containing:
  - `asr_qc`
  - `align_qc`
  - `word_metadata`
  - output file paths

## Troubleshooting

- If `align_qc.fallback_used=true`, inspect:
  - `align_qc.reason`
  - `align_qc.backend_resolution`
- For deterministic test runs, prefer transcript-provided ASR path.

## Backend Validation

Validate installed alignment stacks (whisperx/MFA/gentle):

```bash
nf speech-validate-backends --json
```

Run actionable diagnostics with remediation guidance:

```bash
nf speech-doctor --json
```

Optional runtime check against real audio (and transcript-derived words):

```bash
nf speech-validate-backends \
  --audio-wav input.wav \
  --transcript "this is a runtime backend check" \
  --execution-mode fallback \
  --out-json backend_validation.json \
  --json
```

For MFA-specific setup and `_kalpy` errors, see `docs/mfa_runtime_setup.md`.

## Corpus Benchmarking

Run alignment quality benchmarking against a reference manifest:

```bash
nf speech-benchmark \
  --manifest benchmarks/alignment_manifest.json \
  --backend auto \
  --asr-model small \
  --execution-mode fallback \
  --out-json benchmark_report.json \
  --json
```

Manifest items support:

- `audio_path`
- `reference_ctm` or `reference_textgrid`
- optional `transcript` or `transcript_path`
- optional per-item overrides (`backend`, `language`, `asr_model`)
