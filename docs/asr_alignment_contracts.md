# ASR + Alignment Contracts (P0)

This document defines the P0 runtime contracts for robust speech ASR/alignment outputs.

## Word Event Metadata Contract

All ASR/alignment word `EventSeries` now include:

- `extractor_id`
- `params_hash`
- `extractor_name`
- `code_version`
- `model_revision`
- `asr_model_name`
- `aligner_backend`
- `aligner_version`

These fields are required for provenance, reproducibility, and backend auditability.

## Phoneme Event Metadata Contract

Canonical phoneme `EventSeries` must include:

- `label_namespace` (e.g. `arpabet`, `ipa`, `ctc`)
- `namespace_version`
- `source_word_alignment_id`

Use `speech.phonology.phoneme_event_series(...)` or
`speech.phonology.phoneme_events_from_words(...)` to construct compliant objects.

## Alignment QC Contract

All alignment QC payloads must provide:

- `mode`
- `fallback_used`
- `n_words`
- `low_confidence_words`
- `dropped_words`

The helper `normalize_alignment_qc(...)` fills defaults and validates required fields.

Optional extended fields supported now:

- `coverage_fraction`
- `boundary_jitter_ms_p50`
- `boundary_jitter_ms_p95`
- `duration_outliers`
- `speaker_overlap_fraction`
- `chunk_count`
- `stitch_conflicts`
- `alignment_details` (backend-specific execution diagnostics)

## Backend Resolution Contract

Alignment backend selection is explicit:

- `backend=auto`: prefer `whisperx`, then `mfa`; strict mode raises if neither
  implemented adapter is available
- `backend=whisperx|mfa`: use the requested backend; strict mode raises if it
  is unavailable
- `backend=gentle`: probe the legacy dependency, but raise in strict mode
  because this release has no Gentle runtime adapter
- `backend=none`: explicitly select passthrough, reported as
  `fallback_used=false`

Passthrough caused by an unavailable requested backend is a fallback and only
runs under explicit `execution_mode="fallback"`.

Use `probe_alignment_backends()` and `resolve_aligner_backend(...)` for diagnostics and
explicit backend provenance.

Reports from backend validation and benchmarking include `runtime_pin_metadata`:

- pinned backend version recommendations
- pinned default model IDs
- installed runtime package snapshot

## Compatibility Notes

- Existing ASR/alignment call signatures remain compatible.
- `whisperx_align(...)` now accepts `backend=` and emits truthful fallback metadata/QC.
- WhisperX runtime path performs real boundary refinement when backend assets are available.
- Use `validate_alignment_backends(...)` (or `nf speech-validate-backends`) to verify local runtime readiness.
- Use `build_alignment_doctor_report(...)` (or `nf speech-doctor`) for actionable remediation guidance.
- Use `run_alignment_benchmark(...)` (or `nf speech-benchmark`) for corpus-level MAE/jitter evaluation.
