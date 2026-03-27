# Speech Alignment Backend Policy

This project uses the following alignment policy:

- Default ASR path: `speech.asr.whisper` (faster-whisper backend when available).
- Alignment refinement: `speech.align.whisperx` with backend routing:
  - primary: `whisperx`
  - optional strict backend: `mfa` (requires dictionary + acoustic model paths)
- Legacy backend: `gentle` is supported only as an optional plugin path and is not a default dependency.
- Preferred strict phonetic backend for audio-only posterior features: `speech.phonology.ctc_posteriors` (optional transformers/torch model path, with configurable fallback).

Rationale:

- Keep default setup lightweight and reproducible.
- Prefer actively maintained backends for new workflows.
- Preserve legacy compatibility without coupling core behavior to older tooling.

Expected outputs:

- `segments` and `words` as `EventSeries`.
- Alignment QC summary containing `n_words`, `low_confidence_words`, and `dropped_words`.
