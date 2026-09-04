# Speech Alignment Backend Policy

This project uses the following alignment policy:

- Default ASR path: `speech.asr.whisper` (faster-whisper backend when available).
- Alignment refinement: `speech.align.whisperx` with backend routing:
  - primary: `whisperx`
  - optional strict backend: `mfa` (requires dictionary + acoustic model paths)
- Legacy backend: `gentle` is supported only as an optional plugin path and is not a default dependency.
- Preferred lightweight phonetic backend for audio-only posterior features: `speech.phonology.ctc_posteriors` (optional transformers/torch model path).
- Optional high-fidelity phonetic posteriorgrams: `speech.phonology.ppg_posteriors` via the `ppgs` extra (`posterior_backend="ppgs"`). English CMU 40-phone inventory only.

Rationale:

- Keep default setup lightweight and reproducible.
- Prefer actively maintained backends for new workflows.
- Preserve legacy compatibility without coupling core behavior to older tooling.

Expected outputs:

- `segments` and `words` as `EventSeries`.
- When the MFA backend runs, `phones` is the MFA `phones` TextGrid tier as an
  `EventSeries` (`speech.phones.mfa`). `speech.phonemes` remains the
  uniform word-interval split and is not acoustic phone alignment.
- Alignment QC summary containing `n_words`, `low_confidence_words`, and `dropped_words`.

See `docs/speech_ladder.md` for the graded A1–M feature bands.
