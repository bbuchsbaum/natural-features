# Scientific Semantic Contract Audit

Date: 2026-07-30

## Governing rule

A public feature ID names one scientific quantity. If the implementation
required for that quantity is unavailable or fails, extraction must fail with
an actionable error. A proxy, heuristic, baseline, or cheaper representation
is valid only under its own explicit ID.

This rule is stronger than deterministic execution and truthful provenance.
Recording that a substitution occurred does not make the substituted values an
implementation of the requested method.

## Corrected in API version 3

- Removed same-name substitute implementations for CLIP, DINO, BERT, causal
  language-model hidden states, WavLM, HuBERT, CLAP, AST, face detection,
  diarization, speech emotion, neural VAD, CTC posteriorgrams, openSMILE
  eGeMAPS, OCR, semantic scene labels, motion-energy filters, and optical flow.
- Replaced the token-length and character-diversity score published as
  `language.surprisal` with causal-language-model negative log probability in
  nats, summed over subwords in full preceding context.
- Renamed the spatial-temporal gradient statistic to
  `vision.motion.gradient_change`; it no longer appears in the catalogue as
  optical flow.
- Made execution strict-only. `execution_mode="fallback"` and
  `strict_dependency=False` now raise migration errors.
- Removed fallback tags and controls from catalogue entries and made unknown
  catalogue parameter types fail closed.
- Preserved native CLIP, DINO, CLAP, and AST dimensions. The compatibility
  `dim` parameter now asserts the model width; it never truncates or zero-pads
  representations.
- Added semantic invariant tests and analytical oracles for language-model
  negative log probability and subword-to-word aggregation.
- Made BERT and causal-LM word representations contextual: each lexical
  sequence is encoded once, and fast-tokenizer offsets map subwords back to
  words. Fake-model analytical oracles verify joint evaluation and pooling.
- Replaced the shared CLAP/AST `AutoModel` path with model-specific adapters.
  Both APIs now expose their native whole-clip representation and retain the
  source clip interval instead of inventing frame times from a stride.
  The parity manifest records the companion R catalogue's `stride_s` as an
  R-only legacy default instead of reintroducing it into the Python API.
- Defined `speech.emotion` as a whole-clip classifier, removed its unused
  `hop_s` control, and retained the classified clip interval in the result.
- Parsed catalogue modalities, dependency and cost classes, parameter schemas,
  and output schemas into validated enums and immutable schema dataclasses at
  registration time.
- Added dependency, model-loading, and inference error classes, then split the
  named optional adapters at those phase boundaries: contextual embeddings and
  surprisal, OpenAI embeddings, spaCy, CLAP, AST, openSMILE, OCR,
  faster-whisper, WavLM, HuBERT, phoneme CTC, Silero VAD, pyannote,
  MediaPipe, OpenCV, pymoten, CLIP/DINO, semantic views, and speech emotion.
  Diagnostic probes and benchmark collectors still catch broadly by design
  because their contract is to report failures rather than execute a feature.

Explicitly named lightweight methods remain valid. Examples include
`vision.energy`, `vision.social_proxies`, `speech.vad`,
`speech.phonology.acoustic_posteriors`, and the `local_bow` and `local_hash`
language providers. Their names and documentation identify what they compute.

## Remaining verification boundary

The in-repository tests establish adapter routing, tensor-shape, temporal,
alignment, schema, and failure-phase contracts with deterministic fake
backends. Numerical parity with upstream model releases remains an external,
model-specific gate; it requires the environment-gated real-backend tests and
must not be inferred from the fake-backend suite.

## Verification scope

The local unit and integration suite, Ruff, public R catalogue contract check,
release static checks, and whitespace checks pass after this audit. Optional
real-model tests remain environment-gated; passing fake-backend contract tests
does not certify numerical parity with upstream CLIP, DINO, WavLM, HuBERT,
CLAP, AST, pyannote, or other model implementations.
