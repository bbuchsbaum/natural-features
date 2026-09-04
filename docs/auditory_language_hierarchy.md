# Auditory-Language Hierarchy Vignette

Starting from **one speech wav**, `natural_features` can populate every level of
a hierarchical auditory-language analysis on one shared time grid:

| Level | Block | What it captures | Extractors |
|---|---|---|---|
| 1 | `acoustic.*` | Envelope, spectrum, cepstrum | `rms`, `mel`, `mfcc`, `spectral_stats`, `gammatone` |
| 2 | `prosody.*` | F0, voicing, energy dynamics | `audio_pitch`, `prosody_features` |
| 3 | `phonology.*` | Phone-class posteriors, articulatory features | `extract_acoustic_phonetics` (CTC, PPGs, or acoustic backend) |
| 4 | `semantic.*` | Sentence/paragraph embeddings, lexical controls, optional surprisal | `extract_multiscale_language` |

This vignette is a **shared-grid design-matrix recipe**. For native A1–M bands
and residualization before encoding, see [`speech_ladder.md`](speech_ladder.md).

Each level yields `FeatureSeries` objects on their native grids; rendering them
onto one `build_tr_grid` grid makes them directly stackable into a single
design matrix with per-level column blocks — the layout needed for banded
ridge, hierarchical regression, or variance partitioning across levels of the
speech processing hierarchy.

A complete runnable version of this vignette lives at
[`examples/auditory_language_hierarchy.py`](../examples/auditory_language_hierarchy.py):

```bash
python examples/auditory_language_hierarchy.py \
    --audio-wav tests/stimuli/tier_a/audio_speechlike.wav \
    --transcript tests/stimuli/tier_a/transcript_reference.txt \
    --resolution-s 1.0 \
    --out-prefix hierarchy_demo
```

## 1. One canonical grid

```python
import numpy as np

from natural_features.core.stimulus import AudioStimulus
from natural_features.fmri.resample import build_tr_grid, resample_feature_series

audio = AudioStimulus.from_wav("story.wav")
duration_s = audio.start_offset_s + audio.samples.shape[0] / audio.sr_hz

tr_s = 1.0  # analysis resolution; use your scan TR for fMRI
grid = build_tr_grid(duration_s=duration_s, tr_s=tr_s, start_s=audio.start_offset_s)

def on_grid(fs, method="mean"):
    return resample_feature_series(fs, tr_s=tr_s, method=method, time_grid_s=grid)
```

Every block below is rendered with `on_grid`, so rows always share timestamps
and window support.

## 2. Low-level acoustics

```python
from natural_features.features.audio.lowlevel import mfcc, rms, spectral_stats

acoustic = [
    on_grid(rms(audio)),
    on_grid(mfcc(audio, n_mfcc=13, n_mels=40, include_deltas=True)),
    on_grid(spectral_stats(audio)),
]
```

For batches of clips, `extract_audio_files` wraps the same extractors (plus
`mel`, `pitch`, `prosody`, `vad`, and openSMILE eGeMAPS) with per-file
dataframes; see the [audio batch quickstart](audio_batch_quickstart.md).

## 3. Prosody

```python
from natural_features.features.audio.prosody import prosody_features

prosody = on_grid(prosody_features(audio))
# columns: rms, log_rms, f0_hz, voicing_strength, spectral_centroid, zcr
```

`audio_pitch` is also available separately when only F0/voicing are needed.

## 4. Phonetics and phonology

```python
from natural_features.workflows import extract_acoustic_phonetics

phon = extract_acoustic_phonetics(
    audio,
    posterior_backend="ctc",      # strict wav2vec2 phoneme-CTC model
    ctc_device="auto",            # CUDA, then MPS, then CPU
    ctc_local_files_only=True,    # reproducible/offline
)
phonology = [
    on_grid(phon.posteriorgrams),  # time x phone class probabilities
    on_grid(phon.articulatory),    # time x articulatory features (bilabial, voiced, ...)
]
```

The CTC model loads once per process and is reused across files; audio longer
than 30 s is chunked with 1 s overlap automatically. Without local model
weights, `posterior_backend="acoustic"` provides a deterministic coarse-class
substitute so the pipeline stays runnable offline (details in the
[acoustic phonetics quickstart](acoustic_phonetics_quickstart.md)).

## 5. Sentence- and paragraph-level semantics

```python
from natural_features.workflows import extract_multiscale_language

language = extract_multiscale_language(
    audio,
    transcript_text=open("story_transcript.txt").read(),  # omit to run strict ASR
    scales_s=[tr_s],
    feature_families=[
        "sentence_embeddings",
        "paragraph_embeddings",
        "lexical_controls",
    ],
    provider_config={"provider": "openai", "model": "text-embedding-3-large"},
    standardize=False,
    add_intercept=False,
)
semantic = language.by_scale[tr_s]
```

Words come from your transcript (uniform timing) or strict Whisper ASR;
sentence and paragraph events are segmented from the word stream and embedded
per unit, then rendered onto the same grid. For offline runs, use
`provider_config={"provider": "local_bow", "dim": 1024}`.

Adding `"surprisal"` to `feature_families` contributes word-level
predictability from a causal language model. Unlike the three families above it
is a strict neural backend: it requires `transformers` and `torch` plus local
weights, and raises rather than substituting a proxy. The example script keeps
it behind `--surprisal lm` so the default run stays offline.
`language.words`, `language.sentences`, and `language.paragraphs` expose the
underlying `EventSeries` if you need event-locked rather than gridded designs.

## 6. Stack into one hierarchical design

```python
from natural_features.fmri.design import concat_feature_series

blocks = {
    "acoustic": acoustic,
    "prosody": [prosody],
    "phonology": phonology,
    "semantic": [semantic],
}
design = concat_feature_series(
    [fs for level in blocks.values() for fs in level],
    standardize=True,
    add_intercept=False,
)
```

`concat_feature_series` refuses to stack blocks whose grids, clocks, or window
supports disagree, so misaligned levels fail loudly instead of silently
shifting time. The runnable example additionally records `block_slices`
(column ranges per level) in its JSON sidecar so downstream banded models can
address each level of the hierarchy by name.

## Notes

- **Determinism/offline:** transcript + `posterior_backend="acoustic"` +
  `provider="local_bow"` runs with no model downloads and is what the
  integration test exercises.
- **Strict-by-default:** the neural backends (`ctc`, Whisper ASR, API
  embedding providers) fail loudly when unavailable rather than silently
  substituting a proxy; substitutes are explicit choices.
- **fMRI:** set `tr_s` to your scan TR and map stimulus time onto scan time
  with a `ClockMap` (see the [RMS-energy-to-TR example](../examples/rms_energy_to_tr_grid.py));
  HRF convolution and lagging belong in your modeling layer.
- **Finer phonetic timing:** for event-locked phoneme analyses, align words
  with `whisperx_align`, expand to phones with `phoneme_events_from_words`, and
  map to articulatory features with `articulatory_from_phoneme_events`.
