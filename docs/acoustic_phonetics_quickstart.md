# Acoustic Phonetics Quickstart (Option 1)

Starting from audio only, `natural_features` can produce time-aligned articulatory
feature vectors (`bilabial`, `alveolar`, `voiced`, and related dimensions) using:

1. Phone-like posteriorgrams over short hops.
2. A phone-to-articulatory mapping matrix.

This yields probabilistic articulatory vectors over time.

```python
from natural_features.workflows import extract_acoustic_phonetics

res = extract_acoustic_phonetics(
    "data/audio/speech.wav",
    posterior_backend="ctc",   # preferred strict backend
    ctc_model="bobboyms/wav2vec2-base-en-phoneme-ctc-41h",
    ctc_local_files_only=True, # reproducible/offline
    hop_s=0.02,        # used by the explicit acoustic backend
    resolution_s=1.0,  # optional: aggregate to 1s bins
)

post = res.posteriorgrams  # FeatureSeries(time x phone_like_class)
art = res.articulatory     # FeatureSeries(time x articulatory_feature)
```

CTC inference selects CUDA, then MPS, then CPU (`ctc_device="auto"`) and reuses a
process-level model cache across calls. Stimuli longer than 30 s are split with
1 s overlap; interior overlap frames are dropped before the posteriorgram is
stitched.

Common encodings:

- Probabilities (default): `art.values[t, f]` in `[0, 1]` style occupancy/probability.
- Hard one-hot: `argmax(post.values[t])` then map to articulatory bundle.
- Uncertainty-aware: keep `posterior_entropy` and `posterior_peak` columns.

Current broad posterior classes are coarse by design and optimized for robust,
lightweight extraction when no strict phone recognizer is present.

Strict CTC extraction fails by default when the model is unavailable. To make
that policy explicit in a saved configuration, set:

```python
extract_acoustic_phonetics(
    "data/audio/speech.wav",
    posterior_backend="ctc",
    execution_mode="strict",
)
```

`audio_formants` keeps its requested, sample-rounded formant grid when voicing
gating is enabled. The 40 ms (or longer) pitch windows have different centers
from the default 25 ms formant windows: voicing strength is linearly interpolated
at each formant center before applying the `> 0.3` gate. Centers outside the
pitch-center range are zero-gated, including short clips with no overlapping
centers; edge rows are retained. This periodicity gate does not identify vocals
in polyphonic audio.

Formant metadata records effective framing, the gate policy, and
`code_version="formants-v2"` to distinguish the corrected output's extractor identity.
LPC output includes sample-window bounds and short-input padding. The optional
Praat backend queries its local waveform clock; `start_offset_s` affects only the
output timestamps. Praat output declares point queries rather than LPC window
bounds: its [Gaussian analysis windows](https://www.fon.hum.uva.nl/praat/manual/Sound__To_Formant__burg____.html)
and interpolated estimates have different support.
