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
    execution_mode="fallback", # default; use "strict" to fail loudly
    hop_s=0.02,        # used by acoustic fallback backend
    resolution_s=1.0,  # optional: aggregate to 1s bins
)

post = res.posteriorgrams  # FeatureSeries(time x phone_like_class)
art = res.articulatory     # FeatureSeries(time x articulatory_feature)
```

Common encodings:

- Probabilities (default): `art.values[t, f]` in `[0, 1]` style occupancy/probability.
- Hard one-hot: `argmax(post.values[t])` then map to articulatory bundle.
- Uncertainty-aware: keep `posterior_entropy` and `posterior_peak` columns.

Current broad posterior classes are coarse by design and optimized for robust,
lightweight extraction when no strict phone recognizer is present.

If you want hard failure when strict CTC extraction is unavailable, set:

```python
extract_acoustic_phonetics(
    "data/audio/speech.wav",
    posterior_backend="ctc",
    execution_mode="strict",
)
```
