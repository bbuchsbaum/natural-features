# Onboarding Guide

## 1) Install

```bash
uv sync
uv run python -c "import natural_features as nf; print(nf.API_COMPAT_VERSION)"
```

Optional extras:

```bash
pip install "natural_features[language,speech,audio,vision]"
```

## 2) Validate setup

```bash
make validate-tier-a
uv run pytest -q tests/unit/test_golden_regressions.py
```

## 3) First useful workflows

### A) Short audio clips -> matrices/dataframes

```python
from natural_features.workflows.audio_batch import extract_audio_dir

res = extract_audio_dir(
    "data/audio_clips",
    resolution_s=1.0,
    selected_features=["rms", "mfcc", "spectral_stats", "vad"],
    collapse="mean+sd",
    as_dataframe=True,
)
```

### B) Audio -> acoustic phonetics

```python
from natural_features.workflows import extract_acoustic_phonetics

ap = extract_acoustic_phonetics(
    "tests/stimuli/tier_a/audio_speechlike.wav",
    posterior_backend="ctc",
    execution_mode="fallback",  # use "strict" to fail loudly
    resolution_s=0.5,
)
```

### C) Multiscale language features

```python
from natural_features.workflows import extract_multiscale_language

ml = extract_multiscale_language(
    "A short sentence. Another sentence.",
    scales_s=[2.0, 4.0, 16.0],
    provider_config={"provider": "local_bow", "dim": 1024},
)
```

## 4) Recipe + CLI flow

```bash
nf prep-video movie.mp4 --video-fps 2 --audio-sr 16000 --video-npy movie.npy --audio-wav movie.wav
nf validate tests/fixtures/recipe_baseline.yaml --have video --have audio
nf extract tests/fixtures/recipe_baseline.yaml --video-npy tests/stimuli/tier_a/video_scene_cut.npy --video-fps 10 --audio-wav tests/stimuli/tier_a/audio_speechlike.wav --out nf_catalog
```

## 5) Read next

- `docs/audio_batch_quickstart.md`
- `docs/acoustic_phonetics_quickstart.md`
- `docs/multiscale_language_quickstart.md`
- `docs/public_api_policy.md`
- `docs/testing_strategy.md`
