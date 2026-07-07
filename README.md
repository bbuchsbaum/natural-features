# natural_features

`natural_features` provides core, time-aligned feature data structures and storage primitives for naturalistic stimuli workflows.

This initial scaffold includes:

- Canonical feature object types (`FeatureSeries`, `EventSeries`, `TrackSeries`)
- Deterministic timebase utilities
- Stimulus wrappers with chunked streams
- Minimal storage/catalog support (`zarr`, `parquet`, `npz`)
- Deterministic hashing for IDs and cache fingerprints

## Documentation

- `docs/onboarding.md`
- `docs/audio_batch_quickstart.md`
- `docs/acoustic_phonetics_quickstart.md`
- `docs/multiscale_language_quickstart.md`
- `docs/public_api_policy.md`
- `docs/testing_strategy.md`
- `docs/alignment_benchmarking.md`
- `docs/mfa_runtime_setup.md`
- `docs/fmri_querying.md`
- `docs/api_fmri_query_reference.md`

## Public API Stability

The stable public contract is exported from `natural_features.public_api` and
re-exported at package root (`import natural_features as nf`).

- Stable symbols: `nf.STABLE_EXPORTS`
- Compatibility version: `nf.API_COMPAT_VERSION`
- Experimental namespaces: `nf.EXPERIMENTAL_NAMESPACES`

Policy details: `docs/public_api_policy.md`.

## Execution Modes

Fallback-capable extractors/workflows support:

- `execution_mode="fallback"` (default): use deterministic proxies/fallbacks and annotate provenance.
- `execution_mode="strict"`: fail loudly if optional dependencies/models are unavailable.

Legacy `strict_dependency=True|False` remains supported for backwards compatibility.

## fmrimod Compatibility

The `natural_features.fmri` module is designed to stay compatible with
`fmrimod` for complex HRF/design logic:

- `natural_features.fmri.to_sampling_frame(...)`
- `natural_features.fmri.render_events_with_fmrimod(...)`
- `natural_features.fmri.hrf_kernel(..., backend=\"fmrimod\")`

If `fmrimod` is installed, these adapters let you dogfood existing modeling
interfaces while keeping `natural_features` feature extraction contracts stable.

## CLI

```bash
nf list
nf describe vision.lowlevel.visual_energy
nf validate tests/fixtures/recipe_baseline.yaml --have video --have audio
nf preset-list
nf preset-show fmri_speech_language
nf prep-video movie.mp4 --video-fps 2 --audio-sr 16000 --video-npy movie.npy --audio-wav movie.wav
nf extract tests/fixtures/recipe_baseline.yaml --video-npy clip.npy --video-fps 10 --audio-wav clip.wav --out nf_catalog
nf speech-validate-backends --json
nf speech-doctor --json
nf speech-benchmark --manifest tests/benchmarks/manifests/tier_a_alignment_manifest.json --json
```

## Tier A Test Stimuli

Deterministic, synthetic fixtures live in `tests/stimuli/tier_a/`.

```bash
python scripts/generate_tier_a_stimuli.py
python scripts/validate_tier_a_stimuli.py
```

For broader real-world diagnostics, see Tier B acquisition in `docs/testing_strategy.md`.

Quick test commands:

```bash
make test-smoke
make test-media
make test-nightly
make release-check
make parity-check
```

Optional benchmark gate:

```bash
nf speech-benchmark --manifest tests/benchmarks/manifests/tier_a_alignment_manifest.json --json > /tmp/alignment_report.json
NF_ALIGNMENT_BENCHMARK_REPORT=/tmp/alignment_report.json make release-check
```

Local external dataset tests (SNL sentence stimuli):

```bash
make import-snl       # imports into data/snl_2023_task
make test-external    # runs external-marked tests with dataset enabled
```

## Audio Batch Workflow

For short audio clips (2-6s), there is now a one-call workflow:

```python
from natural_features.workflows.audio_batch import extract_audio_dir

result = extract_audio_dir(
    "data/audio_clips",
    resolution_s=1.0,
    selected_features=["rms", "mfcc", "spectral_stats", "vad"],
    collapse="mean+sd",
    as_dataframe=True,
)
```

See `docs/audio_batch_quickstart.md` for full usage.

## Acoustic Phonetics (Audio-Only)

Option-1 pipeline (`posteriors -> articulatory probabilities`) is available as:

```python
from natural_features.workflows import extract_acoustic_phonetics

res = extract_acoustic_phonetics(
    "data/audio_clips/clip01.wav",
    posterior_backend="ctc",  # "ctc" (preferred) or "acoustic"
    ctc_model="bobboyms/wav2vec2-base-en-phoneme-ctc-41h",
    ctc_local_files_only=True,
    ctc_strict_dependency=False,  # fallback to acoustic backend if unavailable
    resolution_s=0.5,             # optional: 0.5, 1.0, 2.0, ...
)

P = res.posteriorgrams.values  # time x phone-like classes
A = res.articulatory.values    # time x articulatory features (bilabial, alveolar, voiced, ...)
```

For strict CTC-only behavior (no fallback), set `ctc_strict_dependency=True`.

## Multiscale Language Features

Use `extract_multiscale_language` to build language features at multiple scales
(e.g., `2s`, `4s`, `16s`) from transcript text, word events, or audio.

```python
from natural_features.workflows import extract_multiscale_language

res = extract_multiscale_language(
    "data/speech.wav",
    scales_s=[2.0, 4.0, 16.0],
    provider_config={"provider": "local_bow", "dim": 1024},  # or provider="openai"
)
```

See `docs/multiscale_language_quickstart.md` for full usage.

## End-to-End Movie Example

Use `examples/end_to_end_movie_pipeline.py` to go from prepared movie/audio sidecars
to a TR-sampled design matrix (`.npz`) plus metadata:

```bash
python examples/end_to_end_movie_pipeline.py \
  --video-npy movie.npy \
  --video-fps 10 \
  --audio-wav movie.wav \
  --tr-s 1.5 \
  --feature-t0-s 22.3 \
  --out-prefix out/design_run1
```
