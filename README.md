# natural_features

`natural_features` turns naturalistic stimuli into typed, time-aligned feature
objects for analysis and modeling. It provides a feature catalogue and planner,
high-level extraction workflows, explicit timeline alignment, table export, and
provenance-aware storage.

[Documentation](https://bbuchsbaum.github.io/natural-features/) ·
[Start guide](https://bbuchsbaum.github.io/natural-features/start/) ·
[API reference](https://bbuchsbaum.github.io/natural-features/reference/)

The package is currently at version `0.0.1` and is under active development.
Its versioned public contract is exported from `natural_features.public_api`
and re-exported at package root.

## What it provides

- Typed temporal objects: `FeatureSeries`, `EventSeries`, and `TrackSeries`
- Stimulus wrappers for text, image, audio, video, and multimodal inputs
- A queryable feature catalogue with planning before execution
- One-call extraction through `extract_features(...)`
- Explicit alignment to event, frame, regular, or custom timelines
- Audio-batch, acoustic-phonetics, multiscale-language, and video-text workflows
- Run-aware fMRI window queries and optional `fmrimod` adapters
- Storage and interchange through NPZ, Zarr, Parquet, and tabular exports
- A CLI for catalogue inspection, recipes, media preparation, and speech alignment

## Installation

The project is not yet published on PyPI. Install it from a checkout:

```bash
git clone https://github.com/bbuchsbaum/natural-features.git
cd natural-features
python -m pip install -e .
```

For development with `uv`:

```bash
uv sync
uv run nf --help
```

Optional dependency groups are defined in `pyproject.toml`, including
`storage`, `vision`, `speech`, `alignment`, `alignment_mfa`,
`alignment_legacy`, `llm`, `dev`, and `docs`.

## Quick start

The high-level workflow accepts a stimulus, resolves the requested feature
route, and returns typed feature objects:

```python
import natural_features as nf

result = nf.extract_features(
    "The scene opens on a quiet room.",
    features=["text.tokenize", "language.surface"],
)

words = result.features["text.tokenize"]
surface = result.features["language.surface"]

aligned = result.align_to(
    "text.tokenize",
    features="language.surface",
)
rows = aligned.to_rows()
```

Inspect available features before running a workflow:

```bash
nf features --modality text --json
nf features --modality audio --budget all --csv
nf describe vision.lowlevel.visual_energy
```

See the [feature catalog](https://bbuchsbaum.github.io/natural-features/tour/)
for all public feature IDs, brief interpretations, extraction patterns, output
types, dependency and cost classes, explicit approximation semantics, and
method references.

## Execution modes

Named methods fail fast by default when their optional package, model, API
credential, or system tool is unavailable. They never silently substitute a
different computation.

- `execution_mode="strict"` is the default and raises when the requested
  backend or model is unavailable.
- `execution_mode="fallback"` explicitly requests a deterministic proxy, when
  one exists, and records the substitution in provenance.

Legacy `strict_dependency=True|False` arguments remain supported for backward
compatibility.

## Stable public API

The compatibility policy covers the symbols in `nf.STABLE_EXPORTS`, including
the core temporal types, timeline types, catalogue and extraction functions,
workflow entry points, and fMRI query helpers.

```python
import natural_features as nf

print(nf.API_COMPAT_VERSION)
print(nf.STABLE_EXPORTS)
print(nf.EXPERIMENTAL_NAMESPACES)
```

See the [public API policy](https://github.com/bbuchsbaum/natural-features/blob/main/docs/public_api_policy.md)
for the compatibility boundary. Experimental namespaces are explicitly listed
in `nf.EXPERIMENTAL_NAMESPACES`.

## CLI

The `nf` command exposes catalogue, recipe, media, and speech workflows:

```bash
nf list
nf features --modality video
nf describe vision.lowlevel.visual_energy
nf validate tests/fixtures/recipe_baseline.yaml --have video --have audio
nf preset-list
nf preset-show fmri_speech_language
nf prep-video movie.mp4 --video-fps 2 --audio-sr 16000 --video-npy movie.npy --audio-wav movie.wav
nf extract tests/fixtures/recipe_baseline.yaml --video-npy clip.npy --video-fps 10 --audio-wav clip.wav --out nf_catalog
nf video-text movie.mp4 --video-fps 24 --table-out words.csv --json
nf speech-validate-backends --json
nf speech-doctor --json
nf speech-benchmark --manifest tests/benchmarks/manifests/tier_a_alignment_manifest.json --json
```

Run `nf <command> --help` for command-specific options.

## Development and verification

Deterministic synthetic fixtures live in `tests/stimuli/tier_a/`. Common checks
are:

```bash
make test-smoke
make test-media
make test-nightly
make release-check
make parity-check
```

Build the Quarto documentation locally with:

```bash
make -C docs quarto
```

The documentation site is deployed from `main` by GitHub Actions.
