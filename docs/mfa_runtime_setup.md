# MFA Runtime Setup

This guide addresses common Montreal Forced Aligner (MFA) runtime issues in local environments.

## Symptom: `_kalpy` Import Error

If backend validation reports:

- `ModuleNotFoundError: No module named '_kalpy'`

then MFA is installed but its compiled Kaldi bindings are missing for your environment.

## Recommended Setup (Conda)

Use the conda-forge stack for reliable compiled dependencies:

```bash
./scripts/setup_mfa_conda.sh mfa
# or manually:
conda create -n mfa python=3.11 montreal-forced-aligner kalpy openfst pynini -c conda-forge
conda activate mfa
mfa version
```

Make shortcut:

```bash
make setup-mfa-conda
```

## Venv Setup (Best Effort)

If you use this repo venv:

```bash
uv pip install --python .venv/bin/python montreal-forced-aligner
# optional extras path:
# pip install "natural-features[alignment_mfa]"
export PATH="$(pwd)/.venv/bin:$PATH"
mfa version
```

If `_kalpy` is still missing, switch to the conda setup above.

## Verify in `natural_features`

Run:

```bash
nf speech-validate-backends --json
nf speech-doctor --json
```

Expected:

- `backends.mfa.available = true`
- `backends.mfa.runtime_ok = true`

If `runtime_ok` is false, inspect `runtime_details.stderr` in the report.
