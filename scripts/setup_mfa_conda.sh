#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-mfa}"

echo "[mfa-setup] creating conda env: ${ENV_NAME}"
conda create -y -n "${ENV_NAME}" python=3.11 montreal-forced-aligner kalpy openfst pynini -c conda-forge

echo "[mfa-setup] verifying MFA runtime"
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"
mfa version

echo "[mfa-setup] done"
