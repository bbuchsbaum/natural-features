"""Pinned runtime recommendations for speech ASR/alignment stacks."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as pkg_version
from typing import Any


PINNED_BACKEND_VERSIONS: dict[str, str] = {
    "whisperx": "3.8.1",
    "faster-whisper": "1.2.1",
    "montreal-forced-aligner": "3.3.9",
    "gentle": "0.1",
}

PINNED_MODEL_IDS: dict[str, str] = {
    "asr_default": "small",
    "speech_ssl_default": "microsoft/wavlm-base-plus",
    "language_embed_default": "bert-base-uncased",
}


def runtime_version_snapshot() -> dict[str, str | None]:
    """Collect installed versions for known speech dependencies."""

    out: dict[str, str | None] = {}
    for pkg in PINNED_BACKEND_VERSIONS:
        try:
            out[pkg] = pkg_version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
        except Exception:
            out[pkg] = None
    return out


def runtime_pin_metadata() -> dict[str, Any]:
    """Attach stable pin metadata to reports for reproducibility."""

    return {
        "pinned_backend_versions": dict(PINNED_BACKEND_VERSIONS),
        "pinned_model_ids": dict(PINNED_MODEL_IDS),
        "runtime_versions": runtime_version_snapshot(),
    }
