"""Shared helpers for extractors."""

from __future__ import annotations

from typing import Any

from natural_features.util.hashing import stable_hash


def extractor_metadata(
    extractor_name: str,
    params: dict[str, Any] | None = None,
    *,
    code_version: str = "dev",
    model_revision: str = "none",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = params or {}
    extra = extra or {}
    params_hash = stable_hash(params)
    extractor_id = stable_hash(
        {
            "extractor_name": extractor_name,
            "code_version": code_version,
            "model_revision": model_revision,
            "params": params,
        },
        length=20,
    )
    return {
        "extractor_id": extractor_id,
        "params_hash": params_hash,
        "extractor_name": extractor_name,
        "code_version": code_version,
        "model_revision": model_revision,
        **extra,
    }

