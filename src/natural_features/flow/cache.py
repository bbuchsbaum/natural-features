"""Deterministic cache fingerprinting and invalidation reasoning."""

from __future__ import annotations

from typing import Any

from natural_features.util.hashing import stable_hash


def cache_fingerprint(
    *,
    extractor_name: str,
    params: dict[str, Any],
    code_version: str,
    model_revision: str,
    upstream_ids: list[str],
) -> str:
    payload = {
        "extractor_name": extractor_name,
        "params": params,
        "code_version": code_version,
        "model_revision": model_revision,
        "upstream_ids": sorted(upstream_ids),
    }
    return stable_hash(payload, length=24)


def invalidation_reasons(previous: dict[str, Any], current: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    fields = [
        ("extractor_name", "extractor changed"),
        ("params", "parameters changed"),
        ("code_version", "code version changed"),
        ("model_revision", "model revision changed"),
        ("upstream_ids", "upstream lineage changed"),
    ]
    for key, label in fields:
        prev = previous.get(key)
        curr = current.get(key)
        if key == "upstream_ids":
            prev = sorted(prev or [])
            curr = sorted(curr or [])
        if prev != curr:
            reasons.append(label)
    if not reasons:
        reasons.append("cache-valid")
    return reasons

