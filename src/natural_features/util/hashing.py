"""Stable hashing utilities for IDs and cache keys."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

import numpy as np


def _to_json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _to_json_safe(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, set):
        return sorted(_to_json_safe(v) for v in value)
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": True,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
    if isinstance(value, np.generic):
        return value.item()
    return value


def stable_json_dumps(value: Any) -> str:
    """Serialize with stable key order and no incidental whitespace."""
    return json.dumps(_to_json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(value: Any, *, algorithm: str = "sha256", length: int | None = 16) -> str:
    """Compute a deterministic hash from nested Python structures."""
    payload = stable_json_dumps(value).encode("utf-8")
    if algorithm != "sha256":
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    digest = hashlib.sha256(payload).hexdigest()
    if length is None:
        return digest
    if length <= 0:
        raise ValueError("length must be positive or None")
    return digest[:length]
