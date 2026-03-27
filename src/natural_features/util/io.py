"""Safe file write helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
import uuid
from typing import Any

import numpy as np


def _temp_path(path: Path) -> Path:
    return path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"


def atomic_write_bytes(path: str | Path, payload: bytes) -> Path:
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = _temp_path(dst)
    try:
        with tmp.open("wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dst)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return dst


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
    return atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(
    path: str | Path,
    payload: Any,
    *,
    sort_keys: bool = True,
    indent: int | None = 2,
) -> Path:
    txt = json.dumps(payload, sort_keys=sort_keys, indent=indent)
    return atomic_write_text(path, txt, encoding="utf-8")


def atomic_numpy_save(path: str | Path, array: np.ndarray, *, allow_pickle: bool = False) -> Path:
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = _temp_path(dst)
    try:
        with tmp.open("wb") as f:
            np.save(f, array, allow_pickle=allow_pickle)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dst)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return dst


def atomic_numpy_savez(path: str | Path, **arrays: Any) -> Path:
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = _temp_path(dst)
    try:
        with tmp.open("wb") as f:
            np.savez_compressed(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dst)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return dst
