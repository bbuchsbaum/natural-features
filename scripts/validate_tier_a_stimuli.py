#!/usr/bin/env python3
"""Validate Tier A stimuli against manifest hashes and basic metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import wave

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "stimuli" / "tier_a" / "manifest.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    for e in entries:
        rel = e["path"]
        p = ROOT / rel
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {rel}")
        actual = _sha256(p)
        if actual != e["sha256"]:
            raise ValueError(f"SHA mismatch for {rel}: expected={e['sha256']} actual={actual}")

        if e["kind"] == "video_npy":
            arr = np.load(p, allow_pickle=False)
            if arr.ndim != 4:
                raise ValueError(f"{rel} expected 4D array, got shape={arr.shape}")
        elif e["kind"] == "audio_wav":
            with wave.open(str(p), "rb") as w:
                if w.getnchannels() != 1:
                    raise ValueError(f"{rel} expected mono WAV")
                if w.getframerate() <= 0:
                    raise ValueError(f"{rel} invalid sample rate")
        elif e["kind"] in {"text", "ctm"}:
            _ = p.read_text(encoding="utf-8")
    print("Tier A stimuli validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
