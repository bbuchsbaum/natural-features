#!/usr/bin/env python3
"""Prepare short diagnostic clips from Tier B raw media using ffmpeg."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "stimuli" / "tier_b" / "manifest.json"
OUT_DIR = ROOT / "tests" / "stimuli" / "tier_b" / "prepared"


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")


def prepare_video(input_path: Path, output_path: Path, start_s: float, duration_s: float, fps: int = 10) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_s),
        "-i",
        str(input_path),
        "-t",
        str(duration_s),
        "-vf",
        f"fps={fps},scale=224:224",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def prepare_audio(input_path: Path, output_path: Path, start_s: float, duration_s: float, sr: int = 16000) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_s),
        "-i",
        str(input_path),
        "-t",
        str(duration_s),
        "-ac",
        "1",
        "-ar",
        str(sr),
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration-s", type=float, default=8.0)
    ap.add_argument("--start-s", type=float, default=0.0)
    args = ap.parse_args()

    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    existing = []
    for e in entries:
        src = ROOT / e["path"]
        if src.exists():
            existing.append(e)
    if not existing:
        print("Prepared 0 Tier B clips (no raw files found)")
        return 0

    ensure_ffmpeg()
    prepared = 0
    for e in existing:
        rel = e["path"]
        src = ROOT / rel
        tags = set(e.get("diagnostic_tags", []))
        stem = e["id"]
        if "video" in tags:
            out = OUT_DIR / f"{stem}.mp4"
            prepare_video(src, out, args.start_s, args.duration_s)
            prepared += 1
        if "audio" in tags or "speech" in tags:
            out = OUT_DIR / f"{stem}.wav"
            prepare_audio(src, out, args.start_s, args.duration_s)
            prepared += 1
    print(f"Prepared {prepared} Tier B clips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
