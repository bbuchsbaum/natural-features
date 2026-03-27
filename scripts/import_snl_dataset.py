#!/usr/bin/env python3
"""Import SNL 2023 sentence stimuli into repository-local data/ folder."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import wave
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = Path("/Users/bbuchsbaum/Dropbox/analysis/SNL_2023_task")
DEFAULT_DEST = REPO_ROOT / "data" / "snl_2023_task"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return float(w.getnframes() / w.getframerate())


def read_stimuli(src_root: Path) -> list[str]:
    rows: list[str] = []
    p = src_root / "sentence_stimuli.csv"
    with p.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if "stimulus" not in (r.fieldnames or []):
            raise ValueError(f"{p} missing 'stimulus' column")
        for row in r:
            stim = str(row["stimulus"]).strip()
            if stim:
                rows.append(stim)
    return rows


def read_sentence_mapping(src_root: Path) -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}
    for p in sorted(src_root.glob("memory_run_*_version_*.csv")):
        with p.open(newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            required = {"stimulus", "sentence", "length"}
            if not required.issubset(set(r.fieldnames or [])):
                continue
            for row in r:
                stim = str(row.get("stimulus", "")).strip()
                sentence = str(row.get("sentence", "")).strip()
                length = str(row.get("length", "")).strip()
                if not stim or not sentence:
                    continue
                prev = mapping.get(stim)
                if prev is None:
                    mapping[stim] = {"sentence": sentence, "length": length}
                else:
                    if prev["sentence"] != sentence or prev["length"] != length:
                        raise ValueError(f"Conflicting metadata for stimulus: {stim}")
    return mapping


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", type=Path, default=DEFAULT_SRC, help="Source SNL task root")
    ap.add_argument("--dest-root", type=Path, default=DEFAULT_DEST, help="Destination data folder")
    ap.add_argument(
        "--copy-audio",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy WAV files into destination",
    )
    args = ap.parse_args()

    src_root = args.src_root
    dest_root = args.dest_root
    if not src_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {src_root}")

    stimuli = read_stimuli(src_root)
    mapping = read_sentence_mapping(src_root)
    missing = [s for s in stimuli if s not in mapping]
    if missing:
        raise ValueError(f"Missing sentence metadata for {len(missing)} stimuli; sample={missing[:5]}")

    dest_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_root / "sentence_stimuli.csv", dest_root / "sentence_stimuli.csv")

    out_rows = []
    total_bytes = 0
    for stim in stimuli:
        src_wav = src_root / stim
        if not src_wav.exists():
            raise FileNotFoundError(f"Missing source WAV: {src_wav}")
        dest_wav = dest_root / stim
        if args.copy_audio:
            dest_wav.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_wav, dest_wav)
        target = dest_wav if args.copy_audio else src_wav
        size = int(target.stat().st_size)
        total_bytes += size
        out_rows.append(
            {
                "stimulus": stim,
                "sentence": mapping[stim]["sentence"],
                "length_tag": mapping[stim]["length"],
                "duration_s": f"{wav_duration_seconds(target):.6f}",
                "size_bytes": str(size),
                "sha256": sha256_file(target),
            }
        )

    meta_csv = dest_root / "sentence_metadata.csv"
    with meta_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["stimulus", "sentence", "length_tag", "duration_s", "size_bytes", "sha256"],
        )
        w.writeheader()
        for row in out_rows:
            w.writerow(row)

    manifest = {
        "name": "snl_2023_task_sentence_stimuli",
        "version": 1,
        "source_root": str(src_root),
        "n_stimuli": len(stimuli),
        "total_audio_bytes": total_bytes,
        "copied_audio": bool(args.copy_audio),
        "files": {
            "sentence_stimuli_csv": "sentence_stimuli.csv",
            "sentence_metadata_csv": "sentence_metadata.csv",
            "audio_root": "sentences_wav/",
        },
    }
    (dest_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Imported {len(stimuli)} stimuli into {dest_root}")
    print(f"Total audio size: {total_bytes / (1024 * 1024):.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
