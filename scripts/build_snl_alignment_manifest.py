#!/usr/bin/env python3
"""Build an SNL alignment benchmark manifest with deterministic reference CTMs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from natural_features.core.stimulus import AudioStimulus
from natural_features.features.speech.asr import whisper_transcribe
from natural_features.features.speech.formats import write_ctm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snl-root", type=Path, default=Path("data/snl_2023_task"), help="Root SNL dataset directory")
    ap.add_argument("--limit", type=int, default=12, help="Number of clips to include")
    ap.add_argument("--out-dir", type=Path, default=Path("tests/benchmarks/generated/snl"), help="Output directory")
    ap.add_argument("--language", default="en", help="Language code")
    args = ap.parse_args()

    snl_root = args.snl_root
    meta = snl_root / "sentence_metadata.csv"
    if not meta.exists():
        raise FileNotFoundError(f"Missing metadata: {meta}")

    out_dir = args.out_dir
    ref_dir = out_dir / "refs"
    ref_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    with meta.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            stim = str(row.get("stimulus", "")).strip()
            sent = str(row.get("sentence", "")).strip()
            if stim and sent:
                rows.append({"stimulus": stim, "sentence": sent})

    items = []
    for row in rows[: max(0, int(args.limit))]:
        wav = snl_root / row["stimulus"]
        if not wav.exists():
            continue
        audio = AudioStimulus.from_wav(wav)
        words = whisper_transcribe(audio, transcript_text=row["sentence"], language=args.language)["words"]
        ctm = ref_dir / f"{wav.stem}.ctm"
        txt = ref_dir / f"{wav.stem}.txt"
        write_ctm(words, ctm)
        txt.write_text(row["sentence"] + "\n", encoding="utf-8")
        items.append(
            {
                "id": wav.stem,
                "audio_path": str(wav.resolve()),
                "reference_ctm": str(ctm.resolve()),
                "transcript_path": str(txt.resolve()),
                "language": args.language,
            }
        )

    manifest = {
        "manifest_version": 1,
        "description": "Generated SNL subset manifest for alignment benchmarking",
        "items": items,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(manifest_path)
    print(f"items={len(items)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
