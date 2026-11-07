#!/usr/bin/env python3
"""Explicit maintainer operation: refresh the bundled, pinned CMUdict data."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
from urllib.request import urlopen

PIN = "74790861f652b15e4ac49015a90074ad62a27690"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default=PIN)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "src/natural_features/features/speech/data",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("revision must be a full immutable Git SHA")
    base = f"https://raw.githubusercontent.com/cmusphinx/cmudict/{args.revision}"
    fetched = {}
    for name in ("cmudict.dict", "LICENSE"):
        with urlopen(f"{base}/{name}", timeout=60) as response:
            raw = response.read(8_000_001)
        if len(raw) > 8_000_000:
            raise ValueError("resource exceeds expected size")
        fetched[name] = raw
    args.out.mkdir(parents=True, exist_ok=True)
    raw = fetched["cmudict.dict"]
    (args.out / "cmudict.dict.gz").write_bytes(gzip.compress(raw, mtime=0))
    (args.out / "CMUDICT_LICENSE").write_bytes(fetched["LICENSE"])
    manifest = {
        "name": "CMUdict",
        "origin": "dictionary",
        "language": "en-US",
        "revision": args.revision,
        "url": f"{base}/cmudict.dict",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "license_file": "CMUDICT_LICENSE",
        "license_sha256": hashlib.sha256(fetched["LICENSE"]).hexdigest(),
        "uncompressed_bytes": len(raw),
    }
    (args.out / "cmudict.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
