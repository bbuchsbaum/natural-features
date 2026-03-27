#!/usr/bin/env python3
"""Fetch Tier B stimuli declared in manifest with hash verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request
from urllib.parse import urlparse
import uuid


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "stimuli" / "tier_b" / "manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, out_path: Path) -> None:
    parsed = urlparse(url)
    if parsed.scheme.lower() != "https":
        raise ValueError(f"Only https:// URLs are allowed, got: {url}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, method="GET")
    tmp = out_path.parent / f".{out_path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            with tmp.open("wb") as out:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                out.flush()
                os.fsync(out.fileno())
        os.replace(tmp, out_path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default=None, help="Fetch only one entry id")
    ap.add_argument("--allow-missing-sha", action="store_true", help="Allow entries with empty sha256")
    args = ap.parse_args()

    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    fetched = 0
    for e in entries:
        if not e.get("enabled", False):
            continue
        if args.id and e["id"] != args.id:
            continue
        rel = e["path"]
        url = e["source_url"]
        expected = (e.get("sha256") or "").strip().lower()
        out = ROOT / rel
        print(f"Fetching {e['id']} -> {rel}")
        download(url, out)
        actual = sha256_file(out)
        if expected:
            if actual != expected:
                raise ValueError(
                    f"SHA mismatch for {e['id']}: expected={expected} actual={actual}"
                )
        elif not args.allow_missing_sha:
            raise ValueError(
                f"Entry {e['id']} has empty sha256. Fill manifest checksum or use --allow-missing-sha."
            )
        fetched += 1
    print(f"Fetched {fetched} Tier B entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
