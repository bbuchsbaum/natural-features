#!/usr/bin/env python3
"""Fetch six checksum-pinned LibriSpeech utterances, without storing the archive.

FFmpeg is needed only for this explicit preparation step, never for extraction
from the resulting PCM WAVs. The package runtime remains NumPy + stdlib.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
from urllib.request import urlopen
import wave


class LimitedReader:
    def __init__(self, response):
        self.response = response
        self.total = 0

    def read(self, size=-1):
        if size < 0:
            raise ValueError("unbounded archive reads are forbidden")
        data = self.response.read(size)
        self.total += len(data)
        if self.total > 8_000_000:
            raise ValueError(
                "sample exceeds 8 MB download limit; archive layout may have changed"
            )
        return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts/phonology-demo"))
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output directory already exists; use a new directory")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        parser.error(
            "FFmpeg is required for this optional FLAC-to-WAV preparation step"
        )
    source = Path(__file__).resolve().parents[1] / "examples/phonology_demo.json"
    manifest = json.loads(source.read_text())
    hashes = manifest["resource_sha256"]
    blobs = {}
    with urlopen(manifest["source_url"], timeout=45) as response:
        reader = LimitedReader(response)
        with tarfile.open(fileobj=reader, mode="r|gz") as archive:
            scanned = 0
            for member in archive:
                scanned += member.size
                if scanned > 20_000_000:
                    raise ValueError("sample exceeds 20 MB expanded scan limit")
                if member.name in hashes:
                    if not member.isfile() or member.size > 2_000_000:
                        raise ValueError("unexpected sample member type or size")
                    raw = archive.extractfile(member).read()
                    if hashlib.sha256(raw).hexdigest() != hashes[member.name]:
                        raise ValueError(f"source checksum mismatch: {member.name}")
                    blobs[member.name] = raw
                if len(blobs) == len(hashes):
                    break
    if set(blobs) != set(hashes):
        raise ValueError("missing pinned sample resources")
    transcripts = {}
    for name, raw in blobs.items():
        if name.endswith(".trans.txt"):
            transcripts.update(line.split(" ", 1) for line in raw.decode().splitlines())
    args.out.mkdir(parents=True)
    for item in manifest["items"]:
        if transcripts[item["item_id"]] != item["text"]:
            raise ValueError("pinned transcript does not match corpus transcription")
        target = args.out / item["audio"]
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-n",
                "-i",
                "pipe:0",
                "-c:a",
                "pcm_s16le",
                str(target),
            ],
            input=blobs[item["archive_member"]],
            check=True,
        )
        with wave.open(str(target)) as wav:
            if (
                wav.getsampwidth() != 2
                or wav.getnchannels() != 1
                or wav.getframerate() != item["sample_rate"]
            ):
                raise ValueError("decoded audio format mismatch")
            pcm = wav.readframes(wav.getnframes())
        if hashlib.sha256(pcm).hexdigest() != item["pcm_s16le_sha256"]:
            raise ValueError("decoded audio checksum mismatch")
    manifest["resource_manifest_sha256"] = hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out / "LICENSE.txt").write_bytes(blobs["LibriSpeech/LICENSE.TXT"])
    print(
        json.dumps(
            {
                "manifest": str(args.out / "manifest.json"),
                "utterances": len(manifest["items"]),
                "downloaded_bytes": reader.total,
            }
        )
    )


if __name__ == "__main__":
    main()
