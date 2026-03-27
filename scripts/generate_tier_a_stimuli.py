#!/usr/bin/env python3
"""Generate deterministic Tier A test stimuli and manifest."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import wave

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tests" / "stimuli" / "tier_a"


@dataclass(frozen=True)
class StimulusEntry:
    id: str
    path: str
    kind: str
    duration_s: float
    diagnostics: dict[str, object]
    sha256: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_wav(path: Path, samples: np.ndarray, sr_hz: int) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr_hz)
        w.writeframes(pcm.tobytes())


def _make_video() -> Path:
    rng = np.random.default_rng(1337)
    fps = 10
    n_frames = 60
    h, w = 64, 64
    frames = np.zeros((n_frames, h, w, 3), dtype=np.uint8)

    # Pre-cut: dark scene with low motion.
    for t in range(30):
        base = 30 + (t % 5)
        frames[t, :, :, :] = base
        x0 = 4 + (t % 20)
        frames[t, 20:36, x0 : x0 + 8, 1] = 110
        noise = rng.integers(0, 4, size=(h, w, 3), dtype=np.uint8)
        frames[t] = np.clip(frames[t] + noise, 0, 255)

    # Post-cut: bright scene with higher saturation and motion.
    for t in range(30, 60):
        base = 190 + ((t - 30) % 6)
        frames[t, :, :, 0] = base
        frames[t, :, :, 1] = 90
        frames[t, :, :, 2] = 70
        y0 = 6 + ((t - 30) * 2 % 40)
        x0 = 6 + ((t - 30) * 3 % 40)
        frames[t, y0 : y0 + 14, x0 : x0 + 14, :] = np.array([30, 220, 220], dtype=np.uint8)
        noise = rng.integers(0, 6, size=(h, w, 3), dtype=np.uint8)
        frames[t] = np.clip(frames[t] + noise, 0, 255)

    out = OUT_DIR / "video_scene_cut.npy"
    np.save(out, frames)
    return out


def _make_audio() -> Path:
    sr = 16000
    duration_s = 6.0
    t = np.arange(int(sr * duration_s), dtype=np.float32) / sr
    x = np.zeros_like(t, dtype=np.float32)

    # 0-1 silence
    # 1-2 steady tone
    m = (t >= 1.0) & (t < 2.0)
    x[m] += 0.18 * np.sin(2 * np.pi * 220 * t[m])

    # 2-3 dual-tone
    m = (t >= 2.0) & (t < 3.0)
    x[m] += 0.15 * np.sin(2 * np.pi * 220 * t[m]) + 0.08 * np.sin(2 * np.pi * 660 * t[m])

    # 3-4 noise burst
    rng = np.random.default_rng(7331)
    m = (t >= 3.0) & (t < 4.0)
    x[m] += 0.12 * rng.normal(0.0, 1.0, size=np.sum(m)).astype(np.float32)

    # 4-5 high tone
    m = (t >= 4.0) & (t < 5.0)
    x[m] += 0.2 * np.sin(2 * np.pi * 440 * t[m])

    # 5-6 silence with tiny floor
    m = (t >= 5.0) & (t < 6.0)
    x[m] += 0.003 * np.sin(2 * np.pi * 80 * t[m])

    out = OUT_DIR / "audio_speechlike.wav"
    _write_wav(out, x, sr)
    return out


def _make_transcript() -> Path:
    text = (
        "this synthetic clip has one scene cut and changing audio energy "
        "for deterministic integration testing"
    )
    out = OUT_DIR / "transcript_reference.txt"
    out.write_text(text + "\n", encoding="utf-8")
    return out


def _make_reference_ctm(transcript_path: Path, *, duration_s: float = 6.0) -> Path:
    words = transcript_path.read_text(encoding="utf-8").strip().split()
    step = float(duration_s / max(1, len(words)))
    lines: list[str] = []
    for i, token in enumerate(words):
        onset = i * step
        lines.append(f"utt 1 {onset:.6f} {step:.6f} {token} 1.000000")
    out = OUT_DIR / "reference_words.ctm"
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    video = _make_video()
    audio = _make_audio()
    transcript = _make_transcript()
    reference_ctm = _make_reference_ctm(transcript, duration_s=6.0)

    entries = [
        StimulusEntry(
            id="tier_a_video_scene_cut",
            path=str(video.relative_to(ROOT)),
            kind="video_npy",
            duration_s=6.0,
            diagnostics={
                "fps": 10,
                "expected_scene_cut_s": 3.0,
                "expected_motion_high_window_s": [3.0, 6.0],
            },
            sha256=_sha256(video),
        ),
        StimulusEntry(
            id="tier_a_audio_speechlike",
            path=str(audio.relative_to(ROOT)),
            kind="audio_wav",
            duration_s=6.0,
            diagnostics={
                "sr_hz": 16000,
                "high_energy_window_s": [1.0, 5.0],
                "low_energy_windows_s": [[0.0, 1.0], [5.0, 6.0]],
            },
            sha256=_sha256(audio),
        ),
        StimulusEntry(
            id="tier_a_transcript_reference",
            path=str(transcript.relative_to(ROOT)),
            kind="text",
            duration_s=6.0,
            diagnostics={"word_count": len(transcript.read_text(encoding="utf-8").strip().split())},
            sha256=_sha256(transcript),
        ),
        StimulusEntry(
            id="tier_a_reference_ctm",
            path=str(reference_ctm.relative_to(ROOT)),
            kind="ctm",
            duration_s=6.0,
            diagnostics={"word_count": len(transcript.read_text(encoding="utf-8").strip().split())},
            sha256=_sha256(reference_ctm),
        ),
    ]

    manifest = {
        "manifest_version": 1,
        "name": "tier_a_synthetic_stimuli",
        "license": "synthetic-generated",
        "generator": "scripts/generate_tier_a_stimuli.py",
        "entries": [asdict(e) for e in entries],
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote Tier A stimuli to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
