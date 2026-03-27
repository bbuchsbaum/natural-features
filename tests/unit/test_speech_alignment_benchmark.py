from __future__ import annotations

import json
from pathlib import Path
import wave

import numpy as np

from natural_features.core.feature_types import EventSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.features.common import extractor_metadata
from natural_features.features.speech.benchmark import (
    BenchmarkConfig,
    benchmark_alignment_case,
    match_token_pairs,
    run_alignment_benchmark,
)
from natural_features.features.speech.formats import write_ctm


def _md(name: str) -> dict[str, object]:
    return extractor_metadata(name, params={})


def _words(labels: list[str], onset: list[float], offset: list[float]) -> EventSeries:
    return EventSeries(
        onset_s=np.asarray(onset, dtype=np.float64),
        offset_s=np.asarray(offset, dtype=np.float64),
        label=np.asarray(labels, dtype=object),
        confidence=np.ones(len(labels), dtype=np.float32),
        metadata=_md("test.words"),
    )


def _write_wav(path: Path, *, sr: int = 16000, seconds: float = 1.0) -> None:
    n = int(sr * seconds)
    t = np.arange(n, dtype=np.float32) / float(sr)
    x = (0.2 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def test_match_token_pairs_normalizes_tokens() -> None:
    ref = _words(["Hello,", "world!"], [0.0, 0.5], [0.4, 0.9])
    pred = _words(["hello", "world"], [0.01, 0.52], [0.41, 0.92])
    pairs = match_token_pairs(ref, pred)
    assert pairs == [(0, 0), (1, 1)]


def test_benchmark_alignment_case_metrics(monkeypatch) -> None:
    audio = AudioStimulus.from_array(np.zeros(16000, dtype=np.float32), sr_hz=16000)
    ref = _words(["a", "b"], [0.0, 0.5], [0.4, 0.9])
    pred = _words(["a", "b"], [0.01, 0.52], [0.42, 0.95])

    monkeypatch.setattr(
        "natural_features.features.speech.benchmark.whisper_transcribe",
        lambda *args, **kwargs: {"words": ref, "qc": {"mode": "provided_transcript_uniform_alignment"}},
    )
    monkeypatch.setattr(
        "natural_features.features.speech.benchmark.whisperx_align",
        lambda *args, **kwargs: {"words": pred, "qc": {"mode": "whisperx", "fallback_used": False}},
    )

    out = benchmark_alignment_case(
        clip_id="clip1",
        audio=audio,
        reference_words=ref,
        transcript_text="a b",
        backend="whisperx",
        language="en",
    )
    assert out["clip_id"] == "clip1"
    assert out["n_matched_tokens"] == 2
    assert out["fallback_used"] is False
    assert out["boundary_mae_ms"] is not None
    assert float(out["boundary_mae_ms"]) > 0.0
    assert 0.0 <= float(out["token_f1"]) <= 1.0


def test_run_alignment_benchmark_manifest(tmp_path, monkeypatch) -> None:
    wav = tmp_path / "a.wav"
    _write_wav(wav, seconds=1.0)
    ref_words = _words(["alpha", "beta"], [0.0, 0.4], [0.3, 0.8])
    ctm = tmp_path / "a.ctm"
    write_ctm(ref_words, ctm)

    manifest = {
        "items": [
            {
                "id": "a",
                "audio_path": "a.wav",
                "reference_ctm": "a.ctm",
                "transcript": "alpha beta",
            }
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    pred = _words(["alpha", "beta"], [0.02, 0.43], [0.32, 0.83])
    monkeypatch.setattr(
        "natural_features.features.speech.benchmark.whisper_transcribe",
        lambda *args, **kwargs: {"words": ref_words, "qc": {"mode": "provided_transcript_uniform_alignment"}},
    )
    monkeypatch.setattr(
        "natural_features.features.speech.benchmark.whisperx_align",
        lambda *args, **kwargs: {"words": pred, "qc": {"mode": "whisperx", "fallback_used": False}},
    )

    report = run_alignment_benchmark(
        manifest_path,
        config=BenchmarkConfig(backend="whisperx", continue_on_error=False),
    )
    assert report["manifest_items"] == 1
    assert "runtime_pin_metadata" in report
    assert report["summary"]["n_success"] == 1
    assert report["summary"]["n_failed"] == 0
    assert report["summary"]["boundary_mae_ms_mean"] is not None
