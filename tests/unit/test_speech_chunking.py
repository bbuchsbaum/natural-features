from __future__ import annotations

import numpy as np

from natural_features.core.stimulus import AudioStimulus
from natural_features.features.speech.asr import whisper_transcribe_chunked
from natural_features.features.speech.chunking import aggregate_chunk_qc, plan_audio_chunks, stitch_word_events


def _audio(duration_s: float = 6.0, sr: int = 8000) -> AudioStimulus:
    t = np.arange(int(sr * duration_s), dtype=np.float32) / sr
    x = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return AudioStimulus.from_array(x, sr_hz=sr)


def test_plan_audio_chunks_deterministic_and_covering() -> None:
    stim = _audio(duration_s=6.0, sr=8000)
    c1 = plan_audio_chunks(n_samples=stim.samples.shape[0], sr_hz=stim.sr_hz, window_s=2.0, overlap_s=0.5)
    c2 = plan_audio_chunks(n_samples=stim.samples.shape[0], sr_hz=stim.sr_hz, window_s=2.0, overlap_s=0.5)
    assert c1 == c2
    assert c1[0].sample_start == 0
    assert c1[-1].sample_end == stim.samples.shape[0]
    for c in c1:
        assert c.sample_end > c.sample_start
        assert c.end_s > c.start_s


def test_stitch_word_events_dedupes_overlap_tokens() -> None:
    stim = _audio(duration_s=2.0, sr=1000)
    w1 = whisper_transcribe_chunked(
        stim,
        chunk_window_s=2.0,
        chunk_overlap_s=0.0,
        execution_mode="fallback",
    )["words"]
    # Force overlap-style duplicate by stitching same object twice.
    out, conflicts = stitch_word_events([w1, w1], dedupe_tolerance_s=0.2)
    assert len(out) <= len(w1)
    assert conflicts >= 1
    assert np.all(np.diff(out.onset_s) >= 0)


def test_aggregate_chunk_qc_contract() -> None:
    qc = aggregate_chunk_qc(
        [
            {"n_words": 3, "low_confidence_words": 1, "dropped_words": 0, "coverage_fraction": 0.8},
            {"n_words": 2, "low_confidence_words": 0, "dropped_words": 1, "coverage_fraction": 0.6},
        ],
        chunk_count=2,
        stitch_conflicts=1,
    )
    assert qc["mode"] == "chunked"
    assert qc["n_words"] == 5
    assert qc["low_confidence_words"] == 1
    assert qc["dropped_words"] == 1
    assert qc["chunk_count"] == 2
    assert qc["stitch_conflicts"] == 1


def test_whisper_transcribe_chunked_monotonic_and_chunk_qc() -> None:
    stim = _audio(duration_s=7.0, sr=8000)
    out = whisper_transcribe_chunked(
        stim,
        chunk_window_s=2.5,
        chunk_overlap_s=0.5,
        strict_dependency=False,
    )
    words = out["words"]
    qc = out["qc"]
    assert len(words) > 0
    assert np.all(np.diff(words.onset_s) >= 0)
    assert qc["chunk_count"] > 1
    assert "stitch_conflicts" in qc
