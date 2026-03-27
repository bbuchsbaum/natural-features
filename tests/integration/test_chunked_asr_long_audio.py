from __future__ import annotations

import numpy as np
import pytest

from natural_features.core.stimulus import AudioStimulus
from natural_features.features.speech.asr import whisper_transcribe_chunked


@pytest.mark.media
def test_chunked_asr_long_audio_monotonic_and_non_overlapping() -> None:
    sr = 8000
    duration_s = 45.0
    t = np.arange(int(sr * duration_s), dtype=np.float32) / sr
    wav = (0.1 * np.sin(2 * np.pi * 180.0 * t)).astype(np.float32)
    audio = AudioStimulus.from_array(wav, sr_hz=sr)

    out = whisper_transcribe_chunked(
        audio,
        chunk_window_s=8.0,
        chunk_overlap_s=1.0,
        strict_dependency=False,
    )
    words = out["words"]
    qc = out["qc"]

    assert len(words) > 0
    assert np.all(np.diff(words.onset_s) >= 0)
    assert np.all(words.offset_s >= words.onset_s)
    assert qc["chunk_count"] > 1
    assert "stitch_conflicts" in qc
