from __future__ import annotations

import numpy as np

from natural_features.core.stimulus import AudioStimulus
from natural_features.features.speech.align import whisperx_align
from natural_features.features.speech.asr import whisper_transcribe, whisper_transcribe_chunked


def _audio() -> AudioStimulus:
    sr = 8000
    t = np.arange(sr * 3, dtype=np.float32) / sr
    x = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return AudioStimulus.from_array(x, sr_hz=sr)


def test_transcript_driven_asr_is_deterministic() -> None:
    audio = _audio()
    transcript = "HH AH0 L OW1 W ER1 L D"
    a = whisper_transcribe(audio, transcript_text=transcript)
    b = whisper_transcribe(audio, transcript_text=transcript)
    np.testing.assert_allclose(a["words"].onset_s, b["words"].onset_s, atol=0.0)
    np.testing.assert_allclose(a["words"].offset_s, b["words"].offset_s, atol=0.0)
    assert list(a["words"].label) == list(b["words"].label)
    assert a["words"].metadata["params_hash"] == b["words"].metadata["params_hash"]


def test_transcript_driven_chunked_entrypoint_is_deterministic() -> None:
    audio = _audio()
    a = whisper_transcribe_chunked(
        audio,
        transcript_text="hello deterministic world",
        chunk_window_s=1.2,
        chunk_overlap_s=0.2,
    )
    b = whisper_transcribe_chunked(
        audio,
        transcript_text="hello deterministic world",
        chunk_window_s=1.2,
        chunk_overlap_s=0.2,
    )
    np.testing.assert_allclose(a["words"].onset_s, b["words"].onset_s, atol=0.0)
    np.testing.assert_allclose(a["words"].offset_s, b["words"].offset_s, atol=0.0)
    assert list(a["words"].label) == list(b["words"].label)
    assert a["qc"] == b["qc"]


def test_passthrough_alignment_is_deterministic() -> None:
    audio = _audio()
    asr = whisper_transcribe(audio, transcript_text="hello world")
    a = whisperx_align(audio, asr["words"], backend="none")
    b = whisperx_align(audio, asr["words"], backend="none")
    np.testing.assert_allclose(a["words"].onset_s, b["words"].onset_s, atol=0.0)
    np.testing.assert_allclose(a["words"].offset_s, b["words"].offset_s, atol=0.0)
    assert a["qc"]["mode"] == b["qc"]["mode"]
