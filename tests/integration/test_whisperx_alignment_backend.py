from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from natural_features.core.stimulus import AudioStimulus
from natural_features.features.speech.align import whisperx_align
from natural_features.features.speech.asr import whisper_transcribe


ROOT = Path(__file__).resolve().parents[2]
TIER_A_AUDIO = ROOT / "tests" / "stimuli" / "tier_a" / "audio_speechlike.wav"
TIER_A_TXT = ROOT / "tests" / "stimuli" / "tier_a" / "transcript_reference.txt"


@pytest.mark.media
def test_whisperx_backend_refines_boundaries_when_available() -> None:
    if importlib.util.find_spec("whisperx") is None:
        pytest.skip("whisperx is not installed")

    audio = AudioStimulus.from_wav(TIER_A_AUDIO)
    transcript = TIER_A_TXT.read_text(encoding="utf-8").strip()
    asr = whisper_transcribe(audio, transcript_text=transcript, strict_dependency=False)
    before = asr["words"]
    aligned = whisperx_align(
        audio,
        before,
        backend="whisperx",
        strict_dependency=False,
    )
    after = aligned["words"]
    qc = aligned["qc"]

    if qc["mode"] != "whisperx" or qc["fallback_used"]:
        pytest.skip("whisperx runtime path unavailable (model/assets missing)")

    assert len(after) > 0
    assert qc["mode"] == "whisperx"
    assert qc["fallback_used"] is False
    assert (after.onset_s != before.onset_s).any() or (after.offset_s != before.offset_s).any()
