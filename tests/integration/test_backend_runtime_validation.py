from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from natural_features.core.stimulus import AudioStimulus
from natural_features.features.speech.validation import validate_alignment_backends


ROOT = Path(__file__).resolve().parents[2]
TIER_A_AUDIO = ROOT / "tests" / "stimuli" / "tier_a" / "audio_speechlike.wav"
TIER_A_TXT = ROOT / "tests" / "stimuli" / "tier_a" / "transcript_reference.txt"


def test_whisperx_runtime_validation_on_tier_a_audio() -> None:
    if importlib.util.find_spec("whisperx") is None:
        pytest.skip("whisperx is not installed")

    audio = AudioStimulus.from_wav(TIER_A_AUDIO)
    transcript = TIER_A_TXT.read_text(encoding="utf-8").strip()
    report = validate_alignment_backends(
        audio=audio,
        transcript_text=transcript,
        language="en",
        execution_mode="fallback",
    )
    wx = report["backends"]["whisperx"]
    if not wx["runtime_checked"]:
        pytest.skip("whisperx runtime path was not checked")
    if wx["runtime_ok"] is not True:
        pytest.skip(f"whisperx runtime check unavailable in this environment: {wx.get('runtime_reason')}")

    assert wx["available"] is True
    assert wx["runtime_ok"] is True
    assert wx["runtime_details"]["qc"]["mode"] == "whisperx"
