from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from natural_features.core.stimulus import AudioStimulus
from natural_features.features.speech.align import whisperx_align
from natural_features.features.speech.asr import whisper_transcribe
from natural_features.features.speech.phonology import (
    articulatory_from_phoneme_events,
    phoneme_events_from_words,
)


ROOT = Path(__file__).resolve().parents[2]
TIER_A_AUDIO = ROOT / "tests" / "stimuli" / "tier_a" / "audio_speechlike.wav"


@pytest.mark.media
def test_asr_alignment_to_phoneme_to_articulatory_pipeline() -> None:
    # Phone-coded transcript to validate interval and articulatory contracts.
    phone_transcript = "HH AH0 L OW1 W ER1 L D"
    audio = AudioStimulus.from_wav(TIER_A_AUDIO)
    asr = whisper_transcribe(audio, transcript_text=phone_transcript, strict_dependency=False)
    aligned = whisperx_align(audio, asr["words"], backend="none", strict_dependency=False)
    phones = phoneme_events_from_words(
        aligned["words"],
        label_namespace="arpabet",
        namespace_version="cmu-v1",
    )
    art = articulatory_from_phoneme_events(phones)

    assert len(phones) > 0
    assert np.all(np.diff(phones.onset_s) >= 0)
    assert phones.metadata["label_namespace"] == "arpabet"
    assert phones.metadata["namespace_version"] == "cmu-v1"
    assert phones.metadata["source_word_alignment_id"]
    assert art.values.shape[0] == len(phones)
    assert "event_confidence" in art.coords.get("feature", [])
