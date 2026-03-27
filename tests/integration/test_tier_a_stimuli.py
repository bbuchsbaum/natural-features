from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from natural_features.core.stimulus import AudioStimulus, VideoStimulus
from natural_features.features.speech.asr import whisper_transcribe
from natural_features.features.speech.phonology import (
    acoustic_phone_posteriors,
    articulatory_features,
    articulatory_from_posteriors,
)
from natural_features.features.speech.vad import energy_vad
from natural_features.features.vision.motion import optical_flow_mag
from natural_features.features.vision.scene import scene_cuts


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "tests" / "stimuli" / "tier_a" / "manifest.json"


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _entry(entry_id: str) -> dict:
    for e in _manifest()["entries"]:
        if e["id"] == entry_id:
            return e
    raise KeyError(entry_id)


pytestmark = [pytest.mark.media, pytest.mark.smoke]


def test_tier_a_scene_cut_and_motion_diagnostics() -> None:
    ev = _entry("tier_a_video_scene_cut")
    p = ROOT / ev["path"]
    fps = ev["diagnostics"]["fps"]
    expected_cut = float(ev["diagnostics"]["expected_scene_cut_s"])
    high_win = ev["diagnostics"]["expected_motion_high_window_s"]

    video = VideoStimulus.from_npy(p, fps=fps)
    cuts = scene_cuts(video, threshold_z=2.0)
    assert len(cuts) >= 1
    nearest = float(np.min(np.abs(cuts.onset_s - expected_cut)))
    assert nearest <= 0.6

    flow = optical_flow_mag(video)
    t = flow.times_s
    pre = flow.values[(t >= 0.0) & (t < expected_cut), 0].mean()
    post = flow.values[(t >= high_win[0]) & (t < high_win[1]), 0].mean()
    assert post > pre


def test_tier_a_audio_energy_vad_diagnostics() -> None:
    ea = _entry("tier_a_audio_speechlike")
    p = ROOT / ea["path"]
    hi0, hi1 = ea["diagnostics"]["high_energy_window_s"]
    lo_windows = ea["diagnostics"]["low_energy_windows_s"]

    audio = AudioStimulus.from_wav(p)
    vad = energy_vad(audio, hop_s=0.02, win_s=0.03, threshold=0.5)
    t = vad.times_s
    hi = float(vad.values[(t >= hi0) & (t < hi1), 0].mean())
    lo = []
    for lo0, lo1 in lo_windows:
        lo.append(float(vad.values[(t >= lo0) & (t < lo1), 0].mean()))
    assert hi > max(lo) + 0.2


def test_tier_a_transcript_alignment_and_articulatory_features() -> None:
    ea = _entry("tier_a_audio_speechlike")
    et = _entry("tier_a_transcript_reference")
    audio = AudioStimulus.from_wav(ROOT / ea["path"])
    transcript = (ROOT / et["path"]).read_text(encoding="utf-8").strip()
    expected_words = int(et["diagnostics"]["word_count"])

    out = whisper_transcribe(audio, transcript_text=transcript, strict_dependency=False)
    words = out["words"]
    assert len(words) == expected_words
    assert np.all(np.diff(words.onset_s) >= 0)
    assert float(words.onset_s.min()) >= audio.start_offset_s
    audio_end = audio.start_offset_s + (audio.samples.shape[0] / audio.sr_hz)
    assert float(words.offset_s.max()) <= audio_end + 1e-6
    art = articulatory_features(words)
    assert art.values.shape[0] == expected_words


def test_tier_a_acoustic_posterior_articulatory_probabilities() -> None:
    ea = _entry("tier_a_audio_speechlike")
    audio = AudioStimulus.from_wav(ROOT / ea["path"])
    post = acoustic_phone_posteriors(audio, hop_s=0.02)
    art = articulatory_from_posteriors(post, include_uncertainty=True)
    names = list(art.coords.get("feature", []))
    assert post.values.shape[0] == art.values.shape[0]
    assert "bilabial" in names
    assert "alveolar" in names
    assert "posterior_peak" in names
