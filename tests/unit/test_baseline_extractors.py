from __future__ import annotations

import numpy as np

from natural_features.core.stimulus import AudioStimulus, VideoStimulus
from natural_features.features.audio.lowlevel import mel, mfcc, rms, spectral_stats
from natural_features.features.speech.vad import energy_vad
from natural_features.features.vision.dynamics import frame_diffs
from natural_features.features.vision.lowlevel import visual_energy
from natural_features.features.vision.motion import optical_flow_mag
from natural_features.features.vision.scene import scene_cuts


def _video() -> VideoStimulus:
    rng = np.random.default_rng(7)
    frames = (rng.uniform(0, 255, size=(12, 32, 32, 3))).astype(np.uint8)
    frames[6:] = np.clip(frames[6:] + 60, 0, 255)
    return VideoStimulus.from_array(frames, fps=6.0)


def _audio() -> AudioStimulus:
    sr = 16000
    t = np.arange(sr * 2, dtype=np.float32) / sr
    x = (0.3 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return AudioStimulus.from_array(x, sr_hz=sr)


def test_vision_baselines_shapes() -> None:
    v = _video()
    ve = visual_energy(v)
    fd = frame_diffs(v)
    om = optical_flow_mag(v)
    sc = scene_cuts(v)
    assert ve.values.shape[0] == len(v.frame_times_s)
    assert fd.values.shape[1] == 2
    assert om.values.shape[1] == 2
    assert sc.onset_s.ndim == 1


def test_audio_baselines_shapes() -> None:
    a = _audio()
    r = rms(a)
    m = mel(a, n_mels=16)
    c = mfcc(a, n_mfcc=8, n_mels=16)
    s = spectral_stats(a)
    v = energy_vad(a)
    assert r.values.shape[1] == 1
    assert m.values.shape[1] == 16
    assert c.values.shape[1] == 16
    assert s.values.shape[1] == 5
    assert v.values.shape[1] == 1
