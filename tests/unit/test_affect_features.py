from __future__ import annotations

import numpy as np

from natural_features.core.stimulus import AudioStimulus, VideoStimulus
from natural_features.features.affect.audio import audio_affect_proxies
from natural_features.features.affect.visual import social_visual_proxies


def test_audio_affect_proxies_shape() -> None:
    sr = 8000
    t = np.arange(sr, dtype=np.float32) / sr
    x = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    a = AudioStimulus.from_array(x, sr_hz=sr)
    out = audio_affect_proxies(a)
    assert out.values.shape[1] == 3


def test_visual_social_proxies_shape() -> None:
    frames = np.random.default_rng(4).integers(0, 255, size=(10, 12, 12, 3), dtype=np.uint8)
    v = VideoStimulus.from_array(frames, fps=5.0)
    out = social_visual_proxies(v)
    assert out.values.shape == (10, 3)
