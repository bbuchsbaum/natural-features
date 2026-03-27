from __future__ import annotations

import numpy as np
import pytest

from natural_features.core.stimulus import AudioStimulus, VideoStimulus
from natural_features.features.audio.opensmile import egemaps_lld
from natural_features.features.vision.motion_energy import motion_energy_pymoten


def test_motion_energy_wrapper_fallback_or_backend() -> None:
    frames = np.random.default_rng(0).integers(0, 255, size=(8, 16, 16, 3), dtype=np.uint8)
    v = VideoStimulus.from_array(frames, fps=5.0)
    out = motion_energy_pymoten(v, strict_dependency=False)
    assert out.values.shape[0] == len(out.times_s)


def test_motion_energy_wrapper_strict_dependency_error() -> None:
    frames = np.random.default_rng(1).integers(0, 255, size=(6, 8, 8, 3), dtype=np.uint8)
    v = VideoStimulus.from_array(frames, fps=5.0)
    try:
        import moten  # type: ignore  # noqa: F401
    except Exception:
        with pytest.raises(RuntimeError):
            motion_energy_pymoten(v, strict_dependency=True)


def test_opensmile_wrapper_fallback_or_backend() -> None:
    sr = 8000
    t = np.arange(sr, dtype=np.float32) / sr
    a = AudioStimulus.from_array((0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr_hz=sr)
    out = egemaps_lld(a, frame_s=0.02, strict_dependency=False)
    assert out.values.shape[0] == len(out.times_s)


def test_opensmile_wrapper_strict_dependency_error() -> None:
    sr = 8000
    t = np.arange(sr, dtype=np.float32) / sr
    a = AudioStimulus.from_array((0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr_hz=sr)
    try:
        import opensmile  # type: ignore  # noqa: F401
    except Exception:
        with pytest.raises(RuntimeError):
            egemaps_lld(a, strict_dependency=True)
