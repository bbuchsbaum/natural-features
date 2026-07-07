from __future__ import annotations

import sys
import types
from types import SimpleNamespace

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


def test_motion_energy_wrapper_strict_backend_with_fake_moten(monkeypatch) -> None:  # noqa: ANN001
    class FakePyramid:
        def __init__(self, stimulus_vhsize: tuple[int, int], stimulus_fps: float):
            assert stimulus_vhsize == (8, 8)
            assert stimulus_fps == 5.0

        def project_stimulus(self, gray: np.ndarray) -> np.ndarray:
            assert gray.shape == (4, 8, 8)
            return np.column_stack(
                [
                    np.linspace(0.0, 1.0, gray.shape[0], dtype=np.float32),
                    np.linspace(1.0, 0.0, gray.shape[0], dtype=np.float32),
                ]
            )

    moten = types.ModuleType("moten")
    moten.pyramids = SimpleNamespace(MotionEnergyPyramid=FakePyramid)
    monkeypatch.setitem(sys.modules, "moten", moten)

    frames = np.random.default_rng(3).integers(0, 255, size=(4, 8, 8, 3), dtype=np.uint8)
    v = VideoStimulus.from_array(frames, fps=5.0)
    out = motion_energy_pymoten(v, execution_mode="strict", strict_dependency=True)

    assert out.metadata["backend"] == "pymoten"
    assert out.metadata["fallback_used"] is False
    assert out.schema == "FeatureSeries/v1"
    assert out.values.shape == (4, 2)
    assert np.isfinite(out.values).all()


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


def test_opensmile_wrapper_strict_backend_with_fake_opensmile(monkeypatch) -> None:  # noqa: ANN001
    class FakeFrame:
        columns = ["f0", "energy"]
        index = object()

        def to_numpy(self, dtype: object) -> np.ndarray:
            return np.asarray([[1.0, 0.2], [2.0, 0.4], [3.0, 0.6]], dtype=dtype)

        def __len__(self) -> int:
            return 3

    class FakeSmile:
        def __init__(self, feature_set: str, feature_level: str):
            assert feature_set == "eGeMAPSv02"
            assert feature_level == "LowLevelDescriptors"

        def process_signal(self, x: np.ndarray, sr_hz: int) -> FakeFrame:
            assert x.ndim == 1
            assert sr_hz == 8000
            return FakeFrame()

    opensmile = types.ModuleType("opensmile")
    opensmile.FeatureSet = SimpleNamespace(eGeMAPSv02="eGeMAPSv02")
    opensmile.FeatureLevel = SimpleNamespace(LowLevelDescriptors="LowLevelDescriptors")
    opensmile.Smile = FakeSmile
    monkeypatch.setitem(sys.modules, "opensmile", opensmile)

    sr = 8000
    t = np.arange(sr // 2, dtype=np.float32) / sr
    a = AudioStimulus.from_array((0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr_hz=sr)
    out = egemaps_lld(a, frame_s=0.02, execution_mode="strict", strict_dependency=True)

    assert out.metadata["backend"] == "opensmile"
    assert out.metadata["fallback_used"] is False
    assert out.schema == "FeatureSeries/v1"
    assert out.values.shape == (3, 2)
    assert out.coords["feature"] == ["f0", "energy"]
    assert np.isfinite(out.values).all()
