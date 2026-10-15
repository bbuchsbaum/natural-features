from types import SimpleNamespace

import numpy as np
import pytest

from natural_features.core.backend_errors import BackendInferenceError, BackendLoadError
from natural_features.core.stimulus import AudioStimulus
from natural_features.features.audio import vocal_f0 as vf


@pytest.fixture
def runtime(monkeypatch):
    def separate(stereo):
        mono = stereo.mean(axis=1)
        return np.column_stack([mono * 0, mono * 0, mono * 0.5, mono * 0.5])

    def pitch(audio, hop, fmin, fmax):
        n = 1 + len(audio) // hop
        return np.full(n, 220.0), np.full(n, 0.9)

    rt = SimpleNamespace(separate=separate, pitch=pitch,
                         sources=["drums", "bass", "other", "vocals"], versions={"test": "fixed"})
    monkeypatch.setattr(vf, "_load_runtime", lambda *args: rt)
    return rt


def _audio(sr=16000, n=16000, offset=0):
    x = 0.1 * np.sin(2 * np.pi * 220 * np.arange(n) / sr)
    return AudioStimulus.from_array(x.astype(np.float32), sr, start_offset_s=offset, clock="song")


def test_source_support_clock_offset_missingness_and_context(runtime):
    a = vf.audio_vocal_f0(_audio())
    b = vf.audio_vocal_f0(_audio(offset=7.25))
    np.testing.assert_allclose(a.values, b.values, equal_nan=True)
    np.testing.assert_allclose(b.times_s, 7.25 + np.arange(101) / 100)
    np.testing.assert_allclose(a.values[:-1, 2], 0.5, atol=1e-6)
    assert np.all(a.values[:-1, 0] == 220)
    assert np.isnan(a.values[-1, 0]) and a.values[-1, 5] == 1
    assert a.values[0, -1] == 1 and a.values[-1, -1] == 1
    assert a.clock == "song"
    assert b.metadata["computation_context"]["dependency_bounds_s"] == [[7.25, 8.25]]
    assert b.metadata["computation_context"]["policy"] == "bidirectional"
    np.testing.assert_allclose(b.time_bounds_s[[0, -1]], [[7.25, 7.282], [8.218, 8.25]])


def test_silence_and_weak_source_never_become_zero_pitch(runtime):
    silence = AudioStimulus.from_array(np.zeros(16000, np.float32), 16000)
    a = vf.audio_vocal_f0(silence)
    assert np.isnan(a.values[:, 0]).all()
    assert not a.values[:, 3:5].any()
    assert a.values[:, 5].all()
    # High pitch confidence is insufficient when the selected stem is weak.
    old = runtime.separate
    runtime.separate = lambda x: old(x) * np.array([1, 1, 1, 0.01])
    b = vf.audio_vocal_f0(_audio())
    assert np.isnan(b.values[:, 0]).all()
    assert np.all(b.values[:, 1] == pytest.approx(0.9))


def test_awkward_resampling_and_short_input(runtime):
    a = vf.audio_vocal_f0(_audio(sr=11025, n=11026), hop_s=0.0073)
    assert len(a.times_s) == 137
    np.testing.assert_allclose(np.diff(a.times_s), 117 / 16000)
    assert a.metadata["effective_hop_s"] == 117 / 16000
    short = vf.audio_vocal_f0(_audio(n=1))
    assert short.values.shape == (1, len(vf.FEATURES))
    assert short.values[0, -1] == 1
    # A resampler's fractional tail is not real source audio.
    fractional = vf.audio_vocal_f0(_audio(sr=11025, n=11047))
    assert fractional.values[97, -1] == 1
    assert fractional.time_bounds_s[97, 1] == pytest.approx(11047 / 11025)


def test_octave_and_range_flags_are_diagnostics(runtime):
    def pitch(audio, hop, *args):
        n = 1 + len(audio) // hop
        f0 = np.full(n, 220.0)
        f0[1:] = 440
        f0[3] = 50.1
        return f0, np.ones(n)
    runtime.pitch = pitch
    a = vf.audio_vocal_f0(_audio())
    assert a.values[1, 7] == 1
    assert a.values[3, 6] == 1


@pytest.mark.parametrize("failure", ["count", "nan", "confidence", "stem"])
def test_inference_disagreement_fails_without_fallback(runtime, failure):
    if failure == "stem":
        runtime.separate = lambda x: np.zeros(3)
    else:
        def pitch(audio, hop, *args):
            n = 1 + len(audio) // hop
            return (np.full(n - (failure == "count"), np.nan if failure == "nan" else 220),
                    np.full(n, 2 if failure == "confidence" else 0.9))
        runtime.pitch = pitch
    with pytest.raises(BackendInferenceError):
        vf.audio_vocal_f0(_audio())


def test_asset_failure_is_not_reported_as_inference(monkeypatch):
    def fail(*args):
        raise BackendLoadError("test", "missing pinned model")
    monkeypatch.setattr(vf, "_load_runtime", fail)
    with pytest.raises(BackendLoadError):
        vf.audio_vocal_f0(_audio())


@pytest.mark.parametrize("kwargs", [{"fmin": 0}, {"fmax": 3000}, {"hop_s": np.nan},
                                     {"confidence_threshold": 1.1}, {"source_threshold": -1},
                                     {"min_source_rms": 0}, {"local_files_only": "yes"}])
def test_invalid_parameters_rejected(runtime, kwargs):
    with pytest.raises(ValueError):
        vf.audio_vocal_f0(_audio(), **kwargs)


def test_source_gate_cannot_turn_dc_into_vocal_support(monkeypatch):
    torch = pytest.importorskip("torch")
    apply = pytest.importorskip("demucs.apply")

    def separate(model, mix, **kwargs):
        # Nearly all AC energy belongs to the instrumental source.
        return torch.stack([mix, mix * 0.001], dim=1)

    monkeypatch.setattr(apply, "apply_model", separate)
    model = SimpleNamespace(sources=["other", "vocals"])
    rt = vf._Runtime(torch, None, model, None, "cpu", {"test": "fixed"})
    x = 0.1 * np.sin(2 * np.pi * 220 * np.arange(44100) / 44100)
    ordinary = rt.separate(np.repeat(x[:, None], 2, axis=1).astype(np.float32))
    shifted = rt.separate(np.repeat((x + 0.1)[:, None], 2, axis=1).astype(np.float32))
    np.testing.assert_allclose(shifted, ordinary, atol=2e-8)
    for stems in (ordinary, shifted):
        energies = np.mean(stems.astype(np.float64) ** 2, axis=0)
        assert energies[1] / energies.sum() < 1e-5


@pytest.mark.parametrize("failure", ["missing", "unreadable"])
def test_asset_inspection_errors_are_typed(monkeypatch, tmp_path, failure):
    pytest.importorskip("demucs")
    crepe = pytest.importorskip("torchcrepe")
    checkpoint = tmp_path / "checkpoint.th"
    checkpoint.write_bytes(b"present")
    if failure == "missing":
        monkeypatch.setattr(crepe, "__file__", str(tmp_path / "uninstalled" / "__init__.py"))
    else:
        def fail(path):
            raise PermissionError("not readable")
        monkeypatch.setattr(vf, "_sha256", fail)
    with pytest.raises(BackendLoadError) as error:
        vf._load_runtime(str(checkpoint), None, True, "cpu")
    assert isinstance(error.value.__cause__, OSError)
