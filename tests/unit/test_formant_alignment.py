from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from natural_features.core.stimulus import AudioStimulus
from natural_features.features.audio import phonetic


def _tone(sr=8000, n=8000, offset=0.0):
    samples = 0.3 * np.sin(2 * np.pi * 200 * np.arange(n) / sr)
    return AudioStimulus.from_array(samples.astype(np.float32), sr, start_offset_s=offset)


@pytest.mark.parametrize("offset", [0.0, 17.25])
def test_gate_interpolates_strength_at_formant_centers(monkeypatch, offset):
    # Known ramps cross 0.3 at 23.75 and 46.25 ms. The retained centers
    # must be 32.5 and 42.5 ms; a row-index join retains 22.5 and 32.5 ms.
    def pitch(stimulus, **kwargs):
        assert kwargs == {"hop_s": 0.01, "win_s": 0.04}
        return SimpleNamespace(
            times_s=stimulus.start_offset_s + np.array([0.02, 0.03, 0.04, 0.05, 0.06]),
            values=np.array([[200, 0.0], [200, 0.8], [200, 0.8], [200, 0.0], [200, 0.0]]),
        )

    monkeypatch.setattr(phonetic, "audio_pitch", pitch)
    monkeypatch.setattr(phonetic, "_formants_lpc", lambda *a, **kw: np.ones((7, 6)))
    out = phonetic.audio_formants(_tone(offset=offset))
    np.testing.assert_allclose(out.times_s - offset, [0.0125, 0.0225, 0.0325, 0.0425, 0.0525, 0.0625, 0.0725])
    np.testing.assert_array_equal(out.values[:, 0], [0, 0, 1, 1, 0, 0, 0])
    np.testing.assert_array_equal(out.values[[0, 1, 4, 5, 6]], 0)
    assert out.metadata["voicing_alignment"] == "linear_strength_at_formant_centers_zero_outside"
    assert out.metadata["code_version"] == "formants-v2"


@pytest.mark.parametrize("win_s", [0.025, 0.04, 0.055])
@pytest.mark.parametrize("n", [80, 327, 8000])
def test_gate_preserves_grid_and_offset_invariance(win_s, n):
    audio = _tone(n=n)
    raw = phonetic.audio_formants(audio, win_s=win_s, voicing_gate=False)
    gated = phonetic.audio_formants(audio, win_s=win_s)
    shifted = phonetic.audio_formants(_tone(n=n, offset=17.25), win_s=win_s)
    np.testing.assert_array_equal(gated.times_s, raw.times_s)
    np.testing.assert_array_equal(gated.time_bounds_s, raw.time_bounds_s)
    np.testing.assert_allclose(shifted.times_s, gated.times_s + 17.25)
    np.testing.assert_allclose(shifted.time_bounds_s, gated.time_bounds_s + 17.25)
    np.testing.assert_array_equal(shifted.values, gated.values)
    kept = np.any(gated.values != 0, axis=1)
    np.testing.assert_array_equal(gated.values[kept], raw.values[kept])


def test_sample_rounded_clock_and_short_padding(monkeypatch):
    pitch = phonetic.audio_pitch

    def check_pitch_clock(*args, **kwargs):
        out = pitch(*args, **kwargs)
        if args[0].sr_hz == 11025:
            np.testing.assert_allclose(out.times_s[[0, -1]], np.array([220.5, 10780.5]) / 11025)
            np.testing.assert_allclose(np.diff(out.times_s), 80 / 11025)
        return out

    monkeypatch.setattr(phonetic, "audio_pitch", check_pitch_clock)
    out = phonetic.audio_formants(_tone(sr=11025, n=11025), hop_s=0.0073, win_s=0.0251)
    # 80-sample hop, 277-sample window; last start is 10720, not
    # the nominal-hop time accumulated over 134 hops.
    assert len(out.times_s) == 135
    np.testing.assert_allclose(out.times_s[[0, -1]], np.array([138.5, 10858.5]) / 11025)
    np.testing.assert_allclose(out.time_bounds_s[-1], np.array([10720, 10997]) / 11025)
    assert out.timebase.hop_s == 80 / 11025
    assert out.timebase.window_s == 277 / 11025
    short = phonetic.audio_formants(_tone(n=80), voicing_gate=False)
    assert short.metadata["right_padding_samples"] == 120
    np.testing.assert_allclose(short.time_bounds_s, [[0, 0.025]])


@pytest.mark.parametrize("win_s", [0.04, 0.055])
def test_identical_pitch_and_formant_grids_keep_voiced_edge_rows(win_s):
    audio = _tone(sr=11025, n=11025)
    raw = phonetic.audio_formants(audio, win_s=win_s, voicing_gate=False)
    gated = phonetic.audio_formants(audio, win_s=win_s)
    np.testing.assert_array_equal(gated.values, raw.values)


def test_praat_queries_use_local_sample_centers(monkeypatch):
    queries = []

    class Formant:
        def get_value_at_time(self, k, t):
            queries.append(t)
            return 200 * k + t

        def get_bandwidth_at_time(self, k, t):
            return 50 * k + t

    class Sound:
        def __init__(self, samples, sampling_frequency):
            assert len(samples) == 11025
            assert sampling_frequency == 11025

        def to_formant_burg(self, **kwargs):
            return Formant()

    monkeypatch.setitem(sys.modules, "parselmouth", SimpleNamespace(Sound=Sound))
    out = phonetic.audio_formants(
        _tone(sr=11025, n=11025, offset=17.25), backend="parselmouth",
        hop_s=0.0073, win_s=0.0251, voicing_gate=False,
    )
    assert len(out.times_s) == 135
    assert out.time_bounds_s is None
    assert out.timebase.support.kind == "point"
    np.testing.assert_allclose(np.array(queries).reshape(-1, 3)[:, 0], out.times_s - 17.25)
    np.testing.assert_allclose(out.values[:, 0], 200 + out.times_s - 17.25, atol=2e-5)


def test_real_praat_offset_invariance():
    pytest.importorskip("parselmouth")
    a = phonetic.audio_formants(_tone(), backend="parselmouth", voicing_gate=False)
    b = phonetic.audio_formants(_tone(offset=17.25), backend="parselmouth", voicing_gate=False)
    assert np.any(a.values[:, :3] > 0)
    np.testing.assert_array_equal(a.values, b.values)
    np.testing.assert_allclose(b.times_s, a.times_s + 17.25)


@pytest.mark.parametrize("param", ["hop_s", "win_s"])
@pytest.mark.parametrize("value", [0, -0.01, np.nan, np.inf])
def test_invalid_framing_rejected(param, value):
    with pytest.raises(ValueError, match="finite and positive"):
        phonetic.audio_formants(_tone(), **{param: value})
