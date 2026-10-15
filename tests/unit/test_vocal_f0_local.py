"""Independent analytic evidence for the optional local waveform veto."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from natural_features.core.stimulus import AudioStimulus
from natural_features.features.audio import vocal_f0 as vf


def _measure(x, f0=200., center=8000, floor=.01):
    return vf._local_periodicity(x, [center], [f0], floor, len(x) / 16000)[0]


@pytest.mark.parametrize('frequency', [50., 73.7, 100., 220., 440., 799.3, 1000.])
@pytest.mark.parametrize('phase', [0., .4, 1.8])
def test_true_tones_across_phase_and_fractional_period(frequency, phase):
    t = np.arange(16000) / 16000
    x = .1 * np.sin(2 * np.pi * frequency * t + phase)
    row = _measure(x, frequency)
    assert row[1] == 1
    # Direct continuous-wave oracle: a complete-period shift is identical.
    # Linear interpolation of at most1kHz sampled at16kHz changes this by <.001.
    assert row[0] == pytest.approx(1, abs=.001)
    assert row[2] > .001 and row[3] > .001


def test_missing_fundamental_and_sign_reversal():
    t = np.arange(16000) / 16000
    harmonic = sum(.1 / k * np.sin(2 * np.pi * 200 * k * t) for k in (2, 3, 4))
    assert _measure(harmonic)[0] == pytest.approx(1, abs=1e-12)
    tone = .1 * np.sin(2 * np.pi * 200 * t)
    assert _measure(tone, 400)[0] == pytest.approx(-1, abs=1e-12)


def test_silence_dc_and_amplitude_floor_transition():
    t = np.arange(16000) / 16000
    for x in (np.zeros(16000), np.full(16000, .3)):
        assert _measure(x)[0] == pytest.approx(0, abs=1e-20)
    # 200Hz has an integer period and both arms use identical fractional
    # sample phase: verify the floor against a direct sampled-vector energy.
    for amplitude in (.005, .01, .014, .015, .1):
        tone = amplitude * np.sin(2 * np.pi * 200 * t)
        row = _measure(tone)
        positions = 8000 + np.arange(320) - 159.5 - 40
        a = np.interp(positions, np.arange(16000), tone)
        energy = np.var(a)
        assert row[0] == pytest.approx(min(1, energy / .01**2), abs=1e-12)
        shifted = _measure(tone + .3)
        np.testing.assert_allclose(shifted, row, atol=1e-15)


def test_unobserved_support_is_not_zero_padding():
    x = np.sin(2 * np.pi * 50 * np.arange(16000) / 16000)
    rows = vf._local_periodicity(x, [0, 319, 320, 15679, 15680, 15999], np.full(6, 50.), .01, 1.)
    np.testing.assert_array_equal(rows[:, 1], [0, 0, 1, 1, 0, 0])
    assert np.isnan(rows[rows[:, 1] == 0, 0]).all()
    # Fractional resampling tail is not observed support.
    row = vf._local_periodicity(x, [15679], [50.], .01, .9999)[0]
    assert row[1] == 0 and np.isnan(row[0])


def test_boundary_and_noise_are_vetoed_but_periodic_instruments_can_pass():
    t = np.arange(16000) / 16000
    x = .1 * np.sin(2 * np.pi * 100 * t)
    x[t < .5] = 0
    centers = np.array([7200, 8000, 8800])
    rows = vf._local_periodicity(x, centers, np.full(3, 100.), .01, 1.)
    assert rows[0, 0] == 0
    assert rows[1, 0] < .75
    assert rows[2, 0] > .999
    noise = np.random.default_rng(21).normal(0, .1, 16000)
    assert abs(_measure(noise, 200)[0]) < .2
    # Sustained instrumental-like harmonic waves are an honest counterexample
    # to interpreting the statistic as a singer detector.
    instrument = sum(.03 / k * np.cos(2 * np.pi * 200 * k * t) for k in (1, 2, 3))
    assert _measure(instrument)[0] > .999


@pytest.mark.parametrize('threshold', [.25, .5, .75, .9])
def test_local_direct_replay_and_subset(monkeypatch, threshold):
    path = Path(__file__).resolve().parents[2] / 'scripts/calibrate_vocal_f0.py'
    spec = importlib.util.spec_from_file_location('cal', path)
    cal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cal)
    def separate(x):
        return np.column_stack([x[:, 0] * .1, x[:, 0] * .9])
    def pitch(x, hop, *args):
        return np.full(1 + len(x) // hop, 200.), np.full(1 + len(x) // hop, .9)
    monkeypatch.setattr(vf, '_load_runtime', lambda *a: SimpleNamespace(
        separate=separate, pitch=pitch, sources=['other', 'vocals'], versions={}))
    t = np.arange(16000) / 16000
    x = (.1 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    x[(t > .3) & (t < .5)] = 0
    stimulus = AudioStimulus.from_array(x, 16000, start_offset_s=7.25)
    base = vf.audio_vocal_f0(stimulus, hop_s=.0073)
    direct = vf.audio_vocal_f0(stimulus, hop_s=.0073, local_periodicity_threshold=threshold)
    replayed = cal.replay(base, .65, .03, .01, local_periodicity_threshold=threshold)
    np.testing.assert_array_equal(direct.values, replayed.values)
    keep = direct.values[:, 4] > 0
    assert not np.any(keep & (base.values[:, 4] == 0))
    np.testing.assert_array_equal(direct.values[keep, 0], base.values[keep, 0])
    assert np.isnan(direct.values[~keep, 0]).all()
    np.testing.assert_array_equal(direct.times_s, base.times_s)
    np.testing.assert_array_equal(direct.values[:, 3], base.values[:, 3])
    np.testing.assert_array_equal(cal.replay(direct, .65, .03, .01).values, base.values)


@pytest.mark.parametrize('scale', [.3, .5, 2.])
def test_local_agreement_penalizes_arm_amplitude_imbalance(scale):
    # At50Hz the two320sample arms cover adjacent full periods. Their
    # midpoint boundary falls on a zero crossing, so interpolation at that
    # boundary does not mix nonzero samples of different gains.
    t = np.arange(16000) / 16000
    x = .1 * np.sin(2 * np.pi * 50 * t)
    x[t >= .5] *= scale
    assert _measure(x, 50)[0] == pytest.approx(2 * scale / (1 + scale**2), abs=1e-12)


@pytest.mark.parametrize('threshold', [-.1, 1.1, np.nan, np.inf])
def test_invalid_local_threshold_rejected_before_model_loading(threshold, monkeypatch):
    monkeypatch.setattr(vf, '_load_runtime', lambda *a: pytest.fail('must validate before loading models'))
    with pytest.raises(ValueError, match='local_periodicity_threshold'):
        vf.audio_vocal_f0(AudioStimulus.from_array(np.zeros(16000, dtype=np.float32), 16000),
                          local_periodicity_threshold=threshold)
