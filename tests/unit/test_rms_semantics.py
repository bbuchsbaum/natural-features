import numpy as np
import pytest
from natural_features.core.stimulus import AudioStimulus
from natural_features.features.audio.lowlevel import rms, _stft_power
from natural_features.core.registry import Registry


def test_constant_raw_amplitude_and_gain():
    x = np.full(16000, .2, dtype=np.float32)
    a = rms(AudioStimulus(x, 16000), win_s=1, hop_s=1)
    np.testing.assert_allclose(a.values, .2, rtol=1e-6)
    b = rms(AudioStimulus(3*x, 16000), win_s=1, hop_s=1)
    np.testing.assert_allclose(b.values, 3*a.values, rtol=1e-6)
    assert a.metadata['code_version'] == 'rms-untapered-v2'


def test_full_cycle_sine_rms():
    t = np.arange(16000) / 16000
    x = (.2*np.sin(2*np.pi*440*t)).astype(np.float32)
    a = rms(AudioStimulus(x, 16000), win_s=1, hop_s=1)
    np.testing.assert_allclose(a.values, .2/np.sqrt(2), rtol=1e-6)


@pytest.mark.parametrize('position', [0, 3, 80, 159])
def test_impulse_position_does_not_change_energy(position):
    x = np.zeros(160, dtype=np.float32)
    x[position] = .5
    a = rms(AudioStimulus(x, 16000), win_s=.01, hop_s=.01)
    np.testing.assert_allclose(a.values, .5/np.sqrt(160), rtol=1e-6)


def test_native_offset_and_existing_short_input_padding():
    x = np.ones(240, dtype=np.float32)
    a = rms(AudioStimulus(x, 16000, start_offset_s=7), win_s=.01, hop_s=.005)
    np.testing.assert_allclose(a.times_s, [7.005, 7.01])
    np.testing.assert_allclose(a.values, 1, rtol=1e-6)
    b = rms(AudioStimulus(x[:80], 16000), win_s=.01, hop_s=.01)
    np.testing.assert_allclose(b.values, np.sqrt(.5), rtol=1e-6)


def test_spectral_frontend_is_still_periodic_hann():
    x = np.random.default_rng(45).normal(size=800).astype(np.float32)
    actual, freq, starts = _stft_power(x, 16000, .005, .025)
    frame = x[np.arange(400)[None,:]+starts[:,None]]
    frame = frame*np.hanning(401).astype(np.float32)[:-1]
    expected = (np.abs(np.fft.rfft(frame, axis=1))**2).astype(np.float32)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(freq, np.fft.rfftfreq(400, 1/16000))


def test_aliases_have_versioned_cache_semantics():
    r = Registry.with_builtin_specs()
    x = AudioStimulus(np.full(400,.2,dtype=np.float32),16000)
    for name in ['audio.rms','audio.lowlevel.rms']:
        assert r.get(name).version == '1.1'
        out = r.impl(name)(x)
        np.testing.assert_allclose(out.values,.2,rtol=1e-6)
