"""Known-signal, native support and public-contract tests for local modulation."""
import numpy as np
import pytest
from natural_features.core.stimulus import AudioStimulus
from natural_features.features.audio.local_modulation import (
    local_modulation_kernel,
    local_spectrotemporal_modulation,
)


def tone(seconds=4, sr=16000, offset=0):
    t = np.arange(round(seconds * sr)) / sr
    x = ((1 + .4 * np.sin(2 * np.pi * 4 * t)) * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return AudioStimulus(x, sr, start_offset_s=offset)


def test_known_ripple_carrier_rate_scale_direction():
    f = 2 ** np.linspace(6, 13, 128)
    t = np.arange(500) / 100
    x = np.log2(f / 512)
    z = np.cos(2 * np.pi * (x[None, :] + 4 * t[:, None])) * np.exp(-.5 * (x[None, :] / .5) ** 2)
    kw = dict(frequencies_hz=f, frame_rate_hz=100, carriers_hz=(128, 512, 4096),
              rates_hz=(2, 4, 8), scales_cpo=(.5, 1, 2))
    a = local_modulation_kernel(z, **kw).mean(0)
    assert np.unravel_index(a.argmax(), a.shape) == (1, 1, 1, 1)
    assert a[1, 1, 1, 1] > 5 * a[1, 1, 1, 0]
    assert a[1, 1, 1, 1] > 5 * a[2, 1, 1, 1]
    b = local_modulation_kernel(z[::-1], **kw).mean(0)
    assert np.unravel_index(b.argmax(), b.shape) == (1, 1, 1, 0)


def test_log_offset_invariance_and_future_independence():
    f = 2 ** np.linspace(6, 13, 64)
    z = np.random.default_rng(8).normal(size=(400, 64))
    kw = dict(frequencies_hz=f, frame_rate_hz=100)
    a = local_modulation_kernel(z[:300], **kw)
    np.testing.assert_allclose(a, local_modulation_kernel(z, **kw)[:len(a)], atol=1e-12)
    np.testing.assert_allclose(a, local_modulation_kernel(z[:300] + 8, **kw), atol=1e-12)
    np.testing.assert_allclose(local_modulation_kernel(np.ones_like(z), **kw), 0, atol=1e-12)


def test_kernel_matches_direct_manual_correlation():
    # An independent small complex-filter calculation validates numerical normalization.
    f = 2 ** np.linspace(6, 10, 32)
    z = np.random.default_rng(2).normal(size=(240, len(f)))
    x = np.log2(f / 256)
    wf = np.exp(-.5 * (x / .5) ** 2)
    of = np.exp(2j * np.pi * x)
    kf = wf * (of - (wf @ of) / wf.sum())
    kf /= np.linalg.norm(kf)
    t = np.arange(200) / 100
    wt = np.hanning(200)
    ot = np.exp(2j * np.pi * 4 * t)
    kt = wt * (ot - (wt @ ot) / wt.sum())
    kt /= np.linalg.norm(kt)
    expected = np.array([abs(np.vdot(kt, z[i:i+200] @ np.conj(kf))) for i in range(41)])
    a = local_modulation_kernel(z, frequencies_hz=f, frame_rate_hz=100,
                               carriers_hz=(256,), rates_hz=(4,), scales_cpo=(1,))
    np.testing.assert_allclose(a[:, 0, 0, 0, 1], expected, atol=1e-12)


def test_waveform_dependency_cells_offset_and_append():
    a = local_spectrotemporal_modulation(tone(3, offset=10))
    b = local_spectrotemporal_modulation(tone(4, offset=10))
    assert a.dims == ('time', 'carrier_hz', 'rate_hz', 'scale_cpo', 'spectral_sign')
    assert a.values.shape[1:] == (6, 4, 3, 2)
    dep = np.array(a.metadata['computation_context']['dependency_bounds_s'])
    assert dep[0, 0] == 10
    assert dep[-1, 1] <= 13
    np.testing.assert_allclose(a.times_s, dep[:, 1])
    np.testing.assert_allclose(a.time_bounds_s[:, 1], a.times_s)
    np.testing.assert_allclose(np.diff(a.time_bounds_s, axis=1), .1, atol=1e-12)
    assert np.all(a.time_bounds_s[1:, 0] >= a.time_bounds_s[:-1, 1] - 1e-12)
    assert np.all(dep[:, 0] < a.time_bounds_s[:, 0])
    np.testing.assert_allclose(a.values, b.values[:len(a.values)], atol=1e-5)
    shifted = local_spectrotemporal_modulation(tone(3, offset=20))
    np.testing.assert_allclose(a.values, shifted.values)
    np.testing.assert_allclose(a.times_s + 10, shifted.times_s)


def test_complete_parameter_hash_covers_axis_and_frontend_choices():
    x = tone(3)
    a = local_spectrotemporal_modulation(x)
    for kw in [dict(carriers_hz=(250, 500)), dict(rates_hz=(2, 8)),
               dict(scales_cpo=(.5, 2)), dict(fmin=60), dict(output_hop_s=.2),
               dict(spectral_sd_oct=.7)]:
        b = local_spectrotemporal_modulation(x, **kw)
        assert a.metadata['params_hash'] != b.metadata['params_hash']


def test_silence_and_explicit_stereo_mean():
    x = tone(3)
    a = local_spectrotemporal_modulation(x)
    stereo = AudioStimulus(np.column_stack([x.samples, x.samples]), x.sr_hz)
    np.testing.assert_allclose(a.values, local_spectrotemporal_modulation(stereo).values)
    z = AudioStimulus(np.zeros_like(x.samples), x.sr_hz)
    out = local_spectrotemporal_modulation(z)
    assert np.isfinite(out.values).all()
    assert abs(out.values).max() < 1e-8


@pytest.mark.parametrize('kw', [dict(frame_rate_hz=10), dict(scales_cpo=(100,)),
                                dict(carriers_hz=(1,)), dict(rates_hz=()),
                                dict(window_s=True), dict(spectral_sd_oct=-1)])
def test_invalid_kernel_parameters(kw):
    args = dict(frequencies_hz=2 ** np.linspace(6, 13, 64), frame_rate_hz=100)
    args.update(kw)
    with pytest.raises(ValueError):
        local_modulation_kernel(np.ones((400, 64)), **args)


@pytest.mark.parametrize('kw', [dict(n_log_bins=True), dict(n_log_bins=7),
                                dict(hop_s=0), dict(output_hop_s=.001),
                                dict(window_s=10), dict(fmax=9000)])
def test_invalid_waveform_parameters(kw):
    with pytest.raises(ValueError):
        local_spectrotemporal_modulation(tone(3), **kw)


def test_registry_public_entry_and_custom_axes():
    from natural_features.core.registry import Registry
    registry = Registry.with_builtin_specs()
    spec = registry.get('audio.modulation.local')
    assert spec.impl.endswith(':local_spectrotemporal_modulation')
    params = registry.validated_params('audio.modulation.local', {'carriers_hz': [250., 500.], 'rates_hz': [4.]})
    out = registry.impl('audio.modulation.local')(tone(3), **params)
    assert out.values.shape[1:3] == (2, 1)
