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


def test_rms_obeys_positive_amplitude_scaling() -> None:
    audio = _audio()
    scale = 3.5
    scaled = AudioStimulus.from_array(audio.samples * scale, sr_hz=audio.sr_hz)

    baseline_rms = rms(audio).values
    scaled_rms = rms(scaled).values

    np.testing.assert_allclose(
        scaled_rms,
        scale * baseline_rms,
        rtol=2e-6,
        atol=1e-7,
        err_msg="RMS must be homogeneous of degree one under positive amplitude scaling",
    )


def test_spectral_shape_statistics_are_positive_scale_invariant() -> None:
    rng = np.random.default_rng(20260715)
    samples = rng.normal(0.0, 0.05, size=32000).astype(np.float32)
    audio = AudioStimulus.from_array(samples, sr_hz=16000)
    scaled = AudioStimulus.from_array(samples * 4.0, sr_hz=16000)

    baseline = spectral_stats(audio).values
    observed = spectral_stats(scaled).values

    np.testing.assert_array_equal(
        observed[:, [0, 1, 2, 4]],
        baseline[:, [0, 1, 2, 4]],
        err_msg="Centroid, bandwidth, rolloff, and ZCR must ignore positive gain",
    )
    np.testing.assert_allclose(
        observed[:, 3],
        baseline[:, 3],
        rtol=5e-7,
        atol=1.5e-7,
        err_msg="Spectral flatness must ignore positive gain up to float32 rounding",
    )


def test_mfcc_gain_affects_only_the_orthonormal_dc_coefficient() -> None:
    rng = np.random.default_rng(20260715)
    samples = rng.normal(0.0, 0.05, size=16000).astype(np.float32)
    audio = AudioStimulus.from_array(samples, sr_hz=16000)
    scale = 2.5
    scaled = AudioStimulus.from_array(samples * scale, sr_hz=16000)
    n_mels = 24

    baseline = mfcc(
        audio,
        n_mfcc=8,
        n_mels=n_mels,
        include_deltas=False,
    ).values
    observed = mfcc(
        scaled,
        n_mfcc=8,
        n_mels=n_mels,
        include_deltas=False,
    ).values
    delta = observed - baseline
    expected_dc_shift = 2.0 * np.log10(scale) * np.sqrt(n_mels)

    np.testing.assert_allclose(
        delta[:, 0],
        expected_dc_shift,
        rtol=2e-5,
        atol=2e-5,
        err_msg="Log-power gain must map to the MFCC DC coefficient analytically",
    )
    np.testing.assert_allclose(
        delta[:, 1:],
        0.0,
        rtol=0.0,
        atol=3e-5,
        err_msg="Orthonormal non-DC MFCC coefficients must be invariant to constant log gain",
    )
