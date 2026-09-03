"""Behavioural tests for audio.periodicity.

Each asserts a property that has to hold for the feature to mean what its name says:
a clean tone must be more periodic than noise, a stretched partial series must be more
inharmonic than a true harmonic one, and a chord must show more spectral peaks than a
single note. Failing any of these is a bug regardless of what the numbers look like.
"""

from __future__ import annotations

import numpy as np
import pytest

from natural_features.core.stimulus import AudioStimulus
from natural_features.features.audio.periodicity import FEATURE_NAMES, audio_periodicity

SR = 22050


def _stim(x: np.ndarray) -> AudioStimulus:
    return AudioStimulus.from_array(x.astype(np.float32), sr_hz=SR)


def _harmonic(f0: float, dur_s: float = 2.0, n_harm: int = 8, stretch: float = 1.0,
              noise: float = 0.0, seed: int = 0) -> AudioStimulus:
    """Harmonic complex; ``stretch`` > 1 pushes partials sharp of k*f0 like a struck string."""

    t = np.arange(int(SR * dur_s)) / SR
    x = np.zeros_like(t)
    for k in range(1, n_harm + 1):
        fk = f0 * k * (k ** (stretch - 1.0))
        if fk < SR / 2:
            x += (1.0 / k) * np.sin(2 * np.pi * fk * t)
    x /= np.abs(x).max()
    if noise > 0:
        x = x + noise * np.random.default_rng(seed).standard_normal(len(t))
    return _stim(0.4 * x)


def _col(fs, name: str) -> np.ndarray:
    return fs.values[:, list(fs.coords["feature"]).index(name)]


def _voiced(fs) -> np.ndarray:
    return _col(fs, "f0_hz") > 0


# ------------------------------------------------------------------ f0 and confidence


@pytest.mark.parametrize("f0", [110.0, 220.0, 440.0])
def test_f0_is_recovered_for_a_harmonic_complex(f0: float) -> None:
    fs = audio_periodicity(_harmonic(f0))
    est = _col(fs, "f0_hz")
    est = est[est > 0]
    assert est.size > 50
    assert abs(float(np.median(est)) - f0) / f0 < 0.02


def test_noise_is_not_called_voiced() -> None:
    rng = np.random.default_rng(0)
    fs = audio_periodicity(_stim(0.3 * rng.standard_normal(SR * 2)))
    assert _voiced(fs).mean() < 0.2
    assert float(np.median(_col(fs, "f0_confidence"))) < 0.3


def test_confidence_is_higher_for_a_tone_than_for_noise() -> None:
    rng = np.random.default_rng(1)
    tone = audio_periodicity(_harmonic(220.0))
    noise = audio_periodicity(_stim(0.3 * rng.standard_normal(SR * 2)))
    assert np.median(_col(tone, "f0_confidence")) > np.median(_col(noise, "f0_confidence")) + 0.3


# ------------------------------------------------------------------ HNR


def test_hnr_falls_monotonically_as_noise_is_added() -> None:
    """The defining property of a harmonic-to-noise ratio."""

    hnrs = [
        float(np.median(_col(audio_periodicity(_harmonic(220.0, noise=n)), "hnr_db")))
        for n in (0.0, 0.05, 0.2, 0.8)
    ]
    assert all(a > b for a, b in zip(hnrs, hnrs[1:])), hnrs


def test_hnr_is_finite_everywhere_including_silence() -> None:
    fs = audio_periodicity(_stim(np.zeros(SR)))
    assert np.all(np.isfinite(fs.values))


# ------------------------------------------------------------------ harmonic fraction


def test_harmonic_fraction_is_higher_for_a_tone_than_for_noise() -> None:
    rng = np.random.default_rng(2)
    tone = audio_periodicity(_harmonic(220.0))
    noise = audio_periodicity(_stim(0.3 * rng.standard_normal(SR * 2)))
    tv = _col(tone, "harmonic_frac")[_voiced(tone)]
    assert float(np.median(tv)) > 0.5
    assert float(np.median(tv)) > float(np.median(_col(noise, "harmonic_frac"))) + 0.2


def test_harmonic_fraction_is_bounded() -> None:
    fs = audio_periodicity(_harmonic(220.0))
    hf = _col(fs, "harmonic_frac")
    assert hf.min() >= 0.0 and hf.max() <= 1.0 + 1e-5


# ------------------------------------------------------------------ inharmonicity


def test_stretched_partials_read_as_more_inharmonic() -> None:
    """A struck-string spectrum (partials sharp of k*f0) vs a true harmonic one."""

    true_h = audio_periodicity(_harmonic(220.0, stretch=1.0))
    stretched = audio_periodicity(_harmonic(220.0, stretch=1.04))
    a = float(np.median(_col(true_h, "inharmonicity")[_voiced(true_h)]))
    b = float(np.median(_col(stretched, "inharmonicity")[_voiced(stretched)]))
    assert b > a, (a, b)


def test_true_harmonic_series_is_near_zero_inharmonicity() -> None:
    fs = audio_periodicity(_harmonic(220.0))
    val = float(np.median(_col(fs, "inharmonicity")[_voiced(fs)]))
    assert val < 0.01, val


# ------------------------------------------------------------------ peak rate


def test_a_chord_has_more_spectral_peaks_than_a_single_note() -> None:
    t = np.arange(SR * 2) / SR
    one = np.zeros_like(t)
    for k in range(1, 6):
        one += (1.0 / k) * np.sin(2 * np.pi * 220.0 * k * t)
    many = one.copy()
    for f in (277.18, 329.63):
        for k in range(1, 6):
            many += (1.0 / k) * np.sin(2 * np.pi * f * k * t)
    r1 = float(np.median(_col(audio_periodicity(_stim(0.3 * one / np.abs(one).max())),
                              "spectral_peak_rate")))
    r3 = float(np.median(_col(audio_periodicity(_stim(0.3 * many / np.abs(many).max())),
                              "spectral_peak_rate")))
    assert r3 > r1, (r1, r3)


# ------------------------------------------------------------------ jitter


def test_a_steady_tone_has_near_zero_jitter() -> None:
    fs = audio_periodicity(_harmonic(220.0))
    assert float(np.median(_col(fs, "f0_jitter"))) < 0.01


def test_a_glide_has_more_jitter_than_a_steady_tone() -> None:
    t = np.arange(SR * 2) / SR
    f = 180.0 * (2.0 ** (t / 2.0))                      # one octave over 2 s
    phase = 2 * np.pi * np.cumsum(f) / SR
    glide = _stim(0.4 * sum((1.0 / k) * np.sin(k * phase) for k in range(1, 6)) / 2.0)
    a = float(np.mean(_col(audio_periodicity(_harmonic(220.0)), "f0_jitter")))
    b = float(np.mean(_col(audio_periodicity(glide), "f0_jitter")))
    assert b > a, (a, b)


def test_jitter_ignores_unvoiced_gaps() -> None:
    """A silent gap between two tones must not register as a huge f0 jump."""

    t = np.arange(SR) / SR
    tone = 0.4 * np.sin(2 * np.pi * 220.0 * t)
    x = np.concatenate([tone, np.zeros(SR // 2), 0.4 * np.sin(2 * np.pi * 660.0 * t)])
    fs = audio_periodicity(_stim(x))
    assert float(np.max(_col(fs, "f0_jitter"))) < 1.0


# ------------------------------------------------------------------ contract


def test_feature_series_contract() -> None:
    fs = audio_periodicity(_harmonic(220.0))
    assert fs.dims == ("time", "feature")
    assert list(fs.coords["feature"]) == list(FEATURE_NAMES)
    assert fs.values.shape == (fs.times_s.shape[0], len(FEATURE_NAMES))
    assert np.all(np.isfinite(fs.values))
    assert np.all(np.diff(fs.times_s) > 0)
    assert fs.metadata["extractor_name"] == "audio.periodicity"


def test_it_is_wider_than_audio_pitch() -> None:
    """The reason this extractor exists: audio.pitch is two columns."""

    from natural_features.features.audio.prosody import audio_pitch

    stim = _harmonic(220.0)
    assert audio_periodicity(stim).values.shape[1] > audio_pitch(stim).values.shape[1]


@pytest.mark.parametrize("bad", [dict(fmin=0.0), dict(fmin=500.0, fmax=100.0),
                                 dict(n_harmonics=0), dict(harmonic_tol=0.9)])
def test_invalid_parameters_are_rejected(bad: dict) -> None:
    with pytest.raises(ValueError):
        audio_periodicity(_harmonic(220.0, dur_s=0.5), **bad)
