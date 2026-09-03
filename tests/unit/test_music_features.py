"""Behavioural tests for the music and modulation extractors.

These assert *interpretable* properties -- a pure A440 tone must land on pitch class A,
a 120 BPM click train must be called 120 BPM, a ripple at a known (Omega, omega) must
put its energy in the matching cell. Numeric agreement with librosa is checked
separately in ``test_music_parity.py``, which is skipped when librosa is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from natural_features.core.stimulus import AudioStimulus
from natural_features.features.audio.modulation import (
    audio_modulation_spectrum,
    log_cochleagram,
    modulation_power_spectrum,
)
from natural_features.features.audio.music import (
    PITCH_CLASSES,
    music_chroma,
    music_onset_strength,
    music_rhythm,
    music_tempogram,
    music_tonality,
    music_tonnetz,
)

SR = 22050


def _tone(freqs, dur_s=4.0, sr=SR, amp=0.3):
    t = np.arange(int(sr * dur_s), dtype=np.float64) / sr
    x = np.zeros_like(t)
    for f in np.atleast_1d(freqs):
        x += amp * np.sin(2 * np.pi * float(f) * t)
    return AudioStimulus.from_array(x.astype(np.float32), sr_hz=sr)


def _click_train(bpm, dur_s=12.0, sr=SR):
    """Impulses at a fixed tempo, each a short decaying noise burst."""

    n = int(sr * dur_s)
    x = np.zeros(n, dtype=np.float64)
    period = 60.0 / bpm
    rng = np.random.default_rng(0)
    burst_n = int(0.01 * sr)
    env = np.exp(-np.linspace(0, 6, burst_n))
    for k in range(int(dur_s / period)):
        i = int(k * period * sr)
        if i + burst_n < n:
            x[i : i + burst_n] += rng.standard_normal(burst_n) * env
    return AudioStimulus.from_array(x.astype(np.float32), sr_hz=sr)


# ---------------------------------------------------------------- chroma / tonnetz


def test_chroma_pure_tone_lands_on_its_pitch_class() -> None:
    # A4 = 440 Hz is pitch class A, index 9.
    fs = music_chroma(_tone(440.0))
    mean = fs.values.mean(axis=0)
    assert int(np.argmax(mean)) == PITCH_CLASSES.index("A")


@pytest.mark.parametrize("freq,name", [(261.63, "C"), (329.63, "E"), (392.00, "G"), (466.16, "As")])
def test_chroma_tracks_pitch_class_across_the_octave(freq: float, name: str) -> None:
    fs = music_chroma(_tone(freq))
    assert int(np.argmax(fs.values.mean(axis=0))) == PITCH_CLASSES.index(name)


def test_chroma_is_octave_invariant() -> None:
    """C3, C4 and C5 must all peak on C -- that is the whole point of chroma."""

    peaks = {int(np.argmax(music_chroma(_tone(f)).values.mean(axis=0))) for f in (130.81, 261.63, 523.25)}
    assert peaks == {PITCH_CLASSES.index("C")}


def test_chroma_major_triad_has_three_peaks() -> None:
    fs = music_chroma(_tone([261.63, 329.63, 392.00]))  # C major
    mean = fs.values.mean(axis=0)
    top3 = set(np.argsort(mean)[-3:].tolist())
    assert top3 == {PITCH_CLASSES.index(n) for n in ("C", "E", "G")}


def test_chroma_rows_are_unit_norm_by_default() -> None:
    fs = music_chroma(_tone(440.0))
    norms = np.linalg.norm(fs.values, axis=1)
    assert np.allclose(norms[norms > 0], 1.0, atol=1e-5)


def test_tonnetz_puts_fifth_related_keys_closer_than_tritone_related() -> None:
    """C major is a fifth from G and a tritone from F#; tonal distance must agree."""

    c = music_tonnetz(_tone([261.63, 329.63, 392.00])).values.mean(axis=0)  # C E G
    g = music_tonnetz(_tone([392.00, 493.88, 587.33])).values.mean(axis=0)  # G B D
    fsharp = music_tonnetz(_tone([369.99, 466.16, 554.37])).values.mean(axis=0)  # F# A# C#
    assert np.linalg.norm(c - g) < np.linalg.norm(c - fsharp)


# ---------------------------------------------------------------- onset / rhythm


def test_onset_strength_is_positive_and_peaks_at_clicks() -> None:
    stim = _click_train(120.0, dur_s=6.0)
    fs = music_onset_strength(stim)
    env = fs.values[:, 0]
    assert np.all(env >= 0)
    # Peaks should recur near the 0.5 s beat period.
    thresh = env.mean() + 2 * env.std()
    peak_t = fs.times_s[env > thresh]
    assert peak_t.size >= 8
    iois = np.diff(peak_t)
    assert abs(float(np.median(iois)) - 0.5) < 0.05


@pytest.mark.parametrize("bpm", [90.0, 120.0, 150.0])
def test_rhythm_recovers_a_known_tempo(bpm: float) -> None:
    fs = music_rhythm(_click_train(bpm, dur_s=16.0), window_s=8.0, hop_s=4.0)
    tempo = fs.values[:, list(fs.coords["feature"]).index("tempo_bpm")]
    assert np.all(np.abs(tempo - bpm) < 0.05 * bpm)


def test_rhythm_pulse_clarity_is_higher_for_a_beat_than_for_noise() -> None:
    rng = np.random.default_rng(1)
    noise = AudioStimulus.from_array(
        (0.1 * rng.standard_normal(SR * 16)).astype(np.float32), sr_hz=SR
    )
    idx = None
    beat = music_rhythm(_click_train(120.0, dur_s=16.0), window_s=8.0, hop_s=4.0)
    idx = list(beat.coords["feature"]).index("pulse_clarity")
    noisy = music_rhythm(noise, window_s=8.0, hop_s=4.0)
    assert beat.values[:, idx].mean() > noisy.values[:, idx].mean()


def test_tempogram_peaks_at_the_true_tempo() -> None:
    fs = music_tempogram(_click_train(120.0, dur_s=16.0), window_s=8.0, hop_s=4.0, n_bins=64)
    names = list(fs.coords["feature"])
    bpms = np.array([float(n.split("_")[1].replace("bpm", "")) for n in names])
    peak = bpms[int(np.argmax(fs.values.mean(axis=0)))]
    # Autocorrelation also responds at integer multiples of the period; accept the
    # true tempo or an octave of it, but nothing else.
    assert min(abs(peak - 120.0), abs(peak - 60.0), abs(peak - 240.0)) < 8.0


# ---------------------------------------------------------------- tonality


def test_tonality_recovers_c_major_from_a_c_major_triad() -> None:
    fs = music_tonality(_tone([261.63, 329.63, 392.00]), window_s=2.0, hop_s=1.0)
    names = list(fs.coords["feature"])
    key = fs.values[:, names.index("key_index")]
    minor = fs.values[:, names.index("is_minor")]
    assert int(np.round(np.median(key))) == 0  # C
    assert float(np.median(minor)) == 0.0  # major


def test_tonality_emits_the_full_24_key_profile() -> None:
    fs = music_tonality(_tone([261.63, 329.63, 392.00]), window_s=2.0, hop_s=1.0)
    names = list(fs.coords["feature"])
    assert sum(n.startswith("key_r_") for n in names) == 24
    assert len(names) == 5 + 24


def test_tonality_profile_can_be_switched_off() -> None:
    fs = music_tonality(
        _tone(440.0), window_s=2.0, hop_s=1.0, include_profile=False
    )
    assert len(list(fs.coords["feature"])) == 5


# ---------------------------------------------------------------- modulation / MPS


def _ripple(omega_hz, Omega_cpo, *, n_t=512, n_f=128, frame_rate=100.0, oct_per_bin=0.05):
    t = np.arange(n_t)[:, None] / frame_rate
    x = np.arange(n_f)[None, :] * oct_per_bin
    return np.cos(2 * np.pi * (omega_hz * t + Omega_cpo * x))


def test_mps_ripple_energy_lands_in_the_matching_cell() -> None:
    """A ripple at 4 Hz and 1 cyc/oct must dominate the 2-8 Hz, 0.5-2 cyc/oct cell."""

    spec_edges = np.array([0.25, 0.5, 2.0, 8.0])
    temp_edges = np.array([0.5, 2.0, 8.0, 32.0])
    coch = _ripple(4.0, 1.0)
    vals, names = modulation_power_spectrum(
        coch, oct_per_bin=0.05, frame_rate_hz=100.0, spec_edges=spec_edges, temp_edges=temp_edges
    )
    winner = names[int(np.nanargmax(vals))]
    assert winner == "mps_pos_0.5-2cpo_2-8hz", winner


def test_mps_separates_the_two_ripple_directions() -> None:
    """Flipping the sign of Omega must move the energy to the mirrored cell."""

    spec_edges = np.array([0.25, 0.5, 2.0, 8.0])
    temp_edges = np.array([0.5, 2.0, 8.0, 32.0])
    kw = dict(oct_per_bin=0.05, frame_rate_hz=100.0, spec_edges=spec_edges, temp_edges=temp_edges)
    pos, names = modulation_power_spectrum(_ripple(4.0, 1.0), **kw)
    neg, _ = modulation_power_spectrum(_ripple(4.0, -1.0), **kw)
    assert names[int(np.nanargmax(pos))] == "mps_pos_0.5-2cpo_2-8hz"
    assert names[int(np.nanargmax(neg))] == "mps_neg_0.5-2cpo_2-8hz"


@pytest.mark.parametrize("omega,expect", [(1.0, "0.5-2hz"), (4.0, "2-8hz"), (16.0, "8-32hz")])
def test_mps_temporal_axis_is_calibrated_in_hz(omega: float, expect: str) -> None:
    spec_edges = np.array([0.25, 0.5, 2.0, 8.0])
    temp_edges = np.array([0.5, 2.0, 8.0, 32.0])
    vals, names = modulation_power_spectrum(
        _ripple(omega, 1.0),
        oct_per_bin=0.05,
        frame_rate_hz=100.0,
        spec_edges=spec_edges,
        temp_edges=temp_edges,
    )
    assert names[int(np.nanargmax(vals))].endswith(expect)


@pytest.mark.parametrize("Omega,expect", [(0.4, "0.25-0.5cpo"), (1.0, "0.5-2cpo"), (4.0, "2-8cpo")])
def test_mps_spectral_axis_is_calibrated_in_cycles_per_octave(Omega: float, expect: str) -> None:
    spec_edges = np.array([0.25, 0.5, 2.0, 8.0])
    temp_edges = np.array([0.5, 2.0, 8.0, 32.0])
    vals, names = modulation_power_spectrum(
        _ripple(4.0, Omega),
        oct_per_bin=0.05,
        frame_rate_hz=100.0,
        spec_edges=spec_edges,
        temp_edges=temp_edges,
    )
    assert expect in names[int(np.nanargmax(vals))]


def test_log_cochleagram_axis_is_uniform_in_octaves() -> None:
    coch, oct_per_bin = log_cochleagram(_tone(440.0, dur_s=2.0), n_log_bins=48)
    assert coch.shape[1] == 48
    assert oct_per_bin > 0
    # 50 Hz to Nyquist is ~7.8 octaves at SR=22050; spread over 48 bins.
    assert 0.1 < oct_per_bin * 47 / 7.8 < 1.5


def test_modulation_spectrum_extractor_shape_and_times() -> None:
    stim = _tone([440.0, 554.37], dur_s=8.0)
    fs = audio_modulation_spectrum(stim, window_s=2.0, hop_s=1.0)
    assert fs.values.ndim == 2
    assert fs.values.shape[0] == fs.times_s.shape[0]
    assert fs.values.shape[1] == len(list(fs.coords["feature"]))
    assert np.all(np.diff(fs.times_s) > 0)
    # First window is centred half a window in.
    assert abs(float(fs.times_s[0]) - 1.0) < 1e-6


# ---------------------------------------------------------------- contracts


@pytest.mark.parametrize(
    "fn,kw",
    [
        (music_chroma, {}),
        (music_tonnetz, {}),
        (music_onset_strength, {}),
        (music_rhythm, dict(window_s=2.0, hop_s=1.0)),
        (music_tempogram, dict(window_s=2.0, hop_s=1.0)),
        (music_tonality, dict(window_s=2.0, hop_s=1.0)),
    ],
)
def test_feature_series_contract(fn, kw) -> None:
    fs = fn(_tone([261.63, 329.63], dur_s=6.0), **kw)
    assert fs.dims == ("time", "feature")
    assert fs.values.shape == (fs.times_s.shape[0], len(list(fs.coords["feature"])))
    assert np.all(np.isfinite(fs.values))
    assert np.all(np.diff(fs.times_s) > 0)
    assert fs.metadata["extractor_name"].startswith("audio.")


@pytest.mark.parametrize(
    "fn", [music_rhythm, music_tempogram, music_tonality, audio_modulation_spectrum]
)
def test_window_features_reject_signals_shorter_than_one_window(fn) -> None:
    with pytest.raises(ValueError):
        fn(_tone(440.0, dur_s=0.5), window_s=8.0, hop_s=1.0)


def test_start_offset_is_carried_into_times() -> None:
    x = _tone(440.0, dur_s=6.0)
    shifted = AudioStimulus.from_array(x.samples, sr_hz=x.sr_hz, start_offset_s=10.0)
    base = music_chroma(x).times_s
    off = music_chroma(shifted).times_s
    assert np.allclose(off - base, 10.0)
