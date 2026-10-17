"""Limiting-case oracles for missing tonal and rhythmic evidence."""
import numpy as np
import pytest

from natural_features.core.stimulus import AudioStimulus
from natural_features.features.audio import music


def _column(series, name):
    return series.values[:, series.coords["feature"].index(name)]


def _stimulus():
    return AudioStimulus.from_array(np.zeros(10000, np.float32), 1000, start_offset_s=7, clock="recording")


def test_tonality_silence_is_missing_not_c_major():
    out = music.music_tonality(_stimulus())
    assert out["default"].values.shape[1] == 24
    assert all(n.startswith("key_r_") for n in out["default"].coords["feature"])
    assert np.isnan(out["default"].values).all()
    diag = out["diagnostics"]
    assert not _column(diag, "profile_valid").any()
    assert not _column(diag, "key_valid").any()
    assert np.isnan(_column(diag, "key_index")).all()
    assert np.isnan(_column(diag, "is_minor")).all()
    assert out["default"].metadata["output_pair_id"] == diag.metadata["output_pair_id"]


@pytest.mark.parametrize("row,profile_valid", [(np.ones(12) / 12, False), (np.tile([1., 0.], 6) / 6, True)])
def test_degenerate_or_tied_chroma_does_not_produce_a_key(monkeypatch, row, profile_valid):
    monkeypatch.setattr(music, "_chroma_matrix", lambda *a, **k: (np.tile(row, (1000, 1)), np.arange(1000) / 100))
    stim = AudioStimulus.from_array(np.tile([.2, -.2], 5500).astype(np.float32), 1000)
    out = music.music_tonality(stim)
    assert np.all(_column(out["diagnostics"], "profile_valid") == profile_valid)
    assert not _column(out["diagnostics"], "key_valid").any()
    assert np.isnan(_column(out["diagnostics"], "key_index")).all()
    assert np.isfinite(out["default"].values).all() == profile_valid


@pytest.mark.parametrize("count", [0, 1, 2, 3])
def test_rhythm_descriptor_support_depends_on_observed_intervals(monkeypatch, count):
    envelope = np.zeros(1000, np.float32)
    envelope[np.array([100, 150, 200])[:count]] = 1
    monkeypatch.setattr(music, "_onset_envelope", lambda *a, **k: (envelope, np.arange(1000) / 100))
    out = music.music_rhythm(_stimulus(), hop_s=8)
    values, diag = out["default"], out["diagnostics"]
    assert _column(diag, "n_onsets")[0] == count
    assert _column(diag, "n_intervals")[0] == max(0, count - 1)
    assert _column(values, "onset_rate")[0] == count / 8
    assert np.isfinite(_column(values, "tempo_bpm")[0]) == (count >= 3)
    assert np.isfinite(_column(values, "onset_peak_strength")[0]) == (count >= 1)
    assert np.isfinite(_column(values, "ioi_median")[0]) == (count >= 2)
    assert np.isfinite(_column(values, "ioi_cv")[0]) == (count >= 3)
    if count == 3:
        assert _column(values, "ioi_cv")[0] == 0  # measured regularity remains zero
        assert _column(values, "ioi_median")[0] == .5
        assert _column(values, "tempo_bpm")[0] == 120
    assert "log2_tempo" not in values.coords["feature"]
    assert "late_phase_onset_energy_fraction" not in values.coords["feature"]
    assert out["default"].clock == _stimulus().clock


def test_supported_boundary_tempo_is_not_automatically_rejected(monkeypatch):
    envelope = np.zeros(1000, np.float32)
    envelope[[100, 300, 500, 700]] = 1
    monkeypatch.setattr(music, "_onset_envelope", lambda *a, **k: (envelope, np.arange(1000) / 100))
    out = music.music_rhythm(_stimulus(), hop_s=8, prior_bpm=30)
    assert _column(out["default"], "tempo_bpm")[0] == 30
    assert _column(out["diagnostics"], "tempo_boundary")[0] == 1
    assert _column(out["diagnostics"], "tempo_valid")[0] == 1


def test_real_single_click_does_not_become_a_tempo():
    x = np.zeros(12 * 22050, np.float32)
    x[3 * 22050:3 * 22050 + 200] = 1
    out = music.music_rhythm(AudioStimulus.from_array(x, 22050))
    for name in ("tempo_bpm", "pulse_clarity", "ioi_median", "ioi_cv"):
        assert np.isnan(_column(out["default"], name)).all()
    assert np.isnan(_column(out["diagnostics"], "log2_tempo")).all()


def test_window_bounds_use_realized_hop_and_include_flux_dependency():
    sr = 8000
    x = np.zeros(sr * 4, np.float32)
    out = music.music_rhythm(AudioStimulus.from_array(x, sr, start_offset_s=9),
                             window_s=1, hop_s=.3333, env_hop_s=.0103)
    fs = out["default"]
    h = round(.0103 * sr)
    expected_hop = round(.3333 * sr / h) * h / sr
    np.testing.assert_allclose(np.diff(fs.times_s), expected_hop)
    deps = np.asarray(fs.metadata["computation_context"]["dependency_bounds_s"])
    assert np.all(deps[:, 0] <= fs.time_bounds_s[:, 0])
    assert np.all(deps[:, 1] <= 13)


@pytest.mark.parametrize("dc", [.2, -.2, .001])
def test_constant_waveform_has_no_tonal_evidence_and_dc_does_not_change_profiles(dc):
    sr = 8000
    constant = np.full(sr * 3, dc, np.float32)
    silent = music.music_tonality(AudioStimulus.from_array(constant, sr), window_s=1)
    assert np.isnan(silent["default"].values).all()
    assert not _column(silent["diagnostics"], "key_valid").any()
    t = np.arange(sr * 3) / sr
    tone = sum(.02 * np.sin(2 * np.pi * f * t) for f in [261.63, 329.63, 392])
    base = music.music_tonality(AudioStimulus.from_array(tone.astype(np.float32), sr), window_s=1)
    shifted = music.music_tonality(AudioStimulus.from_array((tone + dc).astype(np.float32), sr), window_s=1)
    np.testing.assert_allclose(base["default"].values, shifted["default"].values, atol=1e-5)


def test_loud_prefix_cannot_erase_quiet_local_activity():
    sr = 8000
    quiet = (2e-6 * np.sin(2 * np.pi * 261.63 * np.arange(3 * sr) / sr)).astype(np.float32)
    prefixed = np.concatenate([np.full(128 * sr, .7, np.float32), quiet])
    isolated = music.music_tonality(AudioStimulus.from_array(quiet, sr), window_s=1)
    full = music.music_tonality(AudioStimulus.from_array(prefixed, sr), window_s=1)
    assert _column(isolated["diagnostics"], "profile_valid").all()
    for key in isolated:
        np.testing.assert_array_equal(isolated[key].values, full[key].values[-len(isolated[key].values):])
