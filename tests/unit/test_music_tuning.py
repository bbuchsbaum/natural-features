"""Known frequency offsets isolate tuning from any change of chroma frontend."""
import numpy as np
import pytest

from natural_features.core.stimulus import AudioStimulus
from natural_features.features.audio.music import music_chroma, music_tonality, music_tonnetz
from natural_features.features.audio.music_structure import music_chord_profiles, music_sequence_recurrence, music_tonal_novelty
from natural_features.features.audio.music_tuning import estimate_music_tuning


def _triad(cents=0, root=0, minor=False, seconds=3, noise=0):
    sr = 8000
    t = np.arange(sr * seconds) / sr
    midi = 60 + root + np.array([0, 3 if minor else 4, 7])
    x = sum(.15 / harmonic * np.sin(2 * np.pi * 440 * 2 ** ((m - 69) / 12 + cents / 1200) * harmonic * t)
            for m in midi for harmonic in (1, 2, 3))
    if noise:
        rng = np.random.default_rng(1)
        for i in range(2000, len(x) - 100, 4000):
            x[i:i + 100] += noise * rng.standard_normal(100) * np.exp(-np.linspace(0, 6, 100))
    return AudioStimulus.from_array(x.astype(np.float32), sr, start_offset_s=4, clock="recording")


@pytest.mark.parametrize("cents", [-40, -30, 0, 30, 40])
def test_known_tuning_estimate_and_explicit_reference_agree(cents):
    stim = _triad(cents, noise=.3)
    evidence = estimate_music_tuning(stim)
    assert evidence["supported"]
    # Harmonic3 is naturally ~2c sharp; tolerate its weighted contribution.
    assert abs(evidence["cents_from_a440"] - cents) < 1
    automatic = music_chroma(stim, tuning_mode="auto")
    manual = music_chroma(stim, tuning_hz=440 * 2 ** (cents / 1200))
    np.testing.assert_allclose(automatic.values, manual.values, atol=.01)
    assert automatic.metadata["computation_context"]["policy"] == "bidirectional"
    assert automatic.metadata["computation_context"]["dependency_bounds_s"] == [[4, 7]]
    assert automatic.timebase.reference == stim.clock


@pytest.mark.parametrize("root", range(12))
@pytest.mark.parametrize("minor", [False, True])
def test_auto_tuned_chord_profiles_recover_detuned_triad(root, minor):
    out = music_chord_profiles(_triad(40, root, minor), tuning_mode="auto")["default"]
    assert np.argmax(out.values.mean(axis=0)) == root * 2 + minor
    assert out.metadata["computation_context"]["policy"] == "bidirectional"


def test_unsupported_tuning_is_explicit_for_silence_noise_short_and_mixed_offsets():
    rng = np.random.default_rng(3)
    low = _triad(-25)
    high = _triad(25)
    fixtures = [np.zeros(24000), rng.standard_normal(24000), np.zeros(100), low.samples + high.samples]
    for x in fixtures:
        stim = AudioStimulus.from_array(np.asarray(x, np.float32), 8000)
        evidence = estimate_music_tuning(stim)
        assert not evidence["supported"]
        assert evidence["tuning_hz"] is None
        with pytest.raises(ValueError, match="unsupported or ambiguous"):
            music_chroma(stim, tuning_mode="auto")


@pytest.mark.parametrize("fn,kw", [(music_tonal_novelty, {"mode": "past_only"}), (music_sequence_recurrence, {})])
def test_past_only_rejects_auto_and_fixed_outputs_remain_prefix_invariant(fn, kw):
    prefix = _triad(-30)
    full = AudioStimulus.from_array(np.concatenate([prefix.samples, _triad(30).samples]), 8000,
                                    start_offset_s=4, clock="recording")
    with pytest.raises(ValueError, match="incompatible with past_only"):
        fn(full, tuning_mode="auto", **kw)
    options = {**kw, "tuning_hz": 440 * 2 ** (-30 / 1200)}
    left, right = fn(prefix, **options), fn(full, **options)
    for key in left:
        a, b = left[key], right[key]
        np.testing.assert_array_equal(a.values, b.values[:len(a.values)])
        np.testing.assert_array_equal(a.time_bounds_s, b.time_bounds_s[:len(a.values)])
        assert a.metadata["computation_context"]["policy"] == "past_only"


@pytest.mark.parametrize("fn,kw", [(music_tonnetz, {}), (music_tonality, {"window_s": 1}), (music_tonal_novelty, {})])
def test_tuning_propagates_to_all_derived_outputs(fn, kw):
    out = fn(_triad(30), tuning_mode="auto", **kw)
    outputs = out.values() if isinstance(out, dict) else [out]
    for fs in outputs:
        assert abs(fs.metadata["tuning"]["cents_from_a440"] - 30) < 1
        assert fs.metadata["computation_context"]["dependency_bounds_s"] == [[4, 7]]


def test_fixed_reference_validation_and_conflicting_auto_controls():
    for hz in (0, -440, np.inf, np.nan, True):
        with pytest.raises(ValueError, match="tuning_hz"):
            music_chroma(_triad(), tuning_hz=hz)
    with pytest.raises(ValueError, match="overrides"):
        music_chroma(_triad(), tuning_hz=442, tuning_mode="auto")


@pytest.mark.parametrize("cents", [-49, 49, 49.9, 50])
def test_branch_cut_requires_explicit_reference_even_with_concentrated_peaks(cents):
    stim = _triad(cents)
    evidence = estimate_music_tuning(stim)
    assert evidence["near_semitone_boundary"]
    assert evidence["concentration"] > .9
    assert not evidence["supported"]
    with pytest.raises(ValueError, match="unsupported or ambiguous"):
        music_chord_profiles(stim, tuning_mode="auto")
    manual = music_chord_profiles(stim, tuning_hz=440 * 2 ** (cents / 1200))["default"]
    assert manual.coords["feature"][np.argmax(manual.values.mean(axis=0))] == "C_major"
