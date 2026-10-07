from __future__ import annotations

import numpy as np

from natural_features.core.stimulus import AudioStimulus
from natural_features.features.audio.music_structure import (
    _chord_profile_scores,
    _cosine_novelty,
    _delay_stack_similarity,
    music_chord_profiles, music_sequence_recurrence, music_tonal_novelty,
)


def _tone(freqs, *, seconds=3.0, sr=4000, amp=0.2):
    t = np.arange(round(seconds * sr)) / sr
    signal = sum(amp * np.sin(2 * np.pi * f * t) for f in np.atleast_1d(freqs))
    return AudioStimulus.from_array(signal.astype(np.float32), sr)


def test_chord_profiles_expose_raw_scores_and_silence_missingness():
    out = music_chord_profiles(_tone([261.63, 329.63, 392.0]))
    names = out["default"].coords["feature"]
    assert names[np.argmax(out["default"].values.mean(axis=0))] == "C_major"
    assert np.all(out["diagnostics"].values[:, 0] == 1)
    silence = music_chord_profiles(AudioStimulus.from_array(np.zeros(2000, np.float32), 4000))
    assert not silence["diagnostics"].values[:, 0].any()
    assert not silence["default"].values.any()


def test_chord_profile_oracles_and_gain_invariance_away_from_threshold():
    triad = music_chord_profiles(_tone([261.63, 329.63, 392.0]))["default"].values.mean(axis=0)
    # A unit-normalized C-major chroma has own score 1 and its two-note runners 2/3.
    assert np.isclose(triad[0], 1.0, atol=3e-2)
    # Smooth FFT-bin chroma broadens the waveform fixture slightly around the
    # binary-template ideal of 2/3, while retaining the expected runner order.
    assert 0.65 < np.sort(triad)[-2] < 0.75
    signal = _tone(261.63).samples
    base = music_chord_profiles(AudioStimulus.from_array(signal, 4000))["default"].values
    loud = music_chord_profiles(AudioStimulus.from_array(4 * signal, 4000))["default"].values
    np.testing.assert_allclose(base, loud, atol=2e-5)


def test_isolated_descriptor_oracles_use_exact_synthetic_chroma():
    c_major = np.zeros(12)
    c_major[[0, 4, 7]] = 1 / np.sqrt(3)
    scores, names = _chord_profile_scores(c_major[None, :])
    own = names.index("C_major")
    assert np.isclose(scores[0, own], 1.0)
    normalized = scores[0] / scores[0].sum()
    entropy = -(normalized[normalized > 0] * np.log(normalized[normalized > 0])).sum() / np.log(24)
    assert np.isclose(entropy, 0.7791625887365975)
    assert np.isclose(scores[0, own] - np.sort(scores[0])[-2], 1 / 3)
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    mixed = (a + b) / np.sqrt(2)
    assert _cosine_novelty(a[None], b[None]) == 1.0
    assert np.isclose(_cosine_novelty(a[None], mixed[None]), 1 - 1 / np.sqrt(2))
    assert np.isclose(_delay_stack_similarity(np.stack((a, b)), np.stack((a, b))), 1.0)
    assert np.isclose(_delay_stack_similarity(np.stack((a, b)), np.stack((b, a))), 0.0)
    assert np.isclose(_delay_stack_similarity(np.stack((a, b)), np.stack((a, a))), 0.5)


def test_realized_hop_and_causal_novelty_context_do_not_look_forward():
    stim = _tone(261.63, seconds=2.0, sr=11)
    out = music_tonal_novelty(stim, hop_s=0.14, win_s=0.31, window_s=0.4, mode="past_only")
    series = out["default"]
    assert np.all(series.metadata["computation_context"]["dependency_bounds_s"][-1][1] <= series.times_s[-1])
    assert np.isclose(series.timebase.hop_s, 2 / 11)


def test_causal_novelty_is_prefix_invariant_and_clock_tagged():
    full = _tone([261.63, 329.63], seconds=4.0, sr=4000)
    prefix = AudioStimulus.from_array(full.samples[: 2 * 4000], 4000, start_offset_s=7, clock="recording")
    shifted_full = AudioStimulus.from_array(full.samples, 4000, start_offset_s=7, clock="recording")
    kwargs = dict(hop_s=0.13, win_s=0.23, window_s=0.39, mode="past_only")
    left = music_tonal_novelty(prefix, **kwargs)["default"]
    right = music_tonal_novelty(shifted_full, **kwargs)["default"]
    # Compare only finalized prefix rows: tail placeholders differ because the
    # prefix cannot provide a complete prospective right window.
    finalized = (left.times_s < 9.0) & (left.metadata["computation_context"]["dependency_bounds_s"][-1][1] >= 0)
    np.testing.assert_allclose(left.values[finalized], right.values[: len(left.values)][finalized])
    assert left.metadata["computation_context"]["clock"] == "recording"


def test_centered_split_grid_is_monotonic_with_warmup_and_tail_placeholders():
    out = music_tonal_novelty(_tone(261.63, seconds=1.0, sr=100), hop_s=0.1, win_s=0.2,
                              window_s=0.2, mode="centered")
    series, diag = out["default"], out["diagnostics"].values
    assert np.all(np.diff(series.times_s) > 0)
    assert not diag[:2, 0].any()
    assert not diag[-2:, 0].any()
    assert np.allclose(series.time_bounds_s[:, 0], series.times_s)
    assert np.allclose(series.time_bounds_s[:, 1], series.times_s)


def test_causal_diagnostics_are_unchanged_before_a_mixed_future_append():
    sr = 1000
    tone = _tone(261.63, seconds=2.0, sr=sr).samples
    prefix = AudioStimulus.from_array(tone, sr)
    full = AudioStimulus.from_array(np.concatenate((tone, np.zeros(sr, np.float32))), sr)
    kwargs = dict(hop_s=0.1, win_s=0.2, window_s=0.2, mode="past_only")
    a, b = music_tonal_novelty(prefix, **kwargs), music_tonal_novelty(full, **kwargs)
    np.testing.assert_allclose(a["default"].values, b["default"].values[: len(a["default"].values)])
    np.testing.assert_allclose(a["diagnostics"].values, b["diagnostics"].values[: len(a["diagnostics"].values)])
    np.testing.assert_allclose(a["default"].time_bounds_s, b["default"].time_bounds_s[: len(a["default"].values)])


def test_recurrence_reports_bounded_candidates_and_missing_warmup():
    stim = _tone([261.63, 329.63], seconds=5.0)
    out = music_sequence_recurrence(stim, hop_s=0.1, win_s=0.2, embedding_frames=2,
                                    delay_frames=1, history_s=1.0)
    diag = out["diagnostics"].values
    assert not diag[0, 0]
    assert np.all(np.isfinite(out["default"].values))
    assert np.all(diag[:, 1] >= 0)


def test_recurrence_history_and_exclusion_boundaries():
    stim = _tone(261.63, seconds=3.0)
    permissive = music_sequence_recurrence(stim, hop_s=0.1, win_s=0.2, embedding_frames=2,
                                           history_s=1.0, exclusion_s=0.0)["diagnostics"].values
    excluded = music_sequence_recurrence(stim, hop_s=0.1, win_s=0.2, embedding_frames=2,
                                         history_s=1.0, exclusion_s=2.0)["diagnostics"].values
    assert permissive[:, 1].max() > 0
    assert excluded[:, 1].max() == 0


def test_recurrence_context_contains_query_when_history_is_short_and_is_prefix_stable():
    sr = 1000
    tone = _tone(261.63, seconds=2.0, sr=sr).samples
    prefix = AudioStimulus.from_array(tone, sr)
    full = AudioStimulus.from_array(np.concatenate((tone, np.zeros(sr, np.float32))), sr)
    kwargs = dict(hop_s=0.1, win_s=0.2, embedding_frames=3, delay_frames=1, history_s=0.01)
    a, b = music_sequence_recurrence(prefix, **kwargs), music_sequence_recurrence(full, **kwargs)
    bounds = a["default"].time_bounds_s
    context = np.asarray(a["default"].metadata["computation_context"]["dependency_bounds_s"])
    usable = np.flatnonzero(bounds[:, 1] > bounds[:, 0])
    assert np.all(context[usable, 0] <= bounds[usable, 0])
    keep = a["default"].times_s < 2.0
    np.testing.assert_allclose(a["diagnostics"].values[keep], b["diagnostics"].values[: len(keep)][keep])


def test_recurrence_exact_boundaries_are_clock_offset_invariant():
    signal = _tone(261.63, seconds=3.0, sr=1000).samples
    kwargs = dict(hop_s=0.1, win_s=0.2, embedding_frames=2, delay_frames=1, history_s=1.0)
    zero = music_sequence_recurrence(AudioStimulus.from_array(signal, 1000), **kwargs)
    shifted = music_sequence_recurrence(AudioStimulus.from_array(signal, 1000, start_offset_s=7), **kwargs)
    np.testing.assert_allclose(zero["default"].values, shifted["default"].values)
    np.testing.assert_array_equal(zero["diagnostics"].values, shifted["diagnostics"].values)
    # Once the 1 s history is full, inclusive exact-boundary geometry yields 5 candidates.
    counts = zero["diagnostics"].values[:, 1]
    assert np.all(counts[-5:] == 5)
    # Independent scalar exhaustive reference for m=2, d=1 on the integer
    # sample geometry; do not reuse the extractor's candidate bounds.
    expected = []
    n_frames, h, w, history = len(counts), 100, 200, 1000
    for i in range(n_frames):
        if i < 1:
            expected.append(0)
            continue
        query_start, query_end = (i - 1) * h, i * h + w
        lower = max(0, query_end - history)
        usable = 0
        for j in range(1, n_frames):
            if (j - 1) * h >= lower and j * h + w <= query_start:
                usable += 1
        expected.append(usable)
    np.testing.assert_array_equal(counts, expected)
