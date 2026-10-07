"""Native music outputs through catalogue, recipes, storage and alignment."""
from __future__ import annotations

import numpy as np
import pytest

from natural_features.core.feature_bundle import FeatureBundle
from natural_features.core.registry import Registry
from natural_features.core.stimulus import AudioStimulus
from natural_features.core.timebase import ClockMap, TemporalContext
from natural_features.core.timeline import Timeline, align_feature_to_timeline
from natural_features.storage.readers import read_feature_series
from natural_features.storage.writers import write_feature_series
from natural_features.workflows.extract_features import available_features, extract_features, plan_features

IDS = ["audio.music.chord_profiles", "audio.music.tonal_novelty", "audio.music.sequence_recurrence"]


@pytest.mark.smoke
def test_music_recipe_keeps_native_outputs_and_validity_pairs(tmp_path):
    sr = 4000
    t = np.arange(sr * 3) / sr
    waveform = (0.2 * (np.sin(2 * np.pi * 261.63 * t) + np.sin(2 * np.pi * 329.63 * t))).astype(np.float32)
    waveform[sr:sr + sr // 3] = 0
    audio = AudioStimulus.from_array(
        waveform, sr, start_offset_s=7, clock="recording",
        temporal_context=TemporalContext((ClockMap("recording", "scan", offset_s=-5),)),
    )
    params = {
        IDS[0]: {"hop_s": 0.07},
        IDS[1]: {"hop_s": 0.1, "window_s": 0.3, "mode": "past_only"},
        IDS[2]: {"hop_s": 0.15, "embedding_frames": 2, "history_s": 1.0},
    }
    result = extract_features(audio, features=IDS, feature_params=params)
    bundle = FeatureBundle(result.features)
    target = Timeline("scan windows", np.arange(2, 5, 0.25), np.arange(2.25, 5.25, 0.25),
                      reference="scan", temporal_context=audio.temporal_context)
    lengths = set()
    for feature_id in IDS:
        main, diagnostic = result.features[feature_id], result.features[feature_id + ".diagnostics"]
        lengths.add(len(main.times_s))
        assert main.clock == diagnostic.clock == "recording"
        assert main.metadata["output_pair_id"] == diagnostic.metadata["output_pair_id"]
        assert main.metadata["validity_output"] == "diagnostics"
        assert diagnostic.coords["feature"][0] == main.metadata["validity_column"] == "valid"
        assert np.any(diagnostic.values[:, 0] == 0)
        np.testing.assert_array_equal(main.times_s, diagnostic.times_s)
        main_saved = read_feature_series(write_feature_series(main, tmp_path / feature_id, fmt="npz"))
        diag_saved = read_feature_series(write_feature_series(diagnostic, tmp_path / (feature_id + ".diagnostics"), fmt="npz"))
        assert main_saved.metadata["output_pair_id"] == diag_saved.metadata["output_pair_id"]
        np.testing.assert_array_equal(main_saved.values, main.values)
        np.testing.assert_array_equal(diag_saved.values, diagnostic.values)
        a = align_feature_to_timeline(feature_id, main_saved, target)
        b = align_feature_to_timeline(feature_id + ".diagnostics", diag_saved, target)
        for key in a.mapping:
            np.testing.assert_array_equal(a.mapping[key], b.mapping[key])
        payload = bundle.temporal_payload(feature_id)
        assert payload.metadata["output_pair_id"] == main.metadata["output_pair_id"]
        converted = bundle.in_clock(feature_id, "scan")
        assert converted.metadata["computation_context"]["clock"] == "recording"
        np.testing.assert_allclose(converted.times_s, main.times_s - 5)
    assert len(lengths) == 3


def test_catalogue_exposes_music_backends_with_explicit_opt_in():
    registry = Registry.with_builtin_specs()
    entries = available_features(tags="music", public_only=False, budget="all")
    names = {entry.feature_id for entry in entries}
    assert {*IDS, "audio.music.mert", "audio.music.beat_activations"} <= names
    for name in ("audio.music.mert", "audio.music.beat_activations"):
        assert callable(registry.impl(name))
        with pytest.raises(PermissionError, match="opt-in"):
            plan_features("audio", features=[name])
        plan = plan_features("audio", features=[name], budget="allow_python")
        assert plan.rows[0].params["local_files_only"] is True


def _inject_chroma(monkeypatch, chroma, active=None):
    from natural_features.features.audio import music_structure as module

    chroma = np.asarray(chroma, dtype=np.float64)
    chroma /= np.linalg.norm(chroma, axis=1, keepdims=True)
    n = len(chroma)
    active = np.ones(n, dtype=bool) if active is None else np.asarray(active, dtype=bool)
    bounds = np.column_stack([np.arange(n), np.arange(1, n + 1)]) / 10
    monkeypatch.setattr(module, "_frame_chroma", lambda *a, **k: (
        chroma, active, bounds.mean(axis=1), bounds, 10, 10,
    ))
    audio = AudioStimulus.from_array(np.zeros(n * 10, dtype=np.float32), 100)
    return module, audio


def test_production_chord_diagnostics_have_analytic_entropy_and_margin(monkeypatch):
    triad = np.zeros(12)
    triad[[0, 4, 7]] = 1
    singleton = np.eye(12)[0]
    module, audio = _inject_chroma(monkeypatch, [triad, np.ones(12), singleton])
    result = module.music_chord_profiles(audio)
    scores = result["default"].values
    diagnostic = result["diagnostics"]
    index = {name: i for i, name in enumerate(diagnostic.coords["feature"])}
    np.testing.assert_allclose(scores[0, 0], 1, atol=1e-7)
    np.testing.assert_allclose(diagnostic.values[0, index["top_two_margin"]], 1 / 3, atol=1e-7)
    np.testing.assert_allclose(scores[1], 0.5, atol=1e-7)
    np.testing.assert_allclose(diagnostic.values[1, index["score_entropy"]], 1, atol=1e-7)
    assert diagnostic.values[1, index["top_two_margin"]] == 0
    np.testing.assert_allclose(diagnostic.values[2, index["score_entropy"]],
                               np.log(6) / np.log(24), atol=1e-7)


@pytest.mark.parametrize("window_frames", [2, 5, 8])
@pytest.mark.parametrize("mode", ["centered", "past_only"])
def test_production_tonal_step_has_expected_scale_and_detection_delay(monkeypatch, window_frames, mode):
    # Orthogonal pitch classes switch at exactly 3s on contiguous 0.1s cells.
    # The adjacent pure blocks have distance 1; homogeneous blocks have distance 0.
    pitches = [0] * 30 + [6] * 30
    module, audio = _inject_chroma(monkeypatch, np.eye(12)[pitches])
    result = module.music_tonal_novelty(audio, window_s=window_frames / 10, mode=mode)
    series, diagnostic = result["default"], result["diagnostics"].values
    peak = int(np.argmax(series.values[:, 0]))
    np.testing.assert_allclose(series.values[peak, 0], 1, atol=1e-7)
    assert diagnostic[peak, 0] == 1
    expected_time = 3 if mode == "centered" else 3 + window_frames / 10
    np.testing.assert_allclose(series.times_s[peak], expected_time, atol=1e-12)
    context = np.asarray(series.metadata["computation_context"]["dependency_bounds_s"])
    homogeneous = (diagnostic[:, 0] == 1) & ((context[:, 1] <= 3) | (context[:, 0] >= 3))
    assert homogeneous.any()
    np.testing.assert_allclose(series.values[homogeneous, 0], 0, atol=1e-7)
    assert np.count_nonzero(np.isclose(series.values[:, 0], 1)) == 1


@pytest.mark.parametrize("window_s", [0.2, 0.5, 0.8])
def test_waveform_tonal_transition_is_localized_at_multiple_scales(window_s):
    from natural_features.features.audio.music_structure import music_tonal_novelty

    sr, seconds = 4000, 3
    t = np.arange(sr * seconds) / sr
    # C followed by F-sharp: a large pitch-class change, not an octave change.
    # Nonoverlapping analysis cells align with the step so no cell mixes the tones.
    waveform = np.concatenate([0.2 * np.sin(2 * np.pi * f * t) for f in (261.63, 369.99)])
    audio = AudioStimulus.from_array(waveform.astype(np.float32), sr)
    result = music_tonal_novelty(audio, hop_s=0.1, win_s=0.1, window_s=window_s)
    series, diagnostic = result["default"], result["diagnostics"].values
    peak = int(np.argmax(series.values[:, 0]))
    assert diagnostic[peak, 0] == 1
    assert abs(series.times_s[peak] - seconds) <= 0.05
    assert series.values[peak, 0] > 0.9
    # Steady tones well outside the transition have negligible novelty.
    steady = (diagnostic[:, 0] == 1) & (np.abs(series.times_s - seconds) > window_s + 0.1)
    assert steady.any()
    assert np.max(series.values[steady, 0]) < 0.01


@pytest.mark.parametrize("inactive,changed_query,expected_count,expected_lag,expected_score", [
    (False, False, 7, 0.3, 1.0),
    (True, False, 5, 0.9, 1.0),
    (False, True, 7, 0.3, 0.5),
])
def test_production_recurrence_selects_correct_delayed_match(
    monkeypatch, inactive, changed_query, expected_count, expected_lag, expected_score,
):
    # Independent categorical oracle: two matched stack positions score 1,
    # one matched position scores 1/2. Historical rows 2 and 8 tie initially.
    pitches = [0, 1, 2, 1, 0, 2, 0, 1, 2, 0, 1, 2]
    if changed_query:
        pitches[9] = 3
    active = np.ones(12, dtype=bool)
    if inactive:
        active[6] = False
    module, audio = _inject_chroma(monkeypatch, np.eye(12)[pitches], active)
    result = module.music_sequence_recurrence(
        audio, embedding_frames=2, delay_frames=2, history_s=2,
    )
    np.testing.assert_allclose(result["default"].values[-1], [expected_score, expected_lag], atol=1e-7)
    assert result["diagnostics"].values[-1, 0] == 1
    assert result["diagnostics"].values[-1, 1] == expected_count
