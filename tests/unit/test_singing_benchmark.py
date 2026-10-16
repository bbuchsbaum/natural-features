from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from natural_features.core.feature_types import FeatureSeries
from natural_features.core.timebase import TimebaseSpec
from natural_features.features.audio._music_contract import asset_digest
from natural_features.features.speech.singing_benchmark import _levenshtein, evaluate_singing_phones, native_time_cells


def _post(values, times=(0.05, 0.15), bounds=None):
    labels = ["<b>", "|", "<x>", "AA", "TS", "ZZ"]
    return FeatureSeries(values=np.asarray(values, float), times_s=np.asarray(times, float), coords={"feature": labels}, time_bounds_s=None if bounds is None else np.asarray(bounds, float), metadata={"extractor_id": "toy", "params_hash": "toy"})


def _score(post, refs, windows=(0.0, 0.2), bins=None):
    return evaluate_singing_phones(post, token_roles={"<b>": "blank", "|": "boundary", "<x>": "special", "AA": "phone", "TS": "phone", "ZZ": "phone"}, canonical_inventory={"AA": "aa", "TS": ["t", "s"]}, reference_intervals=[dict(x, role=x.get("role", "phone")) for x in refs], window_bounds_s=windows, bin_bounds_s=bins, native_hop_s=0.1)


def test_ctc_decode_collapses_raw_ids_before_dropping_blank_and_boundary():
    # AA, boundary, AA must stay two phones; consecutive AA collapses first.
    post = _post([[0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]], times=(.05, .15, .25, .35))
    report = _score(post, [{"start_s": 0, "end_s": .1, "phone": "aa"}, {"start_s": .2, "end_s": .3, "phone": "aa"}], windows=(0, .4))
    recognition = report["recognition"]
    assert recognition["reference_phones"] == ["aa", "aa"]
    assert recognition["decoded_phones"] == ["aa", "aa"]
    assert recognition["distance"] == 0 and recognition["full_path_observed"]


def test_true_levenshtein_counts_substitution_deletion_and_insertion():
    post = _post([[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]], times=(.05, .15, .25))
    report = _score(post, [{"start_s": 0, "end_s": .1, "phone": "aa"}, {"start_s": .1, "end_s": .2, "phone": "s"}], windows=(0, .3))
    recognition = report["recognition"]
    # AA and the trailing S match; the TS expansion's T and unmapped ZZ are
    # two insertions, so the hand-counted PER is 2 / 2.
    assert recognition["substitutions"] == 0 and recognition["insertions"] == 2
    assert recognition["deletions"] == 0 and recognition["per"] == pytest.approx(1.0)
    assert "__unmapped_model_token__:ZZ" in recognition["decoded_phones"]


def test_mass_categories_and_all_blank_returns_missing_profile():
    post = _post([[.5, .2, .1, .1, 0, .1], [1, 0, 0, 0, 0, 0]])
    report = _score(post, [], windows=(0, .1, .2))
    first, second = report["composition_diagnostics"]
    assert first["blank"] == pytest.approx(.05)
    assert first["boundary"] == pytest.approx(.02)
    assert first["special"] == pytest.approx(.01)
    assert first["unmapped_phone_mass"] == pytest.approx(.01)
    assert report["composition_profiles"][1] is None
    assert second["mapped_phone_mass"] == 0


def test_duration_weighted_cells_preserve_gaps_without_double_counting():
    post = _post([[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0]], times=(.05, .35), bounds=((0, .1), (.3, .4)))
    cells = native_time_cells(post, native_hop_s=.1)
    np.testing.assert_allclose(cells, [[0, .1], [.3, .4]])
    report = _score(post, [], windows=(0, .4))
    assert report["composition_diagnostics"][0]["observation_s"] == pytest.approx(.2)
    assert report["composition_profiles"][0] == pytest.approx([.5, .25, .25])


def test_ctc_same_token_on_both_sides_of_native_gap_starts_a_new_decode_run():
    post = _post(
        [[0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0]],
        times=(.05, .35), bounds=((0, .1), (.3, .4)),
    )
    report = _score(
        post,
        [{"start_s": 0, "end_s": .1, "phone": "aa"}, {"start_s": .3, "end_s": .4, "phone": "aa"}],
        windows=(0, .4),
    )
    recognition = report["recognition"]
    assert recognition["decoded_phones"] == ["aa", "aa"]
    assert recognition["n_discontinuities"] == 1
    assert not recognition["full_path_observed"]
    assert recognition["interpretation"].startswith("partial CTC path")


def test_expansion_preserves_order_and_unmapped_mass_is_visible():
    post = _post([[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
    report = _score(post, [{"start_s": 0, "end_s": .1, "phone": "t"}, {"start_s": .1, "end_s": .2, "phone": "s"}])
    assert report["recognition"]["decoded_phones"] == ["t", "s", "__unmapped_model_token__:ZZ"]
    assert report["composition_diagnostics"][0]["unmapped_phone_mass"] == pytest.approx(.1)


def test_onset_boundaries_reference_overlap_and_temporal_permutation_are_reported():
    post = _post([[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0]])
    refs = [{"start_s": 0, "end_s": .15, "phone": "aa"}, {"start_s": .1, "end_s": .2, "phone": "t"}]
    # Positive-duration overlap is a rejected reference, never silently assigned.
    with pytest.raises(ValueError, match="must not overlap"):
        _score(post, refs, bins=(0, .1, .2))
    report = _score(post, [{"start_s": 0, "end_s": .1, "phone": "aa"}, {"start_s": .1, "end_s": .2, "phone": "t"}], bins=(0, .1, .2))
    assert report["profile_comparisons"]["ordered_bins"][0]["total_variation"] == pytest.approx(0)
    assert report["ordering_sensitivity"]["n_comparable_transitions"] == 1
    assert report["ordering_sensitivity"]["adjacent_delta_total_variation_mean"] > 0


def test_rejects_non_normalized_or_incomplete_roles():
    with pytest.raises(ValueError, match="normalized"):
        _score(_post([[0, 0, 0, .9, 0, 0], [1, 0, 0, 0, 0, 0]]), [])
    with pytest.raises(ValueError, match="one entry"):
        evaluate_singing_phones(_post([[1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]]), token_roles=["blank"] * 5, canonical_inventory={}, reference_intervals=[], window_bounds_s=(0, .2), native_hop_s=.1)


def test_levenshtein_matches_independent_distance_and_has_consistent_counts():
    alphabet = ("a", "b")

    def oracle(ref, pred):
        if not ref:
            return len(pred)
        if not pred:
            return len(ref)
        return min(oracle(ref[1:], pred[1:]) + (ref[0] != pred[0]), oracle(ref[1:], pred) + 1, oracle(ref, pred[1:]) + 1)

    sequences = [()] + [(x,) for x in alphabet] + [(x, y) for x in alphabet for y in alphabet]
    for reference in sequences:
        for predicted in sequences:
            result = _levenshtein(reference, predicted)
            assert result["distance"] == oracle(reference, predicted)
            assert result["substitutions"] + result["deletions"] + result["insertions"] == result["distance"]
    assert _levenshtein([], ["a", "b"])["distance"] == 2
    assert _levenshtein(["a"], ["x", "x", "a"])["distance"] == 2


def test_reference_only_phone_is_retained_and_nonphone_policy_is_explicit():
    post = _post([[0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0]])
    refs = [{"start_s": 0, "end_s": .1, "phone": "aa", "role": "phone"}, {"start_s": .1, "end_s": .2, "phone": "q", "role": "phone"}]
    report = _score(post, refs)
    assert report["phones"] == ["aa", "t", "s", "q"]
    assert report["recognition"]["reference_phones"] == ["aa", "q"]
    assert report["reference_composition_diagnostics"][0]["reference_mapping_coverage"] == pytest.approx(.5)
    with pytest.raises(ValueError, match="explicit role"):
        evaluate_singing_phones(post, token_roles={"<b>": "blank", "|": "boundary", "<x>": "special", "AA": "phone", "TS": "phone", "ZZ": "phone"}, canonical_inventory={"AA": "aa"}, reference_intervals=[{"start_s": 0, "end_s": .1, "phone": "aa"}], window_bounds_s=(0, .2), native_hop_s=.1)
    report = evaluate_singing_phones(post, token_roles={"<b>": "blank", "|": "boundary", "<x>": "special", "AA": "phone", "TS": "phone", "ZZ": "phone"}, canonical_inventory={"AA": "aa"}, reference_intervals=[{"start_s": 0, "end_s": 0, "phone": "sil"}], reference_nonphone_labels=["sil"], window_bounds_s=(0, .2), native_hop_s=.1)
    assert report["recognition"]["reference_phones"] == []


def _runner_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "qualify_singing_phones.py"
    spec = importlib.util.spec_from_file_location("singing_runner_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_uses_pinned_snapshot_and_retains_native_artifact(tmp_path, monkeypatch):
    runner = _runner_module()
    audio = tmp_path / "audio.wav"
    sf = pytest.importorskip("soundfile")
    source_samples = np.array([0.125, -0.25, 0.5], dtype=np.float32)
    sf.write(audio, source_samples, 11025, subtype="FLOAT")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    series = _post([[0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0]], bounds=((0, .1), (.1, .2)))
    series = FeatureSeries(values=series.values, times_s=series.times_s, coords=series.coords, time_bounds_s=series.time_bounds_s, timebase=TimebaseSpec(kind="frames", reference="model", hop_s=.1), metadata={"extractor_id": "mock", "params_hash": "pinned", "native": "kept"})
    calls = []
    monkeypatch.setattr(runner.AudioStimulus, "from_wav", staticmethod(lambda path: (_ for _ in ()).throw(AssertionError("PCM wave loader must not be used"))))
    monkeypatch.setattr(runner, "ctc_phone_posteriors", lambda stimulus, **kwargs: calls.append((stimulus, kwargs)) or series)
    manifest = {"manifest_version": 1, "state": "development", "token_roles": {"<b>": "blank", "|": "boundary", "<x>": "special", "AA": "phone", "TS": "phone", "ZZ": "phone"}, "canonical_inventory": {"AA": "aa", "TS": ["t", "s"]}, "items": [{"id": "case_1", "audio_path": audio.name, "audio_sha256": runner._sha256(audio), "model_snapshot": snapshot.name, "model_snapshot_sha256": asset_digest(snapshot), "reference_intervals": [{"start_s": 0, "end_s": .1, "phone": "aa", "role": "phone"}], "window_bounds_s": [0, .2], "provenance": {"source_annotation": "pinned-ref", "model_revision": "pinned-model", "rendering": "source-only"}}]}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    checkpoints = []
    report = runner.run_manifest(manifest_path, root=tmp_path, artifact_dir=tmp_path / "artifacts", on_item=lambda current, item: checkpoints.append((len(current["items"]), item["id"])))
    assert report["n_failed"] == 0
    assert len(calls) == 1
    stimulus, kwargs = calls[0]
    assert stimulus.sr_hz == 11025
    np.testing.assert_allclose(stimulus.samples, source_samples)
    assert kwargs == {"model": str(snapshot), "local_files_only": True, "drop_special_tokens": False, "device": "auto"}
    restored = runner._artifact_series(tmp_path / "artifacts" / "case_1.npz")
    assert restored.metadata == series.metadata
    assert restored.timebase.hop_s == .1 and str(restored.clock) == "model"
    np.testing.assert_array_equal(restored.time_bounds_s, series.time_bounds_s)
    assert report["items"][0]["source_hashes"] == {"audio": runner._sha256(audio), "model_snapshot": asset_digest(snapshot), "reference_intervals": runner._json_digest(manifest["items"][0]["reference_intervals"])}
    generated = report["items"][0]["artifacts"]["generated_posterior"]
    assert generated["path"] == str(tmp_path / "artifacts" / "case_1.npz")
    assert generated["sha256"] == runner._sha256(Path(generated["path"]))
    assert report["items"][0]["provenance"] == manifest["items"][0]["provenance"]
    assert report["items"][0]["source_sample_rate_hz"] == 11025
    assert checkpoints == [(1, "case_1")]
    replay_item = dict(manifest["items"][0])
    replay_item.pop("audio_path")
    replay_item.pop("audio_sha256")
    replay_item.pop("model_snapshot")
    replay_item.pop("model_snapshot_sha256")
    replay_item["posterior_npz"] = generated["path"]
    replay_item["posterior_sha256"] = generated["sha256"]
    replay_manifest = dict(manifest, items=[replay_item])
    replay_path = tmp_path / "replay.json"
    replay_path.write_text(json.dumps(replay_manifest))
    replay = runner.run_manifest(replay_path, root=tmp_path, artifact_dir=tmp_path / "unused")
    assert replay["items"][0]["artifacts"]["replay_posterior"] == {"path": generated["path"], "sha256": generated["sha256"]}


@pytest.mark.parametrize("items", [[], [{"id": "same"}, {"id": "same"}], [{"id": "../escape"}]])
def test_runner_rejects_empty_duplicate_or_path_traversal_items(tmp_path, items):
    runner = _runner_module()
    manifest = {"manifest_version": 1, "state": "development", "token_roles": {}, "canonical_inventory": {}, "items": items}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        runner.run_manifest(path, root=tmp_path, artifact_dir=tmp_path / "artifacts")
