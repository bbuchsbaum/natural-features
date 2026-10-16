import numpy as np
import pytest
import importlib.util
import json
from pathlib import Path

from natural_features.core.feature_types import FeatureSeries
from natural_features.core.timebase import TimebaseSpec
from natural_features.features.speech.phone_inventory import PhoneInventory
from natural_features.features.speech.singing_pooling import compare_singing_representations


ROLES = {"<pad>": "blank", "a": "phone", "ab": "phone", "zz": "phone"}


def _post(values, times=(.05, .15, .45)):
    return FeatureSeries(values=np.asarray(values, dtype=float), times_s=np.asarray(times), coords={"feature": list(ROLES)}, metadata={"extractor_id": "test", "params_hash": "test"}, timebase=TimebaseSpec(kind="frames", hop_s=.1))


def _inventory():
    return PhoneInventory({"schema": "EnglishSingingPhoneInventory/v2", "mappings": {"a": {"phones": ["aa"], "relation": "exact", "reason": "test"}, "ab": {"phones": ["aa", "b"], "relation": "sequence", "reason": "test"}}, "ambiguous": {"zz": {"relation": "ambiguous", "alternatives": [["z"], ["zh"]], "reason": "test", "source": "primary"}}})


def _compare(values, refs, times=(.05, .15, .45), legacy=None):
    return compare_singing_representations(_post(values, times), token_roles=ROLES, conservative_inventory=legacy or {"a": "aa", "ab": ["aa", "b"]}, inventory_v2=_inventory(), reference_intervals=refs, window_bounds_s=[0, .3, .6], bin_bounds_s=[0, .3, .6])


def test_mapping_and_pooling_are_a_complete_factorial_with_token_audit():
    report = _compare([[.8, .2, 0, 0], [.4, .6, 0, 0], [.8, .2, 0, 0]], [{"start_s": 0, "end_s": .2, "phone": "aa", "role": "phone"}])
    branch = report["composition"]["mappings"]
    assert set(branch) == {"v1_legacy_exact", "v2_english_decisions"}
    assert set(branch["v2_english_decisions"]["representations"]) == {"raw_posterior_integral", "framewise_phone_conditional", "greedy_ctc_token_events"}
    zz = next(row for row in report["token_audit"] if row["token"] == "zz")
    assert zz["mappings"]["v2_english_decisions"]["relation"] == "ambiguous"
    assert zz["mappings"]["v2_english_decisions"]["mapped"] is False
    assert report["recognition"]["v1_legacy_exact"]["per"] is not None


def test_compound_duplicate_target_adds_mass_and_event_components_once_each():
    inventory = PhoneInventory({"schema": "EnglishSingingPhoneInventory/v2", "mappings": {"ab": {"phones": ["aa", "aa"], "relation": "sequence", "reason": "test"}}})
    report = compare_singing_representations(_post([[0, 0, 1, 0], [0, 0, 1, 0], [1, 0, 0, 0]]), token_roles=ROLES, conservative_inventory={"ab": ["aa", "aa"]}, inventory_v2=inventory, reference_intervals=[{"start_s": 0, "end_s": .3, "phone": "aa", "role": "phone"}], window_bounds_s=[0, .6], bin_bounds_s=[0, .6])
    event = report["composition"]["mappings"]["v2_english_decisions"]["representations"]["greedy_ctc_token_events"]["profiles"][0]
    assert event == [1.0]
    # np.add.at avoids fancy-index loss: duplicate components produce two events.
    assert report["composition"]["mappings"]["v2_english_decisions"]["representations"]["greedy_ctc_token_events"]["event_totals"] == [2.0]
    assert report["token_audit"][2]["greedy_event_count"] == 1


def test_reference_requires_explicit_roles_and_nonoverlap():
    with pytest.raises(ValueError, match="explicit role"):
        _compare([[1, 0, 0, 0]] * 3, [{"start_s": 0, "end_s": .1, "phone": "aa"}])
    with pytest.raises(ValueError, match="must not overlap"):
        _compare([[1, 0, 0, 0]] * 3, [{"start_s": 0, "end_s": .2, "phone": "aa", "role": "phone"}, {"start_s": .1, "end_s": .3, "phone": "aa", "role": "phone"}])


def test_mapping_only_difference_changes_v1_but_not_v2_and_occurrence_is_not_duration():
    report = _compare([[0, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]], [{"start_s": .05, "end_s": .35, "phone": "aa", "role": "phone"}], legacy={"a": "b"})
    branches = report["composition"]["mappings"]
    assert branches["v1_legacy_exact"]["representations"]["raw_posterior_integral"]["profiles"][0] != branches["v2_english_decisions"]["representations"]["raw_posterior_integral"]["profiles"][0]
    assert report["composition"]["reference_duration_composition"][1] is not None
    assert report["composition"]["reference_occurrence_composition"][1] is None


def test_gap_and_blank_reset_ctc_collapse_before_role_filtering():
    report = _compare([[0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]], [{"start_s": 0, "end_s": .5, "phone": "aa", "role": "phone"}])
    # a, blank, a: two phone events; the last is also separated by a native gap.
    assert report["token_audit"][1]["greedy_event_count"] == 2


def test_v2_mapping_difference_leaves_v1_legacy_decision_unchanged():
    inventory = PhoneInventory({"schema": "EnglishSingingPhoneInventory/v2", "mappings": {"ɑ": {"phones": ["aa"], "relation": "coarse-equivalence", "reason": "pre-frozen semantic decision", "source": "primary"}}})
    post = FeatureSeries(values=np.array([[0., 1.]]), times_s=np.array([.05]), coords={"feature": ["<pad>", "ɑ"]}, metadata={"extractor_id": "test", "params_hash": "test"}, timebase=TimebaseSpec(kind="frames", hop_s=.1))
    report = compare_singing_representations(post, token_roles={"<pad>": "blank", "ɑ": "phone"}, conservative_inventory={}, inventory_v2=inventory, reference_intervals=[{"start_s": 0, "end_s": .1, "phone": "aa", "role": "phone"}], window_bounds_s=[0, .1], bin_bounds_s=[0, .1])
    token = report["token_audit"][1]["mappings"]
    assert token["v1_legacy_exact"]["relation"] == "unsupported"
    assert token["v2_english_decisions"]["phones"] == ["aa"]


def test_cached_npz_loader_preserves_required_metadata(tmp_path):
    script = Path(__file__).parents[2] / "scripts/compare_singing_phone_representations.py"
    spec = importlib.util.spec_from_file_location("singing_representation_runner", script)
    assert spec and spec.loader
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    path = tmp_path / "posterior.npz"
    np.savez(path, times_s=np.array([.05]), values=np.array([[1., 0.]]), labels=np.array(["<pad>", "a"]), bounds_s=np.array([[0., .1]]), metadata_json=np.array(json.dumps({"extractor_id": "retained", "params_hash": "abc", "extra": "kept"})), timebase_json=np.array(json.dumps({"kind": "frames", "hop_s": .1})), temporal_context_json=np.array(json.dumps({"mappings": []})))
    series = runner._series(path)
    assert series.metadata["extractor_id"] == "retained"
    assert series.metadata["params_hash"] == "abc"
    assert series.metadata["extra"] == "kept"
