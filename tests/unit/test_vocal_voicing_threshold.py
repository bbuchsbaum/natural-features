import importlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from natural_features.features.audio.vocal_voicing import decode_states
from natural_features.features.audio.vocal_voicing_threshold import (
    apply_threshold_model, canonical_thresholds, effective_score, threshold_mask,
)


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'scripts'))
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    return (importlib.import_module('calibrate_vocal_voicing_threshold'),
            importlib.import_module('calibrate_vocal_voicing'), importlib.import_module('test_vocal_voicing'))


def test_direct_zero_switch_comparison_does_not_lose_tiny_positive_emission():
    p = np.array([.9999, np.nextafter(.5, 1.)])
    np.testing.assert_array_equal(decode_states(p, [True, True], .5, 0), [True, False])
    np.testing.assert_array_equal(threshold_mask(p, [True, True], .5), [True, True])
    np.testing.assert_array_equal(threshold_mask(p[::-1], [True, True], .5), [True, True])


def test_canonical_classes_cover_ties_extremes_and_neighboring_float_thresholds():
    p = np.r_[0., 1e-14, 1e-12, .2, .2, np.nextafter(.2, 1.), .5, np.nextafter(.5, 1.), .9999, 1.]
    valid = np.ones(len(p), bool)
    thresholds = canonical_thresholds([p], [valid])
    masks = {tuple(threshold_mask(p, valid, t)) for t in thresholds}
    assert len(masks) == len(thresholds) == len(np.unique(effective_score(p)))+1
    assert threshold_mask(p, valid, thresholds[0]).all()
    assert not threshold_mask(p, valid, thresholds[-1]).any()
    for t in np.r_[thresholds, np.nextafter(thresholds, 0), np.nextafter(thresholds, 1), (thresholds[:-1]+thresholds[1:])/2]:
        if 0 < t < 1:
            assert tuple(threshold_mask(p, valid, t)) in masks
    assert canonical_thresholds([p], [np.zeros(len(p), bool)]).tolist() == [5e-13]


def _records(v7, fixture, perfect=False):
    rng = np.random.default_rng(88)
    records, probabilities = [], {}
    for group in ('A', 'B'):
        for condition in v7.CONDS:
            for excerpt in (1, 2):
                feature = fixture.native(n=12, offset=0)
                times = np.repeat(feature.times_s, 2)+np.tile([0., .003], 12)
                hz = np.where(np.arange(len(times))%6 < 4, 200., 0.)
                if condition not in v7.POSITIVE:
                    hz[:] = 0
                if not perfect:
                    hz[3] = np.nan
                    hz[7] = 100.
                    times = np.r_[times, 1.]
                    hz = np.r_[hz, 200. if condition in v7.POSITIVE else 0.]
                    feature.values[4, 10] = 0
                key = f'{group}:{condition}:{excerpt}'
                record = v7.Record(key, group, f'S{excerpt}', excerpt, condition, key, None, feature, times, hz, {})
                records.append(record)
                probabilities[key] = (np.where(hz[::2] > 0, .9, .1) if perfect else rng.choice([0., .1, .5, .75, .9, 1.], 12))
    return records, probabilities


@pytest.mark.parametrize('perfect', [False, True])
def test_complete_sweep_matches_brute_public_scoring_with_duplicate_annotations(modules, perfect):
    module, v7, fixture = modules
    records, probabilities = _records(v7, fixture, perfect)
    gates = json.loads((v7.ROOT/'tests/benchmarks/thresholds/vocal_voicing_development_v7.json').read_text())['gates']
    selected, curves = module.exhaustive_search(records, probabilities, gates)
    eligible = []
    for i, threshold in enumerate(curves['threshold']):
        metrics = {}
        for r in records:
            mask = threshold_mask(probabilities[r.key], r.admissible, threshold)
            values = r.feature.values.copy()
            values[:, 0] = np.where(mask, r.candidate, np.nan)
            values[:, 4] = mask
            values[:, 5] = ~mask
            output = replace(r.feature, values=values)
            metrics[r.key] = v7.score_vocal_f0(r.reference_times, r.reference_hz, output)
        groups = v7.summarize(records, metrics, gates)
        passed = all(g['gate']['passed'] for g in groups)
        assert curves['all_gates_pass'][i] == passed
        for group in groups:
            for c, results in group['summary'].items():
                for metric, expected in results.items():
                    if metric == 'median_abs_cents_error' and not curves['monotone_pass'][i]:
                        continue
                    actual = curves['::'.join((group['name'], c, metric))][i]
                    assert np.isnan(actual) if expected is None else actual == pytest.approx(expected, abs=1e-14)
        if passed:
            eligible.append(i)
    assert selected['all_gates_feasible_states'] == len(eligible)
    if perfect:
        assert selected['eligible'] and selected['selected']['threshold'] == .1
    else:
        assert not selected['eligible'] and selected['fallback_used']


def test_v8_output_preserves_clock_candidates_and_identifies_new_decoder(modules):
    _, _, fixture = modules
    native = fixture.native()
    original = native.values.copy()
    profile = {**fixture.profile(), 'switch_penalty': 0., 'protocol_sha256': 'test-v8'}
    result = apply_threshold_model(native, profile)
    np.testing.assert_array_equal(native.values, original)
    np.testing.assert_array_equal(result.times_s, native.times_s)
    np.testing.assert_array_equal(result.values[:, 9], original[:, 9])
    assert result.metadata['code_version'] == 'vocal-voicing-threshold-v8'
    assert result.metadata['computation_context'] == native.metadata['computation_context']
    with pytest.raises(ValueError, match='zero switch'):
        apply_threshold_model(native, fixture.profile())


def test_conditional_octave_gate_can_reject_monotone_feasible_states(modules):
    module, v7, fixture = modules
    records, probabilities = _records(v7, fixture, perfect=True)
    for r in records:
        if r.condition in v7.POSITIVE:
            r.reference_hz[:2] = 100.
            r.__post_init__()
    gates = json.loads((v7.ROOT/'tests/benchmarks/thresholds/vocal_voicing_development_v7.json').read_text())['gates']
    selected, _ = module.exhaustive_search(records, probabilities, gates)
    assert selected['monotone_feasible_states'] > 0
    assert selected['all_gates_feasible_states'] == 0 and not selected['eligible']


def test_reused_selector_is_unchanged_when_outer_labels_are_mutated(modules):
    module, v7, fixture = modules
    protocol = json.loads((v7.ROOT/'tests/benchmarks/thresholds/vocal_voicing_development_v7.json').read_text())
    records = []
    for s in range(1, 18):
        for condition in v7.POSITIVE:
            feature = fixture.native(n=12, offset=0)
            positive = np.arange(12)%4 < 2
            feature.values[:, 1] = np.where(positive, .85, .15)
            key = f'{s}:{condition}'
            records.append(v7.Record(key, 'all', f'S{s}', s, condition, key, None, feature,
                                     feature.times_s, np.where(positive, 200., 0.), {}))
    held = v7.grouped_folds(records, 5, outer=True)[0]
    train = v7.purged_training(records, held)
    trainer = v7.Trainer(protocol, [])
    saved = v7.select(trainer, train, held, 'test')
    reuse = module.ReusedModels({'fit_trace': trainer.trace})
    first_prob = reuse.selector(train, held, 'test', saved)
    first, _ = module.exhaustive_search(train, first_prob, protocol['gates'])
    for r in held:
        r.reference_hz = np.where(r.reference_hz > 0, 0., 200.)
        r.__post_init__()
    second_prob = reuse.selector(train, held, 'test', saved)
    second, _ = module.exhaustive_search(train, second_prob, protocol['gates'])
    assert first == second
    assert reuse.checked == set(reuse.traces)
    corrupted = json.loads(json.dumps(saved))
    corrupted['selector_models'][0]['validation_keys'] = [held[0].key]
    with pytest.raises(ValueError, match='fold mismatch'):
        reuse.selector(train, held, 'test', corrupted)
