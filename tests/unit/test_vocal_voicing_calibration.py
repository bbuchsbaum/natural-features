import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'scripts'))
    return importlib.import_module('calibrate_vocal_voicing')


def test_aggregated_labels_equal_expanded_likelihood_and_gradient(module):
    x = np.array([[1., 2.], [-1., .5]])
    y = np.array([.25, 1.])
    theta = np.array([.1, -.4, .7])
    loss, gradient = module.binary_objective(theta, x, y, np.array([.8, .2]), .01)
    expanded = np.repeat(x, [4, 1], axis=0)
    loss2, gradient2 = module.binary_objective(theta, expanded, np.array([1, 0, 0, 0, 1]), np.full(5, .2), .01)
    assert loss == pytest.approx(loss2)
    np.testing.assert_allclose(gradient, gradient2)
    epsilon = 1e-5
    numerical = [(module.binary_objective(theta+np.eye(3)[j]*epsilon, x, y, np.array([.8, .2]), .01)[0]
                  -module.binary_objective(theta-np.eye(3)[j]*epsilon, x, y, np.array([.8, .2]), .01)[0])/(2*epsilon) for j in range(3)]
    np.testing.assert_allclose(gradient, numerical, atol=1e-9)


def test_platt_gradient_and_mixed_label_brier(module):
    theta = np.array([.7, -.3])
    logits = np.array([-3., 1., 4.])
    y = np.array([0., .5, 1.])
    w = np.array([.2, .3, .5])
    _, gradient = module.platt_objective(theta, logits, y, w, .001)
    eps = 1e-5
    numerical = [(module.platt_objective(theta+np.eye(2)[j]*eps, logits, y, w, .001)[0]
                  -module.platt_objective(theta-np.eye(2)[j]*eps, logits, y, w, .001)[0])/(2*eps) for j in range(2)]
    np.testing.assert_allclose(gradient, numerical, atol=1e-9)
    assert module.probability_statistics([.5], [.5], [1.])['brier'] == .25


def stub(key, singer='S1', condition='solo', family=None, audio=None, excerpt=1):
    return SimpleNamespace(key=key, singer=singer, condition=condition, family=family,
                           audio_hash=audio or key, excerpt=excerpt, training_counts=np.array([1., 2.]))


def test_duplicate_negative_cannot_increase_its_training_mass(module):
    vocal = stub('v')
    a = stub('a', condition='backing', family='backing:F:0', audio='same')
    b = stub('b', singer='S2', condition='backing', family='backing:F:0', audio='same')
    one = module.stream_weights([vocal, a])
    two = module.stream_weights([vocal, a, b])
    np.testing.assert_allclose(one['v'], two['v'])
    np.testing.assert_allclose(one['a'], two['a']+two['b'])


def test_family_purge_covers_prefix_negatives_and_preserves_other_mixtures(module):
    held = stub('held', condition='backing', family='backing:F:0', audio='short')
    prefix = stub('prefix', singer='S2', condition='backing', family='backing:F:0', audio='long')
    mixture = stub('mixture', singer='S2', condition='mixture_minus6db')
    exact = stub('exact', singer='S3', audio='short')
    assert [r.key for r in module.purged_training([held, prefix, mixture, exact], [held])] == ['mixture']


def test_grouped_folds_keep_all_singer_excerpts_and_conditions_together(module):
    records = [stub(f'{s}-{i}', singer=f'S{s}', excerpt=i) for s in range(1, 18) for i in (1, 2)]
    folds = module.grouped_folds(records, 5, outer=True)
    assert sum(map(len, folds)) == len(records)
    for fold, rows in enumerate(folds):
        assert all((int(r.singer[1:])-1)%5 == fold for r in rows)
    assert all(len({i for i, rows in enumerate(folds) if any(r.singer == f'S{s}' for r in rows)}) == 1 for s in range(1, 18))


def test_fixed_fallback_is_not_an_eligible_profile(module):
    params, eligible = module.select_profiles([])
    assert params == {'threshold': .5, 'switch_penalty': 0.} and not eligible


def test_structural_absence_is_inner_only_and_present_controls_must_pass(module):
    protocol = json.loads((Path(__file__).resolve().parents[2]/'tests/benchmarks/thresholds/vocal_voicing_development_v7.json').read_text())
    rows = [SimpleNamespace(key=f'{c}:{i}', condition=c, group='example')
            for c in module.POSITIVE for i in range(6)]
    passing = dict(zip(module.METRICS, [.9, .9, .05, 10., 0., 0.]))
    metrics = {r.key: passing for r in rows}
    inner = module.summarize(rows, metrics, protocol['gates'])[0]
    assert inner['gate']['passed']
    assert inner['untested_controls'] == {'backing': 'ancestor_family_purge', 'silence': 'ancestor_family_purge'}
    outer = module.summarize(rows, metrics, protocol['gates'], complete=True)[0]
    assert not outer['gate']['passed'] and not outer['untested_controls']
    assert outer['gate']['failures'] == ['backing: absent', 'silence: absent']
    rows.append(SimpleNamespace(key='bad_control', condition='backing', group='example'))
    metrics['bad_control'] = {**passing, 'voicing_false_alarm_rate': .06}
    present = module.summarize(rows, metrics, protocol['gates'])[0]
    assert not present['gate']['passed']
    assert 'backing' not in present['untested_controls']


def test_outer_held_labels_cannot_affect_calibration_or_profile_selection(module, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    native = importlib.import_module('test_vocal_voicing').native
    protocol = json.loads((Path(__file__).resolve().parents[2]/'tests/benchmarks/thresholds/vocal_voicing_development_v7.json').read_text())
    records = []
    for singer in range(1, 18):
        group = 'original_development' if singer <= 5 else 'consumed_v2_test' if singer <= 11 else 'consumed_v6_test'
        for condition in module.POSITIVE:
            feature = native(n=20, offset=0)
            positive = np.arange(20)%4 < 2
            feature.values[:, 1] = np.where(positive, .85, .15)
            reference = np.where(positive, 200., 0.)
            key = f'{singer}:{condition}'
            records.append(module.Record(key, group, f'S{singer}', singer, condition, key, None,
                                         feature, feature.times_s, reference, {}))
    held = module.grouped_folds(records, 5, outer=True)[0]
    train = module.purged_training(records, held)
    first = module.Trainer(protocol, [])
    selected1 = module.select(first, train, held, 'test')
    model1 = first.pipeline(train, held, 'test/selected')
    for r in held:
        r.reference_hz = np.where(r.reference_hz > 0, 0., 200.)
        r.__post_init__()
    second = module.Trainer(protocol, [])
    selected2 = module.select(second, module.purged_training(records, held), held, 'test')
    model2 = second.pipeline(train, held, 'test/selected')
    assert selected1 == selected2 and model1 == model2
    held_keys = {r.key for r in held}
    assert all(not held_keys.intersection(fit['training_keys']) for fit in first.trace)
