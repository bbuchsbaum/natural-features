import importlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from natural_features.core.feature_types import FeatureSeries
from natural_features.core.timebase import TimebaseSpec
from natural_features.features.audio.vocal_evidence_calibration import (
    apply_scores, evidence, fit_model, objective, predict,
)
from natural_features.features.audio.vocal_f0_benchmark import score_vocal_f0
from natural_features.features.audio.vocal_voicing_threshold import threshold_mask


def native(times=(.003, .013, .023, .033)):
    times = np.array(times)
    timebase = TimebaseSpec(kind='frames')
    context = {'clock': str(timebase.clock), 'input_bounds_s': [0., .04]}
    common = dict(times_s=times, timebase=timebase, metadata={
        'computation_context': context, 'extractor_id': 'test', 'params_hash': 'test'})
    names = ['f0_hz', 'confidence', 'voiced', 'missing', 'octave_jump',
             'candidate_f0_hz', 'available', 'source_support', 'source_supported', 'range_edge']
    pitch = FeatureSeries(np.tile([np.nan, .4, 0, 1, 0, 200, 1, .5, 1, 0], (len(times), 1)),
                          coords={'feature': names}, **common)
    canonical = FeatureSeries(np.tile([200., .8, .02, .5], (len(times), 1)),
        coords={'feature': ['candidate_f0_hz', 'confidence', 'source_rms', 'source_support']}, **common)
    activity = FeatureSeries(np.tile([.9, 1], (len(times), 1)),
        coords={'feature': ['vocal_activity_score', 'available']}, **common)
    return pitch, canonical, activity


@pytest.fixture
def driver(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'scripts'))
    return importlib.import_module('calibrate_vocal_evidence')


def test_logistic_gradient_against_central_differences():
    rng = np.random.default_rng(2026)
    x, y, w, theta = rng.normal(size=(13, 8)), rng.uniform(size=13), rng.uniform(size=13), rng.normal(size=9)
    w /= w.sum()
    _, actual = objective(theta, x, y, w, .01)
    numerical = []
    for j in range(len(theta)):
        delta = np.eye(len(theta))[j]*1e-5
        numerical.append((objective(theta+delta, x, y, w, .01)[0]-objective(theta-delta, x, y, w, .01)[0])/2e-5)
    np.testing.assert_allclose(actual, numerical, atol=1e-9)


def test_constant_features_recover_weighted_prevalence_and_ignore_zero_weight_outlier():
    x = np.ones((4, 8))*3
    y, w = np.array([0., 0., 1., 1.]), np.array([1., 2., 3., 4.])
    fitted = fit_model(x, y, w)
    np.testing.assert_allclose(predict(x, fitted), .7, atol=1e-7)
    assert fitted['scale'] == [1.]*8
    outlier = fit_model(np.vstack([x, np.full(8, 1e30)]), np.r_[y, 0.], np.r_[w, 0.])
    assert outlier == fitted
    with pytest.raises(ValueError, match='both pitched'):
        fit_model(x, np.zeros(4), w)


def test_missing_activity_is_an_indicator_not_a_veto_and_contexts_are_checked():
    pitch, canonical, activity = native()
    activity.values[1, 1] = 0
    x, admissible = evidence(pitch, canonical, activity)
    assert admissible.all()
    assert x[1, 1] == 0 and x[1, 7] == 1 and x[0, 7] == 0
    with pytest.raises(ValueError, match='share the pitch clock'):
        evidence(pitch, canonical, replace(activity, timebase=replace(activity.timebase, reference='other')))
    with pytest.raises(ValueError, match='computation contexts'):
        evidence(pitch, canonical, replace(activity, metadata={'extractor_id': 'test', 'params_hash': 'test'}))


def test_scores_upsert_without_mutating_raw_confidence_or_candidates():
    pitch, canonical, activity = native()
    original = pitch.values.copy()
    first = apply_scores(pitch, canonical, activity, [.2, .8, .8, .2], .5,
                         model_identity='raw', score_kind='rmvpe_salience')
    second = apply_scores(first, canonical, activity, [.9, .1, .1, .9], .5,
                          model_identity='fitted', score_kind='joint_logistic')
    names = list(second.coords['feature'])
    assert names.count('pitched_voicing_score') == 1
    np.testing.assert_array_equal(second.values[:, names.index('voiced')], [1, 0, 0, 1])
    np.testing.assert_array_equal(second.values[:, names.index('confidence')], original[:, 1])
    np.testing.assert_array_equal(pitch.values, original)
    assert 'uncalibrated' in first.metadata['pitched_voicing_score_semantics']
    assert 'benchmark weights' in second.metadata['pitched_voicing_score_semantics']


def test_reference_projection_uses_rmvpe_grid_and_public_scorer(driver):
    pitch, canonical, activity = native()
    pitch.values[2, 6] = 0
    # Repeated annotations, unknown, unmatched and inadmissible are all retained.
    original = SimpleNamespace(key='case', group='g', singer='S1', excerpt=1, condition='clean',
        audio_hash='h', family=None, provenance={}, reference_times=np.array([.003, .004, .013, .023, .033, .09]),
        reference_hz=np.array([200., 0., np.nan, 200., 100., 200.]))
    record = driver.EvidenceRecord(original, pitch, canonical, activity)
    np.testing.assert_array_equal(record.indices, [0, 0, 1, 2, 3, -1])
    np.testing.assert_array_equal(record.counts, [2, 0, 1, 1])
    np.testing.assert_array_equal(record.training_counts, [2, 0, 0, 1])
    assert record.labels[0] == .5
    scores = np.array([.8, .8, .8, .8])
    output = apply_scores(pitch, canonical, activity, scores, .5, model_identity='test', score_kind='joint_logistic')
    public = score_vocal_f0(record.reference_times, record.reference_hz, output)
    assert public == record.score(threshold_mask(scores, record.admissible, .5))
    assert public['n_unmatched'] == 1 and public['voicing_recall'] == .5


def test_singer_hash_and_negative_family_exclusions_and_duplicate_mass(driver):
    def r(key, singer, audio_hash, family=None):
        return SimpleNamespace(key=key, singer=singer, audio_hash=audio_hash, family=family,
            excerpt=1, condition=driver.v7.POSITIVE[0] if family is None else 'backing_only',
            training_counts=np.array([1., 1.]))
    held = [r('ancestor', 'S1', 'shared', 'family')]
    records = [r('singer', 'S1', 'a'), r('hash', 'S2', 'shared'), r('family', 'S3', 'b', 'family'),
               r('allowed', 'S4', 'c')]
    assert [x.key for x in driver.v7.purged_training(records, held)] == ['allowed']
    baseline = [r('positive', 'S4', 'c'), r('negative', 'S5', 'n', 'neg')]
    w1 = driver.v7.stream_weights(baseline)
    w2 = driver.v7.stream_weights(baseline+[r('copy', 'S6', 'n', 'neg')])
    np.testing.assert_array_equal(w1['positive'], w2['positive'])
    np.testing.assert_array_equal(w1['negative'], w2['negative']+w2['copy'])


def test_gate_changes_fail_before_native_inputs_or_fitting(driver, tmp_path, monkeypatch):
    inherited = {'gates': {'voicing_false_alarm_rate_max': .1}}
    path = tmp_path/'inherited.json'
    path.write_text(json.dumps(inherited))
    report = tmp_path/'comparison.json'
    report.write_text(json.dumps({'protocol': {'consumed_protocol': {'path': path.name, 'sha256': driver.sha(path)}}}))
    protocol = {'gates': {'voicing_false_alarm_rate_max': .2},
                'input_report': {'path': report.name, 'sha256': driver.sha(report)}}
    monkeypatch.setattr(driver, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='gates must be unchanged'):
        driver.load_records(protocol, tmp_path)
