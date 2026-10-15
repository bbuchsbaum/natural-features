from itertools import product

import numpy as np
import pytest

from natural_features.core.feature_types import FeatureSeries
from natural_features.features.audio.vocal_f0 import FEATURES
from natural_features.features.audio.vocal_voicing import (
    FEATURE_NAMES, MODEL_SCHEMA, NEURAL_REVISION, apply_voicing_model,
    decode_states, evidence,
)


def native(n=20, offset=7.25):
    values = np.zeros((n, len(FEATURES)))
    values[:, FEATURES.index('candidate_f0_hz')] = 200.
    values[:, FEATURES.index('confidence')] = .4
    values[:, FEATURES.index('source_support')] = .005
    values[:, FEATURES.index('source_rms')] = .002
    values[:, FEATURES.index('local_periodicity')] = .8
    values[:, FEATURES.index('local_supported')] = 1
    values[:, FEATURES.index('f0_hz')] = np.nan
    values[:, FEATURES.index('missing')] = 1
    return FeatureSeries(values, offset+np.arange(n)*.01, coords={'feature': FEATURES},
                         metadata={'extractor_id': 'audio.vocal_f0', 'params_hash': '5d1a29f54c2ac556',
                                   'code_version': 'vocal-f0-v6', 'model_revision': NEURAL_REVISION,
                                   'effective_hop_s': .01, 'voicing_rule': 'old mask intersection',
                                   'computation_context': {'input_bounds_s': [offset, offset+n*.01],
                                                           'dependency_bounds_s': [[offset, offset+n*.01]],
                                                           'policy': 'bidirectional'}})


def profile():
    return {'threshold': .5, 'switch_penalty': 1., 'model': {
        'schema': MODEL_SCHEMA, 'feature_names': list(FEATURE_NAMES),
        'mean': [0.]*5, 'scale': [1.]*5, 'coefficients': [0.]*5, 'intercept': 3.,
        'calibration': {'slope': 1., 'intercept': 0.}, 'model_revision': NEURAL_REVISION,
        'native_frontends': [['vocal-f0-v6', '5d1a29f54c2ac556']]}}


def test_decoder_matches_exhaustive_binary_paths():
    rng = np.random.default_rng(77)
    for n in (1, 3, 7):
        for threshold, penalty in product((.25, .5, .75), (0., 1., 2.)):
            p = rng.uniform(.05, .95, n)
            valid = rng.random(n) > .2
            emission = np.log(p/(1-p))-np.log(threshold/(1-threshold))
            candidates = [np.array(path, bool) for path in product((False, True), repeat=n)
                          if not np.any(np.array(path) & ~valid)]
            def score(path):
                return float(emission[path].sum()-penalty*np.count_nonzero(path[1:] != path[:-1]))
            best = max(score(path) for path in candidates)
            actual = decode_states(p, valid, threshold, penalty)
            assert score(actual) == pytest.approx(best)
            if penalty == 0:
                np.testing.assert_array_equal(actual, (p > threshold) & valid)
    np.testing.assert_array_equal(decode_states([.5]*5, [True]*5, .5, 0), [False]*5)
    np.testing.assert_array_equal(decode_states([.5]*5, [True]*5, .5, 2), [False]*5)


def test_sequence_can_support_weak_evidence_but_zero_energy_forces_unvoiced():
    np.testing.assert_array_equal(decode_states([.99, .4, .99], [True]*3, .5, 1), [True]*3)
    np.testing.assert_array_equal(decode_states([.99]*3, [True, False, True], .5, 1), [True, False, True])


def test_recovers_below_old_cutoffs_without_changing_native_pitch_or_clock():
    feature = native()
    before = feature.values.copy()
    feature.values[5, FEATURES.index('source_rms')] = 0
    before[5, FEATURES.index('source_rms')] = 0
    result = apply_voicing_model(feature, profile())
    assert np.isnan(result.values[5, 0])
    keep = np.arange(len(result.values)) != 5
    np.testing.assert_array_equal(result.values[keep, 0], 200.)
    np.testing.assert_array_equal(result.times_s, feature.times_s)
    np.testing.assert_array_equal(result.values[:, 9], feature.values[:, 9])
    np.testing.assert_array_equal(feature.values, before)
    assert result.timebase == feature.timebase
    assert result.metadata['computation_context'] == feature.metadata['computation_context']
    assert result.metadata['legacy_voicing_rule'] == 'old mask intersection'
    assert 'old confidence/source/local/temporal gates do not constrain' in result.metadata['voicing_rule']


def test_missing_local_evidence_is_not_an_automatic_veto():
    feature = native()
    feature.values[:, FEATURES.index('local_supported')] = 0
    feature.values[:, FEATURES.index('local_periodicity')] = np.nan
    x, valid = evidence(feature)
    assert np.isfinite(x).all() and valid.all()
    assert (apply_voicing_model(feature, profile()).values[:, 4] > 0).all()


@pytest.mark.parametrize('change', ['missing_revision', 'different_revision', 'different_params', 'different_hop'])
def test_incompatible_frontend_cannot_silently_use_model(change):
    feature = native()
    if change == 'missing_revision':
        feature.metadata.pop('model_revision')
    elif change == 'different_revision':
        feature.metadata['model_revision'] = 'different neural weights'
    elif change == 'different_params':
        feature.metadata['params_hash'] = 'different fmin or local RMS normalization'
    else:
        feature.metadata['effective_hop_s'] = .02
    with pytest.raises(ValueError):
        apply_voicing_model(feature, profile())


def test_nonpositive_candidates_and_outside_duration_are_missing():
    feature = native()
    feature.values[2, 9] = np.nan
    feature.metadata['computation_context']['input_bounds_s'][1] -= .01
    result = apply_voicing_model(feature, profile())
    assert np.isnan(result.values[[2, -1], 0]).all()


def test_malformed_local_support_is_not_treated_as_valid_evidence():
    feature = native()
    feature.values[2, FEATURES.index('local_periodicity')] = np.nan
    with pytest.raises(ValueError, match='available local evidence'):
        evidence(feature)
