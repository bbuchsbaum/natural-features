import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def calibration(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / 'scripts'))
    return importlib.import_module('calibrate_vocal_f0_local')


def test_tied_inclusive_values_are_removed_together(calibration):
    q = np.array([.5, .5, 1., np.nan, -.1])
    points = calibration.decision_breakpoints(q)
    np.testing.assert_array_equal(points, [0., np.nextafter(.5, np.inf), 1.])
    assert np.count_nonzero(q >= .5) == 3
    assert np.count_nonzero(q >= points[1]) == 1
    assert np.count_nonzero(q >= 1.) == 1


def test_complete_intervals_keep_nonmonotone_gate_holes(calibration):
    points = calibration.decision_breakpoints([.2, .4, .6, .8])
    intervals, search = calibration.feasible_intervals(
        points, lambda t: t > .2, lambda t: t <= .8,
        lambda t: t <= .4 or t > .6)
    assert len(intervals) == 2
    assert intervals[0]['lower_inclusive'] == np.nextafter(.2, np.inf)
    assert intervals[0]['upper_inclusive'] == .4
    assert intervals[1]['lower_inclusive'] == np.nextafter(.6, np.inf)
    assert intervals[1]['upper_inclusive'] == .8
    assert search['fully_checked_states'] == 3


def test_no_threshold_can_fix_conflicting_monotone_gates(calibration):
    points = calibration.decision_breakpoints([.2, .4, .6, .8])
    intervals, search = calibration.feasible_intervals(
        points, lambda t: t > .6, lambda t: t <= .4,
        lambda t: pytest.fail('no admissible state should reach full scoring'))
    assert intervals == [] and search['fully_checked_states'] == 0
    assert calibration.select_midpoint(intervals) is None


def test_widest_midpoint_and_tie_rule(calibration):
    assert calibration.select_midpoint([
        {'lower_inclusive': .125, 'upper_inclusive': .25},
        {'lower_inclusive': .5, 'upper_inclusive': .625}]) == .1875
    assert calibration.select_midpoint([
        {'lower_inclusive': .125, 'upper_inclusive': .25},
        {'lower_inclusive': .5, 'upper_inclusive': 1.}]) == .75


def test_prepared_metrics_keep_unknown_unmatched_and_octaves(calibration):
    from natural_features.core.feature_types import FeatureSeries
    from natural_features.features.audio.vocal_f0 import FEATURES
    from natural_features.features.audio.vocal_f0_benchmark import score_vocal_f0
    values = np.zeros((4, len(FEATURES)))
    values[:, 0] = values[:, 9] = [200, 400, 200, 200]
    values[:, 4] = values[:, 11] = values[:, 13] = 1
    values[:, 12] = [.5, .9, .75, .8]
    fs = FeatureSeries(values, np.arange(4) / 100, coords={'feature': FEATURES},
                       metadata={'extractor_id': 'test', 'params_hash': 'test'})
    times = np.array([0., .01, .02, .03, .1])
    hz = np.array([200., 200., 0., np.nan, 200.])
    prepared = calibration.prepare(times, hz, fs)
    for threshold in (.5, np.nextafter(.5, np.inf), .8):
        after = values.copy()
        keep = after[:, 12] >= threshold
        after[:, 0] = np.where(keep, after[:, 0], np.nan)
        after[:, 4] = keep
        candidate = FeatureSeries(after, fs.times_s, coords=fs.coords, metadata=fs.metadata)
        assert calibration.metrics(prepared, threshold) == score_vocal_f0(times, hz, candidate)
