from __future__ import annotations

import numpy as np

from natural_features.core.feature_types import FeatureSeries
from natural_features.features.audio.vocal_activity_benchmark import score_vocal_activity


def _activity(times, scores, available):
    return FeatureSeries(
        values=np.column_stack([scores, available, np.zeros(len(times))]),
        times_s=np.asarray(times, float),
        coords={"feature": ["vocal_activity_score", "available", "padded"]},
        metadata={"extractor_id": "toy_activity", "params_hash": "toy"},
    )


def _brute(times, truth, estimate_times, scores, available, threshold, maximum_error):
    matched = np.zeros(len(times), bool)
    predicted = np.zeros(len(times), bool)
    for i, (time, value) in enumerate(zip(times, truth)):
        if np.isnan(value):
            continue
        candidates = [j for j, estimate_time in enumerate(estimate_times) if abs(estimate_time - time) <= maximum_error]
        if candidates:
            # The production contract chooses the earlier native row on ties.
            j = min(candidates, key=lambda x: (abs(estimate_times[x] - time), estimate_times[x]))
            if available[j]:
                matched[i] = True
                predicted[i] = scores[j] > threshold
    pos, neg = truth == 1, truth == 0
    tp = int((pos & predicted).sum())
    fp = int((neg & predicted).sum())
    tn = int((neg & matched & ~predicted).sum())
    return {"true_positive": tp, "false_positive": fp, "false_negative": int(pos.sum()) - tp, "true_negative": tn, "n_matched": int(matched.sum()), "n_unmatched": int(((~np.isnan(truth)) & ~matched).sum())}


def test_scores_match_independent_nearest_and_count_oracle_with_unknown_and_unavailable():
    times = np.array([0.05, 0.10, 0.15, 0.20, 0.25])
    truth = np.array([1.0, 0.0, np.nan, 1.0, 0.0])
    estimate_times = np.array([0.00, 0.10, 0.20, 0.30])
    scores = np.array([0.9, 0.8, 0.2, 0.7])
    available = np.array([1, 1, 0, 1])
    report = score_vocal_activity(times, truth, _activity(estimate_times, scores, available), threshold=.5, max_time_error_s=.051)
    assert {key: report[key] for key in ("true_positive", "false_positive", "false_negative", "true_negative", "n_matched", "n_unmatched")} == _brute(times, truth, estimate_times, scores, available.astype(bool), .5, .051)
    assert report["n_unknown"] == 1
    assert report["coverage"] == .5
    assert report["false_alarm_rate"] == .5


def test_tie_selects_earlier_native_frame_and_strict_threshold_does_not_accept_equal_score():
    estimate = _activity([0.0, 0.2], [.9, .1], [1, 1])
    report = score_vocal_activity([.1], [1], estimate, threshold=.9, max_time_error_s=.1)
    assert report["n_matched"] == 1
    assert report["true_positive"] == 0  # 0.9 is not greater than 0.9
    report = score_vocal_activity([.1], [1], estimate, threshold=.5, max_time_error_s=.1)
    assert report["true_positive"] == 1  # earlier 0.0 row wins tie, not 0.2
