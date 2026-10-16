"""Vocal-presence scoring, independent of pitched voicing and phone support."""
from __future__ import annotations

import numpy as np

from natural_features.core.feature_types import FeatureSeries
from natural_features.features.audio.vocal_f0_benchmark import _nearest_index


def score_vocal_activity(reference_times_s, reference_present, estimate: FeatureSeries,
                         *, threshold=0.51, max_time_error_s=1 / 140 + 1e-8):
    """Score explicit presence labels (0/1/NaN); never infer them from F0.

    Unmatched labels remain in recall/accuracy denominators. Missing predictions
    are not true negatives. False-alarm rates must be read with class-specific
    coverage; abstention is not evidence of absence.
    """
    times = np.asarray(reference_times_s, dtype=float)
    truth = np.asarray(reference_present, dtype=float)
    if (times.ndim != 1 or truth.shape != times.shape or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError("reference times must be finite, ordered, unique and match labels")
    if not np.isin(truth[~np.isnan(truth)], [0, 1]).all():
        raise ValueError("vocal-presence references must be 0, 1 or NaN, not F0")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must lie in [0, 1]")
    if not np.isfinite(max_time_error_s) or max_time_error_s < 0:
        raise ValueError("max_time_error_s must be finite and nonnegative")
    names = list(estimate.coords["feature"])
    prob = estimate.values[:, names.index("vocal_activity_score")]
    available = estimate.values[:, names.index("available")] > 0
    if not np.isfinite(prob[available]).all() or np.any((prob[available] < 0) | (prob[available] > 1)):
        raise ValueError("available vocal-activity scores must lie in [0, 1]")
    known = ~np.isnan(truth)
    matched = np.zeros(len(times), bool)
    predicted = np.zeros(len(times), bool)
    for j in np.flatnonzero(known):
        i = _nearest_index(estimate.times_s, times[j], max_time_error_s)
        if i is not None and available[i]:
            matched[j] = True
            predicted[j] = prob[i] > threshold
    pos, neg = truth == 1, truth == 0
    tp, fp = int((pos & predicted).sum()), int((neg & predicted).sum())
    tn = int((neg & matched & ~predicted).sum())
    fn = int(pos.sum()) - tp

    def ratio(a, b):
        return float(a / b) if b else None

    return {"reference_kind": "vocal_presence", "threshold": threshold,
            "n_known": int(known.sum()), "n_unknown": int((~known).sum()),
            "n_positive": int(pos.sum()), "n_negative": int(neg.sum()),
            "n_matched": int(matched.sum()), "n_unmatched": int((known & ~matched).sum()),
            "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
            "coverage": ratio(matched.sum(), known.sum()),
            "positive_coverage": ratio((pos & matched).sum(), pos.sum()),
            "negative_coverage": ratio((neg & matched).sum(), neg.sum()),
            "recall": ratio(tp, pos.sum()), "false_alarm_rate": ratio(fp, neg.sum()),
            "precision": ratio(tp, tp + fp), "f1": ratio(2 * tp, 2 * tp + fp + fn),
            "accuracy_with_abstentions_as_errors": ratio(tp + tn, known.sum())}
