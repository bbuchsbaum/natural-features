"""Explicit v8 zero-switch decoder; the frozen v7 implementation is preserved."""
from __future__ import annotations

from dataclasses import replace
import numpy as np

from .vocal_voicing import apply_voicing_model


def effective_score(probability):
    p = np.asarray(probability, dtype=np.float64)
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError('invalid probabilities')
    clipped = np.clip(p, 1e-12, 1-1e-12)
    return np.log(clipped / (1-clipped))


def threshold_score(threshold):
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('threshold must be strictly between zero and one')
    return float(np.log(threshold / (1-threshold)))


def threshold_mask(probability, admissible, threshold):
    score = effective_score(probability)
    valid = np.asarray(admissible, dtype=bool)
    if score.ndim != 1 or valid.shape != score.shape:
        raise ValueError('probabilities and admissibility must be matching vectors')
    return valid & (score > threshold_score(threshold))


def canonical_thresholds(probabilities, admissibilities):
    """Enumerate all attainable masks, including both extreme states."""
    if len(probabilities) != len(admissibilities):
        raise ValueError('probability/admissibility record counts differ')
    values = []
    for probability, admissible in zip(probabilities, admissibilities):
        probability = np.asarray(probability, dtype=np.float64)
        valid = np.asarray(admissible, dtype=bool)
        effective_score(probability)
        if probability.ndim != 1 or valid.shape != probability.shape:
            raise ValueError('probabilities and admissibility must be matching vectors')
        values.append(np.clip(probability[valid], 1e-12, 1-1e-12))
    p = np.sort(np.concatenate(values)) if values else np.array([])
    scores = effective_score(p)
    unique, first = np.unique(scores, return_index=True)
    representatives = p[first]
    if not np.array_equal(effective_score(representatives), unique):
        raise ValueError('invalid score-class representative')
    return np.r_[5e-13, representatives]


def apply_threshold_model(feature, profile):
    """Apply v7 fitted evidence with exact independent zero-switch decisions.

    Reuse v7 model/frontend validation and output provenance, then replace its
    rounded DP mask. No v7 source or retained result is changed.
    """
    if profile.get('switch_penalty') != 0:
        raise ValueError('v8 threshold model requires zero switch penalty')
    out = apply_voicing_model(feature, profile)
    names = list(out.coords['feature'])
    values = out.values.copy()
    voiced = threshold_mask(values[:, names.index('voicing_probability')],
                            values[:, names.index('voicing_admissible')], profile['threshold'])
    candidate = values[:, names.index('candidate_f0_hz')]
    values[:, names.index('f0_hz')] = np.where(voiced, candidate, np.nan)
    values[:, names.index('voiced')] = voiced
    values[:, names.index('missing')] = ~voiced
    jumps = np.zeros(len(values), dtype=bool)
    pair = voiced[1:] & voiced[:-1]
    intervals = np.zeros(max(0, len(values)-1))
    intervals[pair] = abs(1200*np.log2(candidate[1:][pair]/candidate[:-1][pair]))
    jumps[1:] = pair & (abs(intervals-1200) <= 100)
    values[:, names.index('octave_jump')] = jumps
    return replace(out, values=values, metadata={**out.metadata,
                   'code_version': 'vocal-voicing-threshold-v8',
                   'extractor_name': out.metadata['extractor_name'].replace('audio.vocal_f0.voicing-', 'audio.vocal_f0.threshold-v8-'),
                   'voicing_method': 'v7 calibrated joint evidence; direct float64 score threshold; no temporal smoothing',
                   'voicing_rule': 'admissible and clipped probability logit strictly above threshold logit; equality unvoiced; no cumulative DP rounding',
                   'qualification': 'experimental adaptive development; no independent qualification'})
