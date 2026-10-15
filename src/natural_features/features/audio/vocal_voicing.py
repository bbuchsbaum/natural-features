"""Experimental learned voicing for retained vocal-F0 candidates.

This explicit postprocessor does not change the extractor's public defaults.
Its probabilities refer to the benchmark fitting mixture, not arbitrary music.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import numpy as np

from ...core.feature_types import FeatureSeries


FEATURE_NAMES = ("confidence_logit", "log_source_share", "log_source_rms",
                 "local_agreement", "local_available")
MODEL_SCHEMA = "VocalVoicingModel/v7"
NEURAL_REVISION = ("demucs:sha256:8726e21a993978c7ba086d3872e7608d7d5bfca646ca4aca459ffda844faa8b4;"
                   "crepe:sha256:133225604dedd2e4005f8bbd1bd0a2ec073ba8b7a6cd31ff6d5edbbfa3539986")


def sigmoid(logit):
    return np.exp(-np.logaddexp(0., -np.asarray(logit, dtype=np.float64)))


def evidence(feature: FeatureSeries):
    """Return the fixed five predictors and physically admissible native rows."""
    names = list(feature.coords.get("feature", []))
    required = ("confidence", "source_support", "source_rms", "local_periodicity",
                "local_supported", "candidate_f0_hz")
    if any(name not in names for name in required) or len(names) != len(set(names)):
        raise ValueError("learned voicing requires unique native vocal-F0 diagnostics")
    columns = {name: np.asarray(feature.values[:, names.index(name)], dtype=np.float64)
               for name in required}
    confidence, share, rms, q, available, f0 = (columns[name] for name in required)
    if (not np.all(np.isfinite(confidence)) or np.any((confidence < 0) | (confidence > 1))
            or not np.all(np.isfinite(share)) or np.any(share < 0)
            or not np.all(np.isfinite(rms)) or np.any(rms < 0)
            or not np.all(np.isin(available, [0., 1.]))):
        raise ValueError("invalid confidence, energy or availability diagnostics")
    if np.any((available > 0) & (~np.isfinite(q) | (q < -1) | (q > 1))):
        raise ValueError("available local evidence must be finite in [-1,1]")
    bounds = feature.metadata.get("computation_context", {}).get("input_bounds_s")
    if not isinstance(bounds, list) or len(bounds) != 2 or not np.all(np.isfinite(bounds)) or bounds[1] < bounds[0]:
        raise ValueError("learned voicing requires finite real input bounds")
    hop = feature.metadata.get("effective_hop_s")
    if hop != .01 or (len(feature.times_s) > 1 and not np.allclose(np.diff(feature.times_s), hop, rtol=0, atol=1e-12)):
        raise ValueError("v7 model requires the fixed 10ms native grid")
    c = np.clip(confidence, 1e-4, 1 - 1e-4)
    x = np.column_stack([np.log(c / (1-c)), np.log(np.maximum(share, 1e-8)),
                         np.log(np.maximum(rms, 1e-6)), np.where(available > 0, q, 0.), available])
    admissible = ((feature.times_s >= bounds[0]) & (feature.times_s < bounds[1])
                  & np.isfinite(f0) & (f0 > 0) & (rms > 0))
    return x, admissible


def predict_probability(x, model):
    """Apply training-only standardization, logistic fit, and Platt calibration."""
    if model.get("schema") != MODEL_SCHEMA or model.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("unsupported learned voicing model schema")
    x = np.asarray(x, dtype=np.float64)
    mean, scale, beta = (np.asarray(model[key], dtype=np.float64) for key in ("mean", "scale", "coefficients"))
    if (x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not np.all(np.isfinite(x))
            or any(a.shape != (len(FEATURE_NAMES),) or not np.all(np.isfinite(a)) for a in (mean, scale, beta))
            or np.any(scale <= 0)):
        raise ValueError("invalid learned voicing predictors or model arrays")
    intercept = float(model["intercept"])
    calibration = model["calibration"]
    slope, offset = float(calibration["slope"]), float(calibration["intercept"])
    if not np.all(np.isfinite([intercept, slope, offset])) or slope < 0:
        raise ValueError("invalid voicing intercept or monotone calibration")
    return sigmoid(slope * (((x - mean) / scale) @ beta + intercept) + offset)


def decode_states(probability, admissible, threshold, switch_penalty):
    """Maximum-score U/V path; ties prefer U, with no fabricated pitch values.

    U emission is zero and V emission is logit(p)-logit(threshold).
    Each state switch subtracts switch_penalty; initial states have equal prior.
    Inadmissible rows force U. This path is not a calibrated state posterior.
    """
    p = np.asarray(probability, dtype=np.float64)
    valid = np.asarray(admissible, dtype=bool)
    if p.ndim != 1 or valid.shape != p.shape or not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("invalid voicing probabilities or admissibility")
    if not np.isfinite(threshold) or not 0 < threshold < 1 or not np.isfinite(switch_penalty) or switch_penalty < 0:
        raise ValueError("invalid voicing threshold or switch penalty")
    if not len(p):
        return valid.copy()
    p = np.clip(p, 1e-12, 1 - 1e-12)
    emission = np.log(p / (1-p)) - np.log(threshold / (1-threshold))
    emission[~valid] = -np.inf
    previous = np.array([0., emission[0]])
    predecessors = np.zeros((len(p), 2), dtype=np.int8)
    for i in range(1, len(p)):
        # Strict > gives predecessor U every tie, for both destination states.
        to_u = int(previous[1] - switch_penalty > previous[0])
        to_v = int(previous[1] > previous[0] - switch_penalty)
        predecessors[i] = [to_u, to_v]
        previous = np.array([previous[to_u] - (switch_penalty if to_u else 0.),
                             previous[to_v] - (0. if to_v else switch_penalty) + emission[i]])
    state = int(previous[1] > previous[0])
    path = np.zeros(len(p), dtype=bool)
    for i in range(len(p)-1, -1, -1):
        path[i] = bool(state)
        state = int(predecessors[i, state])
    return path


def apply_voicing_model(feature: FeatureSeries, profile: dict) -> FeatureSeries:
    """Replace acceptance using an explicit fitted model and frozen operating point."""
    x, admissible = evidence(feature)
    model = profile["model"]
    if model.get("model_revision") != NEURAL_REVISION or feature.metadata.get("model_revision") != NEURAL_REVISION:
        raise ValueError("voicing profile and native neural-model identities differ")
    native_identity = [feature.metadata.get("code_version"), feature.metadata.get("params_hash")]
    if native_identity not in model.get("native_frontends", []):
        raise ValueError("native frontend parameters are incompatible with voicing profile")
    probability = predict_probability(x, model)
    voiced = decode_states(probability, admissible, profile["threshold"], profile["switch_penalty"])
    names = list(feature.coords["feature"])
    values = np.array(feature.values, dtype=np.float64, copy=True)
    candidate = values[:, names.index("candidate_f0_hz")]
    values[:, names.index("f0_hz")] = np.where(voiced, candidate, np.nan)
    values[:, names.index("voiced")] = voiced
    values[:, names.index("missing")] = ~voiced
    jumps = np.zeros(len(values), dtype=bool)
    positive_pair = np.isfinite(candidate[1:]) & np.isfinite(candidate[:-1]) & (candidate[1:] > 0) & (candidate[:-1] > 0)
    intervals = np.zeros(max(0, len(values)-1))
    intervals[positive_pair] = abs(1200 * np.log2(candidate[1:][positive_pair] / candidate[:-1][positive_pair]))
    jumps[1:] = voiced[1:] & voiced[:-1] & positive_pair & (abs(intervals - 1200) <= 100)
    values[:, names.index("octave_jump")] = jumps
    for name, column in (("voicing_probability", probability), ("voicing_admissible", admissible)):
        if name in names:
            values[:, names.index(name)] = column
        else:
            names.append(name)
            values = np.column_stack([values, column])
    identity = hashlib.sha256(json.dumps(profile, sort_keys=True, allow_nan=False).encode()).hexdigest()
    md = {**feature.metadata, "code_version": "vocal-voicing-v7", "voicing_profile_sha256": identity,
          "native_frontend_identity": native_identity,
          "extractor_name": f"audio.vocal_f0.voicing-{identity[:20]}",
          "params_hash": identity, "voicing_method": "calibrated ridge logistic with explicit U/V maximum-score path",
          "voicing_probability_semantics": "pre-decoder fitted probability under the benchmark weighting; not a population or sequence-posterior guarantee",
          "voicing_admissible_semantics": "real native center, finite positive candidate, and strictly positive source RMS; no positive-amplitude cutoff",
          "legacy_gate_diagnostics": "framewise_voiced/source_supported retain the old heuristic diagnostics and no longer define final acceptance",
          "legacy_voicing_rule": feature.metadata.get("voicing_rule"),
          "voicing_rule": "fitted and calibrated joint evidence followed by explicit U/V decoding; old confidence/source/local/temporal gates do not constrain candidate recovery",
          "qualification": "experimental; development calibration is not independent qualification"}
    return replace(feature, values=values, coords={**feature.coords, "feature": names}, metadata=md)
