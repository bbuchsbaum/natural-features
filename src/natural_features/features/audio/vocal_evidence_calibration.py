"""Experimental joint evidence for pitched voicing, separate from presence.

No model is fitted implicitly and no extractor default is changed. Scores refer
to the explicitly supplied benchmark weighting, not population prevalence.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import numpy as np
from scipy.optimize import minimize

from natural_features.features.audio.vocal_f0_benchmark import _nearest_index
from natural_features.features.audio.vocal_voicing import sigmoid
from natural_features.features.audio.vocal_voicing_threshold import threshold_mask
from natural_features.features.common import extractor_metadata


FEATURE_NAMES = ("rmvpe_salience_logit", "mixture_activity_logit", "crepe_confidence_logit",
                 "log_source_rms", "log_source_share", "pitch_disagreement_octaves",
                 "canonical_missing", "activity_missing")
MODEL_SCHEMA = "JointPitchedVoicing/v1"


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _columns(feature, required):
    names = list(feature.coords.get("feature", []))
    if len(names) != len(set(names)) or any(k not in names for k in required):
        raise ValueError("missing or duplicated evidence columns")
    return {k: np.asarray(feature.values[:, names.index(k)], float) for k in required}


def _logit(p):
    p = np.clip(p, 1e-4, 1-1e-4)
    return np.log(p/(1-p))


def evidence(pitch, canonical, activity):
    """Align fixed native evidence; missing inputs are indicators, not negatives."""
    parents = (pitch, canonical, activity)
    if any(f.clock != pitch.clock for f in parents):
        raise ValueError("evidence must share the pitch clock")
    contexts = [f.metadata.get("computation_context", {}) for f in parents]
    try:
        bounds = np.asarray([c.get("input_bounds_s", []) for c in contexts], float)
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("invalid evidence computation contexts") from exc
    if (bounds.shape != (3, 2) or not np.isfinite(bounds).all()
            or np.any(bounds[:, 1] < bounds[:, 0])
            or any(c.get("clock") != str(pitch.clock) for c in contexts)):
        raise ValueError("invalid evidence computation contexts")
    p = _columns(pitch, ("candidate_f0_hz", "confidence", "available"))
    c = _columns(canonical, ("candidate_f0_hz", "confidence", "source_rms", "source_support"))
    a = _columns(activity, ("vocal_activity_score", "available"))
    for scores in (p["confidence"], c["confidence"], a["vocal_activity_score"]):
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError("confidence/activity evidence must be finite in [0,1]")
    for values in (c["source_rms"], c["source_support"]):
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("source diagnostics must be finite and nonnegative")
    if any(not np.isin(v, [0, 1]).all() for v in (p["available"], a["available"])):
        raise ValueError("availability must be binary")
    x = np.zeros((len(pitch.times_s), len(FEATURE_NAMES)), float)
    x[:, 0] = _logit(p["confidence"])
    x[:, 6:] = 1  # Imputed continuous values are 0; explicit indicators identify them.
    for j, time in enumerate(pitch.times_s):
        ci = _nearest_index(canonical.times_s, float(time), .005001)
        if (ci is not None and bounds[1, 0] <= time < bounds[1, 1]
                and np.isfinite(c["candidate_f0_hz"][ci]) and c["candidate_f0_hz"][ci] > 0):
            x[j, 2:5] = [_logit(c["confidence"][ci]),
                         np.log(max(c["source_rms"][ci], 1e-6)),
                         np.log(max(c["source_support"][ci], 1e-8))]
            if np.isfinite(p["candidate_f0_hz"][j]) and p["candidate_f0_hz"][j] > 0:
                x[j, 5] = min(abs(np.log2(p["candidate_f0_hz"][j]/c["candidate_f0_hz"][ci])), 2.)
            x[j, 6] = 0
        ai = _nearest_index(activity.times_s, float(time), 1/140+1e-8)
        if ai is not None and a["available"][ai] > 0 and bounds[2, 0] <= time < bounds[2, 1]:
            x[j, 1] = _logit(a["vocal_activity_score"][ai])
            x[j, 7] = 0
    admissible = ((p["available"] > 0) & (pitch.times_s >= bounds[0, 0])
                  & (pitch.times_s < bounds[0, 1]) & np.isfinite(p["candidate_f0_hz"])
                  & (p["candidate_f0_hz"] >= 50) & (p["candidate_f0_hz"] <= 1000))
    return x, admissible


def objective(theta, x, y, weights, l2):
    z = x @ theta[:-1] + theta[-1]
    residual = weights * (sigmoid(z)-y)
    return (float(weights @ (np.logaddexp(0., z)-y*z) + .5*l2*np.dot(theta[:-1], theta[:-1])),
            np.r_[x.T @ residual + l2*theta[:-1], residual.sum()])


def fit_model(x, y, weights, *, l2=.01):
    """Fit one fixed ridge-logistic model using supplied training observations."""
    x, y, weights = np.asarray(x, float), np.asarray(y, float), np.asarray(weights, float)
    if (x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or y.shape != (len(x),)
            or weights.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all()
            or not np.isfinite(weights).all() or np.any((y < 0) | (y > 1))
            or np.any(weights < 0) or weights.sum() <= 0 or not np.isfinite(l2) or l2 <= 0):
        raise ValueError("invalid training arrays or regularization")
    keep = weights > 0
    x, y, weights = x[keep], y[keep], weights[keep]/weights[keep].sum()
    prevalence = float(weights @ y)
    if not 0 < prevalence < 1:
        raise ValueError("both pitched-voicing classes are required")
    mean = weights @ x
    variance = weights @ ((x-mean)**2)
    scale = np.where(np.ptp(x, axis=0) == 0, 1., np.sqrt(variance))
    if np.any(scale <= 0):
        raise ValueError("nonconstant training feature has zero numerical variance")
    normalized = (x-mean)/scale
    result = minimize(objective, np.zeros(x.shape[1]+1), args=(normalized, y, weights, l2),
                      method="L-BFGS-B", jac=True,
                      options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8})
    if not result.success or not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise ValueError("joint voicing fit did not converge: " + str(result.message))
    return {"schema": MODEL_SCHEMA, "feature_names": list(FEATURE_NAMES), "l2": l2,
            "mean": mean.tolist(), "scale": scale.tolist(), "coefficients": result.x[:-1].tolist(),
            "intercept": float(result.x[-1]), "training_prevalence": prevalence,
            "loss": float(result.fun), "iterations": int(result.nit)}


def predict(x, model):
    if model.get("schema") != MODEL_SCHEMA or model.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("incompatible joint voicing model")
    x = np.asarray(x, float)
    mean, scale, beta = [np.asarray(model[k], float) for k in ("mean", "scale", "coefficients")]
    if (x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not np.isfinite(x).all()
            or any(v.shape != (len(FEATURE_NAMES),) or not np.isfinite(v).all() for v in (mean, scale, beta))
            or np.any(scale <= 0) or not np.isfinite(model["intercept"])):
        raise ValueError("invalid joint model parameters or observations")
    return sigmoid(((x-mean)/scale) @ beta + model["intercept"])


def apply_scores(pitch, canonical, activity, scores, threshold, *, model_identity, score_kind):
    """Retain RMVPE candidates and publish an explicit pitched-voicing score."""
    if score_kind not in ("rmvpe_salience", "joint_logistic"):
        raise ValueError("explicit raw or fitted score semantics required")
    _, admissible = evidence(pitch, canonical, activity)
    scores = np.asarray(scores, float)
    if scores.shape != admissible.shape:
        raise ValueError("score length differs from native pitch")
    accepted = threshold_mask(scores, admissible, threshold)
    names = list(pitch.coords["feature"])
    values = np.array(pitch.values, copy=True)
    candidate = values[:, names.index("candidate_f0_hz")]
    values[:, names.index("f0_hz")] = np.where(accepted, candidate, np.nan)
    values[:, names.index("voiced")] = accepted
    values[:, names.index("missing")] = ~accepted
    values[:, names.index("octave_jump")] = 0
    with np.errstate(invalid="ignore", divide="ignore"):
        delta = abs(1200*np.diff(np.log2(values[:, names.index("f0_hz")])))
    values[1:, names.index("octave_jump")] = accepted[1:] & accepted[:-1] & (abs(delta-1200) <= 100)
    parents = [p.metadata for p in (pitch, canonical, activity)]
    bounds = np.asarray([p["computation_context"]["input_bounds_s"] for p in parents])
    union = [float(bounds[:, 0].min()), float(bounds[:, 1].max())]
    metadata = extractor_metadata("audio.vocal_f0."+("joint_evidence" if score_kind == "joint_logistic" else "salience_threshold"), code_version="joint-evidence-v1",
        params={"threshold": threshold, "model_identity": model_identity, "score_kind": score_kind,
                "parent_identities": [{k: p.get(k) for k in ("extractor_id", "code_version", "model_revision", "params_hash")} for p in parents]},
        model_revision="joint:"+model_identity, extra={"qualification": "experimental; not qualified",
            "source_metadata": parents, "phonetic_support": "not defined or modified",
            "confidence_semantics": "retained raw maximum RMVPE salience",
            "pitched_voicing_score_semantics": ("raw maximum RMVPE salience; uncalibrated" if score_kind == "rmvpe_salience" else "ridge-logistic score under declared benchmark weights; population calibration unestablished"),
            "score_target": "pitched voicing conditional on an available in-range pitch candidate",
            "computation_context": {"schema": "MusicComputationContext/v1", "clock": str(pitch.clock),
                "policy": "bidirectional", "input_bounds_s": union, "dependency_bounds_s": [union],
                "preprocessing": {"operation": "joint pitched-voicing evidence; union of native parent inputs"}}})
    if "pitched_voicing_score" in names:
        values[:, names.index("pitched_voicing_score")] = scores
    else:
        values = np.column_stack([values, scores])
        names.append("pitched_voicing_score")
    return replace(pitch, values=values, coords={**pitch.coords, "feature": names}, metadata=metadata)
