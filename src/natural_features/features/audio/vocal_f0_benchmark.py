"""Frame-level scoring for vocal F0 estimates against reference annotations."""

from __future__ import annotations

from typing import Any

import numpy as np

from natural_features.core.feature_types import FeatureSeries


_REQUIRED_COLUMNS = (
    "f0_hz",
    "confidence",
    "source_support",
    "source_supported",
    "voiced",
    "missing",
    "range_edge",
    "octave_jump",
)


def _metric(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _reference_arrays(
    reference_times_s: Any, reference_hz: Any
) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(reference_times_s, dtype=np.float64)
    hz = np.asarray(reference_hz, dtype=np.float64)
    if times.ndim != 1 or hz.ndim != 1:
        raise ValueError("reference_times_s and reference_hz must be 1-D")
    if times.shape != hz.shape:
        raise ValueError("reference_times_s and reference_hz must have the same shape")
    if not np.all(np.isfinite(times)):
        raise ValueError("reference_times_s must contain only finite values")
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("reference_times_s must be strictly increasing")
    if np.any(np.isinf(hz)) or np.any(hz < 0):
        raise ValueError("reference_hz must be non-negative finite values or NaN")
    return times, hz


def _estimate_columns(estimate: FeatureSeries) -> tuple[np.ndarray, np.ndarray]:
    names = list(estimate.coords.get("feature", []))
    missing = [name for name in _REQUIRED_COLUMNS if name not in names]
    if missing:
        raise ValueError(f"estimate is missing required feature columns: {missing}")
    if estimate.values.ndim != 2:
        raise ValueError("estimate.values must be a 2-D time-by-feature array")
    if len(names) != estimate.values.shape[1]:
        raise ValueError("estimate feature coordinates must match values columns")
    f0 = np.asarray(estimate.values[:, names.index("f0_hz")], dtype=np.float64)
    voiced = np.asarray(estimate.values[:, names.index("voiced")], dtype=np.float64)
    # A voicing flag without a usable positive F0 cannot support pitch scoring.
    return f0, np.isfinite(voiced) & (voiced > 0.0) & np.isfinite(f0) & (f0 > 0.0)


def _nearest_index(times: np.ndarray, target: float, maximum_error: float) -> int | None:
    right = int(np.searchsorted(times, target, side="left"))
    candidates: list[int] = []
    if right > 0:
        candidates.append(right - 1)
    if right < len(times):
        candidates.append(right)
    if not candidates:
        return None
    # Candidate order preserves the earlier timestamp when distances tie.
    index = min(candidates, key=lambda idx: abs(float(times[idx] - target)))
    if abs(float(times[index] - target)) > maximum_error:
        return None
    return index


def score_vocal_f0(
    reference_times_s: Any,
    reference_hz: Any,
    estimate: FeatureSeries,
    *,
    tolerance_cents: float = 50.0,
    max_time_error_s: float = 0.005001,
) -> dict[str, int | float | None]:
    """Score framewise vocal F0 without interpolating either voicing state.

    NaN reference values are unknown and excluded.  Unmatched voiced references
    remain in recall and raw-pitch-accuracy denominators as misses.
    """

    if not np.isfinite(tolerance_cents) or tolerance_cents < 0:
        raise ValueError("tolerance_cents must be finite and non-negative")
    if not np.isfinite(max_time_error_s) or max_time_error_s < 0:
        raise ValueError("max_time_error_s must be finite and non-negative")
    ref_times, ref_hz = _reference_arrays(reference_times_s, reference_hz)
    est_hz, est_voiced = _estimate_columns(estimate)

    known = ~np.isnan(ref_hz)
    ref_voiced = known & (ref_hz > 0.0)
    ref_unvoiced = known & (ref_hz == 0.0)
    matched = np.zeros(len(ref_times), dtype=bool)
    estimate_indices = np.full(len(ref_times), -1, dtype=np.intp)
    for i in np.flatnonzero(known):
        index = _nearest_index(estimate.times_s, float(ref_times[i]), max_time_error_s)
        if index is not None:
            matched[i] = True
            estimate_indices[i] = index

    n_ref_voiced = int(ref_voiced.sum())
    n_ref_unvoiced = int(ref_unvoiced.sum())
    n_unmatched = int((known & ~matched).sum())
    estimated_voiced = np.zeros(len(ref_times), dtype=bool)
    matched_known = known & matched
    estimated_voiced[matched_known] = est_voiced[estimate_indices[matched_known]]

    joint_voiced = ref_voiced & estimated_voiced
    cents_abs = np.empty(0, dtype=np.float64)
    if np.any(joint_voiced):
        cents_abs = np.abs(
            1200.0
            * np.log2(est_hz[estimate_indices[joint_voiced]] / ref_hz[joint_voiced])
        )
    accurate_voiced = int(np.sum(cents_abs <= tolerance_cents))
    # Include unmatched and predicted-unvoiced reference frames as voiced misses.
    raw_pitch_accuracy = _metric(accurate_voiced, n_ref_voiced)
    voicing_recall = _metric(int(np.sum(ref_voiced & estimated_voiced)), n_ref_voiced)
    voicing_false_alarm_rate = _metric(
        int(np.sum(ref_unvoiced & estimated_voiced)), n_ref_unvoiced
    )
    octave_error_rate = None
    gross_pitch_error_rate = None
    median_abs_cents_error = None
    if cents_abs.size:
        multiples = np.rint(cents_abs / 1200.0)
        octave_errors = (multiples >= 1) & (np.abs(cents_abs - 1200.0 * multiples) <= tolerance_cents)
        octave_error_rate = float(np.mean(octave_errors))
        gross_pitch_error_rate = float(np.mean(cents_abs > tolerance_cents))
        median_abs_cents_error = float(np.median(cents_abs))

    return {
        "n_reference": int(len(ref_times)),
        "n_scored": int(known.sum()),
        "n_reference_voiced": n_ref_voiced,
        "n_reference_unvoiced": n_ref_unvoiced,
        "n_unmatched": n_unmatched,
        "n_joint_voiced": int(joint_voiced.sum()),
        "raw_pitch_accuracy": raw_pitch_accuracy,
        "voicing_recall": voicing_recall,
        "voicing_false_alarm_rate": voicing_false_alarm_rate,
        "median_abs_cents_error": median_abs_cents_error,
        "octave_error_rate": octave_error_rate,
        "gross_pitch_error_rate": gross_pitch_error_rate,
    }
