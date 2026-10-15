"""Reusable, evidence-preserving evaluation for singing-phone posteriors.

This module deliberately evaluates posterior artifacts and CTC output; it does
not choose a recognizer, align supplied labels to audio, or make a claim about
singing recognition generally.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from natural_features.core.feature_types import FeatureSeries

_ROLES = {"blank", "boundary", "special", "phone"}


def _json_array(x: np.ndarray) -> list[float] | None:
    if not x.size or not np.any(np.isfinite(x)):
        return None
    return [float(v) if np.isfinite(v) else None for v in x]


def _roles(labels: Sequence[str], token_roles: Mapping[str, str] | Sequence[str]) -> list[str]:
    if isinstance(token_roles, Mapping):
        missing = [label for label in labels if label not in token_roles]
        extra = set(token_roles) - set(labels)
        if missing or extra:
            raise ValueError("token_roles must specify exactly every posterior column")
        result = [str(token_roles[label]) for label in labels]
    else:
        result = [str(x) for x in token_roles]
        if len(result) != len(labels):
            raise ValueError("token_roles must have one entry per posterior column")
    if any(role not in _ROLES for role in result):
        raise ValueError(f"token_roles must use only {sorted(_ROLES)}")
    if "blank" not in result:
        raise ValueError("token_roles must designate a blank column")
    return result


def _mapping(labels: Sequence[str], roles: Sequence[str], inventory: Mapping[str, str | Sequence[str]]) -> list[list[str] | None]:
    mapped: list[list[str] | None] = []
    for label, role in zip(labels, roles):
        if role != "phone":
            mapped.append(None)
            continue
        raw = inventory.get(label)
        if raw is None:
            mapped.append(None)
            continue
        phones = [str(raw)] if isinstance(raw, str) else [str(x) for x in raw]
        if not phones or any(not phone for phone in phones):
            raise ValueError(f"canonical_inventory[{label!r}] must be a non-empty phone sequence")
        mapped.append(phones)
    return mapped


def native_time_cells(posteriors: FeatureSeries, *, native_hop_s: float | None = None) -> np.ndarray:
    """Return disjoint native integration cells without filling frontend gaps."""
    times = np.asarray(posteriors.times_s, dtype=np.float64)
    if len(times) == 0:
        return np.empty((0, 2), dtype=np.float64)
    if np.any(np.diff(times) <= 0):
        raise ValueError("posterior times_s must be strictly increasing for integration")
    if native_hop_s is None:
        configured = posteriors.timebase.hop_s
        native_hop_s = float(configured) if configured is not None else (float(np.min(np.diff(times))) if len(times) > 1 else None)
    if native_hop_s is None or not np.isfinite(native_hop_s) or native_hop_s <= 0:
        raise ValueError("native_hop_s is required for a single posterior row")
    half = float(native_hop_s) / 2.0
    raw = posteriors.time_bounds_s
    if raw is not None and (
        not np.all(np.isfinite(raw)) or np.any(raw[:, 1] <= raw[:, 0])
    ):
        raise ValueError("explicit posterior supports must be finite and positive")
    observation_lo = float(np.min(raw[:, 0])) if raw is not None else float(times[0] - half)
    observation_hi = float(np.max(raw[:, 1])) if raw is not None else float(times[-1] + half)
    left_mid = np.r_[observation_lo, (times[:-1] + times[1:]) / 2.0]
    right_mid = np.r_[(times[:-1] + times[1:]) / 2.0, observation_hi]
    cells = np.column_stack((np.maximum(times - half, left_mid), np.minimum(times + half, right_mid)))
    # A malformed/very narrow explicit support can make a row empty.  It has no
    # observation weight, rather than borrowing support from a neighbouring row.
    cells[:, 0] = np.maximum(cells[:, 0], observation_lo)
    cells[:, 1] = np.minimum(cells[:, 1], observation_hi)
    if raw is not None:
        cells[:, 0] = np.maximum(cells[:, 0], raw[:, 0])
        cells[:, 1] = np.minimum(cells[:, 1], raw[:, 1])
    if np.any(cells[:, 1] <= cells[:, 0]):
        raise ValueError("posterior supports leave no positive native integration cell")
    return cells


def _bounds(bounds: Sequence[float], name: str) -> np.ndarray:
    out = np.asarray(bounds, dtype=np.float64)
    if out.ndim != 1 or len(out) < 2 or not np.all(np.isfinite(out)) or np.any(np.diff(out) <= 0):
        raise ValueError(f"{name} must be finite, strictly increasing boundaries")
    return out


def _profile_from_mass(mass: np.ndarray) -> np.ndarray:
    total = float(np.sum(mass))
    return mass / total if total > 0 else np.full(len(mass), np.nan)


def _references(
    intervals: Sequence[Mapping[str, Any]], *, nonphone_labels: Sequence[str] | None
) -> list[tuple[float, float, str]]:
    """Validate ordered phone references; non-phone zero durations are ignored."""
    parsed: list[tuple[float, float, str]] = []
    previous_start = -np.inf
    previous_positive_end = -np.inf
    nonphones = set(nonphone_labels or [])
    for raw in intervals:
        if "role" in raw:
            role = str(raw["role"])
        elif str(raw.get("phone", raw.get("label", ""))) in nonphones:
            role = "nonphone"
        else:
            raise ValueError("reference intervals require explicit role or reference_nonphone_labels policy")
        if role not in {"phone", "nonphone"}:
            raise ValueError("reference role must be 'phone' or 'nonphone'")
        start, end = float(raw["start_s"]), float(raw["end_s"])
        if not np.isfinite(start) or not np.isfinite(end) or end < start:
            raise ValueError("reference intervals require finite end_s >= start_s")
        if start < previous_start:
            raise ValueError("reference intervals must be chronological")
        previous_start = start
        if end > start:
            if start < previous_positive_end:
                raise ValueError("positive-duration reference intervals must not overlap")
            previous_positive_end = end
        if role == "nonphone":
            continue
        phone = str(raw["phone"])
        if end <= start:
            raise ValueError("phone reference intervals must have positive duration")
        if phone.startswith("__unmapped_model_token__:"):
            raise ValueError("reference phone uses reserved unmapped-token sentinel")
        parsed.append((start, end, phone))
    return parsed


def _interval_profile(
    intervals: Sequence[tuple[float, float, str]], phones: Sequence[str], bounds: np.ndarray,
    model_phones: set[str],
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Integrate non-overlapping reference phones by exact intersections."""
    phone_i = {phone: i for i, phone in enumerate(phones)}
    output = np.full((len(bounds) - 1, len(phones)), np.nan)
    diagnostics: list[dict[str, float]] = []
    for b, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
        mass = np.zeros(len(phones), dtype=np.float64)
        for start, end, phone in intervals:
            if phone in phone_i:
                mass[phone_i[phone]] += max(0.0, min(hi, end) - max(lo, start))
        output[b] = _profile_from_mass(mass)
        duration = float(mass.sum())
        mapped_duration = sum(
            max(0.0, min(hi, end) - max(lo, start))
            for start, end, phone in intervals if phone in model_phones
        )
        diagnostics.append({"reference_phone_duration_s": duration, "reference_temporal_coverage": duration / (hi - lo), "reference_mapping_coverage": mapped_duration / duration if duration else None})
    return output, diagnostics


def _levenshtein(reference: Sequence[str], predicted: Sequence[str]) -> dict[str, int | float]:
    # Each cell retains S/D/I, so the reported counts are a true edit path.
    table: list[list[tuple[int, int, int, int]]] = [[(j, 0, 0, j) for j in range(len(predicted) + 1)]]
    table += [[(i, 0, i, 0)] + [(0, 0, 0, 0)] * len(predicted) for i in range(1, len(reference) + 1)]
    for i, ref in enumerate(reference, 1):
        for j, pred in enumerate(predicted, 1):
            if ref == pred:
                table[i][j] = table[i - 1][j - 1]
                continue
            sub = table[i - 1][j - 1]
            delete = table[i - 1][j]
            insert = table[i][j - 1]
            options = [(sub[0] + 1, sub[1] + 1, sub[2], sub[3]), (delete[0] + 1, delete[1], delete[2] + 1, delete[3]), (insert[0] + 1, insert[1], insert[2], insert[3] + 1)]
            table[i][j] = min(options, key=lambda x: (x[0], x[1], x[2], x[3]))
    distance, substitutions, deletions, insertions = table[-1][-1]
    return {"substitutions": substitutions, "deletions": deletions, "insertions": insertions, "distance": distance, "per": float(distance / len(reference)) if reference else None}


def greedy_ctc_phone_decode(
    values: np.ndarray,
    labels: Sequence[str],
    roles: Sequence[str],
    mapping: Sequence[list[str] | None],
    *,
    discontinuity_before: np.ndarray | None = None,
) -> list[str]:
    """Free CTC decoding, resetting raw-id collapse at unobserved time gaps."""
    ids = np.argmax(values, axis=1)
    breaks = (
        np.zeros(len(ids), dtype=bool)
        if discontinuity_before is None
        else np.asarray(discontinuity_before, dtype=bool)
    )
    if breaks.shape != ids.shape:
        raise ValueError("discontinuity_before must have one value per posterior row")
    collapsed = [int(token) for row, (token, previous) in enumerate(zip(ids, np.r_[-1, ids[:-1]])) if row == 0 or breaks[row] or token != previous]
    decoded: list[str] = []
    for index in collapsed:
        if roles[index] != "phone":
            continue
        mapped = mapping[index]
        decoded.extend(mapped if mapped is not None else [f"__unmapped_model_token__:{labels[index]}"])
    return decoded


def evaluate_singing_phones(
    posteriors: FeatureSeries,
    *,
    token_roles: Mapping[str, str] | Sequence[str],
    canonical_inventory: Mapping[str, str | Sequence[str]],
    reference_intervals: Sequence[Mapping[str, Any]],
    window_bounds_s: Sequence[float],
    bin_bounds_s: Sequence[float] | None = None,
    native_hop_s: float | None = None,
    reference_nonphone_labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evaluate raw CTC phone posteriors against interval references.

    `window_bounds_s` defines composition windows. `bin_bounds_s` optionally
    defines ordered profile bins; when omitted it equals the windows.
    """
    values = np.asarray(posteriors.values, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("posterior values must be a finite nonnegative 2-D matrix")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=1e-5, atol=1e-7):
        raise ValueError("posterior rows must be normalized probability mass")
    labels = [str(x) for x in posteriors.coords.get("feature", [])]
    if len(labels) != values.shape[1] or len(set(labels)) != len(labels):
        raise ValueError("posterior coords['feature'] must give unique labels for every column")
    roles = _roles(labels, token_roles)
    mapping = _mapping(labels, roles, canonical_inventory)
    parsed_references = _references(reference_intervals, nonphone_labels=reference_nonphone_labels)
    model_phones = set(phone for item in mapping if item is not None for phone in item)
    phones = list(dict.fromkeys(phone for item in mapping if item is not None for phone in item))
    phones.extend(phone for _, _, phone in parsed_references if phone not in phones)
    if not phones:
        raise ValueError("canonical_inventory maps no phone-role token")
    windows = _bounds(window_bounds_s, "window_bounds_s")
    bins = _bounds(bin_bounds_s if bin_bounds_s is not None else window_bounds_s, "bin_bounds_s")
    cells = native_time_cells(posteriors, native_hop_s=native_hop_s)
    role_mass = {role: values[:, [i for i, x in enumerate(roles) if x == role]].sum(axis=1) for role in _ROLES}
    phone_i = {phone: i for i, phone in enumerate(phones)}

    def predicted_profiles(bounds: np.ndarray) -> tuple[np.ndarray, list[dict[str, float | None]]]:
        profiles = np.full((len(bounds) - 1, len(phones)), np.nan)
        diagnostic: list[dict[str, float | None]] = []
        for row, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
            weights = np.maximum(0.0, np.minimum(cells[:, 1], hi) - np.maximum(cells[:, 0], lo))
            category = {role: float(np.dot(weights, role_mass[role])) for role in _ROLES}
            mass = np.zeros(len(phones), dtype=np.float64)
            for col, expanded in enumerate(mapping):
                if expanded is not None:
                    for phone in expanded:
                        mass[phone_i[phone]] += float(np.dot(weights, values[:, col])) / len(expanded)
            profiles[row] = _profile_from_mass(mass)
            category["mapped_phone_mass"] = float(mass.sum())
            category["unmapped_phone_mass"] = category["phone"] - category["mapped_phone_mass"]
            category["observation_s"] = float(weights.sum())
            category["temporal_observation_coverage"] = float(
                weights.sum() / (hi - lo)
            )
            category["mapping_coverage"] = (category["mapped_phone_mass"] / category["phone"] if category["phone"] else None)
            diagnostic.append(category)
        return profiles, diagnostic

    composition, composition_diag = predicted_profiles(windows)
    ordered, ordered_diag = predicted_profiles(bins)
    composition_reference, composition_reference_diag = _interval_profile(parsed_references, phones, windows, model_phones)
    reference, reference_diag = _interval_profile(parsed_references, phones, bins, model_phones)

    def comparisons(predicted: np.ndarray, expected: np.ndarray, diagnostics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for pred, ref, diag in zip(predicted, expected, diagnostics):
            valid = bool(np.all(np.isfinite(pred)) and np.all(np.isfinite(ref)))
            output.append({"total_variation": float(0.5 * np.abs(pred - ref).sum()) if valid else None, "observation_s": diag["observation_s"], "mapping_coverage": diag["mapping_coverage"], "valid": valid})
        return output

    ordered_comparisons = comparisons(ordered, reference, ordered_diag)
    composition_comparisons = comparisons(composition, composition_reference, composition_diag)
    deltas = [float(np.abs((ordered[i] - ordered[i - 1]) - (reference[i] - reference[i - 1])).sum() / 2.0) for i in range(1, len(ordered)) if np.all(np.isfinite(ordered[i])) and np.all(np.isfinite(ordered[i - 1])) and np.all(np.isfinite(reference[i])) and np.all(np.isfinite(reference[i - 1]))]
    reference_sequence = [phone for _, _, phone in parsed_references]
    discontinuity_before = np.zeros(len(cells), dtype=bool)
    if len(cells) > 1:
        discontinuity_before[1:] = cells[1:, 0] > cells[:-1, 1] + 1e-12
    n_discontinuities = int(discontinuity_before.sum())
    decoded = greedy_ctc_phone_decode(
        values, labels, roles, mapping, discontinuity_before=discontinuity_before
    )
    def profile_counts(rows: np.ndarray) -> dict[str, int]:
        valid = sum(_json_array(row) is not None for row in rows)
        return {"valid": valid, "missing": len(rows) - valid}
    return {
        "phones": phones,
        "window_bounds_s": windows.tolist(), "bin_bounds_s": bins.tolist(),
        "composition_profiles": [_json_array(x) for x in composition],
        "reference_composition_profiles": [_json_array(x) for x in composition_reference],
        "ordered_bin_profiles": [_json_array(x) for x in ordered],
        "reference_bin_profiles": [_json_array(x) for x in reference],
        "composition_diagnostics": composition_diag, "ordered_bin_diagnostics": ordered_diag,
        "reference_composition_diagnostics": composition_reference_diag, "reference_bin_diagnostics": reference_diag,
        "profile_comparisons": {"composition": composition_comparisons, "ordered_bins": ordered_comparisons},
        "ordering_sensitivity": {"adjacent_delta_total_variation_mean": float(np.mean(deltas)) if deltas else None, "n_comparable_transitions": len(deltas)},
        "recognition": {"reference_phones": reference_sequence, "decoded_phones": decoded, **_levenshtein(reference_sequence, decoded), "n_discontinuities": n_discontinuities, "full_path_observed": bool(len(values) and n_discontinuities == 0), "interpretation": "continuous supplied CTC path; reference sequence scored as supplied" if len(values) and n_discontinuities == 0 else "partial CTC path: greedy runs reset across native-time gaps; all supplied reference phones remain in PER; no predictions invented inside gaps"},
        "profile_valid": bool(any(x is not None for x in [_json_array(row) for row in ordered])),
        "profile_bin_counts": {"composition_predicted": profile_counts(composition), "composition_reference": profile_counts(composition_reference), "ordered_predicted": profile_counts(ordered), "ordered_reference": profile_counts(reference), "ordered_comparison": {"valid": sum(item["valid"] for item in ordered_comparisons), "missing": sum(not item["valid"] for item in ordered_comparisons)}},
    }
