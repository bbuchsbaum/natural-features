"""Mapping-by-pooling sensitivity comparison for retained CTC posteriors."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from natural_features.core.feature_types import FeatureSeries
from natural_features.features.speech.phone_inventory import PhoneDecision, PhoneInventory
from natural_features.features.speech.singing_benchmark import _levenshtein, native_time_cells


def _bounds(values: Sequence[float]) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    if out.ndim != 1 or len(out) < 2 or not np.all(np.isfinite(out)) or np.any(np.diff(out) <= 0):
        raise ValueError("bounds must be finite and strictly increasing")
    return out


def _profile(mass: np.ndarray) -> list[float] | None:
    return (mass / mass.sum()).tolist() if mass.sum() > 0 else None


def _tv(a: list[float] | None, b: list[float] | None) -> float | None:
    return None if a is None or b is None else float(np.abs(np.asarray(a) - np.asarray(b)).sum() / 2)


def _references(rows: Sequence[Mapping[str, Any]]) -> list[tuple[float, float, str]]:
    output, last_end = [], -np.inf
    for row in rows:
        if "role" not in row:
            raise ValueError("reference intervals require explicit role")
        role, lo, hi = str(row["role"]), float(row["start_s"]), float(row["end_s"])
        if role not in {"phone", "nonphone"} or not np.isfinite(lo) or not np.isfinite(hi) or hi < lo:
            raise ValueError("reference intervals need finite chronological roles and bounds")
        if lo < last_end:
            raise ValueError("reference intervals must not overlap")
        last_end = hi
        if role == "phone":
            if hi <= lo:
                raise ValueError("phone reference intervals need positive duration")
            output.append((lo, hi, str(row["phone"])))
    return output


def _legacy(label: str, role: str, mapping: Mapping[str, str | Sequence[str]]) -> PhoneDecision:
    value = mapping.get(label) if role == "phone" else None
    phones = () if value is None else ((str(value),) if isinstance(value, str) else tuple(str(x) for x in value))
    return PhoneDecision(label, label, phones, "sequence" if len(phones) > 1 else ("exact" if phones else "unsupported"), "Frozen v1 legacy mapping.", "v1 manifest")


def _events(values: np.ndarray, roles: Sequence[str], cells: np.ndarray) -> list[tuple[int, int]]:
    ids, out = np.argmax(values, axis=1), []
    gaps = np.zeros(len(ids), dtype=bool)
    if len(ids) > 1:
        gaps[1:] = cells[1:, 0] > cells[:-1, 1] + 1e-12
    for row, token in enumerate(ids):
        if row and not gaps[row] and token == ids[row - 1]:
            continue  # CTC collapse is on raw IDs before role filtering.
        if roles[token] == "phone":
            out.append((row, int(token)))
    return out


def _reference_profiles(refs: Sequence[tuple[float, float, str]], phones: list[str], bounds: np.ndarray) -> tuple[list[list[float] | None], list[list[float] | None]]:
    index = {phone: i for i, phone in enumerate(phones)}
    duration, occurrence = [], []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        d, o = np.zeros(len(phones)), np.zeros(len(phones))
        for start, end, phone in refs:
            if phone in index:
                d[index[phone]] += max(0., min(hi, end) - max(lo, start))
                if lo <= start < hi:
                    o[index[phone]] += 1.
        duration.append(_profile(d))
        occurrence.append(_profile(o))
    return duration, occurrence


def compare_singing_representations(posteriors: FeatureSeries, *, token_roles: Mapping[str, str], conservative_inventory: Mapping[str, str | Sequence[str]], inventory_v2: PhoneInventory, reference_intervals: Sequence[Mapping[str, Any]], window_bounds_s: Sequence[float], bin_bounds_s: Sequence[float], native_hop_s: float | None = None) -> dict[str, Any]:
    values = np.asarray(posteriors.values, dtype=float)
    labels = [str(x) for x in posteriors.coords.get("feature", ())]
    if values.ndim != 2 or len(labels) != values.shape[1] or set(labels) != set(token_roles) or np.any(values < 0) or not np.allclose(values.sum(axis=1), 1., rtol=1e-5, atol=1e-7):
        raise ValueError("normalized posteriors and exact token roles are required")
    roles = [str(token_roles[x]) for x in labels]
    if set(roles) - {"blank", "boundary", "special", "phone"} or "blank" not in roles:
        raise ValueError("invalid token roles")
    cells, refs = native_time_cells(posteriors, native_hop_s=native_hop_s), _references(reference_intervals)
    mappings = {"v1_legacy_exact": [_legacy(x, r, conservative_inventory) for x, r in zip(labels, roles)], "v2_english_decisions": [inventory_v2.decide(x) if r == "phone" else _legacy(x, r, {}) for x, r in zip(labels, roles)]}
    phones = list(dict.fromkeys(phone for decisions in mappings.values() for decision in decisions for phone in decision.phones))
    phones.extend(phone for _, _, phone in refs if phone not in phones)
    index, events = {phone: i for i, phone in enumerate(phones)}, _events(values, roles, cells)
    widths = cells[:, 1] - cells[:, 0]
    token_audit = []
    for col, label in enumerate(labels):
        item = {"token": label, "role": roles[col], "native_time_posterior_mass": float(np.dot(widths, values[:, col])), "greedy_event_count": int(sum(token == col for _, token in events)), "mappings": {}}
        for name, decisions in mappings.items():
            d = decisions[col]
            item["mappings"][name] = {"relation": d.relation, "phones": list(d.phones), "alternatives": [list(x) for x in d.alternatives], "reason": d.reason, "source": d.source, "canonical_kind": d.canonical_kind, "mapped": d.mapped}
        token_audit.append(item)

    def pooled(bounds: np.ndarray) -> dict[str, Any]:
        duration, occurrence = _reference_profiles(refs, phones, bounds)
        result: dict[str, Any] = {"reference_duration_composition": duration, "reference_occurrence_composition": occurrence, "mappings": {}}
        for name, decisions in mappings.items():
            raw, conditional, counted, event_totals, coverage = [], [], [], [], []
            for lo, hi in zip(bounds[:-1], bounds[1:]):
                weights = np.maximum(0., np.minimum(cells[:, 1], hi) - np.maximum(cells[:, 0], lo))
                r, c, e = np.zeros(len(phones)), np.zeros(len(phones)), np.zeros(len(phones))
                phone_probability = values[:, [i for i, role in enumerate(roles) if role == "phone"]].sum(axis=1)
                for col, decision in enumerate(decisions):
                    if not decision.mapped:
                        continue
                    ids = [index[p] for p in decision.phones]
                    np.add.at(r, ids, float(np.dot(weights, values[:, col])) / len(ids))
                    for frame, weight in enumerate(weights):
                        if weight and phone_probability[frame]:
                            np.add.at(c, ids, weight * values[frame, col] / phone_probability[frame] / len(ids))
                for frame, col in events:
                    if lo <= posteriors.times_s[frame] < hi and decisions[col].mapped:
                        # Events retain each explicit compound component once.
                        np.add.at(e, [index[p] for p in decisions[col].phones], 1.)
                raw.append(_profile(r))
                conditional.append(_profile(c))
                counted.append(_profile(e))
                event_totals.append(float(e.sum()))
                phone_mass = float(np.dot(weights, phone_probability))
                mapped_mass = float(sum(np.dot(weights, values[:, i]) for i, d in enumerate(decisions) if d.mapped))
                conditional_time = float(weights[phone_probability > 0].sum())
                conditional_mapped_mass = float(sum(weight * values[frame, col] / phone_probability[frame] for frame, weight in enumerate(weights) if weight and phone_probability[frame] for col, decision in enumerate(decisions) if decision.mapped))
                bin_events = [col for frame, col in events if lo <= posteriors.times_s[frame] < hi]
                mapped_events = [col for col in bin_events if decisions[col].mapped]
                coverage.append({"observation_s": float(weights.sum()), "phone_mass": phone_mass, "mapped_phone_mass": mapped_mass, "unmapped_or_ambiguous_phone_mass": phone_mass - mapped_mass, "mapping_coverage": mapped_mass / phone_mass if phone_mass else None, "conditional_observed_phone_time_s": conditional_time, "conditional_mapped_mass": conditional_mapped_mass, "conditional_mapping_coverage": conditional_mapped_mass / conditional_time if conditional_time else None, "raw_phone_event_count": len(bin_events), "mapped_raw_phone_event_count": len(mapped_events), "unmapped_or_ambiguous_raw_phone_event_count": len(bin_events) - len(mapped_events), "expanded_canonical_event_count": int(sum(len(decisions[col].phones) for col in mapped_events)), "raw_missing": raw[-1] is None, "conditional_missing": conditional[-1] is None, "event_missing": counted[-1] is None})
            reps = {"raw_posterior_integral": raw, "framewise_phone_conditional": conditional, "greedy_ctc_token_events": counted}
            result["mappings"][name] = {"coverage": coverage, "representations": {key: {"profiles": profiles, "comparisons": [{"duration_total_variation": _tv(profile, d), "occurrence_total_variation": _tv(profile, o)} for profile, d, o in zip(profiles, duration, occurrence)]} for key, profiles in reps.items()}}
            result["mappings"][name]["representations"]["greedy_ctc_token_events"]["event_totals"] = event_totals
        return result

    reference_sequence = [phone for _, _, phone in refs]
    recognition = {}
    for name, decisions in mappings.items():
        decoded = [phone for _, col in events for phone in (decisions[col].phones if decisions[col].mapped else (f"__unmapped_model_token__:{labels[col]}",))]
        recognition[name] = {"reference_phone_count": len(reference_sequence), "decoded_phone_count": len(decoded), **_levenshtein(reference_sequence, decoded), "interpretation": "Greedy CTC token sequence; native gaps reset raw-ID collapse."}
    return {"phones": phones, "mapping_by_pooling": True, "token_audit": token_audit, "recognition": recognition, "composition": pooled(_bounds(window_bounds_s)), "ordered_bins": pooled(_bounds(bin_bounds_s)), "missingness": "Null profiles are retained when a representation or reference composition has zero canonical mass; gaps are never imputed.", "units": {"raw_posterior_integral": "integrated posterior probability mass, not duration", "framewise_phone_conditional": "phone-conditional posterior mass, not duration", "greedy_ctc_token_events": "collapsed token-event occurrence count, not duration"}}
