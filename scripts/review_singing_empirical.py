#!/usr/bin/env python3
"""Read-only independent artifact rescoring. Requires Python + numpy.
Do not use Python -O: assertions are the audit checks.
--complete requires 90 vocal cases, 32 phone cases, and aggregate vocal gates.
Analytic aggregates are checked from retained summaries, not rerun inference.
"""

import argparse, functools, hashlib, json, pathlib, subprocess  # noqa: E401
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check(actual, expected, where=""):
    if isinstance(expected, dict):
        for key, value in expected.items():
            check(actual[key], value, f"{where}.{key}")
    elif expected is None or isinstance(expected, (str, bool)):
        assert actual == expected, (where, actual, expected)
    elif isinstance(expected, (list, tuple, np.ndarray)):
        assert actual is not None and len(actual) == len(expected), where
        for i, (a, b) in enumerate(zip(actual, expected)):
            check(a, b, f"{where}[{i}]")
    elif isinstance(expected, (int, np.integer)):
        assert actual == expected, (where, actual, expected)
    else:
        assert actual is not None and np.isclose(
            actual, expected, rtol=1e-12, atol=1e-12
        ), (where, actual, expected)


def ratio(a, b):
    return float(a / b) if b else None


def audit_vocal(args):
    root = args.root
    report_path = args.vocal_report
    raw = report_path.read_bytes()
    r = json.loads(raw)
    sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()  # noqa: E731
    p = root / "tests/benchmarks/thresholds/vocal_backend_comparison_v1.json"
    assert r["protocol_sha256"] == sha(p)
    for f, h in r["protocol"]["frozen_sources"].items():
        assert sha(root / f) == h, f
    inherited = r["protocol"]["consumed_protocol"]
    assert sha(root / inherited["path"]) == inherited["sha256"]
    protocol = json.loads((root / inherited["path"]).read_text())
    for x in protocol["input_reports"]:
        assert sha(root / x["path"]) == x["sha256"]

    def load(a):
        assert sha(a["path"]) == a["sha256"], a["path"]
        with np.load(a["path"], allow_pickle=False) as n:
            assert str(n["cache_key"].item()) == a["key"]
            return (
                n["times_s"],
                n["values"],
                json.loads(str(n["feature_names_json"].item())),
                json.loads(str(n["source_metadata_json"].item())),
            )

    def nearest(t, query, tol):
        # Full distance matrix, with earliest chronological index winning ties.
        d = abs(query[:, None] - t[None, :])
        ix = d.argmin(axis=1)
        return ix, d[np.arange(len(query)), ix] <= tol

    def ratio(x, n):
        return float(x / n) if n else None

    def compare(actual, expected, where):
        for k, v in expected.items():
            if v is None:
                assert actual[k] is None, (where, k, actual[k], v)
            elif isinstance(v, (int, np.integer)):
                assert actual[k] == v, (where, k, actual[k], v)
            else:
                assert np.isclose(actual[k], v, rtol=1e-12, atol=1e-12), (
                    where,
                    k,
                    actual[k],
                    v,
                )

    n_artifacts = n_pitch = n_activity = n_masks = 0
    for row in r["rows"]:
        a = row["provenance"]["artifact"]
        assert sha(a["artifact"]) == a["artifact_sha256"]
        with np.load(a["artifact"], allow_pickle=False) as orig:
            ot = orig["times_s"]
            meta = json.loads(str(orig["source_metadata_json"].item()))
            bounds = meta["computation_context"]["input_bounds_s"]
            dur = bounds[1] - bounds[0]
        annotation = (
            args.vocal_assets / "vocadito/Annotations/F0"
        ) / f"vocadito_{row['excerpt']}_f0.csv"
        assert sha(annotation) == row["provenance"]["input_hashes"]["reference_sha256"]
        refs = np.loadtxt(annotation, delimiter=",")
        refs = refs[refs[:, 0] < dur]
        q, h = (
            (ot[(ot >= 0) & (ot < dur)], np.zeros(sum((ot >= 0) & (ot < dur))))
            if row["condition"] in ("backing", "silence")
            else (refs[:, 0], refs[:, 1])
        )
        artifacts = {k: load(a) for k, a in row["artifacts"].items()}
        n_artifacts += len(artifacts)
        for arm, actual in row["metrics"].items():
            t, v, n, _ = artifacts[arm]
            ix, match = nearest(t, q, 0.005001)
            known = ~np.isnan(h)
            pos = known & (h > 0)
            neg = known & (h == 0)
            ph = v[ix, n.index("f0_hz")]
            voice = v[ix, n.index("voiced")]
            voiced = (
                known
                & match
                & np.isfinite(ph)
                & (ph > 0)
                & np.isfinite(voice)
                & (voice > 0)
            )
            cents = np.asarray(
                [
                    abs(1200 * np.log2(a / b))
                    for a, b in zip(ph[pos & voiced], h[pos & voiced])
                ]
            )
            octave = [
                min(abs(x - 1200 * k) for k in range(1, max(2, int(x / 1200) + 2)))
                <= 50
                for x in cents
            ]
            expected = dict(
                n_reference=len(q),
                n_scored=int(known.sum()),
                n_reference_voiced=int(pos.sum()),
                n_reference_unvoiced=int(neg.sum()),
                n_unmatched=int((known & ~match).sum()),
                n_joint_voiced=int((pos & voiced).sum()),
                raw_pitch_accuracy=ratio(sum(cents <= 50), sum(pos)),
                voicing_recall=ratio(sum(pos & voiced), sum(pos)),
                voicing_false_alarm_rate=ratio(sum(neg & voiced), sum(neg)),
                median_abs_cents_error=float(np.median(cents)) if len(cents) else None,
                octave_error_rate=ratio(sum(octave), len(cents)),
                gross_pitch_error_rate=ratio(sum(cents > 50), len(cents)),
            )
            compare(actual, expected, (row["key"], arm))
            n_pitch += 1
            if "__" in arm:
                pitch, activity = arm.split("__")
                pt, pv, pn, _ = artifacts[pitch]
                at, av, an, _ = artifacts[activity]
                assert np.array_equal(t, pt)
                ai, am = nearest(at, pt, 1 / 140 + 1e-8)
                accepted = (
                    (pv[:, pn.index("voiced")] > 0)
                    & am
                    & (av[ai, an.index("available")] > 0)
                    & (av[ai, an.index("vocal_activity_score")] > 0.51)
                )
                assert np.array_equal(v[:, n.index("voiced")] > 0, accepted)
                assert np.array_equal(
                    v[:, n.index("f0_hz")][accepted], pv[:, pn.index("f0_hz")][accepted]
                )
                assert np.isnan(v[:, n.index("f0_hz")][~accepted]).all()
                n_masks += 1
        for arm, actual in row["activity_partial_reference"].items():
            t, v, n, _ = artifacts[arm]
            aq, truth = (
                (t[t < dur], np.zeros(sum(t < dur)))
                if row["condition"] in ("backing", "silence")
                else (refs[:, 0], np.where(refs[:, 1] > 0, 1, np.nan))
            )
            ix, match = nearest(t, aq, 1 / 140 + 1e-8)
            known = ~np.isnan(truth)
            match &= known & (v[ix, n.index("available")] > 0)
            pred = match & (v[ix, n.index("vocal_activity_score")] > 0.51)
            pos = truth == 1
            neg = truth == 0
            tp = int(sum(pos & pred))
            fp = int(sum(neg & pred))
            fn = int(sum(pos)) - tp
            tn = int(sum(neg & match & ~pred))
            expected = dict(
                n_known=int(sum(known)),
                n_unknown=int(sum(~known)),
                n_positive=int(sum(pos)),
                n_negative=int(sum(neg)),
                n_matched=int(sum(match)),
                n_unmatched=int(sum(known & ~match)),
                true_positive=tp,
                false_positive=fp,
                false_negative=fn,
                true_negative=tn,
                coverage=ratio(sum(match), sum(known)),
                positive_coverage=ratio(sum(pos & match), sum(pos)),
                negative_coverage=ratio(sum(neg & match), sum(neg)),
                recall=ratio(tp, sum(pos)),
                false_alarm_rate=ratio(fp, sum(neg)),
                precision=ratio(tp, tp + fp),
                f1=ratio(2 * tp, 2 * tp + fp + fn),
                accuracy_with_abstentions_as_errors=ratio(tp + tn, sum(known)),
            )
            compare(actual, expected, (row["key"], arm))
            n_activity += 1

    assert not r["errors"]
    assert r["qualified"] is False and r["state"] == "development"
    expected_arms = {
        p + suffix
        for p in ("canonical", "rmvpe_mixture", "rmvpe_stem")
        for suffix in ("", "__activity_mixture", "__activity_stem")
    }
    conds = ("solo", "mixture_0db", "mixture_minus6db", "backing", "silence")
    expected_keys = {
        f"{g}:{e}:{c}"
        for g, s in protocol["groups"].items()
        for e in s["excerpt_ids"]
        for c in conds
    }
    actual_keys = [x["key"] for x in r["rows"]]
    assert (
        len(actual_keys) == len(set(actual_keys)) and set(actual_keys) <= expected_keys
    )
    for row in r["rows"]:
        assert set(row["metrics"]) == expected_arms
        assert set(row["activity_partial_reference"]) == {
            "activity_mixture",
            "activity_stem",
        }
    gates = {
        "analytic_median_abs_cents_error_max": 50.0,
        "backing_false_alarm_rate_max": 0.05,
        "median_abs_cents_error_max": 50.0,
        "octave_error_rate_max": 0.05,
        "raw_pitch_accuracy_min": 0.8,
        "silence_false_alarm_rate_max": 0.0,
        "voicing_false_alarm_rate_max": 0.1,
        "voicing_recall_min": 0.8,
    }
    assert protocol["gates"] == gates
    gate_count = 0
    if args.complete or "variants" in r:
        assert (
            len(r["rows"]) == 90
            and set(actual_keys) == expected_keys
            and r["completed_cases"] == 90
        )
        assert set(r["variants"]) == expected_arms and r["adopted_model"] is None
        metrics = (
            "raw_pitch_accuracy",
            "voicing_recall",
            "voicing_false_alarm_rate",
            "median_abs_cents_error",
            "octave_error_rate",
            "gross_pitch_error_rate",
        )
        controls = []
        for group, source in protocol["analytic_reports"].items():
            assert sha(root / source["path"]) == source["sha256"]
            gold = json.loads((root / source["path"]).read_text())["analytic"]
            assert r["canonical_analytic"][group]["result"] == gold
            controls.append(gold)
        assert all(x == controls[0] for x in controls)
        for arm, variant in r["variants"].items():
            grouped = {g["name"]: g for g in variant["groups"]}
            assert set(grouped) == set(protocol["groups"])
            passed = []
            for group in protocol["groups"]:
                summary = {}
                failures = 0
                for condition in conds:
                    rows = [
                        x
                        for x in r["rows"]
                        if x["group"] == group and x["condition"] == condition
                    ]
                    assert len(rows) == 6
                    summary[condition] = {
                        k: None
                        if any(x["metrics"][arm][k] is None for x in rows)
                        else float(np.mean([x["metrics"][arm][k] for x in rows]))
                        for k in metrics
                    }
                    if condition in ("backing", "silence"):
                        limits = [
                            (
                                "voicing_false_alarm_rate",
                                "max",
                                gates[condition + "_false_alarm_rate_max"],
                            )
                        ]
                    else:
                        limits = [
                            ("raw_pitch_accuracy", "min", 0.8),
                            ("voicing_recall", "min", 0.8),
                            ("voicing_false_alarm_rate", "max", 0.1),
                            ("median_abs_cents_error", "max", 50.0),
                            ("octave_error_rate", "max", 0.05),
                        ]
                    for metric, direction, limit in limits:
                        value = summary[condition][metric]
                        failures += int(
                            value is None
                            or (value < limit if direction == "min" else value > limit)
                        )
                        gate_count += 1
                actual = grouped[group]
                check(actual["summary"], summary, (arm, group))
                assert actual["absent_conditions"] == []
                assert actual["gate"]["passed"] == (failures == 0)
                assert len(actual["gate"]["failures"]) == failures
                passed.append(failures == 0)
            control = r["rmvpe_analytic"] if arm.startswith("rmvpe") else controls[0]
            cases = control["cases"]
            assert set(cases) == {
                "tone_100",
                "tone_150",
                "tone_220",
                "tone_440",
                "glide_110_440",
                "missing_fundamental_220",
            }
            medians = [x["median_abs_cents_error_interior"] for x in cases.values()]
            complete = all(x is not None and np.isfinite(x) for x in medians)
            assert control["complete"] == complete
            if complete:
                check(
                    control["worst_median_abs_cents_error"],
                    max(medians),
                    "analytic maximum",
                )
            analytic_pass = bool(complete and max(medians) <= 50)
            assert variant["analytic_passed"] == analytic_pass
            assert variant["development_gates_passed"] == (
                all(passed) and analytic_pass
            )
    return dict(
        report_sha256=hashlib.sha256(raw).hexdigest(),
        rows=len(r["rows"]),
        artifact_hashes_checked=n_artifacts,
        pitch_scores_checked=n_pitch,
        activity_scores_checked=n_activity,
        activity_masks_checked=n_masks,
        aggregate_scalar_gates_checked=gate_count,
        complete=len(r["rows"]) == 90 and "variants" in r,
    )


def edit_distance(ref, pred):
    # Scalar-cost DP independent of production's path-count tuples.
    previous = list(range(len(pred) + 1))
    for i, a in enumerate(ref, 1):
        current = [i]
        for j, b in enumerate(pred, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a != b))
            )
        previous = current
    return previous[-1]


def phone_expected(values, times, bounds, hop, labels, item):
    assert np.isfinite(values).all() and (values >= 0).all()
    assert np.allclose(values.sum(1), 1, rtol=1e-5, atol=1e-7)
    assert len(times) == len(values) and np.all(np.diff(times) > 0)
    roles = item["token_roles"]
    roles = [roles[x] for x in labels] if isinstance(roles, dict) else roles
    inventory = item["canonical_inventory"]
    mapping = [
        inventory.get(label) if role == "phone" else None
        for label, role in zip(labels, roles)
    ]
    mapping = [([x] if isinstance(x, str) else x) for x in mapping]
    nonphones = set(item.get("reference_nonphone_labels", []))
    refs = [
        x
        for x in item["reference_intervals"]
        if x.get(
            "role",
            "nonphone" if x.get("phone", x.get("label")) in nonphones else "phone",
        )
        == "phone"
    ]
    modelphones = set(x for expanded in mapping if expanded for x in expanded)
    phones = list(
        dict.fromkeys(
            [x for expanded in mapping if expanded for x in expanded]
            + [x["phone"] for x in refs]
        )
    )
    # Independent row-by-row Voronoi intersections with native hop and support.
    cells = []
    for i, t in enumerate(times):
        lo = max(t - hop / 2, (times[i - 1] + t) / 2 if i else -np.inf)
        hi = min(t + hop / 2, (times[i + 1] + t) / 2 if i + 1 < len(times) else np.inf)
        if bounds is not None:
            lo = max(lo, bounds[i, 0])
            hi = min(hi, bounds[i, 1])
        assert hi > lo
        cells.append((lo, hi))
    gaps = (
        [False] + [cells[i][0] > cells[i - 1][1] + 1e-12 for i in range(1, len(cells))]
        if cells
        else []
    )
    decoded = []
    prev = None
    for i, token in enumerate(values.argmax(1)):
        if token != prev or gaps[i]:
            if roles[token] == "phone":
                decoded.extend(
                    mapping[token] or [f"__unmapped_model_token__:{labels[token]}"]
                )
        prev = token
    refseq = [x["phone"] for x in refs]
    distance = edit_distance(refseq, decoded)
    expected = {
        "phones": phones,
        "recognition": {
            "reference_phones": refseq,
            "decoded_phones": decoded,
            "distance": distance,
            "per": ratio(distance, len(refseq)),
            "n_discontinuities": sum(gaps),
            "full_path_observed": bool(len(values) and not any(gaps)),
        },
    }
    all_profiles = {}
    all_comparisons = {}
    for label, edges in [
        ("composition", item["window_bounds_s"]),
        ("ordered", item.get("bin_bounds_s", item["window_bounds_s"])),
    ]:
        profiles = []
        golds = []
        diagnostics = []
        gold_diags = []
        comparisons = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            # Partition into elementary constant segments; integrate by midpoint ownership.
            cuts = sorted(
                {
                    lo,
                    hi,
                    *[x for cell in cells for x in cell if lo < x < hi],
                    *[
                        r[k]
                        for r in refs
                        for k in ("start_s", "end_s")
                        if lo < r[k] < hi
                    ],
                }
            )
            mids = (np.array(cuts[:-1]) + cuts[1:]) / 2
            widths = np.diff(cuts)
            weights = np.array(
                [widths[(mids >= a) & (mids < b)].sum() for a, b in cells]
            )
            column_mass = weights @ values
            mass = np.zeros(len(phones))
            gold = np.zeros(len(phones))
            category = {
                role: float(
                    sum(column_mass[k] for k, r in enumerate(roles) if r == role)
                )
                for role in ("blank", "boundary", "special", "phone")
            }
            for k, expanded in enumerate(mapping):
                for ph in expanded or []:
                    mass[phones.index(ph)] += column_mass[k] / len(expanded)
            for ref in refs:
                gold[phones.index(ref["phone"])] += widths[
                    (mids >= ref["start_s"]) & (mids < ref["end_s"])
                ].sum()
            profile = (mass / mass.sum()).tolist() if mass.sum() > 0 else None
            reference = (gold / gold.sum()).tolist() if gold.sum() > 0 else None
            profiles.append(profile)
            golds.append(reference)
            category.update(
                mapped_phone_mass=float(mass.sum()),
                unmapped_phone_mass=float(category["phone"] - mass.sum()),
                observation_s=float(weights.sum()),
                temporal_observation_coverage=float(weights.sum() / (hi - lo)),
                mapping_coverage=ratio(mass.sum(), category["phone"]),
            )
            diagnostics.append(category)
            gold_diags.append(
                dict(
                    reference_phone_duration_s=float(gold.sum()),
                    reference_temporal_coverage=float(gold.sum() / (hi - lo)),
                    reference_mapping_coverage=ratio(
                        sum(
                            gold[k] for k, ph in enumerate(phones) if ph in modelphones
                        ),
                        gold.sum(),
                    ),
                )
            )
            valid = profile is not None and reference is not None
            comparisons.append(
                dict(
                    total_variation=float(
                        np.abs(np.array(profile) - reference).sum() / 2
                    )
                    if valid
                    else None,
                    observation_s=category["observation_s"],
                    mapping_coverage=category["mapping_coverage"],
                    valid=valid,
                )
            )
        all_profiles[label] = (profiles, golds)
        all_comparisons[label] = comparisons
        if label == "composition":
            expected.update(
                composition_profiles=profiles,
                reference_composition_profiles=golds,
                composition_diagnostics=diagnostics,
                reference_composition_diagnostics=gold_diags,
            )
        else:
            expected.update(
                ordered_bin_profiles=profiles,
                reference_bin_profiles=golds,
                ordered_bin_diagnostics=diagnostics,
                reference_bin_diagnostics=gold_diags,
            )
    expected["window_bounds_s"] = item["window_bounds_s"]
    expected["bin_bounds_s"] = item.get("bin_bounds_s", item["window_bounds_s"])
    expected["profile_comparisons"] = {
        "composition": all_comparisons["composition"],
        "ordered_bins": all_comparisons["ordered"],
    }
    p, g = all_profiles["ordered"]
    deltas = []
    for i in range(1, len(p)):
        if all(x is not None for x in (p[i], p[i - 1], g[i], g[i - 1])):
            deltas.append(
                float(
                    np.abs(
                        (np.array(p[i]) - p[i - 1]) - (np.array(g[i]) - g[i - 1])
                    ).sum()
                    / 2
                )
            )
    expected["ordering_sensitivity"] = {
        "adjacent_delta_total_variation_mean": float(np.mean(deltas))
        if deltas
        else None,
        "n_comparable_transitions": len(deltas),
    }
    expected["profile_valid"] = any(x is not None for x in p)
    counts = {}
    for label, (pred, gold) in all_profiles.items():
        for suffix, rows in [("predicted", pred), ("reference", gold)]:
            valid = sum(x is not None for x in rows)
            counts[label + "_" + suffix] = {
                "valid": valid,
                "missing": len(rows) - valid,
            }
    valid = sum(x["valid"] for x in all_comparisons["ordered"])
    counts["ordered_comparison"] = {"valid": valid, "missing": len(p) - valid}
    expected["profile_bin_counts"] = counts
    return expected


@functools.lru_cache(None)
def snapshot_sha(root):
    root = pathlib.Path(root)
    h = hashlib.sha256()
    files = sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix in {".json", ".py", ".bin", ".safetensors", ".pt", ".pth", ".ckpt"}
    )
    assert files
    for p in files:
        h.update(p.relative_to(root).as_posix().encode() + b"\0")
        h.update(bytes.fromhex(sha(p)))
    return "sha256:" + h.hexdigest()


def audit_phones(args):
    manifest_raw = args.phone_manifest.read_bytes()
    manifest = json.loads(manifest_raw)
    raw = args.phone_report.read_bytes()
    report = json.loads(raw)
    assert report["manifest_sha256"] == hashlib.sha256(manifest_raw).hexdigest()
    assert manifest["state"] == report["state"] == "development"
    assert report["runtime_failure"] is False and report["n_failed"] == 0
    protocol = manifest["protocol"]
    protocol_path = (
        args.root / "tests/benchmarks/thresholds/singing_phone_pilot_v1.json"
    )
    assert manifest["protocol_sha256"] == sha(protocol_path)
    assert protocol == json.loads(protocol_path.read_text())
    for path, digest in protocol["frozen_sources"].items():
        assert sha(args.root / path) == digest, path
    expected_ids = {
        f"{p['singer']}_{p['song']}_{c}_{m}"
        for p in protocol["pairs"]
        for c in protocol["conditions"]
        for m in protocol["models"]
    }
    items = {x["id"]: x for x in manifest["items"]}
    assert len(manifest["items"]) == len(items) == 32 and set(items) == expected_ids
    ids = [x["id"] for x in report["items"]]
    assert len(ids) == len(set(ids)) and set(ids) <= expected_ids
    if args.complete:
        assert len(ids) == 32 and set(ids) == expected_ids
    windows = 0
    for result in report["items"]:
        assert "error" not in result, result
        item = dict(items[result["id"]])
        item.setdefault("token_roles", manifest["token_roles"])
        item.setdefault("canonical_inventory", manifest["canonical_inventory"])

        def resolve(path):
            path = pathlib.Path(path)
            return path if path.is_absolute() else args.phone_manifest.parent / path

        refhash = hashlib.sha256(
            json.dumps(
                item["reference_intervals"], sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        assert result["source_hashes"]["reference_intervals"] == refhash
        if "posterior_npz" not in item:
            assert (
                sha(resolve(item["audio_path"]))
                == item["audio_sha256"]
                == result["source_hashes"]["audio"]
            )
            assert (
                snapshot_sha(str(resolve(item["model_snapshot"])))
                == item["model_snapshot_sha256"]
                == result["source_hashes"]["model_snapshot"]
            )
        artifacts = list(result["artifacts"].values())
        assert len(artifacts) == 1
        a = artifacts[0]
        path = resolve(a["path"])
        assert sha(path) == a["sha256"]
        if "posterior_npz" in item:
            assert (
                a["sha256"]
                == item["posterior_sha256"]
                == result["source_hashes"]["posterior_npz"]
            )
        with np.load(path, allow_pickle=False) as npz:
            values = np.asarray(npz["values"], float)
            times = np.asarray(npz["times_s"], float)
            labels = [str(x) for x in npz["labels"].tolist()]
            bounds = npz["bounds_s"] if "bounds_s" in npz.files else None
            timebase = (
                json.loads(str(npz["timebase_json"].item()))
                if "timebase_json" in npz.files
                else {}
            )
            if "metadata_json" in npz.files:
                assert (
                    json.loads(str(npz["metadata_json"].item()))
                    == result["native_metadata"]
                )
        hop = (
            item.get("native_hop_s")
            or timebase.get("hop_s")
            or float(np.min(np.diff(times)))
        )
        expected = phone_expected(values, times, bounds, hop, labels, item)
        check(result["evaluation"], expected, result["id"])
        rec = result["evaluation"]["recognition"]
        s, d, i = (rec[k] for k in ("substitutions", "deletions", "insertions"))
        assert all(isinstance(x, int) and x >= 0 for x in (s, d, i))
        assert s + d + i == expected["recognition"]["distance"]
        assert len(rec["reference_phones"]) - d + i == len(rec["decoded_phones"])
        windows += len(expected["composition_profiles"]) + len(
            expected["ordered_bin_profiles"]
        )
    return dict(
        report_sha256=hashlib.sha256(raw).hexdigest(),
        manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        items=len(ids),
        profile_windows_checked=windows,
        complete=len(ids) == 32,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--vocal-report", type=pathlib.Path)
    parser.add_argument("--vocal-assets", type=pathlib.Path)
    parser.add_argument("--phone-report", type=pathlib.Path)
    parser.add_argument("--phone-manifest", type=pathlib.Path)
    parser.add_argument("--complete", action="store_true")
    args = parser.parse_args()
    assert args.vocal_report or args.phone_report, "supply at least one report"
    assert not args.vocal_report or args.vocal_assets, "--vocal-assets is required"
    assert not args.phone_report or args.phone_manifest, "--phone-manifest is required"
    assert not args.complete or (args.vocal_report and args.phone_report), (
        "--complete requires both reports"
    )
    receipt = {
        "source_head": subprocess.check_output(
            ["git", "-C", str(args.root), "rev-parse", "HEAD"], text=True
        ).strip()
    }
    if args.vocal_report:
        receipt["vocal"] = audit_vocal(args)
    if args.phone_report:
        receipt["phones"] = audit_phones(args)
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
