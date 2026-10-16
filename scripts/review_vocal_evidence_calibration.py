"""Independently replay the frozen v2 evidence without fitting or selecting."""

import argparse
import json
import hashlib
import csv
import sys
import subprocess
from pathlib import Path
from dataclasses import replace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--report",
    type=Path,
    default=ROOT / "tests/benchmarks/reports/vocal_evidence_calibration_v2.json",
)
parser.add_argument("--assets", type=Path, required=True)
parser.add_argument("--out-json", type=Path, required=True)
args = parser.parse_args()
from natural_features.core.feature_types import FeatureSeries  # noqa: E402
from natural_features.core.timebase import TimebaseSpec, TemporalContext  # noqa: E402
from natural_features.features.audio.vocal_f0_benchmark import score_vocal_f0  # noqa: E402
from natural_features.features.audio.vocal_evidence_calibration import evidence  # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def digest(o):
    return hashlib.sha256(
        json.dumps(o, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def checked(p, h):
    assert sha(p) == h, ("hash mismatch", p)


def load(a, legacy=False):
    p = a["artifact"] if legacy else a["path"]
    checked(p, a["artifact_sha256"] if legacy else a["sha256"])
    with np.load(p, allow_pickle=False) as z:
        assert z["cache_key"].item() == a["cache_key" if legacy else "key"]
        return FeatureSeries(
            z["values"],
            z["times_s"],
            coords={"feature": json.loads(z["feature_names_json"].item())},
            metadata=json.loads(z["source_metadata_json"].item()),
            time_bounds_s=z["bounds"],
            timebase=TimebaseSpec.from_dict(
                json.loads(z["timebase_json"].item()), default_kind="frames"
            ),
            temporal_context=TemporalContext.from_dict(
                json.loads(z["temporal_context_json"].item())
            ),
        )


def col(f, n):
    return f.values[:, list(f.coords["feature"]).index(n)]


reportpath = args.report
r = json.loads(reportpath.read_text())
p = r["protocol"]
assert digest(p) == r["protocol_sha256"]
assert (
    reportpath.read_bytes()
    == (
        ROOT / "tests/benchmarks/reports/vocal_evidence_calibration_v2.json"
    ).read_bytes()
)
head = "00b37dcfffa149c88d74d62ace43ac97048ad170"
for f, h in p["frozen_sources"].items():
    checked(ROOT / f, h)
    assert (
        hashlib.sha256(
            subprocess.check_output(["git", "show", head + ":" + f], cwd=ROOT)
        ).hexdigest()
        == h
    )
checked(ROOT / p["input_report"]["path"], p["input_report"]["sha256"])
comp = json.load(open(ROOT / p["input_report"]["path"]))
source = comp["protocol"]["consumed_protocol"]
checked(ROOT / source["path"], source["sha256"])
oldp = json.load(open(ROOT / source["path"]))
assert p["gates"] == oldp["gates"]
gates = p["gates"]
assets = args.assets
meta = assets / "vocadito/vocadito_metadata.csv"
checked(meta, oldp["singer_metadata_sha256"])
singers = {int(x["track_id"]): x["singer_id"] for x in csv.DictReader(meta.open())}
old = {}
for spec in oldp["input_reports"]:
    checked(ROOT / spec["path"], spec["sha256"])
    for a in json.load(open(ROOT / spec["path"]))[spec["rows"]]:
        group = spec.get("group", a.get("group"))
        old[f"{group}:{a['excerpt_id']}:{a['condition']}"] = a
records = {}
for row in comp["rows"]:
    key = row["key"]
    o = old[key]
    orig = load(o["artifact"], True)
    duration = np.diff(orig.metadata["computation_context"]["input_bounds_s"])[0]
    refpath = next(assets.rglob(f"vocadito_{row['excerpt']}_f0.csv"))
    checked(refpath, o["input_hashes"]["reference_sha256"])
    ref = np.loadtxt(refpath, delimiter=",")
    ref = ref[ref[:, 0] < duration]
    cond = row["condition"]
    times = (
        orig.times_s[(orig.times_s >= 0) & (orig.times_s < duration)]
        if cond in ["backing", "silence"]
        else ref[:, 0]
    )
    hz = np.zeros(len(times)) if cond in ["backing", "silence"] else ref[:, 1]
    pitch, canon, act = [
        load(row["artifacts"][n])
        for n in ["rmvpe_stem", "canonical", "activity_mixture"]
    ]
    x, admiss = evidence(pitch, canon, act)
    assert score_vocal_f0(times, hz, pitch) == row["metrics"]["rmvpe_stem"], (
        "original projection",
        key,
    )
    assert (
        row["audio_sha256"] == o["input_hashes"]["rendered_16khz_mono_sha256"]
        and row["singer"] == singers[row["excerpt"]]
    )
    fam = (
        "silence"
        if cond == "silence"
        else f"backing:{o['input_hashes']['backing_sha256']}:{oldp['groups'][row['group']]['backing_start_s']}"
        if cond == "backing"
        else None
    )
    records[key] = {
        "row": row,
        "pitch": pitch,
        "x": x,
        "admiss": admiss,
        "times": times,
        "hz": hz,
        "family": fam,
    }
assert len(records) == 90 and set(records) == set(old)
keys = list(records)


def folds(ks, n, outer=False):
    ss = sorted({records[k]["row"]["singer"] for k in ks}, key=lambda s: int(s[1:]))
    groups = [
        {s for i, s in enumerate(ss) if ((int(s[1:]) - 1) if outer else i) % n == j}
        for j in range(n)
    ]
    return [[k for k in ks if records[k]["row"]["singer"] in group] for group in groups]


def purge(ks, held):
    ss = {records[k]["row"]["singer"] for k in held}
    hs = {records[k]["row"]["audio_sha256"] for k in held}
    fs = {records[k]["family"] for k in held if records[k]["family"] is not None}
    return [
        k
        for k in ks
        if records[k]["row"]["singer"] not in ss
        and records[k]["row"]["audio_sha256"] not in hs
        and (records[k]["family"] is None or records[k]["family"] not in fs)
    ]


def prob(k, model):
    if model is None:
        return col(records[k]["pitch"], "confidence")
    z = ((records[k]["x"] - model["mean"]) / model["scale"]) @ np.array(
        model["coefficients"]
    ) + model["intercept"]
    return np.exp(-np.logaddexp(0.0, -z))


def scored(k, pr, threshold):
    f = records[k]["pitch"]
    v = f.values.copy()
    names = list(f.coords["feature"])
    pc = np.clip(pr, 1e-12, 1 - 1e-12)
    accept = records[k]["admiss"] & (
        np.log(pc / (1 - pc)) > np.log(threshold / (1 - threshold))
    )
    v[:, names.index("f0_hz")] = np.where(accept, col(f, "candidate_f0_hz"), np.nan)
    v[:, names.index("voiced")] = accept
    return score_vocal_f0(
        records[k]["times"], records[k]["hz"], replace(f, values=v)
    ), accept


metricsnames = [
    "raw_pitch_accuracy",
    "voicing_recall",
    "voicing_false_alarm_rate",
    "median_abs_cents_error",
    "octave_error_rate",
    "gross_pitch_error_rate",
]
conds = ["solo", "mixture_0db", "mixture_minus6db", "backing", "silence"]


def same_aggregate(actual, expected):
    assert actual.keys() == expected.keys()
    for group in actual:
        assert actual[group].keys() == expected[group].keys()
        for cond in actual[group]:
            for metric in actual[group][cond]:
                a, b = actual[group][cond][metric], expected[group][cond][metric]
                if a is None or b is None:
                    assert a is b, (group, cond, metric, a, b)
                else:
                    assert abs(a - b) <= 1e-14, (group, cond, metric, a, b)


def aggregate(metrics, complete=False):
    result = {}
    passed = True
    for group in sorted({records[k]["row"]["group"] for k in metrics}):
        result[group] = {}
        for cond in conds:
            rows = [
                m
                for k, m in metrics.items()
                if records[k]["row"]["group"] == group
                and records[k]["row"]["condition"] == cond
            ]
            if not rows:
                if complete or cond not in ["backing", "silence"]:
                    passed = False
                continue
            if complete:
                assert len(rows) == 6
            avg = {
                n: None
                if any(a[n] is None for a in rows)
                else float(np.mean([a[n] for a in rows]))
                for n in metricsnames
            }
            result[group][cond] = avg
            limits = (
                [
                    (
                        "voicing_false_alarm_rate",
                        gates[
                            "backing_false_alarm_rate_max"
                            if cond == "backing"
                            else "silence_false_alarm_rate_max"
                        ],
                        False,
                    )
                ]
                if cond in ["backing", "silence"]
                else [
                    ("raw_pitch_accuracy", gates["raw_pitch_accuracy_min"], True),
                    ("voicing_recall", gates["voicing_recall_min"], True),
                    (
                        "voicing_false_alarm_rate",
                        gates["voicing_false_alarm_rate_max"],
                        False,
                    ),
                    (
                        "median_abs_cents_error",
                        gates["median_abs_cents_error_max"],
                        False,
                    ),
                    ("octave_error_rate", gates["octave_error_rate_max"], False),
                ]
            )
            for n, b, lower in limits:
                passed &= avg[n] is not None and (avg[n] >= b if lower else avg[n] <= b)
    return result, bool(passed)


traces = {t["context"]: t for t in r["fit_trace"]}
assert len(traces) == 24
tracechecks = 0
selectors = 0
outputchecks = 0
results = {}


def checkfit(context, train, forbidden, model):
    global tracechecks
    t = traces[context]
    assert t["training_keys"] == train and t["forbidden_keys"] == sorted(forbidden), (
        context,
        "ancestry",
    )
    assert train == purge(train, forbidden)
    assert t["model"] == model and digest(model) == t["model_identity"]
    tracechecks += 1


for arm, a in r["arms"].items():
    metrics = {}
    outerfolds = folds(keys, 5, True)
    assert len(a["outer"]) == 5
    outputmap = {v["key"]: v for v in a["rows"]}
    assert len(a["rows"]) == 90 and set(outputmap) == set(keys)
    for i, held in enumerate(outerfolds):
        out = a["outer"][i]
        train = purge(keys, held)
        assert (
            out["held_keys"] == held
            and out["training_keys"] == train
            and out["fold"] == i
        )
        model = out["model"]
        context = f"{arm}/outer-{i}"
        if model:
            checkfit(context + "/refit", train, held, model)
        for k in held:
            row = outputmap[k]
            assert row["outer_fold"] == i
            pr = prob(k, model)
            threshold = out["selection"]["selected"]["threshold"]
            assert row["threshold"] == threshold
            expect, mask = scored(k, pr, threshold)
            f = load(row["artifact"])
            assert (
                score_vocal_f0(records[k]["times"], records[k]["hz"], f)
                == expect
                == row["metrics"]
            ), ("output scoring", arm, k)
            np.testing.assert_array_equal(f.times_s, records[k]["pitch"].times_s)
            np.testing.assert_array_equal(col(f, "pitched_voicing_score"), pr)
            np.testing.assert_array_equal(
                col(f, "candidate_f0_hz"), col(records[k]["pitch"], "candidate_f0_hz")
            )
            np.testing.assert_array_equal(col(f, "voiced"), mask)
            metrics[k] = expect
            outputchecks += 1
    for i in range(6):
        ancestors = outerfolds[i] if i < 5 else []
        allowed = purge(keys, ancestors)
        context = f"{arm}/outer-{i}" if i < 5 else arm + "/final"
        sel = a["outer"][i]["selection"] if i < 5 else a["final_selection"]
        assert sel["selection_keys"] == allowed and sel[
            "ancestor_forbidden_keys"
        ] == sorted(ancestors)
        sm = {}
        models = sel["selector_models"]
        assert len(models) == 3
        for j, validation in enumerate(folds(allowed, 3)):
            saved = models[j]
            assert saved["validation_keys"] == validation
            model = saved["model"]
            forbidden = ancestors + validation
            if model:
                checkfit(
                    context + f"/inner-{j}", purge(allowed, forbidden), forbidden, model
                )
            for k in validation:
                sm[k] = scored(k, prob(k, model), sel["selected"]["threshold"])[0]
        assert set(sm) == set(allowed)
        agg, eligible = aggregate(sm)
        assert eligible == sel["eligible"]
        same_aggregate({x["name"]: x["summary"] for x in sel["groups"]}, agg)
        state = sel["states_artifact"]
        checked(state["path"], state["sha256"])
        with np.load(state["path"], allow_pickle=False) as z:
            idx = sel["selected_state_index"]
            assert sel["state_count"] == len(z["threshold"])
            assert (
                idx is not None
                and z["threshold"][idx] == sel["selected"]["threshold"]
                and bool(z["all_gates_pass"][idx]) == eligible
            )
            assert int(z["all_gates_pass"].sum()) == sel["all_gates_feasible_states"]
            assert int(z["monotone_pass"].sum()) == sel["monotone_feasible_states"]
            assert {
                "minimum_rpa": float(z["minimum_rpa"][idx]),
                "maximum_false_alarm": float(z["maximum_false_alarm"][idx]),
            } == sel["objective"]
        selectors += 1
    if a["final_model"]:
        checkfit(arm + "/final/refit", keys, [], a["final_model"])
    agg, passed = aggregate(metrics, True)
    same_aggregate({x["name"]: x["summary"] for x in a["groups"]}, agg)
    analytic = comp["rmvpe_analytic"]
    analyticpass = bool(
        analytic["complete"]
        and analytic["worst_median_abs_cents_error"]
        <= gates["analytic_median_abs_cents_error_max"]
    )
    empirical = (
        passed
        and all(x["selection"]["eligible"] for x in a["outer"])
        and a["final_selection"]["eligible"]
    )
    assert (
        a["analytic_passed"] == analyticpass
        and a["empirical_procedure_passed"] == empirical
        and a["all_original_gates_passed"] == (empirical and analyticpass)
    )
    positive = [
        m
        for group in agg.values()
        for c, m in group.items()
        if c not in ["backing", "silence"]
    ]
    results[arm] = {
        "max_fa": max(m["voicing_false_alarm_rate"] for m in positive),
        "min_rpa": min(m["raw_pitch_accuracy"] for m in positive),
        "empirical": empirical,
        "analytic": analyticpass,
        "original_gates": a["all_original_gates_passed"],
        "reported_failures": {
            x["name"]: x["gate"]["failures"]
            for x in a["groups"]
            if not x["gate"]["passed"]
        },
    }
assert tracechecks == 24 and selectors == 12 and outputchecks == 180
receipt = {
    "reviewed_commit": head,
    "report_sha256": sha(reportpath),
    "original_input_projections": 90,
    "verified_fits": tracechecks,
    "verified_selectors": selectors,
    "verified_outputs": outputchecks,
    "results": results,
    "scope": "No fitting, reselection, neural inference or fresh data; scalar replay from retained arrays and fitted coefficients.",
}
args.out_json.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
