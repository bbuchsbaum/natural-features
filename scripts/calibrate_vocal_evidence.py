#!/usr/bin/env python3
"""One frozen nested development comparison of RMVPE and joint voicing evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import calibrate_vocal_voicing as v7
from calibrate_vocal_voicing_threshold import exhaustive_search
from compare_vocal_backends import load_saved
from natural_features.features.audio.vocal_evidence_calibration import (
    FEATURE_NAMES, apply_scores, digest_json, evidence, fit_model, predict,
)
from natural_features.features.audio.vocal_f0_benchmark import _nearest_index, score_vocal_f0
from natural_features.features.audio.vocal_voicing_threshold import threshold_mask

ROOT = v7.ROOT
ARMS = ("rmvpe_threshold", "joint_logistic")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class EvidenceRecord:
    """Reference projection rebuilt on the RMVPE grid, not canonical decisions."""
    def __init__(self, original, pitch, canonical, activity):
        for name in ("key", "group", "singer", "excerpt", "condition", "audio_hash", "family",
                     "reference_times", "reference_hz", "provenance"):
            setattr(self, name, getattr(original, name))
        self.feature, self.canonical, self.activity = pitch, canonical, activity
        self.x, self.admissible = evidence(pitch, canonical, activity)
        self.indices = np.asarray([(-1 if (i := _nearest_index(pitch.times_s, float(t), .005001)) is None else i)
                                   for t in self.reference_times], int)
        self.matched = self.indices >= 0
        self.known = ~np.isnan(self.reference_hz)
        self.positive = self.known & (self.reference_hz > 0)
        self.negative = self.known & (self.reference_hz == 0)
        self.counts = np.bincount(self.indices[self.known & self.matched], minlength=len(self.x)).astype(float)
        positives = np.bincount(self.indices[self.positive & self.matched], minlength=len(self.x))
        self.labels = np.divide(positives, self.counts, out=np.zeros(len(self.x)), where=self.counts > 0)
        self.training_counts = self.counts * self.admissible
        names = list(pitch.coords["feature"])
        self.candidate = pitch.values[:, names.index("candidate_f0_hz")]
        self.salience = pitch.values[:, names.index("confidence")]
        self.cents = np.full(len(self.reference_hz), np.inf)
        keep = self.positive & self.matched
        self.cents[keep] = abs(1200*np.log2(self.candidate[self.indices[keep]]/self.reference_hz[keep]))

    def score(self, mask):
        return v7.Record.score(self, mask)


def load_records(protocol, assets):
    input_path = ROOT/protocol["input_report"]["path"]
    if sha(input_path) != protocol["input_report"]["sha256"]:
        raise ValueError("comparison report changed")
    report = json.loads(input_path.read_text())
    inherited = report["protocol"]["consumed_protocol"]
    if sha(ROOT/inherited["path"]) != inherited["sha256"]:
        raise ValueError("consumed protocol changed")
    inherited_protocol = json.loads((ROOT/inherited["path"]).read_text())
    if protocol["gates"] != inherited_protocol["gates"]:
        raise ValueError("original acceptance gates must be unchanged")
    old, _ = v7.load_records(inherited_protocol, assets, v7._runner())
    originals = {r.key: r for r in old}
    records = []
    for row in report["rows"]:
        loaded = []
        for name in ("rmvpe_stem", "canonical", "activity_mixture"):
            artifact = row["artifacts"][name]
            if sha(artifact["path"]) != artifact["sha256"]:
                raise ValueError("native input artifact changed")
            loaded.append(load_saved(Path(artifact["path"]), artifact["key"]))
        r = EvidenceRecord(originals[row["key"]], *loaded)
        r.provenance = {**r.provenance, "new_evidence_artifacts": {n: row["artifacts"][n] for n in ("rmvpe_stem", "canonical", "activity_mixture")}}
        current = apply_scores(*loaded, r.salience, .03, model_identity="upstream-fixed-salience", score_kind="rmvpe_salience")
        scores = score_vocal_f0(r.reference_times, r.reference_hz, current)
        if r.score(threshold_mask(r.salience, r.admissible, .03)) != scores or scores != row["metrics"]["rmvpe_stem"]:
            raise ValueError("RMVPE native/reference projection failed exact replay")
        records.append(r)
    if len(records) != 90 or {r.key for r in records} != set(originals):
        raise ValueError("expected exactly the 90 consumed cases")
    return records, report["rmvpe_analytic"]


def fit(records, forbidden, context, trace, l2):
    if records != v7.purged_training(records, forbidden):
        raise ValueError("ancestor exclusion violated")
    x, y, weights = v7.training_arrays(records)
    model = fit_model(x, y, weights, l2=l2)
    trace.append({"context": context, "training_keys": [r.key for r in records],
                  "forbidden_keys": sorted(r.key for r in forbidden), "model": model,
                  "model_identity": digest_json(model)})
    return model


def select(records, ancestors, context, arm, protocol, trace, artifact_dir):
    scores, models = {}, []
    for i, validation in enumerate(v7.grouped_folds(records, 3)):
        forbidden = ancestors + validation
        train = v7.purged_training(records, forbidden)
        model = (fit(train, forbidden, f"{context}/inner-{i}", trace, protocol["l2"])
                 if arm == "joint_logistic" else None)
        models.append({"validation_keys": [r.key for r in validation], "model": model})
        for r in validation:
            scores[r.key] = predict(r.x, model) if model else r.salience
    selected, _ = exhaustive_search(records, scores, protocol["gates"], artifact_dir/(context.replace("/", "_")+"_states.npz"))
    selected["selector_models"] = models
    selected["selection_keys"] = [r.key for r in records]
    selected["ancestor_forbidden_keys"] = sorted(r.key for r in ancestors)
    return selected


def probability_diagnostic(r, p):
    # Explicit end-to-end score0 outside admissibility/coverage; no dropped misses.
    q = np.zeros(len(r.reference_hz))
    valid = r.matched.copy()
    valid[r.matched] &= r.admissible[r.indices[r.matched]]
    q[valid] = p[r.indices[valid]]
    return {"brier_on_known_reference": float(np.mean((q[r.known]-r.positive[r.known])**2)),
            "n_known": int(r.known.sum()), "n_matched_admissible": int((valid & r.known).sum()),
            "score_interpretation": "baseline salience is not a calibrated probability; joint score is benchmark-weighted"}


def run(protocol, records, analytic, artifact_dir, out_json):
    runner = v7._runner()
    report = {"schema": "JointVocalEvidenceDevelopment/v2", "state": "development", "qualified": False,
              "protocol": protocol, "protocol_sha256": digest_json(protocol), "errors": [],
              "arms": {}, "fit_trace": [], "fresh_data_consumed": False, "adopted_model": None,
              "inherited_rmvpe_analytic": analytic}
    folds = v7.grouped_folds(records, 5, outer=True)
    for arm in ARMS:
        outputs, outer = {}, []
        for i, held in enumerate(folds):
            context = f"{arm}/outer-{i}"
            train = v7.purged_training(records, held)
            selection = select(train, held, context, arm, protocol, report["fit_trace"], artifact_dir)
            model = fit(train, held, context+"/refit", report["fit_trace"], protocol["l2"]) if arm == "joint_logistic" else None
            threshold = selection["selected"]["threshold"]
            identity = digest_json(model) if model else "RMVPE-native-salience"
            outer.append({"fold": i, "training_keys": [r.key for r in train], "held_keys": [r.key for r in held],
                          "selection": selection, "model": model})
            for r in held:
                p = predict(r.x, model) if model else r.salience
                feature = apply_scores(r.feature, r.canonical, r.activity, p, threshold, model_identity=identity, score_kind="joint_logistic" if model else "rmvpe_salience")
                metrics = score_vocal_f0(r.reference_times, r.reference_hz, feature)
                if metrics != r.score(threshold_mask(p, r.admissible, threshold)):
                    raise ValueError("optimized and public scorer disagree")
                path = artifact_dir/(arm+"_"+r.key.replace(":", "_")+".npz")
                key = digest_json({"protocol": report["protocol_sha256"], "key": r.key, "model": identity,
                                   "threshold": threshold, "inputs": r.provenance["new_evidence_artifacts"]})
                runner._save_feature(path, key, feature)
                outputs[r.key] = {"key": r.key, "group": r.group, "singer": r.singer,
                    "excerpt": r.excerpt, "condition": r.condition, "outer_fold": i,
                    "threshold": threshold, "model_identity": identity, "metrics": metrics,
                    "probability_diagnostic": probability_diagnostic(r, p), "provenance": r.provenance,
                    "artifact": {"path": str(path.resolve()), "sha256": sha(path), "key": key}}
            print(f"{arm}: completed outer fold {i+1}/5; selector eligible={selection['eligible']}", flush=True)
        final = select(records, [], arm+"/final", arm, protocol, report["fit_trace"], artifact_dir)
        final_model = fit(records, [], arm+"/final/refit", report["fit_trace"], protocol["l2"]) if arm == "joint_logistic" else None
        groups = v7.summarize(records, {k: v["metrics"] for k, v in outputs.items()}, protocol["gates"], complete=True)
        empirical = all(g["gate"]["passed"] for g in groups) and all(x["selection"]["eligible"] for x in outer) and final["eligible"]
        analytic_pass = bool(analytic["complete"] and analytic["worst_median_abs_cents_error"] <= protocol["gates"]["analytic_median_abs_cents_error_max"])
        report["arms"][arm] = {"outer": outer, "rows": list(outputs.values()), "groups": groups,
            "final_selection": final, "final_model": final_model, "empirical_procedure_passed": empirical,
            "analytic_passed": analytic_pass, "all_original_gates_passed": empirical and analytic_pass}
        runner._atomic_json(out_json, report)
    report["decision"] = "No automatic adoption or fresh evaluation; preserve both predeclared arms regardless of outcome."
    runner._atomic_json(out_json, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT/"tests/benchmarks/thresholds/vocal_evidence_calibration_v2.json")
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol["state"] != "development" or protocol["qualification_allowed"] is not False:
        raise ValueError("consumed-data development only")
    if protocol["feature_names"] != list(FEATURE_NAMES) or protocol["arms"] != list(ARMS) or protocol["l2"] != .01:
        raise ValueError("fixed experiment changed")
    for path, expected in protocol["frozen_sources"].items():
        if sha(ROOT/path) != expected:
            raise ValueError("frozen source changed: "+path)
    reservation = protocol["reference_reservation"]
    if sha(ROOT/reservation["path"]) != reservation["sha256"]:
        raise ValueError("reference reservation changed")
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    records, analytic = load_records(protocol, args.assets)
    run(protocol, records, analytic, args.artifact_dir, args.out_json)
    return 0  # Execution completed; scientific gates are separate report fields.


if __name__ == "__main__":
    raise SystemExit(main())
