#!/usr/bin/env python3
"""Finite, development-only replay calibration for Vocal F0 v2 diagnostics."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from natural_features.core.feature_types import FeatureSeries  # noqa: E402
from natural_features.features.audio.vocal_f0 import _acceptance_masks  # noqa: E402
from natural_features.features.audio.vocal_f0_benchmark import score_vocal_f0  # noqa: E402

GRID = ([0.21, 0.35, 0.5, 0.65, 0.8], [0.01, 0.03, 0.05, 0.1], [0.001, 0.003, 0.01])
CONDS = ("solo", "mixture_0db", "mixture_minus6db", "backing", "silence")


def _runner():
    spec = importlib.util.spec_from_file_location(
        "qualify_vocal_f0", ROOT / "scripts/qualify_vocal_f0.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load qualification runner")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _duration(feature: FeatureSeries) -> float:
    bounds = feature.metadata.get("computation_context", {}).get("input_bounds_s")
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise ValueError("missing computation_context.input_bounds_s")
    return float(bounds[1] - bounds[0])


def replay(feature: FeatureSeries, c: float, s: float, r: float) -> FeatureSeries:
    names = list(feature.coords["feature"])
    required = [
        "candidate_f0_hz",
        "source_rms",
        "confidence",
        "source_support",
        "source_supported",
        "voiced",
        "missing",
        "octave_jump",
    ]
    if any(name not in names for name in required):
        raise ValueError("raw v2 feature lacks diagnostic columns")
    values = np.asarray(feature.values, dtype=np.float64).copy()

    def at(name):
        return names.index(name)

    candidate, confidence, support, rms = (
        values[:, at(x)]
        for x in ("candidate_f0_hz", "confidence", "source_support", "source_rms")
    )
    start, end = feature.metadata["computation_context"]["input_bounds_s"]
    hop = feature.metadata.get("effective_hop_s")
    preprocessing = feature.metadata["computation_context"].get("preprocessing", {})
    duration = (
        (preprocessing["original_samples"] / preprocessing["original_sr_hz"])
        if "original_samples" in preprocessing
        else end - start
    )
    valid = (
        (np.arange(len(values)) * hop < duration)
        if hop is not None
        else ((feature.times_s >= start) & (feature.times_s < end))
    )
    source, voiced = _acceptance_masks(confidence, support, rms, valid, c, s, r)
    values[:, at("f0_hz")] = np.where(voiced, candidate, np.nan)
    values[:, at("source_supported")] = source
    values[:, at("voiced")] = voiced
    values[:, at("missing")] = ~voiced
    jumps = np.zeros(len(values), dtype=bool)
    cents = np.abs(1200 * np.diff(np.log2(candidate.astype(np.float32))))
    jumps[1:] = voiced[1:] & voiced[:-1] & (np.abs(cents - 1200) <= 100)
    values[:, at("octave_jump")] = jumps
    return FeatureSeries(
        values=values,
        times_s=feature.times_s,
        coords=feature.coords,
        metadata=feature.metadata,
        timebase=feature.timebase,
        time_bounds_s=feature.time_bounds_s,
        temporal_context=feature.temporal_context,
    )


def select(rows: list[dict]) -> dict | None:
    eligible = [x for x in rows if x["eligible"]]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda x: (
            x["objective"]["min_rpa"],
            -x["objective"]["max_fa"],
            -x["objective"]["backing_fa"],
            x["tuple"][0],
            x["tuple"][1],
            x["tuple"][2],
        ),
    )


def _reference(
    assets: Path, excerpt: int, duration: float
) -> tuple[np.ndarray, np.ndarray]:
    path = next(assets.rglob(f"vocadito_{excerpt}_f0.csv"))
    raw = np.loadtxt(path, delimiter=",")
    raw = raw[raw[:, 0] < duration]
    return raw[:, 0], raw[:, 1]


def _score(runner, assets, excerpt, condition, feature):
    times, hz = _reference(assets, excerpt, _duration(feature))
    times, hz = runner._scoring_reference(
        condition, times, hz, feature, _duration(feature)
    )
    return score_vocal_f0(times, hz, feature)


def _summary(runner, rows, gates, analytic):
    report = {
        "per_excerpt": rows,
        "summary": {},
        "analytic": analytic,
        "errors": [],
        "completed_jobs": 30,
    }
    metrics = [
        "raw_pitch_accuracy",
        "voicing_recall",
        "voicing_false_alarm_rate",
        "median_abs_cents_error",
        "octave_error_rate",
        "gross_pitch_error_rate",
    ]
    for cond in CONDS:
        report["summary"][cond] = runner._macro(
            [x for x in rows if x["condition"] == cond], metrics
        )
    passed, failures = runner._gate(report, gates)
    return report["summary"], passed, failures


def _run() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--development-report", type=Path, required=True)
    ap.add_argument("--baseline-report", type=Path, required=True)
    ap.add_argument("--assets", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "tests/benchmarks/thresholds/vocal_f0_calibration_v2.json",
    )
    args = ap.parse_args()
    protocol = json.loads(args.protocol.read_text())
    dev = json.loads(args.development_report.read_text())
    baseline = json.loads(args.baseline_report.read_text())
    runner = _runner()
    protocol_grid = protocol.get("calibration_grid", {})
    expected_grid = {
        "confidence_threshold": GRID[0],
        "source_threshold": GRID[1],
        "min_source_rms": GRID[2],
    }
    if protocol_grid != expected_grid:
        raise ValueError(
            "frozen protocol calibration_grid differs from the only permitted 60 tuples"
        )
    if dev.get("protocol_sha256") != runner._digest(args.protocol):
        raise ValueError("development protocol identity mismatch")
    if dev.get("completed_jobs") != 30 or not dev.get("analytic", {}).get("complete"):
        raise ValueError("development extraction and analytic checks must be complete")
    by_key = {(int(x["excerpt_id"]), x["condition"]): x for x in dev["per_excerpt"]}
    expected_keys = {(i, condition) for i in range(1, 7) for condition in CONDS}
    if (
        set(by_key) != expected_keys
        or len(dev["per_excerpt"]) != 30
        or dev.get("errors")
    ):
        raise ValueError(
            "development report must contain exactly the 30 frozen jobs without errors"
        )
    loaded = {}
    for key, row in by_key.items():
        annotation = runner._find_one(args.assets, f"vocadito_{key[0]}_f0.csv")
        if runner._digest(annotation) != row["input_hashes"]["reference_sha256"]:
            raise ValueError(f"reference annotation identity mismatch for {key}")
        artifact = row["artifact"]
        if runner._digest(Path(artifact["artifact"])) != artifact["artifact_sha256"]:
            raise ValueError(f"raw artifact SHA256 mismatch for {key}")
        fs = runner._load_feature(Path(artifact["artifact"]), artifact["cache_key"])
        if fs is None:
            raise ValueError(f"invalid raw artifact for {key}")
        loaded[key] = fs
    # Exact v1 replay receipt: baseline rows are compared after raw artifacts' first-nine values/masks are retained.
    baseline_keys = {
        (int(x["excerpt_id"]), x["condition"]): x for x in baseline["per_excerpt"]
    }
    for key, fs in loaded.items():
        if key not in baseline_keys:
            raise ValueError(f"baseline missing {key}")
        old = runner._load_feature(
            Path(baseline_keys[key]["artifact"]["artifact"]),
            baseline_keys[key]["artifact"]["cache_key"],
        )
        if old is None or not np.array_equal(fs.times_s, old.times_s):
            raise ValueError(f"baseline native times mismatch {key}")
        if by_key[key]["input_hashes"] != baseline_keys[key]["input_hashes"]:
            raise ValueError(f"baseline input identity mismatch {key}")
        np.testing.assert_allclose(
            fs.values[:, :9], old.values[:, :9], rtol=0, atol=1e-5, equal_nan=True
        )
        replayed = replay(fs, 0.21, 0.1, 0.001)
        np.testing.assert_allclose(
            replayed.values[:, :9], old.values[:, :9], rtol=0, atol=1e-5, equal_nan=True
        )
        recomputed = _score(runner, args.assets, key[0], key[1], replayed)
        for metric, expected in baseline_keys[key]["metrics"].items():
            actual = recomputed[metric]
            if expected is None or actual is None:
                if expected is not actual:
                    raise ValueError(
                        f"baseline score missingness mismatch {key}.{metric}"
                    )
            elif not np.isclose(actual, expected, rtol=0, atol=1e-5):
                raise ValueError(
                    f"baseline score mismatch {key}.{metric}: {actual} != {expected}"
                )
    rows = []
    for c in GRID[0]:
        for s in GRID[1]:
            for r in GRID[2]:
                replay_rows = []
                for (excerpt, cond), fs in loaded.items():
                    out = replay(fs, c, s, r)
                    replay_rows.append(
                        {
                            "excerpt_id": excerpt,
                            "condition": cond,
                            "metrics": _score(runner, args.assets, excerpt, cond, out),
                            "input_hashes": by_key[(excerpt, cond)].get(
                                "input_hashes", {}
                            ),
                            "artifact": by_key[(excerpt, cond)]["artifact"],
                        }
                    )
                summary, passed, failures = _summary(
                    runner, replay_rows, protocol["gates"], dev["analytic"]
                )
                positive = [
                    summary[x] for x in ("solo", "mixture_0db", "mixture_minus6db")
                ]
                objective = {
                    "min_rpa": min(x["raw_pitch_accuracy"] for x in positive),
                    "max_fa": max(x["voicing_false_alarm_rate"] for x in positive),
                    "backing_fa": summary["backing"]["voicing_false_alarm_rate"],
                }
                rows.append(
                    {
                        "tuple": [c, s, r],
                        "summary": summary,
                        "eligible": passed,
                        "failures": failures,
                        "objective": objective,
                        "per_excerpt": replay_rows,
                    }
                )
    chosen = select(rows)
    payload = {
        "schema": "VocalF0Calibration/v2",
        "protocol_sha256": runner._digest(args.protocol),
        "calibration_source_sha256": runner._digest(Path(__file__)),
        "development_report": str(args.development_report),
        "development_report_sha256": runner._digest(args.development_report),
        "baseline_report": str(args.baseline_report),
        "baseline_report_sha256": runner._digest(args.baseline_report),
        "model_revision": next(iter(loaded.values())).metadata["model_revision"],
        "baseline_replay": {
            "verified_first_nine_columns": True,
            "verified_input_identity": True,
            "verified_scores": True,
        },
        "all_tuples": [
            {key: value for key, value in row.items() if key != "per_excerpt"}
            for row in rows
        ],
        "selected": chosen,
        "calibration_passed": chosen is not None,
        "qualified": False,
        "holdout_consumed": False,
    }
    runner._atomic_json(args.out_json, payload)
    return 0 if chosen else 1


def main() -> int:
    try:
        return _run()
    except Exception as exc:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--out-json", type=Path)
        known, _ = parser.parse_known_args()
        if known.out_json is not None:
            known.out_json.parent.mkdir(parents=True, exist_ok=True)
            known.out_json.write_text(
                json.dumps(
                    {
                        "schema": "VocalF0Calibration/v2",
                        "qualified": False,
                        "status": "failed_incomplete_input",
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    indent=2,
                )
                + "\n"
            )
        print(
            f"calibration failed before replay: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
