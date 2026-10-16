#!/usr/bin/env python3
"""Compare two mappings and three poolings on 32 frozen posterior artifacts.

Without --rescore this only verifies the frozen protocol and prints its plan.
The explicit flag is intentional: empirical outputs are not regenerated while
the protocol or implementation is still under review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from natural_features.core.feature_types import FeatureSeries
from natural_features.core.timebase import TemporalContext, TimebaseSpec
from natural_features.features.speech.phone_inventory import load_phone_inventory
from natural_features.features.speech.singing_pooling import compare_singing_representations

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked(raw: dict[str, str], root: Path) -> Path:
    path = Path(raw["path"])
    path = path if path.is_absolute() else root / path
    if not path.is_file() or sha256(path) != raw["sha256"]:
        raise ValueError(f"pinned input differs: {path}")
    return path


def _series(path: Path) -> FeatureSeries:
    with np.load(path, allow_pickle=False) as data:
        required = {"times_s", "values", "labels", "metadata_json", "timebase_json", "temporal_context_json"}
        if required - set(data.files):
            raise ValueError(f"retained posterior lacks required fields: {sorted(required - set(data.files))}")
        bounds = data["bounds_s"] if "bounds_s" in data.files else None
        metadata = json.loads(str(data["metadata_json"].item()))
        if not isinstance(metadata, dict) or not all(metadata.get(key) for key in ("extractor_id", "params_hash")):
            raise ValueError("retained posterior metadata must identify extractor_id and params_hash")
        timebase = TimebaseSpec.from_dict(json.loads(str(data["timebase_json"].item())), default_kind="frames")
        context = TemporalContext.from_dict(json.loads(str(data["temporal_context_json"].item())))
        return FeatureSeries(values=data["values"], times_s=data["times_s"], coords={"feature": [str(x) for x in data["labels"].tolist()]}, metadata=metadata, time_bounds_s=bounds, timebase=timebase, temporal_context=context)


def _legacy_parity(comparison: dict[str, Any], old: dict[str, Any], *, tolerance: float = 1e-10) -> None:
    """Reject a v2 rescore unless its retained v1 branch reproduces the receipt."""
    old_phones, new_phones = old["phones"], comparison["phones"]
    new_index = {phone: i for i, phone in enumerate(new_phones)}
    if any(phone not in new_index for phone in old_phones):
        raise ValueError("v1 parity axis is missing an archived phone")
    for old_key, new_key in (("composition_profiles", "composition"), ("ordered_bin_profiles", "ordered_bins")):
        old_profiles = old[old_key]
        new_profiles = comparison[new_key]["mappings"]["v1_legacy_exact"]["representations"]["raw_posterior_integral"]["profiles"]
        if len(old_profiles) != len(new_profiles):
            raise ValueError("v1 parity profile count differs")
        for expected, observed in zip(old_profiles, new_profiles):
            if expected is None:
                if observed is not None:
                    raise ValueError("v1 parity expected a missing profile")
            elif observed is None or not np.allclose(np.asarray(expected), np.asarray(observed)[[new_index[p] for p in old_phones]], rtol=tolerance, atol=tolerance):
                raise ValueError("v1 raw posterior parity differs")
    old_recognition = old["recognition"]
    new_recognition = comparison["recognition"]["v1_legacy_exact"]
    for key in ("substitutions", "deletions", "insertions", "distance", "per"):
        if old_recognition[key] != new_recognition[key]:
            raise ValueError(f"v1 recognition parity differs: {key}")


def run(protocol_path: Path, *, root: Path, rescore: bool) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema") != "SingingRepresentationComparison/v2" or protocol.get("state") != "frozen_consumed_development":
        raise ValueError("expected frozen v2 representation protocol")
    inputs = protocol["inputs"]
    manifest_path, report_path, mapping_path, reservation_path = (_checked(inputs[key], root) for key in ("manifest", "v1_report", "mapping", "reservation"))
    for name, expected in protocol["frozen_sources"].items():
        if sha256(root / name) != expected:
            raise ValueError(f"frozen source differs: {name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Reading the complete receipt is deliberate: it proves this comparison
    # refers to the archived v1 run even though rescoring reads its NPZ files.
    old_report = json.loads(report_path.read_text(encoding="utf-8"))
    if old_report.get("manifest_sha256") != sha256(manifest_path):
        raise ValueError("v1 receipt does not identify the pinned manifest")
    by_id = {item["id"]: item for item in manifest["items"]}
    receipt_by_id = {item["id"]: item for item in old_report.get("items", [])}
    inventory = load_phone_inventory(mapping_path)
    result: dict[str, Any] = {"schema": "SingingRepresentationComparisonResult/v2", "protocol_sha256": sha256(protocol_path), "state": protocol["state"], "qualified": False, "fresh_data_consumed": False, "rescore_requested": rescore, "reservation_sha256": sha256(reservation_path), "arms": []}
    selected = list(manifest["items"])
    if protocol.get("arm_selection") != "all_manifest_items" or len(selected) != int(protocol.get("expected_item_count", -1)):
        raise ValueError("protocol must freeze every original manifest item")
    for item in selected:
        arm = {"id": item["id"]}
        item = by_id.get(arm["id"])
        receipt = receipt_by_id.get(arm["id"])
        artifact = (receipt or {}).get("artifacts", {}).get("generated_posterior")
        if item is None or not artifact:
            raise ValueError(f"pinned arm missing retained posterior: {arm['id']}")
        posterior = Path(artifact["path"])
        if sha256(posterior) != artifact["sha256"]:
            raise ValueError(f"pinned posterior differs: {arm['id']}")
        row: dict[str, Any] = {"id": arm["id"], **{key: item[key] for key in ("model", "condition", "singer", "song")}, "posterior_sha256": artifact["sha256"], "planned_mappings": protocol["mappings"], "planned_poolings": protocol["poolings"]}
        if rescore:
            series = _series(posterior)
            row["comparison"] = compare_singing_representations(series, token_roles=item["token_roles"], conservative_inventory=item["canonical_inventory"], inventory_v2=inventory, reference_intervals=item["reference_intervals"], window_bounds_s=item["window_bounds_s"], bin_bounds_s=item["bin_bounds_s"])
            _legacy_parity(row["comparison"], receipt["evaluation"])
        result["arms"].append(row)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "tests/benchmarks/thresholds/singing_representation_v2.json")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--rescore", action="store_true", help="Run all 32 frozen cached-posterior arms after protocol validation.")
    args = parser.parse_args()
    report = run(args.protocol, root=args.root, rescore=args.rescore)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.out_json:
        args.out_json.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
