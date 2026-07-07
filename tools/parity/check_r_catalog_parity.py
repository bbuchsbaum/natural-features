#!/usr/bin/env python
"""Check Python feature-catalog parity against the R public contract manifest."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from natural_features.workflows.extract_features import available_features  # noqa: E402
from natural_features.workflows._public_contract import load_r_public_feature_contracts  # noqa: E402


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict) or not isinstance(payload.get("features"), dict):
        raise ValueError(f"Invalid parity manifest: {path}")
    return payload


def _parse_r_catalog(path: Path) -> dict[str, dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    out: dict[str, dict[str, Any]] = {}
    pos = 0
    while True:
        match = re.search(r'list\(\s*\n\s*feature_id\s*=\s*"([^"]+)"', text[pos:])
        if not match:
            break
        start = pos + match.start()
        feature_id = match.group(1)
        next_match = re.search(r'\n\s*list\(\s*\n\s*feature_id\s*=', text[start + 1 :])
        end = start + 1 + next_match.start() if next_match else len(text)
        block = text[start:end]
        schema_match = re.search(r'output_schema\s*=\s*"([^"]+)"', block)
        bundles_match = re.search(r"bundles\s*=\s*c\((.*?)\)", block, flags=re.S)
        defaults_match = re.search(r"default_params\s*=\s*list\((.*?)\)", block, flags=re.S)
        if bundles_match:
            bundles = re.findall(r'"([^"]+)"', bundles_match.group(1))
        elif re.search(r"bundles\s*=\s*character\(0\)", block):
            bundles = []
        else:
            bundles = []
        default_keys = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*=", defaults_match.group(1)) if defaults_match else []
        out[feature_id] = {
            "schema": schema_match.group(1) if schema_match else "",
            "bundles": set(bundles),
            "default_keys": set(default_keys),
        }
        pos = end
    return out


def _schema_compatible(r_schema: str, py_schema: str) -> bool:
    if not r_schema:
        return True
    return r_schema in py_schema.replace("/v1", "")


def _check_python_contract(features: dict[str, Any]) -> list[str]:
    entries = {entry.feature_id: entry for entry in available_features(budget="all", public_only=False)}
    public_entries = {entry.feature_id: entry for entry in available_features(budget="all")}
    errors: list[str] = []

    actual_public_ids = set(public_entries)
    expected_public_ids = set(features)
    if actual_public_ids != expected_public_ids:
        missing = sorted(expected_public_ids - actual_public_ids)
        extra = sorted(actual_public_ids - expected_public_ids)
        if missing:
            errors.append(f"public catalog missing manifest IDs: {missing}")
        if extra:
            errors.append(f"public catalog has IDs absent from manifest: {extra}")

    for feature_id, spec in features.items():
        entry = entries.get(feature_id)
        if entry is None:
            errors.append(f"{feature_id}: missing from Python catalog")
            continue
        expected_schema = str(spec.get("expected_python_schema", ""))
        if entry.output_schema != expected_schema:
            errors.append(f"{feature_id}: output_schema {entry.output_schema!r} != {expected_schema!r}")
        expected_bundles = set(spec.get("bundles") or [])
        if set(entry.bundles) != expected_bundles:
            errors.append(f"{feature_id}: bundles {sorted(entry.bundles)!r} != {sorted(expected_bundles)!r}")
        default_keys = set(entry.default_params)
        required = set(spec.get("required_default_keys") or [])
        allowed_extra = set(spec.get("allowed_python_default_extras") or [])
        missing = sorted(required - default_keys)
        extra = sorted(default_keys - required - allowed_extra)
        if missing:
            errors.append(f"{feature_id}: missing required defaults {missing}")
        if extra:
            errors.append(f"{feature_id}: unlisted Python default extras {extra}")
    return errors


def _check_r_catalog(features: dict[str, Any], r_repo: Path) -> list[str]:
    catalog_file = r_repo / "R" / "extract_features.R"
    if not catalog_file.exists():
        return [f"R catalog not found: {catalog_file}"]
    r_catalog = _parse_r_catalog(catalog_file)
    errors: list[str] = []
    manifest_ids = set(features)
    r_ids = set(r_catalog)
    if manifest_ids != r_ids:
        missing = sorted(r_ids - manifest_ids)
        extra = sorted(manifest_ids - r_ids)
        if missing:
            errors.append(f"manifest missing R feature IDs: {missing}")
        if extra:
            errors.append(f"manifest has feature IDs absent from R catalog: {extra}")
    for feature_id, spec in features.items():
        r_entry = r_catalog.get(feature_id)
        if r_entry is None:
            continue
        r_schema = str(spec.get("r_catalog_schema") or "")
        if not r_schema and not _schema_compatible(r_entry["schema"], str(spec.get("expected_python_schema", ""))):
            errors.append(
                f"{feature_id}: R schema {r_entry['schema']!r} needs r_catalog_schema decision "
                f"for Python schema {spec.get('expected_python_schema')!r}"
            )
        elif r_schema and r_schema != r_entry["schema"]:
            errors.append(f"{feature_id}: manifest R schema {r_schema!r} != live R schema {r_entry['schema']!r}")
        if set(spec.get("bundles") or []) != r_entry["bundles"]:
            errors.append(
                f"{feature_id}: manifest bundles {sorted(spec.get('bundles') or [])!r} "
                f"!= live R bundles {sorted(r_entry['bundles'])!r}"
            )
        required = set(spec.get("required_default_keys") or [])
        missing_defaults = sorted(r_entry["default_keys"] - required)
        extra_required = sorted(required - r_entry["default_keys"])
        if missing_defaults:
            errors.append(f"{feature_id}: manifest required defaults missing live R keys {missing_defaults}")
        if extra_required:
            errors.append(f"{feature_id}: manifest requires defaults absent from live R {extra_required}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to a parity manifest. Defaults to the packaged public feature contract.",
    )
    parser.add_argument(
        "--r-repo",
        type=Path,
        default=Path("~/code/natfeatures").expanduser(),
        help="Optional R package checkout for live comparison.",
    )
    parser.add_argument("--no-r-compare", action="store_true", help="Skip live comparison to the R package checkout.")
    args = parser.parse_args(argv)

    manifest = _load_manifest(args.manifest) if args.manifest is not None else load_r_public_feature_contracts()
    features = manifest["features"]
    errors = _check_python_contract(features)
    if not args.no_r_compare and args.r_repo.exists():
        errors.extend(_check_r_catalog(features, args.r_repo))

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"parity-check: OK ({len(features)} public feature contracts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
