#!/usr/bin/env python3
"""Pre-release static checks for natfeatures."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys


REQUIRED_FILES = [
    "CHANGELOG.md",
    "docs/release_process.md",
    "docs/public_api_policy.md",
    "tests/fixtures/golden_reference_v1.json",
    "tests/benchmarks/thresholds/alignment_quality_gate.json",
    "tests/benchmarks/manifests/tier_a_alignment_manifest.json",
]


def _check_files(root: Path, problems: list[str]) -> None:
    for rel in REQUIRED_FILES:
        p = root / rel
        if not p.exists():
            problems.append(f"Missing required file: {rel}")


def _check_api_compat(root: Path, problems: list[str]) -> None:
    path = root / "src" / "natural_features" / "public_api.py"
    text = path.read_text(encoding="utf-8")
    m = re.search(r"^API_COMPAT_VERSION\s*=\s*(\d+)\s*$", text, flags=re.MULTILINE)
    if not m:
        problems.append("API_COMPAT_VERSION not found in src/natural_features/public_api.py")
        return
    if int(m.group(1)) < 1:
        problems.append("API_COMPAT_VERSION must be >= 1")


def _check_golden(root: Path, problems: list[str]) -> None:
    p = root / "tests" / "fixtures" / "golden_reference_v1.json"
    payload = json.loads(p.read_text(encoding="utf-8"))
    if int(payload.get("reference_version", 0)) != 1:
        problems.append("golden_reference_v1.json must have reference_version=1")


def _check_changelog(root: Path, problems: list[str]) -> None:
    text = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    if "## [Unreleased]" not in text:
        problems.append("CHANGELOG.md must contain an [Unreleased] section")


def _check_ruff(root: Path, problems: list[str]) -> None:
    cmd = [
        sys.executable,
        "-m",
        "ruff",
        "check",
        "src",
        "tests",
        "scripts",
        "tools",
    ]
    proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    if proc.returncode != 0:
        diagnostics = proc.stdout.strip() or proc.stderr.strip() or "no diagnostics"
        problems.append("Ruff check failed:\n" + diagnostics)


def _check_r_public_parity(root: Path, *, no_r_compare: bool, problems: list[str]) -> None:
    parity_script = root / "tools" / "parity" / "check_r_catalog_parity.py"
    if not parity_script.exists():
        problems.append("Missing tools/parity/check_r_catalog_parity.py")
        return
    cmd = [sys.executable, str(parity_script)]
    if no_r_compare:
        cmd.append("--no-r-compare")
    proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    if proc.returncode != 0:
        diagnostics = proc.stdout.strip() or proc.stderr.strip() or "no diagnostics"
        problems.append("R public feature parity check failed:\n" + diagnostics)
    elif proc.stdout.strip():
        print(proc.stdout.strip())


def _check_alignment_gate(root: Path, report_path: Path, problems: list[str]) -> None:
    gate_script = root / "scripts" / "check_alignment_benchmark_gate.py"
    if not gate_script.exists():
        problems.append("Missing scripts/check_alignment_benchmark_gate.py")
        return
    cmd = [
        sys.executable,
        str(gate_script),
        "--report",
        str(report_path),
    ]
    proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    if proc.returncode != 0:
        problems.append(
            "Alignment benchmark hard gate failed:\n"
            + (proc.stdout.strip() or proc.stderr.strip() or "no diagnostics")
        )
    elif proc.stdout.strip():
        print("release-check: alignment gate diagnostics:")
        print(proc.stdout.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--with-tests", action="store_true", help="Run full pytest suite after static checks")
    parser.add_argument(
        "--no-r-compare",
        action="store_true",
        help="Run Python parity checks without comparing to a live ~/code/natfeatures checkout",
    )
    parser.add_argument(
        "--alignment-report",
        type=Path,
        default=None,
        help="Optional alignment benchmark JSON report to evaluate against hard gates",
    )
    args = parser.parse_args()

    root = args.root
    problems: list[str] = []

    _check_files(root, problems)
    if not problems:
        _check_api_compat(root, problems)
        _check_golden(root, problems)
        _check_changelog(root, problems)
        _check_ruff(root, problems)
        no_r_compare = args.no_r_compare or os.environ.get("NF_PARITY_NO_R_COMPARE", "").strip() in {
            "1",
            "true",
            "TRUE",
            "yes",
            "YES",
        }
        _check_r_public_parity(root, no_r_compare=no_r_compare, problems=problems)
        report_arg = args.alignment_report
        report_env = os.environ.get("NF_ALIGNMENT_BENCHMARK_REPORT", "").strip()
        report_path = report_arg or (Path(report_env) if report_env else None)
        if report_path is not None:
            if not report_path.exists():
                problems.append(f"Alignment report not found: {report_path}")
            else:
                _check_alignment_gate(root, report_path, problems)

    if problems:
        for p in problems:
            print(f"ERROR: {p}")
        return 1

    print("release-check: static checks passed")
    if args.with_tests:
        cmd = ["uv", "run", "pytest", "-q"]
        print("release-check: running", " ".join(cmd))
        rc = subprocess.run(cmd, cwd=root).returncode
        return int(rc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
