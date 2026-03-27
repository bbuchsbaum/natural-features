#!/usr/bin/env python3
"""Generate deterministic golden regression references for Tier A stimuli."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from natural_features.testing_helpers import build_tier_a_golden_reference


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "golden_reference_v1.json",
    )
    args = parser.parse_args()

    ref = build_tier_a_golden_reference(args.base_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(ref, indent=2, sort_keys=True), encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
