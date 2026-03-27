from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_tier_a_manifest_validator_script() -> None:
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "validate_tier_a_stimuli.py"
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, check=True)
    assert "validated" in proc.stdout.lower()
