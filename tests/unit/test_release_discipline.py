from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_release_check_script_passes() -> None:
    root = Path(__file__).resolve().parents[2]
    cmd = [sys.executable, "scripts/release_check.py", "--root", str(root)]
    proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert "release-check: static checks passed" in proc.stdout
