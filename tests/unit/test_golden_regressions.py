from __future__ import annotations

import json
from pathlib import Path

from natural_features.testing_helpers import build_tier_a_golden_reference


def test_tier_a_golden_reference_matches() -> None:
    root = Path(__file__).resolve().parents[2]
    expected_path = root / "tests" / "fixtures" / "golden_reference_v1.json"
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    observed = build_tier_a_golden_reference(root)

    assert observed["reference_version"] == expected["reference_version"]
    assert observed["tier"] == expected["tier"]
    assert observed["stimulus"] == expected["stimulus"]
    assert expected["runtime"]["numpy"]
    assert expected["runtime"]["python"]

    assert observed["visual_energy"] == expected["visual_energy"]
    assert observed["mfcc"] == expected["mfcc"]
    assert observed["acoustic_phonetics"] == expected["acoustic_phonetics"]
    assert observed["multiscale_language"] == expected["multiscale_language"]
    assert observed["audio_batch"] == expected["audio_batch"]
