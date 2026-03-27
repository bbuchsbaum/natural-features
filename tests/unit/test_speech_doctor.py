from __future__ import annotations

from natural_features.features.speech.doctor import build_alignment_doctor_report


def _validation_payload(*, wx: dict, mfa: dict, gentle: dict) -> dict:
    return {
        "validated_at": "2026-01-01T00:00:00+00:00",
        "runtime_inputs": {
            "audio_provided": False,
            "words_provided": False,
            "transcript_provided": False,
            "language": "en",
            "execution_mode": "fallback",
        },
        "backends": {
            "whisperx": wx,
            "mfa": mfa,
            "gentle": gentle,
        },
    }


def test_doctor_reports_missing_backends_and_actions() -> None:
    report = build_alignment_doctor_report(
        _validation_payload(
            wx={
                "available": False,
                "reason": "ModuleNotFoundError: No module named 'whisperx'",
                "runtime_ok": None,
                "runtime_checked": False,
                "runtime_reason": "ModuleNotFoundError: No module named 'whisperx'",
                "runtime_details": {},
            },
            mfa={
                "available": False,
                "reason": "mfa executable not found",
                "runtime_ok": None,
                "runtime_checked": False,
                "runtime_reason": "mfa executable not found",
                "runtime_details": {},
            },
            gentle={
                "available": False,
                "reason": "ModuleNotFoundError: No module named 'gentle'",
                "runtime_ok": None,
                "runtime_checked": False,
                "runtime_reason": "ModuleNotFoundError: No module named 'gentle'",
                "runtime_details": {},
            },
        )
    )
    assert report["health"] in {"degraded", "unavailable"}
    assert "whisperx" in report["blockers"]
    assert "mfa" in report["blockers"]
    assert any("whisperx is not installed" in r["message"] for r in report["recommendations"])
    assert any("MFA executable is not on PATH" in r["message"] for r in report["recommendations"])


def test_doctor_flags_kalpy_runtime_failure() -> None:
    report = build_alignment_doctor_report(
        _validation_payload(
            wx={
                "available": True,
                "reason": None,
                "runtime_ok": True,
                "runtime_checked": True,
                "runtime_reason": None,
                "runtime_details": {},
            },
            mfa={
                "available": True,
                "reason": None,
                "runtime_ok": False,
                "runtime_checked": True,
                "runtime_reason": "mfa version command failed",
                "runtime_details": {"stderr": "ModuleNotFoundError: No module named '_kalpy'"},
            },
            gentle={
                "available": True,
                "reason": None,
                "runtime_ok": True,
                "runtime_checked": True,
                "runtime_reason": None,
                "runtime_details": {},
            },
        )
    )
    assert report["backend_status"]["mfa"] == "broken"
    assert any("_kalpy" in r["message"] for r in report["recommendations"])
    assert any(
        "conda create -n mfa" in action["command"]
        for r in report["recommendations"]
        for action in r.get("actions", [])
    )


def test_doctor_ok_when_whisperx_and_mfa_healthy() -> None:
    report = build_alignment_doctor_report(
        _validation_payload(
            wx={
                "available": True,
                "reason": None,
                "runtime_ok": True,
                "runtime_checked": True,
                "runtime_reason": None,
                "runtime_details": {},
            },
            mfa={
                "available": True,
                "reason": None,
                "runtime_ok": True,
                "runtime_checked": True,
                "runtime_reason": None,
                "runtime_details": {},
            },
            gentle={
                "available": False,
                "reason": "ModuleNotFoundError: No module named 'gentle'",
                "runtime_ok": None,
                "runtime_checked": False,
                "runtime_reason": "ModuleNotFoundError: No module named 'gentle'",
                "runtime_details": {},
            },
        )
    )
    assert report["health"] == "ok"
    assert report["blockers"] == []
