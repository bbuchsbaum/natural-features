"""Dependency doctor utilities for speech alignment backends."""

from __future__ import annotations

from typing import Any

from natural_features.features.speech.validation import validate_alignment_backends


def _contains(value: str | None, token: str) -> bool:
    if value is None:
        return False
    return token.lower() in value.lower()


def _backend_status(info: dict[str, Any]) -> str:
    if not bool(info.get("available", False)):
        return "missing"
    runtime_ok = info.get("runtime_ok", None)
    if runtime_ok is True:
        return "healthy"
    if runtime_ok is False:
        return "broken"
    return "unknown"


def _cmd(title: str, command: str) -> dict[str, Any]:
    return {
        "title": title,
        "command": command,
    }


def _recommend_whisperx(info: dict[str, Any]) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []
    reason = str(info.get("reason", "") or "")
    runtime_reason = str(info.get("runtime_reason", "") or "")
    if not bool(info.get("available", False)):
        if _contains(reason, "ModuleNotFoundError"):
            recs.append(
                {
                    "backend": "whisperx",
                    "severity": "high",
                    "message": "whisperx is not installed in this environment.",
                    "actions": [
                        _cmd("Install whisperx in venv", "uv pip install --python .venv/bin/python whisperx"),
                    ],
                }
            )
        return recs

    if _contains(runtime_reason, "no audio provided"):
        recs.append(
            {
                "backend": "whisperx",
                "severity": "info",
                "message": "whisperx is installed, but runtime alignment was not exercised.",
                "actions": [
                    _cmd(
                        "Run runtime check on real audio",
                        "nf speech-validate-backends --audio-wav <file.wav> --transcript \"...\" --json",
                    ),
                ],
            }
        )
    elif info.get("runtime_ok") is False:
        recs.append(
            {
                "backend": "whisperx",
                "severity": "high",
                "message": f"whisperx runtime check failed: {runtime_reason or 'unknown reason'}",
                "actions": [
                    _cmd(
                        "Re-run with debug output",
                        "nf speech-validate-backends --audio-wav <file.wav> --transcript \"...\" --json",
                    ),
                    _cmd("Check torch backend", "PYTHONPATH=src .venv/bin/python -c \"import torch; print(torch.__version__)\""),
                ],
            }
        )
    return recs


def _recommend_mfa(info: dict[str, Any]) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []
    reason = str(info.get("reason", "") or "")
    runtime_reason = str(info.get("runtime_reason", "") or "")
    stderr = str(info.get("runtime_details", {}).get("stderr", "") or "")
    if not bool(info.get("available", False)):
        if _contains(reason, "executable not found"):
            recs.append(
                {
                    "backend": "mfa",
                    "severity": "high",
                    "message": "MFA executable is not on PATH.",
                    "actions": [
                        _cmd("Install Python package", "uv pip install --python .venv/bin/python montreal-forced-aligner"),
                        _cmd("Ensure venv bin on PATH", "export PATH=\"$(pwd)/.venv/bin:$PATH\""),
                        _cmd("Verify CLI", "mfa version"),
                    ],
                }
            )
        return recs

    if info.get("runtime_ok") is True:
        return recs

    if _contains(stderr, "_kalpy"):
        recs.append(
            {
                "backend": "mfa",
                "severity": "high",
                "message": "MFA runtime failed because kalpy bindings are missing (_kalpy import error).",
                "actions": [
                    _cmd("Bootstrap conda MFA env via repo helper", "./scripts/setup_mfa_conda.sh mfa"),
                    _cmd(
                        "Recommended: use conda-forge MFA stack",
                        "conda create -n mfa python=3.11 montreal-forced-aligner kalpy openfst pynini -c conda-forge",
                    ),
                    _cmd("Activate and verify", "conda activate mfa && mfa version"),
                    _cmd(
                        "Alternative: keep using whisperx backend if MFA stack is unavailable",
                        "nf speech-validate-backends --audio-wav <file.wav> --transcript \"...\" --json",
                    ),
                ],
            }
        )
    else:
        recs.append(
            {
                "backend": "mfa",
                "severity": "high",
                "message": f"MFA runtime check failed: {runtime_reason or 'unknown reason'}",
                "actions": [
                    _cmd("Inspect error details", "nf speech-validate-backends --json"),
                ],
            }
        )
    return recs


def _recommend_gentle(info: dict[str, Any]) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []
    reason = str(info.get("reason", "") or "")
    if not bool(info.get("available", False)):
        if _contains(reason, "ModuleNotFoundError"):
            recs.append(
                {
                    "backend": "gentle",
                    "severity": "info",
                    "message": "gentle is not installed; this backend is optional/legacy.",
                    "actions": [
                        _cmd("Install gentle", "uv pip install --python .venv/bin/python gentle"),
                    ],
                }
            )
    elif info.get("runtime_ok") is False:
        recs.append(
            {
                "backend": "gentle",
                "severity": "medium",
                "message": f"gentle runtime check failed: {info.get('runtime_reason', 'unknown reason')}",
                "actions": [
                    _cmd("Re-run backend validation", "nf speech-validate-backends --json"),
                ],
            }
        )
    return recs


def build_alignment_doctor_report(validation_report: dict[str, Any]) -> dict[str, Any]:
    """Convert backend validation payload into prioritized remediation guidance."""

    backends = dict(validation_report.get("backends", {}))
    wx = dict(backends.get("whisperx", {}))
    mfa = dict(backends.get("mfa", {}))
    gentle = dict(backends.get("gentle", {}))

    recommendations: list[dict[str, Any]] = []
    recommendations.extend(_recommend_whisperx(wx))
    recommendations.extend(_recommend_mfa(mfa))
    recommendations.extend(_recommend_gentle(gentle))

    status = {
        "whisperx": _backend_status(wx),
        "mfa": _backend_status(mfa),
        "gentle": _backend_status(gentle),
    }
    blockers: list[str] = []
    if status["whisperx"] in {"missing", "broken"}:
        blockers.append("whisperx")
    if status["mfa"] in {"missing", "broken"}:
        blockers.append("mfa")

    health = "ok"
    if blockers:
        health = "degraded"
    if all(v == "missing" for v in status.values()):
        health = "unavailable"

    return {
        "health": health,
        "backend_status": status,
        "blockers": blockers,
        "recommendations": recommendations,
        "validation_report": validation_report,
    }


def run_alignment_doctor(
    *,
    audio=None,
    words=None,
    transcript_text: str | None = None,
    language: str = "en",
    execution_mode: str = "fallback",
    timeout_s: float = 10.0,
) -> dict[str, Any]:
    """Run backend validation and attach remediation guidance."""

    validation = validate_alignment_backends(
        audio=audio,
        words=words,
        transcript_text=transcript_text,
        language=language,
        execution_mode=execution_mode,
        timeout_s=timeout_s,
    )
    return build_alignment_doctor_report(validation)
