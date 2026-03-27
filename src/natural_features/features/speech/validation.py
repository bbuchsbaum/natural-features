"""Runtime backend validation utilities for speech alignment stacks."""

from __future__ import annotations

from datetime import datetime, timezone
import platform
import shutil
import subprocess
from typing import Any

from natural_features.core.feature_types import EventSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.features.speech.align import whisperx_align
from natural_features.features.speech.asr import whisper_transcribe
from natural_features.features.speech.backends import BackendProbe, probe_alignment_backends
from natural_features.features.speech.runtime_pins import runtime_pin_metadata


def _probe_payload(probe: BackendProbe) -> dict[str, Any]:
    return {
        "available": bool(probe.available),
        "version": probe.version,
        "reason": probe.reason,
    }


def _runtime_check_whisperx(
    *,
    probe: BackendProbe,
    audio: AudioStimulus | None,
    words: EventSeries | None,
    transcript_text: str | None,
    language: str,
    execution_mode: str,
) -> tuple[bool, bool | None, str | None, dict[str, Any]]:
    details: dict[str, Any] = {}
    if not probe.available:
        return False, None, probe.reason or "whisperx unavailable", details
    if audio is None:
        return False, None, "no audio provided; runtime check skipped", details
    check_words = words
    if check_words is None and transcript_text is not None:
        asr = whisper_transcribe(
            audio,
            transcript_text=transcript_text,
            model="small",
            language=language,
            execution_mode=execution_mode,
            strict_dependency=False,
        )
        check_words = asr["words"]
        details["word_source"] = "transcript_uniform_alignment"
    if check_words is None:
        return False, None, "no words or transcript provided for whisperx runtime check", details
    try:
        out = whisperx_align(
            audio,
            check_words,
            backend="whisperx",
            language=language,
            execution_mode=execution_mode,
            strict_dependency=False,
        )
    except Exception as exc:
        return True, False, f"whisperx runtime check failed: {type(exc).__name__}", details
    qc = dict(out.get("qc", {}))
    details["qc"] = qc
    ok = bool((qc.get("mode") == "whisperx") and (not qc.get("fallback_used", True)))
    reason = None if ok else f"whisperx alignment fallback detected: mode={qc.get('mode', 'unknown')}"
    return True, ok, reason, details


def _runtime_check_mfa(*, probe: BackendProbe, timeout_s: float) -> tuple[bool, bool | None, str | None, dict[str, Any]]:
    details: dict[str, Any] = {}
    if not probe.available:
        return False, None, probe.reason or "mfa unavailable", details
    mfa_path = shutil.which("mfa")
    if not mfa_path:
        return True, False, "mfa executable not found on PATH", details
    try:
        proc = subprocess.run(
            [mfa_path, "version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=max(float(timeout_s), 1.0),
        )
    except Exception as exc:
        return True, False, f"mfa runtime check failed: {type(exc).__name__}", details
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    details["returncode"] = int(proc.returncode)
    details["stdout"] = out
    details["stderr"] = err
    ok = bool(proc.returncode == 0 and (out or err))
    reason = None if ok else "mfa version command failed"
    return True, ok, reason, details


def _runtime_check_gentle(*, probe: BackendProbe) -> tuple[bool, bool | None, str | None, dict[str, Any]]:
    details: dict[str, Any] = {}
    if not probe.available:
        return False, None, probe.reason or "gentle unavailable", details
    try:
        import gentle as _gentle  # type: ignore
    except Exception as exc:
        return True, False, f"gentle import failed: {type(exc).__name__}", details
    details["module"] = getattr(_gentle, "__name__", "gentle")
    return True, True, None, details


def validate_alignment_backends(
    *,
    audio: AudioStimulus | None = None,
    words: EventSeries | None = None,
    transcript_text: str | None = None,
    language: str = "en",
    execution_mode: str = "fallback",
    timeout_s: float = 10.0,
) -> dict[str, Any]:
    """Validate alignment backend availability and runtime readiness.

    Runtime checks are best-effort and non-throwing by default. When possible, whisperx
    is validated with an actual alignment pass (audio + words/transcript).
    """

    probes = probe_alignment_backends()
    payload: dict[str, Any] = {
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "runtime_inputs": {
            "audio_provided": audio is not None,
            "words_provided": words is not None,
            "transcript_provided": transcript_text is not None,
            "language": language,
            "execution_mode": execution_mode,
        },
        "runtime_pin_metadata": runtime_pin_metadata(),
        "backends": {},
    }

    whisperx_probe = probes["whisperx"]
    checked, ok, reason, details = _runtime_check_whisperx(
        probe=whisperx_probe,
        audio=audio,
        words=words,
        transcript_text=transcript_text,
        language=language,
        execution_mode=execution_mode,
    )
    payload["backends"]["whisperx"] = {
        **_probe_payload(whisperx_probe),
        "runtime_checked": checked,
        "runtime_ok": ok,
        "runtime_reason": reason,
        "runtime_details": details,
    }

    mfa_probe = probes["mfa"]
    checked, ok, reason, details = _runtime_check_mfa(probe=mfa_probe, timeout_s=timeout_s)
    payload["backends"]["mfa"] = {
        **_probe_payload(mfa_probe),
        "runtime_checked": checked,
        "runtime_ok": ok,
        "runtime_reason": reason,
        "runtime_details": details,
    }

    gentle_probe = probes["gentle"]
    checked, ok, reason, details = _runtime_check_gentle(probe=gentle_probe)
    payload["backends"]["gentle"] = {
        **_probe_payload(gentle_probe),
        "runtime_checked": checked,
        "runtime_ok": ok,
        "runtime_reason": reason,
        "runtime_details": details,
    }
    return payload
