from __future__ import annotations

import numpy as np
import pytest

from natural_features.core.feature_types import EventSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.features.common import extractor_metadata
from natural_features.features.speech.backends import BackendProbe
from natural_features.features.speech.validation import (
    _runtime_check_mfa,
    _runtime_check_whisperx,
    validate_alignment_backends,
)


def _md(name: str) -> dict[str, object]:
    return extractor_metadata(name, params={})


def _words() -> EventSeries:
    return EventSeries(
        onset_s=np.array([0.0, 0.5], dtype=np.float64),
        offset_s=np.array([0.4, 0.9], dtype=np.float64),
        label=np.array(["hello", "world"], dtype=object),
        confidence=np.array([0.9, 0.8], dtype=np.float32),
        metadata=_md("test.words"),
    )


def test_runtime_check_whisperx_skips_without_audio() -> None:
    checked, ok, reason, details = _runtime_check_whisperx(
        probe=BackendProbe(name="whisperx", available=True, version="x"),
        audio=None,
        words=None,
        transcript_text=None,
        language="en",
        execution_mode="fallback",
    )
    assert checked is False
    assert ok is None
    assert "no audio" in str(reason)
    assert details == {}


def test_runtime_check_mfa_handles_missing_executable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _: None)
    checked, ok, reason, _details = _runtime_check_mfa(
        probe=BackendProbe(name="mfa", available=True),
        timeout_s=1.0,
    )
    assert checked is True
    assert ok is False
    assert "not found" in str(reason)


def test_validate_alignment_backends_aggregates_status(monkeypatch: pytest.MonkeyPatch) -> None:
    probes = {
        "whisperx": BackendProbe(name="whisperx", available=True, version="1.0"),
        "mfa": BackendProbe(name="mfa", available=False, reason="missing"),
        "gentle": BackendProbe(name="gentle", available=True, version="0.1"),
    }
    monkeypatch.setattr(
        "natural_features.features.speech.validation.probe_alignment_backends",
        lambda: probes,
    )
    monkeypatch.setattr(
        "natural_features.features.speech.validation._runtime_check_whisperx",
        lambda **_: (True, True, None, {"qc": {"mode": "whisperx"}}),
    )
    monkeypatch.setattr(
        "natural_features.features.speech.validation._runtime_check_mfa",
        lambda **_: (False, None, "missing", {}),
    )
    monkeypatch.setattr(
        "natural_features.features.speech.validation._runtime_check_gentle",
        lambda **_: (True, True, None, {}),
    )

    audio = AudioStimulus.from_array(np.zeros(1600, dtype=np.float32), sr_hz=16000)
    payload = validate_alignment_backends(audio=audio, words=_words(), transcript_text=None)
    assert "validated_at" in payload
    assert "runtime_pin_metadata" in payload
    assert "runtime_versions" in payload["runtime_pin_metadata"]
    assert sorted(payload["backends"].keys()) == ["gentle", "mfa", "whisperx"]
    assert payload["backends"]["whisperx"]["runtime_ok"] is True
    assert payload["backends"]["mfa"]["runtime_ok"] is None
    assert payload["runtime_inputs"]["audio_provided"] is True
