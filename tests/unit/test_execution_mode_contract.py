from __future__ import annotations

import numpy as np
import pytest

from natural_features.core.execution import resolve_execution_mode
from natural_features.core.stimulus import AudioStimulus
from natural_features.features.speech.asr import whisper_transcribe
from natural_features.features.speech.phonology import ctc_phone_posteriors
from natural_features.workflows.multiscale_language import extract_multiscale_language


def _audio() -> AudioStimulus:
    sr = 8000
    t = np.arange(sr * 2, dtype=np.float32) / sr
    x = (0.15 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return AudioStimulus.from_array(x, sr_hz=sr)


def test_resolve_execution_mode_defaults_and_conflicts() -> None:
    mode, strict = resolve_execution_mode()
    assert mode == "fallback"
    assert strict is False

    mode, strict = resolve_execution_mode(strict_dependency=True)
    assert mode == "strict"
    assert strict is True

    with pytest.raises(ValueError):
        resolve_execution_mode(execution_mode="strict", strict_dependency=False)


def test_asr_metadata_has_execution_mode() -> None:
    out = whisper_transcribe(_audio(), execution_mode="fallback")
    assert out["words"].metadata.get("execution_mode") == "fallback"
    assert "fallback_used" in out["words"].metadata


def test_ctc_posteriors_mark_fallback_provenance() -> None:
    post = ctc_phone_posteriors(
        _audio(),
        model="__missing__/__missing__",
        local_files_only=True,
        execution_mode="fallback",
    )
    assert post.metadata.get("execution_mode") == "fallback"
    assert post.metadata.get("fallback_used") is True


def test_ctc_posteriors_strict_mode_fails_loudly() -> None:
    with pytest.raises(RuntimeError):
        ctc_phone_posteriors(
            _audio(),
            model="__missing__/__missing__",
            local_files_only=True,
            execution_mode="strict",
        )


def test_multiscale_language_provider_fallback_when_openai_unavailable(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    res = extract_multiscale_language(
        "hello world this is a fallback provider test",
        scales_s=[2.0],
        provider_config={"provider": "openai", "model": "text-embedding-3-large"},
        execution_mode="fallback",
    )
    prov = res.qc["provider_resolution"]
    assert prov["requested_provider"] == "openai"
    assert prov["resolved_provider"] == "local_bow"
    assert prov["fallback_used"] is True


def test_multiscale_language_provider_strict_mode_fails(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        extract_multiscale_language(
            "hello world",
            scales_s=[2.0],
            provider_config={"provider": "openai", "model": "text-embedding-3-large"},
            execution_mode="strict",
        )
