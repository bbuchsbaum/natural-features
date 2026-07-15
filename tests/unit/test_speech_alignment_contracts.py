from __future__ import annotations

import sys
import types
from unittest.mock import patch

import numpy as np
import pytest

from natural_features.core.feature_types import EventSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.features.common import extractor_metadata
from natural_features.features.speech.align import whisperx_align
from natural_features.features.speech.asr import whisper_transcribe
from natural_features.features.speech.backends import probe_alignment_backends, resolve_aligner_backend
from natural_features.features.speech.contracts import normalize_alignment_qc, validate_alignment_qc
from natural_features.features.speech.phonology import phoneme_event_series, phoneme_events_from_words
from natural_features.features.speech.phonology import articulatory_from_phoneme_events


def _audio() -> AudioStimulus:
    sr = 8000
    t = np.arange(sr * 2, dtype=np.float32) / sr
    x = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return AudioStimulus.from_array(x, sr_hz=sr)


def test_alignment_qc_contract_normalizes_required_fields() -> None:
    qc = normalize_alignment_qc({"mode": "x"})
    assert qc["mode"] == "x"
    assert qc["fallback_used"] is False
    assert qc["n_words"] == 0
    assert qc["low_confidence_words"] == 0
    assert qc["dropped_words"] == 0
    validate_alignment_qc(qc)


def test_alignment_qc_validation_rejects_missing_required() -> None:
    with pytest.raises(ValueError, match="missing required fields"):
        validate_alignment_qc({"mode": "x", "n_words": 1})


def test_backend_probe_and_resolution_contract() -> None:
    probes = probe_alignment_backends()
    assert set(probes.keys()) == {"whisperx", "mfa", "gentle"}
    for name, probe in probes.items():
        assert probe.name == name
        assert isinstance(probe.available, bool)

    forced = resolve_aligner_backend(requested="none")
    assert forced.selected_backend == "passthrough"
    assert forced.fallback_used is False

    auto = resolve_aligner_backend(requested="auto")
    assert auto.selected_backend in {"whisperx", "mfa", "passthrough"}


def test_asr_contract_includes_metadata_and_qc_fields() -> None:
    a = _audio()
    out = whisper_transcribe(a, transcript_text="hello world", strict_dependency=False)
    words = out["words"]
    qc = out["qc"]
    assert words.metadata["asr_model_name"] == "small"
    assert "aligner_backend" in words.metadata
    assert "aligner_version" in words.metadata
    for k in ("mode", "fallback_used", "n_words", "low_confidence_words", "dropped_words"):
        assert k in qc


def test_whisperx_align_explicit_passthrough_is_not_reported_as_fallback() -> None:
    a = _audio()
    out = whisper_transcribe(a, transcript_text="hh ah l ow", strict_dependency=False)
    aligned = whisperx_align(a, out["words"], backend="none")
    qc = aligned["qc"]
    words = aligned["words"]
    assert qc["fallback_used"] is False
    assert qc["mode"] == "passthrough_explicit"
    assert "backend_resolution" in qc
    assert words.metadata["aligner_backend"] == "passthrough"


def test_whisperx_align_uses_backend_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    a = _audio()
    words = EventSeries(
        onset_s=np.array([0.0, 0.5], dtype=np.float64),
        offset_s=np.array([0.5, 1.0], dtype=np.float64),
        label=np.array(["hello", "world"], dtype=object),
        confidence=np.array([0.9, 0.8], dtype=np.float32),
        metadata=extractor_metadata("test.words"),
    )

    fake = types.SimpleNamespace()

    def _load_align_model(language_code: str, device: str):
        assert language_code == "en"
        return object(), {"language_code": language_code, "device": device}

    def _align(segments, model_a, metadata, audio, device, return_char_alignments=False):
        assert segments and isinstance(segments, list)
        return {
            "word_segments": [
                {"word": "hello", "start": 0.05, "end": 0.45, "score": 0.95},
                {"word": "world", "start": 0.55, "end": 0.98, "score": 0.92},
            ]
        }

    fake.load_align_model = _load_align_model
    fake.align = _align
    monkeypatch.setitem(sys.modules, "whisperx", fake)

    aligned = whisperx_align(a, words, backend="whisperx", strict_dependency=True)
    out = aligned["words"]
    qc = aligned["qc"]
    assert qc["mode"] == "whisperx"
    assert qc["fallback_used"] is False
    assert out.metadata["aligner_backend"] == "whisperx"
    assert out.onset_s[0] != words.onset_s[0]


def test_mfa_align_uses_backend_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    a = _audio()
    words = EventSeries(
        onset_s=np.array([0.0, 0.5], dtype=np.float64),
        offset_s=np.array([0.5, 1.0], dtype=np.float64),
        label=np.array(["hello", "world"], dtype=object),
        confidence=np.array([0.9, 0.8], dtype=np.float32),
        metadata=extractor_metadata("test.words"),
    )

    from natural_features.features.speech.backends import AlignerResolution, BackendProbe

    probes = {
        "whisperx": BackendProbe(name="whisperx", available=False),
        "mfa": BackendProbe(name="mfa", available=True, version="3.3.9"),
        "gentle": BackendProbe(name="gentle", available=False),
    }
    monkeypatch.setattr(
        "natural_features.features.speech.align.resolve_aligner_backend",
        lambda requested: AlignerResolution(
            selected_backend="mfa",
            fallback_used=False,
            reason=None,
            probes=probes,
        ),
    )

    refined = EventSeries(
        onset_s=np.array([0.05, 0.55], dtype=np.float64),
        offset_s=np.array([0.45, 0.98], dtype=np.float64),
        label=np.array(["hello", "world"], dtype=object),
        confidence=np.array([0.95, 0.92], dtype=np.float32),
        metadata=words.metadata,
    )
    monkeypatch.setattr(
        "natural_features.features.speech.align._refine_words_with_mfa",
        lambda **kwargs: (refined, 0, {"textgrid_path": "/tmp/clip.TextGrid"}),
    )

    aligned = whisperx_align(
        a,
        words,
        backend="mfa",
        mfa_dictionary_path="/tmp/dict.dict",
        mfa_acoustic_model_path="/tmp/acoustic.zip",
        strict_dependency=True,
    )
    out = aligned["words"]
    qc = aligned["qc"]
    assert qc["mode"] == "mfa"
    assert qc["fallback_used"] is False
    assert out.metadata["aligner_backend"] == "mfa"
    assert out.onset_s[0] != words.onset_s[0]
    assert "alignment_details" in qc


def test_mfa_align_requires_config_in_strict_mode() -> None:
    a = _audio()
    out = whisper_transcribe(a, transcript_text="hello world", strict_dependency=False)
    from natural_features.features.speech.backends import AlignerResolution, BackendProbe

    probes = {
        "whisperx": BackendProbe(name="whisperx", available=False),
        "mfa": BackendProbe(name="mfa", available=True, version="3.3.9"),
        "gentle": BackendProbe(name="gentle", available=False),
    }
    with patch(
        "natural_features.features.speech.align.resolve_aligner_backend",
        lambda requested: AlignerResolution(
            selected_backend="mfa",
            fallback_used=False,
            reason=None,
            probes=probes,
        ),
    ):
        with pytest.raises(RuntimeError, match="mfa backend selected but"):
            whisperx_align(
                a,
                out["words"],
                backend="mfa",
                strict_dependency=True,
            )


def test_selected_backend_without_runtime_adapter_fails_in_strict_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a = _audio()
    words = whisper_transcribe(a, transcript_text="hello world")["words"]
    from natural_features.features.speech.backends import AlignerResolution, BackendProbe

    probes = {
        "whisperx": BackendProbe(name="whisperx", available=False),
        "mfa": BackendProbe(name="mfa", available=False),
        "gentle": BackendProbe(name="gentle", available=True, version="legacy"),
    }
    monkeypatch.setattr(
        "natural_features.features.speech.align.resolve_aligner_backend",
        lambda requested: AlignerResolution(
            selected_backend="gentle",
            fallback_used=False,
            reason=None,
            probes=probes,
        ),
    )

    with pytest.raises(RuntimeError, match="No alignment backend"):
        whisperx_align(a, words, backend="gentle")


def test_phoneme_event_series_contract_metadata() -> None:
    ev = phoneme_event_series(
        onset_s=np.array([0.0, 0.1], dtype=np.float64),
        offset_s=np.array([0.1, 0.2], dtype=np.float64),
        labels=np.array(["AH", "T"], dtype=object),
        label_namespace="arpabet",
        namespace_version="cmu-v1",
        source_word_alignment_id="w123",
    )
    assert ev.metadata["label_namespace"] == "arpabet"
    assert ev.metadata["namespace_version"] == "cmu-v1"
    assert ev.metadata["source_word_alignment_id"] == "w123"


def test_phoneme_events_from_words_splits_phone_strings() -> None:
    words = EventSeries(
        onset_s=np.array([0.0, 1.0], dtype=np.float64),
        offset_s=np.array([1.0, 2.0], dtype=np.float64),
        label=np.array(["HH AH0 L OW1", "W ER1 L D"], dtype=object),
        confidence=np.array([0.9, 0.8], dtype=np.float32),
        metadata=extractor_metadata("test.words"),
    )
    phones = phoneme_events_from_words(words, label_namespace="arpabet", namespace_version="cmu-v1")
    assert len(phones) == 8
    assert np.all(np.diff(phones.onset_s) >= 0)
    assert phones.metadata["label_namespace"] == "arpabet"
    assert phones.metadata["source_word_alignment_id"] == words.metadata["extractor_id"]


def test_articulatory_from_phoneme_events_contract() -> None:
    phones = phoneme_event_series(
        onset_s=np.array([0.0, 0.1, 0.2], dtype=np.float64),
        offset_s=np.array([0.1, 0.2, 0.3], dtype=np.float64),
        labels=np.array(["P", "AH0", "T"], dtype=object),
        confidence=np.array([0.8, 0.9, 0.7], dtype=np.float32),
        label_namespace="arpabet",
        namespace_version="cmu-v1",
        source_word_alignment_id="src1",
    )
    art = articulatory_from_phoneme_events(phones)
    names = list(art.coords.get("feature", []))
    ix = {n: i for i, n in enumerate(names)}
    assert art.values.shape[0] == 3
    assert art.values[0, ix["bilabial"]] > 0
    assert art.values[1, ix["vowel"]] > 0
    assert art.values[2, ix["alveolar"]] > 0
    assert "event_confidence" in ix


def test_non_english_transcript_passthrough_contract() -> None:
    a = _audio()
    transcript = "hola mundo esto es una prueba"
    asr = whisper_transcribe(
        a,
        transcript_text=transcript,
        language="es",
        strict_dependency=False,
    )
    aligned = whisperx_align(
        a,
        asr["words"],
        backend="none",
        language="es",
        strict_dependency=False,
    )
    out = aligned["words"]
    labels = [str(x) for x in np.asarray(out.label, dtype=object)]
    assert labels == transcript.split()
    assert out.metadata["asr_model_name"] == "small"
    assert out.metadata["aligner_backend"] == "passthrough"
    assert aligned["qc"]["mode"] == "passthrough_explicit"
