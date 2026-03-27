"""ASR wrappers with transcript fallback behavior."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from natural_features.core.execution import add_execution_provenance, resolve_execution_mode
from natural_features.core.feature_types import EventSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.features.common import extractor_metadata
from natural_features.features.speech.chunking import (
    aggregate_chunk_qc,
    plan_audio_chunks,
    stitch_word_events,
)
from natural_features.features.speech.contracts import (
    ensure_word_event_metadata,
    normalize_alignment_qc,
)


def _tokenize(text: str) -> list[str]:
    return [w for w in re.split(r"\s+", text.strip()) if w]


def _align_words_uniform(
    words: list[str],
    *,
    start_s: float,
    end_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not words:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    duration = max(end_s - start_s, 1e-6)
    step = duration / len(words)
    onset = start_s + np.arange(len(words), dtype=np.float64) * step
    offset = onset + step
    return onset, offset


def _coverage_fraction(onset_s: np.ndarray, offset_s: np.ndarray, *, start_s: float, end_s: float) -> float:
    duration = max(float(end_s - start_s), 1e-8)
    covered = float(np.clip(np.sum(np.maximum(0.0, offset_s - onset_s)), 0.0, duration))
    return covered / duration


def _from_transcript_text(
    transcript_text: str,
    *,
    start_s: float,
    end_s: float,
    metadata: dict[str, Any],
    asr_model_name: str,
) -> dict[str, Any]:
    words = _tokenize(transcript_text)
    onset, offset = _align_words_uniform(words, start_s=start_s, end_s=end_s)
    md = ensure_word_event_metadata(
        metadata,
        asr_model_name=asr_model_name,
        aligner_backend="provided_transcript_uniform",
        aligner_version="none",
    )
    segments = EventSeries(
        onset_s=np.array([start_s], dtype=np.float64),
        offset_s=np.array([end_s], dtype=np.float64),
        label=np.array([transcript_text], dtype=object),
        confidence=np.array([1.0], dtype=np.float32),
        metadata=md,
    )
    word_events = EventSeries(
        onset_s=onset,
        offset_s=offset,
        label=np.array(words, dtype=object),
        confidence=np.ones(len(words), dtype=np.float32),
        metadata=md,
    )
    qc = normalize_alignment_qc(
        {
            "mode": "provided_transcript_uniform_alignment",
            "execution_mode": metadata.get("execution_mode", "fallback"),
            "fallback_used": False,
            "n_words": len(words),
            "dropped_words": 0,
            "low_confidence_words": 0,
            "coverage_fraction": _coverage_fraction(onset, offset, start_s=start_s, end_s=end_s),
        }
    )
    return {"segments": segments, "words": word_events, "qc": qc}


def _fallback_asr(
    stimulus: AudioStimulus,
    *,
    metadata: dict[str, Any],
    execution_mode: str,
    reason: str,
    asr_model_name: str,
) -> dict[str, Any]:
    start_s = stimulus.start_offset_s
    end_s = stimulus.start_offset_s + (stimulus.samples.shape[0] / stimulus.sr_hz)
    md = ensure_word_event_metadata(
        add_execution_provenance(
            metadata,
            execution_mode=execution_mode,
            fallback_used=True,
            fallback_reason=reason,
        ),
        asr_model_name=asr_model_name,
        aligner_backend="none",
        aligner_version="none",
    )
    segment_text = "[ASR unavailable]"
    segments = EventSeries(
        onset_s=np.array([start_s], dtype=np.float64),
        offset_s=np.array([end_s], dtype=np.float64),
        label=np.array([segment_text], dtype=object),
        confidence=np.array([0.0], dtype=np.float32),
        metadata=md,
    )
    words = EventSeries(
        onset_s=np.array([start_s], dtype=np.float64),
        offset_s=np.array([end_s], dtype=np.float64),
        label=np.array(["UNK"], dtype=object),
        confidence=np.array([0.0], dtype=np.float32),
        metadata=md,
    )
    qc = normalize_alignment_qc(
        {
            "mode": "fallback",
            "execution_mode": execution_mode,
            "fallback_used": True,
            "n_words": 1,
            "dropped_words": 0,
            "low_confidence_words": 1,
            "reason": reason,
            "coverage_fraction": 1.0,
        }
    )
    return {"segments": segments, "words": words, "qc": qc}


def whisper_transcribe(
    stimulus: AudioStimulus,
    *,
    transcript_text: str | None = None,
    model: str = "small",
    language: str = "auto",
    word_timestamps: bool = True,
    device: str = "auto",
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
) -> dict[str, Any]:
    mode, strict_dependency = resolve_execution_mode(
        execution_mode=execution_mode,
        strict_dependency=strict_dependency,
    )
    params = {
        "model": model,
        "language": language,
        "word_timestamps": word_timestamps,
        "transcript_provided": transcript_text is not None,
        "device": device,
    }
    metadata = add_execution_provenance(
        extractor_metadata(
            "speech.asr.whisper",
            params=params,
            model_revision=str(model),
        ),
        execution_mode=mode,
        fallback_used=False,
    )
    start_s = stimulus.start_offset_s
    end_s = stimulus.start_offset_s + (stimulus.samples.shape[0] / stimulus.sr_hz)

    if transcript_text is not None:
        return _from_transcript_text(
            transcript_text,
            start_s=start_s,
            end_s=end_s,
            metadata=metadata,
            asr_model_name=model,
        )

    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError:
        if strict_dependency:
            raise RuntimeError("faster-whisper is not installed. Install optional dependency and retry.")
        return _fallback_asr(
            stimulus,
            metadata=metadata,
            execution_mode=mode,
            reason="faster-whisper unavailable",
            asr_model_name=model,
        )

    wav = stimulus.samples.astype(np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if device == "auto":
        try:
            import torch

            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            resolved_device = "cpu"
    else:
        resolved_device = device
    try:
        whisper_model = WhisperModel(model, device=resolved_device)
        segments_iter, _info = whisper_model.transcribe(
            wav,
            language=None if language == "auto" else language,
            word_timestamps=word_timestamps,
        )
    except Exception as exc:
        if strict_dependency:
            raise RuntimeError("faster-whisper inference failed in strict mode.") from exc
        return _fallback_asr(
            stimulus,
            metadata=metadata,
            execution_mode=mode,
            reason=f"faster-whisper inference failed: {type(exc).__name__}",
            asr_model_name=model,
        )
    seg_on = []
    seg_off = []
    seg_label = []
    seg_conf = []
    word_on = []
    word_off = []
    word_label = []
    word_conf = []
    for seg in segments_iter:
        seg_on.append(float(seg.start) + start_s)
        seg_off.append(float(seg.end) + start_s)
        seg_label.append(seg.text.strip())
        seg_conf.append(float(getattr(seg, "avg_logprob", 0.0)))
        for w in getattr(seg, "words", []) or []:
            word_on.append(float(w.start) + start_s)
            word_off.append(float(w.end) + start_s)
            word_label.append(str(w.word).strip())
            prob = float(getattr(w, "probability", 0.0))
            word_conf.append(prob)
    del whisper_model
    if not word_on:
        if strict_dependency:
            raise RuntimeError("ASR returned no word timestamps in strict mode.")
        return _fallback_asr(
            stimulus,
            metadata=metadata,
            execution_mode=mode,
            reason="ASR returned no word timestamps",
            asr_model_name=model,
        )
    md = ensure_word_event_metadata(
        metadata,
        asr_model_name=model,
        aligner_backend="none",
        aligner_version="none",
    )
    segments = EventSeries(
        onset_s=np.asarray(seg_on, dtype=np.float64),
        offset_s=np.asarray(seg_off, dtype=np.float64),
        label=np.asarray(seg_label, dtype=object),
        confidence=np.asarray(seg_conf, dtype=np.float32),
        metadata=md,
    )
    words = EventSeries(
        onset_s=np.asarray(word_on, dtype=np.float64),
        offset_s=np.asarray(word_off, dtype=np.float64),
        label=np.asarray(word_label, dtype=object),
        confidence=np.asarray(word_conf, dtype=np.float32),
        metadata=md,
    )
    qc = normalize_alignment_qc(
        {
            "mode": "faster_whisper",
            "execution_mode": mode,
            "fallback_used": False,
            "n_words": len(word_label),
            "dropped_words": 0,
            "low_confidence_words": int(np.sum(np.asarray(word_conf) < 0.4)),
            "coverage_fraction": _coverage_fraction(
                np.asarray(word_on, dtype=np.float64),
                np.asarray(word_off, dtype=np.float64),
                start_s=start_s,
                end_s=end_s,
            ),
        }
    )
    return {"segments": segments, "words": words, "qc": qc}


def whisper_transcribe_chunked(
    stimulus: AudioStimulus,
    *,
    transcript_text: str | None = None,
    model: str = "small",
    language: str = "auto",
    word_timestamps: bool = True,
    device: str = "auto",
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
    chunk_window_s: float = 30.0,
    chunk_overlap_s: float = 1.0,
    dedupe_tolerance_s: float = 0.03,
) -> dict[str, Any]:
    """Chunked ASR path with deterministic stitch and QC aggregation."""

    if transcript_text is not None:
        # Keep transcript-driven deterministic path unchunked.
        return whisper_transcribe(
            stimulus,
            transcript_text=transcript_text,
            model=model,
            language=language,
            word_timestamps=word_timestamps,
            device=device,
            execution_mode=execution_mode,
            strict_dependency=strict_dependency,
        )

    mode, strict_flag = resolve_execution_mode(
        execution_mode=execution_mode,
        strict_dependency=strict_dependency,
    )
    chunks = plan_audio_chunks(
        n_samples=stimulus.samples.shape[0],
        sr_hz=stimulus.sr_hz,
        window_s=chunk_window_s,
        overlap_s=chunk_overlap_s,
        start_offset_s=stimulus.start_offset_s,
    )
    if len(chunks) <= 1:
        return whisper_transcribe(
            stimulus,
            transcript_text=None,
            model=model,
            language=language,
            word_timestamps=word_timestamps,
            device=device,
            execution_mode=mode,
            strict_dependency=strict_flag,
        )

    seg_parts: list[EventSeries] = []
    word_parts: list[EventSeries] = []
    chunk_qc: list[dict[str, Any]] = []
    for ch in chunks:
        sub = AudioStimulus(
            samples=stimulus.samples[ch.sample_start : ch.sample_end],
            sr_hz=stimulus.sr_hz,
            start_offset_s=ch.start_s,
            source=stimulus.source,
        )
        res = whisper_transcribe(
            sub,
            transcript_text=None,
            model=model,
            language=language,
            word_timestamps=word_timestamps,
            device=device,
            execution_mode=mode,
            strict_dependency=strict_flag,
        )
        seg_parts.append(res["segments"])
        word_parts.append(res["words"])
        chunk_qc.append(dict(res.get("qc", {})))

    stitched_words, word_conflicts = stitch_word_events(word_parts, dedupe_tolerance_s=dedupe_tolerance_s)
    stitched_segments, seg_conflicts = stitch_word_events(seg_parts, dedupe_tolerance_s=dedupe_tolerance_s)
    md = ensure_word_event_metadata(
        add_execution_provenance(
            extractor_metadata(
                "speech.asr.whisper_chunked",
                params={
                    "model": model,
                    "language": language,
                    "word_timestamps": word_timestamps,
                    "chunk_window_s": chunk_window_s,
                    "chunk_overlap_s": chunk_overlap_s,
                    "dedupe_tolerance_s": dedupe_tolerance_s,
                },
                model_revision=str(model),
            ),
            execution_mode=mode,
            fallback_used=bool(any(bool(q.get("fallback_used", False)) for q in chunk_qc)),
        ),
        asr_model_name=model,
        aligner_backend="none",
        aligner_version="none",
    )
    words = EventSeries(
        onset_s=stitched_words.onset_s,
        offset_s=stitched_words.offset_s,
        label=stitched_words.label,
        confidence=stitched_words.confidence,
        extra=stitched_words.extra,
        metadata=md,
    )
    segments = EventSeries(
        onset_s=stitched_segments.onset_s,
        offset_s=stitched_segments.offset_s,
        label=stitched_segments.label,
        confidence=stitched_segments.confidence,
        extra=stitched_segments.extra,
        metadata=md,
    )
    qc = aggregate_chunk_qc(
        chunk_qc,
        chunk_count=len(chunks),
        stitch_conflicts=int(word_conflicts + seg_conflicts),
    )
    qc["mode"] = "chunked_asr"
    qc["execution_mode"] = mode
    return {"segments": segments, "words": words, "qc": normalize_alignment_qc(qc)}
