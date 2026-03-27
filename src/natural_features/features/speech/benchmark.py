"""Corpus-level benchmarking for ASR/alignment quality."""

from __future__ import annotations

from dataclasses import dataclass
import difflib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from natural_features.core.feature_types import EventSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.features.speech.align import whisperx_align
from natural_features.features.speech.asr import whisper_transcribe
from natural_features.features.speech.formats import read_ctm, read_textgrid
from natural_features.features.speech.runtime_pins import runtime_pin_metadata


def _norm_token(x: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w']+", " ", str(x).strip().lower())).strip()


def _event_labels(events: EventSeries) -> list[str]:
    if events.label is None:
        return [""] * len(events)
    return [str(x) for x in np.asarray(events.label, dtype=object)]


def match_token_pairs(reference: EventSeries, predicted: EventSeries) -> list[tuple[int, int]]:
    """Match token indices with sequence alignment over normalized labels."""

    ref = [_norm_token(x) for x in _event_labels(reference)]
    pred = [_norm_token(x) for x in _event_labels(predicted)]
    matcher = difflib.SequenceMatcher(a=ref, b=pred, autojunk=False)
    pairs: list[tuple[int, int]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            continue
        for k in range(i2 - i1):
            pairs.append((i1 + k, j1 + k))
    return pairs


def _boundary_errors_ms(reference: EventSeries, predicted: EventSeries, pairs: list[tuple[int, int]]) -> np.ndarray:
    if not pairs:
        return np.array([], dtype=np.float64)
    errs: list[float] = []
    for ref_i, pred_i in pairs:
        errs.append(abs(float(reference.onset_s[ref_i]) - float(predicted.onset_s[pred_i])) * 1000.0)
        errs.append(abs(float(reference.offset_s[ref_i]) - float(predicted.offset_s[pred_i])) * 1000.0)
    return np.asarray(errs, dtype=np.float64)


def _safe_float(v: float) -> float | None:
    if not np.isfinite(v):
        return None
    return float(v)


def _event_coverage_fraction(events: EventSeries, *, start_s: float, end_s: float) -> float:
    duration = max(float(end_s - start_s), 1e-8)
    covered = float(np.clip(np.sum(np.maximum(0.0, events.offset_s - events.onset_s)), 0.0, duration))
    return covered / duration


def benchmark_alignment_case(
    *,
    clip_id: str,
    audio: AudioStimulus,
    reference_words: EventSeries,
    transcript_text: str | None = None,
    backend: str = "auto",
    asr_model: str = "small",
    language: str = "en",
    execution_mode: str = "fallback",
    strict_dependency: bool = False,
) -> dict[str, Any]:
    """Run ASR+alignment on one clip and compute quality metrics versus reference."""

    asr = whisper_transcribe(
        audio,
        transcript_text=transcript_text,
        model=asr_model,
        language=language,
        execution_mode=execution_mode,
        strict_dependency=strict_dependency,
    )
    aligned = whisperx_align(
        audio,
        asr["words"],
        backend=backend,
        language=language,
        execution_mode=execution_mode,
        strict_dependency=strict_dependency,
    )
    predicted_words = aligned["words"]
    pairs = match_token_pairs(reference_words, predicted_words)
    boundary_ms = _boundary_errors_ms(reference_words, predicted_words, pairs)
    onset_ms = boundary_ms[0::2] if boundary_ms.size else np.array([], dtype=np.float64)
    offset_ms = boundary_ms[1::2] if boundary_ms.size else np.array([], dtype=np.float64)
    duration_s = float(audio.samples.shape[0] / audio.sr_hz)

    n_ref = int(len(reference_words))
    n_pred = int(len(predicted_words))
    n_matched = int(len(pairs))
    token_precision = float(n_matched / max(1, n_pred))
    token_recall = float(n_matched / max(1, n_ref))
    out: dict[str, Any] = {
        "clip_id": clip_id,
        "backend_requested": backend,
        "align_mode": aligned.get("qc", {}).get("mode", "unknown"),
        "fallback_used": bool(aligned.get("qc", {}).get("fallback_used", False)),
        "n_reference_words": n_ref,
        "n_predicted_words": n_pred,
        "n_matched_tokens": n_matched,
        "token_precision": token_precision,
        "token_recall": token_recall,
        "token_f1": float((2 * token_precision * token_recall) / max(token_precision + token_recall, 1e-8)),
        "onset_mae_ms": _safe_float(float(np.mean(onset_ms))) if onset_ms.size else None,
        "offset_mae_ms": _safe_float(float(np.mean(offset_ms))) if offset_ms.size else None,
        "boundary_mae_ms": _safe_float(float(np.mean(boundary_ms))) if boundary_ms.size else None,
        "boundary_jitter_ms_p50": _safe_float(float(np.percentile(boundary_ms, 50.0))) if boundary_ms.size else None,
        "boundary_jitter_ms_p95": _safe_float(float(np.percentile(boundary_ms, 95.0))) if boundary_ms.size else None,
        "coverage_fraction_reference": _event_coverage_fraction(
            reference_words, start_s=audio.start_offset_s, end_s=audio.start_offset_s + duration_s
        ),
        "coverage_fraction_predicted": _event_coverage_fraction(
            predicted_words, start_s=audio.start_offset_s, end_s=audio.start_offset_s + duration_s
        ),
        "asr_qc": asr.get("qc", {}),
        "align_qc": aligned.get("qc", {}),
    }
    return out


def _resolve_path(root: Path, raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    return root / p


def _load_reference_words(item: dict[str, Any], root: Path) -> EventSeries:
    if item.get("reference_ctm"):
        return read_ctm(_resolve_path(root, str(item["reference_ctm"])))
    if item.get("reference_textgrid"):
        return read_textgrid(_resolve_path(root, str(item["reference_textgrid"])))
    raise ValueError("Benchmark item must provide one of {'reference_ctm','reference_textgrid'}")


def _load_transcript(item: dict[str, Any], root: Path) -> str | None:
    if item.get("transcript") is not None:
        txt = str(item.get("transcript", "")).strip()
        return txt if txt else None
    if item.get("transcript_path"):
        p = _resolve_path(root, str(item["transcript_path"]))
        return p.read_text(encoding="utf-8").strip()
    return None


@dataclass(frozen=True)
class BenchmarkConfig:
    backend: str = "auto"
    asr_model: str = "small"
    language: str = "en"
    execution_mode: str = "fallback"
    strict_dependency: bool = False
    continue_on_error: bool = True


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    def _vals(key: str) -> np.ndarray:
        vals = [float(r[key]) for r in ok if r.get(key) is not None]
        return np.asarray(vals, dtype=np.float64) if vals else np.array([], dtype=np.float64)

    boundary = _vals("boundary_mae_ms")
    onset = _vals("onset_mae_ms")
    offset = _vals("offset_mae_ms")
    token_f1 = _vals("token_f1")
    fallback_rate = float(sum(1 for r in ok if bool(r.get("fallback_used", False))) / max(1, len(ok)))
    mode_counts: dict[str, int] = {}
    for r in ok:
        mode = str(r.get("align_mode", "unknown"))
        mode_counts[mode] = mode_counts.get(mode, 0) + 1

    return {
        "n_items": len(results),
        "n_success": len(ok),
        "n_failed": len(failed),
        "fallback_rate": fallback_rate,
        "align_mode_counts": mode_counts,
        "boundary_mae_ms_mean": _safe_float(float(np.mean(boundary))) if boundary.size else None,
        "boundary_mae_ms_p50": _safe_float(float(np.percentile(boundary, 50.0))) if boundary.size else None,
        "boundary_mae_ms_p95": _safe_float(float(np.percentile(boundary, 95.0))) if boundary.size else None,
        "onset_mae_ms_mean": _safe_float(float(np.mean(onset))) if onset.size else None,
        "offset_mae_ms_mean": _safe_float(float(np.mean(offset))) if offset.size else None,
        "token_f1_mean": _safe_float(float(np.mean(token_f1))) if token_f1.size else None,
    }


def run_alignment_benchmark(
    manifest: dict[str, Any] | str | Path,
    *,
    root: str | Path | None = None,
    config: BenchmarkConfig | None = None,
) -> dict[str, Any]:
    """Run a benchmark suite from a manifest.

    Manifest shape:
    - ``items``: list of benchmark items
      - ``id`` (optional)
      - ``audio_path`` (required)
      - ``reference_ctm`` or ``reference_textgrid`` (required)
      - ``transcript`` or ``transcript_path`` (optional)
      - ``language`` / ``backend`` overrides (optional)
    """

    cfg = config or BenchmarkConfig()
    if isinstance(manifest, (str, Path)):
        manifest_path = Path(manifest)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        base = Path(root) if root is not None else manifest_path.parent
    else:
        payload = manifest
        base = Path(root) if root is not None else Path(".")

    items = list(payload.get("items", []))
    if not items:
        raise ValueError("Benchmark manifest has no items")

    results: list[dict[str, Any]] = []
    for i, raw in enumerate(items):
        item = dict(raw)
        clip_id = str(item.get("id", f"item_{i:03d}"))
        try:
            audio_path = _resolve_path(base, str(item["audio_path"]))
            audio = AudioStimulus.from_wav(audio_path)
            ref_words = _load_reference_words(item, base)
            transcript = _load_transcript(item, base)
            case = benchmark_alignment_case(
                clip_id=clip_id,
                audio=audio,
                reference_words=ref_words,
                transcript_text=transcript,
                backend=str(item.get("backend", cfg.backend)),
                asr_model=str(item.get("asr_model", cfg.asr_model)),
                language=str(item.get("language", cfg.language)),
                execution_mode=str(item.get("execution_mode", cfg.execution_mode)),
                strict_dependency=bool(item.get("strict_dependency", cfg.strict_dependency)),
            )
        except Exception as exc:
            case = {
                "clip_id": clip_id,
                "error": f"{type(exc).__name__}: {exc}",
            }
            results.append(case)
            if not cfg.continue_on_error:
                raise
            continue
        results.append(case)

    return {
        "manifest_items": len(items),
        "config": {
            "backend": cfg.backend,
            "asr_model": cfg.asr_model,
            "language": cfg.language,
            "execution_mode": cfg.execution_mode,
            "strict_dependency": cfg.strict_dependency,
            "continue_on_error": cfg.continue_on_error,
        },
        "runtime_pin_metadata": runtime_pin_metadata(),
        "summary": _aggregate(results),
        "results": results,
    }
