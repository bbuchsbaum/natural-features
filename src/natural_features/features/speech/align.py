"""Alignment wrappers and QC utilities."""

from __future__ import annotations

import difflib
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any
import wave

import numpy as np

from natural_features.core.execution import add_execution_provenance, resolve_execution_mode
from natural_features.core.feature_types import EventSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.features.common import extractor_metadata
from natural_features.features.speech.backends import resolve_aligner_backend
from natural_features.features.speech.contracts import (
    ensure_word_event_metadata,
    normalize_alignment_qc,
)
from natural_features.features.speech.formats import read_textgrid


def alignment_qc(
    words: EventSeries,
    *,
    confidence_threshold: float = 0.4,
    mode: str = "unknown",
    fallback_used: bool = False,
    dropped_words: int = 0,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    conf = words.confidence if words.confidence is not None else np.ones(len(words), dtype=np.float32)
    low = int(np.sum(conf < confidence_threshold))
    qc = {
        "mode": mode,
        "fallback_used": fallback_used,
        "n_words": len(words),
        "low_confidence_words": low,
        "low_confidence_fraction": float(low / max(1, len(words))),
        "dropped_words": int(dropped_words),
    }
    if extra:
        qc.update(extra)
    return normalize_alignment_qc(qc)


def _coverage_fraction(words: EventSeries, *, start_s: float, end_s: float) -> float:
    duration = max(float(end_s - start_s), 1e-8)
    covered = float(np.clip(np.sum(np.maximum(0.0, words.offset_s - words.onset_s)), 0.0, duration))
    return covered / duration


def _boundary_jitter_ms(before: EventSeries, after: EventSeries) -> tuple[float, float]:
    if len(before) == 0 or len(after) == 0:
        return 0.0, 0.0
    n = min(len(before), len(after))
    delta = np.concatenate(
        [
            np.abs(before.onset_s[:n] - after.onset_s[:n]),
            np.abs(before.offset_s[:n] - after.offset_s[:n]),
        ]
    )
    if delta.size == 0:
        return 0.0, 0.0
    ms = 1000.0 * delta
    return float(np.percentile(ms, 50.0)), float(np.percentile(ms, 95.0))


def _refine_words_with_whisperx(
    *,
    stimulus: AudioStimulus,
    words: EventSeries,
    language: str,
) -> tuple[EventSeries, int]:
    import whisperx  # type: ignore

    wav = stimulus.samples.astype(np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    labels = words.label if words.label is not None else np.array([""] * len(words), dtype=object)
    conf = words.confidence if words.confidence is not None else np.ones(len(words), dtype=np.float32)
    audio_start = float(stimulus.start_offset_s)
    local_start = max(0.0, float(words.onset_s[0] - audio_start)) if len(words) else 0.0
    local_end = max(local_start, float(words.offset_s[-1] - audio_start)) if len(words) else float(len(wav) / stimulus.sr_hz)
    segment = {
        "start": local_start,
        "end": local_end,
        "text": " ".join(str(x) for x in labels),
    }

    model_a, metadata = whisperx.load_align_model(language_code=language, device="cpu")
    aligned = whisperx.align(
        [segment],
        model_a,
        metadata,
        wav,
        "cpu",
        return_char_alignments=False,
    )
    word_segments = aligned.get("word_segments", []) if isinstance(aligned, dict) else []
    if not word_segments:
        raise RuntimeError("whisperx returned no aligned word segments")

    on = words.onset_s.astype(np.float64, copy=True)
    off = words.offset_s.astype(np.float64, copy=True)
    out_labels = np.asarray(labels, dtype=object).copy()
    out_conf = np.asarray(conf, dtype=np.float32).copy()

    n_refined = min(len(word_segments), len(words))
    for i in range(n_refined):
        item = word_segments[i]
        start = float(item.get("start", on[i] - audio_start)) + audio_start
        end = float(item.get("end", off[i] - audio_start)) + audio_start
        if end < start:
            end = start
        on[i] = start
        off[i] = end
        tok = str(item.get("word", "")).strip()
        if tok:
            out_labels[i] = tok
        score = item.get("score", None)
        if score is not None:
            out_conf[i] = float(score)

    dropped_words = max(0, len(words) - len(word_segments))
    out = EventSeries(
        onset_s=on,
        offset_s=off,
        label=out_labels,
        confidence=out_conf,
        extra=words.extra,
        metadata=words.metadata,
    )
    return out, dropped_words


def _write_audio_wav(stimulus: AudioStimulus, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = stimulus.samples.astype(np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(stimulus.sr_hz))
        w.writeframes(pcm.tobytes())


def _norm_token(x: str) -> str:
    tok = str(x).strip().lower()
    return "".join(ch for ch in tok if ch.isalnum() or ch in {"'", "_"})


def _map_aligned_words_to_reference(
    reference: EventSeries,
    aligned: EventSeries,
) -> tuple[EventSeries, int]:
    ref_labels = (
        [str(x) for x in np.asarray(reference.label, dtype=object)]
        if reference.label is not None
        else ["" for _ in range(len(reference))]
    )
    aln_labels = (
        [str(x) for x in np.asarray(aligned.label, dtype=object)]
        if aligned.label is not None
        else ["" for _ in range(len(aligned))]
    )
    ref_norm = [_norm_token(x) for x in ref_labels]
    aln_norm = [_norm_token(x) for x in aln_labels]
    matcher = difflib.SequenceMatcher(a=ref_norm, b=aln_norm, autojunk=False)
    pairs: list[tuple[int, int]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            continue
        for k in range(i2 - i1):
            pairs.append((i1 + k, j1 + k))

    on = reference.onset_s.astype(np.float64, copy=True)
    off = reference.offset_s.astype(np.float64, copy=True)
    conf = (
        reference.confidence.astype(np.float32, copy=True)
        if reference.confidence is not None
        else np.ones(len(reference), dtype=np.float32)
    )
    for ref_i, aln_i in pairs:
        on[ref_i] = float(aligned.onset_s[aln_i])
        off[ref_i] = float(aligned.offset_s[aln_i])
        if aligned.confidence is not None and aln_i < len(aligned.confidence):
            conf[ref_i] = float(aligned.confidence[aln_i])
    dropped = max(0, len(reference) - len(pairs))
    out = EventSeries(
        onset_s=on,
        offset_s=off,
        label=np.asarray(ref_labels, dtype=object),
        confidence=conf,
        extra=reference.extra,
        metadata=reference.metadata,
    )
    return out, dropped


def _refine_words_with_mfa(
    *,
    stimulus: AudioStimulus,
    words: EventSeries,
    dictionary_path: str,
    acoustic_model_path: str,
    timeout_s: float,
    tmp_dir: str | None,
    extra_args: list[str] | None,
) -> tuple[EventSeries, int, dict[str, Any]]:
    mfa_exe = shutil.which("mfa")
    if not mfa_exe:
        raise RuntimeError("mfa executable not found on PATH")
    root_ctx = tempfile.TemporaryDirectory(dir=tmp_dir) if tmp_dir else tempfile.TemporaryDirectory()
    with root_ctx as td:
        root = Path(td)
        corpus = root / "corpus"
        output = root / "output"
        corpus.mkdir(parents=True, exist_ok=True)
        output.mkdir(parents=True, exist_ok=True)

        wav_path = corpus / "clip.wav"
        txt_path = corpus / "clip.lab"
        _write_audio_wav(stimulus, wav_path)
        labels = words.label if words.label is not None else np.array([""] * len(words), dtype=object)
        txt_path.write_text(" ".join(str(x) for x in labels), encoding="utf-8")

        cmd = [
            mfa_exe,
            "align",
            str(corpus),
            str(dictionary_path),
            str(acoustic_model_path),
            str(output),
            "--clean",
            "--single_speaker",
            "--output_format",
            "long_textgrid",
        ]
        if extra_args:
            cmd.extend([str(x) for x in extra_args])
        try:
            proc = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=max(float(timeout_s), 1.0),
            )
        except subprocess.CalledProcessError as exc:
            err = (exc.stderr or "").strip()
            raise RuntimeError(f"mfa align failed: {err[:800]}") from exc

        grids = sorted(output.rglob("*.TextGrid"))
        if not grids:
            raise RuntimeError("mfa align produced no TextGrid output")
        aligned = read_textgrid(grids[0])
        if aligned.label is not None:
            keep = np.array([bool(str(x).strip()) for x in aligned.label], dtype=bool)
            aligned = EventSeries(
                onset_s=aligned.onset_s[keep],
                offset_s=aligned.offset_s[keep],
                label=np.asarray(aligned.label, dtype=object)[keep],
                confidence=(aligned.confidence[keep] if aligned.confidence is not None else None),
                extra=aligned.extra,
                metadata=aligned.metadata,
            )

        mapped, dropped = _map_aligned_words_to_reference(words, aligned)
        details = {
            "textgrid_path": str(grids[0]),
            "stdout": (proc.stdout or "").strip()[-500:],
            "stderr": (proc.stderr or "").strip()[-500:],
            "command": cmd,
        }
        return mapped, dropped, details


def whisperx_align(
    stimulus: AudioStimulus,
    words: EventSeries,
    *,
    backend: str = "auto",
    language: str = "en",
    mfa_dictionary_path: str | None = None,
    mfa_acoustic_model_path: str | None = None,
    mfa_timeout_s: float = 300.0,
    mfa_tmp_dir: str | None = None,
    mfa_extra_args: list[str] | None = None,
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
) -> dict[str, Any]:
    mode, strict_dependency = resolve_execution_mode(
        execution_mode=execution_mode,
        strict_dependency=strict_dependency,
    )
    resolution = resolve_aligner_backend(requested=backend)
    selected = resolution.selected_backend
    probe = resolution.probes.get(selected if selected != "passthrough" else "whisperx")
    aligner_version = str(probe.version) if probe and probe.version else "none"
    params = {
        "language": language,
        "backend": backend,
        "mfa_configured": bool(mfa_dictionary_path and mfa_acoustic_model_path),
    }
    base_md = extractor_metadata(
        "speech.align.whisperx",
        params=params,
        model_revision=aligner_version,
        extra={
            "requested_backend": backend,
            "selected_backend": selected,
        },
    )
    asr_model_name = str(words.metadata.get("asr_model_name", "unknown"))
    stim_end_s = stimulus.start_offset_s + (stimulus.samples.shape[0] / stimulus.sr_hz)

    def _passthrough(
        reason: str,
        *,
        aligner_backend: str,
        fallback_used: bool = True,
        mode_name: str = "passthrough",
    ) -> dict[str, Any]:
        md = ensure_word_event_metadata(
            add_execution_provenance(
                    {**words.metadata, **base_md},
                    execution_mode=mode,
                    fallback_used=fallback_used,
                    fallback_reason=reason if fallback_used else None,
                backend=aligner_backend,
            ),
            asr_model_name=asr_model_name,
            aligner_backend=aligner_backend,
            aligner_version=aligner_version,
        )
        qc = alignment_qc(
            words,
            mode=mode_name,
            fallback_used=fallback_used,
            extra={
                "reason": reason,
                "execution_mode": mode,
                "backend_resolution": resolution.as_dict(),
                "coverage_fraction": _coverage_fraction(
                    words,
                    start_s=stimulus.start_offset_s,
                    end_s=stim_end_s,
                ),
            },
        )
        passthrough_words = EventSeries(
            onset_s=words.onset_s,
            offset_s=words.offset_s,
            label=words.label,
            confidence=words.confidence,
            extra=words.extra,
            metadata=md,
        )
        return {"words": passthrough_words, "qc": qc}

    if selected not in {"whisperx", "mfa"}:
        explicit_passthrough = selected == "passthrough" and not resolution.fallback_used
        if strict_dependency and not explicit_passthrough:
            raise RuntimeError(resolution.reason or "No alignment backend is available in strict mode.")
        reason = resolution.reason or (
            "explicit passthrough requested"
            if explicit_passthrough
            else f"backend '{selected}' has no runtime adapter"
        )
        return _passthrough(
            reason,
            aligner_backend=selected,
            fallback_used=not explicit_passthrough,
            mode_name="passthrough_explicit" if explicit_passthrough else "passthrough",
        )

    refine_details: dict[str, Any] = {}
    if selected == "whisperx":
        try:
            import whisperx  # type: ignore  # noqa: F401
        except ImportError as exc:
            if strict_dependency:
                raise RuntimeError("whisperx is required for strict alignment mode.") from exc
            return _passthrough("whisperx import failed", aligner_backend="whisperx")
        try:
            refined_words, dropped_words = _refine_words_with_whisperx(
                stimulus=stimulus,
                words=words,
                language=language,
            )
        except Exception as exc:
            if strict_dependency:
                raise RuntimeError("whisperx refinement failed in strict mode.") from exc
            reason = f"whisperx refinement failed: {type(exc).__name__}"
            return _passthrough(reason, aligner_backend="whisperx", mode_name="whisperx_passthrough")
    else:
        if not mfa_dictionary_path or not mfa_acoustic_model_path:
            reason = "mfa backend selected but mfa_dictionary_path/mfa_acoustic_model_path are not configured"
            if strict_dependency:
                raise RuntimeError(reason)
            return _passthrough(reason, aligner_backend="mfa", mode_name="mfa_passthrough")
        try:
            refined_words, dropped_words, refine_details = _refine_words_with_mfa(
                stimulus=stimulus,
                words=words,
                dictionary_path=str(mfa_dictionary_path),
                acoustic_model_path=str(mfa_acoustic_model_path),
                timeout_s=float(mfa_timeout_s),
                tmp_dir=mfa_tmp_dir,
                extra_args=mfa_extra_args,
            )
        except Exception as exc:
            if strict_dependency:
                raise RuntimeError("mfa refinement failed in strict mode.") from exc
            reason = f"mfa refinement failed: {type(exc).__name__}: {exc}"
            return _passthrough(reason, aligner_backend="mfa", mode_name="mfa_passthrough")

    md = ensure_word_event_metadata(
        add_execution_provenance(
            {**refined_words.metadata, **base_md},
            execution_mode=mode,
            fallback_used=False,
            backend=selected,
        ),
        asr_model_name=asr_model_name,
        aligner_backend=selected,
        aligner_version=aligner_version,
    )
    jitter50, jitter95 = _boundary_jitter_ms(words, refined_words)
    refined_words = EventSeries(
        onset_s=refined_words.onset_s,
        offset_s=refined_words.offset_s,
        label=refined_words.label,
        confidence=refined_words.confidence,
        extra=refined_words.extra,
        metadata=md,
    )
    qc = alignment_qc(
        refined_words,
        mode=selected,
        fallback_used=False,
        dropped_words=dropped_words,
        extra={
            "execution_mode": mode,
            "backend_resolution": resolution.as_dict(),
            "coverage_fraction": _coverage_fraction(
                refined_words,
                start_s=stimulus.start_offset_s,
                end_s=stim_end_s,
            ),
            "boundary_jitter_ms_p50": jitter50,
            "boundary_jitter_ms_p95": jitter95,
            "alignment_details": refine_details,
        },
    )
    return {"words": refined_words, "qc": qc}
