"""Text extraction workflows for video stimuli."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
from typing import Any, Sequence

import numpy as np

from natural_features.core.execution import resolve_execution_mode
from natural_features.core.feature_types import EventSeries
from natural_features.core.frame_timeline import FramePolicy, FrameTimeline
from natural_features.core.interchange import as_event_table
from natural_features.core.stimulus import AudioStimulus, VideoStimulus
from natural_features.features.preprocess import video_audio_extract
from natural_features.features.speech.align import whisperx_align
from natural_features.features.speech.asr import (
    whisper_transcribe,
    whisper_transcribe_chunked,
)

VideoInput = VideoStimulus | str | Path


def _audio_duration_s(audio: AudioStimulus) -> float:
    return float(audio.samples.shape[0] / float(audio.sr_hz))


def _video_source(video: VideoInput) -> str | None:
    if isinstance(video, VideoStimulus):
        return video.source
    return str(Path(video))


def _parse_frame_rate(raw: str) -> float | None:
    for token in raw.splitlines():
        token = token.strip()
        if not token or token == "0/0":
            continue
        try:
            if "/" in token:
                num, den = token.split("/", 1)
                value = float(num) / float(den)
            else:
                value = float(token)
        except (ValueError, ZeroDivisionError):
            continue
        if np.isfinite(value) and value > 0:
            return float(value)
    return None


def _probe_video_fps(
    video: VideoInput,
    *,
    explicit_fps: float | None,
    ffprobe_path: str,
    strict: bool,
) -> tuple[float | None, dict[str, Any]]:
    if explicit_fps is not None:
        fps = float(explicit_fps)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("video_fps must be a positive finite value")
        return fps, {"mode": "provided", "fallback_used": False, "fps": fps}

    if isinstance(video, VideoStimulus):
        fps = float(video.fps)
        return fps, {"mode": "video_stimulus", "fallback_used": False, "fps": fps}

    ffprobe = shutil.which(ffprobe_path)
    if ffprobe is None:
        reason = f"ffprobe executable not found: {ffprobe_path}"
        if strict:
            raise RuntimeError(reason)
        return None, {"mode": "unavailable", "fallback_used": True, "reason": reason}

    path = Path(video)
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except Exception as exc:
        reason = f"ffprobe failed: {type(exc).__name__}"
        if strict:
            raise RuntimeError(reason) from exc
        return None, {"mode": "unavailable", "fallback_used": True, "reason": reason}

    fps = _parse_frame_rate(proc.stdout)
    if fps is None:
        reason = "ffprobe returned no usable frame rate"
        if strict:
            raise RuntimeError(reason)
        return None, {"mode": "unavailable", "fallback_used": True, "reason": reason}
    return fps, {"mode": "ffprobe", "fallback_used": False, "fps": fps, "command": cmd}


def _add_word_extras(words: EventSeries, extras: dict[str, Any]) -> EventSeries:
    n = len(words)
    extra = dict(words.extra)
    for key, value in extras.items():
        if isinstance(value, np.ndarray):
            arr = value
        else:
            arr = np.full(n, value, dtype=object)
        if len(arr) != n:
            raise ValueError(
                f"extra column {key!r} has length {len(arr)}, expected {n}"
            )
        extra[key] = arr
    return EventSeries(
        onset_s=words.onset_s,
        offset_s=words.offset_s,
        label=words.label,
        confidence=words.confidence,
        extra=extra,
        metadata=dict(words.metadata),
        schema=words.schema,
        timebase=words.timebase,
    )


def _with_workflow_metadata(
    words: EventSeries, *, source_video: str | None, frame_policy: str
) -> EventSeries:
    metadata = dict(words.metadata)
    metadata.update(
        {
            "workflow_name": "video_text",
            "source_video": source_video,
            "frame_policy": frame_policy,
        }
    )
    return EventSeries(
        onset_s=words.onset_s,
        offset_s=words.offset_s,
        label=words.label,
        confidence=words.confidence,
        extra=dict(words.extra),
        metadata=metadata,
        schema=words.schema,
        timebase=words.timebase,
    )


def _scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class VideoTextResult:
    """Result bundle for speech-derived text aligned to video time."""

    words: EventSeries
    segments: EventSeries
    frame_timeline: FrameTimeline | None
    asr_qc: dict[str, Any]
    align_qc: dict[str, Any]
    frame_qc: dict[str, Any]
    source_video: str | None = None
    audio: AudioStimulus | None = None

    @property
    def qc(self) -> dict[str, Any]:
        return {
            "asr": dict(self.asr_qc),
            "align": dict(self.align_qc),
            "frame": dict(self.frame_qc),
        }

    def word_table(self) -> Any:
        return as_event_table(self.words)

    def word_rows(self) -> list[dict[str, Any]]:
        labels = (
            self.words.label
            if self.words.label is not None
            else np.full(len(self.words), None, dtype=object)
        )
        conf = (
            self.words.confidence
            if self.words.confidence is not None
            else np.full(len(self.words), np.nan, dtype=np.float32)
        )
        rows: list[dict[str, Any]] = []
        for i in range(len(self.words)):
            row: dict[str, Any] = {
                "onset_s": float(self.words.onset_s[i]),
                "offset_s": float(self.words.offset_s[i]),
                "duration_s": float(self.words.offset_s[i] - self.words.onset_s[i]),
                "label": _scalar(labels[i]),
                "confidence": _scalar(conf[i]),
            }
            for key, values in self.words.extra.items():
                arr = np.asarray(values)
                if len(arr) == len(self.words):
                    row[key] = _scalar(arr[i])
            rows.append(row)
        return rows


def extract_video_text(
    video: VideoInput,
    *,
    sources: Sequence[str] = ("speech",),
    transcript_text: str | None = None,
    align: str | None = "auto",
    language: str = "auto",
    asr_model: str = "small",
    chunked: bool = False,
    chunk_window_s: float = 30.0,
    chunk_overlap_s: float = 1.0,
    audio_sr_hz: int = 16000,
    start_s: float = 0.0,
    duration_s: float | None = None,
    video_fps: float | None = None,
    frame_policy: FramePolicy = "overlap",
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
    ffmpeg_path: str = "ffmpeg",
    ffprobe_path: str = "ffprobe",
    align_backend: str | None = None,
    mfa_dictionary_path: str | None = None,
    mfa_acoustic_model_path: str | None = None,
    mfa_timeout_s: float = 300.0,
    mfa_tmp_dir: str | None = None,
    mfa_extra_args: list[str] | None = None,
) -> VideoTextResult:
    """Extract speech text from a video and align word events to frame indices.

    The workflow currently covers speech text. OCR/object-text can be added as
    additional sources without changing the word/frame result contract.
    """

    requested_sources = {str(source).strip().lower() for source in sources}
    if requested_sources != {"speech"}:
        unsupported = sorted(requested_sources - {"speech"})
        raise NotImplementedError(
            f"Unsupported video text sources for this workflow: {unsupported}"
        )

    mode, strict = resolve_execution_mode(
        execution_mode=execution_mode,
        strict_dependency=strict_dependency,
    )
    source_video = _video_source(video)

    audio = video_audio_extract(
        video,
        sr_hz=int(audio_sr_hz),
        start_s=float(start_s),
        duration_s=duration_s,
        ffmpeg_path=ffmpeg_path,
        execution_mode=mode,
        strict_dependency=strict,
    )
    if chunked and transcript_text is None:
        asr = whisper_transcribe_chunked(
            audio,
            transcript_text=None,
            model=asr_model,
            language=language,
            execution_mode=mode,
            strict_dependency=strict,
            chunk_window_s=float(chunk_window_s),
            chunk_overlap_s=float(chunk_overlap_s),
        )
    else:
        asr = whisper_transcribe(
            audio,
            transcript_text=transcript_text,
            model=asr_model,
            language=language,
            execution_mode=mode,
            strict_dependency=strict,
        )

    words = asr["words"]
    segments = asr["segments"]
    asr_qc = dict(asr.get("qc", {}))

    backend = align_backend if align_backend is not None else align
    backend_norm = "none" if backend is None else str(backend).strip().lower()
    if backend_norm in {"none", "off", "passthrough"}:
        align_qc = {
            "mode": "none",
            "execution_mode": mode,
            "fallback_used": False,
            "n_words": len(words),
            "dropped_words": 0,
        }
    else:
        aligned = whisperx_align(
            audio,
            words,
            backend=backend_norm,
            language="en" if language == "auto" else language,
            mfa_dictionary_path=mfa_dictionary_path,
            mfa_acoustic_model_path=mfa_acoustic_model_path,
            mfa_timeout_s=float(mfa_timeout_s),
            mfa_tmp_dir=mfa_tmp_dir,
            mfa_extra_args=mfa_extra_args,
            execution_mode=mode,
            strict_dependency=strict,
        )
        words = aligned["words"]
        align_qc = dict(aligned.get("qc", {}))

    words = _add_word_extras(
        words,
        {
            "text_source": "speech",
            "source_video": source_video,
        },
    )

    fps, frame_qc = _probe_video_fps(
        video,
        explicit_fps=video_fps,
        ffprobe_path=ffprobe_path,
        strict=strict,
    )
    timeline: FrameTimeline | None = None
    if fps is not None:
        if isinstance(video, VideoStimulus):
            first_frame_index = max(
                0,
                int(
                    np.floor(
                        (float(audio.start_offset_s) - float(video.start_offset_s))
                        * float(fps)
                    )
                ),
            )
        else:
            first_frame_index = int(np.floor(float(start_s) * float(fps)))
        if (
            isinstance(video, VideoStimulus)
            and duration_s is None
            and float(start_s) == float(video.start_offset_s)
        ):
            timeline = FrameTimeline.from_video_stimulus(video, source=source_video)
        else:
            timeline = FrameTimeline.from_fps(
                duration_s=_audio_duration_s(audio),
                fps=fps,
                start_s=float(audio.start_offset_s),
                first_frame_index=first_frame_index,
                source=source_video,
            )
        words = timeline.annotate_events(words, policy=frame_policy)
        frame_qc = {
            **frame_qc,
            "frame_policy": str(frame_policy),
            "mapped_words": len(words),
        }
    else:
        frame_qc = {**frame_qc, "frame_policy": str(frame_policy), "mapped_words": 0}

    words = _with_workflow_metadata(
        words, source_video=source_video, frame_policy=str(frame_policy)
    )
    return VideoTextResult(
        words=words,
        segments=segments,
        frame_timeline=timeline,
        asr_qc=asr_qc,
        align_qc=align_qc,
        frame_qc=frame_qc,
        source_video=source_video,
        audio=audio,
    )


def video_speech_words(
    video: VideoInput,
    *,
    model: str = "small",
    language: str = "auto",
    align: str | None = "auto",
    chunked: bool = False,
    chunk_window_s: float = 30.0,
    chunk_overlap_s: float = 1.0,
    audio_sr_hz: int = 16000,
    start_s: float = 0.0,
    duration_s: float | None = None,
    video_fps: float | None = None,
    frame_policy: FramePolicy = "overlap",
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
) -> dict[str, Any]:
    """Registry-friendly speech text extractor for video inputs."""

    result = extract_video_text(
        video,
        sources=("speech",),
        align=align,
        language=language,
        asr_model=model,
        chunked=chunked,
        chunk_window_s=chunk_window_s,
        chunk_overlap_s=chunk_overlap_s,
        audio_sr_hz=audio_sr_hz,
        start_s=start_s,
        duration_s=duration_s,
        video_fps=video_fps,
        frame_policy=frame_policy,
        execution_mode=execution_mode,
        strict_dependency=strict_dependency,
    )
    return {
        "words": result.words,
        "segments": result.segments,
        "qc": result.qc,
    }


__all__ = ["VideoTextResult", "extract_video_text", "video_speech_words"]
