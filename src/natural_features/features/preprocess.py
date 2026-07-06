"""Plan-visible preprocessing nodes."""

from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np

from natural_features.core.execution import add_execution_provenance, resolve_execution_mode
from natural_features.core.feature_types import EventSeries, FeatureSeries
from natural_features.core.stimulus import AudioStimulus, ImageStimulus, TextStimulus, VideoStimulus
from natural_features.features.common import extractor_metadata
from natural_features.features.vision.common import VisualStimulus
from natural_features.fmri.design import add_lags
from natural_features.fmri.hrf import hrf_convolve
from natural_features.fmri.resample import resample_feature_series


def video_sample_frames(
    stimulus: VisualStimulus,
    *,
    stride_frames: int = 1,
    target_fps: float | None = None,
) -> VisualStimulus:
    if isinstance(stimulus, ImageStimulus):
        return stimulus
    if not isinstance(stimulus, VideoStimulus):
        raise TypeError("video_sample_frames requires a VideoStimulus or ImageStimulus")
    stride = max(1, int(stride_frames))
    if target_fps is not None:
        if target_fps <= 0:
            raise ValueError("target_fps must be > 0")
        stride = max(1, int(round(stimulus.fps / target_fps)))
    idx = np.arange(0, stimulus.frames.shape[0], stride)
    return VideoStimulus(
        frames=stimulus.frames[idx],
        fps=stimulus.fps / stride,
        start_offset_s=stimulus.start_offset_s,
        source=stimulus.source,
    )


def video_trim(stimulus: VisualStimulus, *, start_s: float = 0.0, end_s: float | None = None) -> VisualStimulus:
    if isinstance(stimulus, ImageStimulus):
        onset = np.nan if stimulus.onset_s is None else float(stimulus.onset_s)
        if np.isnan(onset):
            return stimulus
        stop = end_s if end_s is not None else onset
        if start_s <= onset <= stop:
            return stimulus
        raise ValueError("Trim interval does not overlap ImageStimulus")
    if not isinstance(stimulus, VideoStimulus):
        raise TypeError("video_trim requires a VideoStimulus or ImageStimulus")
    times = stimulus.frame_times_s
    frame_step = 1.0 / stimulus.fps
    stop = float(end_s) if end_s is not None else float(times[-1] + frame_step)
    keep = np.where((times >= start_s) & (times < stop))[0]
    if keep.size == 0:
        raise ValueError("Trim interval does not contain any video frames")
    return VideoStimulus(
        frames=stimulus.frames[keep],
        fps=stimulus.fps,
        start_offset_s=float(times[keep[0]]),
        source=stimulus.source,
    )


def _video_source_path(stimulus: VideoStimulus | str | Path) -> Path:
    if isinstance(stimulus, (str, Path)):
        path = Path(stimulus)
    elif isinstance(stimulus, VideoStimulus) and stimulus.source:
        path = Path(stimulus.source)
    else:
        raise RuntimeError("video.audio.extract requires a video file path or VideoStimulus.source")
    if not path.exists():
        raise FileNotFoundError(f"Video source not found: {path}")
    return path


def video_audio_extract(
    stimulus: VideoStimulus | str | Path,
    *,
    sr_hz: int = 16000,
    start_s: float = 0.0,
    duration_s: float | None = None,
    ffmpeg_path: str = "ffmpeg",
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
) -> AudioStimulus:
    _mode, strict = resolve_execution_mode(execution_mode=execution_mode, strict_dependency=strict_dependency)
    source = _video_source_path(stimulus)
    if sr_hz <= 0:
        raise ValueError("sr_hz must be > 0")
    if start_s < 0:
        raise ValueError("start_s must be >= 0")
    if duration_s is not None and duration_s <= 0:
        raise ValueError("duration_s must be > 0 when provided")
    ffmpeg = shutil.which(ffmpeg_path)
    if ffmpeg is None:
        msg = f"ffmpeg executable not found: {ffmpeg_path}"
        if strict:
            raise RuntimeError(msg)
        raise RuntimeError(msg + ". Install ffmpeg or provide an AudioStimulus explicitly.")

    with tempfile.TemporaryDirectory(prefix="nf-audio-") as tmp:
        wav_path = Path(tmp) / "audio.wav"
        cmd = [ffmpeg, "-v", "error", "-y"]
        if start_s > 0:
            cmd.extend(["-ss", str(float(start_s))])
        if duration_s is not None:
            cmd.extend(["-t", str(float(duration_s))])
        cmd.extend(["-i", str(source), "-vn", "-ac", "1", "-ar", str(int(sr_hz)), "-f", "wav", str(wav_path)])
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            diagnostics = proc.stderr.strip() or proc.stdout.strip() or "no ffmpeg diagnostics"
            raise RuntimeError(f"ffmpeg audio extraction failed: {diagnostics}")
        audio = AudioStimulus.from_wav(wav_path, start_offset_s=float(start_s))
    return AudioStimulus(samples=audio.samples, sr_hz=audio.sr_hz, start_offset_s=audio.start_offset_s, source=str(source))


def audio_trim(stimulus: AudioStimulus, *, start_s: float = 0.0, end_s: float | None = None) -> AudioStimulus:
    if not isinstance(stimulus, AudioStimulus):
        raise TypeError("audio_trim requires an AudioStimulus")
    n = stimulus.samples.shape[0]
    source_start = stimulus.start_offset_s
    source_end = source_start + n / float(stimulus.sr_hz)
    start = max(float(start_s), source_start)
    stop = min(float(end_s) if end_s is not None else source_end, source_end)
    if stop <= start:
        raise ValueError("audio_trim requires an interval with end_s > start_s")
    first = max(0, int(np.floor((start - source_start) * stimulus.sr_hz)))
    last = min(n, int(np.ceil((stop - source_start) * stimulus.sr_hz)))
    samples = stimulus.samples[first:last]
    return AudioStimulus(
        samples=samples,
        sr_hz=stimulus.sr_hz,
        start_offset_s=source_start + first / float(stimulus.sr_hz),
        source=stimulus.source,
    )


def audio_resample(stimulus: AudioStimulus, *, target_sr_hz: int = 16000) -> AudioStimulus:
    if not isinstance(stimulus, AudioStimulus):
        raise TypeError("audio_resample requires an AudioStimulus")
    target = int(target_sr_hz)
    if target <= 0:
        raise ValueError("target_sr_hz must be > 0")
    if target == stimulus.sr_hz:
        return stimulus
    samples = stimulus.samples
    n = samples.shape[0]
    duration = n / float(stimulus.sr_hz)
    new_n = max(1, int(np.ceil(duration * target)))
    old_t = np.arange(n, dtype=np.float64) / float(stimulus.sr_hz)
    new_t = np.arange(new_n, dtype=np.float64) / float(target)
    if samples.ndim == 1:
        out = np.interp(new_t, old_t, samples).astype(np.float32)
    else:
        cols = [np.interp(new_t, old_t, samples[:, j]) for j in range(samples.shape[1])]
        out = np.column_stack(cols).astype(np.float32)
    return AudioStimulus(samples=out, sr_hz=target, start_offset_s=stimulus.start_offset_s, source=stimulus.source)


def _tokens(text: str) -> list[str]:
    return re.findall(r"\b[\w']+\b", text)


def text_tokenize(stimulus: TextStimulus | str, *, duration_s: float | None = None) -> EventSeries:
    text = stimulus.text if isinstance(stimulus, TextStimulus) else str(stimulus)
    words = _tokens(text)
    n = len(words)
    if n and isinstance(stimulus, TextStimulus) and stimulus.onset_s is not None and stimulus.offset_s is not None:
        onset = np.asarray(stimulus.onset_s, dtype=np.float64)
        offset = np.asarray(stimulus.offset_s, dtype=np.float64)
        if len(onset) != n or len(offset) != n:
            raise ValueError("TextStimulus word timing must match token count")
    elif n:
        stop = float(duration_s) if duration_s is not None and duration_s > 0 else float(n)
        edges = np.linspace(0.0, stop, n + 1, dtype=np.float64)
        onset, offset = edges[:-1], edges[1:]
    else:
        onset = offset = np.array([], dtype=np.float64)
    return EventSeries(
        onset_s=onset,
        offset_s=offset,
        label=np.asarray(words, dtype=object),
        confidence=np.ones(n, dtype=np.float32),
        extra={"object_type": np.asarray(["word"] * n, dtype=object)},
        metadata=extractor_metadata("text.tokenize", params={"duration_s": duration_s}),
    )


def _empty_ocr_events(extractor_name: str, *, execution_mode: str, reason: str) -> EventSeries:
    md = add_execution_provenance(
        extractor_metadata(extractor_name, params={}, extra={"backend": "empty_fallback"}),
        execution_mode=execution_mode,
        fallback_used=True,
        fallback_reason=reason,
    )
    return EventSeries(
        onset_s=np.array([], dtype=np.float64),
        offset_s=np.array([], dtype=np.float64),
        label=np.array([], dtype=object),
        confidence=np.array([], dtype=np.float32),
        extra={
            "object_type": np.array([], dtype=object),
            "x": np.array([], dtype=np.float32),
            "y": np.array([], dtype=np.float32),
            "width": np.array([], dtype=np.float32),
            "height": np.array([], dtype=np.float32),
            "coordinate_space": np.array([], dtype=object),
        },
        metadata=md,
    )


def _image_to_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    out = np.rint(arr * 255.0).astype(np.uint8)
    if out.ndim == 2:
        return out
    if out.shape[-1] == 1:
        return out[..., 0]
    return out[..., :3]


def _image_ocr_backend(
    stimulus: ImageStimulus,
    *,
    min_confidence: float,
    duration_s: float | None,
    extractor_name: str,
    execution_mode: str,
) -> EventSeries:
    try:
        from PIL import Image  # type: ignore
        import pytesseract  # type: ignore
        from pytesseract import Output  # type: ignore
    except Exception as exc:
        raise RuntimeError("pytesseract and Pillow are required for OCR extraction") from exc

    arr = _image_to_uint8(stimulus.image)
    pil = Image.fromarray(arr)
    data = pytesseract.image_to_data(pil, output_type=Output.DICT)
    labels: list[str] = []
    conf: list[float] = []
    x: list[float] = []
    y: list[float] = []
    width: list[float] = []
    height: list[float] = []
    h, w = stimulus.image.shape[:2]
    n = len(data.get("text", []))
    for i in range(n):
        text = str(data.get("text", [""])[i]).strip()
        if not text:
            continue
        try:
            confidence = float(data.get("conf", [-1])[i])
        except (TypeError, ValueError):
            confidence = -1.0
        if confidence < min_confidence:
            continue
        labels.append(text)
        conf.append(confidence / 100.0 if confidence > 1.0 else confidence)
        left = float(data.get("left", [0])[i])
        top = float(data.get("top", [0])[i])
        bw = float(data.get("width", [0])[i])
        bh = float(data.get("height", [0])[i])
        x.append(left / max(float(w), 1.0))
        y.append(top / max(float(h), 1.0))
        width.append(bw / max(float(w), 1.0))
        height.append(bh / max(float(h), 1.0))

    onset = 0.0 if stimulus.onset_s is None else float(stimulus.onset_s)
    duration = float(duration_s if duration_s is not None else (stimulus.duration_s or 0.0))
    count = len(labels)
    md = add_execution_provenance(
        extractor_metadata(extractor_name, params={"min_confidence": min_confidence}, extra={"backend": "pytesseract"}),
        execution_mode=execution_mode,
        fallback_used=False,
    )
    return EventSeries(
        onset_s=np.full(count, onset, dtype=np.float64),
        offset_s=np.full(count, onset + duration, dtype=np.float64),
        label=np.asarray(labels, dtype=object),
        confidence=np.asarray(conf, dtype=np.float32),
        extra={
            "object_type": np.asarray(["word"] * count, dtype=object),
            "x": np.asarray(x, dtype=np.float32),
            "y": np.asarray(y, dtype=np.float32),
            "width": np.asarray(width, dtype=np.float32),
            "height": np.asarray(height, dtype=np.float32),
            "coordinate_space": np.asarray(["relative"] * count, dtype=object),
        },
        metadata=md,
    )


def image_ocr(
    stimulus: ImageStimulus,
    *,
    min_confidence: float = 0.0,
    duration_s: float | None = None,
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
) -> EventSeries:
    mode, strict = resolve_execution_mode(execution_mode=execution_mode, strict_dependency=strict_dependency)
    if not isinstance(stimulus, ImageStimulus):
        raise TypeError("image_ocr requires an ImageStimulus")
    try:
        return _image_ocr_backend(
            stimulus,
            min_confidence=float(min_confidence),
            duration_s=duration_s,
            extractor_name="image.ocr",
            execution_mode=mode,
        )
    except Exception as exc:
        if strict:
            raise RuntimeError("image.ocr failed in strict mode") from exc
        return _empty_ocr_events("image.ocr", execution_mode=mode, reason=str(exc))


def _concat_event_series(events: list[EventSeries], *, extractor_name: str, execution_mode: str) -> EventSeries:
    if not events:
        return _empty_ocr_events(extractor_name, execution_mode=execution_mode, reason="no frames")
    onset = np.concatenate([ev.onset_s for ev in events])
    offset = np.concatenate([ev.offset_s for ev in events])
    labels = np.concatenate([ev.label if ev.label is not None else np.array([], dtype=object) for ev in events])
    confidence = np.concatenate(
        [ev.confidence if ev.confidence is not None else np.full(len(ev), np.nan, dtype=np.float32) for ev in events]
    )
    extra: dict[str, Any] = {}
    keys = sorted({key for ev in events for key in ev.extra.keys()})
    for key in keys:
        chunks = []
        for ev in events:
            value = ev.extra.get(key)
            if value is None:
                chunks.append(np.full(len(ev), np.nan, dtype=object))
            else:
                chunks.append(np.asarray(value))
        extra[key] = np.concatenate(chunks)
    md = add_execution_provenance(
        extractor_metadata(extractor_name, params={}, extra={"backend": "pytesseract"}),
        execution_mode=execution_mode,
        fallback_used=False,
    )
    return EventSeries(onset_s=onset, offset_s=offset, label=labels, confidence=confidence, extra=extra, metadata=md)


def video_ocr(
    stimulus: VideoStimulus,
    *,
    stride_frames: int = 1,
    min_confidence: float = 0.0,
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
) -> EventSeries:
    mode, strict = resolve_execution_mode(execution_mode=execution_mode, strict_dependency=strict_dependency)
    if not isinstance(stimulus, VideoStimulus):
        raise TypeError("video_ocr requires a VideoStimulus")
    stride = max(1, int(stride_frames))
    events: list[EventSeries] = []
    frame_duration = 1.0 / float(stimulus.fps)
    for frame_index in range(0, stimulus.frames.shape[0], stride):
        image = ImageStimulus.from_array(
            stimulus.frames[frame_index],
            onset_s=float(stimulus.frame_times_s[frame_index]),
            duration_s=frame_duration,
        )
        ev = image_ocr(
            image,
            min_confidence=min_confidence,
            duration_s=frame_duration,
            execution_mode=mode,
            strict_dependency=strict,
        )
        if len(ev):
            ev_extra = dict(ev.extra)
            ev_extra["frame_index"] = np.full(len(ev), frame_index, dtype=np.int64)
            ev = EventSeries(
                onset_s=ev.onset_s,
                offset_s=ev.offset_s,
                label=ev.label,
                confidence=ev.confidence,
                extra=ev_extra,
                metadata=ev.metadata,
                schema=ev.schema,
                timebase=ev.timebase,
            )
            events.append(ev)
    if strict and not events:
        return _empty_ocr_events("video.ocr", execution_mode=mode, reason="no OCR text detected")
    return _concat_event_series(events, extractor_name="video.ocr", execution_mode=mode)


def events_align(events: EventSeries, *, mode: str = "passthrough", **_: object) -> EventSeries:
    if not isinstance(events, EventSeries):
        raise TypeError("events_align requires an EventSeries")
    metadata = dict(events.metadata)
    metadata.update(extractor_metadata("events.align", params={"mode": mode}))
    return EventSeries(
        onset_s=events.onset_s,
        offset_s=events.offset_s,
        label=events.label,
        confidence=events.confidence,
        extra=events.extra,
        metadata=metadata,
        schema=events.schema,
        timebase=events.timebase,
    )


def features_resample(
    feature: FeatureSeries,
    *,
    tr_s: float | None = None,
    step_s: float = 1.0,
    duration_s: float | None = None,
    method: str = "mean",
) -> FeatureSeries:
    return resample_feature_series(feature, tr_s=float(tr_s or step_s), duration_s=duration_s, method=method)


def features_hrf(feature: FeatureSeries, *, tr_s: float | None = None, kind: str = "glover") -> FeatureSeries:
    if tr_s is None:
        diffs = np.diff(feature.times_s)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        tr_s = float(np.median(diffs)) if diffs.size else 1.0
    return hrf_convolve(feature, tr_s=float(tr_s), kind=kind)


def features_lag(feature: FeatureSeries, *, lags: list[int] | None = None) -> FeatureSeries:
    return add_lags(feature, list(lags or [0, 1]))
