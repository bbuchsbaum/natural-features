"""Plan-visible preprocessing nodes."""

from __future__ import annotations

import re

import numpy as np

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


def video_audio_extract(stimulus: VideoStimulus, **_: object) -> AudioStimulus:
    raise NotImplementedError(
        "video.audio.extract is plan-visible but not implemented yet. "
        "Provide an AudioStimulus explicitly or add an ffmpeg-backed extractor."
    )


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


def image_ocr(stimulus: ImageStimulus, **_: object) -> EventSeries:
    raise NotImplementedError("image.ocr is plan-visible but not implemented yet. Provide word EventSeries explicitly.")


def video_ocr(stimulus: VideoStimulus, **_: object) -> EventSeries:
    raise NotImplementedError("video.ocr is plan-visible but not implemented yet. Provide word EventSeries explicitly.")


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
