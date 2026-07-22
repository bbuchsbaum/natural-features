"""Stimulus wrappers with deterministic, aligned timebases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator
import wave

import numpy as np

from .timebase import ClockRef, STIMULUS_CLOCK, TemporalContext, times_from_rate


def _normalize_image_array(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError("image must be numeric")
    if arr.ndim not in (2, 3):
        raise ValueError("image must be shape (H,W) or (H,W,C)")
    if any(dim <= 0 for dim in arr.shape):
        raise ValueError("image dimensions must be positive")
    out = arr.astype(np.float32, copy=False)
    if out.size and np.nanmax(out) > 1.0:
        out = out / 255.0
    return np.clip(out, 0.0, 1.0)


@dataclass(frozen=True)
class ImageStimulus:
    image: np.ndarray
    onset_s: float | None = 0.0
    duration_s: float | None = None
    source: str | None = None
    clock: ClockRef | str = STIMULUS_CLOCK
    temporal_context: TemporalContext = TemporalContext()

    def __post_init__(self) -> None:
        image = _normalize_image_array(self.image)
        if self.onset_s is not None and not np.isfinite(float(self.onset_s)):
            raise ValueError("onset_s must be finite or None")
        if self.duration_s is not None:
            duration = float(self.duration_s)
            if not np.isfinite(duration) or duration < 0:
                raise ValueError("duration_s must be a finite non-negative value or None")
        object.__setattr__(self, "image", image)
        object.__setattr__(self, "clock", ClockRef(self.clock))
        if not isinstance(self.temporal_context, TemporalContext):
            object.__setattr__(self, "temporal_context", TemporalContext.from_dict(self.temporal_context))
        if self.onset_s is not None:
            object.__setattr__(self, "onset_s", float(self.onset_s))
        if self.duration_s is not None:
            object.__setattr__(self, "duration_s", float(self.duration_s))

    @classmethod
    def from_array(
        cls,
        image: np.ndarray,
        *,
        onset_s: float | None = 0.0,
        duration_s: float | None = None,
        clock: ClockRef | str = STIMULUS_CLOCK,
        temporal_context: TemporalContext = TemporalContext(),
    ) -> "ImageStimulus":
        return cls(
            image=np.asarray(image),
            onset_s=onset_s,
            duration_s=duration_s,
            clock=clock,
            temporal_context=temporal_context,
        )

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        onset_s: float | None = 0.0,
        duration_s: float | None = None,
        clock: ClockRef | str = STIMULUS_CLOCK,
        temporal_context: TemporalContext = TemporalContext(),
    ) -> "ImageStimulus":
        try:
            from PIL import Image  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Pillow is required to read image files: pip install natural-features[vision]") from exc

        p = Path(path)
        with Image.open(p) as img:
            if img.mode not in {"L", "RGB", "RGBA"}:
                img = img.convert("RGB")
            data = np.asarray(img)
        return cls(
            image=data,
            onset_s=onset_s,
            duration_s=duration_s,
            source=str(p.resolve()),
            clock=clock,
            temporal_context=temporal_context,
        )

    @property
    def frame_times_s(self) -> np.ndarray:
        return np.array([np.nan if self.onset_s is None else float(self.onset_s)], dtype=np.float64)

    def as_frames(self) -> np.ndarray:
        if self.image.ndim == 2:
            return self.image[None, :, :]
        return self.image[None, :, :, :]


def image_from_array(
    image: np.ndarray,
    *,
    onset_s: float | None = 0.0,
    duration_s: float | None = None,
    clock: ClockRef | str = STIMULUS_CLOCK,
    temporal_context: TemporalContext = TemporalContext(),
) -> ImageStimulus:
    return ImageStimulus.from_array(
        image,
        onset_s=onset_s,
        duration_s=duration_s,
        clock=clock,
        temporal_context=temporal_context,
    )


def image_from_file(
    path: str | Path,
    *,
    onset_s: float | None = 0.0,
    duration_s: float | None = None,
    clock: ClockRef | str = STIMULUS_CLOCK,
    temporal_context: TemporalContext = TemporalContext(),
) -> ImageStimulus:
    return ImageStimulus.from_file(
        path,
        onset_s=onset_s,
        duration_s=duration_s,
        clock=clock,
        temporal_context=temporal_context,
    )


@dataclass(frozen=True)
class VideoStimulus:
    frames: np.ndarray
    fps: float
    start_offset_s: float = 0.0
    source: str | None = None
    clock: ClockRef | str = STIMULUS_CLOCK
    temporal_context: TemporalContext = TemporalContext()

    def __post_init__(self) -> None:
        frames = np.asarray(self.frames)
        if frames.ndim not in (3, 4):
            raise ValueError("frames must be shape (T,H,W) or (T,H,W,C)")
        if frames.shape[0] <= 0:
            raise ValueError("frames must contain at least one frame")
        if self.fps <= 0:
            raise ValueError("fps must be > 0")
        object.__setattr__(self, "frames", frames)
        object.__setattr__(self, "clock", ClockRef(self.clock))
        if not isinstance(self.temporal_context, TemporalContext):
            object.__setattr__(self, "temporal_context", TemporalContext.from_dict(self.temporal_context))

    @classmethod
    def from_array(
        cls,
        frames: np.ndarray,
        fps: float,
        *,
        start_offset_s: float = 0.0,
        clock: ClockRef | str = STIMULUS_CLOCK,
        temporal_context: TemporalContext = TemporalContext(),
    ) -> "VideoStimulus":
        return cls(
            frames=np.asarray(frames),
            fps=fps,
            start_offset_s=start_offset_s,
            clock=clock,
            temporal_context=temporal_context,
        )

    @classmethod
    def from_npy(
        cls,
        path: str | Path,
        fps: float,
        *,
        start_offset_s: float = 0.0,
        clock: ClockRef | str = STIMULUS_CLOCK,
        temporal_context: TemporalContext = TemporalContext(),
    ) -> "VideoStimulus":
        data = np.load(path)
        return cls(
            frames=data,
            fps=fps,
            start_offset_s=start_offset_s,
            source=str(path),
            clock=clock,
            temporal_context=temporal_context,
        )

    @property
    def frame_times_s(self) -> np.ndarray:
        return times_from_rate(len(self.frames), self.fps, start_offset_s=self.start_offset_s)

    def frame_stream(self, *, chunk_size: int = 64) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        t = self.frame_times_s
        for i in range(0, len(self.frames), chunk_size):
            j = min(i + chunk_size, len(self.frames))
            yield t[i:j], self.frames[i:j]


@dataclass(frozen=True)
class AudioStimulus:
    samples: np.ndarray
    sr_hz: int
    start_offset_s: float = 0.0
    source: str | None = None
    clock: ClockRef | str = STIMULUS_CLOCK
    temporal_context: TemporalContext = TemporalContext()

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples)
        if samples.ndim not in (1, 2):
            raise ValueError("samples must be shape (N,) or (N,C)")
        if samples.shape[0] <= 0:
            raise ValueError("samples must contain at least one sample")
        if self.sr_hz <= 0:
            raise ValueError("sr_hz must be > 0")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "clock", ClockRef(self.clock))
        if not isinstance(self.temporal_context, TemporalContext):
            object.__setattr__(self, "temporal_context", TemporalContext.from_dict(self.temporal_context))

    @classmethod
    def from_array(
        cls,
        samples: np.ndarray,
        sr_hz: int,
        *,
        start_offset_s: float = 0.0,
        clock: ClockRef | str = STIMULUS_CLOCK,
        temporal_context: TemporalContext = TemporalContext(),
    ) -> "AudioStimulus":
        return cls(
            samples=np.asarray(samples),
            sr_hz=sr_hz,
            start_offset_s=start_offset_s,
            clock=clock,
            temporal_context=temporal_context,
        )

    @classmethod
    def from_wav(
        cls,
        path: str | Path,
        *,
        start_offset_s: float = 0.0,
        clock: ClockRef | str = STIMULUS_CLOCK,
        temporal_context: TemporalContext = TemporalContext(),
    ) -> "AudioStimulus":
        p = Path(path)
        with wave.open(str(p), "rb") as w:
            n_channels = w.getnchannels()
            sr_hz = w.getframerate()
            sample_width = w.getsampwidth()
            n_frames = w.getnframes()
            raw = w.readframes(n_frames)
        if sample_width not in (1, 2, 4):
            raise ValueError(f"Unsupported WAV sample width: {sample_width}")
        dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[sample_width]
        data = np.frombuffer(raw, dtype=dtype)
        if n_channels > 1:
            data = data.reshape(-1, n_channels)
        if sample_width == 1:
            samples = (data.astype(np.float32) - 128.0) / 128.0
        else:
            max_val = float(np.iinfo(dtype).max)
            samples = data.astype(np.float32) / max_val
        return cls(
            samples=samples,
            sr_hz=sr_hz,
            start_offset_s=start_offset_s,
            source=str(path),
            clock=clock,
            temporal_context=temporal_context,
        )

    @property
    def sample_times_s(self) -> np.ndarray:
        return times_from_rate(self.samples.shape[0], float(self.sr_hz), start_offset_s=self.start_offset_s)

    def audio_stream(
        self,
        *,
        window_s: float = 1.0,
        hop_s: float | None = None,
    ) -> Generator[tuple[float, np.ndarray], None, None]:
        if window_s <= 0:
            raise ValueError("window_s must be > 0")
        if hop_s is None:
            hop_s = window_s
        if hop_s <= 0:
            raise ValueError("hop_s must be > 0")
        win = int(round(window_s * self.sr_hz))
        hop = int(round(hop_s * self.sr_hz))
        if win <= 0 or hop <= 0:
            raise ValueError("window_s/hop_s too small for sample rate")
        n = self.samples.shape[0]
        for i in range(0, n, hop):
            j = min(i + win, n)
            t0 = self.start_offset_s + (i / self.sr_hz)
            yield float(t0), self.samples[i:j]


@dataclass(frozen=True)
class TextStimulus:
    text: str
    onset_s: np.ndarray | None = None
    offset_s: np.ndarray | None = None
    source: str | None = None
    clock: ClockRef | str = STIMULUS_CLOCK
    temporal_context: TemporalContext = TemporalContext()

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")
        object.__setattr__(self, "clock", ClockRef(self.clock))
        if not isinstance(self.temporal_context, TemporalContext):
            object.__setattr__(self, "temporal_context", TemporalContext.from_dict(self.temporal_context))
        if self.onset_s is not None:
            onset_s = np.asarray(self.onset_s, dtype=np.float64)
            object.__setattr__(self, "onset_s", onset_s)
        if self.offset_s is not None:
            offset_s = np.asarray(self.offset_s, dtype=np.float64)
            object.__setattr__(self, "offset_s", offset_s)


@dataclass(frozen=True)
class MultiModalStimulus:
    video: VideoStimulus | None = None
    image: ImageStimulus | None = None
    audio: AudioStimulus | None = None
    text: TextStimulus | None = None
    temporal_context: TemporalContext = TemporalContext()

    def __post_init__(self) -> None:
        if self.video is None and self.image is None and self.audio is None and self.text is None:
            raise ValueError("At least one modality must be provided")
        context = self.temporal_context
        if not isinstance(context, TemporalContext):
            context = TemporalContext.from_dict(context)
        contexts = [context]
        for item in (self.video, self.image, self.audio, self.text):
            if item is not None:
                contexts.append(item.temporal_context)
        combined = contexts[0].merged(*contexts[1:])
        object.__setattr__(self, "temporal_context", combined)

    @property
    def start_offset_s(self) -> float:
        offsets: list[float] = []
        if self.video is not None:
            offsets.append(self.video.start_offset_s)
        if self.image is not None and self.image.onset_s is not None:
            offsets.append(self.image.onset_s)
        if self.audio is not None:
            offsets.append(self.audio.start_offset_s)
        return min(offsets) if offsets else 0.0
