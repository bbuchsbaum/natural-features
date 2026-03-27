"""Stimulus wrappers with deterministic, aligned timebases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator
import wave

import numpy as np

from .timebase import times_from_rate


@dataclass(frozen=True)
class VideoStimulus:
    frames: np.ndarray
    fps: float
    start_offset_s: float = 0.0
    source: str | None = None

    def __post_init__(self) -> None:
        frames = np.asarray(self.frames)
        if frames.ndim not in (3, 4):
            raise ValueError("frames must be shape (T,H,W) or (T,H,W,C)")
        if frames.shape[0] <= 0:
            raise ValueError("frames must contain at least one frame")
        if self.fps <= 0:
            raise ValueError("fps must be > 0")
        object.__setattr__(self, "frames", frames)

    @classmethod
    def from_array(cls, frames: np.ndarray, fps: float, *, start_offset_s: float = 0.0) -> "VideoStimulus":
        return cls(frames=np.asarray(frames), fps=fps, start_offset_s=start_offset_s)

    @classmethod
    def from_npy(cls, path: str | Path, fps: float, *, start_offset_s: float = 0.0) -> "VideoStimulus":
        data = np.load(path)
        return cls(frames=data, fps=fps, start_offset_s=start_offset_s, source=str(path))

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

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples)
        if samples.ndim not in (1, 2):
            raise ValueError("samples must be shape (N,) or (N,C)")
        if samples.shape[0] <= 0:
            raise ValueError("samples must contain at least one sample")
        if self.sr_hz <= 0:
            raise ValueError("sr_hz must be > 0")
        object.__setattr__(self, "samples", samples)

    @classmethod
    def from_array(
        cls,
        samples: np.ndarray,
        sr_hz: int,
        *,
        start_offset_s: float = 0.0,
    ) -> "AudioStimulus":
        return cls(samples=np.asarray(samples), sr_hz=sr_hz, start_offset_s=start_offset_s)

    @classmethod
    def from_wav(cls, path: str | Path, *, start_offset_s: float = 0.0) -> "AudioStimulus":
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
        return cls(samples=samples, sr_hz=sr_hz, start_offset_s=start_offset_s, source=str(path))

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

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")
        if self.onset_s is not None:
            onset_s = np.asarray(self.onset_s, dtype=np.float64)
            object.__setattr__(self, "onset_s", onset_s)
        if self.offset_s is not None:
            offset_s = np.asarray(self.offset_s, dtype=np.float64)
            object.__setattr__(self, "offset_s", offset_s)


@dataclass(frozen=True)
class MultiModalStimulus:
    video: VideoStimulus | None = None
    audio: AudioStimulus | None = None
    text: TextStimulus | None = None

    def __post_init__(self) -> None:
        if self.video is None and self.audio is None and self.text is None:
            raise ValueError("At least one modality must be provided")

    @property
    def start_offset_s(self) -> float:
        offsets: list[float] = []
        if self.video is not None:
            offsets.append(self.video.start_offset_s)
        if self.audio is not None:
            offsets.append(self.audio.start_offset_s)
        return min(offsets) if offsets else 0.0
