"""Low-level audio baseline extractors."""

from __future__ import annotations

import math

import numpy as np

from natural_features.core.feature_types import FeatureSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.core.timebase import TimebaseSpec, times_from_hop
from natural_features.features.common import extractor_metadata


def _mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples.astype(np.float32)
    return samples.astype(np.float32).mean(axis=1)


def _frames(x: np.ndarray, sr_hz: int, hop_s: float, win_s: float) -> tuple[np.ndarray, np.ndarray]:
    hop = max(1, int(round(sr_hz * hop_s)))
    win = max(1, int(round(sr_hz * win_s)))
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    n = 1 + max(0, (len(x) - win) // hop)
    idx = np.arange(win)[None, :] + (np.arange(n)[:, None] * hop)
    frm = x[idx]
    # Match FFT/STFT convention (periodic Hann, equivalent to scipy.signal.windows.hann(sym=False)).
    window = np.hanning(win + 1).astype(np.float32)[:-1]
    return frm * window[None, :], idx[:, 0]


def _stft_power(x: np.ndarray, sr_hz: int, hop_s: float, win_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    framed, starts = _frames(x, sr_hz, hop_s, win_s)
    spec = np.fft.rfft(framed, axis=1)
    power = (np.abs(spec) ** 2).astype(np.float32)
    freqs = np.fft.rfftfreq(framed.shape[1], d=1.0 / sr_hz)
    return power, freqs, starts


def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + (f / 700.0))


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _mel_filterbank(sr_hz: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    mel_edges = np.linspace(_hz_to_mel(np.array([fmin]))[0], _hz_to_mel(np.array([fmax]))[0], n_mels + 2)
    hz_edges = _mel_to_hz(mel_edges)
    bin_edges = np.floor((n_fft + 1) * hz_edges / sr_hz).astype(int)
    fb = np.zeros((n_mels, (n_fft // 2) + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        l = bin_edges[m - 1]
        c = bin_edges[m]
        r = bin_edges[m + 1]
        if c <= l:
            c = l + 1
        if r <= c:
            r = c + 1
        for k in range(l, c):
            if 0 <= k < fb.shape[1]:
                fb[m - 1, k] = (k - l) / max(1, (c - l))
        for k in range(c, r):
            if 0 <= k < fb.shape[1]:
                fb[m - 1, k] = (r - k) / max(1, (r - c))
    return fb


def _dct_matrix(n_mfcc: int, n_mels: int) -> np.ndarray:
    m = np.arange(n_mels, dtype=np.float32)[None, :]
    k = np.arange(n_mfcc, dtype=np.float32)[:, None]
    scale = np.sqrt(2.0 / n_mels)
    basis = scale * np.cos((math.pi / n_mels) * (m + 0.5) * k)
    basis[0, :] *= 1.0 / np.sqrt(2.0)
    return basis.astype(np.float32)


def rms(stimulus: AudioStimulus, *, hop_s: float = 0.01, win_s: float = 0.025) -> FeatureSeries:
    x = _mono(stimulus.samples)
    framed, _ = _frames(x, stimulus.sr_hz, hop_s, win_s)
    vals = np.sqrt(np.mean(framed * framed, axis=1, keepdims=True)).astype(np.float32)
    times = times_from_hop(len(vals), hop_s, start_offset_s=stimulus.start_offset_s, center=True, window_s=win_s)
    metadata = extractor_metadata("audio.lowlevel.rms", params={"hop_s": hop_s, "win_s": win_s})
    return FeatureSeries(
        values=vals,
        times_s=times,
        dims=("time", "feature"),
        coords={"feature": ["rms"]},
        metadata=metadata,
        timebase=TimebaseSpec(kind="audio_hop", hop_s=hop_s, sampling_rate_hz=1.0 / hop_s),
    )


def mel(
    stimulus: AudioStimulus,
    *,
    hop_s: float = 0.01,
    win_s: float = 0.025,
    n_mels: int = 64,
    fmin: float = 50.0,
    fmax: float | None = None,
    log: bool = True,
) -> FeatureSeries:
    x = _mono(stimulus.samples)
    power, _freqs, _ = _stft_power(x, stimulus.sr_hz, hop_s, win_s)
    n_fft = int(2 * (power.shape[1] - 1))
    fmax = float(fmax if fmax is not None else stimulus.sr_hz / 2.0)
    fb = _mel_filterbank(stimulus.sr_hz, n_fft, n_mels, fmin, fmax)
    vals = np.dot(power, fb.T)
    if log:
        vals = np.log10(np.maximum(vals, 1e-10))
    vals = vals.astype(np.float32)
    times = times_from_hop(vals.shape[0], hop_s, start_offset_s=stimulus.start_offset_s, center=True, window_s=win_s)
    metadata = extractor_metadata(
        "audio.lowlevel.mel",
        params={"hop_s": hop_s, "win_s": win_s, "n_mels": n_mels, "fmin": fmin, "fmax": fmax, "log": log},
    )
    return FeatureSeries(
        values=vals,
        times_s=times,
        dims=("time", "feature"),
        coords={"feature": [f"mel_{i}" for i in range(n_mels)]},
        metadata=metadata,
        timebase=TimebaseSpec(kind="audio_hop", hop_s=hop_s, sampling_rate_hz=1.0 / hop_s),
    )


def mfcc(
    stimulus: AudioStimulus,
    *,
    hop_s: float = 0.01,
    win_s: float = 0.025,
    n_mfcc: int = 40,
    n_mels: int = 64,
    include_deltas: bool = True,
    include_delta_deltas: bool = False,
) -> FeatureSeries:
    mel_spec = mel(stimulus, hop_s=hop_s, win_s=win_s, n_mels=n_mels, log=True)
    dct = _dct_matrix(n_mfcc=n_mfcc, n_mels=n_mels)
    base = np.dot(mel_spec.values, dct.T).astype(np.float32)
    names = [f"mfcc_{i}" for i in range(n_mfcc)]
    chunks = [base]
    if include_deltas:
        d1 = np.vstack([np.zeros((1, n_mfcc), dtype=np.float32), np.diff(base, axis=0)])
        chunks.append(d1)
        names.extend([f"delta_mfcc_{i}" for i in range(n_mfcc)])
        if include_delta_deltas:
            d2 = np.vstack([np.zeros((1, n_mfcc), dtype=np.float32), np.diff(d1, axis=0)])
            chunks.append(d2)
            names.extend([f"delta2_mfcc_{i}" for i in range(n_mfcc)])
    vals = np.concatenate(chunks, axis=1).astype(np.float32)
    metadata = extractor_metadata(
        "audio.lowlevel.mfcc",
        params={
            "hop_s": hop_s,
            "win_s": win_s,
            "n_mfcc": n_mfcc,
            "n_mels": n_mels,
            "include_deltas": include_deltas,
            "include_delta_deltas": include_delta_deltas,
        },
    )
    return FeatureSeries(
        values=vals,
        times_s=mel_spec.times_s,
        dims=("time", "feature"),
        coords={"feature": names},
        metadata=metadata,
        timebase=TimebaseSpec(kind="audio_hop", hop_s=hop_s, sampling_rate_hz=1.0 / hop_s),
    )


def spectral_stats(stimulus: AudioStimulus, *, hop_s: float = 0.01, win_s: float = 0.025) -> FeatureSeries:
    x = _mono(stimulus.samples)
    power, freqs, _ = _stft_power(x, stimulus.sr_hz, hop_s, win_s)
    mag = np.sqrt(np.maximum(power, 1e-12))
    wsum = np.maximum(mag.sum(axis=1, keepdims=True), 1e-12)
    centroid = (mag * freqs[None, :]).sum(axis=1) / wsum[:, 0]
    bandwidth = np.sqrt(((freqs[None, :] - centroid[:, None]) ** 2 * mag).sum(axis=1) / wsum[:, 0])
    cdf = np.cumsum(mag, axis=1) / wsum
    rolloff_idx = np.argmax(cdf >= 0.85, axis=1)
    rolloff = freqs[rolloff_idx]
    flatness = np.exp(np.mean(np.log(np.maximum(mag, 1e-12)), axis=1)) / np.maximum(np.mean(mag, axis=1), 1e-12)
    framed, _ = _frames(x, stimulus.sr_hz, hop_s, win_s)
    sign = np.sign(framed)
    zcr = (np.abs(np.diff(sign, axis=1)) > 0).mean(axis=1)
    vals = np.column_stack([centroid, bandwidth, rolloff, flatness, zcr]).astype(np.float32)
    times = times_from_hop(vals.shape[0], hop_s, start_offset_s=stimulus.start_offset_s, center=True, window_s=win_s)
    metadata = extractor_metadata("audio.lowlevel.spectral_stats", params={"hop_s": hop_s, "win_s": win_s})
    return FeatureSeries(
        values=vals,
        times_s=times,
        dims=("time", "feature"),
        coords={"feature": ["centroid", "bandwidth", "rolloff85", "flatness", "zcr"]},
        metadata=metadata,
        timebase=TimebaseSpec(kind="audio_hop", hop_s=hop_s, sampling_rate_hz=1.0 / hop_s),
    )
