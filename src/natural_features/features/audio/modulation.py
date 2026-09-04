"""Spectrotemporal modulation features.

The modulation power spectrum (MPS) is the 2-D Fourier transform of a log-frequency,
log-power cochleagram patch. Its axes are *spectral modulation* in cycles per octave
(how finely energy is rippled across frequency) and *temporal modulation* in Hz (how
fast it fluctuates in time). It is the standard characterisation of early auditory
cortex, and it is not recoverable from a cochleagram by any per-frame summary --
which is why it lives here rather than in :mod:`~natural_features.features.audio.lowlevel`.

The spectral axis is signed on purpose. A ripple sweeping upward in frequency over time
and one sweeping downward have the same ``|Omega|`` and the same ``omega``; only the
sign of ``Omega`` relative to ``omega`` distinguishes them, and auditory neurons are
routinely selective for that direction. Cells are labelled ``pos``/``neg`` by the sign
of ``Omega`` rather than "up"/"down", because the perceptual direction depends on a
sign convention: here the time axis is transformed with an rFFT so ``omega >= 0``, and
a component at ``(+Omega, +omega)`` has constant-phase contours ``x = -(omega/Omega) t``
in octaves, i.e. a *downward* sweep.

:func:`modulation_power_spectrum` is the numeric kernel and takes a cochleagram
directly, so it can be exercised with synthetic ripples of known ``(Omega, omega)``
without going through audio synthesis. :func:`audio_modulation_spectrum` is the
extractor that builds the cochleagram from a stimulus first.

:func:`spectrotemporal_modulation` is a separate hop-rate STFT rate/scale energy
band used by the speech ladder (``backend=stft_rate_scale``). It is not an NSL
Chi/Shamma ``aud2cor`` implementation and it is not interchangeable with
``audio.modulation.mps``.
"""

from __future__ import annotations

import numpy as np

from natural_features.core.feature_types import FeatureSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.core.timebase import TimebaseSpec, times_from_hop
from natural_features.features.audio.cochlear import (
    _erb_filterbank,
    _erb_to_hz,
    _hz_to_erb,
)
from natural_features.features.audio.lowlevel import _mono, _stft_power
from natural_features.features.common import extractor_metadata

__all__ = [
    "audio_modulation_spectrum",
    "log_cochleagram",
    "modulation_power_spectrum",
    "spectrotemporal_modulation",
]


def _erb_centers(n_channels: int, fmin: float, fmax: float) -> np.ndarray:
    """Centre frequencies of the ERB-spaced bank used by ``audio.gammatone``."""

    erb_edges = np.linspace(
        _hz_to_erb(np.array([fmin]))[0], _hz_to_erb(np.array([fmax]))[0], n_channels + 2
    )
    return _erb_to_hz(erb_edges)[1:-1]


def log_cochleagram(
    stimulus: AudioStimulus,
    *,
    hop_s: float = 0.005,
    win_s: float = 0.025,
    n_channels: int = 64,
    fmin: float = 50.0,
    fmax: float | None = None,
    n_log_bins: int = 64,
) -> tuple[np.ndarray, float]:
    """Return ``(cochleagram[time, n_log_bins], octaves_per_bin)``.

    The ERB bank is resampled onto a grid that is uniform in ``log2(f)``. That step is
    what makes the spectral modulation axis mean cycles *per octave*: ERB spacing is
    only approximately logarithmic, and below ~500 Hz it is markedly not, so taking an
    FFT straight along ERB channels would give an axis in cycles per channel whose
    octave calibration drifts with frequency.
    """

    top = float(fmax if fmax is not None else stimulus.sr_hz / 2.0)
    power, _freqs, _ = _stft_power(
        _mono(stimulus.samples), stimulus.sr_hz, hop_s, win_s
    )
    n_fft = int(2 * (power.shape[1] - 1))
    fb = _erb_filterbank(stimulus.sr_hz, n_fft, n_channels, fmin, top)
    erb_power = power @ fb.T
    coch = np.log10(np.maximum(erb_power, 1e-10))

    centers = _erb_centers(n_channels, fmin, top)
    lo, hi = np.log2(centers[0]), np.log2(centers[-1])
    grid = np.linspace(lo, hi, n_log_bins)
    out = np.empty((coch.shape[0], n_log_bins), dtype=np.float32)
    src = np.log2(centers)
    for t in range(coch.shape[0]):
        out[t] = np.interp(grid, src, coch[t]).astype(np.float32)
    oct_per_bin = float((hi - lo) / max(1, n_log_bins - 1))
    return out, oct_per_bin


def modulation_power_spectrum(
    coch: np.ndarray,
    *,
    oct_per_bin: float,
    frame_rate_hz: float,
    spec_edges: np.ndarray | None = None,
    temp_edges: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """2-D modulation power of one cochleagram patch.

    Parameters
    ----------
    coch:
        ``(n_frames, n_bins)`` log-power cochleagram on a log-frequency axis.
    oct_per_bin:
        Octaves between adjacent frequency bins; sets the ``cyc/oct`` calibration.
    frame_rate_hz:
        Cochleagram frame rate; sets the ``Hz`` calibration.
    spec_edges:
        Band edges in ``|cyc/oct|``. Defaults to four log bands from 0.25 to 8.
    temp_edges:
        Band edges in Hz. Defaults to five log bands from 0.5 to 32.

    Returns
    -------
    ``(values, names)`` where ``values`` is the flattened log mean power over each
    ``(signed spectral band, temporal band)`` cell.
    """

    if coch.ndim != 2:
        raise ValueError("coch must be 2-D (frames x bins)")
    n_t, n_f = coch.shape
    if n_t < 4 or n_f < 4:
        raise ValueError("coch must have at least 4 frames and 4 bins")
    if oct_per_bin <= 0 or frame_rate_hz <= 0:
        raise ValueError("oct_per_bin and frame_rate_hz must be > 0")

    if spec_edges is None:
        spec_edges = np.geomspace(0.25, 8.0, 5)
    if temp_edges is None:
        temp_edges = np.geomspace(0.5, 32.0, 6)
    spec_edges = np.asarray(spec_edges, dtype=np.float64)
    temp_edges = np.asarray(temp_edges, dtype=np.float64)

    patch = coch.astype(np.float64)
    patch = patch - patch.mean()
    # Separable Hann taper: without it the patch edges act as step discontinuities and
    # smear energy across every modulation band.
    wt = np.hanning(n_t)[:, None]
    wf = np.hanning(n_f)[None, :]
    patch = patch * wt * wf

    # rFFT along time first -- it needs a real input, and a real patch has a symmetric
    # temporal spectrum so omega >= 0 loses nothing. The full FFT along frequency then
    # runs on the complex result and keeps the sign of Omega, which is what separates
    # up- from down-sweeping ripples.
    spec = np.fft.fft(np.fft.rfft(patch, axis=0), axis=1)
    power = np.abs(spec) ** 2

    omega = np.fft.rfftfreq(n_t, d=1.0 / frame_rate_hz)  # Hz, >= 0
    Omega = np.fft.fftfreq(n_f, d=oct_per_bin)  # cyc/oct, signed

    values: list[float] = []
    names: list[str] = []
    for s_i in range(len(spec_edges) - 1):
        s_lo, s_hi = spec_edges[s_i], spec_edges[s_i + 1]
        for sign, tag in ((1.0, "pos"), (-1.0, "neg")):
            if sign > 0:
                s_mask = (Omega >= s_lo) & (Omega < s_hi)
            else:
                s_mask = (Omega <= -s_lo) & (Omega > -s_hi)
            for t_i in range(len(temp_edges) - 1):
                t_lo, t_hi = temp_edges[t_i], temp_edges[t_i + 1]
                t_mask = (omega >= t_lo) & (omega < t_hi)
                if not np.any(s_mask) or not np.any(t_mask):
                    values.append(np.nan)
                else:
                    cell = power[np.ix_(t_mask, s_mask)]
                    values.append(float(np.log10(max(cell.mean(), 1e-20))))
                names.append(f"mps_{tag}_{s_lo:g}-{s_hi:g}cpo_{t_lo:g}-{t_hi:g}hz")

    return np.asarray(values, dtype=np.float32), names


def audio_modulation_spectrum(
    stimulus: AudioStimulus,
    *,
    window_s: float = 2.0,
    hop_s: float = 0.5,
    coch_hop_s: float = 0.005,
    coch_win_s: float = 0.025,
    n_channels: int = 64,
    n_log_bins: int = 64,
    fmin: float = 50.0,
    fmax: float | None = None,
    spec_edges: np.ndarray | None = None,
    temp_edges: np.ndarray | None = None,
) -> FeatureSeries:
    """Return the spectrotemporal modulation power spectrum per analysis window.

    ``coch_hop_s`` sets the Nyquist of the temporal modulation axis: the default
    0.005 s reaches 100 Hz, comfortably above the 32 Hz top of the default temporal
    bands. ``window_s`` sets its resolution, and a window shorter than a few cycles of
    the slowest band cannot resolve it -- at the 0.5 Hz default floor, ``window_s``
    should be at least 2 s.
    """

    if window_s <= 0 or hop_s <= 0:
        raise ValueError("window_s and hop_s must be > 0")

    coch, oct_per_bin = log_cochleagram(
        stimulus,
        hop_s=coch_hop_s,
        win_s=coch_win_s,
        n_channels=n_channels,
        fmin=fmin,
        fmax=fmax,
        n_log_bins=n_log_bins,
    )
    frame_rate = 1.0 / coch_hop_s
    w = int(round(window_s * frame_rate))
    h = max(1, int(round(hop_s * frame_rate)))
    if coch.shape[0] < w:
        raise ValueError("signal is shorter than one analysis window")
    starts = np.arange(0, coch.shape[0] - w + 1, h, dtype=int)

    rows = []
    names: list[str] = []
    for s in starts:
        vals, names = modulation_power_spectrum(
            coch[s : s + w],
            oct_per_bin=oct_per_bin,
            frame_rate_hz=frame_rate,
            spec_edges=spec_edges,
            temp_edges=temp_edges,
        )
        rows.append(vals)
    values = np.vstack(rows).astype(np.float32)

    times = (starts / frame_rate) + (window_s / 2.0) + stimulus.start_offset_s
    md = extractor_metadata(
        "audio.modulation.mps",
        params={
            "window_s": window_s,
            "hop_s": hop_s,
            "coch_hop_s": coch_hop_s,
            "coch_win_s": coch_win_s,
            "n_channels": n_channels,
            "n_log_bins": n_log_bins,
            "fmin": fmin,
            "fmax": fmax,
            "spec_edges": None
            if spec_edges is None
            else np.asarray(spec_edges).tolist(),
            "temp_edges": None
            if temp_edges is None
            else np.asarray(temp_edges).tolist(),
        },
        extra={"backend": "log_cochleagram_2dfft"},
    )
    return FeatureSeries(
        values=values,
        times_s=times,
        dims=("time", "feature"),
        coords={"feature": names},
        metadata=md,
        timebase=TimebaseSpec(
            kind="audio_hop", hop_s=hop_s, sampling_rate_hz=1.0 / hop_s
        ),
    )


DEFAULT_RATE_BANDS_HZ: tuple[tuple[float, float], ...] = (
    (1.0, 4.0),
    (4.0, 8.0),
    (8.0, 16.0),
    (16.0, 32.0),
)
DEFAULT_SCALE_BANDS: tuple[tuple[float, float], ...] = (
    (0.0, 0.25),
    (0.25, 0.75),
    (0.75, 2.0),
)


def _bandpass_axis(
    values: np.ndarray,
    *,
    axis: int,
    sample_period: float,
    lo: float,
    hi: float,
) -> np.ndarray:
    n = values.shape[axis]
    freqs = np.fft.rfftfreq(n, d=sample_period)
    spec = np.fft.rfft(values, axis=axis)
    mask = (freqs >= lo) & (freqs < hi)
    indexer = [slice(None)] * spec.ndim
    indexer[axis] = ~mask
    spec[tuple(indexer)] = 0.0
    out = np.fft.irfft(spec, n=n, axis=axis)
    return np.asarray(out, dtype=np.float32)


def _rate_name(lo: float, hi: float) -> str:
    return f"rate_{int(lo)}_{int(hi)}"


def _scale_name(lo: float, hi: float) -> str:
    return f"scale_{lo:g}_{hi:g}"


def spectrotemporal_modulation(
    stimulus: AudioStimulus,
    *,
    hop_s: float = 0.01,
    win_s: float = 0.025,
    compressed: bool = True,
) -> FeatureSeries:
    """Return temporal-rate × spectral-scale energy of an STFT spectrogram.

    Default output is a compressed ``time × (rate × scale)`` matrix. Set
    ``compressed=False`` to keep flattened per-frequency rate/scale channels.
    """

    power, _freqs, _starts = _stft_power(
        _mono(stimulus.samples), stimulus.sr_hz, hop_s, win_s
    )
    log_power = np.log10(np.maximum(power, 1e-10)).astype(np.float32)
    n_freq = log_power.shape[1]
    # Spectral modulation is defined on a channel index axis (cycles/channel).
    channel_period = 1.0
    names: list[str] = []
    cols: list[np.ndarray] = []
    for rate_lo, rate_hi in DEFAULT_RATE_BANDS_HZ:
        rate_filt = _bandpass_axis(
            log_power, axis=0, sample_period=hop_s, lo=rate_lo, hi=rate_hi
        )
        rate_label = _rate_name(rate_lo, rate_hi)
        for scale_lo, scale_hi in DEFAULT_SCALE_BANDS:
            scale_filt = _bandpass_axis(
                rate_filt,
                axis=1,
                sample_period=channel_period,
                lo=scale_lo,
                hi=scale_hi,
            )
            energy = np.sqrt(np.mean(scale_filt * scale_filt, axis=1))
            label = f"{rate_label}__{_scale_name(scale_lo, scale_hi)}"
            if compressed:
                names.append(label)
                cols.append(energy.astype(np.float32))
            else:
                for freq_i in range(n_freq):
                    names.append(f"{label}__f{freq_i}")
                    cols.append(np.abs(scale_filt[:, freq_i]).astype(np.float32))
    values = np.stack(cols, axis=1).astype(np.float32)
    times = times_from_hop(
        values.shape[0],
        hop_s,
        start_offset_s=stimulus.start_offset_s,
        center=True,
        window_s=win_s,
    )
    md = extractor_metadata(
        "audio.modulation.spectrotemporal",
        params={"hop_s": hop_s, "win_s": win_s, "compressed": compressed},
        extra={"backend": "stft_rate_scale"},
    )
    return FeatureSeries(
        values=values,
        times_s=times,
        dims=("time", "feature"),
        coords={"feature": names},
        metadata=md,
        timebase=TimebaseSpec(
            kind="audio_hop", hop_s=hop_s, sampling_rate_hz=1.0 / hop_s
        ),
    )
