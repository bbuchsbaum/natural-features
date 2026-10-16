"""Finite-support carrier-resolved modulation on a log-frequency energy frontend.

A local Gabor descriptor, not a physiological auditory-model implementation.
Observation cells and computational input dependencies are recorded separately.
"""
from __future__ import annotations

import numpy as np
from natural_features.core.feature_types import FeatureSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.core.timebase import SupportSpec, TimebaseSpec
from natural_features.features.common import extractor_metadata
from natural_features.features.audio.modulation import _erb_centers, _erb_filters


def local_modulation_kernel(
    coch,
    *,
    frequencies_hz,
    frame_rate_hz,
    carriers_hz=(125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0),
    rates_hz=(2.0, 4.0, 8.0, 16.0),
    scales_cpo=(0.5, 1.0, 2.0),
    window_s=2.0,
    spectral_sd_oct=0.5,
):
    """Return valid-window amplitudes [time, carrier, rate, scale, direction].

    Direction is the sign of spectral frequency while temporal frequency is
    positive. Positive sign gives downward constant-phase contours. Spectral
    kernels and temporal kernels are separately zero-DC and unit L2 norm.
    """

    if any(isinstance(v, (bool, np.bool_)) for v in (frame_rate_hz, window_s, spectral_sd_oct)):
        raise ValueError("numeric positive kernel widths and frame rate required")
    z = np.asarray(coch, dtype=float)
    f = np.asarray(frequencies_hz, dtype=float)
    rates = np.asarray(rates_hz, dtype=float)
    scales = np.asarray(scales_cpo, dtype=float)
    carriers = np.asarray(carriers_hz, dtype=float)
    if f.ndim != 1 or z.ndim != 2 or z.shape[1] != f.size or f.size < 3 or not np.all(np.isfinite(z)):
        raise ValueError("finite time by frequency cochleagram required")
    if np.any(f <= 0) or not np.all(np.isfinite(f)) or not np.all(np.diff(f) > 0):
        raise ValueError("frequencies must be positive and increasing")
    dx = np.diff(np.log2(f))
    if not np.allclose(dx, dx[0], rtol=1e-5):
        raise ValueError("log-frequency grid required")
    if not np.isfinite(frame_rate_hz) or frame_rate_hz <= 0:
        raise ValueError("positive frame rate required")
    if (
        not np.isfinite(window_s)
        or window_s <= 0
        or not np.isfinite(spectral_sd_oct)
        or spectral_sd_oct <= 0
    ):
        raise ValueError("positive finite kernel widths required")
    if any(
        a.ndim != 1 or not len(a) or not np.all(np.isfinite(a)) or np.any(a <= 0)
        for a in (rates, scales, carriers)
    ):
        raise ValueError("nonempty positive carrier/rate/scale vectors required")
    if np.any(rates >= frame_rate_hz / 2) or np.any(scales >= 0.5 / dx[0]):
        raise ValueError("modulation frequency reaches Nyquist")
    if np.any(carriers < f[0]) or np.any(carriers > f[-1]):
        raise ValueError("carrier outside frequency grid")
    nt = int(round(window_s * frame_rate_hz))
    if nt < 3 or len(z) < nt:
        raise ValueError("insufficient complete temporal support")
    t = np.arange(nt) / frame_rate_hz
    wt = np.hanning(nt)
    if np.count_nonzero(wt) < 2:
        raise ValueError("at least two nonzero temporal taper samples required")
    result = np.empty((len(z) - nt + 1, len(carriers), len(rates), len(scales), 2))
    for ci, c in enumerate(carriers):
        x = np.log2(f / c)
        wf = np.exp(-0.5 * (x / spectral_sd_oct) ** 2)
        for si, s in enumerate(scales):
            for di, sign in enumerate((-1, 1)):
                oscill = np.exp(2j * np.pi * sign * s * x)
                kf = wf * (oscill - np.sum(wf * oscill) / np.sum(wf))
                nf = np.linalg.norm(kf)
                if not np.isfinite(nf) or nf < 1e-14:
                    raise ValueError("degenerate spectral kernel")
                kf /= nf
                signal = z @ np.conj(kf)
                for ri, r in enumerate(rates):
                    osc = np.exp(2j * np.pi * r * t)
                    kt = wt * (osc - np.sum(wt * osc) / np.sum(wt))
                    nn = np.linalg.norm(kt)
                    if not np.isfinite(nn) or nn < 1e-14:
                        raise ValueError("degenerate temporal kernel")
                    kt /= nn
                    result[:, ci, ri, si, di] = np.abs(
                        np.convolve(signal, np.conj(kt[::-1]), mode="valid")
                    )
    return result


def local_spectrotemporal_modulation(
    stimulus: AudioStimulus,
    *,
    hop_s: float = 0.01,
    win_s: float = 0.025,
    window_s: float = 2.0,
    output_hop_s: float = 0.1,
    n_log_bins: int = 64,
    fmin: float = 50.0,
    fmax: float = 7500.0,
    carriers_hz=(125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0),
    rates_hz=(2.0, 4.0, 8.0, 16.0),
    scales_cpo=(0.5, 1.0, 2.0),
    spectral_sd_oct: float = 0.5,
) -> FeatureSeries:
    """Return amplitude [time, carrier, rate, scale, spectral sign].

    Complete 25-ms Hann-windowed FFT-power frames are projected to 64 ERB
    triangles and interpolated onto a uniform log2-frequency grid. Local
    spectral and temporal kernels are zero-DC and unit L2 norm. Values are
    absolute complex-filter responses on log10 power, not a loudness measure.

    Timestamps are causal availability endpoints. ``time_bounds_s`` are
    non-overlapping output-hop observation cells ending at those timestamps;
    ``metadata['computation_context']['dependency_bounds_s']`` separately
    records the complete waveform used by each filter window. There is no
    padding: incomplete frontend/filter windows are omitted. Channels are
    averaged before extraction. Frequency limits must match the sample rate;
    lower-rate audio is never silently resampled or truncated in frequency.
    """
    raw = np.asarray(stimulus.samples)
    if not np.issubdtype(raw.dtype, np.floating) or raw.size == 0:
        raise ValueError("nonempty floating audio required; scale integer PCM explicitly")
    if raw.ndim not in (1, 2) or not np.all(np.isfinite(raw)):
        raise ValueError("finite mono or time-by-channel audio required")
    if raw.ndim == 2 and raw.shape[1] == 0:
        raise ValueError("at least one audio channel required")
    sr = stimulus.sr_hz
    if isinstance(sr, (bool, np.bool_)) or not np.isfinite(sr) or sr <= 0 or int(sr) != sr:
        raise ValueError("positive integer sample rate required")
    sr = int(sr)
    offset = float(stimulus.start_offset_s)
    if not np.isfinite(offset):
        raise ValueError("finite audio start offset required")
    if isinstance(n_log_bins, (bool, np.bool_)) or not isinstance(n_log_bins, (int, np.integer)) or n_log_bins < 8:
        raise ValueError("n_log_bins must be an integer of at least eight")
    for name, value in dict(hop_s=hop_s, win_s=win_s, window_s=window_s,
                            output_hop_s=output_hop_s, spectral_sd_oct=spectral_sd_oct).items():
        if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not np.isfinite(fmin) or not np.isfinite(fmax) or not 0 < fmin < fmax <= sr / 2:
        raise ValueError("frequency bounds must lie below audio Nyquist")
    x = raw.mean(axis=1, dtype=np.float64) if raw.ndim == 2 else raw.astype(np.float64)
    hop, win = int(round(hop_s * sr)), int(round(win_s * sr))
    if hop < 1 or win < 2 or len(x) < win:
        raise ValueError("insufficient complete audio/frontend support")
    actual_hop = hop / sr
    step = int(round(output_hop_s / actual_hop))
    if step < 1:
        raise ValueError("output hop cannot be shorter than input hop")
    starts = np.arange(0, len(x) - win + 1, hop)
    frames = np.lib.stride_tricks.sliding_window_view(x, win)[::hop]
    taper = np.hanning(win + 1)[:-1]
    power = np.abs(np.fft.rfft(frames * taper, axis=1)) ** 2
    erb = power @ _erb_filters(sr, win, 64, fmin, fmax).T
    centers = _erb_centers(64, fmin, fmax)
    log_grid = np.linspace(np.log2(centers[0]), np.log2(centers[-1]), n_log_bins)
    log_power = np.log10(np.maximum(erb, 1e-10))
    coch = np.asarray([np.interp(log_grid, np.log2(centers), row) for row in log_power])
    values = local_modulation_kernel(
        coch, frequencies_hz=2 ** log_grid, frame_rate_hz=1 / actual_hop,
        carriers_hz=carriers_hz, rates_hz=rates_hz, scales_cpo=scales_cpo,
        window_s=window_s, spectral_sd_oct=spectral_sd_oct,
    )
    nt = int(round(window_s / actual_hop))
    index = np.arange(0, len(values), step)
    dependency = np.column_stack((starts[index] / sr,
                                 (starts[index + nt - 1] + win) / sr)) + offset
    times = dependency[:, 1]
    out_hop = step * actual_hop
    observation = np.column_stack((times - out_hop, times))
    # For a large requested output hop, the first cell may precede the input.
    observation[:, 0] = np.maximum(observation[:, 0], offset)
    params = dict(hop_s=hop_s, win_s=win_s, window_s=window_s,
                  output_hop_s=output_hop_s, n_log_bins=int(n_log_bins),
                  fmin=fmin, fmax=fmax, carriers_hz=[float(v) for v in carriers_hz],
                  rates_hz=[float(v) for v in rates_hz], scales_cpo=[float(v) for v in scales_cpo],
                  spectral_sd_oct=spectral_sd_oct)
    context = dict(schema="FeatureComputationContext/v1", clock=str(stimulus.clock),
                   policy="past_only", input_bounds_s=[offset, offset + len(x) / sr],
                   dependency_bounds_s=dependency.tolist(),
                   preprocessing=dict(channel_policy="mean", sample_rate_hz=sr,
                                      frontend="64 triangular ERB FFT-power bands; log2-frequency interpolation",
                                      taper="periodic Hann", log_power_floor=1e-10,
                                      padding="none; incomplete windows omitted"))
    metadata = extractor_metadata(
        "audio.modulation.local", params=params,
        extra=dict(backend="numpy_local_gabor", computation_context=context,
                   actual_frame_hop_s=actual_hop, actual_frame_win_s=win / sr,
                   actual_filter_frames=nt, actual_output_hop_s=out_hop,
                   value_units="absolute zero-DC unit-L2 Gabor response on log10 power",
                   direction_convention="positive spectral sign with positive temporal rate means downward contour",
                   qualification="synthetic-signal descriptor; no physiological auditory-model equivalence claimed"),
    )
    return FeatureSeries(
        values=values[index].astype(np.float32), times_s=times,
        dims=("time", "carrier_hz", "rate_hz", "scale_cpo", "spectral_sign"),
        coords=dict(carrier_hz=list(carriers_hz), rate_hz=list(rates_hz),
                    scale_cpo=list(scales_cpo), spectral_sign=[-1, 1]),
        metadata=metadata, time_bounds_s=observation,
        timebase=TimebaseSpec(kind="audio_hop", reference=stimulus.clock,
                              hop_s=out_hop, sampling_rate_hz=1 / out_hop,
                              alignment="offset", support=SupportSpec(kind="interval", anchor="offset")),
        temporal_context=stimulus.temporal_context,
    )
