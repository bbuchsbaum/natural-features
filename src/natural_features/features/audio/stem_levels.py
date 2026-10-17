"""Framewise level summaries of the four pinned-Demucs source estimates."""
from __future__ import annotations

import numpy as np

from natural_features.core.backend_errors import BackendInferenceError
from natural_features.core.feature_types import FeatureSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.core.timebase import SupportSpec, TimebaseSpec
from natural_features.features.audio._music_contract import computation_context, mono_audio
from natural_features.features.audio.source_separation import (
    SEPARATOR_SHA256, _resample, load_separator_runtime,
)
from natural_features.features.common import extractor_metadata
from natural_features.util.hashing import stable_hash

STEMS = ("vocals", "drums", "bass", "other")
_load_runtime = load_separator_runtime  # deliberate test seam, matching vocal_f0


def _parameters(hop_s, window_s, db_floor, units):
    values = []
    for value, name in ((hop_s, "hop_s"), (window_s, "window_s"), (db_floor, "db_floor")):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be numeric, not boolean")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be finite") from exc
        if not np.isfinite(numeric):
            raise ValueError(f"{name} must be finite")
        values.append(numeric)
    hop_s, window_s, db_floor = values
    if hop_s <= 0 or window_s <= 0:
        raise ValueError("hop_s and window_s must be finite and positive")
    if abs(db_floor) > float(np.finfo(np.float32).max):
        raise ValueError("db_floor must be representable as finite float32")
    if units not in {"dbfs", "rms"}:
        raise ValueError("units must be 'dbfs' or 'rms'")
    return hop_s, window_s, db_floor


def _frames(n, sr, hop_s, window_s):
    hop, window = round(float(hop_s) * sr), round(float(window_s) * sr)
    if hop < 1 or window < 1:
        raise ValueError("hop_s and window_s must each span at least one input sample")
    starts = np.arange(0, max(0, n - window + 1), hop, dtype=np.int64)
    return starts, hop, window


def _batched_window_rms(stems, starts, window, *, batch_size=32):
    """RMS of complete windows without an allocation proportional to all rows.

    ``stems`` is ``(samples, signals, channels)``; channels are reduced as
    energy dimensions while signals remain output columns.
    """
    x = np.asarray(stems)
    starts = np.asarray(starts, dtype=np.int64)
    if x.ndim != 3 or starts.ndim != 1 or window < 1 or batch_size < 1:
        raise ValueError("expected (samples, signals, channels), one-dimensional starts, and positive sizes")
    result = np.empty((len(starts), x.shape[1]), dtype=np.float64)
    offsets = np.arange(window, dtype=np.int64)
    for first in range(0, len(starts), batch_size):
        stop = min(first + batch_size, len(starts))
        frames = x[starts[first:stop, None] + offsets]
        result[first:stop] = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=(1, 3)))
    return result


def audio_stem_levels(
    stimulus: AudioStimulus, *, separator_checkpoint: str | None = None,
    cache_dir: str | None = None, local_files_only: bool = True, device: str = "cpu",
    hop_s: float = 0.1, window_s: float = 0.25, units: str = "dbfs",
    db_floor: float = -120.0, include_ratio: bool = False,
) -> dict[str, FeatureSeries]:
    """Estimate complete-window source levels from fixed HTDemucs estimates.

    ``dbfs`` values are ``20*log10(RMS)`` floored at ``db_floor``.  The level
    RMS uses every sample of both output channels; it never averages stereo to
    mono.  Separation is whole-input and therefore every row has a whole-input
    dependency even though its observation support is one exact input frame.
    """
    mono_audio(stimulus)  # shared finite floating input validation
    if not isinstance(local_files_only, bool) or not isinstance(device, str) or not device:
        raise ValueError("local_files_only must be boolean and device a non-empty string")
    if not isinstance(include_ratio, bool):
        raise ValueError("include_ratio must be boolean")
    hop_s, window_s, db_floor = _parameters(hop_s, window_s, db_floor, units)
    x = np.asarray(stimulus.samples, dtype=np.float32)
    if x.ndim == 1:
        stereo = np.repeat(x[:, None], 2, axis=1)
    elif x.shape[1] == 1:
        stereo = np.repeat(x, 2, axis=1)
    elif x.shape[1] == 2:
        stereo = x
    else:
        raise ValueError("source separation accepts mono or stereo audio")
    starts, hop, window = _frames(len(stereo), stimulus.sr_hz, hop_s, window_s)
    runtime = _load_runtime(separator_checkpoint, cache_dir, local_files_only, device)
    separator_input = _resample(stereo, stimulus.sr_hz, 44100)
    try:
        estimated = runtime.separate_stereo(separator_input)
        if np.shape(estimated) != (len(separator_input), len(runtime.sources), 2) or not np.all(np.isfinite(estimated)):
            raise ValueError("separator returned invalid stems")
        restored = _resample(estimated, 44100, stimulus.sr_hz)
        if len(restored) < len(stereo):
            raise ValueError("separator resampling returned too few samples")
        restored = restored[:len(stereo)]
        if not np.all(np.isfinite(restored)):
            raise ValueError("separator resampling returned nonfinite samples")
        indices = [list(runtime.sources).index(name) for name in STEMS]
        stems = restored[:, indices, :]
    except Exception as exc:
        raise BackendInferenceError("stem_levels", "source-level separation failed") from exc
    rms = _batched_window_rms(stems, starts, window)
    silent = rms == 0
    raw_db = np.full_like(rms, -np.inf)
    positive = rms > 0
    raw_db[positive] = 20 * np.log10(rms[positive])
    below_floor = raw_db < db_floor
    levels = np.maximum(raw_db, db_floor) if units == "dbfs" else rms
    valid = np.isfinite(rms)
    names = list(STEMS)
    diagnostic_names = [f"{name}_{flag}" for name in STEMS for flag in ("valid", "silent", "below_floor")]
    diagnostics = np.column_stack([np.column_stack([valid[:, j], silent[:, j], below_floor[:, j]])
                                   for j in range(len(STEMS))]).astype(np.float32)
    if include_ratio:
        accompaniment = stems[:, 1:, :].sum(axis=1, keepdims=True, dtype=np.float64)
        acc_rms = _batched_window_rms(accompaniment, starts, window)[:, 0]
        ratio_valid = (rms[:, 0] > 0) & (acc_rms > 0)
        ratio = np.full(len(starts), np.nan, dtype=np.float64)
        ratio[ratio_valid] = 20 * np.log10(rms[ratio_valid, 0] / acc_rms[ratio_valid])
        levels = np.column_stack([levels, ratio])
        names.append("vocals_to_accompaniment_db")
        diagnostics = np.column_stack([diagnostics, ratio_valid.astype(np.float32)])
        diagnostic_names.append("ratio_valid")
    start = float(stimulus.start_offset_s)
    times = start + (starts + window / 2) / stimulus.sr_hz
    bounds = start + np.column_stack([starts, starts + window]) / stimulus.sr_hz
    duration = len(stereo) / stimulus.sr_hz
    context = computation_context(
        stimulus, policy="bidirectional", bounds_s=[[start, start + duration]],
        preprocessing={"resampler": "scipy.signal.resample_poly; default Kaiser 5; constant zero boundary",
                       "original_sr_hz": stimulus.sr_hz, "separator_sr_hz": 44100,
                       "original_samples": len(stereo), "separator_samples": len(separator_input),
                       "separator_normalization": "whole-input stereo RMS; gain restored; per-stem/channel DC removed; shifts=0; split=True; overlap=0.25",
                       "stereo_level_policy": "mean square over both channels; no mono averaging",
                       "separator_segment_s": 7.8},
    )
    params = {"hop_s": hop_s, "window_s": window_s, "units": units,
              "db_floor": db_floor, "include_ratio": include_ratio}
    pair_id = stable_hash({"samples": np.asarray(stimulus.samples), "sr_hz": stimulus.sr_hz,
                           "clock": str(stimulus.clock), "start_offset_s": stimulus.start_offset_s,
                           "params": params, "model_sha256": SEPARATOR_SHA256}, length=24)
    column_units = {name: units for name in STEMS}
    if include_ratio:
        column_units["vocals_to_accompaniment_db"] = "db"
    common = dict(code_version="stem-levels-v1", params=params,
                  model_revision=f"demucs:sha256:{SEPARATOR_SHA256}",
                  extra={"backend": "demucs_htdemucs", "fallback_used": False,
                         "execution_mode": "strict", "units": units,
                         "level_semantics": "estimated source waveform RMS; not source presence, leakage, or voice confidence",
                         "dbfs_semantics": "20*log10(RMS), with db_floor applied only to reported dbfs values",
                         "diagnostics_semantics": "valid is numerical validity; silent is exact zero RMS; below_floor is pre-floor dbfs below db_floor",
                         "ratio_semantics": "vocal/accompaniment waveform-sum RMS dB; NaN unless both RMS values are positive",
                         "column_units": column_units, "output_pair_id": pair_id,
                         "computation_context": context, "versions": runtime.versions})
    timebase = TimebaseSpec(kind="audio_hop", reference=stimulus.clock, hop_s=hop / stimulus.sr_hz,
                            window_s=window / stimulus.sr_hz, support=SupportSpec(kind="interval", anchor="center"))
    return {
        "default": FeatureSeries(levels.astype(np.float32), times, coords={"feature": names},
                                 metadata=extractor_metadata("audio.stem_levels", **common), timebase=timebase,
                                 time_bounds_s=bounds, temporal_context=stimulus.temporal_context),
        "diagnostics": FeatureSeries(diagnostics, times, coords={"feature": diagnostic_names},
                                     metadata=extractor_metadata("audio.stem_levels.diagnostics", **common), timebase=timebase,
                                     time_bounds_s=bounds, temporal_context=stimulus.temporal_context),
    }
