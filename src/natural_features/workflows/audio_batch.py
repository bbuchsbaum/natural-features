"""Ergonomic batch workflow for short audio clips."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from natural_features.core.feature_types import FeatureSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.features.audio.lowlevel import mel, mfcc, rms, spectral_stats
from natural_features.features.audio.opensmile import egemaps_lld
from natural_features.features.speech.vad import energy_vad
from natural_features.fmri.design import concat_feature_series
from natural_features.fmri.resample import build_tr_grid, resample_feature_series


@dataclass
class AudioFileResult:
    file_id: str
    path: Path
    times_s: np.ndarray
    feature_names: list[str]
    matrix: np.ndarray
    dataframe: Any | None = None
    collapsed_feature_names: list[str] | None = None
    collapsed_vector: np.ndarray | None = None
    collapsed_dataframe: Any | None = None


@dataclass
class AudioBatchResult:
    files: dict[str, AudioFileResult]
    long_dataframe: Any | None = None
    collapsed_dataframe: Any | None = None


_FEATURES = {
    "rms": (rms, {"hop_s": 0.01, "win_s": 0.03}),
    "mel": (mel, {"hop_s": 0.01, "win_s": 0.03, "n_mels": 64}),
    "mfcc": (mfcc, {"hop_s": 0.01, "win_s": 0.03, "n_mfcc": 13, "n_mels": 40, "include_deltas": True}),
    "spectral_stats": (spectral_stats, {"hop_s": 0.01, "win_s": 0.03}),
    "vad": (energy_vad, {"hop_s": 0.02, "win_s": 0.03, "threshold": 0.5}),
    "opensmile_egemaps": (egemaps_lld, {"frame_s": 0.01}),
}


def _prefix_feature_names(fs: FeatureSeries, prefix: str) -> FeatureSeries:
    names = fs.coords.get("feature", [f"f{i}" for i in range(fs.values.reshape(fs.values.shape[0], -1).shape[1])])
    prefixed = [f"{prefix}{n}" for n in names]
    return FeatureSeries(
        values=fs.values,
        times_s=fs.times_s,
        dims=fs.dims,
        coords={"feature": prefixed},
        metadata=fs.metadata,
        timebase=fs.timebase,
    )


def _get_pandas():
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return None
    return pd


def _extract_selected_features(
    stimulus: AudioStimulus,
    *,
    selected_features: list[str],
    feature_params: dict[str, dict[str, Any]] | None,
    resolution_s: float,
    resample_method: str,
    execution_mode: str | None,
) -> FeatureSeries:
    duration_s = stimulus.samples.shape[0] / stimulus.sr_hz
    grid = build_tr_grid(duration_s=duration_s, tr_s=resolution_s, start_s=stimulus.start_offset_s)
    feat_spaces: list[FeatureSeries] = []
    feature_params = feature_params or {}
    for name in selected_features:
        if name not in _FEATURES:
            raise ValueError(
                f"Unknown feature '{name}'. Available: {sorted(_FEATURES.keys())}"
            )
        fn, defaults = _FEATURES[name]
        params = dict(defaults)
        params.update(feature_params.get(name, {}))
        if name == "opensmile_egemaps" and execution_mode is not None and "execution_mode" not in params:
            params["execution_mode"] = execution_mode
        fs = fn(stimulus, **params)
        fs = resample_feature_series(fs, tr_s=resolution_s, method=resample_method, time_grid_s=grid)
        fs = _prefix_feature_names(fs, f"{name}.")
        feat_spaces.append(fs)
    return concat_feature_series(feat_spaces, standardize=False, add_intercept=False)


def _parse_collapse(collapse: str | list[str] | None) -> list[str] | None:
    if collapse is None:
        return None
    if isinstance(collapse, str):
        token = collapse.strip().lower()
        if token in {"", "none"}:
            return None
        parts = [p.strip().lower() for p in token.replace(",", "+").split("+") if p.strip()]
    else:
        parts = [str(p).strip().lower() for p in collapse if str(p).strip()]
    allowed = {"mean", "sd", "min", "max"}
    invalid = [p for p in parts if p not in allowed]
    if invalid:
        raise ValueError(
            f"Unsupported collapse statistic(s): {invalid}. "
            f"Allowed: {sorted(allowed)}"
        )
    # Preserve user order while removing duplicates.
    out: list[str] = []
    for p in parts:
        if p not in out:
            out.append(p)
    return out if out else None


def _collapse_matrix(
    matrix: np.ndarray,
    feature_names: list[str],
    stats: list[str],
) -> tuple[np.ndarray, list[str]]:
    vals: list[np.ndarray] = []
    names: list[str] = []
    for stat in stats:
        if stat == "mean":
            vec = np.nanmean(matrix, axis=0)
        elif stat == "sd":
            vec = np.nanstd(matrix, axis=0)
        elif stat == "min":
            vec = np.nanmin(matrix, axis=0)
        elif stat == "max":
            vec = np.nanmax(matrix, axis=0)
        else:
            raise ValueError(f"Unsupported collapse stat: {stat}")
        vals.append(np.asarray(vec, dtype=np.float32))
        names.extend([f"{n}.{stat}" for n in feature_names])
    return np.concatenate(vals, axis=0).astype(np.float32), names


def extract_audio_files(
    paths: list[str | Path],
    *,
    resolution_s: float = 1.0,
    selected_features: list[str] | None = None,
    feature_params: dict[str, dict[str, Any]] | None = None,
    resample_method: str = "mean",
    execution_mode: str | None = None,
    as_dataframe: bool = True,
    collapse: str | list[str] | None = None,
) -> AudioBatchResult:
    if resolution_s <= 0:
        raise ValueError("resolution_s must be > 0")
    selected_features = selected_features or ["rms", "mfcc", "spectral_stats", "vad"]
    collapse_stats = _parse_collapse(collapse)
    pd = _get_pandas() if as_dataframe else None
    if as_dataframe and pd is None:
        raise RuntimeError(
            "pandas is required for as_dataframe=True. "
            "Install with: pip install natural-features[storage]"
        )
    pd_collapse = _get_pandas() if collapse_stats is not None else None

    files: dict[str, AudioFileResult] = {}
    long_rows = []
    collapsed_rows = []
    for raw in paths:
        p = Path(raw)
        stim = AudioStimulus.from_wav(p)
        dm = _extract_selected_features(
            stim,
            selected_features=selected_features,
            feature_params=feature_params,
            resolution_s=resolution_s,
            resample_method=resample_method,
            execution_mode=execution_mode,
        )
        names = [str(n) for n in dm.coords.get("feature", [])]
        df = None
        collapsed_vector = None
        collapsed_names = None
        collapsed_df = None
        if pd is not None:
            df = pd.DataFrame(dm.values, columns=names)
            df.insert(0, "time_s", dm.times_s)
            df.insert(0, "file_id", p.stem)
            long_rows.append(df)
        if collapse_stats is not None:
            collapsed_vector, collapsed_names = _collapse_matrix(dm.values, names, collapse_stats)
            if pd_collapse is not None:
                collapsed_df = pd_collapse.DataFrame([collapsed_vector], columns=collapsed_names)
                collapsed_df.insert(0, "file_id", p.stem)
                collapsed_rows.append(collapsed_df)
        files[p.stem] = AudioFileResult(
            file_id=p.stem,
            path=p,
            times_s=dm.times_s,
            feature_names=names,
            matrix=dm.values,
            dataframe=df,
            collapsed_feature_names=collapsed_names,
            collapsed_vector=collapsed_vector,
            collapsed_dataframe=collapsed_df,
        )
    long_df = None
    if pd is not None:
        long_df = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame()
    collapsed_batch_df = None
    if pd_collapse is not None:
        collapsed_batch_df = (
            pd_collapse.concat(collapsed_rows, ignore_index=True) if collapsed_rows else pd_collapse.DataFrame()
        )
    return AudioBatchResult(
        files=files,
        long_dataframe=long_df,
        collapsed_dataframe=collapsed_batch_df,
    )


def extract_audio_dir(
    directory: str | Path,
    *,
    pattern: str = "*.wav",
    resolution_s: float = 1.0,
    selected_features: list[str] | None = None,
    feature_params: dict[str, dict[str, Any]] | None = None,
    resample_method: str = "mean",
    execution_mode: str | None = None,
    as_dataframe: bool = True,
    collapse: str | list[str] | None = None,
) -> AudioBatchResult:
    d = Path(directory)
    paths = sorted(d.glob(pattern))
    return extract_audio_files(
        paths=paths,
        resolution_s=resolution_s,
        selected_features=selected_features,
        feature_params=feature_params,
        resample_method=resample_method,
        execution_mode=execution_mode,
        as_dataframe=as_dataframe,
        collapse=collapse,
    )
