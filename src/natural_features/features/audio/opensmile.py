"""openSMILE wrapper extractors."""

from __future__ import annotations

import numpy as np

from natural_features.core.backend_errors import (
    BackendDependencyError,
    BackendInferenceError,
    BackendLoadError,
)
from natural_features.core.execution import add_execution_provenance, resolve_execution_mode
from natural_features.core.feature_types import FeatureSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.core.timebase import TimebaseSpec
from natural_features.features.common import extractor_metadata


def egemaps_lld(
    stimulus: AudioStimulus,
    *,
    frame_s: float = 0.01,
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
) -> FeatureSeries:
    mode, _strict = resolve_execution_mode(
        execution_mode=execution_mode,
        strict_dependency=strict_dependency,
    )
    try:
        import opensmile  # type: ignore
    except ImportError as exc:
        raise BackendDependencyError("openSMILE", "the opensmile package is required") from exc

    x = stimulus.samples.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    try:
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )
    except Exception as exc:
        raise BackendLoadError("openSMILE", "eGeMAPSv02 configuration failed") from exc
    try:
        df = smile.process_signal(x, stimulus.sr_hz)
    except Exception as exc:
        raise BackendInferenceError("openSMILE", "signal processing failed") from exc
    values = df.to_numpy(dtype=np.float32)
    # opensmile index stores times; convert to seconds if available.
    if hasattr(df.index, "get_level_values"):
        try:
            starts = df.index.get_level_values("start").total_seconds().to_numpy(dtype=np.float64)
        except (AttributeError, KeyError, TypeError, ValueError):
            starts = np.arange(len(df), dtype=np.float64) * frame_s
    else:
        starts = np.arange(len(df), dtype=np.float64) * frame_s
    times = starts + stimulus.start_offset_s
    md = add_execution_provenance(
        extractor_metadata(
        "audio.opensmile.egemaps_lld",
        params={"frame_s": frame_s},
        extra={"backend": "opensmile"},
        ),
        execution_mode=mode,
        fallback_used=False,
    )
    return FeatureSeries(
        values=values,
        times_s=times,
        dims=("time", "feature"),
        coords={"feature": [str(c) for c in df.columns]},
        metadata=md,
        timebase=TimebaseSpec(kind="audio_hop", hop_s=frame_s, sampling_rate_hz=1.0 / frame_s),
    )
