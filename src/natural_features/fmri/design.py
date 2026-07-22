"""Design-matrix assembly utilities."""

from __future__ import annotations

import numpy as np

from natural_features.core.feature_types import FeatureSeries
from natural_features.core.timebase import TimebaseSpec
from natural_features.features.common import extractor_metadata


def _zscore(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    return (x - mu) / np.maximum(sd, 1e-8)


def add_lags(feature: FeatureSeries, lags: list[int]) -> FeatureSeries:
    if feature.values.ndim != 2:
        raise ValueError("add_lags currently supports 2-D FeatureSeries only")
    lags_sorted = sorted(set(lags))
    if not lags_sorted:
        raise ValueError("lags cannot be empty")
    cols = []
    names = feature.coords.get("feature", [f"f{i}" for i in range(feature.values.shape[1])])
    out_names: list[str] = []
    for lag in lags_sorted:
        if lag < 0:
            raise ValueError("Negative lags are not supported")
        if lag == 0:
            shifted = feature.values
        elif lag >= feature.values.shape[0]:
            shifted = np.zeros_like(feature.values, dtype=np.float32)
        else:
            shifted = np.vstack([np.zeros((lag, feature.values.shape[1]), dtype=np.float32), feature.values[:-lag]])
        cols.append(shifted)
        out_names.extend([f"{n}_lag{lag}" for n in names])
    vals = np.concatenate(cols, axis=1)
    metadata = dict(feature.metadata)
    metadata.update(extractor_metadata("fmri.add_lags", params={"lags": lags_sorted}))
    return FeatureSeries(
        values=vals.astype(np.float32),
        times_s=feature.times_s,
        dims=("time", "feature"),
        coords={"feature": out_names},
        metadata=metadata,
        timebase=feature.timebase,
        time_bounds_s=feature.time_bounds_s,
        temporal_context=feature.temporal_context,
    )


def concat_feature_series(
    features: list[FeatureSeries],
    *,
    standardize: bool = True,
    add_intercept: bool = True,
) -> FeatureSeries:
    if not features:
        raise ValueError("features cannot be empty")
    base_times = features[0].times_s
    base_clock = features[0].clock
    base_bounds = features[0].temporal_bounds_s
    for f in features[1:]:
        if len(f.times_s) != len(base_times) or not np.allclose(f.times_s, base_times):
            raise ValueError("All feature spaces must share the same time grid")
        if f.clock != base_clock:
            raise ValueError("All feature spaces must share the same clock")
        if not np.allclose(f.temporal_bounds_s, base_bounds):
            raise ValueError("All feature spaces must share the same temporal support")
    mats = [f.values if f.values.ndim == 2 else f.values.reshape(f.values.shape[0], -1) for f in features]
    names: list[str] = []
    for i, f in enumerate(features):
        n = f.values.reshape(f.values.shape[0], -1).shape[1]
        feature_names = f.coords.get("feature", [f"space{i}_f{j}" for j in range(n)])
        if len(feature_names) != n:
            feature_names = [f"space{i}_f{j}" for j in range(n)]
        names.extend([str(x) for x in feature_names])
    x = np.concatenate(mats, axis=1).astype(np.float32)
    if standardize:
        x = _zscore(x).astype(np.float32)
    if add_intercept:
        x = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
        names.append("intercept")
    metadata = extractor_metadata(
        "fmri.design.concat",
        params={"standardize": standardize, "add_intercept": add_intercept},
    )
    context = features[0].temporal_context
    for feature in features[1:]:
        context = context.merged(feature.temporal_context)
    return FeatureSeries(
        values=x,
        times_s=base_times,
        dims=("time", "feature"),
        coords={"feature": names},
        metadata=metadata,
        timebase=TimebaseSpec(
            kind="windows",
            reference=base_clock,
            support=features[0].timebase.support,
        ),
        time_bounds_s=features[0].time_bounds_s,
        temporal_context=context,
    )
