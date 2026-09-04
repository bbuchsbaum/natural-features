"""OLS residualization of one FeatureSeries on others.

Full-stimulus residuals are a constructed feature space. Nested encoding must
residualize inside cross-validation folds downstream; this helper does not
replace commonality analysis.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from natural_features.core.feature_types import FeatureSeries
from natural_features.core.timebase import TimebaseSpec
from natural_features.features.common import extractor_metadata
from natural_features.fmri.resample import resample_feature_series


def _as_predictor_list(
    predictors: FeatureSeries | Sequence[FeatureSeries],
) -> list[FeatureSeries]:
    if isinstance(predictors, FeatureSeries):
        return [predictors]
    items = list(predictors)
    if not items:
        raise ValueError("predictors must contain at least one FeatureSeries")
    for item in items:
        if not isinstance(item, FeatureSeries):
            raise TypeError("predictors must be FeatureSeries objects")
    return items


def residualize_feature_series(
    target: FeatureSeries,
    predictors: FeatureSeries | Sequence[FeatureSeries],
    *,
    hop_s: float = 0.01,
    method: str = "linear",
    add_intercept: bool = True,
) -> FeatureSeries:
    """Return the OLS residual of ``target`` after predicting it from ``predictors``.

    All series are resampled onto a shared analysis hop before the fit. Each
    target column is residualized independently. Metadata records predictor
    extractor names, the analysis hop, and per-column in-sample R².
    """

    if hop_s <= 0:
        raise ValueError("hop_s must be > 0")
    if target.values.ndim != 2:
        raise ValueError("target must be a 2-D FeatureSeries")
    pred_list = _as_predictor_list(predictors)
    for pred in pred_list:
        if pred.values.ndim != 2:
            raise ValueError("predictors must be 2-D FeatureSeries objects")

    t0 = min(float(target.times_s[0]), min(float(p.times_s[0]) for p in pred_list))
    t1 = max(float(target.times_s[-1]), max(float(p.times_s[-1]) for p in pred_list))
    n = int(np.floor((t1 - t0) / hop_s)) + 1
    if n < 2:
        raise ValueError("residualize requires at least two analysis frames")
    grid = t0 + np.arange(n, dtype=np.float64) * hop_s

    tgt = resample_feature_series(target, hop_s, method=method, time_grid_s=grid)
    pred_names: list[str] = []
    x_parts: list[np.ndarray] = []
    if add_intercept:
        x_parts.append(np.ones((n, 1), dtype=np.float64))
    for pred in pred_list:
        resampled = resample_feature_series(
            pred, hop_s, method=method, time_grid_s=grid
        )
        x_parts.append(np.asarray(resampled.values, dtype=np.float64))
        pred_names.append(str(pred.metadata.get("extractor_name", "predictor")))
    x = np.concatenate(x_parts, axis=1)
    y = np.asarray(tgt.values, dtype=np.float64)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ beta
    resid = y - fitted
    ss_res = np.sum(resid**2, axis=0)
    ss_tot = np.sum((y - y.mean(axis=0, keepdims=True)) ** 2, axis=0)
    r2 = 1.0 - (ss_res / np.maximum(ss_tot, 1e-12))
    names = [
        str(x) for x in tgt.coords.get("feature", [f"f{i}" for i in range(y.shape[1])])
    ]
    md = extractor_metadata(
        "features.residualize",
        params={
            "hop_s": hop_s,
            "method": method,
            "add_intercept": add_intercept,
            "predictors": pred_names,
        },
        extra={
            "backend": "ols",
            "analysis_hop_s": hop_s,
            "predictors": pred_names,
            "r2_per_column": {name: float(val) for name, val in zip(names, r2)},
            "cv_note": (
                "Full-stimulus OLS. Residualize inside CV folds for nested encoding."
            ),
        },
    )
    return FeatureSeries(
        values=resid.astype(np.float32),
        times_s=grid,
        dims=("time", "feature"),
        coords={"feature": names},
        metadata=md,
        timebase=TimebaseSpec(
            kind="audio_hop", hop_s=hop_s, sampling_rate_hz=1.0 / hop_s
        ),
        temporal_context=target.temporal_context,
    )
