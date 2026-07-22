"""HRF kernels and convolution."""

from __future__ import annotations

import math

import numpy as np

from natural_features.core.feature_types import FeatureSeries
from natural_features.core.timebase import TimebaseSpec
from natural_features.features.common import extractor_metadata
from natural_features.fmri.compat import has_fmrimod, map_hrf_name


def _gamma_pdf(t: np.ndarray, a: float, b: float) -> np.ndarray:
    t = np.maximum(t, 1e-12)
    return np.exp((a - 1) * np.log(t) - (t / b) - a * np.log(b) - math.lgamma(a))


def hrf_kernel(
    tr_s: float,
    *,
    kind: str = "glover",
    duration_s: float = 32.0,
    backend: str = "auto",
) -> np.ndarray:
    if tr_s <= 0:
        raise ValueError("tr_s must be > 0")
    t = np.arange(0.0, duration_s, tr_s, dtype=np.float64)
    use_fmrimod = backend == "fmrimod" or (backend == "auto" and has_fmrimod())
    if use_fmrimod:
        from fmrimod.hrf import get_hrf  # type: ignore

        hrf = get_hrf(map_hrf_name(kind))
        h = np.asarray(hrf(t), dtype=np.float64)
    elif kind == "glover":
        # Glover (1999) canonical double-gamma parameters.
        h = _gamma_pdf(t, 6, 0.9) - 0.35 * _gamma_pdf(t, 16, 0.9)
    elif kind == "spm":
        h = _gamma_pdf(t, 6, 1) - (1.0 / 6.0) * _gamma_pdf(t, 16, 1)
    else:
        raise ValueError(f"Unsupported HRF kind: {kind}")
    h = h / np.maximum(np.max(np.abs(h)), 1e-12)
    return h.astype(np.float32)


def hrf_convolve(
    feature: FeatureSeries,
    *,
    tr_s: float,
    kind: str = "glover",
    backend: str = "auto",
) -> FeatureSeries:
    if feature.values.ndim != 2:
        raise ValueError("hrf_convolve currently supports 2-D FeatureSeries only")
    h = hrf_kernel(tr_s=tr_s, kind=kind, backend=backend)
    out = np.zeros_like(feature.values, dtype=np.float32)
    for j in range(feature.values.shape[1]):
        out[:, j] = np.convolve(feature.values[:, j], h, mode="full")[: feature.values.shape[0]]
    metadata = dict(feature.metadata)
    metadata.update(extractor_metadata("fmri.hrf_convolve", params={"tr_s": tr_s, "kind": kind, "backend": backend}))
    return FeatureSeries(
        values=out,
        times_s=feature.times_s,
        dims=feature.dims,
        coords=feature.coords,
        metadata=metadata,
        timebase=TimebaseSpec(
            kind="windows",
            reference=feature.clock,
            stride_s=tr_s,
            window_s=tr_s,
            alignment="center",
        ),
        time_bounds_s=feature.time_bounds_s,
        temporal_context=feature.temporal_context,
    )
