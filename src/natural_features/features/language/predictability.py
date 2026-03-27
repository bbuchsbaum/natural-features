"""Predictability features such as surprisal."""

from __future__ import annotations

import numpy as np

from natural_features.core.feature_types import EventSeries, FeatureSeries
from natural_features.core.timebase import TimebaseSpec
from natural_features.features.common import extractor_metadata


def surprisal(words: EventSeries, *, model: str = "gpt2") -> FeatureSeries:
    labels = words.label if words.label is not None else np.array([""] * len(words), dtype=object)
    vals = np.zeros((len(words), 1), dtype=np.float32)
    # Deterministic heuristic fallback: longer/rarer-looking tokens -> higher surprisal proxy.
    for i, token in enumerate(labels):
        txt = str(token).strip().lower()
        if not txt:
            vals[i, 0] = 0.0
            continue
        uniq = len(set(txt))
        vals[i, 0] = float(np.log1p(len(txt)) + 0.05 * uniq)
    md = extractor_metadata("language.predict.surprisal", params={"model": model}, extra={"backend": "heuristic"})
    return FeatureSeries(
        values=vals,
        times_s=words.onset_s,
        dims=("time", "feature"),
        coords={"feature": ["surprisal_proxy"]},
        metadata=md,
        timebase=TimebaseSpec(kind="tokens"),
    )

