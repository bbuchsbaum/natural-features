"""Cheap word-level discourse and surface-language features."""

from __future__ import annotations

import numpy as np

from natural_features.core.feature_types import EventSeries, FeatureSeries
from natural_features.core.timebase import TimebaseSpec
from natural_features.features.common import extractor_metadata

_STOP_WORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "can",
    "shall",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "it",
    "this",
    "that",
}

_FEATURE_NAMES = [
    "sentence_position",
    "is_repeated",
    "recurrence_distance",
    "ttr_local",
    "is_content_word",
]


def discourse_features(words: EventSeries, *, window_size: int = 5) -> FeatureSeries:
    if not isinstance(words, EventSeries):
        raise TypeError("discourse_features requires an EventSeries")
    labels = words.label if words.label is not None else np.asarray([""] * len(words), dtype=object)
    n_words = len(words)
    vals = np.zeros((n_words, len(_FEATURE_NAMES)), dtype=np.float32)
    lower = [str(label).strip().lower() for label in labels]
    window = max(0, int(window_size))
    last_seen: dict[str, int] = {}

    for i, token in enumerate(lower):
        vals[i, 0] = (i + 1) / max(n_words, 1)
        if token in last_seen:
            vals[i, 1] = 1.0
            vals[i, 2] = float(i - last_seen[token])
        last_seen[token] = i

        start = max(0, i - window)
        stop = min(n_words, i + window + 1)
        window_words = [w for w in lower[start:stop] if w]
        vals[i, 3] = len(set(window_words)) / max(len(window_words), 1)
        vals[i, 4] = float(len(token) >= 4 and token not in _STOP_WORDS)

    md = extractor_metadata(
        "language.discourse.features",
        params={"window_size": window},
        extra={"extractor_class": "heuristic", "backend": "python_native"},
    )
    return FeatureSeries(
        values=vals,
        times_s=words.onset_s,
        dims=("time", "feature"),
        coords={"feature": list(_FEATURE_NAMES)},
        metadata=md,
        timebase=TimebaseSpec(kind="tokens"),
    )
