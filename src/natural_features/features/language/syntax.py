"""Word-aligned syntactic proxy features."""

from __future__ import annotations

import numpy as np

from natural_features.core.execution import add_execution_provenance, resolve_execution_mode
from natural_features.core.feature_types import EventSeries, FeatureSeries
from natural_features.core.timebase import TimebaseSpec
from natural_features.features.common import extractor_metadata

_FUNCTION_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}

_FEATURES = [
    "token_length",
    "is_function_word",
    "is_capitalized",
    "is_punctuation",
    "noun_like",
    "verb_like",
    "modifier_like",
    "boundary_after",
]


def _labels(words: EventSeries) -> list[str]:
    raw = words.label if words.label is not None else np.asarray([""] * len(words), dtype=object)
    return [str(x) for x in raw]


def _heuristic_syntax(words: EventSeries) -> np.ndarray:
    labels = _labels(words)
    values = np.zeros((len(labels), len(_FEATURES)), dtype=np.float32)
    for i, token in enumerate(labels):
        stripped = token.strip()
        lower = stripped.lower()
        values[i, 0] = float(len(stripped))
        values[i, 1] = float(lower in _FUNCTION_WORDS)
        values[i, 2] = float(bool(stripped[:1].isupper()))
        values[i, 3] = float(bool(stripped) and all(not ch.isalnum() for ch in stripped))
        values[i, 4] = float(lower.endswith(("tion", "ment", "ness", "ity")) or (stripped[:1].isupper() and i > 0))
        values[i, 5] = float(lower.endswith(("ing", "ed", "ize", "ise")) or lower in {"is", "are", "was", "were", "be"})
        values[i, 6] = float(lower.endswith(("ly", "ous", "ive", "al", "ic")))
        values[i, 7] = float(stripped.endswith((".", "?", "!", ";", ":")))
    return values


def _spacy_syntax(words: EventSeries, *, model: str) -> tuple[np.ndarray, list[str]]:
    import spacy  # type: ignore

    nlp = spacy.load(model)
    labels = _labels(words)
    doc = nlp(" ".join(labels))
    tokens = list(doc)
    values = _heuristic_syntax(words)
    for i, token in enumerate(tokens[: len(labels)]):
        pos = getattr(token, "pos_", "")
        values[i, 1] = float(pos in {"ADP", "AUX", "CCONJ", "DET", "PART", "PRON", "SCONJ"})
        values[i, 3] = float(pos == "PUNCT")
        values[i, 4] = float(pos in {"NOUN", "PROPN", "PRON"})
        values[i, 5] = float(pos in {"AUX", "VERB"})
        values[i, 6] = float(pos in {"ADJ", "ADV"})
        values[i, 7] = float(getattr(token, "is_sent_end", False) or labels[i].endswith((".", "?", "!")))
    return values, [getattr(token, "pos_", "") for token in tokens[: len(labels)]]


def syntactic_features(
    words: EventSeries,
    *,
    model: str = "en_core_web_sm",
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
) -> FeatureSeries:
    """Return word-level syntactic indicator features."""

    if not isinstance(words, EventSeries):
        raise TypeError("syntactic_features requires an EventSeries")
    mode, strict = resolve_execution_mode(execution_mode=execution_mode, strict_dependency=strict_dependency)
    params = {"model": model}
    backend = "heuristic"
    fallback_used = True
    fallback_reason: str | None = "spacy unavailable"
    try:
        values, pos_tags = _spacy_syntax(words, model=model)
        backend = "spacy"
        fallback_used = False
        fallback_reason = None
        extra = {"backend": backend, "pos_tags": pos_tags}
    except Exception as exc:
        if strict:
            raise RuntimeError("spaCy syntax extraction failed in strict mode.") from exc
        values = _heuristic_syntax(words)
        extra = {"backend": backend}
        fallback_reason = f"spaCy unavailable: {type(exc).__name__}"
    md = add_execution_provenance(
        extractor_metadata("language.syntax", params=params, extra=extra),
        execution_mode=mode,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
    )
    return FeatureSeries(
        values=values.astype(np.float32),
        times_s=words.onset_s,
        dims=("time", "feature"),
        coords={"feature": list(_FEATURES)},
        metadata=md,
        timebase=TimebaseSpec(kind="tokens"),
    )
