"""Contextual, word-aligned language-model representations."""

from __future__ import annotations

from typing import Any

import numpy as np

from natural_features.core.backend_errors import (
    BackendDependencyError,
    BackendInferenceError,
    BackendLoadError,
)
from natural_features.core.execution import add_execution_provenance, resolve_execution_mode
from natural_features.core.feature_types import EventSeries, FeatureSeries
from natural_features.core.timebase import TimebaseSpec
from natural_features.features.common import extractor_metadata


def _word_text_and_spans(words: EventSeries) -> tuple[list[str], str, list[tuple[int, int]]]:
    if not isinstance(words, EventSeries):
        raise TypeError("contextual embeddings require an EventSeries")
    if words.label is None:
        raise ValueError("contextual embeddings require word labels")

    labels = [str(value).strip() for value in words.label]
    empty = [index for index, value in enumerate(labels) if not value]
    if empty:
        raise ValueError(f"contextual embeddings require non-empty labels; empty indices: {empty}")

    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for label in labels:
        if parts:
            parts.append(" ")
            cursor += 1
        start = cursor
        parts.append(label)
        cursor += len(label)
        spans.append((start, cursor))
    return labels, "".join(parts), spans


def _offset_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    offsets = np.asarray(value)
    if offsets.ndim == 3 and offsets.shape[0] == 1:
        offsets = offsets[0]
    if offsets.ndim != 2 or offsets.shape[1] != 2:
        raise RuntimeError("tokenizer offset_mapping must have shape (token, 2)")
    return offsets.astype(np.int64, copy=False)


def _encode_joint_context(
    tokenizer: Any,
    *,
    text: str,
    backend: str,
) -> tuple[dict[str, Any], np.ndarray]:
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=False,
        )
    except Exception as exc:
        raise BackendInferenceError(backend, "joint tokenization failed") from exc
    offsets_value = encoded.pop("offset_mapping", None)
    if offsets_value is None:
        raise BackendInferenceError(backend, "the tokenizer did not return offset_mapping")
    return dict(encoded), _offset_array(offsets_value)


def _layer_array(hidden_states: Any, layer: int, *, backend: str) -> np.ndarray:
    if hidden_states is None:
        raise BackendInferenceError(backend, "the model did not return hidden states")
    index = int(layer)
    if index < 0 or index >= len(hidden_states):
        raise ValueError(
            f"Requested {backend} layer {index} is outside [0, {len(hidden_states) - 1}]"
        )
    value = hidden_states[index]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 3 or array.shape[0] != 1:
        raise BackendInferenceError(
            backend,
            f"hidden state must have shape (1, token, unit), received {array.shape}",
        )
    return array[0]


def _word_token_indices(
    offsets: np.ndarray,
    word_spans: list[tuple[int, int]],
) -> list[np.ndarray]:
    assignments: list[np.ndarray] = []
    for word_index, (word_start, word_end) in enumerate(word_spans):
        matches = np.flatnonzero(
            (offsets[:, 1] > offsets[:, 0])
            & (offsets[:, 1] > word_start)
            & (offsets[:, 0] < word_end)
        )
        if matches.size == 0:
            raise RuntimeError(
                f"Tokenizer produced no lexical token for word index {word_index}"
            )
        assignments.append(matches)

    lexical = [int(index) for matches in assignments for index in matches]
    if len(lexical) != len(set(lexical)):
        raise RuntimeError("Tokenizer offsets do not map uniquely to input words")
    return assignments


def _pool_contextual_hidden_states(
    hidden_states: Any,
    *,
    offsets: np.ndarray,
    word_spans: list[tuple[int, int]],
    layers: list[int],
    pooling: str,
    backend: str,
) -> np.ndarray:
    token_indices = _word_token_indices(offsets, word_spans)
    pooled_layers: list[np.ndarray] = []
    for layer in layers:
        values = _layer_array(hidden_states, int(layer), backend=backend)
        if values.shape[0] != offsets.shape[0]:
            raise BackendInferenceError(
                backend,
                "hidden-state token count does not match tokenizer offsets",
            )
        word_values: list[np.ndarray] = []
        for indices in token_indices:
            selected = values[indices]
            if pooling in {"mean_subwords", "token"}:
                vector = selected.mean(axis=0)
            elif pooling == "first_subword":
                vector = selected[0]
            elif pooling == "last_subword":
                vector = selected[-1]
            else:
                raise ValueError(f"Unsupported pooling: {pooling}")
            word_values.append(np.asarray(vector, dtype=np.float32))
        pooled_layers.append(np.stack(word_values, axis=0))
    return np.stack(pooled_layers, axis=1).astype(np.float32, copy=False)


def _contextual_result(
    words: EventSeries,
    *,
    values: np.ndarray,
    layers: list[int],
    metadata: dict[str, Any],
) -> FeatureSeries:
    return FeatureSeries(
        values=values,
        times_s=words.onset_s,
        dims=("time", "layer", "unit"),
        coords={"layer": layers, "unit": [f"u{i}" for i in range(values.shape[2])]},
        metadata=metadata,
        timebase=TimebaseSpec(kind="tokens", reference=words.clock),
        temporal_context=words.temporal_context,
    )


def bert_word_embeddings(
    words: EventSeries,
    *,
    model: str = "bert-base-uncased",
    layers: list[int] | None = None,
    pooling: str = "mean_subwords",
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
) -> FeatureSeries:
    """Return BERT word vectors from one jointly encoded lexical sequence."""

    mode, _strict = resolve_execution_mode(
        execution_mode=execution_mode,
        strict_dependency=strict_dependency,
    )
    layer_ids = [12] if layers is None else [int(layer) for layer in layers]
    _labels, text, word_spans = _word_text_and_spans(words)
    backend = "BERT"
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise BackendDependencyError(
            backend,
            "transformers and torch are required for language embeddings",
        ) from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, use_fast=True)
        net = AutoModel.from_pretrained(model, local_files_only=True)
    except Exception as exc:
        raise BackendLoadError(backend, f"model '{model}' is unavailable locally") from exc

    encoded, offsets = _encode_joint_context(tokenizer, text=text, backend=backend)
    try:
        with torch.no_grad():
            model_output = net(**encoded, output_hidden_states=True)
    except Exception as exc:
        raise BackendInferenceError(backend, "model evaluation failed") from exc
    values = _pool_contextual_hidden_states(
        getattr(model_output, "hidden_states", None),
        offsets=offsets,
        word_spans=word_spans,
        layers=layer_ids,
        pooling=pooling,
        backend=backend,
    )
    metadata = add_execution_provenance(
        extractor_metadata(
            "language.embed.bert_words",
            params={"model": model, "layers": layer_ids, "pooling": pooling},
            model_revision=model,
            extra={
                "backend": "transformers_bert",
                "context_encoding": "joint_sequence",
                "word_alignment": "tokenizer_offsets",
            },
        ),
        execution_mode=mode,
        fallback_used=False,
    )
    return _contextual_result(
        words,
        values=values,
        layers=layer_ids,
        metadata=metadata,
    )


def lm_hidden_states(
    words: EventSeries,
    *,
    model: str = "gpt2",
    layers: list[int] | None = None,
    pooling: str = "mean_subwords",
    local_files_only: bool = True,
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
) -> FeatureSeries:
    """Return causal-LM word states from one jointly encoded lexical sequence."""

    mode, _strict = resolve_execution_mode(
        execution_mode=execution_mode,
        strict_dependency=strict_dependency,
    )
    layer_ids = [6, 12] if layers is None else [int(layer) for layer in layers]
    _labels, text, word_spans = _word_text_and_spans(words)
    backend = "causal language model"
    params = {
        "model": model,
        "layers": layer_ids,
        "pooling": pooling,
        "local_files_only": local_files_only,
    }
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise BackendDependencyError(
            backend,
            "transformers and torch are required for hidden states",
        ) from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model,
            local_files_only=local_files_only,
            use_fast=True,
        )
        net = AutoModelForCausalLM.from_pretrained(
            model,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        raise BackendLoadError(backend, f"model '{model}' is unavailable") from exc

    encoded, offsets = _encode_joint_context(tokenizer, text=text, backend=backend)
    try:
        with torch.no_grad():
            model_output = net(**encoded, output_hidden_states=True)
    except Exception as exc:
        raise BackendInferenceError(backend, "model evaluation failed") from exc
    values = _pool_contextual_hidden_states(
        getattr(model_output, "hidden_states", None),
        offsets=offsets,
        word_spans=word_spans,
        layers=layer_ids,
        pooling=pooling,
        backend=backend,
    )
    metadata = add_execution_provenance(
        extractor_metadata(
            "language.hidden_states",
            params=params,
            model_revision=model,
            extra={
                "backend": "transformers_causal_lm",
                "context_encoding": "joint_sequence",
                "word_alignment": "tokenizer_offsets",
            },
        ),
        execution_mode=mode,
        fallback_used=False,
    )
    return _contextual_result(
        words,
        values=values,
        layers=layer_ids,
        metadata=metadata,
    )
