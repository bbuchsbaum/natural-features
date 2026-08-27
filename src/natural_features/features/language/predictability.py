"""Language-model predictability features."""

from __future__ import annotations

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


def _token_negative_log_probabilities(
    logits: np.ndarray,
    target_ids: np.ndarray,
) -> np.ndarray:
    logits_arr = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(target_ids, dtype=np.int64)
    if logits_arr.ndim != 2:
        raise ValueError("logits must have shape (token, vocabulary)")
    if targets.shape != (logits_arr.shape[0],):
        raise ValueError("target_ids must contain one id per logits row")
    if not np.all(np.isfinite(logits_arr)):
        raise ValueError("logits must be finite")
    if np.any(targets < 0) or np.any(targets >= logits_arr.shape[1]):
        raise ValueError("target_ids contain an out-of-range vocabulary id")
    row_max = np.max(logits_arr, axis=1, keepdims=True)
    log_normalizer = row_max[:, 0] + np.log(
        np.exp(logits_arr - row_max).sum(axis=1)
    )
    return log_normalizer - logits_arr[np.arange(len(targets)), targets]


def _aggregate_subwords_to_words(
    token_surprisal: np.ndarray,
    offsets: np.ndarray,
    word_spans: list[tuple[int, int]],
    words: list[str],
) -> np.ndarray:
    nll = np.asarray(token_surprisal, dtype=np.float64)
    token_offsets = np.asarray(offsets, dtype=np.int64)
    if token_offsets.shape != (len(nll), 2):
        raise ValueError("offsets must have shape (token, 2)")
    values = np.zeros((len(words), 1), dtype=np.float32)
    assigned = np.zeros(len(words), dtype=np.int64)
    for token_nll, (token_start, token_end) in zip(nll, token_offsets):
        if int(token_end) <= int(token_start):
            continue
        matches = [
            i
            for i, (word_start, word_end) in enumerate(word_spans)
            if int(token_end) > word_start and int(token_start) < word_end
        ]
        if len(matches) != 1:
            raise RuntimeError("Tokenizer offsets do not map uniquely to input words")
        word_index = matches[0]
        values[word_index, 0] += np.float32(token_nll)
        assigned[word_index] += 1
    missing = [i for i, count in enumerate(assigned) if count == 0 and words[i]]
    if missing:
        raise RuntimeError(
            f"Tokenizer produced no lexical token for word indices {missing}"
        )
    return values


def surprisal(
    words: EventSeries,
    *,
    model: str = "gpt2",
    local_files_only: bool = True,
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
) -> FeatureSeries:
    """Return word surprisal as summed subword negative log probability.

    Values are measured in nats.  Each lexical token is predicted from a
    model-specific beginning-of-sequence token and all preceding lexical
    tokens.  This function never substitutes orthographic heuristics for
    language-model probability.
    """

    if not isinstance(words, EventSeries):
        raise TypeError("surprisal requires an EventSeries")
    mode, _strict = resolve_execution_mode(
        execution_mode=execution_mode,
        strict_dependency=strict_dependency,
    )
    labels = words.label if words.label is not None else np.array([""] * len(words), dtype=object)
    normalized = [str(token).strip() for token in labels]
    if not normalized:
        values = np.zeros((0, 1), dtype=np.float32)
    else:
        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except ImportError as exc:
            raise BackendDependencyError(
                "language-model surprisal",
                "transformers+torch are required",
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
            raise BackendLoadError(
                "language-model surprisal",
                f"causal model '{model}' is unavailable",
            ) from exc

        text_parts: list[str] = []
        word_spans: list[tuple[int, int]] = []
        cursor = 0
        for token in normalized:
            if text_parts:
                text_parts.append(" ")
                cursor += 1
            start = cursor
            text_parts.append(token)
            cursor += len(token)
            word_spans.append((start, cursor))
        text = "".join(text_parts)

        try:
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
        except Exception as exc:
            raise BackendInferenceError(
                "language-model surprisal",
                "tokenization failed",
            ) from exc
        offsets_value = encoded.pop("offset_mapping", None)
        if offsets_value is None:
            raise BackendInferenceError(
                "language-model surprisal",
                f"tokenizer for '{model}' does not provide offset mappings",
            )
        input_ids = encoded.get("input_ids")
        if input_ids is None:
            raise BackendInferenceError(
                "language-model surprisal",
                f"tokenizer for '{model}' did not return input_ids",
            )
        bos_id = getattr(tokenizer, "bos_token_id", None)
        if bos_id is None:
            bos_id = getattr(tokenizer, "eos_token_id", None)
        if bos_id is None:
            raise BackendInferenceError(
                "language-model surprisal",
                f"tokenizer for '{model}' has neither a BOS nor EOS token",
            )

        prefix = torch.tensor([[int(bos_id)]], dtype=input_ids.dtype, device=input_ids.device)
        model_input_ids = torch.cat([prefix, input_ids], dim=1)
        try:
            with torch.no_grad():
                logits = net(input_ids=model_input_ids).logits
        except Exception as exc:
            raise BackendInferenceError(
                "language-model surprisal",
                "causal model evaluation failed",
            ) from exc

        logits_np = logits.detach().cpu().numpy()[0, :-1, :].astype(np.float64)
        target_ids = input_ids.detach().cpu().numpy()[0].astype(np.int64)
        if logits_np.shape[0] != target_ids.shape[0]:
            raise BackendInferenceError(
                "language-model surprisal",
                "logits do not align with token ids",
            )
        token_surprisal = _token_negative_log_probabilities(
            logits_np,
            target_ids,
        )

        if hasattr(offsets_value, "detach"):
            offsets = offsets_value.detach().cpu().numpy()[0]
        else:
            offsets = np.asarray(offsets_value)[0]
        values = _aggregate_subwords_to_words(
            token_surprisal,
            offsets,
            word_spans,
            normalized,
        )

    md = add_execution_provenance(
        extractor_metadata(
            "language.predict.surprisal",
            params={"model": model, "local_files_only": local_files_only},
            model_revision=model,
            extra={
                "backend": "transformers_causal_lm",
                "quantity": "negative_log_probability",
                "unit": "nat",
                "subword_aggregation": "sum",
            },
        ),
        execution_mode=mode,
        fallback_used=False,
    )
    return FeatureSeries(
        values=values,
        times_s=words.onset_s,
        dims=("time", "feature"),
        coords={"feature": ["surprisal_nats"]},
        metadata=md,
        timebase=TimebaseSpec(kind="tokens"),
    )
