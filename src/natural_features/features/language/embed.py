"""Word-aligned language embeddings."""

from __future__ import annotations

import numpy as np

from natural_features.core.execution import add_execution_provenance, resolve_execution_mode
from natural_features.core.feature_types import EventSeries, FeatureSeries
from natural_features.core.timebase import TimebaseSpec
from natural_features.features.common import extractor_metadata
from natural_features.util.hashing import stable_hash


def _fallback_embeddings(
    words: EventSeries,
    *,
    extractor_name: str = "language.embed.bert_words",
    model: str | None = None,
    layers: list[int],
    execution_mode: str,
    fallback_reason: str,
    pooling: str | None = None,
    dim: int = 64,
) -> FeatureSeries:
    labels = words.label if words.label is not None else np.array([""] * len(words), dtype=object)
    out = np.zeros((len(words), len(layers), dim), dtype=np.float32)
    for i, token in enumerate(labels):
        txt = str(token).lower()
        for j, layer in enumerate(layers):
            seed = int(
                stable_hash(
                    {
                        "extractor": extractor_name,
                        "model": model,
                        "token": txt,
                        "layer": int(layer),
                    },
                    length=8,
                ),
                16,
            ) % (2**32)
            rng = np.random.default_rng(seed)
            out[i, j, :] = rng.normal(0.0, 1.0, size=(dim,)).astype(np.float32)
    params: dict[str, object] = {"layers": layers}
    if model is not None:
        params["model"] = model
    if pooling is not None:
        params["pooling"] = pooling
    md = add_execution_provenance(
        extractor_metadata(extractor_name, params=params, extra={"backend": "fallback"}),
        execution_mode=execution_mode,
        fallback_used=True,
        fallback_reason=fallback_reason,
    )
    return FeatureSeries(
        values=out,
        times_s=words.onset_s,
        dims=("time", "layer", "unit"),
        coords={"layer": layers, "unit": [f"u{i}" for i in range(dim)]},
        metadata=md,
        timebase=TimebaseSpec(kind="tokens"),
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
    mode, strict_dependency = resolve_execution_mode(
        execution_mode=execution_mode,
        strict_dependency=strict_dependency,
    )
    layers = layers or [12]
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except Exception:
        if strict_dependency:
            raise RuntimeError("transformers+torch required for strict language embeddings.")
        return _fallback_embeddings(
            words,
            model=model,
            layers=layers,
            execution_mode=mode,
            fallback_reason="transformers/torch unavailable",
            pooling=pooling,
        )

    labels = words.label if words.label is not None else np.array([""] * len(words), dtype=object)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        net = AutoModel.from_pretrained(model, local_files_only=True)
    except Exception:
        if strict_dependency:
            raise RuntimeError(f"Model '{model}' unavailable locally for strict mode.")
        return _fallback_embeddings(
            words,
            model=model,
            layers=layers,
            execution_mode=mode,
            fallback_reason="local model unavailable",
            pooling=pooling,
        )

    emb_list = []
    try:
        for token in labels:
            txt = str(token) if str(token).strip() else "[UNK]"
            toks = tokenizer(txt, return_tensors="pt")
            with torch.no_grad():
                out = net(**toks, output_hidden_states=True)
            hidden = out.hidden_states
            layer_vecs = []
            for layer in layers:
                l_idx = max(0, min(int(layer), len(hidden) - 1))
                arr = hidden[l_idx][0].detach().cpu().numpy().astype(np.float32)
                if pooling in {"mean_subwords", "token"}:
                    vec = arr.mean(axis=0)
                else:
                    raise ValueError(f"Unsupported pooling: {pooling}")
                layer_vecs.append(vec)
            emb_list.append(np.stack(layer_vecs, axis=0))
    except Exception as exc:
        if strict_dependency:
            raise RuntimeError("Language embedding inference failed in strict mode.") from exc
        return _fallback_embeddings(
            words,
            model=model,
            layers=layers,
            execution_mode=mode,
            fallback_reason=f"inference failed: {type(exc).__name__}",
            pooling=pooling,
        )
    out = np.stack(emb_list, axis=0)
    md = add_execution_provenance(
        extractor_metadata(
            "language.embed.bert_words",
            params={"model": model, "layers": layers, "pooling": pooling},
            extra={"backend": "transformers_local"},
        ),
        execution_mode=mode,
        fallback_used=False,
    )
    return FeatureSeries(
        values=out,
        times_s=words.onset_s,
        dims=("time", "layer", "unit"),
        coords={"layer": layers, "unit": [f"u{i}" for i in range(out.shape[2])]},
        metadata=md,
        timebase=TimebaseSpec(kind="tokens"),
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
    """Return word-aligned hidden states from a causal language model."""

    if not isinstance(words, EventSeries):
        raise TypeError("lm_hidden_states requires an EventSeries")
    mode, strict_dependency = resolve_execution_mode(
        execution_mode=execution_mode,
        strict_dependency=strict_dependency,
    )
    layers = layers or [6, 12]
    params = {
        "model": model,
        "layers": layers,
        "pooling": pooling,
        "local_files_only": local_files_only,
    }
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except Exception:
        if strict_dependency:
            raise RuntimeError("transformers+torch required for strict causal LM hidden states.")
        return _fallback_embeddings(
            words,
            extractor_name="language.hidden_states",
            model=model,
            layers=layers,
            execution_mode=mode,
            fallback_reason="transformers/torch unavailable",
            pooling=pooling,
        )

    labels = words.label if words.label is not None else np.array([""] * len(words), dtype=object)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=local_files_only)
        net = AutoModelForCausalLM.from_pretrained(model, local_files_only=local_files_only)
    except Exception:
        if strict_dependency:
            raise RuntimeError(f"Model '{model}' unavailable locally for strict mode.")
        return _fallback_embeddings(
            words,
            extractor_name="language.hidden_states",
            model=model,
            layers=layers,
            execution_mode=mode,
            fallback_reason="local model unavailable",
            pooling=pooling,
        )

    emb_list = []
    try:
        for token in labels:
            txt = str(token) if str(token).strip() else getattr(tokenizer, "unk_token", None) or " "
            toks = tokenizer(txt, return_tensors="pt")
            with torch.no_grad():
                out = net(**toks, output_hidden_states=True)
            hidden = out.hidden_states
            layer_vecs = []
            for layer in layers:
                l_idx = max(0, min(int(layer), len(hidden) - 1))
                arr = hidden[l_idx][0].detach().cpu().numpy().astype(np.float32)
                if pooling == "mean_subwords":
                    vec = arr.mean(axis=0)
                elif pooling == "first_subword":
                    vec = arr[0]
                elif pooling == "last_subword":
                    vec = arr[-1]
                else:
                    raise ValueError(f"Unsupported pooling: {pooling}")
                layer_vecs.append(vec)
            emb_list.append(np.stack(layer_vecs, axis=0))
    except Exception as exc:
        if strict_dependency:
            raise RuntimeError("Causal LM hidden-state inference failed in strict mode.") from exc
        return _fallback_embeddings(
            words,
            extractor_name="language.hidden_states",
            model=model,
            layers=layers,
            execution_mode=mode,
            fallback_reason=f"inference failed: {type(exc).__name__}",
            pooling=pooling,
        )
    out_arr = np.stack(emb_list, axis=0)
    md = add_execution_provenance(
        extractor_metadata(
            "language.hidden_states",
            params=params,
            extra={"backend": "transformers_causal_lm"},
        ),
        execution_mode=mode,
        fallback_used=False,
    )
    return FeatureSeries(
        values=out_arr,
        times_s=words.onset_s,
        dims=("time", "layer", "unit"),
        coords={"layer": layers, "unit": [f"u{i}" for i in range(out_arr.shape[2])]},
        metadata=md,
        timebase=TimebaseSpec(kind="tokens"),
    )
