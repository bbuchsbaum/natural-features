"""Multiscale language feature workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

import numpy as np

from natural_features.core.execution import resolve_execution_mode
from natural_features.core.feature_types import EventSeries, FeatureSeries
from natural_features.core.stimulus import AudioStimulus, TextStimulus
from natural_features.features.audio.lowlevel import rms
from natural_features.features.common import extractor_metadata
from natural_features.features.language.predictability import surprisal
from natural_features.features.language.providers import (
    LocalBoWEmbeddingProvider,
    make_embedding_provider,
    sanitize_provider_config,
)
from natural_features.features.speech.asr import whisper_transcribe
from natural_features.fmri.design import concat_feature_series
from natural_features.fmri.resample import build_tr_grid, resample_feature_series
from natural_features.util.hashing import stable_hash
from natural_features.util.io import atomic_numpy_save


@dataclass
class MultiscaleLanguageResult:
    by_scale: dict[float, FeatureSeries]
    by_scale_dataframe: dict[float, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    qc: dict[str, Any] = field(default_factory=dict)
    words: EventSeries | None = None
    sentences: EventSeries | None = None
    paragraphs: EventSeries | None = None


def _tokenize(text: str) -> list[str]:
    return [w for w in re.split(r"\s+", text.strip()) if w]


def _words_from_text(
    text: str,
    *,
    start_s: float = 0.0,
    word_duration_s: float = 0.4,
) -> EventSeries:
    tokens = _tokenize(text)
    onset = start_s + np.arange(len(tokens), dtype=np.float64) * word_duration_s
    offset = onset + word_duration_s
    md = extractor_metadata("language.words.from_text", params={"word_duration_s": word_duration_s})
    return EventSeries(
        onset_s=onset,
        offset_s=offset,
        label=np.asarray(tokens, dtype=object),
        confidence=np.ones(len(tokens), dtype=np.float32),
        metadata=md,
    )


def _resolve_words(
    input_data: AudioStimulus | EventSeries | TextStimulus | str | Path,
    *,
    transcript_text: str | None = None,
    execution_mode: str,
    strict_dependency: bool,
) -> tuple[EventSeries, float, dict[str, Any]]:
    if isinstance(input_data, EventSeries):
        labels = input_data.label if input_data.label is not None else np.array([], dtype=object)
        n = int(len(labels))
        end_s = float(input_data.offset_s[-1]) if len(input_data.offset_s) else 0.0
        return input_data, end_s, {"source": "event_series", "n_words": n}

    if isinstance(input_data, AudioStimulus):
        stim = input_data
    else:
        p = Path(str(input_data))
        if p.exists():
            if p.suffix.lower() == ".wav":
                stim = AudioStimulus.from_wav(p)
            else:
                txt = p.read_text(encoding="utf-8")
                words = _words_from_text(txt)
                return words, float(words.offset_s[-1]) if len(words) else 0.0, {"source": "text_file", "n_words": len(words)}
        elif isinstance(input_data, TextStimulus):
            words = _words_from_text(input_data.text)
            return words, float(words.offset_s[-1]) if len(words) else 0.0, {"source": "text_stimulus", "n_words": len(words)}
        else:
            words = _words_from_text(str(input_data))
            return words, float(words.offset_s[-1]) if len(words) else 0.0, {"source": "plain_text", "n_words": len(words)}

    asr = whisper_transcribe(
        stim,
        transcript_text=transcript_text,
        execution_mode=execution_mode,
        strict_dependency=strict_dependency,
    )
    words = asr["words"]
    duration_s = stim.start_offset_s + (stim.samples.shape[0] / stim.sr_hz)
    qc = dict(asr.get("qc", {}))
    qc["source"] = "audio_asr"
    return words, float(duration_s), qc


def _sentence_events(words: EventSeries, *, max_words_without_punct: int = 12) -> EventSeries:
    labels = words.label if words.label is not None else np.array([], dtype=object)
    if len(labels) == 0:
        md = extractor_metadata("language.units.sentences", params={})
        return EventSeries(
            onset_s=np.array([], dtype=np.float64),
            offset_s=np.array([], dtype=np.float64),
            label=np.array([], dtype=object),
            confidence=np.array([], dtype=np.float32),
            metadata=md,
        )
    out_on, out_off, out_txt, out_conf = [], [], [], []
    cur_idx: list[int] = []
    conf = words.confidence if words.confidence is not None else np.ones(len(labels), dtype=np.float32)
    punct_breaks = 0
    for i, tok in enumerate(labels):
        t = str(tok)
        cur_idx.append(i)
        is_break = t.endswith((".", "!", "?")) or len(cur_idx) >= max_words_without_punct
        if is_break:
            punct_breaks += 1
            out_on.append(float(words.onset_s[cur_idx[0]]))
            out_off.append(float(words.offset_s[cur_idx[-1]]))
            out_txt.append(" ".join(str(labels[j]) for j in cur_idx))
            out_conf.append(float(np.mean(conf[cur_idx])))
            cur_idx = []
    if cur_idx:
        out_on.append(float(words.onset_s[cur_idx[0]]))
        out_off.append(float(words.offset_s[cur_idx[-1]]))
        out_txt.append(" ".join(str(labels[j]) for j in cur_idx))
        out_conf.append(float(np.mean(conf[cur_idx])))
    md = extractor_metadata("language.units.sentences", params={"max_words_without_punct": max_words_without_punct})
    return EventSeries(
        onset_s=np.asarray(out_on, dtype=np.float64),
        offset_s=np.asarray(out_off, dtype=np.float64),
        label=np.asarray(out_txt, dtype=object),
        confidence=np.asarray(out_conf, dtype=np.float32),
        metadata=md,
    )


def _paragraph_events(
    sentences: EventSeries,
    *,
    target_duration_s: float = 16.0,
    max_sentences: int = 6,
) -> EventSeries:
    labels = sentences.label if sentences.label is not None else np.array([], dtype=object)
    if len(labels) == 0:
        md = extractor_metadata("language.units.paragraphs", params={})
        return EventSeries(
            onset_s=np.array([], dtype=np.float64),
            offset_s=np.array([], dtype=np.float64),
            label=np.array([], dtype=object),
            confidence=np.array([], dtype=np.float32),
            metadata=md,
        )
    conf = sentences.confidence if sentences.confidence is not None else np.ones(len(labels), dtype=np.float32)
    out_on, out_off, out_txt, out_conf = [], [], [], []
    block_idx: list[int] = []
    for i in range(len(labels)):
        block_idx.append(i)
        dur = float(sentences.offset_s[block_idx[-1]] - sentences.onset_s[block_idx[0]])
        if dur >= target_duration_s or len(block_idx) >= max_sentences:
            out_on.append(float(sentences.onset_s[block_idx[0]]))
            out_off.append(float(sentences.offset_s[block_idx[-1]]))
            out_txt.append(" ".join(str(labels[j]) for j in block_idx))
            out_conf.append(float(np.mean(conf[block_idx])))
            block_idx = []
    if block_idx:
        out_on.append(float(sentences.onset_s[block_idx[0]]))
        out_off.append(float(sentences.offset_s[block_idx[-1]]))
        out_txt.append(" ".join(str(labels[j]) for j in block_idx))
        out_conf.append(float(np.mean(conf[block_idx])))
    md = extractor_metadata(
        "language.units.paragraphs",
        params={"target_duration_s": target_duration_s, "max_sentences": max_sentences},
    )
    return EventSeries(
        onset_s=np.asarray(out_on, dtype=np.float64),
        offset_s=np.asarray(out_off, dtype=np.float64),
        label=np.asarray(out_txt, dtype=object),
        confidence=np.asarray(out_conf, dtype=np.float32),
        metadata=md,
    )


def _event_to_feature_series(
    events: EventSeries,
    values: np.ndarray,
    *,
    prefix: str,
    extractor_name: str,
    metadata_extra: dict[str, Any] | None = None,
) -> FeatureSeries:
    if values.ndim != 2:
        raise ValueError("values must be 2-D for conversion to FeatureSeries")
    if values.shape[0] != len(events.onset_s):
        raise ValueError("values rows must match number of events")
    times = 0.5 * (events.onset_s + events.offset_s)
    names = [f"{prefix}u{i}" for i in range(values.shape[1])]
    md = extractor_metadata(extractor_name, params={}, extra=metadata_extra or {})
    return FeatureSeries(
        values=values.astype(np.float32),
        times_s=times.astype(np.float64),
        dims=("time", "feature"),
        coords={"feature": names},
        metadata=md,
        timebase=events.timebase,
    )


def _normalize_text_for_cache(text: str) -> str:
    return " ".join(str(text).strip().split())


def _embed_with_cache(
    texts: list[str],
    *,
    provider: Any,
    cache_dir: Path,
) -> tuple[np.ndarray, dict[str, int]]:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32), {"hits": 0, "misses": 0, "unique_misses": 0}
    cache_dir.mkdir(parents=True, exist_ok=True)
    desc = provider.cache_descriptor()
    vectors: list[np.ndarray | None] = [None] * len(texts)
    missing_by_key: dict[str, dict[str, Any]] = {}
    hits = 0
    misses = 0
    for i, t in enumerate(texts):
        norm = _normalize_text_for_cache(t)
        key = stable_hash({"text": norm, "provider": desc}, length=32)
        p = cache_dir / f"{key}.npy"
        if p.exists():
            vectors[i] = np.load(p).astype(np.float32)
            hits += 1
        else:
            slot = missing_by_key.get(key)
            if slot is None:
                slot = {"norm_text": norm, "row_indices": []}
                missing_by_key[key] = slot
            slot["row_indices"].append(i)
            misses += 1
    if missing_by_key:
        missing_keys = list(missing_by_key.keys())
        unique_missing_texts = [str(missing_by_key[k]["norm_text"]) for k in missing_keys]
        embs = provider.embed_text_batch(unique_missing_texts).astype(np.float32)
        if embs.shape[0] != len(unique_missing_texts):
            raise ValueError("Provider returned unexpected batch size")
        for j, key in enumerate(missing_keys):
            vec = embs[j]
            p = cache_dir / f"{key}.npy"
            atomic_numpy_save(p, vec.astype(np.float32), allow_pickle=False)
            for row_idx in missing_by_key[key]["row_indices"]:
                vectors[row_idx] = vec.astype(np.float32)
    out = np.stack([v for v in vectors if v is not None], axis=0).astype(np.float32)
    return out, {"hits": hits, "misses": misses, "unique_misses": len(missing_by_key)}


def _lexical_controls(words: EventSeries) -> FeatureSeries:
    labels = words.label if words.label is not None else np.array([], dtype=object)
    vals = np.zeros((len(labels), 2), dtype=np.float32)
    for i, tok in enumerate(labels):
        txt = str(tok).strip()
        vals[i, 0] = float(len(txt))
        vals[i, 1] = float(txt.isalpha())
    md = extractor_metadata("language.lexical.controls", params={})
    return FeatureSeries(
        values=vals,
        times_s=words.onset_s,
        dims=("time", "feature"),
        coords={"feature": ["lex.word_length", "lex.is_alpha"]},
        metadata=md,
        timebase=words.timebase,
    )


def _prefix_features(fs: FeatureSeries, *, prefix: str) -> FeatureSeries:
    names = [str(n) for n in fs.coords.get("feature", [f"f{i}" for i in range(fs.values.shape[1])])]
    md = dict(fs.metadata)
    return FeatureSeries(
        values=fs.values,
        times_s=fs.times_s,
        dims=fs.dims,
        coords={"feature": [f"{prefix}{n}" for n in names]},
        metadata=md,
        timebase=fs.timebase,
    )


def _resample_multiscale(
    feature: FeatureSeries,
    *,
    scale_s: float,
    duration_s: float,
    method: str,
    window_policy: str,
    start_s: float,
) -> FeatureSeries:
    grid = build_tr_grid(duration_s=duration_s, tr_s=scale_s, start_s=start_s)
    if window_policy == "centered":
        return resample_feature_series(feature, tr_s=scale_s, method=method, time_grid_s=grid)
    if window_policy == "causal":
        shifted = grid - (0.5 * scale_s)
        shifted_fs = resample_feature_series(feature, tr_s=scale_s, method=method, time_grid_s=shifted)
        md = dict(shifted_fs.metadata)
        md["window_policy"] = "causal"
        return FeatureSeries(
            values=shifted_fs.values,
            times_s=grid,
            dims=shifted_fs.dims,
            coords=shifted_fs.coords,
            metadata=md,
            timebase=shifted_fs.timebase,
        )
    raise ValueError("window_policy must be one of {'centered','causal'}")


def _get_pandas():
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return None
    return pd


def extract_multiscale_language(
    input_data: AudioStimulus | EventSeries | TextStimulus | str | Path,
    *,
    scales_s: list[float] | tuple[float, ...] = (2.0, 4.0, 16.0),
    feature_families: list[str] | None = None,
    provider_config: dict[str, Any] | None = None,
    transcript_text: str | None = None,
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
    aggregation: str = "mean",
    window_policy: str = "centered",
    standardize: bool = True,
    add_intercept: bool = False,
    as_dataframe: bool = False,
    cache_dir: str | Path = ".nf_cache/language_embeddings",
) -> MultiscaleLanguageResult:
    mode, strict_dependency = resolve_execution_mode(
        execution_mode=execution_mode,
        strict_dependency=strict_dependency,
    )
    families = feature_families or [
        "sentence_embeddings",
        "paragraph_embeddings",
        "surprisal",
        "lexical_controls",
    ]
    scales = sorted({float(s) for s in scales_s})
    if not scales or min(scales) <= 0:
        raise ValueError("scales_s must contain positive values")

    words, duration_s, source_qc = _resolve_words(
        input_data,
        transcript_text=transcript_text,
        execution_mode=mode,
        strict_dependency=strict_dependency,
    )
    if len(words) == 0:
        raise ValueError("No words available for language feature extraction")
    sentences = _sentence_events(words)
    paragraphs = _paragraph_events(sentences)

    provider = None
    cache_stats = {"hits": 0, "misses": 0, "unique_misses": 0}
    family_features: list[FeatureSeries] = []
    cache_path = Path(cache_dir)

    provider_resolution = {
        "requested_provider": str((provider_config or {}).get("provider", "local_hash")),
        "resolved_provider": "none",
        "fallback_used": False,
    }
    if "sentence_embeddings" in families or "paragraph_embeddings" in families:
        try:
            provider = make_embedding_provider(provider_config)
            provider_resolution["resolved_provider"] = str(provider.provider_name)
        except Exception as exc:
            if strict_dependency:
                raise
            provider = LocalBoWEmbeddingProvider()
            provider_resolution["resolved_provider"] = str(provider.provider_name)
            provider_resolution["fallback_used"] = True
            provider_resolution["fallback_reason"] = f"{type(exc).__name__}: {exc}"

    if "sentence_embeddings" in families:
        txt = [str(x) for x in (sentences.label if sentences.label is not None else np.array([], dtype=object))]
        emb, stats = _embed_with_cache(txt, provider=provider, cache_dir=cache_path)
        cache_stats["hits"] += stats["hits"]
        cache_stats["misses"] += stats["misses"]
        cache_stats["unique_misses"] += stats["unique_misses"]
        fs = _event_to_feature_series(
            sentences,
            emb,
            prefix="sem.sent.",
            extractor_name="language.embed.sentences",
            metadata_extra={"provider_name": provider.provider_name, "model_name": provider.model_name},
        )
        family_features.append(fs)

    if "paragraph_embeddings" in families:
        txt = [str(x) for x in (paragraphs.label if paragraphs.label is not None else np.array([], dtype=object))]
        emb, stats = _embed_with_cache(txt, provider=provider, cache_dir=cache_path)
        cache_stats["hits"] += stats["hits"]
        cache_stats["misses"] += stats["misses"]
        cache_stats["unique_misses"] += stats["unique_misses"]
        fs = _event_to_feature_series(
            paragraphs,
            emb,
            prefix="sem.par.",
            extractor_name="language.embed.paragraphs",
            metadata_extra={"provider_name": provider.provider_name, "model_name": provider.model_name},
        )
        family_features.append(fs)

    if "surprisal" in families:
        family_features.append(_prefix_features(surprisal(words), prefix="lang."))
    if "lexical_controls" in families:
        family_features.append(_lexical_controls(words))
    if "speech_energy" in families:
        if isinstance(input_data, AudioStimulus) or (isinstance(input_data, (str, Path)) and Path(str(input_data)).exists()):
            stim = input_data if isinstance(input_data, AudioStimulus) else AudioStimulus.from_wav(Path(str(input_data)))
            family_features.append(_prefix_features(rms(stim), prefix="audio."))

    if not family_features:
        raise ValueError("No feature families selected")

    method = "mean" if aggregation in {"mean", "duration_weighted_mean"} else aggregation
    by_scale: dict[float, FeatureSeries] = {}
    by_scale_df: dict[float, Any] | None = {} if as_dataframe else None
    pd = _get_pandas() if as_dataframe else None
    if as_dataframe and pd is None:
        raise RuntimeError("pandas is required for as_dataframe=True")

    start_s = float(words.onset_s[0]) if len(words.onset_s) else 0.0
    for scale in scales:
        rendered = [
            _resample_multiscale(
                f,
                scale_s=scale,
                duration_s=duration_s,
                method=method,
                window_policy=window_policy,
                start_s=start_s,
            )
            for f in family_features
        ]
        dm = concat_feature_series(rendered, standardize=standardize, add_intercept=add_intercept)
        by_scale[scale] = dm
        if by_scale_df is not None and pd is not None:
            names = [str(x) for x in dm.coords.get("feature", [f"f{i}" for i in range(dm.values.shape[1])])]
            df = pd.DataFrame(dm.values, columns=names)
            df.insert(0, "time_s", dm.times_s)
            df.insert(0, "scale_s", scale)
            by_scale_df[scale] = df

    total = cache_stats["hits"] + cache_stats["misses"]
    qc = {
        "source_qc": source_qc,
        "execution_mode": mode,
        "n_words": int(len(words)),
        "n_sentences": int(len(sentences)),
        "n_paragraphs": int(len(paragraphs)),
        "cache_hits": int(cache_stats["hits"]),
        "cache_misses": int(cache_stats["misses"]),
        "cache_unique_misses": int(cache_stats["unique_misses"]),
        "cache_hit_fraction": float(cache_stats["hits"] / max(1, total)),
        "provider_resolution": provider_resolution,
    }
    md = extractor_metadata(
        "workflow.multiscale_language",
        params={
            "scales_s": scales,
            "feature_families": families,
            "aggregation": aggregation,
            "window_policy": window_policy,
            "standardize": standardize,
            "add_intercept": add_intercept,
        },
        extra={
            "provider_config": sanitize_provider_config(provider_config),
            "execution_mode": mode,
            "provider_resolution": provider_resolution,
        },
    )
    return MultiscaleLanguageResult(
        by_scale=by_scale,
        by_scale_dataframe=by_scale_df,
        metadata=md,
        qc=qc,
        words=words,
        sentences=sentences,
        paragraphs=paragraphs,
    )
