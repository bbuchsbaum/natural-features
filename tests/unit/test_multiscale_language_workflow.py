from __future__ import annotations

import numpy as np
import pytest

from natural_features.core.feature_types import EventSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.core.timebase import ClockMap, TemporalContext, TimebaseSpec
from natural_features.features.common import extractor_metadata
from natural_features.workflows.multiscale_language import _embed_with_cache, extract_multiscale_language


def _audio() -> AudioStimulus:
    sr = 8000
    t = np.arange(sr * 3, dtype=np.float32) / sr
    x = (0.15 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return AudioStimulus.from_array(x, sr_hz=sr)


def test_multiscale_language_from_text_local_hash(tmp_path) -> None:
    text = "This is sentence one. This is sentence two with more words."
    res = extract_multiscale_language(
        text,
        scales_s=[2.0, 4.0, 16.0],
        provider_config={"provider": "local_hash", "dim": 64},
        cache_dir=tmp_path / "cache",
        as_dataframe=False,
    )
    assert sorted(res.by_scale.keys()) == [2.0, 4.0, 16.0]
    for s, fs in res.by_scale.items():
        assert fs.values.ndim == 2
        assert fs.values.shape[0] == len(fs.times_s)
        assert fs.values.shape[1] > 0
    assert res.qc["n_words"] > 0
    assert res.qc["cache_misses"] >= 1


def test_multiscale_language_cache_hits_on_repeat(tmp_path) -> None:
    text = "Short text for caching behavior check."
    kwargs = dict(
        scales_s=[2.0, 4.0],
        provider_config={"provider": "local_hash", "dim": 32},
        cache_dir=tmp_path / "cache",
        as_dataframe=False,
    )
    first = extract_multiscale_language(text, **kwargs)
    second = extract_multiscale_language(text, **kwargs)
    assert first.qc["cache_misses"] >= 1
    assert second.qc["cache_hits"] >= first.qc["cache_misses"]
    assert second.qc["cache_hit_fraction"] > 0


def test_multiscale_language_audio_asr_fallback() -> None:
    a = _audio()
    res = extract_multiscale_language(
        a,
        scales_s=[2.0, 4.0],
        provider_config={"provider": "local_hash", "dim": 32},
        feature_families=["sentence_embeddings", "surprisal", "lexical_controls"],
        execution_mode="fallback",
        as_dataframe=False,
    )
    assert sorted(res.by_scale.keys()) == [2.0, 4.0]
    assert res.qc["source_qc"]["source"] == "audio_asr"


def test_multiscale_language_openai_requires_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        extract_multiscale_language(
            "hello world",
            scales_s=[2.0],
            provider_config={"provider": "openai", "model": "text-embedding-3-large"},
            execution_mode="strict",
        )


def test_embed_with_cache_deduplicates_missing_texts(tmp_path) -> None:
    class _Provider:
        provider_name = "stub"
        model_name = "stub-model"

        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def cache_descriptor(self):
            return {"provider": self.provider_name, "model": self.model_name}

        def embed_text_batch(self, texts: list[str]) -> np.ndarray:
            self.calls.append(list(texts))
            out = np.zeros((len(texts), 4), dtype=np.float32)
            for i, t in enumerate(texts):
                out[i, 0] = float(len(t))
            return out

    p = _Provider()
    texts = ["repeat me", "repeat me", "unique token"]
    emb, stats = _embed_with_cache(texts, provider=p, cache_dir=tmp_path / "cache")
    assert emb.shape == (3, 4)
    assert stats["misses"] == 3
    assert stats["unique_misses"] == 2
    assert len(p.calls) == 1
    assert sorted(p.calls[0]) == sorted(["repeat me", "unique token"])

    emb2, stats2 = _embed_with_cache(texts, provider=p, cache_dir=tmp_path / "cache")
    assert emb2.shape == (3, 4)
    assert stats2["hits"] == 3
    assert stats2["unique_misses"] == 0
    assert len(p.calls) == 1


def test_multiscale_language_preserves_explicit_word_clock() -> None:
    context = TemporalContext((ClockMap("stimulus", "scan:run-01", offset_s=-23.0),))
    words = EventSeries(
        onset_s=np.array([7.0, 7.5, 8.0]),
        offset_s=np.array([7.4, 7.9, 8.4]),
        label=np.array(["one", "two", "three"], dtype=object),
        metadata=extractor_metadata("test.words"),
        timebase=TimebaseSpec(kind="events", reference="scan:run-01"),
        temporal_context=context,
    )

    result = extract_multiscale_language(
        words,
        scales_s=[0.5],
        feature_families=["lexical_controls"],
        standardize=False,
    )

    assert result.words is words
    assert result.sentences is not None
    assert result.sentences.clock == "scan:run-01"
    assert result.by_scale[0.5].clock == "scan:run-01"
    assert result.by_scale[0.5].temporal_context == context
