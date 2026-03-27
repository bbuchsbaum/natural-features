from __future__ import annotations

import numpy as np

from natural_features.features.language.providers import (
    LocalBoWEmbeddingProvider,
    make_embedding_provider,
    sanitize_provider_config,
)


def test_local_bow_provider_basic_and_deterministic() -> None:
    p = LocalBoWEmbeddingProvider(dim=128)
    texts = ["the cat sat on the mat", "the cat sat near the mat", "quantum fields and gauge bosons"]
    emb1 = p.embed_text_batch(texts)
    emb2 = p.embed_text_batch(texts)
    assert emb1.shape == (3, 128)
    np.testing.assert_allclose(emb1, emb2)

    sim12 = float(np.dot(emb1[0], emb1[1]))
    sim13 = float(np.dot(emb1[0], emb1[2]))
    assert sim12 > sim13


def test_make_provider_local_bow() -> None:
    p = make_embedding_provider({"provider": "local_bow", "dim": 64})
    emb = p.embed_text_batch(["hello world", "hello there"])
    assert emb.shape == (2, 64)


def test_sanitize_provider_config_redacts_api_key() -> None:
    cfg = sanitize_provider_config({"provider": "openai", "api_key": "SECRET"})
    assert cfg["api_key"] == "***REDACTED***"
