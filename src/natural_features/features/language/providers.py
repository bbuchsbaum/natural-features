"""Embedding provider interfaces for language features."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from natural_features.util.hashing import stable_hash


class EmbeddingProvider(Protocol):
    provider_name: str
    model_name: str

    def embed_text_batch(self, texts: list[str]) -> np.ndarray:
        ...

    def cache_descriptor(self) -> dict[str, Any]:
        ...


def _bow_tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-zA-Z0-9']+", str(text).lower()) if t]


@dataclass
class LocalHashEmbeddingProvider:
    """Deterministic local fallback provider (no external dependencies)."""

    model_name: str = "local-hash-emb-256"
    dim: int = 256
    provider_name: str = "local_hash"

    def embed_text_batch(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = int(stable_hash({"text": str(t), "model": self.model_name}, length=8), 16) % (2**32)
            rng = np.random.default_rng(seed)
            v = rng.normal(0.0, 1.0, size=(self.dim,)).astype(np.float32)
            n = float(np.linalg.norm(v))
            out[i] = v / max(n, 1e-8)
        return out

    def cache_descriptor(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "model": self.model_name,
            "dim": self.dim,
        }


@dataclass
class LocalBoWEmbeddingProvider:
    """Deterministic lexical bag-of-words embedding fallback."""

    model_name: str = "local-bow-emb-1024"
    dim: int = 1024
    provider_name: str = "local_bow"

    def embed_text_batch(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        token_cache: dict[str, tuple[int, float]] = {}
        for i, t in enumerate(texts):
            toks = _bow_tokens(t)
            if not toks:
                continue
            for tok in toks:
                cached = token_cache.get(tok)
                if cached is None:
                    # Signed hashing improves robustness to collisions.
                    h = int(stable_hash({"tok": tok}, length=8), 16)
                    cached = (h % self.dim, 1.0 if ((h >> 1) & 1) == 0 else -1.0)
                    token_cache[tok] = cached
                idx, sign = cached
                out[i, idx] += sign
            n = float(np.linalg.norm(out[i]))
            out[i] = out[i] / max(n, 1e-8)
        return out

    def cache_descriptor(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "model": self.model_name,
            "dim": self.dim,
        }


@dataclass
class OpenAIEmbeddingProvider:
    """OpenAI embeddings provider (opt-in via API key)."""

    model_name: str = "text-embedding-3-large"
    api_key: str | None = None
    api_key_env_var: str = "OPENAI_API_KEY"
    timeout_s: float = 60.0
    max_retries: int = 3
    batch_size: int = 128
    provider_name: str = "openai"

    def __post_init__(self) -> None:
        key = self.api_key or os.environ.get(self.api_key_env_var)
        if not key:
            raise RuntimeError(
                f"Missing API key for provider=openai. "
                f"Set `{self.api_key_env_var}` or provide `api_key` in provider_config."
            )
        object.__setattr__(self, "api_key", key)

    def embed_text_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "openai package is required for provider=openai. "
                "Install optional dependency and retry."
            ) from e
        client = OpenAI(
            api_key=self.api_key,
            timeout=self.timeout_s,
            max_retries=self.max_retries,
        )
        vectors: list[np.ndarray] = []
        step = max(1, int(self.batch_size))
        for i in range(0, len(texts), step):
            batch = texts[i : i + step]
            last_exc: Exception | None = None
            for attempt in range(max(1, int(self.max_retries)) + 1):
                try:
                    resp = client.embeddings.create(model=self.model_name, input=batch)
                    vecs = [np.asarray(item.embedding, dtype=np.float32) for item in resp.data]
                    if len(vecs) != len(batch):
                        raise RuntimeError("OpenAI embeddings response size mismatch")
                    vectors.extend(vecs)
                    last_exc = None
                    break
                except Exception as exc:  # pragma: no cover - network/provider dependent
                    last_exc = exc
                    if attempt >= int(self.max_retries):
                        raise RuntimeError("OpenAI embedding request failed after retries") from exc
                    time.sleep(min(0.5 * (2**attempt), 4.0))
            if last_exc is not None:  # defensive
                raise RuntimeError("OpenAI embedding request failed")
        return np.stack(vectors, axis=0).astype(np.float32)

    def cache_descriptor(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "model": self.model_name,
            "timeout_s": self.timeout_s,
            "max_retries": self.max_retries,
            "batch_size": self.batch_size,
        }


def make_embedding_provider(provider_config: dict[str, Any] | None) -> EmbeddingProvider:
    cfg = dict(provider_config or {})
    provider = str(cfg.pop("provider", "local_hash"))
    if provider == "local_hash":
        return LocalHashEmbeddingProvider(
            model_name=str(cfg.get("model", "local-hash-emb-256")),
            dim=int(cfg.get("dim", 256)),
        )
    if provider == "local_bow":
        return LocalBoWEmbeddingProvider(
            model_name=str(cfg.get("model", "local-bow-emb-1024")),
            dim=int(cfg.get("dim", 1024)),
        )
    if provider == "openai":
        return OpenAIEmbeddingProvider(
            model_name=str(cfg.get("model", "text-embedding-3-large")),
            api_key=cfg.get("api_key"),
            api_key_env_var=str(cfg.get("api_key_env_var", "OPENAI_API_KEY")),
            timeout_s=float(cfg.get("timeout_s", 60.0)),
            max_retries=int(cfg.get("max_retries", 3)),
            batch_size=int(cfg.get("batch_size", 128)),
        )
    raise ValueError(f"Unsupported embedding provider: {provider}")


def sanitize_provider_config(provider_config: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(provider_config or {})
    if "api_key" in cfg:
        cfg["api_key"] = "***REDACTED***"
    return cfg
