"""Chunked map helpers for large-output ergonomics."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any


def chunked_map(
    items: Iterable[Any],
    fn: Callable[[list[Any]], Any],
    *,
    chunk_size: int,
    merge_fn: Callable[[list[Any]], Any] | None = None,
) -> Any:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    batch: list[Any] = []
    partials: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) >= chunk_size:
            partials.append(fn(batch))
            batch = []
    if batch:
        partials.append(fn(batch))
    if merge_fn is None:
        return partials
    return merge_fn(partials)

