"""Local backend markers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LocalBackendConfig:
    max_workers: int = 1
