"""Execution policy and provenance for scientific extractors."""

from __future__ import annotations

from enum import Enum
from typing import Any


class ExecutionMode(str, Enum):
    """Supported execution policies for named scientific methods.

    A named extractor may use another implementation only when that
    implementation computes the same scientific quantity.  Proxy or surrogate
    quantities belong under their own extractor names, so the public execution
    policy is intentionally strict-only.
    """

    STRICT = "strict"

    def __str__(self) -> str:
        return self.value


def resolve_execution_mode(
    *,
    execution_mode: str | ExecutionMode | None = None,
    strict_dependency: bool | None = None,
    default_mode: str | ExecutionMode = ExecutionMode.STRICT,
) -> tuple[ExecutionMode, bool]:
    """Resolve the strict-only execution policy.

    ``strict_dependency=True`` remains a compatibility alias.  The former
    ``fallback`` mode and ``strict_dependency=False`` are rejected because they
    allowed scientifically different quantities to be returned under a named
    method's feature identifier.
    """

    raw_mode = execution_mode if execution_mode is not None else default_mode
    try:
        mode = ExecutionMode(str(raw_mode).strip().lower())
    except ValueError as exc:
        raise ValueError(
            "execution_mode must be 'strict'; proxy and surrogate methods "
            "must use their own explicit extractor names"
        ) from exc

    if strict_dependency is not None:
        strict_flag = bool(strict_dependency)
        if not strict_flag:
            raise ValueError(
                "strict_dependency=False is no longer supported; proxy and "
                "surrogate methods must use their own explicit extractor names"
            )

    return mode, True


def add_execution_provenance(
    metadata: dict[str, Any],
    *,
    execution_mode: str | ExecutionMode,
    fallback_used: bool,
    fallback_reason: str | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    """Attach normalized execution provenance fields."""

    out = dict(metadata)
    out["execution_mode"] = str(execution_mode)
    out["fallback_used"] = bool(fallback_used)
    if backend is not None:
        out["backend"] = backend
    if fallback_reason:
        out["fallback_reason"] = str(fallback_reason)
    return out
