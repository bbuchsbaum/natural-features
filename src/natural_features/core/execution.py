"""Execution-mode helpers for fail-fast vs explicit fallback behavior."""

from __future__ import annotations

from typing import Any

_VALID_MODES = {"strict", "fallback"}


def resolve_execution_mode(
    *,
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
    default_mode: str = "strict",
) -> tuple[str, bool]:
    """Resolve execution mode, defaulting named methods to fail-fast execution.

    ``strict_dependency`` remains a compatibility alias. Passing ``False`` is
    therefore an explicit request for fallback behavior; omitting both controls
    selects strict execution.
    """

    mode = str(execution_mode).strip().lower() if execution_mode is not None else str(default_mode).strip().lower()
    if mode not in _VALID_MODES:
        raise ValueError(f"execution_mode must be one of {_VALID_MODES}, got '{mode}'")

    if strict_dependency is not None:
        strict_flag = bool(strict_dependency)
        strict_from_mode = mode == "strict"
        if execution_mode is not None and strict_flag != strict_from_mode:
            raise ValueError("Conflicting execution controls: strict_dependency disagrees with execution_mode")
        if execution_mode is None:
            mode = "strict" if strict_flag else "fallback"

    return mode, mode == "strict"


def add_execution_provenance(
    metadata: dict[str, Any],
    *,
    execution_mode: str,
    fallback_used: bool,
    fallback_reason: str | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    """Attach normalized execution provenance fields."""

    out = dict(metadata)
    out["execution_mode"] = execution_mode
    out["fallback_used"] = bool(fallback_used)
    if backend is not None:
        out["backend"] = backend
    if fallback_reason:
        out["fallback_reason"] = str(fallback_reason)
    return out
