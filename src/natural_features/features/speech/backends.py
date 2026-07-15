"""Optional aligner backend probing and selection."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as package_version
import shutil
from typing import Any, Literal, Protocol


BackendName = Literal["whisperx", "mfa", "gentle", "none", "passthrough"]


class AlignerBackend(Protocol):
    """Protocol for runtime aligner backends."""

    name: str

    def available(self) -> bool:
        """Return True when backend can run in current environment."""


@dataclass(frozen=True)
class BackendProbe:
    name: str
    available: bool
    version: str | None = None
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "version": self.version,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class AlignerResolution:
    selected_backend: BackendName
    fallback_used: bool
    reason: str | None
    probes: dict[str, BackendProbe]

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_backend": self.selected_backend,
            "fallback_used": self.fallback_used,
            "reason": self.reason,
            "probes": {k: v.as_dict() for k, v in self.probes.items()},
        }


def _pkg_version(name: str) -> str | None:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None
    except Exception:
        return None


def _probe_whisperx() -> BackendProbe:
    try:
        import_module("whisperx")
    except Exception as exc:
        return BackendProbe(
            name="whisperx",
            available=False,
            reason=f"{type(exc).__name__}: {exc}",
        )
    return BackendProbe(name="whisperx", available=True, version=_pkg_version("whisperx"))


def _probe_mfa() -> BackendProbe:
    mfa_path = shutil.which("mfa")
    if not mfa_path:
        return BackendProbe(name="mfa", available=False, reason="mfa executable not found")
    return BackendProbe(name="mfa", available=True, version=None)


def _probe_gentle() -> BackendProbe:
    try:
        import_module("gentle")
    except Exception as exc:
        return BackendProbe(
            name="gentle",
            available=False,
            reason=f"{type(exc).__name__}: {exc}",
        )
    return BackendProbe(name="gentle", available=True, version=_pkg_version("gentle"))


def probe_alignment_backends() -> dict[str, BackendProbe]:
    """Probe all known alignment backends without importing heavy runtimes."""

    return {
        "whisperx": _probe_whisperx(),
        "mfa": _probe_mfa(),
        "gentle": _probe_gentle(),
    }


def resolve_aligner_backend(
    *,
    requested: str = "auto",
    preferred_order: tuple[str, ...] = ("whisperx", "mfa"),
) -> AlignerResolution:
    probes = probe_alignment_backends()
    req = str(requested).strip().lower()

    if req in {"none", "passthrough"}:
        return AlignerResolution("passthrough", False, "explicit passthrough requested", probes)

    if req != "auto":
        if req not in probes:
            raise ValueError(f"Unknown aligner backend '{requested}'. Expected one of: auto, whisperx, mfa, gentle, none")
        probe = probes[req]
        if probe.available:
            return AlignerResolution(req, False, None, probes)  # type: ignore[arg-type]
        return AlignerResolution(
            "passthrough",
            True,
            f"requested backend '{req}' unavailable: {probe.reason or 'not available'}",
            probes,
        )

    for name in preferred_order:
        probe = probes.get(name)
        if probe is not None and probe.available:
            return AlignerResolution(name, False, None, probes)  # type: ignore[arg-type]
    return AlignerResolution("passthrough", True, "no alignment backend available", probes)
