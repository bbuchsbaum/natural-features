"""Phase-specific failures for optional scientific backends."""

from __future__ import annotations


class OptionalBackendError(RuntimeError):
    """Base class for an optional backend failure at a known lifecycle phase."""

    phase = "backend"

    def __init__(self, backend: str, detail: str) -> None:
        self.backend = str(backend)
        self.detail = str(detail)
        super().__init__(f"{self.backend} {self.phase} failed: {self.detail}")


class BackendDependencyError(OptionalBackendError):
    """The backend's Python dependency could not be imported."""

    phase = "dependency"


class BackendLoadError(OptionalBackendError):
    """A dependency imported, but its configured model/resource did not load."""

    phase = "load"


class BackendInferenceError(OptionalBackendError):
    """A loaded backend failed while preprocessing or evaluating an input."""

    phase = "inference"


__all__ = [
    "BackendDependencyError",
    "BackendInferenceError",
    "BackendLoadError",
    "OptionalBackendError",
]
