"""Lossless temporal references, support, and clock conversion utilities."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Literal

import numpy as np

TimebaseKind = str
SupportKind = Literal["point", "window", "interval"]
SupportAnchor = Literal["onset", "center", "offset"]


class ClockRef(str):
    """Stable identifier for the clock used by temporal coordinates.

    ``ClockRef`` is a ``str`` subclass so legacy callers and JSON encoders keep
    working while APIs can require an explicit temporal-reference type.
    """

    def __new__(cls, value: str) -> "ClockRef":
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("clock reference cannot be empty")
        return str.__new__(cls, normalized)

    @property
    def id(self) -> str:
        return str(self)


STIMULUS_CLOCK = ClockRef("stimulus")


@dataclass(frozen=True)
class ClockMap:
    """Directed affine mapping between two clocks.

    The mapping is always interpreted as
    ``target_time = scale * source_time + offset_s``.
    """

    source: ClockRef | str
    target: ClockRef | str
    scale: float = 1.0
    offset_s: float = 0.0

    def __post_init__(self) -> None:
        source = ClockRef(self.source)
        target = ClockRef(self.target)
        scale = float(self.scale)
        offset = float(self.offset_s)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("ClockMap.scale must be a positive finite value")
        if not np.isfinite(offset):
            raise ValueError("ClockMap.offset_s must be finite")
        if source == target and (not np.isclose(scale, 1.0) or not np.isclose(offset, 0.0)):
            raise ValueError("a clock cannot map to itself with a non-identity transform")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "offset_s", offset)

    def apply(self, values: Any) -> np.ndarray | float:
        arr = np.asarray(values, dtype=np.float64)
        converted = self.scale * arr + self.offset_s
        if arr.ndim == 0:
            return float(converted)
        return converted

    def inverse(self) -> "ClockMap":
        return ClockMap(
            source=self.target,
            target=self.source,
            scale=1.0 / self.scale,
            offset_s=-self.offset_s / self.scale,
        )

    def then(self, following: "ClockMap") -> "ClockMap":
        """Compose this mapping with ``following`` in application order."""

        if self.target != following.source:
            raise ValueError(
                f"cannot compose {self.source}->{self.target} with "
                f"{following.source}->{following.target}"
            )
        return ClockMap(
            source=self.source,
            target=following.target,
            scale=following.scale * self.scale,
            offset_s=(following.scale * self.offset_s) + following.offset_s,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": str(self.source),
            "target": str(self.target),
            "scale": self.scale,
            "offset_s": self.offset_s,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ClockMap":
        return cls(**value)


@dataclass(frozen=True)
class SupportSpec:
    """Meaning of a temporal row: point, fixed window, or row interval."""

    kind: SupportKind = "point"
    anchor: SupportAnchor = "center"
    width_s: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"point", "window", "interval"}:
            raise ValueError("support kind must be one of {'point','window','interval'}")
        if self.anchor not in {"onset", "center", "offset"}:
            raise ValueError("support anchor must be one of {'onset','center','offset'}")
        if self.width_s is not None:
            width = float(self.width_s)
            if not np.isfinite(width) or width < 0:
                raise ValueError("support width_s must be finite and non-negative")
            object.__setattr__(self, "width_s", width)
        if self.kind == "point" and self.width_s not in {None, 0.0}:
            raise ValueError("point support cannot have a positive width_s")
        if self.kind == "window" and (self.width_s is None or self.width_s <= 0):
            raise ValueError("window support requires a positive width_s")
        if self.kind == "interval" and self.width_s is not None:
            raise ValueError("interval support uses per-row bounds, not width_s")

    def bounds(self, times_s: Any, *, explicit_bounds_s: Any | None = None) -> np.ndarray:
        times = np.asarray(times_s, dtype=np.float64)
        if times.ndim != 1:
            raise ValueError("times_s must be 1-D")
        if explicit_bounds_s is not None:
            bounds = np.asarray(explicit_bounds_s, dtype=np.float64)
            if bounds.shape != (len(times), 2):
                raise ValueError("time_bounds_s must have shape (n_time, 2)")
            if not np.all(np.isfinite(bounds)) or np.any(bounds[:, 1] < bounds[:, 0]):
                raise ValueError("time_bounds_s must be finite with offset >= onset")
            return bounds
        if self.kind == "interval":
            raise ValueError("interval support requires per-row time_bounds_s")
        if self.kind == "point":
            return np.column_stack([times, times])
        assert self.width_s is not None
        if self.anchor == "onset":
            onset = times
            offset = times + self.width_s
        elif self.anchor == "offset":
            onset = times - self.width_s
            offset = times
        else:
            onset = times - (self.width_s / 2.0)
            offset = times + (self.width_s / 2.0)
        return np.column_stack([onset, offset])

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "anchor": self.anchor, "width_s": self.width_s}

    def scaled(self, scale: float) -> "SupportSpec":
        numeric = float(scale)
        if not np.isfinite(numeric) or numeric <= 0:
            raise ValueError("support scale must be a positive finite value")
        width = None if self.width_s is None else self.width_s * numeric
        return SupportSpec(kind=self.kind, anchor=self.anchor, width_s=width)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SupportSpec":
        return cls(**value)


def _maps_close(left: ClockMap, right: ClockMap) -> bool:
    return (
        left.source == right.source
        and left.target == right.target
        and np.isclose(left.scale, right.scale, rtol=1e-10, atol=1e-12)
        and np.isclose(left.offset_s, right.offset_s, rtol=1e-10, atol=1e-12)
    )


@dataclass(frozen=True)
class TemporalContext:
    """A validated, serializable graph of mappings between named clocks."""

    mappings: tuple[ClockMap, ...] | Iterable[ClockMap] = ()

    def __post_init__(self) -> None:
        mappings: list[ClockMap] = []
        for value in self.mappings:
            mapping = value if isinstance(value, ClockMap) else ClockMap.from_dict(value)
            if not any(_maps_close(mapping, existing) for existing in mappings):
                mappings.append(mapping)
        object.__setattr__(self, "mappings", tuple(mappings))
        self._validate_consistency()

    def _adjacency(self) -> dict[ClockRef, list[ClockMap]]:
        graph: dict[ClockRef, list[ClockMap]] = {}
        for mapping in self.mappings:
            graph.setdefault(ClockRef(mapping.source), []).append(mapping)
            graph.setdefault(ClockRef(mapping.target), []).append(mapping.inverse())
        return graph

    def _validate_consistency(self) -> None:
        graph = self._adjacency()
        for origin in graph:
            resolved: dict[ClockRef, ClockMap] = {
                origin: ClockMap(origin, origin)
            }
            queue = [origin]
            while queue:
                node = queue.pop(0)
                base = resolved[node]
                for edge in graph.get(node, []):
                    candidate = base.then(edge)
                    existing = resolved.get(ClockRef(edge.target))
                    if existing is None:
                        resolved[ClockRef(edge.target)] = candidate
                        queue.append(ClockRef(edge.target))
                    elif not _maps_close(existing, candidate):
                        raise ValueError(
                            f"inconsistent clock mappings between {origin!s} and {edge.target!s}"
                        )

    @property
    def clocks(self) -> tuple[ClockRef, ...]:
        values = {ClockRef(m.source) for m in self.mappings} | {
            ClockRef(m.target) for m in self.mappings
        }
        return tuple(sorted(values, key=str))

    def resolve(self, source: ClockRef | str, target: ClockRef | str) -> ClockMap:
        source_ref = ClockRef(source)
        target_ref = ClockRef(target)
        if source_ref == target_ref:
            return ClockMap(source_ref, target_ref)
        graph = self._adjacency()
        queue: list[tuple[ClockRef, ClockMap]] = [
            (source_ref, ClockMap(source_ref, source_ref))
        ]
        seen = {source_ref}
        while queue:
            node, base = queue.pop(0)
            for edge in graph.get(node, []):
                candidate = base.then(edge)
                edge_target = ClockRef(edge.target)
                if edge_target == target_ref:
                    return candidate
                if edge_target not in seen:
                    seen.add(edge_target)
                    queue.append((edge_target, candidate))
        raise KeyError(f"no clock mapping from {source_ref!s} to {target_ref!s}")

    def convert(
        self,
        values: Any,
        *,
        source: ClockRef | str,
        target: ClockRef | str,
    ) -> np.ndarray | float:
        return self.resolve(source, target).apply(values)

    def merged(self, *others: "TemporalContext") -> "TemporalContext":
        mappings = list(self.mappings)
        for other in others:
            mappings.extend(other.mappings)
        return TemporalContext(tuple(mappings))

    def to_dict(self) -> dict[str, Any]:
        return {"mappings": [mapping.to_dict() for mapping in self.mappings]}

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "TemporalContext":
        if not value:
            return cls()
        return cls(tuple(ClockMap.from_dict(item) for item in value.get("mappings", [])))

    @property
    def digest(self) -> str:
        payload: list[dict[str, Any]] = []
        clocks = self.clocks
        for index, source in enumerate(clocks):
            for target in clocks[index + 1 :]:
                try:
                    payload.append(self.resolve(source, target).to_dict())
                except KeyError:
                    continue
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:20]


def _normalize_anchor(alignment: str | None) -> SupportAnchor:
    value = str(alignment or "center").lower()
    if value in {"start", "left", "onset"}:
        return "onset"
    if value in {"end", "right", "offset"}:
        return "offset"
    return "center"


@dataclass(frozen=True)
class TimebaseSpec:
    kind: TimebaseKind
    reference: ClockRef | str = STIMULUS_CLOCK
    sampling_rate_hz: float | None = None
    hop_s: float | None = None
    window_s: float | None = None
    stride_s: float | None = None
    alignment: str | None = None
    support: SupportSpec | dict[str, Any] | None = None

    def __post_init__(self) -> None:
        kind = str(self.kind).strip()
        if not kind:
            raise ValueError("timebase kind cannot be empty")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "reference", ClockRef(self.reference))
        for name in ("sampling_rate_hz", "hop_s", "window_s", "stride_s"):
            value = getattr(self, name)
            if value is not None:
                numeric = float(value)
                if not np.isfinite(numeric) or numeric <= 0:
                    raise ValueError(f"{name} must be a positive finite value")
                object.__setattr__(self, name, numeric)
        support = self.support
        if isinstance(support, dict):
            support = SupportSpec.from_dict(support)
        if support is not None and not isinstance(support, SupportSpec):
            raise TypeError("support must be a SupportSpec, mapping, or None")
        if support is None:
            if self.kind == "events":
                support = SupportSpec(kind="interval", anchor="onset")
            elif self.window_s is not None:
                support = SupportSpec(
                    kind="window",
                    anchor=_normalize_anchor(self.alignment),
                    width_s=self.window_s,
                )
            else:
                support = SupportSpec(kind="point", anchor=_normalize_anchor(self.alignment))
        object.__setattr__(self, "support", support)

    @property
    def clock(self) -> ClockRef:
        return ClockRef(self.reference)

    def with_reference(self, reference: ClockRef | str) -> "TimebaseSpec":
        return TimebaseSpec(
            kind=self.kind,
            reference=reference,
            sampling_rate_hz=self.sampling_rate_hz,
            hop_s=self.hop_s,
            window_s=self.window_s,
            stride_s=self.stride_s,
            alignment=self.alignment,
            support=self.support,
        )

    def to_dict(self) -> dict[str, Any]:
        assert isinstance(self.support, SupportSpec)
        return {
            "kind": self.kind,
            "reference": str(self.reference),
            "sampling_rate_hz": self.sampling_rate_hz,
            "hop_s": self.hop_s,
            "window_s": self.window_s,
            "stride_s": self.stride_s,
            "alignment": self.alignment,
            "support": self.support.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None, *, default_kind: str) -> "TimebaseSpec":
        if not value:
            return cls(kind=default_kind)
        return cls(**value)


def validate_time_bounds(bounds_s: Any, n_time: int) -> np.ndarray:
    bounds = np.asarray(bounds_s, dtype=np.float64)
    if bounds.shape != (n_time, 2):
        raise ValueError("time_bounds_s must have shape (n_time, 2)")
    if not np.all(np.isfinite(bounds)):
        raise ValueError("time_bounds_s must contain only finite values")
    if np.any(bounds[:, 1] < bounds[:, 0]):
        raise ValueError("time_bounds_s offsets must be >= onsets")
    if len(bounds) > 1 and np.any(np.diff(bounds[:, 0]) < 0):
        raise ValueError("time_bounds_s onsets must be monotonic non-decreasing")
    return bounds


def times_from_rate(
    n_samples: int,
    sampling_rate_hz: float,
    *,
    start_offset_s: float = 0.0,
) -> np.ndarray:
    if n_samples < 0:
        raise ValueError("n_samples must be >= 0")
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be > 0")
    idx = np.arange(n_samples, dtype=np.float64)
    return start_offset_s + (idx / float(sampling_rate_hz))


def times_from_hop(
    n_steps: int,
    hop_s: float,
    *,
    start_offset_s: float = 0.0,
    center: bool = False,
    window_s: float | None = None,
) -> np.ndarray:
    if n_steps < 0:
        raise ValueError("n_steps must be >= 0")
    if hop_s <= 0:
        raise ValueError("hop_s must be > 0")
    shift = 0.0
    if center and window_s is not None:
        if window_s <= 0:
            raise ValueError("window_s must be > 0 when center=True")
        shift = window_s / 2.0
    return start_offset_s + shift + (np.arange(n_steps, dtype=np.float64) * hop_s)


def is_monotonic_non_decreasing(times_s: np.ndarray) -> bool:
    if times_s.ndim != 1:
        return False
    if len(times_s) < 2:
        return True
    return bool(np.all(np.diff(times_s) >= 0))


__all__ = [
    "ClockMap",
    "ClockRef",
    "STIMULUS_CLOCK",
    "SupportSpec",
    "TemporalContext",
    "TimebaseKind",
    "TimebaseSpec",
    "is_monotonic_non_decreasing",
    "times_from_hop",
    "times_from_rate",
    "validate_time_bounds",
]
