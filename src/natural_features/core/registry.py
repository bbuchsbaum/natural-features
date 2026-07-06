"""Extractor registry and zoo spec loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module, metadata
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import yaml


ExtractorCallable = Callable[..., Any]


@dataclass(frozen=True)
class ExtractorSpec:
    name: str
    impl: str
    version: str = "1.0"
    label: str | None = None
    modalities: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)
    dependency_class: str | None = None
    cost_class: str | None = None
    bundles: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)


def _load_impl(impl: str) -> ExtractorCallable:
    if ":" not in impl:
        raise ValueError(f"Invalid impl '{impl}' (expected module:object)")
    mod_name, obj_name = impl.split(":", 1)
    mod: ModuleType = import_module(mod_name)
    fn = getattr(mod, obj_name, None)
    if fn is None:
        raise ValueError(f"Impl target not found: {impl}")
    return fn


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _validate_scalar_type(type_name: str, value: Any) -> bool:
    if type_name == "bool":
        return _is_bool(value)
    if type_name == "int":
        return isinstance(value, int) and not _is_bool(value)
    if type_name == "float":
        return (isinstance(value, (int, float)) and not _is_bool(value)) or isinstance(value, float)
    if type_name == "str":
        return isinstance(value, str)
    if type_name == "model_id":
        return isinstance(value, str)
    return True


def _validate_param_value(name: str, value: Any, spec: dict[str, Any]) -> None:
    if value is None:
        if bool(spec.get("nullable", False)) or ("default" in spec and spec.get("default") is None):
            return
        raise ValueError(f"Parameter '{name}' may not be null")

    t = str(spec.get("type", "")).strip()
    if t:
        if t.startswith("list[") and t.endswith("]"):
            subtype = t[5:-1].strip()
            if not isinstance(value, list):
                raise ValueError(f"Parameter '{name}' must be a list[{subtype}]")
            if subtype in {"int", "float", "str", "bool"}:
                for i, item in enumerate(value):
                    if not _validate_scalar_type(subtype, item):
                        raise ValueError(f"Parameter '{name}[{i}]' must be {subtype}")
        elif t.startswith("tuple[") and t.endswith("]"):
            parts = [x.strip() for x in t[6:-1].split(",") if x.strip()]
            if not isinstance(value, (list, tuple)) or len(value) != len(parts):
                raise ValueError(f"Parameter '{name}' must be {t}")
            for i, subtype in enumerate(parts):
                if not _validate_scalar_type(subtype, value[i]):
                    raise ValueError(f"Parameter '{name}[{i}]' must be {subtype}")
        elif not _validate_scalar_type(t, value):
            raise ValueError(f"Parameter '{name}' must be {t}")

    if "choices" in spec and value not in spec["choices"]:
        raise ValueError(f"Parameter '{name}' must be one of {spec['choices']}")

    if isinstance(value, (int, float)) and not _is_bool(value):
        if "min" in spec and float(value) < float(spec["min"]):
            raise ValueError(f"Parameter '{name}' must be >= {spec['min']}")
        if "max" in spec and float(value) > float(spec["max"]):
            raise ValueError(f"Parameter '{name}' must be <= {spec['max']}")

    if isinstance(value, list):
        if "min_items" in spec and len(value) < int(spec["min_items"]):
            raise ValueError(f"Parameter '{name}' must have at least {spec['min_items']} items")
        if "max_items" in spec and len(value) > int(spec["max_items"]):
            raise ValueError(f"Parameter '{name}' must have at most {spec['max_items']} items")


def _validate_extractor_spec_payload(spec_payload: dict[str, Any]) -> None:
    params = spec_payload.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError(f"Extractor '{spec_payload.get('name', '<unknown>')}' params must be a mapping")
    for p_name, p_spec in params.items():
        if not isinstance(p_spec, dict):
            raise ValueError(f"Extractor '{spec_payload.get('name', '<unknown>')}' param '{p_name}' must be a mapping")
        if "choices" in p_spec and not isinstance(p_spec["choices"], list):
            raise ValueError(f"Extractor '{spec_payload.get('name', '<unknown>')}' param '{p_name}' choices must be a list")
        if "default" in p_spec:
            _validate_param_value(p_name, p_spec["default"], p_spec)
    outputs = spec_payload.get("outputs", {})
    if outputs is not None and not isinstance(outputs, dict):
        raise ValueError(f"Extractor '{spec_payload.get('name', '<unknown>')}' outputs must be a mapping")


class Registry:
    def __init__(self) -> None:
        self._specs: dict[str, ExtractorSpec] = {}
        self._impls: dict[str, ExtractorCallable] = {}

    def register_spec(self, spec: ExtractorSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"Extractor already registered: {spec.name}")
        self._specs[spec.name] = spec
        self._impls[spec.name] = _load_impl(spec.impl)

    def register(self, spec_payload: dict[str, Any]) -> None:
        required = {"name", "impl"}
        missing = sorted(required - set(spec_payload.keys()))
        if missing:
            raise ValueError(f"Extractor spec missing required fields: {missing}")
        _validate_extractor_spec_payload(spec_payload)
        self.register_spec(ExtractorSpec(**spec_payload))

    def load_spec_file(self, path: str | Path) -> None:
        p = Path(path)
        payload = yaml.safe_load(p.read_text(encoding="utf-8"))
        extractors = payload.get("extractors", [])
        if not isinstance(extractors, list):
            raise ValueError(f"'extractors' must be a list in {p}")
        for item in extractors:
            self.register(item)

    def load_spec_dir(self, path: str | Path) -> None:
        p = Path(path)
        for file in sorted(p.glob("*.yaml")):
            self.load_spec_file(file)
        for file in sorted(p.glob("*.yml")):
            self.load_spec_file(file)
        for file in sorted(p.glob("*.json")):
            self.load_spec_file(file)

    def load_plugins(self, *, group: str = "natural_features.extractors") -> None:
        eps = metadata.entry_points()
        try:
            group_eps = eps.select(group=group)  # py311+
        except AttributeError:
            group_eps = eps.get(group, [])  # py310 fallback
        for ep in group_eps:
            if ep.name in self._specs:
                continue
            target = f"{ep.module}:{ep.attr}" if ep.attr else ep.module
            spec = ExtractorSpec(name=ep.name, impl=target, version="plugin")
            self.register_spec(spec)

    def list(self) -> list[ExtractorSpec]:
        return [self._specs[k] for k in sorted(self._specs.keys())]

    def get(self, name: str) -> ExtractorSpec:
        if name not in self._specs:
            raise KeyError(f"Unknown extractor: {name}")
        return self._specs[name]

    def impl(self, name: str) -> ExtractorCallable:
        if name not in self._impls:
            raise KeyError(f"Unknown extractor: {name}")
        return self._impls[name]

    def validated_params(self, name: str, params: dict[str, Any] | None) -> dict[str, Any]:
        spec = self.get(name)
        params = dict(params or {})
        declared = spec.params or {}
        unknown = sorted(set(params.keys()) - set(declared.keys()))
        if unknown:
            raise ValueError(f"Unknown parameter(s) for extractor '{name}': {unknown}")

        for p_name, value in params.items():
            p_spec = declared.get(p_name, {})
            if isinstance(p_spec, dict):
                _validate_param_value(p_name, value, p_spec)
        return params

    @classmethod
    def with_builtin_specs(cls) -> "Registry":
        reg = cls()
        spec_dir = Path(__file__).resolve().parents[1] / "zoo" / "specs"
        reg.load_spec_dir(spec_dir)
        reg.load_plugins()
        return reg
