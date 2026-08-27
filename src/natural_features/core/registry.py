"""Typed extractor catalogue contracts and registry loading."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from importlib import import_module, metadata
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import yaml


ExtractorCallable = Callable[..., Any]
_MISSING = object()


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class Modality(_StringEnum):
    AUDIO = "audio"
    EVENTS = "events"
    FEATURES = "features"
    IMAGE = "image"
    TEXT = "text"
    TOKENS = "tokens"
    VIDEO = "video"
    WORDS = "words"


class DependencyClass(_StringEnum):
    BASE_PYTHON = "base_python"
    OPTIONAL_API = "optional_api"
    OPTIONAL_PYTHON = "optional_python"


class CostClass(_StringEnum):
    CHEAP = "cheap"
    MODERATE = "moderate"
    EXPENSIVE = "expensive"


class ScalarParameterType(_StringEnum):
    BOOL = "bool"
    FLOAT = "float"
    INT = "int"
    MODEL_ID = "model_id"
    STR = "str"


class ParameterContainer(_StringEnum):
    SCALAR = "scalar"
    LIST = "list"
    TUPLE = "tuple"


class OutputSchemaId(_StringEnum):
    AUDIO_STIMULUS = "AudioStimulus/v1"
    EVENT_SERIES = "EventSeries/v1"
    FEATURE_SERIES = "FeatureSeries/v1"
    TRACK_SERIES = "TrackSeries/v1"
    VIDEO_STIMULUS = "VideoStimulus/v1"
    DICT = "dict"


class OutputKind(_StringEnum):
    AUDIO = "audio"
    EVENTS = "events"
    FEATURES = "features"
    PHONEMES = "phonemes"
    QC = "qc"
    TRACKS = "tracks"
    VIDEO = "video"
    WORDS = "words"


@dataclass(frozen=True)
class ParameterType:
    """Parsed parameter type, including list and fixed-tuple structure."""

    container: ParameterContainer
    items: tuple[ScalarParameterType, ...]

    def __post_init__(self) -> None:
        if self.container in {ParameterContainer.SCALAR, ParameterContainer.LIST}:
            if len(self.items) != 1:
                raise ValueError(f"{self.container} parameter types require exactly one item type")
        elif not self.items:
            raise ValueError("tuple parameter types require at least one item type")

    @classmethod
    def parse(cls, value: str) -> "ParameterType":
        token = str(value).strip()
        try:
            return cls(ParameterContainer.SCALAR, (ScalarParameterType(token),))
        except ValueError:
            pass
        if token.startswith("list[") and token.endswith("]"):
            subtype = token[5:-1].strip()
            try:
                item = ScalarParameterType(subtype)
            except ValueError as exc:
                raise ValueError(f"Unsupported catalogue parameter type '{token}'") from exc
            return cls(ParameterContainer.LIST, (item,))
        if token.startswith("tuple[") and token.endswith("]"):
            parts = [part.strip() for part in token[6:-1].split(",") if part.strip()]
            try:
                items = tuple(ScalarParameterType(part) for part in parts)
            except ValueError as exc:
                raise ValueError(f"Unsupported catalogue parameter type '{token}'") from exc
            return cls(ParameterContainer.TUPLE, items)
        raise ValueError(f"Unsupported catalogue parameter type '{token}'")

    def __str__(self) -> str:
        if self.container is ParameterContainer.SCALAR:
            return str(self.items[0])
        if self.container is ParameterContainer.LIST:
            return f"list[{self.items[0]}]"
        return f"tuple[{','.join(str(item) for item in self.items)}]"


@dataclass(frozen=True, eq=False)
class ParameterSchema(Mapping[str, Any]):
    """Immutable, validated parameter declaration with mapping compatibility."""

    value_type: ParameterType | None = None
    default: Any = _MISSING
    choices: tuple[Any, ...] | None = None
    nullable: bool = False
    minimum: float | int | None = None
    maximum: float | int | None = None
    min_items: int | None = None
    max_items: int | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ParameterSchema":
        allowed = {
            "type",
            "default",
            "choices",
            "nullable",
            "min",
            "max",
            "min_items",
            "max_items",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"Unknown catalogue parameter schema fields: {unknown}")
        choices = value.get("choices")
        if choices is not None and not isinstance(choices, list):
            raise ValueError("catalogue parameter choices must be a list")
        value_type = None
        if "type" in value:
            value_type = ParameterType.parse(str(value["type"]))
        schema = cls(
            value_type=value_type,
            default=value.get("default", _MISSING),
            choices=None if choices is None else tuple(choices),
            nullable=bool(value.get("nullable", False)),
            minimum=value.get("min"),
            maximum=value.get("max"),
            min_items=value.get("min_items"),
            max_items=value.get("max_items"),
        )
        schema._validate_structure()
        return schema

    @property
    def has_default(self) -> bool:
        return self.default is not _MISSING

    def _validate_structure(self) -> None:
        if self.min_items is not None and int(self.min_items) < 0:
            raise ValueError("min_items must be non-negative")
        if self.max_items is not None and int(self.max_items) < 0:
            raise ValueError("max_items must be non-negative")
        if (
            self.min_items is not None
            and self.max_items is not None
            and int(self.min_items) > int(self.max_items)
        ):
            raise ValueError("min_items may not exceed max_items")
        if self.minimum is not None and self.maximum is not None:
            if float(self.minimum) > float(self.maximum):
                raise ValueError("parameter min may not exceed max")
        if self.has_default:
            _validate_param_value("default", self.default, self)

    def __iter__(self) -> Iterator[str]:
        if self.value_type is not None:
            yield "type"
        if self.has_default:
            yield "default"
        if self.choices is not None:
            yield "choices"
        if self.nullable:
            yield "nullable"
        if self.minimum is not None:
            yield "min"
        if self.maximum is not None:
            yield "max"
        if self.min_items is not None:
            yield "min_items"
        if self.max_items is not None:
            yield "max_items"

    def __len__(self) -> int:
        return sum(1 for _key in self)

    def __getitem__(self, key: str) -> Any:
        if key == "type" and self.value_type is not None:
            return str(self.value_type)
        if key == "default" and self.has_default:
            return self.default
        if key == "choices" and self.choices is not None:
            return list(self.choices)
        if key == "nullable" and self.nullable:
            return True
        if key == "min" and self.minimum is not None:
            return self.minimum
        if key == "max" and self.maximum is not None:
            return self.maximum
        if key == "min_items" and self.min_items is not None:
            return self.min_items
        if key == "max_items" and self.max_items is not None:
            return self.max_items
        raise KeyError(key)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Mapping) and dict(self) == dict(other)


@dataclass(frozen=True, eq=False)
class OutputSpec(Mapping[str, Any]):
    """Immutable output schema and semantic routing kind."""

    schema: OutputSchemaId
    kind: OutputKind | None = None
    dims: tuple[str, ...] | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OutputSpec":
        unknown = sorted(set(value) - {"schema", "kind", "dims"})
        if unknown:
            raise ValueError(f"Unknown catalogue output schema fields: {unknown}")
        if "schema" not in value:
            raise ValueError("catalogue output declarations require a schema")
        try:
            schema = OutputSchemaId(str(value["schema"]))
        except ValueError as exc:
            raise ValueError(f"Unsupported catalogue output schema '{value['schema']}'") from exc
        kind_value = value.get("kind")
        try:
            kind = None if kind_value is None else OutputKind(str(kind_value))
        except ValueError as exc:
            raise ValueError(f"Unsupported catalogue output kind '{kind_value}'") from exc
        dims_value = value.get("dims")
        if dims_value is not None:
            if not isinstance(dims_value, list) or not dims_value:
                raise ValueError("catalogue output dims must be a non-empty list")
            if any(not isinstance(dim, str) or not dim.strip() for dim in dims_value):
                raise ValueError("catalogue output dims must contain non-empty strings")
            dims = tuple(dims_value)
        else:
            dims = None
        return cls(schema=schema, kind=kind, dims=dims)

    def __iter__(self) -> Iterator[str]:
        yield "schema"
        if self.kind is not None:
            yield "kind"
        if self.dims is not None:
            yield "dims"

    def __len__(self) -> int:
        return 1 + int(self.kind is not None) + int(self.dims is not None)

    def __getitem__(self, key: str) -> str:
        if key == "schema":
            return str(self.schema)
        if key == "kind" and self.kind is not None:
            return str(self.kind)
        if key == "dims" and self.dims is not None:
            return list(self.dims)
        raise KeyError(key)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Mapping) and dict(self) == dict(other)


@dataclass(frozen=True)
class ExtractorSpec:
    name: str
    impl: str
    version: str = "1.0"
    label: str | None = None
    modalities: tuple[Modality, ...] | list[str] = field(default_factory=tuple)
    requires: list[str] = field(default_factory=list)
    dependency_class: DependencyClass | str | None = None
    cost_class: CostClass | str | None = None
    bundles: list[str] = field(default_factory=list)
    infer_bundles: bool = True
    tags: list[str] = field(default_factory=list)
    params: dict[str, ParameterSchema] | dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, OutputSpec] | dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            modalities = tuple(Modality(str(value)) for value in self.modalities)
        except ValueError as exc:
            raise ValueError(f"Unsupported catalogue modality: {exc}") from exc
        dependency = self.dependency_class
        if dependency is not None and not isinstance(dependency, DependencyClass):
            try:
                dependency = DependencyClass(str(dependency))
            except ValueError as exc:
                raise ValueError(f"Unsupported catalogue dependency class '{dependency}'") from exc
        cost = self.cost_class
        if cost is not None and not isinstance(cost, CostClass):
            try:
                cost = CostClass(str(cost))
            except ValueError as exc:
                raise ValueError(f"Unsupported catalogue cost class '{cost}'") from exc

        typed_params: dict[str, ParameterSchema] = {}
        for name, value in self.params.items():
            if isinstance(value, ParameterSchema):
                typed_params[str(name)] = value
            elif isinstance(value, Mapping):
                typed_params[str(name)] = ParameterSchema.from_mapping(value)
            else:
                raise ValueError(f"Extractor '{self.name}' param '{name}' must be a mapping")
        typed_outputs: dict[str, OutputSpec] = {}
        for name, value in self.outputs.items():
            if isinstance(value, OutputSpec):
                typed_outputs[str(name)] = value
            elif isinstance(value, Mapping):
                typed_outputs[str(name)] = OutputSpec.from_mapping(value)
            else:
                raise ValueError(f"Extractor '{self.name}' output '{name}' must be a mapping")

        object.__setattr__(self, "modalities", modalities)
        object.__setattr__(self, "dependency_class", dependency)
        object.__setattr__(self, "cost_class", cost)
        object.__setattr__(self, "params", typed_params)
        object.__setattr__(self, "outputs", typed_outputs)


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


def _validate_scalar_type(type_name: ScalarParameterType, value: Any) -> bool:
    if type_name is ScalarParameterType.BOOL:
        return _is_bool(value)
    if type_name is ScalarParameterType.INT:
        return isinstance(value, int) and not _is_bool(value)
    if type_name is ScalarParameterType.FLOAT:
        return isinstance(value, (int, float)) and not _is_bool(value)
    if type_name in {ScalarParameterType.STR, ScalarParameterType.MODEL_ID}:
        return isinstance(value, str)
    raise AssertionError(f"unhandled parameter type {type_name}")


def _validate_param_value(name: str, value: Any, spec: ParameterSchema) -> None:
    if value is None:
        if spec.nullable or (spec.has_default and spec.default is None):
            return
        raise ValueError(f"Parameter '{name}' may not be null")

    value_type = spec.value_type
    if value_type is not None:
        if value_type.container is ParameterContainer.LIST:
            if not isinstance(value, list):
                raise ValueError(f"Parameter '{name}' must be a {value_type}")
            for index, item in enumerate(value):
                if not _validate_scalar_type(value_type.items[0], item):
                    raise ValueError(f"Parameter '{name}[{index}]' must be {value_type.items[0]}")
        elif value_type.container is ParameterContainer.TUPLE:
            if not isinstance(value, (list, tuple)) or len(value) != len(value_type.items):
                raise ValueError(f"Parameter '{name}' must be {value_type}")
            for index, item_type in enumerate(value_type.items):
                if not _validate_scalar_type(item_type, value[index]):
                    raise ValueError(f"Parameter '{name}[{index}]' must be {item_type}")
        elif not _validate_scalar_type(value_type.items[0], value):
            raise ValueError(f"Parameter '{name}' must be {value_type.items[0]}")

    if spec.choices is not None and value not in spec.choices:
        raise ValueError(f"Parameter '{name}' must be one of {list(spec.choices)}")
    if isinstance(value, (int, float)) and not _is_bool(value):
        if spec.minimum is not None and float(value) < float(spec.minimum):
            raise ValueError(f"Parameter '{name}' must be >= {spec.minimum}")
        if spec.maximum is not None and float(value) > float(spec.maximum):
            raise ValueError(f"Parameter '{name}' must be <= {spec.maximum}")
    if isinstance(value, list):
        if spec.min_items is not None and len(value) < int(spec.min_items):
            raise ValueError(f"Parameter '{name}' must have at least {spec.min_items} items")
        if spec.max_items is not None and len(value) > int(spec.max_items):
            raise ValueError(f"Parameter '{name}' must have at most {spec.max_items} items")


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
        missing = sorted(required - set(spec_payload))
        if missing:
            raise ValueError(f"Extractor spec missing required fields: {missing}")
        self.register_spec(ExtractorSpec(**spec_payload))

    def load_spec_file(self, path: str | Path) -> None:
        source = Path(path)
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"Catalogue file {source} must contain a mapping")
        extractors = payload.get("extractors", [])
        if not isinstance(extractors, list):
            raise ValueError(f"'extractors' must be a list in {source}")
        for item in extractors:
            if not isinstance(item, dict):
                raise ValueError(f"Extractor entries in {source} must be mappings")
            self.register(item)

    def load_spec_dir(self, path: str | Path) -> None:
        directory = Path(path)
        for pattern in ("*.yaml", "*.yml", "*.json"):
            for file in sorted(directory.glob(pattern)):
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
            self.register_spec(ExtractorSpec(name=ep.name, impl=target, version="plugin"))

    def list(self) -> list[ExtractorSpec]:
        return [self._specs[name] for name in sorted(self._specs)]

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
        values = dict(params or {})
        unknown = sorted(set(values) - set(spec.params))
        if unknown:
            raise ValueError(f"Unknown parameter(s) for extractor '{name}': {unknown}")
        for param_name, value in values.items():
            _validate_param_value(param_name, value, spec.params[param_name])
        return values

    @classmethod
    def with_builtin_specs(cls) -> "Registry":
        registry = cls()
        spec_dir = Path(__file__).resolve().parents[1] / "zoo" / "specs"
        registry.load_spec_dir(spec_dir)
        registry.load_plugins()
        return registry


__all__ = [
    "CostClass",
    "DependencyClass",
    "ExtractorSpec",
    "Modality",
    "OutputKind",
    "OutputSchemaId",
    "OutputSpec",
    "ParameterContainer",
    "ParameterSchema",
    "ParameterType",
    "Registry",
    "ScalarParameterType",
]
