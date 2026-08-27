from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from natural_features.core.registry import (
    CostClass,
    DependencyClass,
    Modality,
    OutputKind,
    OutputSchemaId,
    OutputSpec,
    ParameterContainer,
    ParameterSchema,
    Registry,
    ScalarParameterType,
)


def test_builtin_catalogue_is_typed_at_registration() -> None:
    spec = Registry.with_builtin_specs().get("audio.clap")

    assert spec.modalities == (Modality.AUDIO,)
    assert spec.dependency_class is DependencyClass.OPTIONAL_PYTHON
    assert spec.cost_class is CostClass.EXPENSIVE
    assert isinstance(spec.params["model"], ParameterSchema)
    assert spec.params["model"].value_type is not None
    assert spec.params["model"].value_type.container is ParameterContainer.SCALAR
    assert spec.params["model"].value_type.items == (ScalarParameterType.STR,)
    assert isinstance(spec.outputs["default"], OutputSpec)
    assert spec.outputs["default"].schema is OutputSchemaId.FEATURE_SERIES
    assert spec.outputs["default"].kind is OutputKind.FEATURES


def test_typed_schemas_remain_read_compatible_and_immutable() -> None:
    spec = Registry.with_builtin_specs().get("speech.words")

    assert spec.outputs["qc"] == {"schema": "dict", "kind": "qc"}
    assert spec.params["execution_mode"]["choices"] == ["strict"]
    with pytest.raises(FrozenInstanceError):
        spec.outputs["qc"].kind = OutputKind.FEATURES  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("modalities", ["soundish"], "modality"),
        ("dependency_class", "sometimes", "dependency class"),
        ("cost_class", "huge", "cost class"),
        ("outputs", {"default": {"schema": "Array/v9"}}, "output schema"),
    ],
)
def test_invalid_catalogue_domains_fail_during_registration(
    field: str,
    value: object,
    message: str,
) -> None:
    payload = {
        "name": f"test.invalid_{field}",
        "impl": "natural_features.features.preprocess:text_tokenize",
        field: value,
    }

    with pytest.raises(ValueError, match=message):
        Registry().register(payload)
