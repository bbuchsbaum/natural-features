"""Recipe parsing and execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from natural_features.core.feature_types import EventSeries, FeatureSeries, TrackSeries
from natural_features.core.registry import Registry


@dataclass
class RecipeExecutionResult:
    steps: dict[str, dict[str, Any]]


@dataclass
class RecipeValidationResult:
    step_ids: list[str]
    outputs_by_step: dict[str, list[str]]


_ALLOWED_STEP_KEYS = {"id", "use", "params", "inputs", "postprocess", "enabled"}


def _validate_recipe_payload(payload: dict[str, Any]) -> None:
    if "features" not in payload:
        raise ValueError("Recipe missing 'features'")
    if not isinstance(payload["features"], list):
        raise ValueError("Recipe field 'features' must be a list")
    seen_ids: set[str] = set()
    for idx, spec in enumerate(payload["features"], start=1):
        if not isinstance(spec, dict):
            raise ValueError(f"Recipe feature entry at index {idx} must be a mapping")
        unknown = sorted(set(spec.keys()) - _ALLOWED_STEP_KEYS)
        if unknown:
            raise ValueError(f"Recipe step at index {idx} has unknown keys: {unknown}")
        if "use" not in spec or not isinstance(spec["use"], str) or not spec["use"].strip():
            raise ValueError(f"Recipe step at index {idx} must define non-empty 'use'")
        step_id = spec.get("id", f"step{idx}")
        if not isinstance(step_id, str) or not step_id.strip():
            raise ValueError(f"Recipe step at index {idx} has invalid 'id'")
        if step_id in seen_ids:
            raise ValueError(f"Duplicate recipe step id: '{step_id}'")
        seen_ids.add(step_id)
        if "params" in spec and not isinstance(spec["params"], dict):
            raise ValueError(f"Recipe step '{step_id}' field 'params' must be a mapping")
        if "inputs" in spec and not isinstance(spec["inputs"], dict):
            raise ValueError(f"Recipe step '{step_id}' field 'inputs' must be a mapping")
        if "postprocess" in spec and not isinstance(spec["postprocess"], dict):
            raise ValueError(f"Recipe step '{step_id}' field 'postprocess' must be a mapping")
        if "enabled" in spec and not isinstance(spec["enabled"], bool):
            raise ValueError(f"Recipe step '{step_id}' field 'enabled' must be bool")


def load_recipe(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    payload = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Recipe must be a mapping")
    _validate_recipe_payload(payload)
    return payload


def _resolve_ref(ref: str, context: dict[str, dict[str, Any]]) -> Any:
    raw = ref.strip()
    if not raw.startswith("ref:"):
        return ref
    step_id, out_key = _parse_ref_token(raw)
    if step_id not in context:
        raise KeyError(f"Unknown recipe ref step: {step_id}")
    outputs = context[step_id]
    if out_key not in outputs:
        raise KeyError(f"Unknown recipe ref output '{out_key}' for step '{step_id}'")
    return outputs[out_key]


def _parse_ref_token(ref: str) -> tuple[str, str]:
    token = ref.split("ref:", 1)[1].strip()
    if "." not in token:
        return token, "default"
    return token.split(".", 1)


def _postprocess(obj: Any, post: dict[str, Any]) -> Any:
    if not isinstance(obj, FeatureSeries):
        return obj
    cur = obj
    rename = post.get("rename", {})
    prefix = rename.get("prefix")
    if prefix:
        names = cur.coords.get("feature", [f"f{i}" for i in range(cur.values.shape[1])])
        names = [f"{prefix}{n}" for n in names]
        cur = FeatureSeries(
            values=cur.values,
            times_s=cur.times_s,
            dims=cur.dims,
            coords={"feature": names},
            metadata=cur.metadata,
            timebase=cur.timebase,
        )
    standardize = post.get("standardize")
    if standardize:
        vals = cur.values.astype(np.float32)
        mu = vals.mean(axis=0, keepdims=True)
        sd = vals.std(axis=0, keepdims=True)
        vals = (vals - mu) / np.maximum(sd, 1e-8)
        cur = FeatureSeries(
            values=vals,
            times_s=cur.times_s,
            dims=cur.dims,
            coords=cur.coords,
            metadata=cur.metadata,
            timebase=cur.timebase,
        )
    reduce = post.get("reduce")
    if reduce and isinstance(reduce, dict) and reduce.get("method") == "pca":
        n_components = int(reduce.get("n_components", min(cur.values.shape)))
        x = cur.values.astype(np.float64)
        x = x - x.mean(axis=0, keepdims=True)
        u, s, _vh = np.linalg.svd(x, full_matrices=False)
        k = max(1, min(n_components, u.shape[1]))
        vals = (u[:, :k] * s[:k]).astype(np.float32)
        cur = FeatureSeries(
            values=vals,
            times_s=cur.times_s,
            dims=("time", "feature"),
            coords={"feature": [f"pc{i}" for i in range(vals.shape[1])]},
            metadata=cur.metadata,
            timebase=cur.timebase,
        )
    return cur


def _validate_output_schema(
    *,
    step_id: str,
    output_key: str,
    schema: str | None,
    value: Any,
) -> None:
    if not schema:
        return
    if schema.startswith("FeatureSeries") and not isinstance(value, FeatureSeries):
        raise TypeError(f"Step '{step_id}' output '{output_key}' must be FeatureSeries for schema '{schema}'")
    if schema.startswith("EventSeries") and not isinstance(value, EventSeries):
        raise TypeError(f"Step '{step_id}' output '{output_key}' must be EventSeries for schema '{schema}'")
    if schema.startswith("TrackSeries") and not isinstance(value, TrackSeries):
        raise TypeError(f"Step '{step_id}' output '{output_key}' must be TrackSeries for schema '{schema}'")


def _validate_output_contracts(
    *,
    step_id: str,
    outputs: dict[str, Any],
    expected_outputs: dict[str, Any],
) -> None:
    if not expected_outputs:
        return
    for out_key, out_spec in expected_outputs.items():
        if out_key not in outputs:
            raise ValueError(f"Step '{step_id}' missing declared output '{out_key}'")
        schema = out_spec.get("schema") if isinstance(out_spec, dict) else None
        _validate_output_schema(step_id=step_id, output_key=out_key, schema=schema, value=outputs[out_key])


def validate_recipe(
    recipe: dict[str, Any],
    *,
    registry: Registry,
    input_keys: set[str] | None = None,
) -> RecipeValidationResult:
    _validate_recipe_payload(recipe)
    input_keys = set(input_keys or set())
    available_outputs: dict[str, set[str]] = {"input": set(input_keys)}
    step_ids: list[str] = []

    for idx, spec in enumerate(recipe.get("features", []), start=1):
        if spec.get("enabled", True) is False:
            continue
        step_id = str(spec.get("id", f"step{idx}"))
        name = str(spec["use"])
        ex_spec = registry.get(name)
        registry.validated_params(name, spec.get("params", {}))
        input_map = spec.get("inputs")
        if input_map:
            for in_key, in_value in input_map.items():
                if isinstance(in_value, str) and in_value.strip().startswith("ref:"):
                    ref_step, ref_out = _parse_ref_token(in_value)
                    if ref_step not in available_outputs:
                        raise KeyError(f"Unknown recipe ref step '{ref_step}' for input '{in_key}' in step '{step_id}'")
                    if ref_out not in available_outputs[ref_step]:
                        raise KeyError(
                            f"Unknown recipe ref output '{ref_out}' from step '{ref_step}' for input '{in_key}' in step '{step_id}'"
                        )
        else:
            if input_keys and ex_spec.modalities:
                modality = ex_spec.modalities[0]
                if modality not in input_keys:
                    raise ValueError(f"Step '{step_id}' requires modality '{modality}', not present in declared inputs")
        declared_outputs = set(ex_spec.outputs.keys()) if ex_spec.outputs else {"default"}
        available_outputs[step_id] = declared_outputs
        step_ids.append(step_id)

    outputs_by_step = {k: sorted(v) for k, v in available_outputs.items() if k != "input"}
    return RecipeValidationResult(step_ids=step_ids, outputs_by_step=outputs_by_step)


def execute_recipe(
    recipe: dict[str, Any],
    *,
    registry: Registry,
    inputs: dict[str, Any],
) -> RecipeExecutionResult:
    _validate_recipe_payload(recipe)
    context: dict[str, dict[str, Any]] = {"input": dict(inputs)}
    for idx, spec in enumerate(recipe.get("features", []), start=1):
        if spec.get("enabled", True) is False:
            continue
        name = spec["use"]
        step_id = spec.get("id", f"step{idx}")
        params = registry.validated_params(name, spec.get("params", {}))
        input_map = spec.get("inputs")
        fn = registry.impl(name)
        ex_spec = registry.get(name)

        if input_map:
            resolved = {k: _resolve_ref(v, context) if isinstance(v, str) else v for k, v in input_map.items()}
            if len(resolved) == 1:
                first_key = next(iter(resolved.keys()))
                result = fn(resolved[first_key], **params)
            else:
                result = fn(**resolved, **params)
        else:
            if not ex_spec.modalities:
                raise ValueError(f"Extractor {name} does not declare modalities and no explicit inputs were provided")
            modality = ex_spec.modalities[0]
            if modality not in inputs:
                raise ValueError(f"Recipe step {step_id} requires modality '{modality}'")
            result = fn(inputs[modality], **params)

        outputs: dict[str, Any]
        if isinstance(result, dict):
            outputs = result
        else:
            outputs = {"default": result}
        _validate_output_contracts(step_id=step_id, outputs=outputs, expected_outputs=ex_spec.outputs)
        post = spec.get("postprocess", {})
        if post:
            outputs = {k: _postprocess(v, post) for k, v in outputs.items()}
        context[step_id] = outputs
    return RecipeExecutionResult(steps=context)
