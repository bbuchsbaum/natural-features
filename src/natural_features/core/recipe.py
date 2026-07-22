"""Recipe parsing and execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from natural_features.core.feature_bundle import inherit_temporal_contract
from natural_features.core.feature_types import EventSeries, FeatureSeries, TrackSeries
from natural_features.core.registry import Registry


@dataclass
class RecipeExecutionResult:
    steps: dict[str, dict[str, Any]]


@dataclass
class RecipeValidationResult:
    step_ids: list[str]
    outputs_by_step: dict[str, list[str]]


@dataclass
class RecipeDag:
    nodes: list[dict[str, str]]
    edges: list[dict[str, str]]
    recipe: dict[str, Any]


_ALLOWED_STEP_KEYS = {"id", "use", "params", "inputs", "outputs", "depends_on", "postprocess", "enabled"}
_PREPROCESS_IDS = {
    "video.frames.sample",
    "video.trim",
    "video.audio.extract",
    "audio.trim",
    "audio.resample",
    "text.tokenize",
    "image.ocr",
    "video.ocr",
    "events.align",
    "features.resample",
    "features.hrf",
    "features.lag",
}


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
        if "outputs" in spec and not isinstance(spec["outputs"], dict):
            raise ValueError(f"Recipe step '{step_id}' field 'outputs' must be a mapping")
        if "depends_on" in spec and not isinstance(spec["depends_on"], (str, list, tuple)):
            raise ValueError(f"Recipe step '{step_id}' field 'depends_on' must be a string or list")
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


def _step_id(spec: dict[str, Any], idx: int) -> str:
    return str(spec.get("id", f"step{idx}"))


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(x) for x in value]


def _step_output_spec(spec: dict[str, Any], ex_spec: Any) -> dict[str, Any]:
    outputs = spec.get("outputs")
    if outputs:
        return dict(outputs)
    if ex_spec.outputs:
        return dict(ex_spec.outputs)
    return {"default": {"schema": ""}}


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


def _resolve_recipe_input(value: Any, context: dict[str, dict[str, Any]]) -> Any:
    if not isinstance(value, str):
        return value
    raw = value.strip()
    if raw.startswith("ref:"):
        return _resolve_ref(raw, context)
    if raw.startswith("input:"):
        key = raw.split("input:", 1)[1].strip()
        if key not in context["input"]:
            raise KeyError(f"Unknown recipe input key: {key}")
        return context["input"][key]
    return value


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
            time_bounds_s=cur.time_bounds_s,
            temporal_context=cur.temporal_context,
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
            time_bounds_s=cur.time_bounds_s,
            temporal_context=cur.temporal_context,
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
            time_bounds_s=cur.time_bounds_s,
            temporal_context=cur.temporal_context,
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
    all_step_ids = [
        _step_id(spec, idx)
        for idx, spec in enumerate(recipe.get("features", []), start=1)
        if spec.get("enabled", True) is not False
    ]
    edges: list[tuple[str, str]] = []

    for idx, spec in enumerate(recipe.get("features", []), start=1):
        if spec.get("enabled", True) is False:
            continue
        step_id = _step_id(spec, idx)
        name = str(spec["use"])
        ex_spec = registry.get(name)
        registry.validated_params(name, spec.get("params", {}))
        input_map = spec.get("inputs")
        if input_map:
            for in_key, in_value in input_map.items():
                if isinstance(in_value, str) and in_value.strip().startswith("input:"):
                    input_name = in_value.split("input:", 1)[1].strip()
                    if input_keys and input_name not in input_keys:
                        raise KeyError(f"Unknown recipe input key '{input_name}' for input '{in_key}' in step '{step_id}'")
                elif isinstance(in_value, str) and in_value.strip().startswith("ref:"):
                    ref_step, ref_out = _parse_ref_token(in_value)
                    if ref_step not in available_outputs:
                        raise KeyError(f"Unknown recipe ref step '{ref_step}' for input '{in_key}' in step '{step_id}'")
                    if ref_out not in available_outputs[ref_step]:
                        raise KeyError(
                            f"Unknown recipe ref output '{ref_out}' from step '{ref_step}' for input '{in_key}' in step '{step_id}'"
                        )
                    edges.append((ref_step, step_id))
        else:
            if input_keys and ex_spec.modalities:
                modality = ex_spec.modalities[0]
                if modality not in input_keys:
                    raise ValueError(f"Step '{step_id}' requires modality '{modality}', not present in declared inputs")
        for dep in _as_str_list(spec.get("depends_on")):
            if dep not in all_step_ids:
                raise KeyError(f"Unknown recipe dependency step '{dep}' for step '{step_id}'")
            edges.append((dep, step_id))
        declared_outputs = set(_step_output_spec(spec, ex_spec).keys())
        available_outputs[step_id] = declared_outputs
        step_ids.append(step_id)

    _check_cycles(step_ids, edges)
    outputs_by_step = {k: sorted(v) for k, v in available_outputs.items() if k != "input"}
    return RecipeValidationResult(step_ids=step_ids, outputs_by_step=outputs_by_step)


def execute_recipe(
    recipe: dict[str, Any],
    *,
    registry: Registry,
    inputs: dict[str, Any],
) -> RecipeExecutionResult:
    _validate_recipe_payload(recipe)
    validate_recipe(recipe, registry=registry, input_keys=set(inputs.keys()))
    context: dict[str, dict[str, Any]] = {"input": dict(inputs)}
    for idx, spec in _execution_steps(recipe):
        name = spec["use"]
        step_id = _step_id(spec, idx)
        params = registry.validated_params(name, spec.get("params", {}))
        input_map = spec.get("inputs")
        fn = registry.impl(name)
        ex_spec = registry.get(name)

        if input_map:
            resolved = {k: _resolve_recipe_input(v, context) for k, v in input_map.items()}
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
        temporal_inputs = resolved.values() if input_map else [inputs[modality]]
        outputs = {
            key: inherit_temporal_contract(value, temporal_inputs)
            for key, value in outputs.items()
        }
        _validate_output_contracts(step_id=step_id, outputs=outputs, expected_outputs=_step_output_spec(spec, ex_spec))
        post = spec.get("postprocess", {})
        if post:
            outputs = {k: _postprocess(v, post) for k, v in outputs.items()}
        context[step_id] = outputs
    return RecipeExecutionResult(steps=context)


def _check_cycles(step_ids: list[str], edges: list[tuple[str, str]]) -> None:
    children: dict[str, list[str]] = {step_id: [] for step_id in step_ids}
    for src, dst in edges:
        if src in children and dst in children:
            children[src].append(dst)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str, stack: list[str]) -> None:
        if node in visiting:
            raise ValueError(f"Recipe DAG contains a cycle: {' -> '.join([*stack, node])}")
        if node in visited:
            return
        visiting.add(node)
        for child in children[node]:
            visit(child, [*stack, node])
        visiting.remove(node)
        visited.add(node)

    for step_id in step_ids:
        visit(step_id, [])


def _execution_steps(recipe: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    enabled = [
        (idx, spec, _step_id(spec, idx))
        for idx, spec in enumerate(recipe.get("features", []), start=1)
        if spec.get("enabled", True) is not False
    ]
    original_index = {step_id: pos for pos, (_idx, _spec, step_id) in enumerate(enabled)}
    by_id = {step_id: (idx, spec) for idx, spec, step_id in enabled}
    deps: dict[str, set[str]] = {step_id: set() for _idx, _spec, step_id in enabled}

    for _idx, spec, step_id in enabled:
        for value in (spec.get("inputs") or {}).values():
            if isinstance(value, str) and value.strip().startswith("ref:"):
                ref_step, _ref_out = _parse_ref_token(value)
                deps[step_id].add(ref_step)
        deps[step_id].update(_as_str_list(spec.get("depends_on")))

    ordered: list[tuple[int, dict[str, Any]]] = []
    emitted: set[str] = set()
    remaining = set(deps)
    while remaining:
        ready = sorted(
            [step_id for step_id in remaining if deps[step_id].issubset(emitted)],
            key=lambda step_id: original_index[step_id],
        )
        if not ready:
            cycle = " -> ".join(sorted(remaining))
            raise ValueError(f"Recipe DAG contains a cycle or unsatisfied dependency: {cycle}")
        for step_id in ready:
            ordered.append(by_id[step_id])
            emitted.add(step_id)
            remaining.remove(step_id)
    return ordered


def _node_type(feature_id: str) -> str:
    return "preprocess" if feature_id in _PREPROCESS_IDS else "extract"


def plan_dag(
    recipe: dict[str, Any],
    *,
    registry: Registry,
    input_keys: set[str] | None = None,
    include_merge: bool = True,
) -> RecipeDag:
    _validate_recipe_payload(recipe)
    input_keys = set(input_keys or set())
    nodes: list[dict[str, str]] = []
    edges: list[dict[str, str]] = []
    step_ids: list[str] = []
    outgoing: set[str] = set()

    for key in sorted(input_keys):
        nodes.append({"id": f"input:{key}", "type": "input", "label": key, "use": "", "output_schema": ""})

    for idx, spec in enumerate(recipe.get("features", []), start=1):
        if spec.get("enabled", True) is False:
            continue
        step_id = _step_id(spec, idx)
        feature_id = str(spec["use"])
        ex_spec = registry.get(feature_id)
        outputs = _step_output_spec(spec, ex_spec)
        schemas = [str(v.get("schema", "")) for v in outputs.values() if isinstance(v, dict) and v.get("schema")]
        nodes.append(
            {
                "id": step_id,
                "type": _node_type(feature_id),
                "label": feature_id,
                "use": feature_id,
                "output_schema": ",".join(schemas),
            }
        )
        step_ids.append(step_id)
        input_map = spec.get("inputs") or {}
        for input_name, value in input_map.items():
            if not isinstance(value, str):
                continue
            raw = value.strip()
            if raw.startswith("input:"):
                key = raw.split("input:", 1)[1].strip()
                input_id = f"input:{key}"
                if not any(n["id"] == input_id for n in nodes):
                    nodes.insert(0, {"id": input_id, "type": "input", "label": key, "use": "", "output_schema": ""})
                edges.append({"from": input_id, "to": step_id, "output": key, "input": str(input_name), "kind": "input"})
            elif raw.startswith("ref:"):
                ref_step, ref_out = _parse_ref_token(raw)
                edges.append({"from": ref_step, "to": step_id, "output": ref_out, "input": str(input_name), "kind": "ref"})
                outgoing.add(ref_step)
        for dep in _as_str_list(spec.get("depends_on")):
            edges.append({"from": dep, "to": step_id, "output": "", "input": "", "kind": "depends_on"})
            outgoing.add(dep)
        if spec.get("postprocess"):
            post_id = f"{step_id}__postprocess"
            nodes.append({"id": post_id, "type": "postprocess", "label": f"{step_id} postprocess", "use": "", "output_schema": ",".join(schemas)})
            edges.append({"from": step_id, "to": post_id, "output": "default", "input": "default", "kind": "postprocess"})
            outgoing.add(step_id)
            outgoing.discard(post_id)

    validate_recipe(recipe, registry=registry, input_keys=input_keys)
    if include_merge and step_ids:
        nodes.append({"id": "merge", "type": "merge", "label": "merge", "use": "", "output_schema": "table"})
        leaves = [step_id for step_id in step_ids if step_id not in outgoing]
        if not leaves:
            leaves = [step_ids[-1]]
        for step_id in leaves:
            post_id = f"{step_id}__postprocess"
            src = post_id if any(n["id"] == post_id for n in nodes) else step_id
            edges.append({"from": src, "to": "merge", "output": "default", "input": "table", "kind": "merge"})

    unique_nodes: list[dict[str, str]] = []
    seen: set[str] = set()
    for node in nodes:
        if node["id"] in seen:
            continue
        seen.add(node["id"])
        unique_nodes.append(node)
    return RecipeDag(nodes=unique_nodes, edges=edges, recipe=recipe)


def _mermaid_id(raw: str) -> str:
    out = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in raw)
    return out if out and (out[0].isalpha() or out[0] == "_") else f"n_{out}"


def as_mermaid(dag_or_recipe: RecipeDag | dict[str, Any], *, registry: Registry | None = None, direction: str = "TD") -> str:
    if isinstance(dag_or_recipe, RecipeDag):
        dag = dag_or_recipe
    else:
        if registry is None:
            registry = Registry.with_builtin_specs()
        dag = plan_dag(dag_or_recipe, registry=registry)
    lines = [f"flowchart {direction}"]
    for node in dag.nodes:
        label = f"{node['label']}\\n{node['type']}".replace('"', "'")
        lines.append(f"  {_mermaid_id(node['id'])}[\"{label}\"]")
    for edge in dag.edges:
        label = " -> ".join(x for x in [edge.get("output", ""), edge.get("input", "")] if x)
        if label:
            lines.append(f"  {_mermaid_id(edge['from'])} -->|{label.replace(chr(34), chr(39))}| {_mermaid_id(edge['to'])}")
        else:
            lines.append(f"  {_mermaid_id(edge['from'])} --> {_mermaid_id(edge['to'])}")
    return "\n".join(lines)
