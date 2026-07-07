"""High-level feature catalogue, planning, and extraction helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Any, Iterable

from natural_features.core.feature_types import EventSeries, FeatureSeries, TrackSeries
from natural_features.core.interchange import merge_feature_tables
from natural_features.core.recipe import execute_recipe
from natural_features.core.registry import ExtractorSpec, Registry
from natural_features.core.stimulus import (
    AudioStimulus,
    ImageStimulus,
    MultiModalStimulus,
    TextStimulus,
    VideoStimulus,
)
from natural_features.core.timeline import FeatureAlignment, Timeline, align_feature_to_timeline
from natural_features.workflows._public_contract import public_feature_ids

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
_VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
_KNOWN_MODALITIES = {
    "audio",
    "video",
    "image",
    "text",
    "words",
    "events",
    "features",
    "tokens",
    "phonemes",
    "movie",
}
_BASE_DEPENDENCY_CLASSES = {"base", "base_python", "python", "stdlib"}
_EXPENSIVE_REQUIRES = {
    "transformers",
    "torch",
    "mediapipe",
    "opensmile",
    "pymoten",
    "speech_asr",
    "whisperx",
    "ocr",
    "ffmpeg",
}
_SOURCE_ALIASES = {
    "tokens": ["tokens", "words", "events"],
    "words": ["words", "tokens", "events"],
    "events": ["events", "words", "phonemes"],
    "phonemes": ["phonemes", "events", "tokens"],
}
_PREFERRED_FEATURE_IDS = {
    "audio.lowlevel.rms": "audio.rms",
    "audio.lowlevel.mel": "audio.mel",
    "audio.lowlevel.mfcc": "audio.mfcc",
    "audio.lowlevel.spectral_stats": "audio.spectral_stats",
    "audio.opensmile.egemaps_lld": "audio.egemaps",
    "vision.dynamics.frame_diffs": "vision.frame_diffs",
    "vision.motion.optical_flow_mag": "vision.motion",
    "vision.motion.motion_energy": "vision.motion_energy",
    "affect.visual.social_proxies": "vision.social_proxies",
    "speech.asr.whisper": "speech.words",
    "speech.articulatory.features": "speech.articulatory",
    "speech.phonology.ctc_posteriors": "speech.ctc",
    "speech.ssl.wavlm": "speech.wavlm",
    "language.embed.bert_words": "language.bert",
    "language.predict.surprisal": "language.surprisal",
}
_PUBLIC_FEATURE_IDS = public_feature_ids()


@dataclass(frozen=True)
class FeatureCatalogEntry:
    feature_id: str
    label: str
    modalities: list[str]
    requires: list[str]
    dependency_class: str
    cost_class: str
    output_schema: str
    bundles: list[str]
    tags: list[str]
    default_params: dict[str, Any]
    requires_opt_in: bool
    is_public: bool


@dataclass(frozen=True)
class FeaturePlanRow:
    feature_id: str
    step_id: str
    use: str
    input_key: str
    input_token: str
    params: dict[str, Any]
    output_schema: str
    requires_opt_in: bool


@dataclass
class FeaturePlan:
    rows: list[FeaturePlanRow]
    input_modalities: list[str]

    def to_recipe(self) -> dict[str, Any]:
        return {
            "features": [
                {
                    "id": row.step_id,
                    "use": row.use,
                    "inputs": {row.input_key: row.input_token},
                    "params": dict(row.params),
                }
                for row in self.rows
            ]
        }

    def to_dataframe(self) -> Any:
        pd = _require_pandas()
        return pd.DataFrame([asdict(row) for row in self.rows])


@dataclass
class AlignedFeatureSet:
    features: dict[str, Any]
    target: Timeline
    alignments: dict[str, FeatureAlignment]
    policy: str = "overlap"

    def to_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for alignment in self.alignments.values():
            rows.extend(alignment.to_rows())
        return rows

    def to_dataframe(self) -> Any:
        pd = _require_pandas()
        return pd.DataFrame(self.to_rows())

    def annotated_features(self, *, prefix: str | None = None) -> dict[str, Any]:
        out = dict(self.features)
        for name, alignment in self.alignments.items():
            if isinstance(alignment.source, EventSeries):
                out[name] = alignment.annotated_events(prefix=prefix)
        return out


@dataclass
class ExtractFeaturesResult:
    features: dict[str, Any]
    plan: FeaturePlan
    recipe: dict[str, Any]
    steps: dict[str, dict[str, Any]]
    table: Any | None = None
    inputs: dict[str, Any] = field(default_factory=dict)
    timelines: dict[str, Timeline] = field(default_factory=dict)

    def to_table(
        self,
        *,
        format: str = "long",
        include_objects: bool = True,
        include_metadata: bool = True,
    ) -> Any:
        return merge_feature_tables(
            self.features,
            format=format,
            include_objects=include_objects,
            include_metadata=include_metadata,
        )

    def timeline(self, target: str | Timeline) -> Timeline:
        return _resolve_timeline(self, target)

    def align_to(
        self,
        target: str | Timeline,
        *,
        features: str | Iterable[str] | None = None,
        policy: str = "overlap",
    ) -> AlignedFeatureSet:
        target_timeline = self.timeline(target)
        names = _as_list(features) if features is not None else _temporal_feature_names(self.features)
        alignments: dict[str, FeatureAlignment] = {}
        for name in names:
            if name not in self.features:
                raise KeyError(f"Unknown feature output: {name}")
            obj = self.features[name]
            if not _is_temporal_output(obj):
                raise TypeError(f"Feature output '{name}' is not a temporal feature object")
            alignments[name] = align_feature_to_timeline(name, obj, target_timeline, policy=policy)
        return AlignedFeatureSet(
            features=self.features,
            target=target_timeline,
            alignments=alignments,
            policy=policy,
        )


def _require_pandas() -> Any:
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "pandas is required for dataframe output: "
            "pip install natural-features[storage]"
        ) from exc
    return pd


def _as_list(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _title_from_id(feature_id: str) -> str:
    return feature_id.replace(".", " ").replace("_", " ").title()


def _param_defaults(spec: ExtractorSpec) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for name, param_spec in (spec.params or {}).items():
        if isinstance(param_spec, dict) and "default" in param_spec:
            defaults[name] = param_spec["default"]
    return defaults


def _output_schema(spec: ExtractorSpec) -> str:
    schemas = []
    for out_spec in (spec.outputs or {}).values():
        if isinstance(out_spec, dict) and out_spec.get("schema"):
            schemas.append(str(out_spec["schema"]))
    return ",".join(schemas)


def _infer_dependency_class(spec: ExtractorSpec) -> str:
    if spec.dependency_class:
        return str(spec.dependency_class)
    requires = {str(x) for x in spec.requires}
    if requires & _EXPENSIVE_REQUIRES or "fallback" in spec.tags:
        return "optional_python"
    return "base_python"


def _infer_cost_class(spec: ExtractorSpec) -> str:
    if spec.cost_class:
        return str(spec.cost_class)
    requires = {str(x) for x in spec.requires}
    tags = {str(x) for x in spec.tags}
    name = spec.name
    if requires & {"transformers", "torch"} or any(
        token in name for token in ("clip", "dino", "bert", "whisper", "ssl")
    ):
        return "expensive"
    if requires & {"mediapipe", "opensmile", "pymoten", "ffmpeg", "ocr"}:
        return "moderate"
    if tags & {"embedding", "asr"}:
        return "expensive"
    return "cheap"


def _infer_bundles(spec: ExtractorSpec) -> list[str]:
    if spec.bundles:
        return list(spec.bundles)
    if not spec.infer_bundles:
        return []
    bundles: list[str] = []
    tags = set(spec.tags)
    if "baseline" in tags:
        bundles.append("baseline")
    if spec.name in {"audio.mel", "vision.energy", "vision.motion", "vision.clip"}:
        bundles.append("movie_encoding")
    if spec.name.startswith("speech."):
        bundles.append("speech_encoding")
    if spec.name.startswith("language."):
        bundles.append("language_encoding")
    if spec.name in {"vision.face", "vision.clip"}:
        bundles.append("social_vision")
    return bundles


def _requires_opt_in(spec: ExtractorSpec, dependency_class: str, cost_class: str) -> bool:
    tags = set(spec.tags)
    requires = {str(x) for x in spec.requires}
    if "placeholder" in tags:
        return True
    if dependency_class not in _BASE_DEPENDENCY_CLASSES:
        return True
    return cost_class == "expensive" or bool(requires & _EXPENSIVE_REQUIRES)


def _entry_from_spec(spec: ExtractorSpec) -> FeatureCatalogEntry:
    dependency_class = _infer_dependency_class(spec)
    cost_class = _infer_cost_class(spec)
    return FeatureCatalogEntry(
        feature_id=spec.name,
        label=spec.label or _title_from_id(spec.name),
        modalities=list(spec.modalities),
        requires=list(spec.requires),
        dependency_class=dependency_class,
        cost_class=cost_class,
        output_schema=_output_schema(spec),
        bundles=_infer_bundles(spec),
        tags=list(spec.tags),
        default_params=_param_defaults(spec),
        requires_opt_in=_requires_opt_in(spec, dependency_class, cost_class),
        is_public=spec.name in _PUBLIC_FEATURE_IDS,
    )


def feature_catalog(*, registry: Registry | None = None) -> list[FeatureCatalogEntry]:
    registry = registry or Registry.with_builtin_specs()
    return [_entry_from_spec(spec) for spec in registry.list()]


def _allowed_by_budget(entry: FeatureCatalogEntry, budget: str) -> bool:
    token = budget.lower().strip()
    if token in {"all", "unsafe"}:
        return True
    if token in {"allow_python", "allow-python", "python"}:
        return "placeholder" not in set(entry.tags)
    if token in {"local", "default", "base"}:
        return not entry.requires_opt_in
    raise ValueError("budget must be one of: default, local, allow_python, all")


def available_features(
    *,
    modality: str | Iterable[str] | None = None,
    budget: str = "default",
    bundle: str | None = None,
    tags: str | Iterable[str] | None = None,
    include_placeholders: bool = False,
    public_only: bool = True,
    as_dataframe: bool = False,
    registry: Registry | None = None,
) -> list[FeatureCatalogEntry] | Any:
    modalities = set(_as_list(modality))
    if "movie" in modalities:
        modalities.update({"audio", "video"})
    tags_filter = set(_as_list(tags))
    out = []
    for entry in feature_catalog(registry=registry):
        entry_tags = set(entry.tags)
        if not include_placeholders and "placeholder" in entry_tags:
            continue
        if public_only and not entry.is_public:
            continue
        if modalities and not (set(entry.modalities) & modalities):
            continue
        if bundle is not None and bundle not in set(entry.bundles):
            continue
        if tags_filter and not tags_filter.issubset(entry_tags):
            continue
        if not _allowed_by_budget(entry, budget):
            continue
        out.append(entry)
    if as_dataframe:
        pd = _require_pandas()
        return pd.DataFrame([asdict(entry) for entry in out])
    return out


def _input_modalities(stimulus: Any) -> set[str]:
    if stimulus is None:
        return set()
    if isinstance(stimulus, str) and stimulus in _KNOWN_MODALITIES:
        if stimulus == "movie":
            return {"audio", "video"}
        return {stimulus}
    if isinstance(stimulus, AudioStimulus):
        return {"audio"}
    if isinstance(stimulus, VideoStimulus):
        return {"video"}
    if isinstance(stimulus, ImageStimulus):
        return {"image"}
    if isinstance(stimulus, TextStimulus):
        return {"text"}
    if isinstance(stimulus, MultiModalStimulus):
        out = set()
        if stimulus.audio is not None:
            out.add("audio")
        if stimulus.video is not None:
            out.add("video")
        if stimulus.image is not None:
            out.add("image")
        if stimulus.text is not None:
            out.add("text")
        return out
    if isinstance(stimulus, dict):
        return {str(key) for key in stimulus.keys()}
    if isinstance(stimulus, (str, Path)):
        p = Path(stimulus)
        if p.exists():
            suffix = p.suffix.lower()
            if suffix == ".wav":
                return {"audio"}
            if suffix == ".npy":
                return {"video"}
            if suffix in _VIDEO_SUFFIXES:
                return {"video"}
            if suffix in _IMAGE_SUFFIXES:
                return {"image"}
            if suffix in {".txt", ".text"}:
                return {"text"}
        return {"text"}
    return set()


def _coerce_path(path: str | Path, *, video_fps: float) -> Any:
    p = Path(path)
    if p.exists():
        suffix = p.suffix.lower()
        if suffix == ".wav":
            return AudioStimulus.from_wav(p)
        if suffix == ".npy":
            return VideoStimulus.from_npy(p, fps=video_fps)
        if suffix in _VIDEO_SUFFIXES:
            return p
        if suffix in _IMAGE_SUFFIXES:
            return ImageStimulus.from_file(p)
        if suffix in {".txt", ".text"}:
            return TextStimulus(p.read_text(encoding="utf-8"))
    return TextStimulus(str(path))


def _coerce_inputs(stimulus: Any, *, video_fps: float) -> dict[str, Any]:
    if isinstance(stimulus, dict):
        return {
            str(key): _coerce_path(value, video_fps=video_fps)
            if isinstance(value, (str, Path))
            else value
            for key, value in stimulus.items()
        }
    if isinstance(stimulus, MultiModalStimulus):
        out: dict[str, Any] = {}
        if stimulus.audio is not None:
            out["audio"] = stimulus.audio
        if stimulus.video is not None:
            out["video"] = stimulus.video
        if stimulus.image is not None:
            out["image"] = stimulus.image
        if stimulus.text is not None:
            out["text"] = stimulus.text
        return out
    if isinstance(stimulus, AudioStimulus):
        return {"audio": stimulus}
    if isinstance(stimulus, VideoStimulus):
        return {"video": stimulus}
    if isinstance(stimulus, ImageStimulus):
        return {"image": stimulus}
    if isinstance(stimulus, TextStimulus):
        return {"text": stimulus}
    if isinstance(stimulus, (str, Path)):
        coerced = _coerce_path(stimulus, video_fps=video_fps)
        modality = next(iter(_input_modalities(coerced)))
        return {modality: coerced}
    raise TypeError("Unsupported stimulus input")


def _is_temporal_output(value: Any) -> bool:
    return isinstance(value, (FeatureSeries, EventSeries, TrackSeries))


def _temporal_feature_names(features: dict[str, Any]) -> list[str]:
    return [name for name, value in features.items() if _is_temporal_output(value)]


def _default_timelines_from_inputs(inputs: dict[str, Any]) -> dict[str, Timeline]:
    timelines: dict[str, Timeline] = {}
    video = inputs.get("video")
    if isinstance(video, VideoStimulus):
        timelines["video_frames"] = Timeline.from_video_stimulus(video)
    return timelines


def _resolve_timeline(result: ExtractFeaturesResult, target: str | Timeline) -> Timeline:
    if isinstance(target, Timeline):
        return target
    target_name = str(target)
    if target_name in result.timelines:
        return result.timelines[target_name]
    if target_name in result.features:
        obj = result.features[target_name]
        if not _is_temporal_output(obj):
            raise TypeError(f"Feature output '{target_name}' is not temporal and cannot define a timeline")
        return Timeline.from_feature(target_name, obj)
    feature_prefix = "feature:"
    if target_name.startswith(feature_prefix):
        feature_name = target_name[len(feature_prefix) :]
        if feature_name not in result.features:
            raise KeyError(f"Unknown feature output: {feature_name}")
        obj = result.features[feature_name]
        if not _is_temporal_output(obj):
            raise TypeError(f"Feature output '{feature_name}' is not temporal and cannot define a timeline")
        return Timeline.from_feature(feature_name, obj)
    known = sorted([*result.timelines.keys(), *result.features.keys()])
    raise KeyError(f"Unknown timeline target: {target_name}. Known targets: {', '.join(known)}")


def _sanitize_step_id(feature_id: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]+", "_", feature_id).strip("_")
    return token or "feature"


def _unique_step_id(feature_id: str, used: set[str]) -> str:
    base = _sanitize_step_id(feature_id)
    step_id = base
    i = 2
    while step_id in used:
        step_id = f"{base}_{i}"
        i += 1
    used.add(step_id)
    return step_id


def _source_candidates(modality: str) -> list[str]:
    return _SOURCE_ALIASES.get(modality, [modality])


def _find_source(
    modalities: list[str],
    available: dict[str, str],
) -> tuple[str, str] | None:
    for modality in modalities:
        for candidate in _source_candidates(modality):
            if candidate in available:
                return modality, available[candidate]
    return None


def _register_outputs(
    available: dict[str, str],
    *,
    step_id: str,
    outputs: dict[str, Any],
) -> None:
    for out_key, out_spec in outputs.items():
        token = f"ref:{step_id}.{out_key}"
        kind = out_spec.get("kind") if isinstance(out_spec, dict) else None
        schema = out_spec.get("schema") if isinstance(out_spec, dict) else None
        if kind:
            available[str(kind)] = token
        if out_key != "default":
            available[str(out_key)] = token
        if isinstance(schema, str) and schema.startswith("EventSeries"):
            available.setdefault("events", token)
            if kind == "words" or out_key == "words":
                available["words"] = token
            if kind == "phonemes" or out_key == "phonemes":
                available["phonemes"] = token
                available["tokens"] = token
        if isinstance(schema, str) and schema.startswith("FeatureSeries"):
            available["features"] = token
        if kind in {"audio", "video", "image", "text"}:
            available[str(kind)] = token


def _requested_features(
    features: str | Iterable[str] | None,
    *,
    bundle: str | None,
    input_modalities: set[str],
    budget: str,
    include_placeholders: bool,
    registry: Registry,
) -> list[str]:
    if features is not None:
        return _as_list(features)
    entries = available_features(
        modality=input_modalities or None,
        budget=budget,
        bundle=bundle or "baseline",
        include_placeholders=include_placeholders,
        registry=registry,
    )
    entry_ids = {entry.feature_id for entry in entries}
    out: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        feature_id = _PREFERRED_FEATURE_IDS.get(entry.feature_id, entry.feature_id)
        if feature_id in seen:
            continue
        if feature_id not in entry_ids:
            continue
        seen.add(feature_id)
        out.append(feature_id)
    return out


def plan_features(
    stimulus: Any = None,
    *,
    features: str | Iterable[str] | None = None,
    bundle: str | None = None,
    budget: str = "default",
    feature_params: dict[str, dict[str, Any]] | None = None,
    include_placeholders: bool = False,
    as_dataframe: bool = False,
    registry: Registry | None = None,
) -> FeaturePlan | Any:
    registry = registry or Registry.with_builtin_specs()
    input_modalities = _input_modalities(stimulus)
    requested = _requested_features(
        features,
        bundle=bundle,
        input_modalities=input_modalities,
        budget=budget,
        include_placeholders=include_placeholders,
        registry=registry,
    )
    catalog = {entry.feature_id: entry for entry in feature_catalog(registry=registry)}
    feature_params = feature_params or {}
    available_sources = {key: f"input:{key}" for key in input_modalities}
    rows: list[FeaturePlanRow] = []
    used_ids: set[str] = set()

    for feature_id in requested:
        if feature_id not in catalog:
            raise KeyError(f"Unknown feature: {feature_id}")
        entry = catalog[feature_id]
        if not include_placeholders and "placeholder" in set(entry.tags):
            raise ValueError(f"Feature '{feature_id}' is a placeholder and cannot be planned by default")
        if not _allowed_by_budget(entry, budget):
            raise PermissionError(
                f"Feature '{feature_id}' requires opt-in. "
                "Use budget='allow_python' or budget='all'."
            )
        spec = registry.get(feature_id)
        source = _find_source(spec.modalities, available_sources)
        if source is None:
            have = ", ".join(sorted(available_sources)) or "none"
            need = ", ".join(spec.modalities) or "unknown"
            raise ValueError(
                f"Cannot route feature '{feature_id}': requires {need}; available {have}"
            )
        input_key, input_token = source
        params = dict(entry.default_params)
        params.update(feature_params.get(feature_id, {}))
        params.update(feature_params.get(spec.name, {}))
        step_id = _unique_step_id(feature_id, used_ids)
        rows.append(
            FeaturePlanRow(
                feature_id=feature_id,
                step_id=step_id,
                use=spec.name,
                input_key=input_key,
                input_token=input_token,
                params=params,
                output_schema=entry.output_schema,
                requires_opt_in=entry.requires_opt_in,
            )
        )
        _register_outputs(available_sources, step_id=step_id, outputs=spec.outputs or {})

    plan = FeaturePlan(rows=rows, input_modalities=sorted(input_modalities))
    return plan.to_dataframe() if as_dataframe else plan


def _collect_features(plan: FeaturePlan, steps: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for row in plan.rows:
        outputs = steps.get(row.step_id, {})
        if len(outputs) == 1:
            out[row.feature_id] = next(iter(outputs.values()))
        elif "default" in outputs:
            out[row.feature_id] = outputs["default"]
        else:
            out[row.feature_id] = outputs
            for key, value in outputs.items():
                out[f"{row.feature_id}.{key}"] = value
    return out


def extract_features(
    stimulus: Any,
    *,
    features: str | Iterable[str] | None = None,
    bundle: str | None = None,
    budget: str = "default",
    feature_params: dict[str, dict[str, Any]] | None = None,
    format: str = "features",
    table_format: str = "long",
    include_objects: bool = True,
    include_placeholders: bool = False,
    video_fps: float = 10.0,
    registry: Registry | None = None,
) -> ExtractFeaturesResult | FeaturePlan | dict[str, Any] | Any:
    registry = registry or Registry.with_builtin_specs()
    inputs = _coerce_inputs(stimulus, video_fps=video_fps)
    plan = plan_features(
        inputs,
        features=features,
        bundle=bundle,
        budget=budget,
        feature_params=feature_params,
        include_placeholders=include_placeholders,
        registry=registry,
    )
    assert isinstance(plan, FeaturePlan)
    recipe = plan.to_recipe()
    fmt = format.lower().strip()
    if fmt == "recipe":
        return recipe
    if fmt == "plan":
        return plan
    result = execute_recipe(recipe, registry=registry, inputs=inputs)
    features_out = _collect_features(plan, result.steps)
    if fmt in {"table", "dataframe"}:
        return merge_feature_tables(
            features_out,
            format=table_format,
            include_objects=include_objects,
        )
    if fmt not in {"features", "result"}:
        raise ValueError("format must be one of: features, recipe, plan, table")
    return ExtractFeaturesResult(
        features=features_out,
        plan=plan,
        recipe=recipe,
        steps=result.steps,
        inputs=inputs,
        timelines=_default_timelines_from_inputs(inputs),
    )
