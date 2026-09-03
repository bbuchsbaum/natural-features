"""Semantic view event extractors."""

from __future__ import annotations

from typing import Any

import numpy as np

from natural_features.core.backend_errors import (
    BackendDependencyError,
    BackendInferenceError,
    BackendLoadError,
)
from natural_features.core.execution import add_execution_provenance, resolve_execution_mode
from natural_features.core.feature_types import EventSeries
from natural_features.core.stimulus import VideoStimulus
from natural_features.core.timebase import SupportSpec, TimebaseSpec
from natural_features.features.common import extractor_metadata
from natural_features.features.vision.common import frame_duration_s, frame_times_s
from natural_features.features.vision.neural import _batch_iter, _frames_to_pil_images, _to_device


_DEFAULT_SEMANTIC_LABELS = [
    "indoor_scene",
    "outdoor_scene",
    "person_or_face",
    "text_or_graphics",
    "object_closeup",
    "landscape",
]


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    logits = logits - np.nanmax(logits, axis=1, keepdims=True)
    exp = np.exp(logits).astype(np.float32)
    denom = np.maximum(exp.sum(axis=1, keepdims=True, dtype=np.float32), 1e-8)
    return exp / denom


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _prompt_labels(labels: list[str]) -> list[str]:
    return [str(label).replace("_", " ") for label in labels]


def _clip_semantic_views(
    stimulus: VideoStimulus,
    *,
    model: str,
    labels: list[str],
    stride_frames: int,
    batch_size: int,
    local_files_only: bool,
    execution_mode: str,
    params: dict[str, object],
) -> EventSeries:
    try:
        import torch  # type: ignore
        from transformers import CLIPModel, CLIPProcessor  # type: ignore
    except ImportError as exc:
        raise BackendDependencyError("CLIP semantic views", "transformers and torch are required") from exc

    try:
        processor = CLIPProcessor.from_pretrained(model, local_files_only=local_files_only)
        net = CLIPModel.from_pretrained(model, local_files_only=local_files_only)
        device = "cuda" if getattr(torch, "cuda", None) is not None and torch.cuda.is_available() else "cpu"
        to_method = getattr(net, "to", None)
        if callable(to_method):
            net = to_method(device)
        eval_method = getattr(net, "eval", None)
        if callable(eval_method):
            eval_method()
    except Exception as exc:
        raise BackendLoadError("CLIP semantic views", f"model '{model}' could not be loaded") from exc
    prompts = [f"a video frame showing {label}" for label in _prompt_labels(labels)]
    images = _frames_to_pil_images(stimulus, stride_frames=stride_frames)
    try:
        prob_chunks = []
        for batch in _batch_iter(images, batch_size):
            inputs = processor(text=prompts, images=batch, return_tensors="pt", padding=True)
            inputs = _to_device(inputs, device)
            with torch.no_grad():
                out = net(**inputs)
            logits = getattr(out, "logits_per_image", None)
            if logits is None:
                logits = out[0]
            prob_chunks.append(_softmax_rows(_as_numpy(logits)))
        probs = np.concatenate(prob_chunks, axis=0).astype(np.float32)
    except Exception as exc:
        raise BackendInferenceError("CLIP semantic views", "zero-shot classification failed") from exc
    label_idx = np.argmax(probs, axis=1).astype(np.int64)
    out_labels = np.asarray([labels[int(i)] for i in label_idx], dtype=object)
    confidence = probs[np.arange(probs.shape[0]), label_idx].astype(np.float32)
    stride = max(1, int(stride_frames))
    onset = frame_times_s(stimulus)[::stride]
    dur = frame_duration_s(stimulus, stride_frames=stride) or 0.0
    md = add_execution_provenance(
        extractor_metadata(
            "vision.semantic_views",
            params=params,
            extra={"backend": "transformers_clip_zero_shot", "candidate_labels": labels},
        ),
        execution_mode=execution_mode,
        fallback_used=False,
    )
    return EventSeries(
        onset_s=onset,
        offset_s=onset + float(dur),
        label=out_labels,
        confidence=confidence,
        extra={
            "view_type": out_labels.copy(),
            "label_index": label_idx,
        },
        metadata=md,
        timebase=TimebaseSpec(
            kind="frame_events",
            reference=stimulus.clock,
            sampling_rate_hz=stimulus.fps / stride,
            support=SupportSpec(kind="interval", anchor="onset"),
        ),
        temporal_context=stimulus.temporal_context,
    )


def vision_semantic_views(
    stimulus: VideoStimulus,
    *,
    model: str = "openai/clip-vit-base-patch32",
    labels: list[str] | None = None,
    stride_frames: int = 1,
    batch_size: int = 32,
    local_files_only: bool = True,
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
) -> EventSeries:
    """Return frame-aligned semantic view labels."""

    mode, _strict = resolve_execution_mode(execution_mode=execution_mode, strict_dependency=strict_dependency)
    candidate_labels = list(labels or _DEFAULT_SEMANTIC_LABELS)
    if not candidate_labels:
        raise ValueError("labels must contain at least one label")
    stride = max(1, int(stride_frames))
    params: dict[str, object] = {
        "model": model,
        "labels": candidate_labels,
        "stride_frames": stride,
        "batch_size": batch_size,
        "local_files_only": local_files_only,
    }
    return _clip_semantic_views(
        stimulus,
        model=model,
        labels=candidate_labels,
        stride_frames=stride,
        batch_size=batch_size,
        local_files_only=local_files_only,
        execution_mode=mode,
        params=params,
    )
