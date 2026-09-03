"""Vision neural embedding extractors."""

from __future__ import annotations

import numpy as np

from natural_features.core.backend_errors import (
    BackendDependencyError,
    BackendInferenceError,
    BackendLoadError,
)
from natural_features.core.execution import add_execution_provenance, resolve_execution_mode
from natural_features.core.feature_types import FeatureSeries
from natural_features.core.timebase import TimebaseSpec
from natural_features.features.common import extractor_metadata
from natural_features.features.vision.common import VisualStimulus, ensure_frames, frame_sampling_rate_hz, frame_times_s


def _require_native_dim(values: np.ndarray, dim: int | None) -> np.ndarray:
    """Preserve model geometry and optionally assert its native width."""

    if dim is not None:
        expected = int(dim)
        if expected <= 0:
            raise ValueError("dim must be > 0")
        if values.shape[1] != expected:
            raise ValueError(
                f"Requested dim={expected}, but the model returned its native "
                f"dimension {values.shape[1]}; dimensionality reduction must be "
                "an explicit downstream transform"
            )
    return values.astype(np.float32)


def _frames_to_pil_images(stimulus: VisualStimulus, *, stride_frames: int) -> list[object]:
    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:
        raise BackendDependencyError("vision neural models", "Pillow is required") from exc

    frames = ensure_frames(stimulus)[:: max(1, int(stride_frames))].astype(np.float32)
    if frames.size and np.nanmax(frames) <= 1.0:
        frames = frames * 255.0
    frames = np.clip(frames, 0, 255).astype(np.uint8)
    images = []
    for frame in frames:
        if frame.ndim == 2:
            images.append(Image.fromarray(frame).convert("RGB"))
        elif frame.shape[-1] == 1:
            images.append(Image.fromarray(frame[..., 0]).convert("RGB"))
        else:
            images.append(Image.fromarray(frame[..., :3]).convert("RGB"))
    return images


def _batch_iter(items: list[object], batch_size: int) -> list[list[object]]:
    size = max(1, int(batch_size))
    return [items[i : i + size] for i in range(0, len(items), size)]


def _to_device(inputs: object, device: object) -> object:
    to_method = getattr(inputs, "to", None)
    if callable(to_method):
        return to_method(device)
    if isinstance(inputs, dict):
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    return inputs


def vision_clip_embeddings(
    stimulus: VisualStimulus,
    *,
    model: str = "openai/clip-vit-base-patch32",
    stride_frames: int = 1,
    dim: int | None = None,
    batch_size: int = 32,
    local_files_only: bool = True,
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
) -> FeatureSeries:
    mode, _strict = resolve_execution_mode(execution_mode=execution_mode, strict_dependency=strict_dependency)
    try:
        import torch  # type: ignore
        from transformers import CLIPModel, CLIPProcessor  # type: ignore
    except ImportError as exc:
        raise BackendDependencyError("CLIP", "transformers+torch are required") from exc
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
        raise BackendLoadError("CLIP", f"model '{model}' could not be loaded") from exc
    images = _frames_to_pil_images(stimulus, stride_frames=stride_frames)
    try:
        chunks = []
        for batch in _batch_iter(images, batch_size):
            inputs = processor(images=batch, return_tensors="pt")
            inputs = _to_device(inputs, device)
            with torch.no_grad():
                features = net.get_image_features(**inputs)
            chunks.append(features.detach().cpu().numpy().astype(np.float32))
        raw_values = np.concatenate(chunks, axis=0)
    except Exception as exc:
        raise BackendInferenceError("CLIP", "image projection failed") from exc
    vals = _require_native_dim(raw_values, dim)
    md = add_execution_provenance(
        extractor_metadata(
            "vision.clip",
            params={
                "model": model,
                "stride_frames": stride_frames,
                "dim": dim,
                "batch_size": batch_size,
                "local_files_only": local_files_only,
            },
            extra={"backend": "transformers_clip"},
        ),
        execution_mode=mode,
        fallback_used=False,
    )
    stride = max(1, int(stride_frames))
    return FeatureSeries(
        values=vals,
        times_s=frame_times_s(stimulus)[::stride],
        dims=("time", "feature"),
        coords={"feature": [f"dim_{i}" for i in range(vals.shape[1])]},
        metadata=md,
        timebase=TimebaseSpec(kind="frames", sampling_rate_hz=frame_sampling_rate_hz(stimulus, stride_frames=stride)),
    )


def vision_dino_embeddings(
    stimulus: VisualStimulus,
    *,
    model: str = "facebook/dinov2-base",
    stride_frames: int = 1,
    layers: list[int] | None = None,
    pooling: str = "cls",
    dim: int | None = None,
    batch_size: int = 32,
    local_files_only: bool = True,
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
) -> FeatureSeries:
    mode, _strict = resolve_execution_mode(execution_mode=execution_mode, strict_dependency=strict_dependency)
    layers = layers or [6, 12]
    if pooling not in {"cls", "mean"}:
        raise ValueError("pooling must be 'cls' or 'mean'")
    try:
        import torch  # type: ignore
        from transformers import AutoImageProcessor, AutoModel  # type: ignore
    except ImportError as exc:
        raise BackendDependencyError("DINO", "transformers+torch are required") from exc
    try:
        processor = AutoImageProcessor.from_pretrained(model, local_files_only=local_files_only)
        net = AutoModel.from_pretrained(model, local_files_only=local_files_only)
        device = "cuda" if getattr(torch, "cuda", None) is not None and torch.cuda.is_available() else "cpu"
        to_method = getattr(net, "to", None)
        if callable(to_method):
            net = to_method(device)
        eval_method = getattr(net, "eval", None)
        if callable(eval_method):
            eval_method()
    except Exception as exc:
        raise BackendLoadError("DINO", f"model '{model}' could not be loaded") from exc
    images = _frames_to_pil_images(stimulus, stride_frames=stride_frames)
    try:
        chunks = []
        for batch in _batch_iter(images, batch_size):
            inputs = processor(images=batch, return_tensors="pt")
            inputs = _to_device(inputs, device)
            with torch.no_grad():
                outputs = net(**inputs, output_hidden_states=True)
            hidden_states = getattr(outputs, "hidden_states")
            layer_chunks = []
            for layer in layers:
                layer_idx = int(layer)
                if layer_idx < 0 or layer_idx >= len(hidden_states):
                    raise ValueError(
                        f"Requested DINO layer {layer_idx} is outside "
                        f"[0, {len(hidden_states) - 1}]"
                    )
                arr = hidden_states[layer_idx].detach().cpu().numpy().astype(np.float32)
                if pooling == "cls":
                    emb = arr[:, 0, :]
                else:
                    emb = arr[:, 1:, :].mean(axis=1) if arr.shape[1] > 1 else arr.mean(axis=1)
                layer_chunks.append(_require_native_dim(emb, dim))
            chunks.append(np.concatenate(layer_chunks, axis=1))
        vals = np.concatenate(chunks, axis=0).astype(np.float32)
    except ValueError:
        raise
    except Exception as exc:
        raise BackendInferenceError("DINO", "hidden-state extraction failed") from exc
    md = add_execution_provenance(
        extractor_metadata(
            "vision.dino",
            params={
                "model": model,
                "stride_frames": stride_frames,
                "layers": layers,
                "pooling": pooling,
                "dim": dim,
                "batch_size": batch_size,
                "local_files_only": local_files_only,
            },
            extra={"backend": "transformers_dino"},
        ),
        execution_mode=mode,
        fallback_used=False,
    )
    stride = max(1, int(stride_frames))
    return FeatureSeries(
        values=vals,
        times_s=frame_times_s(stimulus)[::stride],
        dims=("time", "feature"),
        coords={"feature": [f"dim_{i}" for i in range(vals.shape[1])]},
        metadata=md,
        timebase=TimebaseSpec(kind="frames", sampling_rate_hz=frame_sampling_rate_hz(stimulus, stride_frames=stride)),
    )
