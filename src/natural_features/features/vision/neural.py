"""Vision neural embedding extractors with deterministic fallbacks."""

from __future__ import annotations

import numpy as np

from natural_features.core.execution import add_execution_provenance, resolve_execution_mode
from natural_features.core.feature_types import FeatureSeries
from natural_features.core.timebase import TimebaseSpec
from natural_features.features.common import extractor_metadata
from natural_features.features.vision.common import VisualStimulus, ensure_frames, frame_sampling_rate_hz, frame_times_s
from natural_features.features.vision.lowlevel import _edge_energy, _saturation, _to_gray
from natural_features.util.hashing import stable_hash


def _fit_dim(values: np.ndarray, dim: int) -> np.ndarray:
    target = int(dim)
    if target <= 0:
        raise ValueError("dim must be > 0")
    if values.shape[1] == target:
        return values.astype(np.float32)
    if values.shape[1] > target:
        return values[:, :target].astype(np.float32)
    pad = np.zeros((values.shape[0], target - values.shape[1]), dtype=np.float32)
    return np.concatenate([values.astype(np.float32), pad], axis=1)


def _frames_to_pil_images(stimulus: VisualStimulus, *, stride_frames: int) -> list[object]:
    from PIL import Image  # type: ignore

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


def _frame_descriptors(stimulus: VisualStimulus) -> np.ndarray:
    frames = ensure_frames(stimulus).astype(np.float32)
    gray = _to_gray(frames)
    flat = gray.reshape(gray.shape[0], -1)
    q = np.quantile(flat, [0.1, 0.5, 0.9], axis=1).T
    desc = np.column_stack(
        [
            flat.mean(axis=1),
            flat.std(axis=1),
            q,
            _saturation(frames),
            _edge_energy(gray),
        ]
    )
    return desc.astype(np.float32)


def _fallback_embedding(
    stimulus: VisualStimulus,
    *,
    extractor_name: str,
    dim: int,
    stride_frames: int,
    execution_mode: str,
    reason: str,
    params: dict[str, object],
) -> FeatureSeries:
    stride = max(1, int(stride_frames))
    desc = _frame_descriptors(stimulus)[::stride]
    seed = int(stable_hash({"extractor": extractor_name, "dim": dim}, length=8), 16) % (2**32)
    rng = np.random.default_rng(seed)
    proj = rng.normal(0.0, 1.0 / np.sqrt(desc.shape[1]), size=(desc.shape[1], dim)).astype(np.float32)
    vals = np.tanh(desc @ proj).astype(np.float32)
    md = add_execution_provenance(
        extractor_metadata(extractor_name, params=params, extra={"backend": "fallback_projection"}),
        execution_mode=execution_mode,
        fallback_used=True,
        fallback_reason=reason,
    )
    return FeatureSeries(
        values=vals,
        times_s=frame_times_s(stimulus)[::stride],
        dims=("time", "feature"),
        coords={"feature": [f"dim_{i}" for i in range(dim)]},
        metadata=md,
        timebase=TimebaseSpec(kind="frames", sampling_rate_hz=frame_sampling_rate_hz(stimulus, stride_frames=stride)),
    )


def vision_clip_embeddings(
    stimulus: VisualStimulus,
    *,
    model: str = "openai/clip-vit-base-patch32",
    stride_frames: int = 1,
    dim: int = 64,
    batch_size: int = 32,
    local_files_only: bool = True,
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
) -> FeatureSeries:
    mode, strict = resolve_execution_mode(execution_mode=execution_mode, strict_dependency=strict_dependency)
    try:
        import torch  # type: ignore
        from transformers import CLIPModel, CLIPProcessor  # type: ignore
    except Exception as exc:
        if strict:
            raise RuntimeError("transformers+torch are required for strict CLIP extraction.") from exc
        return _fallback_embedding(
            stimulus,
            extractor_name="vision.clip",
            dim=dim,
            stride_frames=stride_frames,
            execution_mode=mode,
            reason="transformers/torch unavailable",
            params={"model": model, "stride_frames": stride_frames, "dim": dim, "local_files_only": local_files_only},
        )
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
        images = _frames_to_pil_images(stimulus, stride_frames=stride_frames)
        chunks = []
        for batch in _batch_iter(images, batch_size):
            inputs = processor(images=batch, return_tensors="pt")
            inputs = _to_device(inputs, device)
            with torch.no_grad():
                features = net.get_image_features(**inputs)
            chunks.append(features.detach().cpu().numpy().astype(np.float32))
        vals = _fit_dim(np.concatenate(chunks, axis=0), dim)
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
    except Exception as exc:
        if strict:
            raise RuntimeError("CLIP extraction failed in strict mode.") from exc
        return _fallback_embedding(
            stimulus,
            extractor_name="vision.clip",
            dim=dim,
            stride_frames=stride_frames,
            execution_mode=mode,
            reason=f"CLIP backend failed: {type(exc).__name__}",
            params={"model": model, "stride_frames": stride_frames, "dim": dim, "local_files_only": local_files_only},
        )


def vision_dino_embeddings(
    stimulus: VisualStimulus,
    *,
    model: str = "facebook/dinov2-base",
    stride_frames: int = 1,
    layers: list[int] | None = None,
    pooling: str = "cls",
    dim: int = 64,
    batch_size: int = 32,
    local_files_only: bool = True,
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
) -> FeatureSeries:
    mode, strict = resolve_execution_mode(execution_mode=execution_mode, strict_dependency=strict_dependency)
    layers = layers or [6, 12]
    if pooling not in {"cls", "mean"}:
        raise ValueError("pooling must be 'cls' or 'mean'")
    try:
        import torch  # type: ignore
        from transformers import AutoImageProcessor, AutoModel  # type: ignore
    except Exception as exc:
        if strict:
            raise RuntimeError("transformers+torch are required for strict DINO extraction.") from exc
        return _fallback_embedding(
            stimulus,
            extractor_name="vision.dino",
            dim=dim * len(layers),
            stride_frames=stride_frames,
            execution_mode=mode,
            reason="transformers/torch unavailable",
            params={"model": model, "stride_frames": stride_frames, "layers": layers, "dim": dim, "local_files_only": local_files_only},
        )
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
        images = _frames_to_pil_images(stimulus, stride_frames=stride_frames)
        chunks = []
        for batch in _batch_iter(images, batch_size):
            inputs = processor(images=batch, return_tensors="pt")
            inputs = _to_device(inputs, device)
            with torch.no_grad():
                outputs = net(**inputs, output_hidden_states=True)
            hidden_states = getattr(outputs, "hidden_states")
            layer_chunks = []
            for layer in layers:
                layer_idx = max(0, min(int(layer), len(hidden_states) - 1))
                arr = hidden_states[layer_idx].detach().cpu().numpy().astype(np.float32)
                if pooling == "cls":
                    emb = arr[:, 0, :]
                else:
                    emb = arr[:, 1:, :].mean(axis=1) if arr.shape[1] > 1 else arr.mean(axis=1)
                layer_chunks.append(_fit_dim(emb, dim))
            chunks.append(np.concatenate(layer_chunks, axis=1))
        vals = np.concatenate(chunks, axis=0).astype(np.float32)
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
    except Exception as exc:
        if strict:
            raise RuntimeError("DINO extraction failed in strict mode.") from exc
        return _fallback_embedding(
            stimulus,
            extractor_name="vision.dino",
            dim=dim * len(layers),
            stride_frames=stride_frames,
            execution_mode=mode,
            reason=f"DINO backend failed: {type(exc).__name__}",
            params={"model": model, "stride_frames": stride_frames, "layers": layers, "dim": dim, "local_files_only": local_files_only},
        )
