"""Face-detection style visual features."""

from __future__ import annotations

import numpy as np

from natural_features.core.execution import add_execution_provenance, resolve_execution_mode
from natural_features.core.feature_types import FeatureSeries
from natural_features.core.timebase import TimebaseSpec
from natural_features.features.common import extractor_metadata
from natural_features.features.vision.common import VisualStimulus, ensure_frames, frame_sampling_rate_hz, frame_times_s
from natural_features.features.vision.lowlevel import _saturation, _to_gray

_FEATURE_NAMES = ["face_presence", "face_count", "face_area_frac", "face_center_x", "face_center_y"]


def _fallback_face_proxy(stimulus: VisualStimulus, *, execution_mode: str, reason: str) -> FeatureSeries:
    frames = ensure_frames(stimulus).astype(np.float32)
    sat = _saturation(frames)
    gray = _to_gray(frames)
    contrast = gray.reshape(gray.shape[0], -1).std(axis=1)
    presence = np.clip((sat * 1.5) + (contrast > 0.05).astype(np.float32) * 0.1, 0.0, 1.0)
    count = (presence > 0.25).astype(np.float32)
    area = np.clip(sat, 0.0, 1.0).astype(np.float32)
    center_x = np.full_like(presence, 0.5, dtype=np.float32)
    center_y = np.full_like(presence, 0.5, dtype=np.float32)
    vals = np.column_stack([presence, count, area, center_x, center_y]).astype(np.float32)
    md = add_execution_provenance(
        extractor_metadata(
            "vision.face",
            params={},
            extra={"backend": "fallback_proxy"},
        ),
        execution_mode=execution_mode,
        fallback_used=True,
        fallback_reason=reason,
    )
    return FeatureSeries(
        values=vals,
        times_s=frame_times_s(stimulus),
        dims=("time", "feature"),
        coords={"feature": list(_FEATURE_NAMES)},
        metadata=md,
        timebase=TimebaseSpec(kind="frames", sampling_rate_hz=frame_sampling_rate_hz(stimulus)),
    )


def _as_uint8_rgb(frames: np.ndarray) -> np.ndarray:
    arr = frames.astype(np.float32)
    if arr.size and np.nanmax(arr) <= 1.0:
        arr = arr * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3:
        return np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] == 1:
        return np.repeat(arr[..., :1], 3, axis=-1)
    return arr[..., :3]


def _float_attr(obj: object, name: str, default: float = 0.0) -> float:
    try:
        return float(getattr(obj, name))
    except Exception:
        return float(default)


def _mediapipe_face_detection(
    stimulus: VisualStimulus,
    *,
    execution_mode: str,
    min_detection_confidence: float,
) -> FeatureSeries:
    import mediapipe as mp  # type: ignore

    frames = _as_uint8_rgb(ensure_frames(stimulus))
    face_det = mp.solutions.face_detection
    detector = face_det.FaceDetection(
        model_selection=0,
        min_detection_confidence=float(min_detection_confidence),
    )
    try:
        vals = np.zeros((frames.shape[0], len(_FEATURE_NAMES)), dtype=np.float32)
        vals[:, 3] = 0.5
        vals[:, 4] = 0.5
        for i, frame in enumerate(frames):
            result = detector.process(frame)
            detections = getattr(result, "detections", None) or []
            if not detections:
                continue
            count = float(len(detections))
            area_sum = 0.0
            cx_sum = 0.0
            cy_sum = 0.0
            for det in detections:
                location = getattr(det, "location_data", None)
                bbox = getattr(location, "relative_bounding_box", None)
                if bbox is None:
                    continue
                width = max(0.0, _float_attr(bbox, "width"))
                height = max(0.0, _float_attr(bbox, "height"))
                xmin = _float_attr(bbox, "xmin")
                ymin = _float_attr(bbox, "ymin")
                area_sum += width * height
                cx_sum += xmin + (width / 2.0)
                cy_sum += ymin + (height / 2.0)
            vals[i, 0] = 1.0
            vals[i, 1] = count
            vals[i, 2] = min(area_sum, 1.0)
            vals[i, 3] = cx_sum / count
            vals[i, 4] = cy_sum / count
    finally:
        close = getattr(detector, "close", None)
        if callable(close):
            close()
    md = add_execution_provenance(
        extractor_metadata(
            "vision.face",
            params={"min_detection_confidence": min_detection_confidence},
            extra={"backend": "mediapipe"},
        ),
        execution_mode=execution_mode,
        fallback_used=False,
    )
    return FeatureSeries(
        values=vals,
        times_s=frame_times_s(stimulus),
        dims=("time", "feature"),
        coords={"feature": list(_FEATURE_NAMES)},
        metadata=md,
        timebase=TimebaseSpec(kind="frames", sampling_rate_hz=frame_sampling_rate_hz(stimulus)),
    )


def face_detection(
    stimulus: VisualStimulus,
    *,
    min_detection_confidence: float = 0.5,
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
) -> FeatureSeries:
    mode, strict = resolve_execution_mode(execution_mode=execution_mode, strict_dependency=strict_dependency)
    try:
        return _mediapipe_face_detection(
            stimulus,
            execution_mode=mode,
            min_detection_confidence=float(min_detection_confidence),
        )
    except Exception as exc:
        if strict:
            raise RuntimeError("mediapipe face detection failed in strict mode.") from exc
        return _fallback_face_proxy(stimulus, execution_mode=mode, reason=str(exc))
