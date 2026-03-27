"""pymoten-compatible motion-energy wrapper."""

from __future__ import annotations

import numpy as np

from natural_features.core.execution import add_execution_provenance, resolve_execution_mode
from natural_features.core.feature_types import FeatureSeries
from natural_features.core.stimulus import VideoStimulus
from natural_features.core.timebase import TimebaseSpec
from natural_features.features.common import extractor_metadata
from natural_features.features.vision.motion import optical_flow_mag
from natural_features.features.vision.lowlevel import _to_gray


def motion_energy_pymoten(
    stimulus: VideoStimulus,
    *,
    fps_downsample: float | None = None,
    roi: tuple[int, int, int, int] | None = None,
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
) -> FeatureSeries:
    mode, strict_dependency = resolve_execution_mode(
        execution_mode=execution_mode,
        strict_dependency=strict_dependency,
    )
    frames = stimulus.frames
    times = stimulus.frame_times_s
    fps = stimulus.fps
    if fps_downsample is not None and fps_downsample > 0 and fps_downsample < stimulus.fps:
        step = max(1, int(round(stimulus.fps / fps_downsample)))
        frames = frames[::step]
        times = times[::step]
        fps = stimulus.fps / step
    if roi is not None:
        y0, y1, x0, x1 = roi
        frames = frames[:, y0:y1, x0:x1, ...]
    if frames.shape[0] == 0:
        raise ValueError("No frames available after fps_downsample/roi filtering")

    try:
        import moten  # type: ignore
    except ImportError:
        if strict_dependency:
            raise RuntimeError(
                "pymoten/moten is not installed. Install optional dependency and retry."
            )
        proxy = optical_flow_mag(VideoStimulus.from_array(frames, fps=fps, start_offset_s=float(times[0])))
        md = dict(proxy.metadata)
        md["backend"] = "proxy"
        md["backend_reason"] = "moten unavailable"
        md["execution_mode"] = mode
        md["fallback_used"] = True
        md["fallback_reason"] = "moten unavailable"
        return FeatureSeries(
            values=proxy.values,
            times_s=proxy.times_s,
            dims=proxy.dims,
            coords=proxy.coords,
            metadata=md,
            timebase=proxy.timebase,
        )

    # This branch intentionally keeps API assumptions minimal across moten versions.
    gray = _to_gray(frames.astype(np.float32))
    try:
        pyramid = moten.pyramids.MotionEnergyPyramid(
            stimulus_vhsize=gray.shape[1:3],
            stimulus_fps=fps,
        )
        values = pyramid.project_stimulus(gray).astype(np.float32)
    except Exception:
        if strict_dependency:
            raise
        proxy = optical_flow_mag(VideoStimulus.from_array(frames, fps=fps, start_offset_s=float(times[0])))
        md = dict(proxy.metadata)
        md["backend"] = "proxy"
        md["backend_reason"] = "moten projection failed"
        md["execution_mode"] = mode
        md["fallback_used"] = True
        md["fallback_reason"] = "moten projection failed"
        return FeatureSeries(
            values=proxy.values,
            times_s=proxy.times_s,
            dims=proxy.dims,
            coords=proxy.coords,
            metadata=md,
            timebase=proxy.timebase,
        )

    metadata = add_execution_provenance(
        extractor_metadata(
            "vision.motion.motion_energy",
            params={"fps_downsample": fps_downsample, "roi": roi},
            extra={"backend": "pymoten"},
        ),
        execution_mode=mode,
        fallback_used=False,
    )
    return FeatureSeries(
        values=values,
        times_s=times,
        dims=("time", "feature"),
        coords={"feature": [f"motion_energy_{i}" for i in range(values.shape[1])]},
        metadata=metadata,
        timebase=TimebaseSpec(kind="frames", sampling_rate_hz=fps),
    )
