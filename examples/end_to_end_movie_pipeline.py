#!/usr/bin/env python3
"""End-to-end example: movie/audio features -> run-aware TR matrix export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from natural_features import build_experiment_grid, query_feature_zoo_window_tr
from natural_features.core.feature_types import FeatureSeries
from natural_features.core.recipe import execute_recipe
from natural_features.core.registry import Registry
from natural_features.core.stimulus import AudioStimulus, VideoStimulus
from natural_features.fmri.design import concat_feature_series
from natural_features.util.io import atomic_numpy_savez, atomic_write_json


def _default_recipe() -> dict[str, Any]:
    return {
        "features": [
            {"id": "vision_energy", "use": "vision.lowlevel.visual_energy", "params": {"fps": 10, "include_deltas": True}},
            {"id": "audio_rms", "use": "audio.lowlevel.rms", "params": {"hop_s": 0.01, "win_s": 0.03}},
            {"id": "audio_mfcc", "use": "audio.lowlevel.mfcc", "params": {"n_mfcc": 13, "n_mels": 40, "include_deltas": True}},
            {"id": "speech_vad", "use": "speech.vad.energy_vad", "params": {"hop_s": 0.02, "threshold": 0.5}},
        ]
    }


def _flatten_recipe_outputs(steps: dict[str, dict[str, Any]]) -> dict[str, FeatureSeries]:
    out: dict[str, FeatureSeries] = {}
    for step_id, outputs in steps.items():
        for out_key, obj in outputs.items():
            if not isinstance(obj, FeatureSeries):
                continue
            name = f"{step_id}.{out_key}"
            out[name] = obj
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-npy", required=True, help="Prepared RGB frames .npy")
    ap.add_argument("--video-fps", type=float, default=10.0, help="Frame rate for video_npy")
    ap.add_argument("--audio-wav", required=True, help="Prepared mono wav")
    ap.add_argument("--tr-s", type=float, required=True, help="Target TR in seconds")
    ap.add_argument("--run-index", type=int, default=1, help="Run index label")
    ap.add_argument("--run-start-s", type=float, default=0.0, help="Run start (scan) time")
    ap.add_argument(
        "--feature-t0-s",
        type=float,
        default=0.0,
        help="Feature timeline t=0 mapped onto scan time (stimulus onset offset)",
    )
    ap.add_argument("--out-prefix", default="example_design", help="Output file prefix")
    args = ap.parse_args()

    video = VideoStimulus.from_npy(args.video_npy, fps=float(args.video_fps))
    audio = AudioStimulus.from_wav(args.audio_wav)
    duration_s = float(max(len(video.frames) / video.fps, audio.samples.shape[0] / audio.sr_hz))
    n_trs = int(np.floor(duration_s / float(args.tr_s)))
    if n_trs <= 0:
        raise ValueError("Computed n_trs <= 0. Check TR and stimulus duration.")

    reg = Registry.with_builtin_specs()
    result = execute_recipe(
        _default_recipe(),
        registry=reg,
        inputs={"video": video, "audio": audio},
    )
    zoo = _flatten_recipe_outputs(result.steps)

    grid = build_experiment_grid(
        tr_s=float(args.tr_s),
        n_trs_by_run=[n_trs],
        run_starts_s=[float(args.run_start_s)],
        feature_t0_s=float(args.feature_t0_s),
    )
    tr_zoo = query_feature_zoo_window_tr(
        zoo,
        grid,
        run_index=int(args.run_index),
        t_start_s=0.0,
        t_end_s=n_trs * float(args.tr_s),
        relative_to_run=True,
        method="mean",
        output_time="run_relative",
    )
    matrix = concat_feature_series(list(tr_zoo.values()), standardize=True, add_intercept=True)
    feature_names = [str(x) for x in matrix.coords.get("feature", [])]

    out_prefix = Path(args.out_prefix)
    npz_path = out_prefix.with_suffix(".npz")
    meta_path = out_prefix.with_suffix(".json")
    atomic_numpy_savez(
        npz_path,
        X=matrix.values.astype(np.float32),
        times_s=matrix.times_s.astype(np.float64),
        feature_names=np.asarray(feature_names, dtype=object),
    )
    atomic_write_json(
        meta_path,
        {
            "video_npy": args.video_npy,
            "audio_wav": args.audio_wav,
            "tr_s": float(args.tr_s),
            "n_trs": n_trs,
            "run_start_s": float(args.run_start_s),
            "feature_t0_s": float(args.feature_t0_s),
            "spaces": sorted(tr_zoo.keys()),
            "npz_path": str(npz_path),
        },
        sort_keys=True,
        indent=2,
    )
    print(f"wrote design matrix: {npz_path}")
    print(f"wrote metadata: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
