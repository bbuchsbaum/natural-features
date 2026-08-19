#!/usr/bin/env python3
"""Extract RMS energy and sample it onto a scan TR grid.

The default timing matches a common single-run layout: TR = 2 s, and audio
t = 0 occurs 0.67 s after the scan begins.

    python examples/rms_energy_to_tr_grid.py
    python examples/rms_energy_to_tr_grid.py \\
        --wav data/stimuli/DownTheRabbitHoleFinal_mono_exp120_NR16_pad.wav \\
        --tr-s 2 --stim-onset-s 0.67 --out rabbit_hole_rms_tr.npz
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from natural_features import (
    build_experiment_grid,
    extract_features,
    query_feature_window_tr,
)
from natural_features.util.io import atomic_numpy_savez, atomic_write_json

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WAV_NAME = "DownTheRabbitHoleFinal_mono_exp120_NR16_pad.wav"
DEFAULT_WAV = REPO_ROOT / "data" / "stimuli" / DEFAULT_WAV_NAME
WAV_ENV = "NF_RABBIT_HOLE_WAV"


def _resolve_wav(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get(WAV_ENV, "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_WAV


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--wav",
        default=None,
        help=f"Audio path (default: ${WAV_ENV} or {DEFAULT_WAV})",
    )
    ap.add_argument(
        "--tr-s", type=float, default=2.0, help="Scan repetition time in seconds"
    )
    ap.add_argument(
        "--stim-onset-s",
        type=float,
        default=0.67,
        help="Scan time at which audio t=0 occurs",
    )
    ap.add_argument("--hop-s", type=float, default=0.01, help="RMS hop in seconds")
    ap.add_argument("--win-s", type=float, default=0.025, help="RMS window in seconds")
    ap.add_argument(
        "--method",
        default="mean",
        choices=["mean", "linear", "nearest"],
        help="How to sample native RMS rows onto each TR",
    )
    ap.add_argument("--out", default=None, help="Optional .npz output path")
    args = ap.parse_args()

    wav_path = _resolve_wav(args.wav)
    if not wav_path.is_file():
        raise FileNotFoundError(
            f"Audio not found: {wav_path}. Copy the WAV into data/stimuli/ "
            f"or pass --wav / set {WAV_ENV}."
        )
    if args.tr_s <= 0:
        raise ValueError("--tr-s must be > 0")
    if not np.isfinite(args.stim_onset_s):
        raise ValueError("--stim-onset-s must be finite")

    result = extract_features(
        wav_path,
        features=["audio.rms"],
        feature_params={"audio.rms": {"hop_s": args.hop_s, "win_s": args.win_s}},
    )
    rms = result.features["audio.rms"]
    audio = result.inputs["audio"]
    duration_s = float(audio.samples.shape[0] / audio.sr_hz)
    scan_span_s = float(args.stim_onset_s) + duration_s
    n_trs = int(np.ceil(scan_span_s / float(args.tr_s)))
    if n_trs <= 0:
        raise ValueError("Computed n_trs <= 0. Check TR, duration, and stimulus onset.")

    grid = build_experiment_grid(
        tr_s=float(args.tr_s),
        n_trs_by_run=[n_trs],
        run_starts_s=[0.0],
        feature_t0_s=float(args.stim_onset_s),
    )
    sampled = query_feature_window_tr(
        rms,
        grid,
        run_index=1,
        t_start_s=0.0,
        t_end_s=n_trs * float(args.tr_s),
        relative_to_run=True,
        method=str(args.method),
        output_time="run_relative",
    )
    energy = sampled.values.reshape(sampled.values.shape[0], -1)[:, 0]

    print(f"wav: {wav_path}")
    print(f"audio_duration_s: {duration_s:.6f}")
    print(f"stim_onset_s: {float(args.stim_onset_s):.6f}")
    print(f"tr_s: {float(args.tr_s):.6f}")
    print(f"n_trs: {n_trs}")
    print(f"native_rms_rows: {rms.values.shape[0]}")
    print(f"tr_rms_rows: {sampled.values.shape[0]}")
    print(f"tr_times_s: {sampled.times_s[0]:.6f} .. {sampled.times_s[-1]:.6f}")
    print(f"tr_rms_mean: {float(np.mean(energy)):.8f}")
    print(f"tr_rms_max: {float(np.max(energy)):.8f}")

    if args.out:
        out = Path(args.out)
        npz_path = out if out.suffix == ".npz" else out.with_suffix(".npz")
        atomic_numpy_savez(
            npz_path,
            times_s=sampled.times_s.astype(np.float64),
            rms=energy.astype(np.float32),
        )
        atomic_write_json(
            npz_path.with_suffix(".json"),
            {
                "wav": str(wav_path),
                "tr_s": float(args.tr_s),
                "stim_onset_s": float(args.stim_onset_s),
                "method": str(args.method),
                "n_trs": n_trs,
                "audio_duration_s": duration_s,
                "native_rms_rows": int(rms.values.shape[0]),
                "npz_path": str(npz_path),
            },
            sort_keys=True,
            indent=2,
        )
        print(f"wrote: {npz_path}")
        print(f"wrote: {npz_path.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
