#!/usr/bin/env python3
"""Extract RMS energy and convolve it with fmrimod's SPMG1 HRF.

Native RMS stays on its hop grid. fmrimod builds the predicted BOLD
regressor and evaluates it on the scan TR grid.

    PYTHONPATH=/path/to/fmrimod python examples/rms_spmg1_hrf.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from natural_features import (
    ClockMap,
    TemporalContext,
    extract_features,
    temporal_object_in_clock,
)
from natural_features.fmri.compat import has_fmrimod, hrf_regressor
from natural_features.util.io import atomic_numpy_savez, atomic_write_json

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WAV = (
    REPO_ROOT / "data" / "stimuli" / "DownTheRabbitHoleFinal_mono_exp120_NR16_pad.wav"
)
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
        "--wav", default=None, help=f"Audio path (default: ${WAV_ENV} or {DEFAULT_WAV})"
    )
    ap.add_argument("--tr-s", type=float, default=2.0)
    ap.add_argument("--stim-onset-s", type=float, default=0.67)
    ap.add_argument("--hrf", default="spmg1", help="fmrimod HRF name")
    ap.add_argument("--precision", type=float, default=0.1)
    ap.add_argument("--out", default=None, help="Optional .npz output path")
    args = ap.parse_args()

    if not has_fmrimod():
        raise RuntimeError(
            "fmrimod is required. Install from "
            "https://github.com/bbuchsbaum/fmrimod or add it to PYTHONPATH."
        )

    wav_path = _resolve_wav(args.wav)
    if not wav_path.is_file():
        raise FileNotFoundError(f"Audio not found: {wav_path}")

    result = extract_features(wav_path, features=["audio.rms"])
    rms = result.features["audio.rms"]
    audio = result.inputs["audio"]
    duration_s = float(audio.samples.shape[0] / audio.sr_hz)
    n_trs = int(np.ceil((float(args.stim_onset_s) + duration_s) / float(args.tr_s)))
    rms_scan = temporal_object_in_clock(
        rms,
        "scan:run-01",
        context=TemporalContext(
            (ClockMap("stimulus", "scan:run-01", offset_s=float(args.stim_onset_s)),)
        ),
    )
    bold = hrf_regressor(
        rms_scan,
        tr_s=float(args.tr_s),
        n_scans=n_trs,
        hrf=str(args.hrf),
        precision=float(args.precision),
        start_time=0.0,
    )
    energy = rms.values[:, 0]
    pred = bold.values[:, 0]
    print(f"wav: {wav_path}")
    print(f"native_rms_rows: {energy.shape[0]}")
    print(f"hrf: {args.hrf}")
    print(f"n_trs: {n_trs}")
    print(f"tr_times_s: {bold.times_s[0]:.6f} .. {bold.times_s[-1]:.6f}")
    print(f"rms_max: {float(np.max(energy)):.8f}")
    print(f"spmg1_max: {float(np.max(pred)):.8f}")
    print(f"spmg1_peak_s: {float(bold.times_s[int(np.argmax(pred))]):.3f}")

    if args.out:
        out = Path(args.out)
        npz_path = out if out.suffix == ".npz" else out.with_suffix(".npz")
        atomic_numpy_savez(
            npz_path,
            rms_times_s=rms_scan.times_s.astype(np.float64),
            rms=energy.astype(np.float32),
            tr_times_s=bold.times_s.astype(np.float64),
            hrf_regressor=pred.astype(np.float32),
        )
        atomic_write_json(
            npz_path.with_suffix(".json"),
            {
                "wav": str(wav_path),
                "tr_s": float(args.tr_s),
                "stim_onset_s": float(args.stim_onset_s),
                "hrf": str(args.hrf),
                "n_trs": n_trs,
                "npz_path": str(npz_path),
            },
            sort_keys=True,
            indent=2,
        )
        print(f"wrote: {npz_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
