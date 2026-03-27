"""CLI entrypoint for natural_features."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np

from natural_features.core.feature_types import EventSeries, FeatureSeries, TrackSeries
from natural_features.core.recipe import execute_recipe, load_recipe, validate_recipe
from natural_features.core.registry import Registry
from natural_features.core.stimulus import AudioStimulus, VideoStimulus
from natural_features.features.speech.align import whisperx_align
from natural_features.features.speech.asr import whisper_transcribe, whisper_transcribe_chunked
from natural_features.features.speech.benchmark import BenchmarkConfig, run_alignment_benchmark
from natural_features.features.speech.doctor import build_alignment_doctor_report
from natural_features.features.speech.formats import read_ctm, write_ctm, write_textgrid
from natural_features.features.speech.validation import validate_alignment_backends
from natural_features.storage.catalog import Catalog
from natural_features.util.io import atomic_numpy_save, atomic_write_json
import yaml


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nf", description="natural_features CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List registered extractors")

    d = sub.add_parser("describe", help="Describe an extractor")
    d.add_argument("name")

    v = sub.add_parser("validate", help="Validate a recipe without executing extractors")
    v.add_argument("recipe")
    v.add_argument("--have", action="append", default=[], help="Declared available input modality (repeatable)")
    v.add_argument("--json", action="store_true", help="Emit JSON output")

    pl = sub.add_parser("preset-list", help="List built-in presets")
    pl.add_argument("--json", action="store_true", help="Emit JSON output")

    ps = sub.add_parser("preset-show", help="Show built-in preset YAML/JSON")
    ps.add_argument("name", help="Preset name or path")
    ps.add_argument("--json", action="store_true", help="Emit JSON output")

    e = sub.add_parser("extract", help="Run a recipe")
    e.add_argument("recipe")
    e.add_argument("--video-npy", default=None)
    e.add_argument("--video-fps", type=float, default=10.0)
    e.add_argument("--audio-wav", default=None)
    e.add_argument("--out", default="nf_catalog")
    e.add_argument("--run-id", default=None)
    e.add_argument("--json", action="store_true", help="Emit JSON output")

    pv = sub.add_parser("prep-video", help="Prepare video+audio sidecars for feature extraction")
    pv.add_argument("input_video", help="Input video file path (e.g., mp4/mkv)")
    pv.add_argument("--video-npy", default=None, help="Output .npy path for RGB frames")
    pv.add_argument("--audio-wav", default=None, help="Output .wav path for mono audio")
    pv.add_argument("--video-fps", type=float, default=10.0, help="Frame sampling rate in Hz")
    pv.add_argument("--video-width", type=int, default=224, help="Output frame width")
    pv.add_argument("--video-height", type=int, default=224, help="Output frame height")
    pv.add_argument("--audio-sr", type=int, default=16000, help="Output audio sample rate in Hz")
    pv.add_argument("--start-s", type=float, default=0.0, help="Start time (seconds)")
    pv.add_argument("--duration-s", type=float, default=None, help="Clip duration (seconds)")
    pv.add_argument("--no-video", action="store_true", help="Skip video frame extraction")
    pv.add_argument("--no-audio", action="store_true", help="Skip audio extraction")
    pv.add_argument("--json", action="store_true", help="Emit JSON output")

    sa = sub.add_parser("speech-align", help="One-pass ASR + alignment with optional CTM/TextGrid export")
    sa.add_argument("--audio-wav", required=True, help="Input mono/stereo WAV file")
    sa.add_argument("--model", default="small", help="ASR model id")
    sa.add_argument("--language", default="auto", help="ASR language code or auto")
    sa.add_argument("--execution-mode", default="fallback", choices=["fallback", "strict"])
    sa.add_argument("--strict-dependency", action="store_true", help="Treat missing deps as hard failures")
    sa.add_argument("--chunked", action="store_true", help="Use chunked ASR path for long audio")
    sa.add_argument("--chunk-window-s", type=float, default=30.0, help="Chunk window (s)")
    sa.add_argument("--chunk-overlap-s", type=float, default=1.0, help="Chunk overlap (s)")
    sa.add_argument("--align-backend", default="auto", help="Alignment backend: auto|whisperx|mfa|gentle|none")
    sa.add_argument("--mfa-dictionary", default=None, help="MFA dictionary path (required for backend=mfa)")
    sa.add_argument("--mfa-acoustic-model", default=None, help="MFA acoustic model path (required for backend=mfa)")
    sa.add_argument("--mfa-timeout-s", type=float, default=300.0, help="MFA align timeout in seconds")
    sa.add_argument("--mfa-tmp-dir", default=None, help="Optional temp dir for MFA intermediate files")
    sa.add_argument(
        "--mfa-extra-arg",
        action="append",
        default=[],
        help="Additional MFA CLI argument (repeatable), forwarded to `mfa align`",
    )
    sa.add_argument("--ctm-out", default=None, help="Optional CTM output path")
    sa.add_argument("--textgrid-out", default=None, help="Optional TextGrid output path")
    sa.add_argument("--out-json", default=None, help="Optional JSON summary output path")
    sa.add_argument("--json", action="store_true", help="Emit JSON output")

    sv = sub.add_parser("speech-validate-backends", help="Validate whisperx/MFA/gentle backend runtime readiness")
    sv.add_argument("--audio-wav", default=None, help="Optional WAV used for whisperx runtime check")
    sv.add_argument("--words-ctm", default=None, help="Optional CTM words for whisperx runtime check")
    sv.add_argument("--transcript", default=None, help="Optional transcript text for generated check words")
    sv.add_argument("--language", default="en", help="Language code for alignment check")
    sv.add_argument("--execution-mode", default="fallback", choices=["fallback", "strict"])
    sv.add_argument("--timeout-s", type=float, default=10.0, help="Timeout for backend subprocess checks")
    sv.add_argument("--out-json", default=None, help="Optional JSON report output path")
    sv.add_argument("--json", action="store_true", help="Emit JSON output")

    sd = sub.add_parser("speech-doctor", help="Diagnose alignment backend failures and suggest fixes")
    sd.add_argument("--audio-wav", default=None, help="Optional WAV used for runtime checks")
    sd.add_argument("--words-ctm", default=None, help="Optional CTM words for whisperx runtime check")
    sd.add_argument("--transcript", default=None, help="Optional transcript text for generated check words")
    sd.add_argument("--language", default="en", help="Language code for alignment check")
    sd.add_argument("--execution-mode", default="fallback", choices=["fallback", "strict"])
    sd.add_argument("--timeout-s", type=float, default=10.0, help="Timeout for backend subprocess checks")
    sd.add_argument("--out-json", default=None, help="Optional JSON doctor output path")
    sd.add_argument("--json", action="store_true", help="Emit JSON output")

    sb = sub.add_parser("speech-benchmark", help="Benchmark ASR+alignment against a manifest of reference alignments")
    sb.add_argument("--manifest", required=True, help="Benchmark manifest JSON path")
    sb.add_argument("--root", default=None, help="Optional root directory for relative manifest paths")
    sb.add_argument("--backend", default="auto", help="Alignment backend: auto|whisperx|mfa|gentle|none")
    sb.add_argument("--asr-model", default="small", help="ASR model id")
    sb.add_argument("--language", default="en", help="Language code")
    sb.add_argument("--execution-mode", default="fallback", choices=["fallback", "strict"])
    sb.add_argument("--strict-dependency", action="store_true", help="Fail on missing runtime deps")
    sb.add_argument("--fail-fast", action="store_true", help="Stop on first benchmark failure")
    sb.add_argument("--out-json", default=None, help="Optional benchmark JSON output path")
    sb.add_argument("--json", action="store_true", help="Emit JSON output")
    return p


def _load_inputs(args: argparse.Namespace) -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    if args.video_npy:
        inputs["video"] = VideoStimulus.from_npy(args.video_npy, fps=args.video_fps)
    if args.audio_wav:
        inputs["audio"] = AudioStimulus.from_wav(args.audio_wav)
    return inputs


def _cmd_list(reg: Registry) -> int:
    for spec in reg.list():
        print(f"{spec.name}\tmodalities={','.join(spec.modalities)}\trequires={','.join(spec.requires)}")
    return 0


def _cmd_describe(reg: Registry, name: str) -> int:
    spec = reg.get(name)
    print(json.dumps(spec.__dict__, indent=2, sort_keys=True))
    return 0


def _preset_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "zoo" / "presets"


def _cmd_preset_list(args: argparse.Namespace) -> int:
    pdir = _preset_dir()
    names = sorted([p.stem for p in pdir.glob("*.yaml")])
    if args.json:
        print(json.dumps({"presets": names}, indent=2, sort_keys=True))
    else:
        for name in names:
            print(name)
    return 0


def _resolve_preset_path(name_or_path: str) -> Path:
    raw = Path(name_or_path)
    if raw.exists():
        return raw
    pdir = _preset_dir()
    cand = pdir / f"{name_or_path}.yaml"
    if cand.exists():
        return cand
    raise FileNotFoundError(f"Preset not found: {name_or_path}")


def _cmd_preset_show(args: argparse.Namespace) -> int:
    p = _resolve_preset_path(args.name)
    payload = yaml.safe_load(p.read_text(encoding="utf-8"))
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(p.read_text(encoding="utf-8"))
    return 0


def _cmd_validate(reg: Registry, args: argparse.Namespace) -> int:
    recipe = load_recipe(args.recipe)
    validation = validate_recipe(recipe, registry=reg, input_keys=set(args.have))
    if args.json:
        print(
            json.dumps(
                {
                    "valid": True,
                    "steps": validation.step_ids,
                    "outputs_by_step": validation.outputs_by_step,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print("valid=true")
        print(f"n_steps={len(validation.step_ids)}")
        for step_id in validation.step_ids:
            outputs = ",".join(validation.outputs_by_step.get(step_id, []))
            print(f"{step_id}\toutputs={outputs}")
    return 0


def _cmd_extract(reg: Registry, args: argparse.Namespace) -> int:
    recipe = load_recipe(args.recipe)
    inputs = _load_inputs(args)
    result = execute_recipe(recipe, registry=reg, inputs=inputs)

    run_id = args.run_id or f"run-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    catalog = Catalog(args.out)
    created_at = datetime.now(timezone.utc).isoformat()
    records: list[dict[str, Any]] = []
    for step_id, outputs in result.steps.items():
        for out_key, obj in outputs.items():
            if not isinstance(obj, (FeatureSeries, EventSeries, TrackSeries)):
                continue
            rec = catalog.put(
                obj,
                run_id=run_id,
                stage_id=f"{step_id}.{out_key}",
                code_version="dev",
                created_at=created_at,
                preferred_format="npz",
            )
            records.append(rec.__dict__)
    if args.json:
        print(json.dumps({"run_id": run_id, "artifacts": records}, indent=2, sort_keys=True))
    else:
        print(f"run_id={run_id}")
        for rec in records:
            print(f"{rec['artifact_id']}\t{rec['stage_id']}\t{rec['path']}")
    return 0


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")


def _ffmpeg_trim_args(*, start_s: float, duration_s: float | None) -> list[str]:
    args: list[str] = []
    if start_s > 0:
        args.extend(["-ss", str(start_s)])
    if duration_s is not None:
        if duration_s <= 0:
            raise ValueError("--duration-s must be > 0 when provided")
        args.extend(["-t", str(duration_s)])
    return args


def _cmd_prep_video(args: argparse.Namespace) -> int:
    _require_ffmpeg()
    src = Path(args.input_video)
    if not src.exists():
        raise FileNotFoundError(f"Input video not found: {src}")
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be > 0")
    if args.video_width <= 0 or args.video_height <= 0:
        raise ValueError("--video-width/--video-height must be > 0")
    if args.audio_sr <= 0:
        raise ValueError("--audio-sr must be > 0")
    if args.start_s < 0:
        raise ValueError("--start-s must be >= 0")
    if args.no_video and args.no_audio:
        raise ValueError("Nothing to do: both --no-video and --no-audio are set")

    base = src.with_suffix("")
    video_out = Path(args.video_npy) if args.video_npy else base.with_suffix(".npy")
    audio_out = Path(args.audio_wav) if args.audio_wav else base.with_suffix(".wav")
    trim = _ffmpeg_trim_args(start_s=args.start_s, duration_s=args.duration_s)

    payload: dict[str, Any] = {
        "input_video": str(src),
        "video_prepared": False,
        "audio_prepared": False,
    }

    if not args.no_video:
        vf = f"fps={args.video_fps},scale={args.video_width}:{args.video_height}"
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            *trim,
            "-i",
            str(src),
            "-vf",
            vf,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
        raw = subprocess.check_output(cmd)
        bpf = int(args.video_width * args.video_height * 3)
        n_frames = len(raw) // bpf
        if n_frames <= 0:
            raise RuntimeError("ffmpeg returned no frames for the requested clip parameters")
        buf = raw[: n_frames * bpf]
        frames = np.frombuffer(buf, dtype=np.uint8).reshape(n_frames, args.video_height, args.video_width, 3)
        atomic_numpy_save(video_out, frames, allow_pickle=False)
        payload.update(
            {
                "video_prepared": True,
                "video_npy": str(video_out),
                "video_shape": list(frames.shape),
                "video_fps": float(args.video_fps),
            }
        )

    if not args.no_audio:
        audio_out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            *trim,
            "-i",
            str(src),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(args.audio_sr),
            str(audio_out),
        ]
        subprocess.run(cmd, check=True)
        payload.update(
            {
                "audio_prepared": True,
                "audio_wav": str(audio_out),
                "audio_sr": int(args.audio_sr),
            }
        )

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        if payload["video_prepared"]:
            print(f"video_npy={payload['video_npy']}\tshape={payload['video_shape']}\tfps={payload['video_fps']}")
        if payload["audio_prepared"]:
            print(f"audio_wav={payload['audio_wav']}\tsr={payload['audio_sr']}")
    return 0


def _cmd_speech_align(args: argparse.Namespace) -> int:
    audio = AudioStimulus.from_wav(args.audio_wav)
    strict = bool(args.strict_dependency) or (args.execution_mode == "strict")

    if args.chunked:
        asr = whisper_transcribe_chunked(
            audio,
            model=args.model,
            language=args.language,
            execution_mode=args.execution_mode,
            strict_dependency=strict,
            chunk_window_s=float(args.chunk_window_s),
            chunk_overlap_s=float(args.chunk_overlap_s),
        )
    else:
        asr = whisper_transcribe(
            audio,
            model=args.model,
            language=args.language,
            execution_mode=args.execution_mode,
            strict_dependency=strict,
        )

    aligned = whisperx_align(
        audio,
        asr["words"],
        backend=args.align_backend,
        language="en" if args.language == "auto" else args.language,
        mfa_dictionary_path=args.mfa_dictionary,
        mfa_acoustic_model_path=args.mfa_acoustic_model,
        mfa_timeout_s=float(args.mfa_timeout_s),
        mfa_tmp_dir=args.mfa_tmp_dir,
        mfa_extra_args=list(args.mfa_extra_arg or []),
        execution_mode=args.execution_mode,
        strict_dependency=strict,
    )
    words = aligned["words"]
    payload = {
        "audio_wav": str(args.audio_wav),
        "n_words": int(len(words)),
        "execution_mode": args.execution_mode,
        "asr_qc": asr.get("qc", {}),
        "align_qc": aligned.get("qc", {}),
        "word_metadata": words.metadata,
    }

    if args.ctm_out:
        ctm_path = write_ctm(words, args.ctm_out)
        payload["ctm_out"] = str(ctm_path)
    if args.textgrid_out:
        tg_path = write_textgrid(words, args.textgrid_out)
        payload["textgrid_out"] = str(tg_path)
    if args.out_json:
        out = Path(args.out_json)
        atomic_write_json(out, payload, sort_keys=True, indent=2)
        payload["out_json"] = str(out)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"n_words={payload['n_words']}")
        print(f"align_mode={payload['align_qc'].get('mode', 'unknown')}")
        if "ctm_out" in payload:
            print(f"ctm_out={payload['ctm_out']}")
        if "textgrid_out" in payload:
            print(f"textgrid_out={payload['textgrid_out']}")
        if "out_json" in payload:
            print(f"out_json={payload['out_json']}")
    return 0


def _cmd_speech_validate_backends(args: argparse.Namespace) -> int:
    audio = AudioStimulus.from_wav(args.audio_wav) if args.audio_wav else None
    words = read_ctm(args.words_ctm) if args.words_ctm else None
    payload = validate_alignment_backends(
        audio=audio,
        words=words,
        transcript_text=args.transcript,
        language=args.language,
        execution_mode=args.execution_mode,
        timeout_s=float(args.timeout_s),
    )
    if args.out_json:
        out = Path(args.out_json)
        atomic_write_json(out, payload, sort_keys=True, indent=2)
        payload["out_json"] = str(out)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for name in ("whisperx", "mfa", "gentle"):
            info = payload["backends"][name]
            print(
                f"{name}\tavailable={info['available']}\t"
                f"runtime_checked={info['runtime_checked']}\truntime_ok={info['runtime_ok']}"
            )
            if info.get("runtime_reason"):
                print(f"{name}.reason={info['runtime_reason']}")
        if "out_json" in payload:
            print(f"out_json={payload['out_json']}")
    return 0


def _cmd_speech_benchmark(args: argparse.Namespace) -> int:
    cfg = BenchmarkConfig(
        backend=args.backend,
        asr_model=args.asr_model,
        language=args.language,
        execution_mode=args.execution_mode,
        strict_dependency=bool(args.strict_dependency),
        continue_on_error=not bool(args.fail_fast),
    )
    payload = run_alignment_benchmark(
        args.manifest,
        root=args.root,
        config=cfg,
    )
    if args.out_json:
        out = Path(args.out_json)
        atomic_write_json(out, payload, sort_keys=True, indent=2)
        payload["out_json"] = str(out)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        s = payload.get("summary", {})
        print(
            f"n_items={s.get('n_items', 0)}\tn_success={s.get('n_success', 0)}\t"
            f"n_failed={s.get('n_failed', 0)}\tfallback_rate={s.get('fallback_rate', 0.0)}"
        )
        print(
            f"boundary_mae_ms_mean={s.get('boundary_mae_ms_mean')}\t"
            f"boundary_mae_ms_p95={s.get('boundary_mae_ms_p95')}"
        )
        if "out_json" in payload:
            print(f"out_json={payload['out_json']}")
    return 0


def _cmd_speech_doctor(args: argparse.Namespace) -> int:
    audio = AudioStimulus.from_wav(args.audio_wav) if args.audio_wav else None
    words = read_ctm(args.words_ctm) if args.words_ctm else None
    validation = validate_alignment_backends(
        audio=audio,
        words=words,
        transcript_text=args.transcript,
        language=args.language,
        execution_mode=args.execution_mode,
        timeout_s=float(args.timeout_s),
    )
    payload = build_alignment_doctor_report(validation)
    if args.out_json:
        out = Path(args.out_json)
        atomic_write_json(out, payload, sort_keys=True, indent=2)
        payload["out_json"] = str(out)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"health={payload.get('health', 'unknown')}")
        print(f"blockers={','.join(payload.get('blockers', [])) or 'none'}")
        for rec in payload.get("recommendations", []):
            print(f"{rec.get('backend')}\tseverity={rec.get('severity')}\t{rec.get('message')}")
            for action in rec.get("actions", []):
                print(f"  - {action.get('title')}: {action.get('command')}")
        if "out_json" in payload:
            print(f"out_json={payload['out_json']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    reg = Registry.with_builtin_specs()
    if args.cmd == "list":
        return _cmd_list(reg)
    if args.cmd == "describe":
        return _cmd_describe(reg, args.name)
    if args.cmd == "validate":
        return _cmd_validate(reg, args)
    if args.cmd == "preset-list":
        return _cmd_preset_list(args)
    if args.cmd == "preset-show":
        return _cmd_preset_show(args)
    if args.cmd == "extract":
        return _cmd_extract(reg, args)
    if args.cmd == "prep-video":
        return _cmd_prep_video(args)
    if args.cmd == "speech-align":
        return _cmd_speech_align(args)
    if args.cmd == "speech-validate-backends":
        return _cmd_speech_validate_backends(args)
    if args.cmd == "speech-doctor":
        return _cmd_speech_doctor(args)
    if args.cmd == "speech-benchmark":
        return _cmd_speech_benchmark(args)
    parser.error(f"Unsupported command: {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
