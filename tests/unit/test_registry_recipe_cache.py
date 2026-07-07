from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
from io import StringIO
import wave

import numpy as np
import pytest

from natural_features.cli.main import main as cli_main
from natural_features.core.feature_types import EventSeries
from natural_features.core.recipe import as_mermaid, execute_recipe, plan_dag, validate_recipe
from natural_features.core.registry import Registry
from natural_features.core.stimulus import AudioStimulus, ImageStimulus, VideoStimulus
from natural_features.flow.cache import cache_fingerprint, invalidation_reasons


def test_builtin_registry_loads_specs() -> None:
    reg = Registry.with_builtin_specs()
    names = [s.name for s in reg.list()]
    assert "vision.lowlevel.visual_energy" in names
    assert "vision.energy" in names
    assert "vision.face" in names
    assert "vision.clip" in names
    assert "audio.lowlevel.rms" in names
    assert "speech.phonology.ctc_posteriors" in names
    assert "speech.articulatory.from_phoneme_events" in names


def test_registry_rejects_invalid_param_spec_default_type() -> None:
    reg = Registry()
    with pytest.raises(ValueError):
        reg.register(
            {
                "name": "test.bad_spec",
                "impl": "natural_features.features.vision.lowlevel:visual_energy",
                "modalities": ["video"],
                "params": {"include_deltas": {"type": "bool", "default": "yes"}},
            }
        )


def test_registry_accepts_nullable_param_default_none() -> None:
    reg = Registry()
    reg.register(
        {
            "name": "test.nullable_ok",
            "impl": "natural_features.features.vision.lowlevel:visual_energy",
            "modalities": ["video"],
            "params": {"x": {"type": "str", "nullable": True, "default": None}},
        }
    )
    assert reg.get("test.nullable_ok").name == "test.nullable_ok"


def test_recipe_ref_wiring_with_custom_registry() -> None:
    reg = Registry()
    reg.register(
        {
            "name": "test.step1",
            "impl": "natural_features.features.vision.lowlevel:visual_energy",
            "modalities": ["video"],
        }
    )
    reg.register(
        {
            "name": "test.step2",
            "impl": "natural_features.testing_helpers:pass_through_feature_series",
            "modalities": ["features"],
            "params": {"scale": {"type": "float", "default": 1.0}},
        }
    )
    frames = np.zeros((6, 8, 8, 3), dtype=np.uint8)
    recipe = {
        "features": [
            {"id": "a", "use": "test.step1"},
            {"id": "b", "use": "test.step2", "inputs": {"x": "ref: a.default"}, "params": {"scale": 2.0}},
        ]
    }
    out = execute_recipe(recipe, registry=reg, inputs={"video": VideoStimulus.from_array(frames, fps=5.0)})
    assert "a" in out.steps and "b" in out.steps
    assert out.steps["b"]["default"].values.shape[1] == out.steps["a"]["default"].values.shape[1]


def test_builtin_registry_executes_image_vision_aliases() -> None:
    reg = Registry.with_builtin_specs()
    img = ImageStimulus.from_array(np.ones((4, 5, 3), dtype=np.float32), onset_s=0.25)
    recipe = {
        "features": [
            {"id": "energy", "use": "vision.energy", "inputs": {"image": "input:image"}},
            {"id": "face", "use": "vision.face", "inputs": {"image": "input:image"}},
            {"id": "clip", "use": "vision.clip", "inputs": {"image": "input:image"}, "params": {"dim": 8}},
        ]
    }
    out = execute_recipe(recipe, registry=reg, inputs={"image": img})
    assert out.steps["energy"]["default"].values.shape[0] == 1
    assert out.steps["face"]["default"].values.shape[0] == 1
    assert out.steps["clip"]["default"].values.shape == (1, 8)


def test_recipe_rejects_unknown_params() -> None:
    reg = Registry.with_builtin_specs()
    frames = np.zeros((6, 8, 8, 3), dtype=np.uint8)
    recipe = {
        "features": [
            {"id": "a", "use": "vision.lowlevel.visual_energy", "params": {"unknown_param": 1}},
        ]
    }
    with pytest.raises(ValueError):
        execute_recipe(recipe, registry=reg, inputs={"video": VideoStimulus.from_array(frames, fps=5.0)})


def test_recipe_rejects_param_type_mismatch() -> None:
    reg = Registry.with_builtin_specs()
    frames = np.zeros((6, 8, 8, 3), dtype=np.uint8)
    recipe = {
        "features": [
            {"id": "a", "use": "vision.lowlevel.visual_energy", "params": {"include_deltas": "yes"}},
        ]
    }
    with pytest.raises(ValueError):
        execute_recipe(recipe, registry=reg, inputs={"video": VideoStimulus.from_array(frames, fps=5.0)})


def test_recipe_rejects_duplicate_step_ids() -> None:
    reg = Registry.with_builtin_specs()
    frames = np.zeros((6, 8, 8, 3), dtype=np.uint8)
    recipe = {
        "features": [
            {"id": "dup", "use": "vision.lowlevel.visual_energy"},
            {"id": "dup", "use": "vision.dynamics.frame_diffs"},
        ]
    }
    with pytest.raises(ValueError):
        execute_recipe(recipe, registry=reg, inputs={"video": VideoStimulus.from_array(frames, fps=5.0)})


def test_recipe_rejects_unknown_step_keys() -> None:
    reg = Registry.with_builtin_specs()
    frames = np.zeros((6, 8, 8, 3), dtype=np.uint8)
    recipe = {
        "features": [
            {"id": "a", "use": "vision.lowlevel.visual_energy", "bogus": 1},
        ]
    }
    with pytest.raises(ValueError):
        execute_recipe(recipe, registry=reg, inputs={"video": VideoStimulus.from_array(frames, fps=5.0)})


def test_recipe_enforces_declared_output_schema() -> None:
    reg = Registry()
    reg.register(
        {
            "name": "test.step1",
            "impl": "natural_features.features.vision.lowlevel:visual_energy",
            "modalities": ["video"],
        }
    )
    reg.register(
        {
            "name": "test.bad_output",
            "impl": "natural_features.testing_helpers:wrong_typed_output",
            "modalities": ["features"],
            "outputs": {"default": {"schema": "FeatureSeries/v1", "kind": "features"}},
        }
    )
    frames = np.zeros((6, 8, 8, 3), dtype=np.uint8)
    recipe = {
        "features": [
            {"id": "a", "use": "test.step1"},
            {"id": "b", "use": "test.bad_output", "inputs": {"x": "ref: a.default"}},
        ]
    }
    with pytest.raises(TypeError):
        execute_recipe(recipe, registry=reg, inputs={"video": VideoStimulus.from_array(frames, fps=5.0)})


def test_validate_recipe_static_contracts() -> None:
    reg = Registry.with_builtin_specs()
    recipe = {
        "features": [
            {"id": "asr", "use": "speech.asr.whisper"},
            {"id": "art", "use": "speech.articulatory.features", "inputs": {"words": "ref: asr.words"}},
        ]
    }
    val = validate_recipe(recipe, registry=reg, input_keys={"audio"})
    assert val.step_ids == ["asr", "art"]
    assert val.outputs_by_step["asr"] == ["qc", "segments", "words"]


def test_recipe_dag_outputs_depends_on_input_tokens_and_mermaid() -> None:
    reg = Registry.with_builtin_specs()
    recipe = {
        "features": [
            {
                "id": "rms",
                "use": "audio.lowlevel.rms",
                "inputs": {"audio": "input:audio"},
                "outputs": {"default": {"schema": "FeatureSeries/v1", "kind": "features"}},
            },
            {
                "id": "mel",
                "use": "audio.lowlevel.mel",
                "inputs": {"audio": "input:audio"},
                "depends_on": "rms",
            },
        ]
    }
    val = validate_recipe(recipe, registry=reg, input_keys={"audio"})
    assert val.step_ids == ["rms", "mel"]
    assert val.outputs_by_step["rms"] == ["default"]
    dag = plan_dag(recipe, registry=reg, input_keys={"audio"})
    assert any(node["id"] == "merge" for node in dag.nodes)
    assert any(edge["from"] == "rms" and edge["to"] == "mel" for edge in dag.edges)
    mermaid = as_mermaid(dag)
    assert "flowchart TD" in mermaid
    assert "audio.lowlevel.rms" in mermaid


def test_execute_recipe_honors_depends_on_execution_order() -> None:
    reg = Registry.with_builtin_specs()
    audio = AudioStimulus.from_array(np.linspace(-1.0, 1.0, 1000, dtype=np.float32), sr_hz=1000)
    recipe = {
        "features": [
            {
                "id": "mel",
                "use": "audio.lowlevel.mel",
                "inputs": {"audio": "input:audio"},
                "depends_on": "rms",
                "params": {"n_mels": 8},
            },
            {
                "id": "rms",
                "use": "audio.lowlevel.rms",
                "inputs": {"audio": "input:audio"},
            },
        ]
    }

    out = execute_recipe(recipe, registry=reg, inputs={"audio": audio})

    assert list(out.steps) == ["input", "rms", "mel"]
    assert out.steps["mel"]["default"].values.shape[1] == 8


def test_recipe_dag_detects_depends_on_cycles() -> None:
    reg = Registry.with_builtin_specs()
    recipe = {
        "features": [
            {"id": "a", "use": "audio.lowlevel.rms", "depends_on": "b"},
            {"id": "b", "use": "audio.lowlevel.mel", "depends_on": "a"},
        ]
    }
    with pytest.raises(ValueError, match="cycle"):
        validate_recipe(recipe, registry=reg, input_keys={"audio"})


def test_cache_fingerprint_and_invalidation() -> None:
    prev = {
        "extractor_name": "x",
        "params": {"a": 1},
        "code_version": "c1",
        "model_revision": "m1",
        "upstream_ids": ["u1"],
    }
    curr = dict(prev)
    assert cache_fingerprint(**prev) == cache_fingerprint(**curr)
    curr["params"] = {"a": 2}
    reasons = invalidation_reasons(prev, curr)
    assert "parameters changed" in reasons


def test_cli_list_describe_and_extract(tmp_path, capsys) -> None:
    # list
    assert cli_main(["list"]) == 0
    listed = capsys.readouterr().out
    assert "vision.lowlevel.visual_energy" in listed

    # describe
    assert cli_main(["describe", "audio.lowlevel.rms"]) == 0
    desc = capsys.readouterr().out
    assert "\"name\": \"audio.lowlevel.rms\"" in desc

    # extract
    recipe_path = Path("tests/fixtures/recipe_baseline.yaml")
    frames = np.zeros((10, 8, 8, 3), dtype=np.uint8)
    video_path = tmp_path / "video.npy"
    np.save(video_path, frames)

    wav_path = tmp_path / "audio.wav"
    sr = 8000
    t = np.arange(sr, dtype=np.float32) / sr
    x = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())

    out_dir = tmp_path / "catalog"
    assert (
        cli_main(
            [
                "extract",
                str(recipe_path),
                "--video-npy",
                str(video_path),
                "--video-fps",
                "5",
                "--audio-wav",
                str(wav_path),
                "--out",
                str(out_dir),
                "--json",
            ]
        )
        == 0
    )
    payload = capsys.readouterr().out
    assert "\"artifacts\"" in payload
    assert (out_dir / "catalog.sqlite3").exists()


def test_cli_features_exports_public_catalog_text_json_and_csv(capsys) -> None:
    assert cli_main(["features", "--modality", "audio", "--budget", "all"]) == 0
    listed = capsys.readouterr().out
    assert "feature_id\tmodalities\toutput_schema" in listed
    assert "audio.rms" in listed
    assert "audio.lowlevel.rms" not in listed

    assert cli_main(["features", "--modality", "audio", "--budget", "all", "--include-internal", "--csv"]) == 0
    csv_text = capsys.readouterr().out
    rows = list(csv.DictReader(StringIO(csv_text)))
    ids = {row["feature_id"] for row in rows}
    assert "audio.rms" in ids
    assert "audio.lowlevel.rms" in ids
    assert next(row for row in rows if row["feature_id"] == "audio.lowlevel.rms")["is_public"] == "false"

    assert cli_main(["features", "--modality", "text", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [row["feature_id"] for row in payload["features"]] == ["text.tokenize"]
    assert payload["features"][0]["is_public"] is True


def test_cli_validate_and_presets(capsys) -> None:
    recipe_path = Path("tests/fixtures/recipe_baseline.yaml")
    assert cli_main(["validate", str(recipe_path), "--have", "video", "--have", "audio", "--json"]) == 0
    validated = capsys.readouterr().out
    assert "\"valid\": true" in validated

    assert cli_main(["preset-list"]) == 0
    listed = capsys.readouterr().out
    assert "fmri_speech_language" in listed

    assert cli_main(["preset-show", "fmri_speech_language"]) == 0
    shown = capsys.readouterr().out
    assert "speech.asr.whisper" in shown


def test_cli_prep_video_with_mocked_ffmpeg(tmp_path, capsys, monkeypatch) -> None:
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"fake")
    video_out = tmp_path / "clip.npy"
    audio_out = tmp_path / "clip.wav"

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None)

    def _fake_check_output(cmd):
        assert cmd[0] == "ffmpeg"
        # Two RGB frames at 2x2 resolution.
        return bytes(range(24))

    def _fake_run(cmd, check):
        assert cmd[0] == "ffmpeg"
        assert check is True
        out = Path(cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(8000)
            w.writeframes(np.zeros(16, dtype=np.int16).tobytes())
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("subprocess.check_output", _fake_check_output)
    monkeypatch.setattr("subprocess.run", _fake_run)

    assert (
        cli_main(
            [
                "prep-video",
                str(src),
                "--video-npy",
                str(video_out),
                "--audio-wav",
                str(audio_out),
                "--video-fps",
                "2",
                "--video-width",
                "2",
                "--video-height",
                "2",
                "--audio-sr",
                "8000",
                "--duration-s",
                "1.0",
                "--json",
            ]
        )
        == 0
    )
    payload = capsys.readouterr().out
    assert "\"video_prepared\": true" in payload
    assert "\"audio_prepared\": true" in payload
    arr = np.load(video_out, allow_pickle=False)
    assert arr.shape == (2, 2, 2, 3)
    assert audio_out.exists()


def test_cli_speech_align_exports_ctm_textgrid_and_json(tmp_path, capsys) -> None:
    wav_path = tmp_path / "audio.wav"
    sr = 8000
    t = np.arange(sr, dtype=np.float32) / sr
    x = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())

    ctm_out = tmp_path / "words.ctm"
    tg_out = tmp_path / "words.TextGrid"
    json_out = tmp_path / "summary.json"

    assert (
        cli_main(
            [
                "speech-align",
                "--audio-wav",
                str(wav_path),
                "--align-backend",
                "none",
                "--ctm-out",
                str(ctm_out),
                "--textgrid-out",
                str(tg_out),
                "--out-json",
                str(json_out),
                "--json",
            ]
        )
        == 0
    )
    payload = capsys.readouterr().out
    assert "\"n_words\"" in payload
    assert ctm_out.exists()
    assert tg_out.exists()
    assert json_out.exists()


def test_cli_speech_validate_backends_json(capsys, monkeypatch, tmp_path) -> None:
    report = {
        "validated_at": "2026-01-01T00:00:00+00:00",
        "backends": {
            "whisperx": {"available": True, "runtime_checked": True, "runtime_ok": True},
            "mfa": {"available": False, "runtime_checked": False, "runtime_ok": None},
            "gentle": {"available": False, "runtime_checked": False, "runtime_ok": None},
        },
    }
    monkeypatch.setattr("natural_features.cli.main.validate_alignment_backends", lambda **_: report)
    out_json = tmp_path / "backend_report.json"
    assert cli_main(["speech-validate-backends", "--out-json", str(out_json), "--json"]) == 0
    payload = capsys.readouterr().out
    assert "\"whisperx\"" in payload
    assert out_json.exists()


def test_cli_speech_benchmark_json(capsys, monkeypatch, tmp_path) -> None:
    report = {
        "manifest_items": 1,
        "summary": {"n_items": 1, "n_success": 1, "n_failed": 0, "fallback_rate": 0.0},
        "results": [{"clip_id": "x"}],
    }
    monkeypatch.setattr("natural_features.cli.main.run_alignment_benchmark", lambda *args, **kwargs: report)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{\"items\":[]}", encoding="utf-8")
    out_json = tmp_path / "benchmark_report.json"
    assert cli_main(["speech-benchmark", "--manifest", str(manifest), "--out-json", str(out_json), "--json"]) == 0
    payload = capsys.readouterr().out
    assert "\"summary\"" in payload
    assert out_json.exists()


def test_cli_speech_doctor_json(capsys, monkeypatch, tmp_path) -> None:
    validation = {
        "validated_at": "2026-01-01T00:00:00+00:00",
        "backends": {
            "whisperx": {"available": False, "reason": "ModuleNotFoundError: No module named 'whisperx'"},
            "mfa": {"available": False, "reason": "mfa executable not found"},
            "gentle": {"available": False, "reason": "ModuleNotFoundError: No module named 'gentle'"},
        },
    }
    doctor = {"health": "unavailable", "blockers": ["whisperx", "mfa"], "recommendations": []}
    monkeypatch.setattr("natural_features.cli.main.validate_alignment_backends", lambda **_: validation)
    monkeypatch.setattr("natural_features.cli.main.build_alignment_doctor_report", lambda _v: doctor)
    out_json = tmp_path / "doctor_report.json"
    assert cli_main(["speech-doctor", "--out-json", str(out_json), "--json"]) == 0
    payload = capsys.readouterr().out
    assert "\"health\": \"unavailable\"" in payload
    assert out_json.exists()


def test_cli_speech_align_forwards_mfa_args(monkeypatch, tmp_path, capsys) -> None:
    wav_path = tmp_path / "audio.wav"
    sr = 8000
    t = np.arange(sr, dtype=np.float32) / sr
    x = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())

    monkeypatch.setattr(
        "natural_features.cli.main.whisper_transcribe",
        lambda *args, **kwargs: {
            "words": EventSeries(
                onset_s=np.array([0.0], dtype=np.float64),
                offset_s=np.array([0.5], dtype=np.float64),
                label=np.array(["hello"], dtype=object),
                confidence=np.array([1.0], dtype=np.float32),
                metadata={"extractor_id": "x", "params_hash": "y", "asr_model_name": "small"},
            ),
            "qc": {},
        },
    )

    captured = {}

    def _fake_align(*args, **kwargs):
        captured.update(kwargs)
        return {
            "words": EventSeries(
                onset_s=np.array([0.0], dtype=np.float64),
                offset_s=np.array([0.5], dtype=np.float64),
                label=np.array(["hello"], dtype=object),
                confidence=np.array([1.0], dtype=np.float32),
                metadata={"extractor_id": "x", "params_hash": "y", "asr_model_name": "small"},
            ),
            "qc": {"mode": "mfa"},
        }

    monkeypatch.setattr("natural_features.cli.main.whisperx_align", _fake_align)

    assert (
        cli_main(
            [
                "speech-align",
                "--audio-wav",
                str(wav_path),
                "--align-backend",
                "mfa",
                "--mfa-dictionary",
                "/tmp/dict.dict",
                "--mfa-acoustic-model",
                "/tmp/acoustic.zip",
                "--mfa-extra-arg",
                "beam=20",
                "--json",
            ]
        )
        == 0
    )
    _ = capsys.readouterr().out
    assert captured["backend"] == "mfa"
    assert captured["mfa_dictionary_path"] == "/tmp/dict.dict"
    assert captured["mfa_acoustic_model_path"] == "/tmp/acoustic.zip"
    assert "beam=20" in captured["mfa_extra_args"]
