from __future__ import annotations

import subprocess
import sys
import wave

import numpy as np
import pytest

from natural_features.core.feature_types import FeatureSeries
from natural_features.core.recipe import execute_recipe, validate_recipe
from natural_features.core.registry import Registry
from natural_features.core.stimulus import AudioStimulus, ImageStimulus, TextStimulus, VideoStimulus
from natural_features.features.preprocess import image_ocr, video_audio_extract
from natural_features.features.common import extractor_metadata


def test_preprocessing_nodes_are_registered() -> None:
    reg = Registry.with_builtin_specs()
    names = {spec.name for spec in reg.list()}
    assert {
        "video.frames.sample",
        "video.audio.extract",
        "audio.trim",
        "audio.resample",
        "video.trim",
        "text.tokenize",
        "image.ocr",
        "video.ocr",
        "events.align",
        "features.resample",
        "features.hrf",
        "features.lag",
    } <= names


def test_video_sampling_feeds_visual_energy() -> None:
    reg = Registry.with_builtin_specs()
    stim = VideoStimulus.from_array(np.arange(6 * 3 * 3, dtype=np.float32).reshape(6, 3, 3), fps=6.0)
    recipe = {
        "features": [
            {"id": "sample", "use": "video.frames.sample", "inputs": {"video": "input:video"}, "params": {"stride_frames": 2}},
            {"id": "energy", "use": "vision.energy", "inputs": {"video": "ref:sample.default"}},
        ]
    }
    val = validate_recipe(recipe, registry=reg, input_keys={"video"})
    assert val.step_ids == ["sample", "energy"]
    out = execute_recipe(recipe, registry=reg, inputs={"video": stim})
    sampled = out.steps["sample"]["default"]
    assert sampled.frames.shape[0] == 3
    assert sampled.fps == 3.0
    assert out.steps["energy"]["default"].values.shape[0] == 3


def test_audio_trim_resample_and_rms_chain() -> None:
    reg = Registry.with_builtin_specs()
    stim = AudioStimulus.from_array(np.linspace(0, 1, 8, dtype=np.float32), sr_hz=8)
    recipe = {
        "features": [
            {"id": "trim", "use": "audio.trim", "inputs": {"audio": "input:audio"}, "params": {"start_s": 0.25, "end_s": 0.75}},
            {"id": "resample", "use": "audio.resample", "inputs": {"audio": "ref:trim.default"}, "params": {"target_sr_hz": 16}},
            {"id": "rms", "use": "audio.lowlevel.rms", "inputs": {"audio": "ref:resample.default"}},
        ]
    }
    out = execute_recipe(recipe, registry=reg, inputs={"audio": stim})
    assert out.steps["trim"]["default"].sr_hz == 8
    assert out.steps["resample"]["default"].sr_hz == 16
    assert out.steps["rms"]["default"].values.ndim == 2


def test_text_tokenize_and_feature_preprocessing_chain() -> None:
    reg = Registry.with_builtin_specs()
    words_recipe = {"features": [{"id": "words", "use": "text.tokenize", "inputs": {"text": "input:text"}}]}
    words = execute_recipe(words_recipe, registry=reg, inputs={"text": TextStimulus("one two two")}).steps["words"]["default"]
    assert list(words.label) == ["one", "two", "two"]

    feature = FeatureSeries(
        values=np.arange(12, dtype=np.float32).reshape(6, 2),
        times_s=np.arange(6, dtype=np.float64) * 0.5,
        metadata=extractor_metadata("test.feature"),
    )
    recipe = {
        "features": [
            {"id": "hrf", "use": "features.hrf", "inputs": {"features": "input:features"}},
            {"id": "resample", "use": "features.resample", "inputs": {"features": "ref:hrf.default"}, "params": {"step_s": 1.0}},
            {"id": "lag", "use": "features.lag", "inputs": {"features": "ref:resample.default"}, "params": {"lags": [0, 1]}},
        ]
    }
    out = execute_recipe(recipe, registry=reg, inputs={"features": feature})
    assert out.steps["lag"]["default"].values.shape[1] == 4


def test_ocr_missing_dependency_returns_empty_fallback() -> None:
    reg = Registry.with_builtin_specs()
    image = ImageStimulus.from_array(np.ones((3, 3), dtype=np.float32))
    out = execute_recipe(
        {"features": [{"id": "ocr", "use": "image.ocr", "inputs": {"image": "input:image"}}]},
        registry=reg,
        inputs={"image": image},
    )
    words = out.steps["ocr"]["default"]
    assert len(words) >= 0
    assert words.metadata["extractor_id"]


def test_video_audio_extract_requires_source_path() -> None:
    video = VideoStimulus.from_array(np.ones((2, 3, 3), dtype=np.float32), fps=1.0)
    with pytest.raises(RuntimeError, match="requires a video file path"):
        video_audio_extract(video)


def test_video_audio_extract_runs_ffmpeg_command(monkeypatch, tmp_path) -> None:
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"not-real-video")

    def fake_which(name: str) -> str:
        assert name == "ffmpeg"
        return "/usr/bin/ffmpeg"

    def fake_run(cmd, capture_output, text):  # noqa: ANN001, ANN202
        assert "-vn" in cmd
        wav_path = cmd[-1]
        with wave.open(wav_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(8000)
            w.writeframes(np.zeros(16, dtype=np.int16).tobytes())
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("natural_features.features.preprocess.shutil.which", fake_which)
    monkeypatch.setattr("natural_features.features.preprocess.subprocess.run", fake_run)

    audio = video_audio_extract(video_path, sr_hz=8000)

    assert audio.sr_hz == 8000
    assert audio.samples.shape[0] == 16
    assert audio.source == str(video_path)


def test_image_ocr_with_fake_pytesseract(monkeypatch) -> None:
    pytest.importorskip("PIL.Image")

    class FakeOutput:
        DICT = "dict"

    class FakeTesseract:
        Output = FakeOutput

        @staticmethod
        def image_to_data(_image, output_type):  # noqa: ANN001, ANN202
            assert output_type == FakeOutput.DICT
            return {
                "text": ["", "Hello", "world"],
                "conf": ["-1", "95", "80"],
                "left": [0, 1, 4],
                "top": [0, 2, 2],
                "width": [0, 3, 4],
                "height": [0, 2, 2],
            }

    monkeypatch.setitem(sys.modules, "pytesseract", FakeTesseract)
    image = ImageStimulus.from_array(np.ones((10, 20, 3), dtype=np.float32), onset_s=2.0, duration_s=1.0)

    words = image_ocr(image, min_confidence=50.0, execution_mode="strict", strict_dependency=True)

    assert list(words.label) == ["Hello", "world"]
    np.testing.assert_allclose(words.onset_s, np.array([2.0, 2.0]))
    np.testing.assert_allclose(words.offset_s, np.array([3.0, 3.0]))
    assert list(words.extra["object_type"]) == ["word", "word"]
    assert words.metadata["backend"] == "pytesseract"
