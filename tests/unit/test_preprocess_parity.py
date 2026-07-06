from __future__ import annotations

import numpy as np
import pytest

from natural_features.core.feature_types import FeatureSeries
from natural_features.core.recipe import execute_recipe, validate_recipe
from natural_features.core.registry import Registry
from natural_features.core.stimulus import AudioStimulus, ImageStimulus, TextStimulus, VideoStimulus
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


def test_ocr_and_video_audio_placeholders_fail_clearly() -> None:
    reg = Registry.with_builtin_specs()
    image = ImageStimulus.from_array(np.ones((3, 3), dtype=np.float32))
    with pytest.raises(NotImplementedError, match="image.ocr"):
        execute_recipe({"features": [{"id": "ocr", "use": "image.ocr", "inputs": {"image": "input:image"}}]}, registry=reg, inputs={"image": image})

    video = VideoStimulus.from_array(np.ones((2, 3, 3), dtype=np.float32), fps=1.0)
    with pytest.raises(NotImplementedError, match="video.audio.extract"):
        execute_recipe(
            {"features": [{"id": "audio", "use": "video.audio.extract", "inputs": {"video": "input:video"}}]},
            registry=reg,
            inputs={"video": video},
        )
