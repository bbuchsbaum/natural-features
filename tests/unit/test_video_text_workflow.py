from __future__ import annotations

import csv
import json

import numpy as np

from natural_features.cli.main import main as cli_main
from natural_features.core.feature_types import EventSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.features.common import extractor_metadata
from natural_features.workflows.extract_features import plan_features
from natural_features.workflows.video_text import VideoTextResult, extract_video_text


def _word_events(*, shifted: float = 0.0) -> EventSeries:
    return EventSeries(
        onset_s=np.array([2.05 + shifted, 2.55 + shifted], dtype=np.float64),
        offset_s=np.array([2.25 + shifted, 2.95 + shifted], dtype=np.float64),
        label=np.array(["hello", "world"], dtype=object),
        confidence=np.array([0.9, 0.8], dtype=np.float32),
        metadata=extractor_metadata("test.asr", {"shifted": shifted}),
    )


def _segment_events() -> EventSeries:
    return EventSeries(
        onset_s=np.array([2.0], dtype=np.float64),
        offset_s=np.array([4.0], dtype=np.float64),
        label=np.array(["hello world"], dtype=object),
        confidence=np.array([1.0], dtype=np.float32),
        metadata=extractor_metadata("test.asr.segment"),
    )


def test_extract_video_text_aligns_asr_words_to_source_frames(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_audio_extract(video, **kwargs):
        calls["video"] = video
        calls["audio_kwargs"] = kwargs
        return AudioStimulus.from_array(
            np.zeros(20, dtype=np.float32), sr_hz=10, start_offset_s=kwargs["start_s"]
        )

    def fake_transcribe(audio, **kwargs):
        calls["asr_kwargs"] = kwargs
        return {
            "segments": _segment_events(),
            "words": _word_events(),
            "qc": {"mode": "fake_asr", "fallback_used": False},
        }

    monkeypatch.setattr(
        "natural_features.workflows.video_text.video_audio_extract", fake_audio_extract
    )
    monkeypatch.setattr(
        "natural_features.workflows.video_text.whisper_transcribe", fake_transcribe
    )

    result = extract_video_text(
        "movie.mp4",
        align="none",
        asr_model="tiny",
        start_s=2.0,
        duration_s=2.0,
        video_fps=10.0,
    )

    assert calls["video"] == "movie.mp4"
    assert calls["asr_kwargs"]["model"] == "tiny"
    assert result.asr_qc["mode"] == "fake_asr"
    assert result.align_qc["mode"] == "none"
    assert result.frame_qc["mode"] == "provided"
    assert result.words.extra["frame_start"].tolist() == [20, 25]
    assert result.words.extra["frame_end"].tolist() == [22, 29]
    assert result.words.extra["text_source"].tolist() == ["speech", "speech"]
    assert result.words.metadata["workflow_name"] == "video_text"


def test_extract_video_text_can_refine_word_alignment(monkeypatch) -> None:
    def fake_audio_extract(_video, **kwargs):
        return AudioStimulus.from_array(
            np.zeros(20, dtype=np.float32), sr_hz=10, start_offset_s=kwargs["start_s"]
        )

    def fake_transcribe(_audio, **_kwargs):
        return {
            "segments": _segment_events(),
            "words": _word_events(),
            "qc": {"mode": "fake_asr", "fallback_used": False},
        }

    def fake_align(_audio, words, **kwargs):
        assert kwargs["backend"] == "whisperx"
        assert len(words) == 2
        return {
            "words": _word_events(shifted=0.01),
            "qc": {"mode": "whisperx", "fallback_used": False, "n_words": 2},
        }

    monkeypatch.setattr(
        "natural_features.workflows.video_text.video_audio_extract", fake_audio_extract
    )
    monkeypatch.setattr(
        "natural_features.workflows.video_text.whisper_transcribe", fake_transcribe
    )
    monkeypatch.setattr(
        "natural_features.workflows.video_text.whisperx_align", fake_align
    )

    result = extract_video_text(
        "movie.mp4",
        align="whisperx",
        start_s=2.0,
        duration_s=2.0,
        video_fps=10.0,
    )

    assert result.align_qc["mode"] == "whisperx"
    assert np.allclose(result.words.onset_s, [2.06, 2.56])
    assert result.words.extra["frame_start"].tolist() == [20, 25]


def test_video_speech_words_is_plan_visible() -> None:
    plan = plan_features(
        "video", features=["video.speech.words"], budget="allow_python"
    )

    assert [row.feature_id for row in plan.rows] == ["video.speech.words"]
    assert plan.rows[0].input_key == "video"
    assert plan.rows[0].params["frame_policy"] == "overlap"


def test_video_text_cli_writes_summary_and_word_table(monkeypatch, tmp_path) -> None:
    words = EventSeries(
        onset_s=np.array([0.1, 0.4], dtype=np.float64),
        offset_s=np.array([0.2, 0.55], dtype=np.float64),
        label=np.array(["alpha", "beta"], dtype=object),
        confidence=np.array([0.7, 0.8], dtype=np.float32),
        extra={
            "frame_start": np.array([1, 4], dtype=np.int64),
            "frame_end": np.array([2, 5], dtype=np.int64),
            "text_source": np.array(["speech", "speech"], dtype=object),
        },
        metadata=extractor_metadata("test.video_text"),
    )
    result = VideoTextResult(
        words=words,
        segments=_segment_events(),
        frame_timeline=None,
        asr_qc={"mode": "fake_asr", "fallback_used": False},
        align_qc={"mode": "none", "fallback_used": False},
        frame_qc={"mode": "provided", "fallback_used": False},
        source_video="movie.mp4",
    )
    captured: dict[str, object] = {}

    def fake_extract_video_text(video, **kwargs):
        captured["video"] = video
        captured["kwargs"] = kwargs
        return result

    monkeypatch.setattr(
        "natural_features.cli.main.extract_video_text", fake_extract_video_text
    )
    table_out = tmp_path / "words.csv"
    json_out = tmp_path / "summary.json"

    rc = cli_main(
        [
            "video-text",
            "movie.mp4",
            "--align-backend",
            "none",
            "--video-fps",
            "10",
            "--table-out",
            str(table_out),
            "--out-json",
            str(json_out),
            "--json",
        ]
    )

    assert rc == 0
    assert captured["video"] == "movie.mp4"
    assert captured["kwargs"]["video_fps"] == 10.0
    assert json.loads(json_out.read_text(encoding="utf-8"))["n_words"] == 2
    with table_out.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["label"] == "alpha"
    assert rows[0]["frame_start"] == "1"
