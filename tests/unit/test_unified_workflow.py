from __future__ import annotations

import numpy as np

import natural_features as nf
from natural_features.core.feature_types import EventSeries, FeatureSeries
from natural_features.core.stimulus import TextStimulus, VideoStimulus
from natural_features.core.timeline import Timeline, align_feature_to_timeline
from natural_features.features.common import extractor_metadata
from natural_features.workflows.extract_features import extract_features


def test_timeline_aligns_dense_features_to_event_timeline() -> None:
    words = EventSeries(
        onset_s=np.array([0.0, 1.0, 2.0], dtype=np.float64),
        offset_s=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        label=np.array(["one", "two", "three"], dtype=object),
        metadata=extractor_metadata("test.words"),
    )
    surface = FeatureSeries(
        values=np.arange(6, dtype=np.float32).reshape(3, 2),
        times_s=np.array([0.0, 1.0, 2.0], dtype=np.float64),
        coords={"feature": ["a", "b"]},
        metadata=extractor_metadata("test.surface"),
    )

    target = Timeline.from_feature("words", words)
    alignment = align_feature_to_timeline("surface", surface, target)

    assert alignment.mapping["target_index_start"].tolist() == [0, 1, 2]
    assert alignment.mapping["target_index_end"].tolist() == [0, 1, 2]
    assert alignment.to_rows()[1]["target_name"] == "words"


def test_extract_features_result_aligns_language_features_to_words() -> None:
    result = extract_features(
        TextStimulus("one two two"),
        features=["text.tokenize", "language.surface"],
    )

    assert isinstance(result, nf.ExtractFeaturesResult)
    aligned = result.align_to("text.tokenize", features=["language.surface"])
    rows = aligned.to_rows()

    assert isinstance(aligned, nf.AlignedFeatureSet)
    assert [row["target_index_start"] for row in rows] == [0, 1, 2]
    assert [row["source_name"] for row in rows] == ["language.surface"] * 3
    table = result.to_table(include_metadata=False)
    assert {"text.tokenize", "language.surface"} <= set(table["output_name"])


def test_extract_features_result_aligns_video_features_to_frame_timeline() -> None:
    frames = np.ones((4, 5, 5, 3), dtype=np.float32)
    video = VideoStimulus.from_array(frames, fps=2.0)

    result = extract_features(video, features=["vision.energy"])
    aligned = result.align_to("video_frames", features="vision.energy")

    assert "video_frames" in result.timelines
    assert aligned.target.kind == "frames"
    assert aligned.alignments["vision.energy"].mapping["target_index_start"].tolist() == [0, 1, 2, 3]


def test_aligned_feature_set_can_annotate_event_outputs() -> None:
    result = extract_features(
        TextStimulus("alpha beta"),
        features=["text.tokenize"],
    )

    annotated = result.align_to("text.tokenize", features="text.tokenize").annotated_features(prefix="word")
    words = annotated["text.tokenize"]

    assert isinstance(words, EventSeries)
    assert words.extra["word_index_start"].tolist() == [0, 1]
    assert words.extra["word_index_end"].tolist() == [0, 1]
