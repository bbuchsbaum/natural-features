from __future__ import annotations

import numpy as np
import pytest

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


@pytest.mark.parametrize("policy", ["start", "center", "nearest", "overlap"])
def test_timeline_mapping_is_equivariant_to_time_origin(policy: str) -> None:
    target = Timeline(
        name="irregular",
        onset_s=np.array([0.0, 1.0, 3.0]),
        offset_s=np.array([1.0, 3.0, 4.0]),
        index=np.array([10, 20, 30]),
    )
    source_onset = np.array([0.2, 1.4, 2.8])
    source_offset = np.array([0.7, 2.6, 3.2])
    baseline = target.map_intervals(source_onset, source_offset, policy=policy)

    shift = 1_000_000.25
    shifted_target = Timeline(
        name="shifted",
        onset_s=target.onset_s + shift,
        offset_s=target.offset_s + shift,
        index=target.index,
    )
    shifted = shifted_target.map_intervals(
        source_onset + shift,
        source_offset + shift,
        policy=policy,
    )

    for key in (
        "target_position_start",
        "target_position_end",
        "target_index_start",
        "target_index_end",
    ):
        np.testing.assert_array_equal(
            shifted[key], baseline[key], err_msg=f"{policy=} changed {key} after time shift"
        )
    np.testing.assert_allclose(shifted["target_time_s"] - shift, baseline["target_time_s"])
    np.testing.assert_allclose(
        shifted["target_end_time_s"] - shift, baseline["target_end_time_s"]
    )


def test_timeline_rejects_non_finite_target_and_source_times() -> None:
    with pytest.raises(ValueError, match="finite"):
        Timeline(name="bad", onset_s=np.array([0.0, np.nan]), offset_s=np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="finite"):
        Timeline(name="bad", onset_s=np.array([0.0]), offset_s=np.array([np.inf]))

    target = Timeline(name="valid", onset_s=np.array([0.0]), offset_s=np.array([1.0]))
    with pytest.raises(ValueError, match="finite"):
        target.map_intervals(np.array([np.nan]), np.array([0.5]))


def test_empty_timeline_returns_explicit_unmapped_sentinels() -> None:
    target = Timeline(name="empty", onset_s=np.array([]), offset_s=np.array([]))
    mapping = target.map_intervals(np.array([0.0, 1.0]), np.array([0.5, 1.5]))

    np.testing.assert_array_equal(mapping["target_position_start"], [-1, -1])
    np.testing.assert_array_equal(mapping["target_position_end"], [-1, -1])
    assert mapping["target_index_start"].tolist() == [None, None]
    assert mapping["target_index_end"].tolist() == [None, None]
    assert np.isnan(mapping["target_time_s"]).all()
    assert np.isnan(mapping["target_end_time_s"]).all()


def test_timeline_from_points_uses_median_positive_step_for_last_interval() -> None:
    target = Timeline.from_points("samples", np.array([0.0, 1.0, 3.0]))
    np.testing.assert_allclose(target.onset_s, [0.0, 1.0, 3.0])
    np.testing.assert_allclose(target.offset_s, [1.0, 3.0, 4.5])
