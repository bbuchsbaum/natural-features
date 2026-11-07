from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys
import wave

import numpy as np
import pytest

from natural_features.core.stimulus import AudioStimulus
from natural_features.features.speech.phone_sequences import ItemFeatures
from natural_features.workflows.sentence_comparison import (
    AcousticConfig,
    FeatureScaler,
    combine_scaled_blocks,
    extract_sentence_acoustics,
    feature_distances,
    overlap_weights,
)

_spec = importlib.util.spec_from_file_location(
    "phonology_comparison_example",
    Path(__file__).resolve().parents[2] / "examples/phonology_comparison.py",
)
example = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = example
_spec.loader.exec_module(example)


def tone(duration=2.5, gain=0.2, frequency=180):
    t = np.arange(round(duration * 16000)) / 16000
    # Amplitude modulation and harmonic content avoid a degenerate MPS example.
    x = gain * (
        (0.7 + 0.3 * np.sin(2 * np.pi * 4 * t))
        * (
            np.sin(2 * np.pi * frequency * t)
            + 0.3 * np.sin(2 * np.pi * frequency * 2 * t)
        )
        + 0.01 * np.random.default_rng(43).normal(size=len(t))
    )
    return AudioStimulus.from_array(x, 16000)


def table(values, valid=None, names=("x", "y"), ids=("a", "b", "c")):
    values = np.asarray(values, float)
    return ItemFeatures(
        ids, names, values, np.isfinite(values) if valid is None else valid, {}
    )


def test_overlap_weights_analytic_union_and_clipping():
    np.testing.assert_allclose(
        overlap_weights(np.array([[0, 2], [1, 3]]), start=0, stop=3), [1.5, 1.5]
    )
    np.testing.assert_allclose(
        overlap_weights(np.array([[0, 2], [1, 3]]), start=1, stop=2), [0.5, 0.5]
    )
    np.testing.assert_allclose(
        overlap_weights(np.array([[0, 1], [2, 4]]), start=0, stop=4), [1, 2]
    )
    np.testing.assert_allclose(
        overlap_weights(np.array([[0, 1], [0, 1]]), start=0, stop=1), [0.5, 0.5]
    )


@pytest.mark.smoke
def test_native_support_short_clips_silence_and_pitch():
    clips = {
        "silence": AudioStimulus.from_array(np.zeros(40000), 16000),
        "short": tone(0.01),
        "tone": tone(),
    }
    result = extract_sentence_acoustics(
        clips, config=AcousticConfig(include_pitch=True)
    )
    assert not result.blocks["spectral_shape"].eligible[0]
    assert not result.blocks["modulation"].eligible[:2].any()
    assert not result.blocks["spectral_shape"].eligible[1]
    assert result.blocks["modulation"].eligible[2]
    assert not result.blocks["pitch"].valid[0, 0]  # zero Hz must not enter the mean
    assert np.isnan(result.blocks["pitch"].values[0, 0])
    assert result.blocks["controls"].values[0, 1] == 0
    assert result.blocks["controls"].values[2, 2] > 0.9
    native = result.native["tone"]["modulation"]
    np.testing.assert_allclose(native.series.time_bounds_s, [[0, 2.02]])
    assert native.coverage_s == pytest.approx(2.02)
    assert (
        result.native["short"]["spectral_raw"].series.metadata["right_padding_samples"]
        == 240
    )


def test_changed_gain_shape_vs_level_and_duration():
    result = extract_sentence_acoustics(
        {"a": tone(), "b": tone(gain=0.4), "long": tone(3.5)}
    )
    # The expected log-power shift is 2 log10(2); floors limit invariance at tiny energy.
    raw = result.blocks["spectral_raw"].values
    np.testing.assert_allclose(raw[1, :32] - raw[0, :32], 2 * np.log10(2), atol=2e-5)
    shape = result.blocks["spectral_shape"].values
    np.testing.assert_allclose(shape[0], shape[1], atol=2e-5)
    controls = result.blocks["controls"].values
    assert controls[1, 1] == pytest.approx(controls[0, 1] * 2, rel=1e-6)
    assert controls[2, 0] == 3.5
    assert (
        result.native["long"]["modulation"].coverage_s
        > result.native["a"]["modulation"].coverage_s
    )


def test_origin_and_sample_rounded_clocks_and_no_silent_resample():
    original = tone()
    shifted = AudioStimulus(original.samples, 16000, start_offset_s=9.5, clock="audio")
    cfg = AcousticConfig(hop_s=0.01003, coch_hop_s=0.00503)
    result = extract_sentence_acoustics({"x": shifted}, config=cfg)
    native = result.native["x"]["spectral_raw"]
    assert str(native.series.clock) == "audio"
    assert native.series.time_bounds_s[0, 0] == 9.5
    assert native.series.time_bounds_s[1, 0] == pytest.approx(
        9.5 + round(0.01003 * 16000) / 16000
    )
    with pytest.raises(ValueError, match="resample"):
        extract_sentence_acoustics({"x": AudioStimulus(original.samples, 22050)})


def test_scaler_fit_is_frozen_schema_persisted_and_missing_masked():
    source = table([[0, 4], [2, 4], [100, 9]])
    scaler = FeatureScaler.fit(source, reference_ids=["a", "b"])
    assert scaler.retained == (0,)
    scaled = scaler.transform(source)
    np.testing.assert_allclose(scaled.values[:, 0], [-1, 1, 99])
    restored = FeatureScaler.from_dict(json.loads(json.dumps(scaler.to_dict())))
    np.testing.assert_array_equal(restored.transform(source).values, scaled.values)
    with pytest.raises(ValueError, match="schema"):
        scaler.transform(table([[0, 4], [2, 4], [100, 9]], names=("y", "x")))
    incomplete = table([[0, 4], [2, 4], [np.nan, 9]])
    distances = feature_distances(scaler.transform(incomplete), metric="rms_euclidean")
    assert distances[0, 1] == 2 and np.isnan(distances[0, 2])
    assert FeatureScaler.fit(
        table([[0, np.nan], [2, 4], [3, 5]]), reference_ids=["a", "b"]
    ).retained == (0,)


def test_distance_zero_vectors_and_block_dimension_balance():
    features = table([[1, 0], [0, 1], [0, 0]])
    d = feature_distances(features)
    assert d[0, 0] == 0 and d[0, 1] == 1
    assert np.isnan(d[2]).all()
    one = table([[0], [1], [2]], names=("one",))
    many = table([[0] * 8, [1] * 8, [2] * 8], names=tuple(map(str, range(8))))
    a = FeatureScaler.fit(one, reference_ids=one.item_ids).transform(one)
    b = FeatureScaler.fit(many, reference_ids=many.item_ids).transform(many)
    combined = combine_scaled_blocks({"a": a, "b": b}, weights={"a": 1.0, "b": 1.0})
    np.testing.assert_allclose(
        feature_distances(combined, metric="euclidean"),
        feature_distances(a, metric="rms_euclidean"),
    )
    with pytest.raises(ValueError, match="explicit"):
        combine_scaled_blocks({"a": a}, weights={})


def test_invalid_configs_and_inputs_fail():
    with pytest.raises(ValueError):
        AcousticConfig(temporal_edges=(1, 1, 2))
    with pytest.raises(ValueError):
        AcousticConfig(fmax=9000)
    with pytest.raises(ValueError, match="finite"):
        extract_sentence_acoustics({"x": AudioStimulus(np.array([np.nan]), 16000)})


def test_pair_restrictions_common_support_and_rank_ties():
    ids = ("a", "b", "c", "d")
    pairs = example.analysis_pairs(
        ids, groups={"a": "r1", "b": "r1", "c": "r2", "d": "r2"}, within_group=True
    )
    assert pairs == [(0, 1), (2, 3)]
    with pytest.raises(ValueError, match="duplicate"):
        example.analysis_pairs(ids, explicit=[("a", "b"), ("b", "a")])
    np.testing.assert_array_equal(
        example.ranks(np.array([4, 2, 2, 8])), [3, 1.5, 1.5, 4]
    )
    a = np.array([[0, 1, 3, 5], [1, 0, 2, 4], [3, 2, 0, 6], [5, 4, 6, 0]], float)
    b = a * 2
    b[0, 2] = b[2, 0] = np.nan
    all_pairs = example.analysis_pairs(ids)
    audit = example.audit_rdms(
        {"a": example.DistanceMatrix(ids, a), "b": example.DistanceMatrix(ids, b)},
        pairs=all_pairs,
    )
    assert audit["matched_pairs"] == 5
    assert audit["pairs"][1]["excluded_by"] == ["b"]
    assert audit["comparisons"]["a vs b"]["pearson"]["raw"] == pytest.approx(1)
    assert audit["comparisons"]["a vs b"]["spearman"]["raw"] == pytest.approx(1)
    with pytest.raises(ValueError, match="order"):
        example.audit_rdms(
            {
                "a": example.DistanceMatrix(ids, a),
                "b": example.DistanceMatrix(ids[::-1], b),
            },
            pairs=all_pairs,
        )


def test_conditional_correlation_against_balanced_group_contrast():
    # Six pair observations; the nuisance is constant within each triple.
    # Removing it leaves [1,-1,0,1,-1,0] and [1,-1,0,-1,1,0], orthogonal.
    ids = ("a", "b", "c", "d")
    pairs = example.analysis_pairs(ids)
    nuisance = np.array([0, 0, 0, 10, 10, 10.0])
    x = 20 + nuisance + [1, -1, 0, 1, -1, 0]
    y = 20 + nuisance + [1, -1, 0, -1, 1, 0]
    matrices = {}
    for name, values in (("x", x), ("y", y)):
        matrix = np.zeros((4, 4))
        for (i, j), value in zip(pairs, values):
            matrix[i, j] = matrix[j, i] = value
        matrices[name] = example.DistanceMatrix(ids, matrix)
    audit = example.audit_rdms(matrices, pairs=pairs, controls={"nuisance": nuisance})
    corr = audit["comparisons"]["x vs y"]["pearson"]
    assert corr["raw"] > 0.95
    assert corr["conditional"] == pytest.approx(0, abs=1e-12)
    assert audit["control_design"]["pearson"]["rank"] == 2
    degenerate = example.audit_rdms(
        matrices, pairs=pairs, controls={"x": x, "duplicate": x}
    )
    assert degenerate["control_design"]["pearson"]["rank_deficient"]
    assert degenerate["comparisons"]["x vs y"]["pearson"]["conditional"] is None
    json.dumps(degenerate, allow_nan=False)


def test_no_matched_pairs_and_no_p_values():
    missing = example.DistanceMatrix(("a", "b"), np.full((2, 2), np.nan))
    audit = example.audit_rdms({"x": missing, "y": missing}, pairs=[(0, 1)])
    assert audit["matched_pairs"] == 0
    assert audit["comparisons"]["x vs y"]["pearson"]["raw"] is None
    assert audit["comparisons"]["x vs y"]["pearson"]["conditional"] is None
    json.dumps(audit, allow_nan=False)


def test_end_to_end_exports_and_embedding_identity(tmp_path):
    items = []
    for i, text in enumerate(("cat tack", "tack cat", "the cat", "a tack")):
        audio = tone(2.1 + i * 0.1, frequency=140 + i * 30)
        path = tmp_path / f"{i}.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            wav.writeframes((audio.samples * 32767).astype("<i2").tobytes())
        items.append(
            {"item_id": f"s{i}", "text": text, "audio": path.name, "group": "one"}
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"items": items}))
    out = tmp_path / "result"
    report = example.run_comparison(
        manifest,
        out,
        selection="dictionary_order",
        reference_ids=[i["item_id"] for i in items],
        within_group=True,
    )
    assert report["matched_pairs"] == 6
    assert (out / "articulatory_sequences.json").is_file()
    assert json.loads((out / "coverage.json").read_text())["eligible_items"] == 4
    restored = ItemFeatures.load(out / "bigram_within")
    assert restored.item_ids == ("s0", "s1", "s2", "s3")
    with np.load(out / "native_0000_modulation.npz", allow_pickle=False) as data:
        assert data["bounds_s"].shape[1] == 2
        assert data["values"].shape == data["valid"].shape
    with pytest.raises(FileExistsError):
        example.run_comparison(
            manifest, out, selection="dictionary_order", reference_ids=["s0", "s1"]
        )
    embedding = tmp_path / "embedding.npz"
    np.savez(
        embedding, item_ids=np.array([i["item_id"] for i in items]), values=np.eye(4)
    )
    embedding_meta = tmp_path / "embedding.json"
    embedding_meta.write_text(
        json.dumps(
            {
                "model": "synthetic_test_fixture",
                "revision": "test",
                "text_sha256": {"s0": "wrong"},
            }
        )
    )
    with pytest.raises(ValueError, match="hashes"):
        example.run_comparison(
            manifest,
            tmp_path / "mismatch",
            selection="dictionary_order",
            reference_ids=["s0", "s1"],
            embeddings=embedding,
            embedding_metadata=embedding_meta,
        )
    assert not (tmp_path / "mismatch").exists()
