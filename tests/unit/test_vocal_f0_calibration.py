from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np


def _mod():
    p = Path(__file__).resolve().parents[2] / "scripts/calibrate_vocal_f0.py"
    s = importlib.util.spec_from_file_location("cal", p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def _fs():
    from natural_features.core.feature_types import FeatureSeries

    names = [
        "f0_hz",
        "confidence",
        "source_support",
        "source_supported",
        "voiced",
        "missing",
        "range_edge",
        "octave_jump",
        "padded",
        "candidate_f0_hz",
        "source_rms",
    ]
    x = np.zeros((3, len(names)))
    x[:, 1] = [0.21, 0.5, 0.8]
    x[:, 2] = 0.1
    x[:, 9] = 220
    x[:, 10] = 0.001
    return FeatureSeries(
        x,
        np.array([0.0, 0.01, 0.02]),
        coords={"feature": names},
        metadata={
            "extractor_id": "x",
            "params_hash": "x",
            "computation_context": {"input_bounds_s": [0.0, 0.02]},
        },
    )


def test_replay_threshold_and_end_center_rejection():
    m = _mod()
    out = m.replay(_fs(), 0.5, 0.1, 0.001)
    np.testing.assert_allclose(
        out.values[:, 0], [np.nan, 220.0, np.nan], equal_nan=True
    )


def test_replay_uses_clock_tagged_input_end():
    from dataclasses import replace

    m = _mod()
    source = _fs()
    shifted = replace(
        source,
        times_s=source.times_s + 7,
        metadata={
            **source.metadata,
            "computation_context": {"input_bounds_s": [7.0, 7.02]},
        },
    )
    out = m.replay(shifted, 0.5, 0.1, 0.001)
    np.testing.assert_allclose(
        out.values[:, 0], [np.nan, 220.0, np.nan], equal_nan=True
    )


def test_calibration_negatives_use_native_grid(monkeypatch, tmp_path):
    m = _mod()
    monkeypatch.setattr(
        m, "_reference", lambda *a: (np.array([0.0, 0.006, 0.012, 0.018]), np.zeros(4))
    )
    out = m.replay(_fs(), 0.21, 0.1, 0.001)
    metrics = m._score(m._runner(), tmp_path, 1, "backing", out)
    assert metrics["n_reference"] == 2
    assert metrics["voicing_false_alarm_rate"] == 1.0


def test_select_is_deterministic_and_no_eligible_is_none():
    m = _mod()
    assert m.select([]) is None
    rows = [
        {
            "eligible": True,
            "tuple": [0.21, 0.01, 0.001],
            "objective": {"min_rpa": 0.8, "max_fa": 0.1, "backing_fa": 0.01},
        },
        {
            "eligible": True,
            "tuple": [0.8, 0.1, 0.01],
            "objective": {"min_rpa": 0.8, "max_fa": 0.1, "backing_fa": 0.01},
        },
    ]
    assert m.select(rows)["tuple"] == [0.8, 0.1, 0.01]


def test_replay_matches_direct_extraction_all_frozen_thresholds(monkeypatch):
    from types import SimpleNamespace
    from natural_features.core.stimulus import AudioStimulus
    from natural_features.features.audio import vocal_f0 as vf

    m = _mod()
    # Global mean zero, but window means are nonzero. Frame demeaning would
    # erase valid gate energy. Confidence lies at every frozen boundary.
    n = 4410
    stem = np.where(np.arange(n) < n // 2, 0.002, -0.002)
    stem += 1e-4 * np.sin(2 * np.pi * 220 * np.arange(n) / 44100)

    def separate(x):
        return np.column_stack([stem * 3, stem])

    def pitch(x, hop, *args):
        count = 1 + len(x) // hop
        return np.full(count, 220, dtype=np.float32), np.resize(
            np.asarray(m.GRID[0], dtype=np.float32), count
        )

    rt = SimpleNamespace(
        separate=separate, pitch=pitch, sources=["other", "vocals"], versions={}
    )
    monkeypatch.setattr(vf, "_load_runtime", lambda *a: rt)
    audio = AudioStimulus.from_array(
        stem.astype(np.float32), 44100, start_offset_s=7.25
    )
    raw = vf.audio_vocal_f0(audio)
    centers = np.arange(len(raw.values)) * 160
    exact_rms = vf._frame_rms(vf._resample(stem, 44100, 16000), centers)
    np.testing.assert_array_equal(raw.values[:, 10], exact_rms)
    assert np.max(exact_rms) > 0.0019
    for c in m.GRID[0]:
        for s in m.GRID[1]:
            for r in m.GRID[2]:
                direct = vf.audio_vocal_f0(
                    audio, confidence_threshold=c, source_threshold=s, min_source_rms=r
                )
                replayed = m.replay(raw, c, s, r)
                np.testing.assert_array_equal(replayed.values, direct.values)


def test_gate_precision_boundaries():
    from natural_features.features.audio.vocal_f0 import _acceptance_masks

    source, voiced = _acceptance_masks(
        np.array([0.21], dtype=np.float32),
        np.array([0.1]),
        np.array([0.001]),
        np.array([True]),
        0.21,
        0.1,
        0.001,
    )
    assert source[0] and voiced[0]
    source, _ = _acceptance_masks(
        np.array([0.21], dtype=np.float32),
        np.nextafter(np.array([0.1]), 0),
        np.array([0.001]),
        np.array([True]),
        0.21,
        0.1,
        0.001,
    )
    assert not source[0]


def test_temporal_replay_matches_direct_and_rebuilds_original_gate(monkeypatch):
    from types import SimpleNamespace
    from natural_features.core.stimulus import AudioStimulus
    from natural_features.features.audio import vocal_f0 as vf
    m = _mod()
    def separate(x):
        return np.column_stack([x[:, 0] * .5, x[:, 0] * .5])
    def pitch(x, hop, *args):
        n = 1 + len(x) // hop
        confidence = np.full(n, .9, dtype=np.float32)
        confidence[10:12] = .5
        return np.where(np.arange(n) < 30, 220., 440.), confidence
    monkeypatch.setattr(vf, '_load_runtime', lambda *a: SimpleNamespace(
        separate=separate, pitch=pitch, sources=['other', 'vocals'], versions={}))
    samples = (.1 * np.sin(2 * np.pi * 220 * np.arange(16000) / 16000)).astype(np.float32)
    audio = AudioStimulus.from_array(samples, 16000, start_offset_s=7.25)
    for hop in (.01, .0073):
        direct = vf.audio_vocal_f0(audio, hop_s=hop, voicing_guard_s=.02)
        replayed = m.replay(direct, .65, .03, .01, voicing_guard_s=.02)
        np.testing.assert_array_equal(replayed.values, direct.values)
        unguarded = m.replay(direct, .65, .03, .01)
        np.testing.assert_array_equal(unguarded.values, vf.audio_vocal_f0(audio, hop_s=hop).values)


def test_temporal_artifact_keys_do_not_overwrite_repeated_inputs(monkeypatch, tmp_path):
    import json
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / 'scripts'))
    from evaluate_vocal_f0_temporal import _replay_cache_key
    source = {'artifact_sha256': 'same-native-input', 'cached': False}
    hashes = {k: 'frozen' for k in ('protocol_sha256', 'extractor_source_sha256',
                                   'replay_source_sha256', 'evaluation_source_sha256')}
    first = _replay_cache_key(source, hashes, 'consumed_v2_test', {'excerpt_id': 7, 'condition': 'silence'})
    second = _replay_cache_key({**source, 'cached': True}, hashes, 'consumed_v2_test', {'excerpt_id': 12, 'condition': 'silence'})
    # A second row's provenance must never mutate the file hashed by the first.
    (tmp_path / first).write_text(json.dumps(source))
    (tmp_path / second).write_text(json.dumps({**source, 'cached': True}))
    assert first != second
    assert json.loads((tmp_path / first).read_text())['cached'] is False
    assert _replay_cache_key(source, hashes, 'consumed_v2_test', {'excerpt_id': 7, 'condition': 'silence'}) == first
