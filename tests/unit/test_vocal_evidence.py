from __future__ import annotations

import hashlib

import numpy as np
import pytest

from natural_features.core.backend_errors import BackendLoadError
from natural_features.core.feature_types import FeatureSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.core.timebase import TimebaseSpec
from natural_features.features.audio import vocal_evidence as evidence


def _pitch(context_end=1.0, *, extractor_id="pitch"):
    names = ["f0_hz", "confidence", "source_support", "source_supported", "voiced", "missing", "range_edge", "octave_jump"]
    values = np.array([[220., .9, np.nan, np.nan, 1, 0, 0, 0], [440., .9, np.nan, np.nan, 1, 0, 0, 0]])
    return FeatureSeries(values=values, times_s=np.array([.1, .2]), coords={"feature": names}, timebase=TimebaseSpec(kind="frames", reference="stimulus", hop_s=.1), metadata={"extractor_id": extractor_id, "params_hash": "pitch-a", "model_revision": "pitch-r", "computation_context": {"clock": "stimulus", "input_bounds_s": [0., context_end]}})


def _activity(context_end=2.0):
    return FeatureSeries(values=np.array([[.6, 1, 0], [.4, 1, 0]]), times_s=np.array([.1, .2]), coords={"feature": ["vocal_activity_score", "available", "padded"]}, timebase=TimebaseSpec(kind="frames", reference="stimulus", hop_s=.1), metadata={"extractor_id": "activity", "params_hash": "activity-a", "model_revision": "activity-r", "computation_context": {"clock": "stimulus", "input_bounds_s": [.05, context_end]}})


def test_pitch_with_activity_has_threshold_specific_identity_and_unions_parent_context():
    pitch, activity = _pitch(), _activity()
    low = evidence.pitch_with_activity(pitch, activity, threshold=.5)
    high = evidence.pitch_with_activity(pitch, activity, threshold=.7)
    names = list(low.coords["feature"])
    assert low.metadata["params_hash"] != high.metadata["params_hash"]
    assert low.metadata["computation_context"]["input_bounds_s"] == [0., 2.]
    assert low.values[:, names.index("voiced")].tolist() == [1., 0.]
    assert np.isnan(low.values[1, names.index("f0_hz")])
    assert low.metadata["activity_intersection"]["phonetic_support"] == "not defined or modified"


def test_pitch_with_activity_rejects_wrong_context_clock_and_binds_parent_identity():
    pitch, activity = _pitch(), _activity()
    wrong = FeatureSeries(values=activity.values, times_s=activity.times_s, coords=activity.coords, timebase=activity.timebase, metadata={**activity.metadata, "computation_context": {"clock": "other", "input_bounds_s": [0, 2]}})
    with pytest.raises(ValueError, match="output clock"):
        evidence.pitch_with_activity(pitch, wrong)
    changed_parent = evidence.pitch_with_activity(_pitch(extractor_id="pitch-v2"), activity)
    baseline = evidence.pitch_with_activity(pitch, activity)
    assert changed_parent.metadata["params_hash"] != baseline.metadata["params_hash"]


def test_pinned_asset_hash_guard_rejects_before_runtime_or_network(tmp_path, monkeypatch):
    asset = tmp_path / "rmvpe.py"
    asset.write_text("local but wrong")
    with pytest.raises(BackendLoadError, match="identity mismatch"):
        evidence._verified(asset, "0" * 64)
    expected = hashlib.sha256(asset.read_bytes()).hexdigest()
    assert evidence._verified(asset, expected) == asset.resolve()
    called = []
    monkeypatch.setattr(evidence, "_rmvpe_runtime", lambda *args: called.append(args))
    with pytest.raises(BackendLoadError):
        evidence.audio_rmvpe(AudioStimulus.from_array(np.zeros(1600), 16000), source_root=str(tmp_path))
    assert called == []
