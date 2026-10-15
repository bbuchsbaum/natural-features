from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from natural_features.core.feature_types import FeatureSeries
from natural_features.core.timebase import TimebaseSpec, TemporalContext


def _comparison_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "compare_vocal_backends.py"
    spec = importlib.util.spec_from_file_location("comparison_runner_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


def test_cached_feature_requires_exact_variant_key_and_preserves_rendering_identity(tmp_path):
    runner = _comparison_module()
    feature = FeatureSeries(values=np.array([[1.]]), times_s=np.array([.1]), coords={"feature": ["x"]}, timebase=TimebaseSpec(kind="frames", hop_s=.1), temporal_context=TemporalContext(), metadata={"extractor_id": "toy", "params_hash": "p"})
    path = tmp_path / "cached.npz"
    np.savez(path, cache_key="variant-a", values=feature.values, times_s=feature.times_s, bounds=feature.temporal_bounds_s, feature_names_json=json.dumps(["x"]), source_metadata_json=json.dumps(feature.metadata), timebase_json=json.dumps(feature.timebase.to_dict()), temporal_context_json=json.dumps(feature.temporal_context.to_dict()))
    restored = runner.load_saved(path, "variant-a")
    assert restored.metadata == feature.metadata
    assert restored.timebase.hop_s == .1
    with pytest.raises(ValueError, match="cache identity mismatch"):
        runner.load_saved(path, "variant-b")
