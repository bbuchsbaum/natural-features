"""Corrected named outputs survive recipe and native storage boundaries."""
from types import SimpleNamespace

import numpy as np
import pytest

from natural_features.core.recipe import execute_recipe
from natural_features.core.registry import Registry
from natural_features.core.stimulus import AudioStimulus
from natural_features.features.audio import stem_levels
from natural_features.flow.cache import cache_fingerprint
from natural_features.storage.readers import read_feature_series
from natural_features.storage.writers import write_feature_series
from natural_features.workflows.extract_features import plan_features


@pytest.mark.parametrize("name,version,dimensions", [("tonality", "2.0", 24), ("rhythm", "3.0", 6)])
def test_window_outputs_recipe_storage_and_cache_identity(tmp_path, name, version, dimensions):
    registry = Registry.with_builtin_specs()
    feature = "audio.music." + name
    assert registry.get(feature).version == version
    recipe = {"features": [{"id": "music", "use": feature, "params": {"window_s": 1.}}]}
    stimulus = AudioStimulus.from_array(np.zeros(3000, np.float32), 1000, start_offset_s=9, clock="song")
    out = execute_recipe(recipe, registry=registry, inputs={"audio": stimulus}).steps["music"]
    assert set(out) == {"default", "diagnostics"}
    assert out["default"].values.shape[1] == dimensions
    for key, fs in out.items():
        path = write_feature_series(fs, tmp_path / key, fmt="npz")
        restored = read_feature_series(path)
        np.testing.assert_array_equal(restored.values, fs.values)
        np.testing.assert_array_equal(restored.time_bounds_s, fs.time_bounds_s)
        assert restored.clock == stimulus.clock
        assert restored.metadata["computation_context"] == fs.metadata["computation_context"]
    fingerprint = dict(extractor_name=feature, params={"window_s": 1.}, model_revision="none", upstream_ids=[])
    old = "dev" if name == "tonality" else "music-rhythm-v2"
    assert cache_fingerprint(**fingerprint, code_version=old) != cache_fingerprint(
        **fingerprint, code_version=out["default"].metadata["code_version"])


def test_new_stem_extractor_registry_recipe_and_optional_ratio(monkeypatch):
    sources = ["vocals", "drums", "bass", "other"]
    runtime = SimpleNamespace(sources=sources, versions={},
                              separate_stereo=lambda x: np.stack([x, x * .5, x * .25, x * .125], axis=1))
    monkeypatch.setattr(stem_levels, "_load_runtime", lambda *a, **k: runtime)
    recipe = {"features": [{"id": "stems", "use": "audio.stem_levels",
                             "params": {"units": "rms", "include_ratio": True}}]}
    stimulus = AudioStimulus.from_array(np.ones((22050, 2), np.float32) * .1, 44100)
    out = execute_recipe(recipe, registry=Registry.with_builtin_specs(), inputs={"audio": stimulus}).steps["stems"]
    assert out["default"].coords["feature"] == sources + ["vocals_to_accompaniment_db"]
    np.testing.assert_allclose(out["default"].values[:, -1], 20 * np.log10(1 / .875), atol=1e-6)
    assert np.all(out["diagnostics"].values[:, -1] == 1)


@pytest.mark.parametrize("name", ["tonality", "rhythm", "chord_profiles"])
def test_planner_routes_feature_consumers_to_profiles_not_diagnostics(name):
    plan = plan_features("audio", features=["audio.music." + name, "features.residualize"])
    steps = plan.to_recipe()["features"]
    assert steps[1]["inputs"] == {"features": f"ref:audio_music_{name}.default"}
