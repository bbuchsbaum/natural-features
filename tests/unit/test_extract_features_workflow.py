from __future__ import annotations

import numpy as np
import pytest

from natural_features.core.stimulus import ImageStimulus, TextStimulus
from natural_features.workflows.extract_features import (
    ExtractFeaturesResult,
    available_features,
    extract_features,
    plan_features,
)


def test_available_features_filters_by_budget_and_modality() -> None:
    default_image = available_features(modality="image")
    default_ids = {entry.feature_id for entry in default_image}
    assert "vision.energy" in default_ids
    assert "vision.clip" not in default_ids

    python_image = available_features(modality="image", budget="allow_python")
    python_ids = {entry.feature_id for entry in python_image}
    assert {"vision.clip", "vision.face"} <= python_ids
    clip = next(entry for entry in python_image if entry.feature_id == "vision.clip")
    assert clip.requires_opt_in


def test_available_features_dataframe_output() -> None:
    pytest.importorskip("pandas")
    table = available_features(modality="audio", as_dataframe=True)
    assert {"feature_id", "default_params", "requires_opt_in", "is_public"} <= set(table.columns)
    assert "audio.rms" in set(table["feature_id"])


def test_available_features_defaults_to_public_catalog_surface() -> None:
    public_entries = available_features(modality="audio", budget="all")
    public_ids = {entry.feature_id for entry in public_entries}
    all_ids = {entry.feature_id for entry in available_features(modality="audio", budget="all", public_only=False)}

    assert all(entry.is_public for entry in public_entries)
    assert "audio.rms" in public_ids
    assert "audio.lowlevel.rms" not in public_ids
    assert "audio.lowlevel.rms" in all_ids
    assert public_ids < all_ids


def test_plan_features_routes_image_features_to_image_input() -> None:
    image = ImageStimulus.from_array(np.ones((4, 5, 3), dtype=np.float32))
    plan = plan_features(
        image,
        features=["vision.clip", "vision.face"],
        budget="allow_python",
        feature_params={"vision.clip": {"dim": 8}},
    )

    assert [row.feature_id for row in plan.rows] == ["vision.clip", "vision.face"]
    assert [row.input_key for row in plan.rows] == ["image", "image"]
    recipe = plan.to_recipe()
    assert recipe["features"][0]["inputs"] == {"image": "input:image"}
    assert recipe["features"][0]["params"]["dim"] == 8


def test_plan_features_bundle_prefers_public_aliases() -> None:
    plan = plan_features("audio", bundle="baseline")
    ids = [row.feature_id for row in plan.rows]
    assert {"audio.rms", "audio.mel", "audio.spectral_stats"} <= set(ids)
    assert "audio.mfcc" not in ids
    assert not any(feature_id.startswith("audio.lowlevel.") for feature_id in ids)


def test_default_budget_rejects_opt_in_features() -> None:
    image = ImageStimulus.from_array(np.ones((4, 5, 3), dtype=np.float32))
    with pytest.raises(PermissionError, match="requires opt-in"):
        plan_features(image, features="vision.clip")


def test_extract_features_executes_image_fallbacks_from_path(tmp_path) -> None:
    pil = pytest.importorskip("PIL.Image")
    path = tmp_path / "image.png"
    data = np.full((4, 5, 3), 128, dtype=np.uint8)
    pil.fromarray(data).save(path)

    result = extract_features(
        path,
        features=["vision.clip", "vision.face"],
        budget="allow_python",
        feature_params={"vision.clip": {"dim": 8}},
    )

    assert isinstance(result, ExtractFeaturesResult)
    assert result.features["vision.clip"].values.shape == (1, 8)
    assert result.features["vision.face"].values.shape[0] == 1
    assert result.plan.input_modalities == ["image"]


def test_extract_features_text_preprocessing_chain() -> None:
    result = extract_features(
        TextStimulus("one two two"),
        features=["text.tokenize", "language.surface"],
    )

    assert isinstance(result, ExtractFeaturesResult)
    words = result.features["text.tokenize"]
    surface = result.features["language.surface"]
    assert list(words.label) == ["one", "two", "two"]
    assert surface.values.shape == (3, 5)
    assert result.recipe["features"][1]["inputs"] == {"words": "ref:text_tokenize.default"}


def test_extract_features_table_output_includes_features_and_objects() -> None:
    pytest.importorskip("pandas")
    table = extract_features(
        "one two two",
        features=["text.tokenize", "language.surface"],
        format="table",
    )

    assert {"output_type", "output_name"} <= set(table.columns)
    assert {"features", "objects"} <= set(table["output_type"])
    assert "object_id" in table.columns
    assert list(table.loc[table["output_name"] == "text.tokenize", "object_id"]) == [
        "word_0001",
        "word_0002",
        "word_0003",
    ]


def test_recipe_format_plans_ocr_chain_without_executing() -> None:
    image = ImageStimulus.from_array(np.ones((3, 3), dtype=np.float32))
    recipe = extract_features(
        image,
        features=["image.ocr", "language.surface"],
        budget="all",
        format="recipe",
    )

    assert [step["use"] for step in recipe["features"]] == ["image.ocr", "language.surface"]
    assert recipe["features"][1]["inputs"] == {"words": "ref:image_ocr.default"}
