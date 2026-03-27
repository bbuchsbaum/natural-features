from __future__ import annotations

from datetime import datetime, timezone
import json

import numpy as np
import pytest

from natural_features.core.feature_types import EventSeries, FeatureSeries
from natural_features.storage.catalog import Catalog
from natural_features.storage.readers import read_event_series, read_feature_series


def _meta() -> dict[str, str]:
    return {"extractor_id": "vision.lowlevel.visual_energy", "params_hash": "abc123"}


def test_catalog_put_and_list_feature_npz(tmp_path) -> None:
    cat = Catalog(tmp_path / "catalog")
    fs = FeatureSeries(
        values=np.ones((4, 2), dtype=np.float32),
        times_s=np.array([0.0, 0.1, 0.2, 0.3]),
        metadata=_meta(),
    )
    rec = cat.put(
        fs,
        run_id="run-1",
        stage_id="stage-1",
        code_version="git:deadbeef",
        model_name="none",
        model_revision="none",
        created_at=datetime.now(timezone.utc).isoformat(),
        preferred_format="npz",
    )
    assert rec.artifact_id
    listed = cat.list_artifacts(stage_id="stage-1")
    assert len(listed) == 1
    obj = read_feature_series((cat.root / listed[0].path))
    np.testing.assert_allclose(obj.values, fs.values)


def test_catalog_put_event_npz(tmp_path) -> None:
    cat = Catalog(tmp_path / "catalog")
    es = EventSeries(
        onset_s=np.array([0.1, 0.5]),
        offset_s=np.array([0.2, 0.7]),
        label=np.array(["a", "b"]),
        metadata=_meta(),
    )
    rec = cat.put(
        es,
        run_id="run-2",
        stage_id="stage-2",
        code_version="git:cafebabe",
        created_at=datetime.now(timezone.utc).isoformat(),
        preferred_format="npz",
    )
    loaded = read_event_series(cat.root / rec.path)
    assert len(loaded) == 2


def test_manifest_exports_payload_and_object_provenance(tmp_path) -> None:
    cat = Catalog(tmp_path / "catalog")
    fs = FeatureSeries(
        values=np.ones((3, 2), dtype=np.float32),
        times_s=np.array([0.0, 0.5, 1.0]),
        metadata={"extractor_id": "x", "params_hash": "y", "model_revision": "m1"},
    )
    rec = cat.put(
        fs,
        run_id="r1",
        stage_id="s1",
        code_version="dev",
        created_at=datetime.now(timezone.utc).isoformat(),
        preferred_format="npz",
    )
    manifest = cat.export_manifest(tmp_path / "manifest.json")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["manifest_version"] == 2
    assert payload["manifest_id"]
    assert len(payload["artifacts"]) == 1
    entry = payload["artifacts"][0]
    assert entry["artifact"]["artifact_id"] == rec.artifact_id
    assert entry["object_metadata"]["extractor_id"] == "x"
    assert entry["payload_sha256"]
    assert int(entry["payload_bytes"]) > 0


def test_export_manifest_raises_on_corrupt_metadata_json(tmp_path) -> None:
    cat = Catalog(tmp_path / "catalog")
    fs = FeatureSeries(
        values=np.ones((2, 2), dtype=np.float32),
        times_s=np.array([0.0, 1.0]),
        metadata={"extractor_id": "x", "params_hash": "y"},
    )
    rec = cat.put(
        fs,
        run_id="r1",
        stage_id="s1",
        code_version="dev",
        created_at=datetime.now(timezone.utc).isoformat(),
        preferred_format="npz",
    )
    meta_path = cat.root / "artifacts" / rec.artifact_id / "metadata.json"
    meta_path.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="Failed to parse metadata JSON"):
        cat.export_manifest(tmp_path / "manifest.json")


def test_catalog_put_leaves_no_temp_artifacts(tmp_path) -> None:
    cat = Catalog(tmp_path / "catalog")
    fs = FeatureSeries(
        values=np.ones((3, 2), dtype=np.float32),
        times_s=np.array([0.0, 0.5, 1.0]),
        metadata={"extractor_id": "x", "params_hash": "y"},
    )
    cat.put(
        fs,
        run_id="r1",
        stage_id="s1",
        code_version="dev",
        created_at=datetime.now(timezone.utc).isoformat(),
        preferred_format="npz",
    )
    leftovers = [p for p in (cat.root / "artifacts").rglob("*") if ".tmp" in p.name]
    assert leftovers == []


def test_query_alignment_artifacts_filters_by_backend_and_fallback(tmp_path) -> None:
    cat = Catalog(tmp_path / "catalog")
    words = EventSeries(
        onset_s=np.array([0.0, 0.4], dtype=np.float64),
        offset_s=np.array([0.2, 0.6], dtype=np.float64),
        label=np.array(["a", "b"], dtype=object),
        confidence=np.array([0.9, 0.2], dtype=np.float32),
        metadata={
            "extractor_id": "x",
            "params_hash": "y",
            "asr_model_name": "small",
            "aligner_backend": "whisperx",
            "fallback_used": False,
        },
    )
    cat.put(
        words,
        run_id="r-align",
        stage_id="align.words",
        code_version="dev",
        created_at=datetime.now(timezone.utc).isoformat(),
        preferred_format="npz",
    )
    res = cat.query_alignment_artifacts(aligner_backend="whisperx", fallback_used=False, asr_model_name="small")
    assert len(res) == 1
    assert res[0].stage_id == "align.words"
