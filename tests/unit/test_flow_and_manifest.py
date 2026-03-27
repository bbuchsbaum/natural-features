from __future__ import annotations

from datetime import datetime, timezone
import json

import numpy as np
import pytest

from natural_features.core.feature_types import FeatureSeries
from natural_features.flow.backends.slurm import SlurmResources, profile_to_resources, to_sbatch_args
from natural_features.flow.chunking import chunked_map
from natural_features.flow.engine import run_flow
from natural_features.flow.report import write_html_report
from natural_features.flow.runlog import write_run_json
from natural_features.flow.spec import FlowSpec, RetryPolicy, StageSpec
from natural_features.storage.catalog import Catalog

pytestmark = [pytest.mark.nightly]


def _stage_a(**kwargs):
    return {"x": 1, "kwargs": kwargs}


def _stage_b(dep_outputs, **kwargs):
    return dep_outputs["a"]["x"] + 1


def _flaky_factory():
    state = {"n": 0}

    def _flaky(dep_outputs, **kwargs):
        state["n"] += 1
        if state["n"] < 2:
            raise RuntimeError("boom")
        return 42

    return _flaky


def test_flow_run_plan_retry_and_reports(tmp_path) -> None:
    flaky = _flaky_factory()
    flow = FlowSpec(
        flow_id="f1",
        stages=[
            StageSpec(stage_id="a", fn=_stage_a),
            StageSpec(stage_id="b", fn=_stage_b, deps=["a"]),
            StageSpec(stage_id="c", fn=flaky, deps=["b"], retry_policy=RetryPolicy(retries=1, retry_backoff_s=0.0)),
        ],
    )
    planned = run_flow(flow, mode="plan")
    assert all(r.status in {"pending", "cached"} for r in planned.stage_records)
    ran = run_flow(flow, mode="run")
    assert [r.status for r in ran.stage_records] == ["completed", "completed", "completed"]
    run_json = write_run_json(ran, tmp_path / "run.json")
    report_html = write_html_report(ran, tmp_path / "report.html")
    assert run_json.exists() and report_html.exists()
    payload = json.loads(run_json.read_text(encoding="utf-8"))
    assert payload["flow_id"] == "f1"


def test_html_report_escapes_user_strings(tmp_path) -> None:
    flow = FlowSpec(
        flow_id="f<script>",
        stages=[StageSpec(stage_id="<stage>", fn=_stage_a)],
    )
    result = run_flow(flow, mode="run", run_id="run<&>")
    out = write_html_report(result, tmp_path / "report.html")
    html = out.read_text(encoding="utf-8")
    assert "<script>" not in html
    assert "run&lt;&amp;&gt;" in html


def test_chunking_and_slurm_mapping() -> None:
    out = chunked_map(range(10), lambda xs: sum(xs), chunk_size=3, merge_fn=sum)
    assert out == sum(range(10))
    prof = profile_to_resources("gpu")
    args = to_sbatch_args(prof)
    assert "--gpus=1" in args
    custom = to_sbatch_args(SlurmResources(cpus=8, mem_gb=32, time_min=60))
    assert "--cpus-per-task=8" in custom


def test_catalog_query_and_manifest_roundtrip(tmp_path) -> None:
    c1 = Catalog(tmp_path / "c1")
    fs = FeatureSeries(
        values=np.ones((3, 2), dtype=np.float32),
        times_s=np.array([0.0, 1.0, 2.0]),
        metadata={"extractor_id": "x", "params_hash": "p"},
    )
    rec = c1.put(
        fs,
        run_id="r1",
        stage_id="s1",
        code_version="dev",
        created_at=datetime.now(timezone.utc).isoformat(),
        preferred_format="npz",
    )
    q = c1.query_artifacts(run_id="r1", extractor_id="x")
    assert len(q) == 1 and q[0].artifact_id == rec.artifact_id
    manifest = c1.export_manifest(tmp_path / "manifest.json")

    c2 = Catalog(tmp_path / "c2")
    inserted = c2.import_manifest(manifest, source_root=c1.root, copy_artifacts=True)
    assert inserted == 1
    assert c2.get_artifact(rec.artifact_id) is not None
    meta_json = json.loads((c2.root / "artifacts" / rec.artifact_id / "metadata.json").read_text(encoding="utf-8"))
    assert meta_json["object_metadata"]["extractor_id"] == "x"


def test_manifest_import_rejects_payload_hash_mismatch(tmp_path) -> None:
    c1 = Catalog(tmp_path / "c1")
    fs = FeatureSeries(
        values=np.ones((2, 2), dtype=np.float32),
        times_s=np.array([0.0, 1.0]),
        metadata={"extractor_id": "x", "params_hash": "p"},
    )
    c1.put(
        fs,
        run_id="r2",
        stage_id="s2",
        code_version="dev",
        created_at=datetime.now(timezone.utc).isoformat(),
        preferred_format="npz",
    )
    manifest = c1.export_manifest(tmp_path / "manifest.json")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["artifacts"][0]["payload_sha256"] = "badbad"
    tampered = tmp_path / "manifest_tampered.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")

    c2 = Catalog(tmp_path / "c2")
    with pytest.raises(ValueError):
        c2.import_manifest(tampered, source_root=c1.root, copy_artifacts=True)


def test_manifest_import_rejects_path_escape(tmp_path) -> None:
    c2 = Catalog(tmp_path / "c2")
    manifest_payload = {
        "manifest_version": 2,
        "catalog_root": str(tmp_path / "c1"),
        "artifacts": [
            {
                "artifact": {
                    "artifact_id": "a1",
                    "run_id": "r1",
                    "stage_id": "s1",
                    "schema": "FeatureSeries/v1",
                    "dtype": "float32",
                    "shape": [1, 1],
                    "timebase": {"kind": "frames"},
                    "params_hash": "p",
                    "extractor_id": "e",
                    "code_version": "dev",
                    "model_name": "none",
                    "model_revision": "none",
                    "upstream_ids": [],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "backend": "local",
                    "hostname": "localhost",
                    "path": "../../escape.npz",
                },
                "object_metadata": {},
                "payload_sha256": None,
                "payload_bytes": None,
            }
        ],
    }
    manifest = tmp_path / "manifest_escape.json"
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    with pytest.raises(ValueError):
        c2.import_manifest(manifest, copy_artifacts=False)
