"""Minimal artifact catalog backed by SQLite."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import shutil
from typing import Any

from natural_features.core.feature_types import EventSeries, FeatureSeries, TrackSeries
from natural_features.storage.writers import write_event_series, write_feature_series, write_track_series
from natural_features.util.hashing import stable_hash
from natural_features.util.io import atomic_write_json


@dataclass(frozen=True)
class ArtifactMetadata:
    artifact_id: str
    run_id: str
    stage_id: str
    schema: str
    dtype: str
    shape: list[int]
    timebase: dict[str, Any]
    params_hash: str
    extractor_id: str
    code_version: str
    model_name: str
    model_revision: str
    upstream_ids: list[str]
    created_at: str
    backend: str
    hostname: str
    path: str


class Catalog:
    """Filesystem + SQLite catalog for artifact metadata."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.artifacts_dir = self.root / "artifacts"
        self.index_db = self.root / "catalog.sqlite3"
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.index_db)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    stage_id TEXT NOT NULL,
                    schema TEXT NOT NULL,
                    dtype TEXT NOT NULL,
                    shape_json TEXT NOT NULL,
                    timebase_json TEXT NOT NULL,
                    params_hash TEXT NOT NULL,
                    extractor_id TEXT NOT NULL,
                    code_version TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_revision TEXT NOT NULL,
                    upstream_ids_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    hostname TEXT NOT NULL,
                    path TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_stage ON artifacts(stage_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_run ON artifacts(run_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_schema ON artifacts(schema)")

    @staticmethod
    def make_artifact_id(payload: dict[str, Any]) -> str:
        return stable_hash(payload, length=20)

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _file_sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def _read_metadata_payload(self, artifact_id: str) -> dict[str, Any]:
        meta_path = self.artifacts_dir / artifact_id / "metadata.json"
        if not meta_path.exists():
            return {}
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Failed to parse metadata JSON for artifact '{artifact_id}': {meta_path}") from exc

    @staticmethod
    def _resolve_within(root: Path, rel_path: str) -> Path:
        rel = Path(rel_path)
        if rel.is_absolute():
            raise ValueError(f"Absolute paths are not allowed in manifest: {rel_path}")
        resolved = (root / rel).resolve()
        root_resolved = root.resolve()
        if root_resolved != resolved and root_resolved not in resolved.parents:
            raise ValueError(f"Path escapes catalog root: {rel_path}")
        return resolved

    def put(
        self,
        obj: FeatureSeries | EventSeries | TrackSeries,
        *,
        run_id: str,
        stage_id: str,
        code_version: str,
        model_name: str = "none",
        model_revision: str = "none",
        upstream_ids: list[str] | None = None,
        backend: str = "local",
        hostname: str = "localhost",
        created_at: str = "",
        preferred_format: str | None = None,
    ) -> ArtifactMetadata:
        upstream_ids = upstream_ids or []
        created_at = created_at or self._utc_now_iso()
        if isinstance(obj, EventSeries):
            shape_hint = [len(obj)]
        elif isinstance(obj, (FeatureSeries, TrackSeries)):
            shape_hint = list(obj.values.shape)
        else:
            shape_hint = []
        base_payload = {
            "run_id": run_id,
            "stage_id": stage_id,
            "schema": obj.schema,
            "params_hash": obj.metadata["params_hash"],
            "extractor_id": obj.metadata["extractor_id"],
            "code_version": code_version,
            "model_name": model_name,
            "model_revision": model_revision,
            "upstream_ids": upstream_ids,
            "shape": shape_hint,
        }
        artifact_id = self.make_artifact_id(base_payload)
        artifact_dir = self.artifacts_dir / artifact_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(obj, FeatureSeries):
            fmt = preferred_format or "npz"
            payload_path = write_feature_series(obj, artifact_dir, fmt=fmt)
            dtype = str(obj.values.dtype)
            shape = list(obj.values.shape)
        elif isinstance(obj, EventSeries):
            fmt = preferred_format or "npz"
            payload_path = write_event_series(obj, artifact_dir, fmt=fmt)
            dtype = "event"
            shape = [len(obj)]
        else:
            fmt = preferred_format or "npz"
            payload_path = write_track_series(obj, artifact_dir, fmt=fmt)
            dtype = str(obj.values.dtype)
            shape = list(obj.values.shape)
        payload_abs = payload_path
        payload_sha256 = self._file_sha256(payload_abs)
        payload_bytes = int(payload_abs.stat().st_size)

        record = ArtifactMetadata(
            artifact_id=artifact_id,
            run_id=run_id,
            stage_id=stage_id,
            schema=obj.schema,
            dtype=dtype,
            shape=shape,
            timebase=asdict(obj.timebase),
            params_hash=str(obj.metadata["params_hash"]),
            extractor_id=str(obj.metadata["extractor_id"]),
            code_version=code_version,
            model_name=model_name,
            model_revision=model_revision,
            upstream_ids=upstream_ids,
            created_at=created_at,
            backend=backend,
            hostname=hostname,
            path=str(payload_path.relative_to(self.root)),
        )
        self._write_metadata_json(
            record,
            obj.metadata,
            payload_sha256=payload_sha256,
            payload_bytes=payload_bytes,
        )
        try:
            self._upsert(record, obj.metadata)
        except Exception:
            (artifact_dir / "metadata.json").unlink(missing_ok=True)
            raise
        return record

    def _write_metadata_json(
        self,
        metadata: ArtifactMetadata,
        object_metadata: dict[str, Any],
        *,
        payload_sha256: str | None = None,
        payload_bytes: int | None = None,
    ) -> None:
        meta_path = self.artifacts_dir / metadata.artifact_id / "metadata.json"
        payload = {
            "artifact": asdict(metadata),
            "object_metadata": object_metadata,
            "payload": {
                "sha256": payload_sha256,
                "bytes": payload_bytes,
            },
        }
        atomic_write_json(meta_path, payload, sort_keys=True, indent=2)

    def _upsert(self, metadata: ArtifactMetadata, object_metadata: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO artifacts (
                    artifact_id, run_id, stage_id, schema, dtype, shape_json, timebase_json,
                    params_hash, extractor_id, code_version, model_name, model_revision,
                    upstream_ids_json, created_at, backend, hostname, path, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata.artifact_id,
                    metadata.run_id,
                    metadata.stage_id,
                    metadata.schema,
                    metadata.dtype,
                    json.dumps(metadata.shape),
                    json.dumps(metadata.timebase),
                    metadata.params_hash,
                    metadata.extractor_id,
                    metadata.code_version,
                    metadata.model_name,
                    metadata.model_revision,
                    json.dumps(metadata.upstream_ids),
                    metadata.created_at,
                    metadata.backend,
                    metadata.hostname,
                    metadata.path,
                    json.dumps(object_metadata, sort_keys=True),
                ),
            )

    def list_artifacts(self, *, schema: str | None = None, stage_id: str | None = None) -> list[ArtifactMetadata]:
        query = "SELECT * FROM artifacts"
        params: list[str] = []
        clauses: list[str] = []
        if schema is not None:
            clauses.append("schema = ?")
            params.append(schema)
        if stage_id is not None:
            clauses.append("stage_id = ?")
            params.append(stage_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at, artifact_id"
        rows: list[tuple[Any, ...]] = []
        with self._connect() as conn:
            rows = list(conn.execute(query, params).fetchall())
        return [self._row_to_artifact(row) for row in rows]

    def query_artifacts(
        self,
        *,
        extractor_id: str | None = None,
        run_id: str | None = None,
        stage_id: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
    ) -> list[ArtifactMetadata]:
        query = "SELECT * FROM artifacts"
        params: list[str] = []
        clauses: list[str] = []
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if stage_id is not None:
            clauses.append("stage_id = ?")
            params.append(stage_id)
        if created_after is not None:
            clauses.append("created_at >= ?")
            params.append(created_after)
        if created_before is not None:
            clauses.append("created_at <= ?")
            params.append(created_before)
        if extractor_id is not None:
            clauses.append("extractor_id = ?")
            params.append(extractor_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at, artifact_id"
        with self._connect() as conn:
            rows = list(conn.execute(query, params).fetchall())
        return [self._row_to_artifact(row) for row in rows]

    def query_alignment_artifacts(
        self,
        *,
        aligner_backend: str | None = None,
        fallback_used: bool | None = None,
        asr_model_name: str | None = None,
    ) -> list[ArtifactMetadata]:
        """Query speech alignment artifacts using object metadata fields."""

        query = "SELECT * FROM artifacts ORDER BY created_at, artifact_id"
        out: list[ArtifactMetadata] = []
        with self._connect() as conn:
            rows = list(conn.execute(query).fetchall())
        for row in rows:
            art = self._row_to_artifact(row)
            try:
                obj_meta = json.loads(row[17]) if row[17] else {}
            except Exception:
                obj_meta = {}
            if not isinstance(obj_meta, dict):
                continue
            if "aligner_backend" not in obj_meta and "asr_model_name" not in obj_meta:
                continue
            if aligner_backend is not None and str(obj_meta.get("aligner_backend")) != str(aligner_backend):
                continue
            if fallback_used is not None and bool(obj_meta.get("fallback_used", False)) != bool(fallback_used):
                continue
            if asr_model_name is not None and str(obj_meta.get("asr_model_name")) != str(asr_model_name):
                continue
            out.append(art)
        return out

    def get_artifact(self, artifact_id: str) -> ArtifactMetadata | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_artifact(row)

    def export_manifest(self, path: str | Path, *, artifact_ids: list[str] | None = None) -> Path:
        if artifact_ids:
            rows = [self.get_artifact(aid) for aid in artifact_ids]
            artifacts = [r for r in rows if r is not None]
        else:
            artifacts = self.list_artifacts()
        entries: list[dict[str, Any]] = []
        for artifact in artifacts:
            meta_payload = self._read_metadata_payload(artifact.artifact_id)
            object_metadata = meta_payload.get("object_metadata", {})
            payload_info = meta_payload.get("payload", {}) if isinstance(meta_payload.get("payload", {}), dict) else {}
            payload_sha256 = payload_info.get("sha256")
            payload_bytes = payload_info.get("bytes")
            artifact_path = self.root / artifact.path
            if (not payload_sha256 or payload_bytes is None) and artifact_path.exists():
                payload_sha256 = self._file_sha256(artifact_path)
                payload_bytes = int(artifact_path.stat().st_size)
            entries.append(
                {
                    "artifact": asdict(artifact),
                    "object_metadata": object_metadata,
                    "payload_sha256": payload_sha256,
                    "payload_bytes": payload_bytes,
                }
            )
        manifest_core = [
            {
                "artifact_id": e["artifact"]["artifact_id"],
                "path": e["artifact"]["path"],
                "payload_sha256": e.get("payload_sha256"),
            }
            for e in entries
        ]
        payload = {
            "manifest_version": 2,
            "catalog_root": str(self.root),
            "exported_at_utc": self._utc_now_iso(),
            "manifest_id": stable_hash(manifest_core, length=20),
            "artifacts": entries,
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(p, payload, sort_keys=True, indent=2)
        return p

    def import_manifest(
        self,
        path: str | Path,
        *,
        source_root: str | Path | None = None,
        copy_artifacts: bool = False,
    ) -> int:
        p = Path(path)
        payload = json.loads(p.read_text(encoding="utf-8"))
        manifest_version = int(payload.get("manifest_version", 1))
        if manifest_version not in {1, 2}:
            raise ValueError(f"Unsupported manifest_version={manifest_version}")
        source_root = Path(source_root) if source_root is not None else Path(payload.get("catalog_root", "."))
        inserted = 0
        for item in payload.get("artifacts", []):
            if manifest_version >= 2 and isinstance(item, dict) and "artifact" in item:
                meta_payload = item["artifact"]
                object_metadata = item.get("object_metadata", {})
                expected_sha256 = item.get("payload_sha256")
                expected_bytes = item.get("payload_bytes")
            else:
                meta_payload = item
                object_metadata = {}
                expected_sha256 = None
                expected_bytes = None
            if not isinstance(meta_payload, dict):
                raise ValueError("Manifest artifact entry is not a mapping")
            meta = ArtifactMetadata(**meta_payload)
            rel_parts = Path(meta.path).parts
            if len(rel_parts) < 3 or rel_parts[0] != "artifacts" or rel_parts[1] != meta.artifact_id:
                raise ValueError(f"Manifest artifact path does not match artifact id: {meta.artifact_id} -> {meta.path}")
            dst = self._resolve_within(self.root, meta.path)
            artifact_dir = dst.parent
            artifact_dir.mkdir(parents=True, exist_ok=True)
            if copy_artifacts:
                src = self._resolve_within(source_root, meta.path)
                if not src.exists():
                    raise FileNotFoundError(f"Manifest references missing artifact payload: {src}")
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
            if expected_sha256 and dst.exists():
                observed = self._file_sha256(dst)
                if observed != expected_sha256:
                    raise ValueError(f"Payload hash mismatch for artifact {meta.artifact_id}")
            if expected_bytes is None and dst.exists():
                expected_bytes = int(dst.stat().st_size)
            self._write_metadata_json(
                meta,
                object_metadata,
                payload_sha256=expected_sha256,
                payload_bytes=expected_bytes,
            )
            self._upsert(meta, object_metadata=object_metadata)
            inserted += 1
        return inserted

    def _row_to_artifact(self, row: tuple[Any, ...]) -> ArtifactMetadata:
        (
            artifact_id,
            run_id,
            stage_id,
            schema,
            dtype,
            shape_json,
            timebase_json,
            params_hash,
            extractor_id,
            code_version,
            model_name,
            model_revision,
            upstream_ids_json,
            created_at,
            backend,
            hostname,
            path,
            _metadata_json,
        ) = row
        return ArtifactMetadata(
            artifact_id=artifact_id,
            run_id=run_id,
            stage_id=stage_id,
            schema=schema,
            dtype=dtype,
            shape=json.loads(shape_json),
            timebase=json.loads(timebase_json),
            params_hash=params_hash,
            extractor_id=extractor_id,
            code_version=code_version,
            model_name=model_name,
            model_revision=model_revision,
            upstream_ids=json.loads(upstream_ids_json),
            created_at=created_at,
            backend=backend,
            hostname=hostname,
            path=path,
        )
