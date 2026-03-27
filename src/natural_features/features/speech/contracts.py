"""Speech alignment contracts: metadata and QC schema helpers."""

from __future__ import annotations

from typing import Any


REQUIRED_ALIGNMENT_QC_FIELDS = (
    "mode",
    "fallback_used",
    "n_words",
    "low_confidence_words",
    "dropped_words",
)


def _non_empty_str(value: Any, *, default: str) -> str:
    txt = str(value).strip() if value is not None else ""
    return txt if txt else default


def ensure_word_event_metadata(
    metadata: dict[str, Any],
    *,
    asr_model_name: str,
    aligner_backend: str,
    aligner_version: str = "none",
) -> dict[str, Any]:
    """Ensure canonical metadata keys for aligned word EventSeries."""

    out = dict(metadata)
    out["asr_model_name"] = _non_empty_str(asr_model_name, default="unknown")
    out["aligner_backend"] = _non_empty_str(aligner_backend, default="none")
    out["aligner_version"] = _non_empty_str(aligner_version, default="none")
    return out


def ensure_phoneme_event_metadata(
    metadata: dict[str, Any],
    *,
    label_namespace: str,
    namespace_version: str,
    source_word_alignment_id: str,
) -> dict[str, Any]:
    """Ensure canonical metadata keys for aligned phoneme EventSeries."""

    out = dict(metadata)
    out["label_namespace"] = _non_empty_str(label_namespace, default="unknown")
    out["namespace_version"] = _non_empty_str(namespace_version, default="unknown")
    out["source_word_alignment_id"] = _non_empty_str(source_word_alignment_id, default="unknown")
    return out


def normalize_alignment_qc(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize and validate QC payload required by alignment contract."""

    out = dict(payload)
    out.setdefault("mode", "unknown")
    out.setdefault("fallback_used", False)
    out.setdefault("n_words", 0)
    out.setdefault("low_confidence_words", 0)
    out.setdefault("dropped_words", 0)

    out["mode"] = str(out["mode"])
    out["fallback_used"] = bool(out["fallback_used"])
    out["n_words"] = int(out["n_words"])
    out["low_confidence_words"] = int(out["low_confidence_words"])
    out["dropped_words"] = int(out["dropped_words"])

    for key in ("coverage_fraction", "boundary_jitter_ms_p50", "boundary_jitter_ms_p95"):
        if key in out and out[key] is not None:
            out[key] = float(out[key])
    for key in ("duration_outliers", "chunk_count", "stitch_conflicts"):
        if key in out and out[key] is not None:
            out[key] = int(out[key])
    if "speaker_overlap_fraction" in out and out["speaker_overlap_fraction"] is not None:
        out["speaker_overlap_fraction"] = float(out["speaker_overlap_fraction"])

    validate_alignment_qc(out)
    return out


def validate_alignment_qc(payload: dict[str, Any]) -> None:
    missing = [k for k in REQUIRED_ALIGNMENT_QC_FIELDS if k not in payload]
    if missing:
        raise ValueError(f"alignment QC missing required fields: {missing}")
