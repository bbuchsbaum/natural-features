"""MFA phone-tier events."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from natural_features.core.backend_errors import (
    BackendDependencyError,
    BackendInferenceError,
)
from natural_features.core.execution import (
    add_execution_provenance,
    resolve_execution_mode,
)
from natural_features.core.feature_types import EventSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.features.common import extractor_metadata
from natural_features.features.speech.formats import read_textgrid
from natural_features.features.speech.phonology import phoneme_event_series


PHONE_TIER_NAMES = ("phones", "phone")
WORD_TIER_NAMES = ("words", "word")


def _empty_phones(extractor_name: str, params: dict[str, Any]) -> EventSeries:
    return phoneme_event_series(
        onset_s=np.array([], dtype=np.float64),
        offset_s=np.array([], dtype=np.float64),
        labels=np.array([], dtype=object),
        confidence=np.array([], dtype=np.float32),
        label_namespace="arpabet",
        namespace_version="mfa",
        source_word_alignment_id="speech.phones.mfa",
        metadata=extractor_metadata(extractor_name, params=params),
    )


def phones_from_textgrid(
    path: str | Path,
    *,
    tier: str | None = None,
) -> EventSeries:
    """Read a TextGrid phone tier into a phoneme EventSeries."""

    preferred = (tier,) if tier else PHONE_TIER_NAMES
    events = None
    for name in preferred:
        try:
            events = read_textgrid(path, tier=name)
        except ValueError:
            events = None
        if events is not None and len(events) > 0:
            break
    if events is None or len(events) == 0:
        return _empty_phones(
            "speech.phones.from_textgrid", {"path": str(Path(path).name), "tier": tier}
        )
    keep = np.array(
        [
            bool(str(x).strip())
            for x in (events.label if events.label is not None else [])
        ],
        dtype=bool,
    )
    if events.label is None or keep.size != len(events):
        keep = np.ones(len(events), dtype=bool)
    md = extractor_metadata(
        "speech.phones.from_textgrid",
        params={"path": str(Path(path).name), "tier": tier or "phones"},
        extra={"backend": "textgrid_tier"},
    )
    return phoneme_event_series(
        onset_s=events.onset_s[keep],
        offset_s=events.offset_s[keep],
        labels=np.asarray(events.label, dtype=object)[keep],
        confidence=(
            events.confidence[keep]
            if events.confidence is not None
            else np.ones(int(np.sum(keep)), dtype=np.float32)
        ),
        label_namespace="arpabet",
        namespace_version="mfa",
        source_word_alignment_id="textgrid",
        metadata=md,
    )


def mfa_phone_events(
    stimulus: AudioStimulus,
    words: EventSeries,
    *,
    mfa_dictionary_path: str,
    mfa_acoustic_model_path: str,
    mfa_timeout_s: float = 300.0,
    mfa_tmp_dir: str | None = None,
    mfa_extra_args: list[str] | None = None,
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
) -> EventSeries:
    """Run MFA and return the phones TextGrid tier as events."""

    mode, _strict = resolve_execution_mode(
        execution_mode=execution_mode, strict_dependency=strict_dependency
    )
    from natural_features.features.speech.align import _refine_words_with_mfa

    params = {
        "mfa_dictionary_path": mfa_dictionary_path,
        "mfa_acoustic_model_path": mfa_acoustic_model_path,
    }
    try:
        _mapped, _dropped, details = _refine_words_with_mfa(
            stimulus=stimulus,
            words=words,
            dictionary_path=mfa_dictionary_path,
            acoustic_model_path=mfa_acoustic_model_path,
            timeout_s=mfa_timeout_s,
            tmp_dir=mfa_tmp_dir,
            extra_args=mfa_extra_args,
        )
    except FileNotFoundError as exc:
        raise BackendDependencyError("mfa", "mfa executable is required") from exc
    except Exception as exc:
        raise BackendInferenceError(
            "mfa", f"MFA phone alignment failed: {exc}"
        ) from exc
    phones = details.get("phones")
    if not isinstance(phones, EventSeries):
        phones = _empty_phones("speech.phones.mfa", params)
    md = add_execution_provenance(
        extractor_metadata(
            "speech.phones.mfa", params=params, extra={"backend": "mfa"}
        ),
        execution_mode=mode,
        fallback_used=False,
    )
    return phoneme_event_series(
        onset_s=phones.onset_s,
        offset_s=phones.offset_s,
        labels=phones.label if phones.label is not None else np.array([], dtype=object),
        confidence=phones.confidence,
        label_namespace="arpabet",
        namespace_version="mfa",
        source_word_alignment_id=str(words.metadata.get("extractor_id", "unknown")),
        metadata=md,
    )
