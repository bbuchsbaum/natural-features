"""Phonological and articulatory feature extractors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from natural_features.core.backend_errors import (
    BackendDependencyError,
    BackendInferenceError,
    BackendLoadError,
)
from natural_features.core.execution import (
    add_execution_provenance,
    resolve_execution_mode,
)
from natural_features.core.feature_types import EventSeries, FeatureSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.core.timebase import TimebaseSpec, times_from_hop
from natural_features.features.audio.lowlevel import mel
from natural_features.features.common import extractor_metadata
from natural_features.features.speech.chunking import plan_audio_chunks
from natural_features.features.speech.contracts import ensure_phoneme_event_metadata


_VOWELS = set("aeiou")
_LABIAL = set("pbmfwv")
_CORONAL = set("tdnslzrchj")
_DORSAL = set("kgqx")

DEFAULT_ARTICULATORY_FEATURES: list[str] = [
    "bilabial",
    "labiodental",
    "dental",
    "alveolar",
    "postalveolar",
    "palatal",
    "velar",
    "glottal",
    "stop",
    "fricative",
    "affricate",
    "nasal",
    "approximant",
    "vowel",
    "consonant",
    "sonorant",
    "obstruent",
    "voiced",
    "front",
    "central",
    "back",
    "silence",
]

DEFAULT_DISTINCTIVE_FEATURES: list[str] = list(DEFAULT_ARTICULATORY_FEATURES) + [
    "continuant",
    "strident",
    "lateral",
    "aspirated",
    "rounded",
    "high",
    "low",
]

_DISTINCTIVE_EXTRAS: dict[str, set[str]] = {
    "S": {"continuant", "strident"},
    "Z": {"continuant", "strident"},
    "SH": {"continuant", "strident"},
    "ZH": {"continuant", "strident"},
    "F": {"continuant", "strident"},
    "V": {"continuant", "strident"},
    "TH": {"continuant"},
    "DH": {"continuant"},
    "HH": {"continuant"},
    "L": {"continuant", "lateral"},
    "R": {"continuant"},
    "Y": {"continuant"},
    "W": {"continuant", "rounded"},
    "CH": {"strident"},
    "JH": {"strident"},
    "IY": {"continuant", "high"},
    "IH": {"continuant", "high"},
    "EH": {"continuant"},
    "AE": {"continuant", "low"},
    "EY": {"continuant"},
    "AY": {"continuant"},
    "AH": {"continuant"},
    "ER": {"continuant"},
    "AA": {"continuant", "low"},
    "AO": {"continuant", "rounded"},
    "OW": {"continuant", "rounded"},
    "AW": {"continuant", "rounded"},
    "OY": {"continuant", "rounded"},
    "UH": {"continuant", "high", "rounded"},
    "UW": {"continuant", "high", "rounded"},
}

_CLASS_DISTINCTIVE_EXTRAS: dict[str, set[str]] = {
    "labiodental_fricative": {"continuant", "strident"},
    "dental_fricative": {"continuant"},
    "alveolar_fricative": {"continuant", "strident"},
    "postalveolar_fricative": {"continuant", "strident"},
    "approximant": {"continuant"},
    "front_vowel": {"continuant"},
    "central_vowel": {"continuant"},
    "back_vowel": {"continuant"},
}

# Coarse phone-like classes for posteriorgrams when a strict phone recognizer
# is unavailable. These classes are intentionally broad and easy to map to
# articulatory feature probabilities.
DEFAULT_ACOUSTIC_PHONE_CLASSES: list[str] = [
    "silence",
    "bilabial_stop",
    "alveolar_stop",
    "velar_stop",
    "labiodental_fricative",
    "dental_fricative",
    "alveolar_fricative",
    "postalveolar_fricative",
    "nasal",
    "approximant",
    "front_vowel",
    "central_vowel",
    "back_vowel",
]

_CLASS_TO_FEATURES: dict[str, set[str]] = {
    "silence": {"silence"},
    "bilabial_stop": {"bilabial", "stop", "consonant", "obstruent"},
    "alveolar_stop": {"alveolar", "stop", "consonant", "obstruent"},
    "velar_stop": {"velar", "stop", "consonant", "obstruent"},
    "labiodental_fricative": {"labiodental", "fricative", "consonant", "obstruent"},
    "dental_fricative": {"dental", "fricative", "consonant", "obstruent"},
    "alveolar_fricative": {"alveolar", "fricative", "consonant", "obstruent"},
    "postalveolar_fricative": {"postalveolar", "fricative", "consonant", "obstruent"},
    "nasal": {"nasal", "consonant", "sonorant", "voiced"},
    "approximant": {"approximant", "consonant", "sonorant", "voiced"},
    "front_vowel": {"vowel", "sonorant", "voiced", "front"},
    "central_vowel": {"vowel", "sonorant", "voiced", "central"},
    "back_vowel": {"vowel", "sonorant", "voiced", "back"},
}

_ARPABET_TO_FEATURES: dict[str, set[str]] = {
    "P": {"bilabial", "stop", "consonant", "obstruent"},
    "B": {"bilabial", "stop", "consonant", "obstruent", "voiced"},
    "M": {"bilabial", "nasal", "consonant", "sonorant", "voiced"},
    "F": {"labiodental", "fricative", "consonant", "obstruent"},
    "V": {"labiodental", "fricative", "consonant", "obstruent", "voiced"},
    "TH": {"dental", "fricative", "consonant", "obstruent"},
    "DH": {"dental", "fricative", "consonant", "obstruent", "voiced"},
    "T": {"alveolar", "stop", "consonant", "obstruent"},
    "D": {"alveolar", "stop", "consonant", "obstruent", "voiced"},
    "N": {"alveolar", "nasal", "consonant", "sonorant", "voiced"},
    "S": {"alveolar", "fricative", "consonant", "obstruent"},
    "Z": {"alveolar", "fricative", "consonant", "obstruent", "voiced"},
    "SH": {"postalveolar", "fricative", "consonant", "obstruent"},
    "ZH": {"postalveolar", "fricative", "consonant", "obstruent", "voiced"},
    "CH": {"postalveolar", "affricate", "consonant", "obstruent"},
    "JH": {"postalveolar", "affricate", "consonant", "obstruent", "voiced"},
    "K": {"velar", "stop", "consonant", "obstruent"},
    "G": {"velar", "stop", "consonant", "obstruent", "voiced"},
    "NG": {"velar", "nasal", "consonant", "sonorant", "voiced"},
    "HH": {"glottal", "fricative", "consonant", "obstruent"},
    "L": {"alveolar", "approximant", "consonant", "sonorant", "voiced"},
    "R": {"alveolar", "approximant", "consonant", "sonorant", "voiced"},
    "Y": {"palatal", "approximant", "consonant", "sonorant", "voiced"},
    "W": {"bilabial", "velar", "approximant", "consonant", "sonorant", "voiced"},
    "IY": {"vowel", "sonorant", "voiced", "front"},
    "IH": {"vowel", "sonorant", "voiced", "front"},
    "EH": {"vowel", "sonorant", "voiced", "front"},
    "AE": {"vowel", "sonorant", "voiced", "front"},
    "EY": {"vowel", "sonorant", "voiced", "front"},
    "AY": {"vowel", "sonorant", "voiced", "front"},
    "AH": {"vowel", "sonorant", "voiced", "central"},
    "ER": {"vowel", "sonorant", "voiced", "central"},
    "AA": {"vowel", "sonorant", "voiced", "back"},
    "AO": {"vowel", "sonorant", "voiced", "back"},
    "OW": {"vowel", "sonorant", "voiced", "back"},
    "AW": {"vowel", "sonorant", "voiced", "back"},
    "OY": {"vowel", "sonorant", "voiced", "front", "back"},
    "UH": {"vowel", "sonorant", "voiced", "back"},
    "UW": {"vowel", "sonorant", "voiced", "back"},
}

_IPA_CHAR_TO_FEATURES: dict[str, set[str]] = {
    "p": {"bilabial", "stop", "consonant", "obstruent"},
    "b": {"bilabial", "stop", "consonant", "obstruent", "voiced"},
    "m": {"bilabial", "nasal", "consonant", "sonorant", "voiced"},
    "f": {"labiodental", "fricative", "consonant", "obstruent"},
    "v": {"labiodental", "fricative", "consonant", "obstruent", "voiced"},
    "θ": {"dental", "fricative", "consonant", "obstruent"},
    "ð": {"dental", "fricative", "consonant", "obstruent", "voiced"},
    "t": {"alveolar", "stop", "consonant", "obstruent"},
    "d": {"alveolar", "stop", "consonant", "obstruent", "voiced"},
    "n": {"alveolar", "nasal", "consonant", "sonorant", "voiced"},
    "s": {"alveolar", "fricative", "consonant", "obstruent"},
    "z": {"alveolar", "fricative", "consonant", "obstruent", "voiced"},
    "ʃ": {"postalveolar", "fricative", "consonant", "obstruent"},
    "ʒ": {"postalveolar", "fricative", "consonant", "obstruent", "voiced"},
    "k": {"velar", "stop", "consonant", "obstruent"},
    "g": {"velar", "stop", "consonant", "obstruent", "voiced"},
    "ŋ": {"velar", "nasal", "consonant", "sonorant", "voiced"},
    "h": {"glottal", "fricative", "consonant", "obstruent"},
    "l": {"alveolar", "approximant", "consonant", "sonorant", "voiced"},
    "ɹ": {"alveolar", "approximant", "consonant", "sonorant", "voiced"},
    "r": {"alveolar", "approximant", "consonant", "sonorant", "voiced"},
    "j": {"palatal", "approximant", "consonant", "sonorant", "voiced"},
    "w": {"bilabial", "velar", "approximant", "consonant", "sonorant", "voiced"},
    "i": {"vowel", "sonorant", "voiced", "front"},
    "e": {"vowel", "sonorant", "voiced", "front"},
    "ɛ": {"vowel", "sonorant", "voiced", "front"},
    "a": {"vowel", "sonorant", "voiced", "central"},
    "ə": {"vowel", "sonorant", "voiced", "central"},
    "ɜ": {"vowel", "sonorant", "voiced", "central"},
    "o": {"vowel", "sonorant", "voiced", "back"},
    "u": {"vowel", "sonorant", "voiced", "back"},
    "ɔ": {"vowel", "sonorant", "voiced", "back"},
}

_IPA_VOWEL_FRONT = {"i", "y", "ɪ", "e", "ø", "ɛ", "œ", "æ"}
_IPA_VOWEL_CENTRAL = {"ə", "ɜ", "ɐ", "a", "ɞ", "ɘ", "ɵ"}
_IPA_VOWEL_BACK = {"u", "ʊ", "o", "ɔ", "ɑ", "ɒ", "ɯ"}
_IPA_VOWEL_ALL = _IPA_VOWEL_FRONT | _IPA_VOWEL_CENTRAL | _IPA_VOWEL_BACK
_IPA_CONSONANT_OBSTRUENT = {
    "p",
    "b",
    "t",
    "d",
    "k",
    "g",
    "f",
    "v",
    "θ",
    "ð",
    "s",
    "z",
    "ʃ",
    "ʒ",
    "h",
}
_IPA_CONSONANT_SONORANT = {"m", "n", "ŋ", "l", "ɹ", "r", "j", "w"}

_SPECIAL_TOKEN_VALUES = {
    "",
    "<pad>",
    "<s>",
    "</s>",
    "<unk>",
    "[pad]",
    "[unk]",
    "[cls]",
    "[sep]",
    "[mask]",
}


def _normalize_phone_label(label: str) -> str:
    token = str(label).strip()
    if not token:
        return ""
    # Remove stress markers in ARPABET-style vowels (e.g., AH0 -> AH).
    if token[-1].isdigit():
        token = token[:-1]
    return token.upper()


def _normalize_ctc_token(label: str) -> str:
    token = str(label).strip()
    if not token:
        return ""
    token = token.replace("▁", "").replace("Ġ", "").replace(" ", "")
    if token in {"|", "sil", "sp", "spn"}:
        return "silence"
    token_lower = token.lower()
    if token_lower in _SPECIAL_TOKEN_VALUES:
        return ""
    return token


def _ipa_features_from_token(token: str) -> set[str]:
    classes: set[str] = set()
    if not token:
        return classes
    # Remove common length/syllabic/aspiration-like diacritics.
    cleaned = (
        token.replace("ː", "")
        .replace("̩", "")
        .replace("ʰ", "")
        .replace("ˑ", "")
        .replace("ˈ", "")
        .replace("ˌ", "")
    )
    # Include per-character mappings when available.
    for ch in cleaned:
        classes.update(_IPA_CHAR_TO_FEATURES.get(ch, set()))
    chars = set(cleaned)
    # Add coarse vowel/consonant and frontness/backness cues for multi-char tokens.
    if chars & _IPA_VOWEL_ALL:
        classes.add("vowel")
        classes.add("sonorant")
        classes.add("voiced")
        if chars & _IPA_VOWEL_FRONT:
            classes.add("front")
        if chars & _IPA_VOWEL_CENTRAL:
            classes.add("central")
        if chars & _IPA_VOWEL_BACK:
            classes.add("back")
    if chars & (_IPA_CONSONANT_OBSTRUENT | _IPA_CONSONANT_SONORANT):
        classes.add("consonant")
    if chars & _IPA_CONSONANT_OBSTRUENT:
        classes.add("obstruent")
    if chars & _IPA_CONSONANT_SONORANT:
        classes.add("sonorant")
        classes.add("voiced")
    # Affricate cues for common IPA bigrams.
    if "dʒ" in cleaned or "tʃ" in cleaned:
        classes.add("affricate")
        classes.add("consonant")
        classes.add("obstruent")
    return classes


def _looks_like_ipa_token(token: str) -> bool:
    if not token:
        return False
    cleaned = (
        str(token)
        .replace("ː", "")
        .replace("̩", "")
        .replace("ʰ", "")
        .replace("ˑ", "")
        .replace("ˈ", "")
        .replace("ˌ", "")
        .strip()
    )
    if not cleaned:
        return False
    if any(ord(ch) > 127 for ch in cleaned):
        return True
    # Restrict ASCII IPA fallback to short tokens to avoid treating words as phones.
    return len(cleaned) <= 2 and any(ch in _IPA_CHAR_TO_FEATURES for ch in cleaned)


def _resample_audio_linear(wav: np.ndarray, *, from_sr: int, to_sr: int) -> np.ndarray:
    if from_sr <= 0 or to_sr <= 0:
        raise ValueError("sampling rates must be > 0")
    if from_sr == to_sr:
        return wav.astype(np.float32, copy=False)
    n_in = wav.shape[0]
    duration_s = n_in / float(from_sr)
    n_out = max(1, int(round(duration_s * float(to_sr))))
    x_in = np.linspace(0.0, duration_s, num=n_in, endpoint=False, dtype=np.float64)
    x_out = np.linspace(0.0, duration_s, num=n_out, endpoint=False, dtype=np.float64)
    out = np.interp(x_out, x_in, wav.astype(np.float64))
    return out.astype(np.float32)


def _features_for_label(raw: str) -> set[str]:
    raw_tok = str(raw)
    # pypar/ppgs encode silence as a single space; keep that mappable.
    if raw_tok in {" ", "\t", "\n"} or raw_tok.lower() in {"<silent>", "<silence>"}:
        return set(_CLASS_TO_FEATURES["silence"])
    raw_tok = raw_tok.strip()
    if not raw_tok:
        return set()

    ctc_norm = _normalize_ctc_token(raw_tok)
    if ctc_norm:
        from_ctc = _CLASS_TO_FEATURES.get(ctc_norm.lower())
        if from_ctc is not None:
            extras = set(_CLASS_DISTINCTIVE_EXTRAS.get(ctc_norm.lower(), set()))
            return set(from_ctc) | extras

    from_raw = _CLASS_TO_FEATURES.get(raw_tok.lower())
    if from_raw is not None:
        extras = set(_CLASS_DISTINCTIVE_EXTRAS.get(raw_tok.lower(), set()))
        return set(from_raw) | extras

    arpabet = _normalize_phone_label(raw_tok)
    from_arpabet = _ARPABET_TO_FEATURES.get(arpabet)
    if from_arpabet is not None:
        return set(from_arpabet) | set(_DISTINCTIVE_EXTRAS.get(arpabet, set()))

    ipa_tok = ctc_norm if ctc_norm else raw_tok
    if _looks_like_ipa_token(ipa_tok):
        classes = _ipa_features_from_token(ipa_tok)
        if "fricative" in classes or "approximant" in classes or "vowel" in classes:
            classes.add("continuant")
        return classes
    return set()


def _labels_to_feature_matrix(
    labels: list[str],
    *,
    feature_names: list[str],
) -> np.ndarray:
    mat = np.zeros((len(labels), len(feature_names)), dtype=np.float32)
    feature_ix = {f: i for i, f in enumerate(feature_names)}
    for row_ix, raw in enumerate(labels):
        classes = _features_for_label(raw)
        for f in classes:
            col_ix = feature_ix.get(f)
            if col_ix is not None:
                mat[row_ix, col_ix] = 1.0
    return mat


def phoneme_event_series(
    onset_s: np.ndarray,
    offset_s: np.ndarray,
    labels: np.ndarray,
    *,
    confidence: np.ndarray | None = None,
    extra: dict[str, np.ndarray] | None = None,
    label_namespace: str = "unknown",
    namespace_version: str = "v1",
    source_word_alignment_id: str = "unknown",
    metadata: dict[str, object] | None = None,
) -> EventSeries:
    """Build canonical phoneme EventSeries with namespace metadata."""

    md_in = dict(metadata or {})
    if "extractor_id" not in md_in or "params_hash" not in md_in:
        md_in.update(
            extractor_metadata(
                "speech.phonology.phoneme_events",
                params={
                    "label_namespace": label_namespace,
                    "namespace_version": namespace_version,
                },
            )
        )
    md = ensure_phoneme_event_metadata(
        md_in,
        label_namespace=label_namespace,
        namespace_version=namespace_version,
        source_word_alignment_id=source_word_alignment_id,
    )
    return EventSeries(
        onset_s=np.asarray(onset_s, dtype=np.float64),
        offset_s=np.asarray(offset_s, dtype=np.float64),
        label=np.asarray(labels, dtype=object),
        confidence=None
        if confidence is None
        else np.asarray(confidence, dtype=np.float32),
        extra=dict(extra or {}),
        metadata=md,
    )


def phoneme_events_from_words(
    words: EventSeries,
    *,
    label_namespace: str = "arpabet",
    namespace_version: str = "v1",
) -> EventSeries:
    """Construct coarse phoneme events from word-level phone labels.

    Expected input labels are either single phones (e.g., ``"AH0"``) or
    whitespace-separated phone strings (e.g., ``"DH AH0"``). Word intervals are
    split uniformly over contained phones.
    """

    if len(words) == 0:
        return phoneme_event_series(
            onset_s=np.array([], dtype=np.float64),
            offset_s=np.array([], dtype=np.float64),
            labels=np.array([], dtype=object),
            confidence=np.array([], dtype=np.float32),
            label_namespace=label_namespace,
            namespace_version=namespace_version,
            source_word_alignment_id=str(words.metadata.get("extractor_id", "unknown")),
            metadata=extractor_metadata(
                "speech.phonology.events_from_words", params={}
            ),
        )

    labels_in = words.label if words.label is not None else np.array([], dtype=object)
    conf_in = (
        words.confidence
        if words.confidence is not None
        else np.ones(len(words), dtype=np.float32)
    )

    out_on: list[float] = []
    out_off: list[float] = []
    out_lab: list[str] = []
    out_conf: list[float] = []

    for i in range(len(words)):
        token = str(labels_in[i]).strip() if i < len(labels_in) else ""
        phones = [p for p in token.split() if p]
        if not phones:
            continue
        w_on = float(words.onset_s[i])
        w_off = float(words.offset_s[i])
        dur = max(w_off - w_on, 1e-6)
        step = dur / float(len(phones))
        for j, phone in enumerate(phones):
            out_on.append(w_on + (j * step))
            out_off.append(w_on + ((j + 1) * step))
            out_lab.append(phone)
            out_conf.append(float(conf_in[i]))

    return phoneme_event_series(
        onset_s=np.asarray(out_on, dtype=np.float64),
        offset_s=np.asarray(out_off, dtype=np.float64),
        labels=np.asarray(out_lab, dtype=object),
        confidence=np.asarray(out_conf, dtype=np.float32),
        label_namespace=label_namespace,
        namespace_version=namespace_version,
        source_word_alignment_id=str(words.metadata.get("extractor_id", "unknown")),
        metadata=extractor_metadata(
            "speech.phonology.events_from_words",
            params={
                "label_namespace": label_namespace,
                "namespace_version": namespace_version,
            },
        ),
    )


def articulatory_features(words: EventSeries) -> FeatureSeries:
    labels = (
        words.label
        if words.label is not None
        else np.array([""] * len(words), dtype=object)
    )
    vals = np.zeros((len(words), 5), dtype=np.float32)
    for i, token in enumerate(labels):
        w = str(token).lower()
        chars = [c for c in w if c.isalpha()]
        if not chars:
            continue
        n = float(len(chars))
        vals[i, 0] = sum(c in _VOWELS for c in chars) / n
        vals[i, 1] = sum(c in _LABIAL for c in chars) / n
        vals[i, 2] = sum(c in _CORONAL for c in chars) / n
        vals[i, 3] = sum(c in _DORSAL for c in chars) / n
        vals[i, 4] = float(chars[0] in _VOWELS)
    md = extractor_metadata("speech.articulatory.features", params={})
    return FeatureSeries(
        values=vals,
        times_s=words.onset_s,
        dims=("time", "feature"),
        coords={
            "feature": [
                "vowel_ratio",
                "labial_ratio",
                "coronal_ratio",
                "dorsal_ratio",
                "starts_vowel",
            ]
        },
        metadata=md,
        timebase=TimebaseSpec(kind="tokens"),
    )


def articulatory_from_phoneme_events(
    phonemes: EventSeries,
    *,
    feature_names: list[str] | None = None,
    include_confidence: bool = True,
    extractor_name: str = "speech.articulatory.from_phoneme_events",
) -> FeatureSeries:
    """Map phoneme-labeled events to articulatory feature vectors.

    This is the canonical articulatory path for phone-level labels. It uses
    label normalization (`ARPABET`/`IPA`/class token handling) and emits one row
    per phoneme event at event onset times.
    """

    labels = (
        phonemes.label if phonemes.label is not None else np.array([], dtype=object)
    )
    feature_names = feature_names or list(DEFAULT_ARTICULATORY_FEATURES)
    vals = _labels_to_feature_matrix(
        [str(x) for x in labels], feature_names=feature_names
    ).astype(np.float32)
    out_names = list(feature_names)
    conf = phonemes.confidence if phonemes.confidence is not None else None
    if include_confidence:
        c = (
            np.ones((len(phonemes), 1), dtype=np.float32)
            if conf is None
            else np.asarray(conf, dtype=np.float32).reshape(-1, 1)
        )
        vals = np.concatenate([vals, c], axis=1)
        out_names.append("event_confidence")
    md = extractor_metadata(
        extractor_name,
        params={
            "feature_names": out_names,
            "include_confidence": include_confidence,
        },
        extra={
            "source_label_namespace": phonemes.metadata.get(
                "label_namespace", "unknown"
            ),
            "source_word_alignment_id": phonemes.metadata.get(
                "source_word_alignment_id", "unknown"
            ),
        },
    )
    return FeatureSeries(
        values=vals.astype(np.float32),
        times_s=phonemes.onset_s.astype(np.float64),
        dims=("time", "feature"),
        coords={"feature": out_names},
        metadata=md,
        timebase=TimebaseSpec(kind="tokens"),
    )


def phoneme_posteriorgrams(
    stimulus: AudioStimulus,
    *,
    hop_s: float = 0.02,
    n_classes: int = 8,
    class_labels: list[str] | None = None,
) -> FeatureSeries:
    m = mel(
        stimulus,
        hop_s=hop_s,
        win_s=max(0.025, 2 * hop_s),
        n_mels=max(16, n_classes * 2),
    )
    x = m.values.astype(np.float32)
    bins = np.array_split(np.arange(x.shape[1]), n_classes)
    logits = np.stack([x[:, b].mean(axis=1) for b in bins], axis=1)
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-8)
    if class_labels is None:
        names = [f"phn_{i}" for i in range(n_classes)]
    else:
        if len(class_labels) != n_classes:
            raise ValueError("len(class_labels) must match n_classes")
        names = [str(c) for c in class_labels]
    md = extractor_metadata(
        "speech.phonology.posteriorgrams",
        params={"hop_s": hop_s, "n_classes": n_classes, "class_labels": names},
    )
    return FeatureSeries(
        values=probs.astype(np.float32),
        times_s=m.times_s,
        dims=("time", "feature"),
        coords={"feature": names},
        metadata=md,
        timebase=TimebaseSpec(
            kind="audio_hop", hop_s=hop_s, sampling_rate_hz=1.0 / hop_s
        ),
    )


def acoustic_phone_posteriors(
    stimulus: AudioStimulus,
    *,
    hop_s: float = 0.02,
) -> FeatureSeries:
    labels = list(DEFAULT_ACOUSTIC_PHONE_CLASSES)
    return phoneme_posteriorgrams(
        stimulus,
        hop_s=hop_s,
        n_classes=len(labels),
        class_labels=labels,
    )


_VALID_CTC_DEVICES = {"auto", "cuda", "mps", "cpu"}


@dataclass
class CTCModelRuntime:
    """A loaded CTC processor/model pair reusable across many stimuli."""

    processor: Any
    model: Any
    torch: Any
    device: str
    model_id: str
    sample_rate_hz: int
    local_files_only: bool


_CTC_RUNTIME_CACHE: dict[tuple[str, bool, str], CTCModelRuntime] = {}


def clear_ctc_runtime() -> None:
    """Drop any process-level cached CTC runtimes."""

    _CTC_RUNTIME_CACHE.clear()


def _resolve_torch_device(torch: Any, requested: str) -> str:
    """Resolve ``auto`` to CUDA, MPS, or CPU and validate explicit devices."""

    device = str(requested).strip().lower()
    if device not in _VALID_CTC_DEVICES:
        raise ValueError("device must be one of {'auto', 'cuda', 'mps', 'cpu'}")
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        return "mps" if mps is not None and mps.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for CTC extraction but is unavailable.")
    if device == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError(
                "MPS was requested for CTC extraction but is unavailable."
            )
    return device


def _move_to_device(value: Any, device: str) -> Any:
    to_method = getattr(value, "to", None)
    if callable(to_method):
        return to_method(device)
    return value


def load_ctc_runtime(
    *,
    model: str = "bobboyms/wav2vec2-base-en-phoneme-ctc-41h",
    local_files_only: bool = True,
    device: str = "auto",
) -> CTCModelRuntime:
    """Load one CTC processor/model pair onto the requested device."""

    try:
        import torch
        from transformers import AutoModelForCTC, AutoProcessor  # type: ignore
    except ImportError as error:
        raise BackendDependencyError(
            "phoneme CTC",
            "transformers and torch are required",
        ) from error

    selected_device = _resolve_torch_device(torch, device)
    try:
        processor = AutoProcessor.from_pretrained(
            model, local_files_only=local_files_only
        )
        net = AutoModelForCTC.from_pretrained(model, local_files_only=local_files_only)
    except Exception as error:
        raise BackendLoadError(
            "phoneme CTC", f"model '{model}' is unavailable"
        ) from error
    try:
        net = net.to(selected_device)
        eval_method = getattr(net, "eval", None)
        if callable(eval_method):
            eval_method()
    except Exception as error:
        raise BackendLoadError(
            "phoneme CTC",
            f"model '{model}' is unavailable on {selected_device}",
        ) from error
    model_sr = int(
        getattr(getattr(processor, "feature_extractor", None), "sampling_rate", 16000)
    )
    return CTCModelRuntime(
        processor=processor,
        model=net,
        torch=torch,
        device=selected_device,
        model_id=model,
        sample_rate_hz=model_sr,
        local_files_only=bool(local_files_only),
    )


def _get_ctc_runtime(
    *,
    model: str,
    local_files_only: bool,
    device: str,
    runtime: CTCModelRuntime | None,
) -> CTCModelRuntime:
    if runtime is not None:
        if runtime.model_id != model:
            raise ValueError(
                f"Preloaded CTC runtime uses '{runtime.model_id}', not '{model}'."
            )
        return runtime
    try:
        import torch
    except ImportError as error:
        raise BackendDependencyError(
            "phoneme CTC",
            "transformers and torch are required",
        ) from error
    selected_device = _resolve_torch_device(torch, device)
    key = (model, bool(local_files_only), selected_device)
    cached = _CTC_RUNTIME_CACHE.get(key)
    if cached is not None:
        return cached
    loaded = load_ctc_runtime(
        model=model,
        local_files_only=local_files_only,
        device=selected_device,
    )
    _CTC_RUNTIME_CACHE[key] = loaded
    return loaded


def _ctc_forward_chunk(
    runtime: CTCModelRuntime,
    wav_chunk: np.ndarray,
) -> np.ndarray:
    torch = runtime.torch
    inputs = runtime.processor(
        wav_chunk,
        sampling_rate=runtime.sample_rate_hz,
        return_tensors="pt",
    )
    if hasattr(inputs, "items"):
        model_inputs = {
            key: _move_to_device(value, runtime.device) for key, value in inputs.items()
        }
    else:
        model_inputs = {"input_values": _move_to_device(inputs, runtime.device)}
    inference_context = getattr(torch, "inference_mode", torch.no_grad)
    with inference_context():
        logits = runtime.model(**model_inputs).logits[0]
    return torch.softmax(logits, dim=-1).detach().cpu().numpy().astype(np.float32)


def _decode_ctc_labels(
    processor: Any,
    probs: np.ndarray,
    *,
    drop_special_tokens: bool,
) -> tuple[np.ndarray, list[str]]:
    vocab_size = int(probs.shape[1])
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "convert_ids_to_tokens"):
        labels = [
            str(tokenizer.convert_ids_to_tokens(int(i))) for i in range(vocab_size)
        ]
    else:
        labels = [f"tok_{i}" for i in range(vocab_size)]
    normalized_labels = [_normalize_ctc_token(x) for x in labels]
    if drop_special_tokens:
        keep_idx = [i for i, norm in enumerate(normalized_labels) if norm != ""]
        if keep_idx:
            probs = probs[:, keep_idx]
            labels = [normalized_labels[i] for i in keep_idx]
        else:
            probs = probs[:, :0]
            labels = []
    else:
        labels = [
            norm if norm else str(raw) for norm, raw in zip(normalized_labels, labels)
        ]
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-8)
    return probs, labels


def _keep_chunk_frames(
    times: np.ndarray,
    *,
    chunk_index: int,
    n_chunks: int,
    chunk_start_s: float,
    chunk_end_s: float,
    overlap_s: float,
) -> np.ndarray:
    lo = chunk_start_s if chunk_index == 0 else chunk_start_s + overlap_s / 2.0
    hi = chunk_end_s if chunk_index == n_chunks - 1 else chunk_end_s - overlap_s / 2.0
    return (times >= lo) & (times < hi)


def _ctc_inference_error(error: BaseException, device: str) -> BackendInferenceError:
    hint = ""
    text = str(error).lower()
    if any(
        token in text
        for token in ("out of memory", "cuda oom", "mps backend out of memory")
    ):
        hint = " Reduce chunk_window_s if this was an accelerator memory failure."
    return BackendInferenceError(
        "phoneme CTC",
        f"posterior inference failed on {device}: {error}.{hint}",
    )


def ctc_phone_posteriors(
    stimulus: AudioStimulus,
    *,
    model: str = "bobboyms/wav2vec2-base-en-phoneme-ctc-41h",
    stride_s: float = 0.02,
    local_files_only: bool = True,
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
    drop_special_tokens: bool = True,
    runtime: CTCModelRuntime | None = None,
    device: str = "auto",
    chunk_window_s: float = 30.0,
    chunk_overlap_s: float = 1.0,
) -> FeatureSeries:
    """Extract CTC phone posteriors with cached runtimes and long-audio chunking.

    Audio shorter than ``chunk_window_s`` uses one forward pass. Longer stimuli
    are split with ``plan_audio_chunks`` and interior overlap frames are dropped
    before concatenation.
    """

    mode, _strict = resolve_execution_mode(
        execution_mode=execution_mode,
        strict_dependency=strict_dependency,
    )
    if chunk_window_s <= 0:
        raise ValueError("chunk_window_s must be greater than zero")
    if chunk_overlap_s < 0:
        raise ValueError("chunk_overlap_s must be >= 0")
    if chunk_overlap_s >= chunk_window_s:
        raise ValueError("chunk_overlap_s must be < chunk_window_s")
    requested_device = str(device).strip().lower()
    if requested_device not in _VALID_CTC_DEVICES:
        raise ValueError("device must be one of {'auto', 'cuda', 'mps', 'cpu'}")
    if runtime is not None and runtime.model_id != model:
        raise ValueError(
            f"Preloaded CTC runtime uses '{runtime.model_id}', not '{model}'."
        )
    params = {
        "model": model,
        "stride_s": stride_s,
        "local_files_only": local_files_only,
        "drop_special_tokens": drop_special_tokens,
        "device": requested_device,
        "chunk_window_s": float(chunk_window_s),
        "chunk_overlap_s": float(chunk_overlap_s),
    }
    active_runtime = _get_ctc_runtime(
        model=model,
        local_files_only=local_files_only,
        device=requested_device,
        runtime=runtime,
    )

    try:
        wav = stimulus.samples.astype(np.float32)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        wav_model = _resample_audio_linear(
            wav,
            from_sr=stimulus.sr_hz,
            to_sr=active_runtime.sample_rate_hz,
        )
        duration_s = float(len(wav_model) / active_runtime.sample_rate_hz)
        if duration_s <= float(chunk_window_s):
            raw_probs = _ctc_forward_chunk(active_runtime, wav_model)
            hop_s = max(duration_s / max(1, raw_probs.shape[0]), 1e-6)
            times = times_from_hop(
                raw_probs.shape[0],
                hop_s,
                start_offset_s=stimulus.start_offset_s,
            )
            chunk_count = 1
        else:
            chunks = plan_audio_chunks(
                n_samples=len(wav_model),
                sr_hz=active_runtime.sample_rate_hz,
                window_s=float(chunk_window_s),
                overlap_s=float(chunk_overlap_s),
                start_offset_s=stimulus.start_offset_s,
            )
            probability_parts: list[np.ndarray] = []
            time_parts: list[np.ndarray] = []
            for index, chunk in enumerate(chunks):
                wav_slice = wav_model[chunk.sample_start : chunk.sample_end]
                chunk_probs = _ctc_forward_chunk(active_runtime, wav_slice)
                chunk_duration_s = float(len(wav_slice) / active_runtime.sample_rate_hz)
                chunk_hop_s = max(chunk_duration_s / max(1, chunk_probs.shape[0]), 1e-6)
                chunk_times = times_from_hop(
                    chunk_probs.shape[0],
                    chunk_hop_s,
                    start_offset_s=chunk.start_s,
                )
                keep = _keep_chunk_frames(
                    chunk_times,
                    chunk_index=index,
                    n_chunks=len(chunks),
                    chunk_start_s=chunk.start_s,
                    chunk_end_s=chunk.end_s,
                    overlap_s=float(chunk_overlap_s),
                )
                if not np.any(keep):
                    continue
                probability_parts.append(chunk_probs[keep])
                time_parts.append(chunk_times[keep])
            if not probability_parts:
                raise RuntimeError("CTC chunking produced no posterior frames")
            raw_probs = np.concatenate(probability_parts, axis=0)
            times = np.concatenate(time_parts)
            if len(times) > 1 and np.any(np.diff(times) < 0):
                order = np.argsort(times, kind="mergesort")
                times = times[order]
                raw_probs = raw_probs[order]
            hop_s = max(duration_s / max(1, raw_probs.shape[0]), 1e-6)
            chunk_count = len(chunks)
        probs, labels = _decode_ctc_labels(
            active_runtime.processor,
            raw_probs,
            drop_special_tokens=drop_special_tokens,
        )
    except Exception as error:
        raise _ctc_inference_error(error, active_runtime.device) from error

    md = add_execution_provenance(
        extractor_metadata(
            "speech.phonology.ctc_posteriors",
            params=params,
            extra={
                "backend": "transformers_ctc",
                "device": active_runtime.device,
                "chunk_count": chunk_count,
                "chunk_window_s": float(chunk_window_s),
                "chunk_overlap_s": float(chunk_overlap_s),
            },
        ),
        execution_mode=mode,
        fallback_used=False,
    )
    return FeatureSeries(
        values=probs.astype(np.float32),
        times_s=times,
        dims=("time", "feature"),
        coords={"feature": labels},
        metadata=md,
        timebase=TimebaseSpec(
            kind="audio_hop", hop_s=hop_s, sampling_rate_hz=1.0 / hop_s
        ),
    )


def articulatory_from_posteriors(
    posteriors: FeatureSeries,
    *,
    feature_names: list[str] | None = None,
    renormalize_posteriors: bool = True,
    include_uncertainty: bool = True,
    extractor_name: str = "speech.articulatory.from_posteriors",
) -> FeatureSeries:
    if posteriors.values.ndim != 2:
        raise ValueError(
            "posteriors must be a 2D FeatureSeries with dims (time, feature)"
        )
    labels = [
        str(x)
        for x in posteriors.coords.get(
            "feature", [f"phn_{i}" for i in range(posteriors.values.shape[1])]
        )
    ]
    feature_names = feature_names or list(DEFAULT_ARTICULATORY_FEATURES)
    mat = _labels_to_feature_matrix(labels, feature_names=feature_names)
    p = np.asarray(posteriors.values, dtype=np.float32)
    p = np.clip(p, 0.0, None)
    if renormalize_posteriors:
        denom = np.maximum(p.sum(axis=1, keepdims=True), 1e-8)
        p = p / denom
    vals = p @ mat
    out_names = list(feature_names)
    if include_uncertainty:
        entropy = -np.sum(p * np.log(np.maximum(p, 1e-8)), axis=1, keepdims=True)
        peak = np.max(p, axis=1, keepdims=True)
        vals = np.concatenate(
            [vals, entropy.astype(np.float32), peak.astype(np.float32)], axis=1
        )
        out_names.extend(["posterior_entropy", "posterior_peak"])
    md = extractor_metadata(
        extractor_name,
        params={
            "feature_names": out_names,
            "renormalize_posteriors": renormalize_posteriors,
            "include_uncertainty": include_uncertainty,
        },
        extra={
            "source_extractor": posteriors.metadata.get("extractor_name", "unknown")
        },
    )
    return FeatureSeries(
        values=vals.astype(np.float32),
        times_s=posteriors.times_s,
        dims=("time", "feature"),
        coords={"feature": out_names},
        metadata=md,
        timebase=posteriors.timebase,
    )


def distinctive_from_posteriors(
    posteriors: FeatureSeries,
    *,
    renormalize_posteriors: bool = True,
    include_uncertainty: bool = True,
) -> FeatureSeries:
    """English distinctive-feature occupancy from phone posteriors."""

    return articulatory_from_posteriors(
        posteriors,
        feature_names=list(DEFAULT_DISTINCTIVE_FEATURES),
        renormalize_posteriors=renormalize_posteriors,
        include_uncertainty=include_uncertainty,
        extractor_name="speech.phonology.distinctive_from_posteriors",
    )


def distinctive_from_phoneme_events(
    phonemes: EventSeries,
    *,
    include_confidence: bool = True,
) -> FeatureSeries:
    """English distinctive-feature vectors from phone events."""

    return articulatory_from_phoneme_events(
        phonemes,
        feature_names=list(DEFAULT_DISTINCTIVE_FEATURES),
        include_confidence=include_confidence,
        extractor_name="speech.phonology.distinctive_from_phoneme_events",
    )
