"""Experimental, untimed English pronunciations from pinned dictionary data.

No recognizer, G2P fallback, timing, or confidence estimation is performed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import gzip
import hashlib
from importlib import metadata, resources
import json
from pathlib import Path
import re
from typing import Mapping, Sequence
import unicodedata

from natural_features.features.common import extractor_metadata

VOWELS = tuple("AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split())
PHONES = tuple(
    sorted(
        (*VOWELS, *"B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH".split())
    )
)
_TOKEN = re.compile(
    r"\d+(?:[.,:/]\d+)*|['’‘]?\w+(?:['’‘\-‐‑]\w+)*['’‘]?|[^\w\s]", re.UNICODE
)
_PUNCTUATION = set(".,!?;:()[]{}\"'’‘“”«»—–-…")
_TRANSLATION = str.maketrans({"’": "'", "‘": "'", "‐": "-", "‑": "-"})
NORMALIZATION = "nfc-lower-apostrophe-hyphen-v2"


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def provenance(name: str, params: dict, **extra: object) -> dict:
    """Include an installed version and hashes of the implementing source files."""
    try:
        version = metadata.version("natural-features")
    except metadata.PackageNotFoundError:
        version = "uninstalled"
    params = json.loads(json.dumps(params, allow_nan=False))
    extra = json.loads(json.dumps(extra, allow_nan=False))
    return extractor_metadata(
        name,
        params,
        code_version=version,
        extra={
            "params": params,
            "package_version": version,
            "implementation_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (
                    Path(__file__),
                    Path(__file__).with_name("phone_sequences.py"),
                )
                if p.exists()
            },
            **extra,
        },
    )


def normalize_word(word: str) -> str:
    word = unicodedata.normalize("NFC", word).translate(_TRANSLATION).lower()
    return (
        word[1:-1]
        if len(word) > 2 and word.startswith("'") and word.endswith("'")
        else word
    )


def validate_phones(phones: Sequence[str]) -> tuple[str, ...]:
    if isinstance(phones, str) or not phones:
        raise ValueError(
            "pronunciation must be a nonempty sequence of ARPABET phone labels"
        )
    out = tuple(phones)
    for phone in out:
        if not isinstance(phone, str):
            raise ValueError("phone labels must be strings")
        base = phone.rstrip("012")
        if base not in PHONES or phone not in (
            tuple(base + stress for stress in "012") if base in VOWELS else (base,)
        ):
            raise ValueError(
                f"unsupported ARPABET phone or missing vowel stress: {phone!r}"
            )
    return out


@dataclass(frozen=True)
class PronunciationLexicon:
    entries: Mapping[str, tuple[tuple[str, ...], ...]]
    labels: Mapping[str, tuple[str, ...]]
    metadata: dict

    @classmethod
    def from_mapping(
        cls, entries: Mapping[str, Sequence[Sequence[str]]], *, name: str
    ) -> PronunciationLexicon:
        normalized = {}
        for word, variants in entries.items():
            key = normalize_word(word)
            if key in normalized or not variants:
                raise ValueError(
                    f"duplicate normalized key or empty variants: {word!r}"
                )
            normalized[key] = tuple(validate_phones(p) for p in variants)
        return cls(
            normalized,
            {
                w: tuple(f"{w}[{i}]" for i in range(len(v)))
                for w, v in normalized.items()
            },
            {
                "name": name,
                "sha256": digest(normalized),
                "origin": "user_supplied_unverified",
            },
        )


@lru_cache(maxsize=1)
def load_cmudict() -> PronunciationLexicon:
    root = resources.files("natural_features.features.speech").joinpath("data")
    manifest = json.loads(root.joinpath("cmudict.json").read_text())
    raw = gzip.decompress(root.joinpath("cmudict.dict.gz").read_bytes())
    if hashlib.sha256(raw).hexdigest() != manifest["sha256"]:
        raise ValueError("bundled CMUdict checksum mismatch")
    entries: dict[str, list[tuple[str, ...]]] = {}
    labels: dict[str, list[str]] = {}
    for line in raw.decode("utf-8").splitlines():
        fields = line.split("#", 1)[0].split()
        if not fields or fields[0].startswith(";;;"):
            continue
        key = re.sub(r"\(\d+\)$", "", fields[0]).lower()
        entries.setdefault(key, []).append(validate_phones(fields[1:]))
        labels.setdefault(key, []).append(fields[0])
    return PronunciationLexicon(
        {k: tuple(v) for k, v in entries.items()},
        {k: tuple(v) for k, v in labels.items()},
        manifest,
    )


@dataclass(frozen=True)
class WordPronunciation:
    index: int
    start: int
    end: int
    text: str
    normalized: str
    alternatives: tuple[tuple[str, ...], ...]
    alternative_labels: tuple[str, ...]
    selected: int | None
    origin: str
    reason: str | None = None

    def __post_init__(self) -> None:
        if len(self.alternatives) != len(self.alternative_labels):
            raise ValueError("alternative labels must match alternatives")
        for variant in self.alternatives:
            validate_phones(variant)
        if self.selected is not None and (
            type(self.selected) is not int
            or not 0 <= self.selected < len(self.alternatives)
        ):
            raise ValueError("invalid pronunciation selection")

    @property
    def phones(self) -> tuple[str, ...] | None:
        return None if self.selected is None else self.alternatives[self.selected]


@dataclass(frozen=True)
class PronunciationSequence:
    item_id: str
    text: str
    words: tuple[WordPronunciation, ...]
    metadata: dict
    schema: str = "PronunciationSequence/v1"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.item_id, str)
            or not self.item_id
            or self.schema != "PronunciationSequence/v1"
        ):
            raise ValueError(
                "a nonempty item ID and supported sequence schema are required"
            )
        previous = 0
        for i, word in enumerate(self.words):
            if (
                word.index != i
                or not previous <= word.start < word.end <= len(self.text)
                or self.text[word.start : word.end] != word.text
            ):
                raise ValueError(
                    "word indices and source spans must preserve original text order"
                )
            previous = word.end
        if (
            self.metadata.get("text_sha256")
            != hashlib.sha256(self.text.encode()).hexdigest()
        ):
            raise ValueError("text checksum mismatch")

    @property
    def eligible(self) -> bool:
        return bool(self.words) and all(w.selected is not None for w in self.words)

    @property
    def phones(self) -> tuple[str, ...]:
        if not self.eligible:
            raise ValueError(
                f"{self.item_id}: empty, unresolved or unselected pronunciation sequence"
            )
        return tuple(p for w in self.words for p in w.phones or ())

    @property
    def coverage(self) -> dict:
        n = len(self.words)
        known = sum(bool(w.alternatives) for w in self.words)
        return {
            "word_count": n,
            "resolved_words": known,
            "selected_words": sum(w.selected is not None for w in self.words),
            "word_coverage": known / n if n else None,
            "eligible": self.eligible,
            "unresolved_indices": [w.index for w in self.words if not w.alternatives],
            "unselected_indices": [
                w.index for w in self.words if w.alternatives and w.selected is None
            ],
        }

    def to_dict(self) -> dict:
        return {**asdict(self), "coverage": self.coverage}

    @classmethod
    def from_dict(cls, data: dict) -> PronunciationSequence:
        words = tuple(
            WordPronunciation(
                **{
                    **w,
                    "alternatives": tuple(tuple(p) for p in w["alternatives"]),
                    "alternative_labels": tuple(w["alternative_labels"]),
                }
            )
            for w in data["words"]
        )
        return cls(
            data["item_id"], data["text"], words, data["metadata"], data["schema"]
        )


def extract_pronunciations(
    items: Mapping[str, str],
    *,
    selection: str = "retain",
    lexicon: PronunciationLexicon | None = None,
    supplemental: PronunciationLexicon | None = None,
    overrides: Mapping[str, Mapping[int, Sequence[str]]] | None = None,
) -> list[PronunciationSequence]:
    """Preserve every item and unresolved token. Selection is never inferred.

    ``retain`` selects only unambiguous entries; ``dictionary_order`` explicitly
    chooses the first variant. Overrides supersede lookup for one item/word only.
    """
    if selection not in {"retain", "dictionary_order"}:
        raise ValueError("selection must be 'retain' or 'dictionary_order'")
    lexicon = load_cmudict() if lexicon is None else lexicon
    overrides = overrides or {}
    if set(overrides) - set(items):
        raise ValueError("override item ID does not exist")

    def _known(key: str) -> bool:
        return key in lexicon.entries or (
            supplemental is not None and key in supplemental.entries
        )

    result = []
    for item_id, text in items.items():
        if not isinstance(text, str):
            raise TypeError("transcripts must be strings")
        words = []
        item_overrides = overrides.get(item_id, {})
        for match in _TOKEN.finditer(text):
            raw = match.group()
            if raw in _PUNCTUATION:
                continue
            key = normalize_word(raw)
            unquoted = key.strip("'")
            if unquoted and not _known(key) and _known(unquoted):
                # Unmatched quotation marks ('Stop, he said) are not part of the
                # word; listed apostrophe forms ('em, dogs') are looked up first.
                key = unquoted
            index = len(words)
            origin = "dictionary"
            source = (
                supplemental
                if supplemental and key in supplemental.entries
                else lexicon
            )
            if source is supplemental:
                origin = "supplemental_unverified"
            alternatives = source.entries.get(key, ())
            labels = source.labels.get(key, ())
            chosen = (
                0
                if len(alternatives) == 1
                or (alternatives and selection == "dictionary_order")
                else None
            )
            reason = (
                None
                if chosen is not None
                else "alternatives_unselected"
                if alternatives
                else "not_in_lexicon"
            )
            if index in item_overrides:
                variant = validate_phones(item_overrides[index])
                if variant in alternatives:
                    chosen = alternatives.index(variant)
                else:
                    chosen = len(alternatives)
                    alternatives = (*alternatives, variant)
                    labels = (*labels, "item_override")
                origin, reason = "item_override_unverified", None
            if not alternatives:
                origin = "unresolved"
            words.append(
                WordPronunciation(
                    index,
                    match.start(),
                    match.end(),
                    raw,
                    key,
                    alternatives,
                    labels,
                    chosen,
                    origin,
                    reason,
                )
            )
        if any(type(i) is not int or not 0 <= i < len(words) for i in item_overrides):
            raise ValueError(f"{item_id}: override word index does not exist")
        md = provenance(
            "speech.transcript.pronunciations",
            {
                "selection": selection,
                "normalization": NORMALIZATION,
                "lexicon": lexicon.metadata,
                "supplemental": supplemental.metadata if supplemental else None,
                "overrides": {str(k): list(v) for k, v in item_overrides.items()},
            },
            text_sha256=hashlib.sha256(text.encode()).hexdigest(),
            language="en-US",
            phone_namespace="CMU-ARPABET",
            stress="original_0_1_2",
            pronunciation_scope="canonical_not_audio_observed",
        )
        result.append(PronunciationSequence(item_id, text, tuple(words), md))
    return result


def coverage_report(
    sequences: Sequence[PronunciationSequence], *, pairs: Sequence[tuple[str, str]] = ()
) -> dict:
    """Prioritize unresolved types by affected pairs; losses are not additive."""
    by_id = {s.item_id: s for s in sequences}
    if len(by_id) != len(sequences):
        raise ValueError("duplicate item IDs")
    if any(a not in by_id or b not in by_id or a == b for a, b in pairs):
        raise ValueError("pairs must reference distinct existing items")
    unresolved: dict[str, dict] = {}
    for seq in sequences:
        for word in seq.words:
            if not word.alternatives:
                row = unresolved.setdefault(
                    word.normalized,
                    {"token": word.normalized, "occurrences": 0, "item_ids": []},
                )
                row["occurrences"] += 1
                if seq.item_id not in row["item_ids"]:
                    row["item_ids"].append(seq.item_id)
    for row in unresolved.values():
        row["affected_pairs"] = sum(
            a in row["item_ids"] or b in row["item_ids"] for a, b in pairs
        )
    return {
        "items": {s.item_id: s.coverage for s in sequences},
        "eligible_items": sum(s.eligible for s in sequences),
        "total_items": len(sequences),
        "eligible_pairs": sum(
            by_id[a].eligible and by_id[b].eligible for a, b in pairs
        ),
        "total_pairs": len(pairs),
        "unresolved": sorted(
            unresolved.values(),
            key=lambda r: (-r["affected_pairs"], -r["occurrences"], r["token"]),
        ),
    }
