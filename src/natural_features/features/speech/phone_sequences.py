"""Experimental fixed-vocabulary phone features and descriptive sequence distance."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .pronunciation import PHONES, VOWELS, PronunciationSequence, digest, provenance


@dataclass(frozen=True)
class ItemFeatures:
    """Untimed item-by-coordinate matrix. False validity never means measured zero."""

    item_ids: tuple[str, ...]
    names: tuple[str, ...]
    values: np.ndarray
    valid: np.ndarray
    metadata: dict
    schema: str = "ItemFeatures/v1"

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64).copy()
        valid = np.asarray(self.valid, dtype=bool).copy()
        if (
            self.schema != "ItemFeatures/v1"
            or values.shape != (len(self.item_ids), len(self.names))
            or valid.shape != values.shape
        ):
            raise ValueError("invalid item-feature schema or dimensions")
        if len(set(self.item_ids)) != len(self.item_ids) or len(set(self.names)) != len(
            self.names
        ):
            raise ValueError("item IDs and feature coordinates must be unique")
        if any(not isinstance(x, str) or not x for x in (*self.item_ids, *self.names)):
            raise ValueError("item IDs and coordinates must be nonempty strings")
        if np.any(valid & ~np.isfinite(values)):
            raise ValueError("valid features must be finite")
        values.setflags(write=False)
        valid.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "valid", valid)

    @property
    def eligible(self) -> np.ndarray:
        return self.valid.all(axis=1) & bool(self.names)

    @staticmethod
    def _paths(prefix: str | Path) -> tuple[Path, Path]:
        # Append, never replace: dotted prefixes (sub-01.run-1) must not collide.
        prefix = Path(prefix)
        return (
            prefix.with_name(prefix.name + ".npz"),
            prefix.with_name(prefix.name + ".json"),
        )

    def save(self, prefix: str | Path) -> None:
        data_path, json_path = self._paths(prefix)
        np.savez_compressed(data_path, values=self.values, valid=self.valid)
        import hashlib

        payload = {
            "schema": self.schema,
            "item_ids": self.item_ids,
            "names": self.names,
            "metadata": self.metadata,
            "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        }
        json_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    @classmethod
    def load(cls, prefix: str | Path) -> ItemFeatures:
        import hashlib

        data_path, json_path = cls._paths(prefix)
        payload = json.loads(json_path.read_text())
        if hashlib.sha256(data_path.read_bytes()).hexdigest() != payload["data_sha256"]:
            raise ValueError("feature data checksum mismatch")
        with np.load(data_path, allow_pickle=False) as data:
            return cls(
                tuple(payload["item_ids"]),
                tuple(payload["names"]),
                data["values"],
                data["valid"],
                payload["metadata"],
                payload["schema"],
            )


@dataclass(frozen=True)
class NgramSchema:
    n: int = 2
    boundary: str = "within"
    stress: str = "strip"
    normalization: str = "frequency"

    def __post_init__(self) -> None:
        if type(self.n) is not int or self.n not in {1, 2}:
            raise ValueError("fixed inventory supports unigrams and bigrams")
        if self.boundary not in {"within", "cross", "all"} or (
            self.n == 1 and self.boundary != "within"
        ):
            raise ValueError("invalid ngram word-boundary policy")
        if self.stress not in {"strip", "keep"} or self.normalization not in {
            "count",
            "frequency",
        }:
            raise ValueError("invalid stress or normalization policy")

    @property
    def inventory(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                p + s
                for p in PHONES
                for s in ("012" if self.stress == "keep" and p in VOWELS else ("",))
            )
        )

    @property
    def coordinates(self) -> tuple[str, ...]:
        return tuple(
            " ".join(ngram)
            for ngram in itertools.product(self.inventory, repeat=self.n)
        )

    def to_dict(self) -> dict:
        payload = {
            "schema": "PhoneNgrams/v1",
            "n": self.n,
            "boundary": self.boundary,
            "stress": self.stress,
            "normalization": self.normalization,
            "coordinates": self.coordinates,
            "unseen_policy": "error",
            "fit": "fixed_inventory_no_fit",
        }
        return {**payload, "sha256": digest(payload)}

    @classmethod
    def from_dict(cls, payload: dict) -> NgramSchema:
        result = cls(
            **{k: payload[k] for k in ("n", "boundary", "stress", "normalization")}
        )
        if (
            digest({k: v for k, v in payload.items() if k != "sha256"})
            != payload.get("sha256")
            or payload["sha256"] != result.to_dict()["sha256"]
        ):
            raise ValueError("unsupported or corrupted ngram schema")
        return result

    def transform(self, sequences: Sequence[PronunciationSequence]) -> ItemFeatures:
        names = self.coordinates
        idx = {name: i for i, name in enumerate(names)}
        values = np.zeros((len(sequences), len(names)))
        valid = np.zeros_like(values, dtype=bool)
        for row, seq in enumerate(sequences):
            words = [
                None
                if w.phones is None
                else tuple(
                    p.rstrip("012") if self.stress == "strip" else p for p in w.phones
                )
                for w in seq.words
            ]
            grams = []
            if self.n == 1:
                grams = [(p,) for word in words if word is not None for p in word]
            else:
                if self.boundary in {"within", "all"}:
                    grams.extend(
                        (a, b)
                        for word in words
                        if word is not None
                        for a, b in zip(word, word[1:])
                    )
                if self.boundary in {"cross", "all"}:
                    # Adjacent source words only: an unresolved word remains a gap.
                    grams.extend(
                        (a[-1], b[0]) for a, b in zip(words, words[1:]) if a and b
                    )
            for gram in grams:
                values[row, idx[" ".join(gram)]] += 1
            if self.normalization == "frequency" and grams:
                values[row] /= len(grams)
            valid[row] = seq.eligible
        return ItemFeatures(
            tuple(s.item_id for s in sequences),
            names,
            values,
            valid,
            provenance(
                "speech.transcript.phone_ngrams",
                self.to_dict(),
                sources=[s.metadata for s in sequences],
                zero_vector_distance="undefined",
            ),
        )


def pronunciation_counts(sequences: Sequence[PronunciationSequence]) -> ItemFeatures:
    values = np.zeros((len(sequences), 3))
    valid = np.zeros_like(values, dtype=bool)
    for i, s in enumerate(sequences):
        selected = [p for w in s.words for p in w.phones or ()]
        values[i] = (
            len(s.words),
            len(selected),
            sum(p.rstrip("012") in VOWELS for p in selected),
        )
        valid[i] = True, s.eligible, s.eligible
    return ItemFeatures(
        tuple(s.item_id for s in sequences),
        ("word_count", "phone_count", "syllable_count"),
        values,
        valid,
        provenance(
            "speech.transcript.counts",
            {
                "syllables": "selected_vowel_nuclei",
                "word_count": "nonpunctuation_source_tokens",
            },
            sources=[s.metadata for s in sequences],
        ),
    )


# Explicit categorical bundles, independently specified for this English model.
# Consonant place/manner follows the existing package map; vowels use endpoint
# height/backness/rounding categories. These are modeling conventions, not EMA.
_CONSONANTS = {
    "P": ("bilabial", "stop", "unvoiced"),
    "B": ("bilabial", "stop", "voiced"),
    "M": ("bilabial", "nasal", "voiced"),
    "F": ("labiodental", "fricative", "unvoiced"),
    "V": ("labiodental", "fricative", "voiced"),
    "TH": ("dental", "fricative", "unvoiced"),
    "DH": ("dental", "fricative", "voiced"),
    "T": ("alveolar", "stop", "unvoiced"),
    "D": ("alveolar", "stop", "voiced"),
    "N": ("alveolar", "nasal", "voiced"),
    "S": ("alveolar", "fricative", "unvoiced"),
    "Z": ("alveolar", "fricative", "voiced"),
    "SH": ("postalveolar", "fricative", "unvoiced"),
    "ZH": ("postalveolar", "fricative", "voiced"),
    "CH": ("postalveolar", "affricate", "unvoiced"),
    "JH": ("postalveolar", "affricate", "voiced"),
    "K": ("velar", "stop", "unvoiced"),
    "G": ("velar", "stop", "voiced"),
    "NG": ("velar", "nasal", "voiced"),
    "HH": ("glottal", "fricative", "unvoiced"),
    "L": ("alveolar", "lateral", "voiced"),
    "R": ("alveolar", "approximant", "voiced"),
    "Y": ("palatal", "approximant", "voiced"),
    "W": ("labial_velar", "approximant", "voiced"),
}
# (height, backness, rounding, rhoticity); diphthongs preserve ordered endpoints.
_VOWEL_POINTS = {
    "IY": ("high", "front", "unrounded", "plain"),
    "IH": ("near_high", "front", "unrounded", "plain"),
    "EH": ("mid", "front", "unrounded", "plain"),
    "AE": ("low", "front", "unrounded", "plain"),
    "AA": ("low", "back", "unrounded", "plain"),
    "AO": ("mid", "back", "rounded", "plain"),
    "AH": ("mid", "central", "unrounded", "plain"),
    "ER": ("mid", "central", "unrounded", "rhotic"),
    "UH": ("near_high", "back", "rounded", "plain"),
    "UW": ("high", "back", "rounded", "plain"),
}
_VOWEL_PATHS = {p: (v, v) for p, v in _VOWEL_POINTS.items()}
_VOWEL_PATHS.update(
    {
        "EY": (_VOWEL_POINTS["EH"], _VOWEL_POINTS["IH"]),
        "AY": (("low", "central", "unrounded", "plain"), _VOWEL_POINTS["IH"]),
        "OW": (_VOWEL_POINTS["AO"], _VOWEL_POINTS["UH"]),
        "AW": (("low", "central", "unrounded", "plain"), _VOWEL_POINTS["UH"]),
        "OY": (_VOWEL_POINTS["AO"], _VOWEL_POINTS["IH"]),
    }
)


def phone_feature_bundles() -> dict:
    """Inspectable ordered feature mapping; returned data do not depend on a model."""
    return {
        "schema": "EnglishPhoneBundles/v1",
        "consonants": dict(_CONSONANTS),
        "vowels": dict(_VOWEL_PATHS),
        "consonant_groups": ["place", "manner", "voicing"],
        "vowel_groups": ["height", "backness", "rounding", "rhoticity"],
        "stress": "separate_0_1_2",
        "scope": "canonical_English_modeling_conventions",
    }


def articulatory_sequence(sequence: PronunciationSequence) -> list[dict]:
    """One ordered feature bundle per selected phone; refuses gaps and ambiguity."""
    sequence.phones  # validate completeness before building any result
    return [
        {
            "word_index": w.index,
            "phone_index": i,
            "label": p,
            "stress": int(p[-1]) if p[-1] in "012" else None,
            "features": _CONSONANTS.get(
                p.rstrip("012"), _VOWEL_PATHS.get(p.rstrip("012"))
            ),
        }
        for w in sequence.words
        for i, p in enumerate(w.phones or ())
    ]


@dataclass(frozen=True)
class PhoneEditDistance:
    """Group-weighted substitutions, symmetric unit indels; no metric claim."""

    weighted: bool = True
    consonant_weights: tuple[float, ...] = (1.0, 1.0, 1.0)
    vowel_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    stress_weight: float = 0.0
    normalize: bool = True

    def __post_init__(self) -> None:
        for weights, size in ((self.consonant_weights, 3), (self.vowel_weights, 4)):
            if (
                len(weights) != size
                or not np.isfinite(weights).all()
                or min(weights) < 0
                or sum(weights) <= 0
            ):
                raise ValueError(
                    "feature weights must be finite, nonnegative and have positive sum"
                )
        if not np.isfinite(self.stress_weight) or self.stress_weight < 0:
            raise ValueError("stress_weight must be finite and nonnegative")

    def to_dict(self) -> dict:
        return {
            "schema": "PhoneEditDistance/v1",
            "weighted": self.weighted,
            "consonant_weights": self.consonant_weights,
            "vowel_weights": self.vowel_weights,
            "stress_weight": self.stress_weight,
            "insertion": 1.0,
            "deletion": 1.0,
            "normalization": "max_phone_count" if self.normalize else "none",
            "boundary": "word_boundaries_not_edit_symbols",
            "mapping": phone_feature_bundles(),
            "metric_claim": False,
        }

    def substitution(self, a: str, b: str) -> float:
        # Accept stress-stripped or original CMU labels, never arbitrary symbols.
        for p in (a, b):
            base = p.rstrip("012")
            if base not in PHONES or p not in (
                (base, *(base + s for s in "012")) if base in VOWELS else (base,)
            ):
                raise ValueError(f"unsupported phone: {p!r}")
        pa, pb = a.rstrip("012"), b.rstrip("012")
        stress_term = 0.0
        if self.stress_weight and pa in VOWELS and pb in VOWELS:
            if a[-1] not in "012" or b[-1] not in "012":
                raise ValueError(
                    "stress-weighted distance requires original vowel stress labels"
                )
            stress_term = self.stress_weight * (a[-1] != b[-1])
        if not self.weighted:
            # Vowel pairs use the weighted path's convex identity + stress form;
            # consonant and cross-class substitutions carry no stress term.
            if pa in VOWELS and pb in VOWELS:
                return (float(pa != pb) + stress_term) / (1.0 + self.stress_weight)
            return float(pa != pb)
        if (pa in VOWELS) != (pb in VOWELS):
            return 1.0
        if pa not in VOWELS:
            return float(
                np.average(
                    np.array(_CONSONANTS[pa]) != np.array(_CONSONANTS[pb]),
                    weights=self.consonant_weights,
                )
            )
        diff = (np.array(_VOWEL_PATHS[pa]) != np.array(_VOWEL_PATHS[pb])).mean(axis=0)
        value = float(np.dot(diff, self.vowel_weights)) + stress_term
        return value / (sum(self.vowel_weights) + self.stress_weight)

    def __call__(self, a: Sequence[str], b: Sequence[str]) -> float:
        for p in (*a, *b):
            self.substitution(p, p)
        previous = np.arange(len(b) + 1, dtype=float)
        for i, pa in enumerate(a, 1):
            current = np.empty(len(b) + 1)
            current[0] = i
            for j, pb in enumerate(b, 1):
                current[j] = min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + self.substitution(pa, pb),
                )
            previous = current
        value = float(previous[-1])
        return value / max(len(a), len(b), 1) if self.normalize else value


def sequence_distances(
    sequences: Sequence[PronunciationSequence],
    distance: PhoneEditDistance | None = None,
) -> np.ndarray:
    distance = PhoneEditDistance() if distance is None else distance
    result = np.full((len(sequences), len(sequences)), np.nan)
    for i, a in enumerate(sequences):
        if not a.eligible:
            continue
        for j in range(i, len(sequences)):
            b = sequences[j]
            if b.eligible:
                result[i, j] = result[j, i] = distance(a.phones, b.phones)
    return result
