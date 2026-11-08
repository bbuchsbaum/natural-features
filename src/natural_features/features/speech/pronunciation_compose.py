"""Deterministic pronunciation estimates for words absent from a lexicon.

A missing word is rebuilt from lexicon entries by a small, fixed set of English
spelling rules: regular suffixes (``hug`` + ``-able``), possessives and plurals
(``snowflake`` + ``'s``), closed compounds (``mouth`` + ``watering``) and
hyphenated compounds. Nothing is learned from data and no statistical
grapheme-to-phoneme model is used, so the same lexicon always gives the same
result. Every result is an estimate: callers must keep it distinguishable from
dictionary pronunciations. Words that no rule explains stay unresolved.
"""

from __future__ import annotations

from typing import Mapping, Sequence

COMPOSITION_RULES = "affix-compound-v1"

_SIBILANT = frozenset("S Z SH ZH CH JH".split())
_VOICELESS = frozenset("P T K F TH".split())
_VOICELESS_ED = _VOICELESS | frozenset("S SH CH".split())

# Suffixes that begin with a vowel letter may drop a final silent "e" or double
# a final consonant of the base (hug -> huggable, hope -> hoping).
_VOWEL_SUFFIXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("able", ("AH0", "B", "AH0", "L")),
    ("ing", ("IH0", "NG")),
    ("est", ("AH0", "S", "T")),
    ("ish", ("IH0", "SH")),
    ("er", ("ER0",)),
    ("y", ("IY0",)),
)
_CONSONANT_SUFFIXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ness", ("N", "AH0", "S")),
    ("less", ("L", "AH0", "S")),
    ("ment", ("M", "AH0", "N", "T")),
    ("ful", ("F", "AH0", "L")),
    ("ly", ("L", "IY0")),
)
_MIN_STEM = 2
_MIN_COMPOUND_LEFT = 3
_MIN_COMPOUND_RIGHT = 4

Lookup = Mapping[str, Sequence[Sequence[str]]]
Alternative = tuple[tuple[str, ...], str]


def _s_ending(phones: tuple[str, ...]) -> tuple[str, ...]:
    last = phones[-1].rstrip("012")
    if last in _SIBILANT:
        return ("IH0", "Z")
    return ("S",) if last in _VOICELESS else ("Z",)


def _ed_ending(phones: tuple[str, ...]) -> tuple[str, ...]:
    last = phones[-1].rstrip("012")
    if last in {"T", "D"}:
        return ("IH0", "D")
    return ("T",) if last in _VOICELESS_ED else ("D",)


def _demote(phones: Sequence[str]) -> tuple[str, ...]:
    """Primary stress becomes secondary in the non-initial part of a compound."""
    return tuple(p[:-1] + "2" if p.endswith("1") else p for p in phones)


def _vowel_stems(stem: str, entries: Lookup) -> list[str]:
    """Candidate bases before a vowel-initial suffix, most plausible first."""
    if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiouy":
        # Doubled consonant (hugg-able): the base has one (hug).
        return [stem[:-1], stem]
    if stem.endswith("i"):
        return [stem[:-1] + "y", stem]
    if (
        len(stem) > 2
        and stem[-1] not in "aeiouwxy"
        and stem[-2] in "aeiou"
        and stem[-3] not in "aeiou"
        and stem + "e" in entries
    ):
        # A single consonant after a single vowel would have been doubled had
        # the base ended there (hop -> hopping), so hoping comes from hope.
        return [stem + "e"]
    return [stem, stem + "e"]


def _consonant_stems(stem: str) -> list[str]:
    return [stem, stem[:-1] + "y"] if stem.endswith("i") else [stem]


class _Composer:
    def __init__(self, entries: Lookup, max_depth: int) -> None:
        self.entries = entries
        self.max_depth = max_depth

    def lookup(
        self, word: str, depth: int, *, compounds: bool = True
    ) -> list[Alternative]:
        if word in self.entries:
            return [(tuple(p), word) for p in self.entries[word]]
        return self.compose(word, depth, compounds=compounds) if depth > 0 else []

    def _attach(
        self, stems: list[str], depth: int, ending, suffix: str
    ) -> list[Alternative]:
        out = []
        for stem in dict.fromkeys(stems):
            if len(stem) < _MIN_STEM:
                continue
            # A suffix never attaches to a guessed compound (chip+lee+y).
            for phones, label in self.lookup(stem, depth - 1, compounds=False):
                tail = ending(phones) if callable(ending) else ending
                out.append(((*phones, *tail), f"{label}+{suffix}"))
        return out

    def suffixes(self, word: str, depth: int) -> list[Alternative]:
        out: list[Alternative] = []
        if word.endswith("'s"):
            out += self._attach([word[:-2]], depth, _s_ending, "'s")
        elif word.endswith("s'"):
            # Plural possessive sounds like the plural (dogs' = dogs).
            out += self._attach([word[:-1]], depth, (), "'")
        if "'" in word:
            return out
        if word.endswith("ies"):
            out += self._attach([word[:-3] + "y"], depth, _s_ending, "s")
        elif word.endswith("es") and word[:-2].endswith(("s", "x", "z", "ch", "sh")):
            out += self._attach([word[:-2]], depth, _s_ending, "es")
        elif word.endswith("s") and not word.endswith("ss"):
            out += self._attach([word[:-1]], depth, _s_ending, "s")
        if word.endswith("ied"):
            out += self._attach([word[:-3] + "y"], depth, _ed_ending, "ed")
        elif word.endswith("ed"):
            stem = word[:-2]
            out += self._attach(
                _vowel_stems(stem, self.entries), depth, _ed_ending, "ed"
            )
        for suffix, phones in _VOWEL_SUFFIXES:
            if word.endswith(suffix):
                stems = _vowel_stems(word[: -len(suffix)], self.entries)
                out += self._attach(stems, depth, phones, suffix)
        for suffix, phones in _CONSONANT_SUFFIXES:
            if word.endswith(suffix):
                stems = _consonant_stems(word[: -len(suffix)])
                out += self._attach(stems, depth, phones, suffix)
        return out

    def compounds(self, word: str, depth: int) -> list[Alternative]:
        out: list[Alternative] = []
        for i in range(_MIN_COMPOUND_LEFT, len(word) - _MIN_COMPOUND_RIGHT + 1):
            left, right = word[:i], word[i:]
            if left not in self.entries:
                continue
            for lp, ll in self.lookup(left, 0):
                for rp, rl in self.lookup(right, depth - 1, compounds=False):
                    out.append(((*lp, *_demote(rp)), f"{ll}+{rl}"))
        return out

    def hyphenated(self, word: str, depth: int) -> list[Alternative]:
        parts = [p for p in word.split("-")]
        if len(parts) < 2 or not all(parts):
            return []
        combos: list[Alternative] = [((), "")]
        for part in parts:
            options = self.lookup(part, depth - 1)
            if not options:
                return []
            combos = [
                ((*cp, *op), f"{cl}-{ol}" if cl else ol)
                for cp, cl in combos
                for op, ol in options
            ]
        return combos

    def compose(
        self, word: str, depth: int, *, compounds: bool = True
    ) -> list[Alternative]:
        if depth <= 0 or not word:
            return []
        if "-" in word:
            return self.hyphenated(word, depth)
        # Suffix analyses take precedence: arbitrary splits of an inflected word
        # into two dictionary entries (rust + les) are far more often wrong.
        found = self.suffixes(word, depth)
        if not found and compounds:
            found = self.compounds(word, depth)
        return found


def compose_pronunciation(
    word: str, entries: Lookup, *, max_depth: int = 2
) -> tuple[tuple[tuple[str, ...], ...], tuple[str, ...]]:
    """Return estimated variants and their derivation labels for a missing word.

    ``word`` must already be normalized. Variants are deduplicated in rule
    order; distinct analyses that disagree are all returned, so an ambiguous
    word is never resolved silently. ``max_depth`` bounds how many rules may be
    stacked (``unhuggable`` would need three). An empty result means no rule
    applies. The word's own entry, if any, is never consulted.
    """
    if not 1 <= max_depth <= 3:
        raise ValueError("max_depth must be between 1 and 3")
    found: dict[tuple[str, ...], str] = {}
    for phones, label in _Composer(entries, max_depth).compose(word, max_depth):
        found.setdefault(phones, f"composed:{label}")
    return tuple(found), tuple(found.values())
