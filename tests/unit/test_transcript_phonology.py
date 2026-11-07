from __future__ import annotations

import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from natural_features.features.speech.pronunciation import (
    PHONES,
    PronunciationLexicon,
    PronunciationSequence,
    coverage_report,
    extract_pronunciations,
    load_cmudict,
)
from natural_features.features.speech.phone_sequences import (
    ItemFeatures,
    NgramSchema,
    PhoneEditDistance,
    articulatory_sequence,
    pronunciation_counts,
    sequence_distances,
)


def lexicon():
    return PronunciationLexicon.from_mapping(
        {
            "cat": [("K", "AE1", "T")],
            "tack": [("T", "AE1", "K")],
            "a": [("AH0",)],
            "read": [("R", "EH1", "D"), ("R", "IY1", "D")],
            "it's": [("IH1", "T", "S")],
            "cat's": [("K", "AE1", "T", "S")],
        },
        name="test-exact",
    )


@pytest.mark.smoke
def test_pinned_dictionary_exact_homophones_and_alternatives():
    dictionary = load_cmudict()
    assert dictionary.entries["cat"] == (("K", "AE1", "T"),)
    assert dictionary.entries["read"] == (("R", "EH1", "D"), ("R", "IY1", "D"))
    seqs = extract_pronunciations(
        {"a": "sea", "b": "see"}, selection="dictionary_order"
    )
    assert seqs[0].phones == seqs[1].phones
    assert sequence_distances(seqs)[0, 1] == 0


def test_alternatives_explicit_selection_and_overrides_keep_originals():
    (a,) = extract_pronunciations({"a": "read cat"}, lexicon=lexicon())
    assert not a.eligible and a.coverage["word_coverage"] == 1
    with pytest.raises(ValueError, match="unselected"):
        _ = a.phones
    (selected,) = extract_pronunciations(
        {"a": "read cat"}, lexicon=lexicon(), overrides={"a": {0: ["R", "IY1", "D"]}}
    )
    assert selected.words[0].selected == 1
    assert selected.words[0].alternatives == a.words[0].alternatives
    assert selected.words[0].origin == "item_override_unverified"
    (first,) = extract_pronunciations(
        {"a": "read"}, selection="dictionary_order", lexicon=lexicon()
    )
    assert first.phones == ("R", "EH1", "D")


def test_tokenization_punctuation_possessives_and_unresolved_symbols():
    text = "‘CAT’, it's cat’s dogs’ 3.14 snow-man 🙂"
    (seq,) = extract_pronunciations({"s": text}, lexicon=lexicon())
    assert [w.normalized for w in seq.words] == [
        "cat",
        "it's",
        "cat's",
        "dogs'",
        "3.14",
        "snow-man",
        "🙂",
    ]
    assert [w.origin for w in seq.words[:3]] == ["dictionary"] * 3
    assert seq.coverage["unresolved_indices"] == [3, 4, 5, 6]
    assert all(text[w.start : w.end] == w.text for w in seq.words)
    assert "onset" not in json.dumps(seq.to_dict())
    assert "confidence" not in json.dumps(seq.to_dict())


def test_roundtrip_repeatability_empty_and_no_silent_gaps():
    seqs = extract_pronunciations(
        {"s": "cat missing tack", "empty": "..."}, lexicon=lexicon()
    )
    again = extract_pronunciations(
        {"s": "cat missing tack", "empty": "..."}, lexicon=lexicon()
    )
    assert [s.to_dict() for s in seqs] == [s.to_dict() for s in again]
    for s in seqs:
        assert PronunciationSequence.from_dict(json.loads(json.dumps(s.to_dict()))) == s
    cross = NgramSchema(boundary="cross", normalization="count").transform(seqs)
    assert cross.values.sum() == 0 and not cross.valid.any()
    within = NgramSchema(normalization="count").transform(seqs)
    assert within.values[0].sum() == 4  # diagnostic partial counts retained but invalid
    assert not within.valid.any()
    assert seqs[1].coverage["word_coverage"] is None


def test_coverage_pair_loss_and_supplement():
    seqs = extract_pronunciations(
        {"s": "x x", "t": "cat", "u": "x cat"}, lexicon=lexicon()
    )
    report = coverage_report(seqs, pairs=[("s", "t"), ("u", "t"), ("s", "u")])
    assert report["unresolved"] == [
        {"token": "x", "occurrences": 3, "item_ids": ["s", "u"], "affected_pairs": 3}
    ]
    supplement = PronunciationLexicon.from_mapping(
        {"x": [("EH1", "K", "S")]}, name="review-list"
    )
    (repaired,) = extract_pronunciations(
        {"s": "x"}, lexicon=lexicon(), supplemental=supplement
    )
    assert repaired.eligible and repaired.words[0].origin == "supplemental_unverified"


def test_fixed_coordinates_boundaries_order_and_normalization(tmp_path):
    seqs = extract_pronunciations(
        {"s": "cat tack", "r": "tack cat", "c": "cat", "t": "tack"}, lexicon=lexicon()
    )
    unigrams = NgramSchema(n=1).transform(seqs)
    np.testing.assert_array_equal(unigrams.values[2], unigrams.values[3])
    within = NgramSchema(normalization="count").transform(seqs)
    assert not np.array_equal(within.values[2], within.values[3])
    assert within.values[0, within.names.index("T T")] == 0
    cross = NgramSchema(boundary="cross", normalization="count").transform(seqs)
    assert cross.values[0, cross.names.index("T T")] == 1
    assert cross.values[1, cross.names.index("K K")] == 1
    assert not cross.values[2].any()
    schema = NgramSchema()
    restored = NgramSchema.from_dict(json.loads(json.dumps(schema.to_dict())))
    assert (
        restored.coordinates == schema.coordinates and len(schema.coordinates) == 39**2
    )
    frequencies = schema.transform(seqs)
    np.testing.assert_allclose(frequencies.values.sum(axis=1), 1)
    frequencies.save(tmp_path / "features")
    loaded = ItemFeatures.load(tmp_path / "features")
    np.testing.assert_array_equal(loaded.values, frequencies.values)
    assert loaded.names == frequencies.names and loaded.metadata == frequencies.metadata
    altered = schema.to_dict()
    altered["coordinates"] = tuple(reversed(altered["coordinates"]))
    with pytest.raises(ValueError, match="schema"):
        NgramSchema.from_dict(altered)


def test_stress_counts_and_ordered_features():
    seqs = extract_pronunciations({"s": "cat a"}, lexicon=lexicon())
    table = NgramSchema(n=1, stress="keep", normalization="count").transform(seqs)
    assert table.values[0, table.names.index("AE1")] == 1
    assert table.values[0, table.names.index("AE0")] == 0
    np.testing.assert_array_equal(pronunciation_counts(seqs).values, [[2, 4, 2]])
    ordered = articulatory_sequence(seqs[0])
    assert [r["label"] for r in ordered] == ["K", "AE1", "T", "AH0"]
    assert [r["word_index"] for r in ordered] == [0, 0, 0, 1]


def test_edit_analytic_cases_and_symmetric_table():
    edit = PhoneEditDistance(normalize=False)
    assert edit(["P"], ["B"]) == pytest.approx(1 / 3)
    assert edit(["P"], ["S"]) == pytest.approx(2 / 3)
    assert edit([], ["P", "B"]) == 2
    assert edit(["P", "B"], ["P"]) == 1
    assert PhoneEditDistance()([], []) == 0
    assert PhoneEditDistance()(["P", "B"], ["P"]) == 0.5
    assert PhoneEditDistance(stress_weight=4)(["AH0"], ["AH1"]) == 0.5
    for a, b in itertools.product(PHONES, repeat=2):
        assert edit([a], [b]) == edit([b], [a])
        assert 0 <= edit([a], [b]) <= 1
    assert all(edit([p], [p]) == 0 for p in PHONES)


def test_edit_matches_exhaustive_alignment_reference():
    # Enumerate all alignments of tiny strings, independently of the DP table.
    def all_costs(a, b):
        if not a:
            return [len(b)]
        if not b:
            return [len(a)]
        substitution = 0 if a[0] == b[0] else 1
        return (
            [1 + c for c in all_costs(a[1:], b)]
            + [1 + c for c in all_costs(a, b[1:])]
            + [substitution + c for c in all_costs(a[1:], b[1:])]
        )

    edit = PhoneEditDistance(weighted=False, normalize=False)
    strings = [()] + [
        p for n in (1, 2, 3) for p in itertools.product(("P", "B"), repeat=n)
    ]
    for a, b in itertools.product(strings, repeat=2):
        assert edit(a, b) == min(all_costs(a, b))


@pytest.mark.parametrize("phones", [["ZZ"], ["AH"], ["T1"], ["AH00"], []])
def test_unsupported_pronunciations_fail(phones):
    with pytest.raises(ValueError):
        PronunciationLexicon.from_mapping({"x": [phones]}, name="invalid")


def test_bad_overrides_and_unsupported_edit_fail():
    for overrides in ({"missing": {0: ["P"]}}, {"s": {2: ["P"]}}, {"s": {"0": ["P"]}}):
        with pytest.raises(ValueError):
            extract_pronunciations({"s": "cat"}, lexicon=lexicon(), overrides=overrides)
    with pytest.raises(ValueError, match="unsupported"):
        PhoneEditDistance()(["<unk>"], [])
    with pytest.raises(ValueError, match="stress"):
        PhoneEditDistance(stress_weight=1)(["AH"], ["AH1"])


def test_no_optional_runtime_dependencies():
    src = Path(__file__).resolve().parents[2] / "src"
    code = """
import sys, importlib.abc
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','tensorflow','keras','pandas','scipy','sklearn','cmudict','pronouncing','pincelate','panphon','librosa'}:
            raise AssertionError('unexpected optional dependency: ' + fullname)
sys.meta_path.insert(0, Guard())
from natural_features.features.speech.pronunciation import extract_pronunciations
from natural_features.features.speech.phone_sequences import NgramSchema
from natural_features.workflows.sentence_comparison import extract_sentence_acoustics
from natural_features.core.stimulus import AudioStimulus
import numpy as np
seqs = extract_pronunciations({'s': 'cat'}, selection='dictionary_order')
assert NgramSchema().transform(seqs).eligible[0]
assert extract_sentence_acoustics({'s': AudioStimulus(np.ones(3200), 16000)}).blocks['controls'].eligible[0]
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; sys.path.insert(0, {str(src)!r});\n" + code,
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "text", ["‘Stop, he said.", "'Stop,' he said.", "He said stop’ and left"]
)
def test_unmatched_quotation_marks_do_not_drop_words(text):
    seq = extract_pronunciations({"q": text})[0]
    stop = next(w for w in seq.words if "top" in w.text.lower())
    assert stop.normalized == "stop" and stop.origin == "dictionary"
    assert text[stop.start : stop.end] == stop.text  # span keeps the mark
    assert seq.coverage["unresolved_indices"] == []


def test_listed_apostrophe_forms_take_precedence_over_quote_stripping():
    words = extract_pronunciations({"a": "tell 'em the dogs' names"})[0].words
    assert [w.normalized for w in words] == ["tell", "'em", "the", "dogs'", "names"]
    lex = PronunciationLexicon.from_mapping({"stop": [["S", "T", "AA1", "P"]]}, name="t")
    unknown = extract_pronunciations({"b": "'Zzyzx"}, lexicon=lex)[0].words[0]
    assert unknown.normalized == "'zzyzx" and unknown.origin == "unresolved"


def test_dotted_save_prefixes_do_not_collide(tmp_path):
    seqs = extract_pronunciations({"a": "cat", "b": "dog"})
    first = NgramSchema(n=1).transform(seqs)
    second = NgramSchema(normalization="count").transform(seqs)
    first.save(tmp_path / "sub-01.run-1")
    second.save(tmp_path / "sub-01.run-2")
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "sub-01.run-1.json",
        "sub-01.run-1.npz",
        "sub-01.run-2.json",
        "sub-01.run-2.npz",
    ]
    assert ItemFeatures.load(tmp_path / "sub-01.run-1").names == first.names
    assert ItemFeatures.load(tmp_path / "sub-01.run-2").names == second.names


def test_unweighted_stress_weight_is_graded_not_boolean():
    small = PhoneEditDistance(weighted=False, stress_weight=0.01)
    assert small(["AH0"], ["AH1"]) == pytest.approx(0.01 / 1.01)
    assert small(["AH1"], ["IY1"]) == pytest.approx(1 / 1.01)
    assert small(["AH0"], ["IY1"]) == pytest.approx(1.0)
    assert small(["T"], ["D"]) == 1.0 and small(["T"], ["T"]) == 0.0
    with pytest.raises(ValueError, match="stress"):
        small(["AH"], ["AH1"])
    plain = PhoneEditDistance(weighted=False)
    assert plain(["AH0"], ["AH1"]) == 0.0 and plain(["AH"], ["AH1"]) == 0.0
