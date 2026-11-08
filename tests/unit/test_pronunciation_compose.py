from __future__ import annotations

import pytest

from natural_features.features.speech.pronunciation import (
    PronunciationLexicon,
    PronunciationSequence,
    coverage_report,
    extract_pronunciations,
    load_cmudict,
)
from natural_features.features.speech.pronunciation_compose import (
    COMPOSITION_RULES,
    compose_pronunciation,
)

TOY = {
    "hug": [["HH", "AH1", "G"]],
    "hugg": [["HH", "AH1", "G"]],
    "hop": [["HH", "AA1", "P"]],
    "hope": [["HH", "OW1", "P"]],
    "mint": [["M", "IH1", "N", "T"]],
    "rustle": [["R", "AH1", "S", "AH0", "L"]],
    "snow": [["S", "N", "OW1"]],
    "flake": [["F", "L", "EY1", "K"]],
    "mouth": [["M", "AW1", "TH"]],
    "water": [["W", "AO1", "T", "ER0"]],
    "box": [["B", "AA1", "K", "S"]],
    "city": [["S", "IH1", "T", "IY0"]],
    "wait": [["W", "EY1", "T"]],
    "walk": [["W", "AO1", "K"]],
    "rust": [["R", "AH1", "S", "T"]],
    "les": [["L", "EY1"]],
    "chip": [["CH", "IH1", "P"]],
    "lee": [["L", "IY1"]],
    "read": [["R", "IY1", "D"], ["R", "EH1", "D"]],
}


def phones(word, entries=TOY, **kw):
    alts, labels = compose_pronunciation(word, entries, **kw)
    return [" ".join(a) for a in alts], list(labels)


@pytest.mark.parametrize(
    "word, expected, label",
    [
        ("huggable", "HH AH1 G AH0 B AH0 L", "composed:hug+able"),
        ("minty", "M IH1 N T IY0", "composed:mint+y"),
        ("rustles", "R AH1 S AH0 L Z", "composed:rustle+s"),
        ("boxes", "B AA1 K S IH0 Z", "composed:box+es"),
        ("cities", "S IH1 T IY0 Z", "composed:city+s"),
        ("walks", "W AO1 K S", "composed:walk+s"),
        ("waited", "W EY1 T IH0 D", "composed:wait+ed"),
        ("walked", "W AO1 K T", "composed:walk+ed"),
        ("hoping", "HH OW1 P IH0 NG", "composed:hope+ing"),
        ("hopping", "HH AA1 P IH0 NG", "composed:hop+ing"),
        ("mouthwatering", "M AW1 TH W AO2 T ER0 IH0 NG", "composed:mouth+water+ing"),
        ("snow-flake", "S N OW1 F L EY1 K", "composed:snow-flake"),
    ],
)
def test_rules_produce_expected_estimates(word, expected, label):
    assert phones(word) == ([expected], [label])


def test_possessive_and_plural_allomorphs_follow_final_phone():
    entries = {**TOY, "snowflake": [["S", "N", "OW1", "F", "L", "EY2", "K"]]}
    assert phones("snowflake's", entries)[0] == ["S N OW1 F L EY2 K S"]
    # A plural possessive sounds like the plural, which may itself be composed.
    assert phones("hugs'", entries) == (["HH AH1 G Z"], ["composed:hug+s+'"])
    boxes = {**entries, "boxes": [["B", "AA1", "K", "S", "IH0", "Z"]]}
    assert phones("boxes'", boxes) == (["B AA1 K S IH0 Z"], ["composed:boxes+'"])


def test_compound_demotes_second_primary_stress():
    assert phones("snowflake") == (["S N OW1 F L EY2 K"], ["composed:snow+flake"])


def test_suffix_analyses_block_spurious_compounds():
    # rust + les would be a dictionary split; the plural rule wins.
    assert phones("rustles")[1] == ["composed:rustle+s"]


def test_disagreeing_analyses_are_all_kept_and_variants_propagate():
    out, labels = phones("reading")
    assert out == ["R IY1 D IH0 NG", "R EH1 D IH0 NG"]
    assert labels == ["composed:read+ing", "composed:read+ing"]


def test_unexplained_words_stay_unresolved_and_own_entry_is_ignored():
    assert phones("cerulean") == ([], [])
    assert phones("hug") == ([], [])  # no rule rebuilds a bare dictionary word
    with pytest.raises(ValueError, match="max_depth"):
        compose_pronunciation("hugs", TOY, max_depth=0)
    assert phones("mouthwatering", max_depth=1) == ([], [])


def test_extraction_is_opt_in_and_marks_estimates():
    text = "The huggable puppy has a cerulean collar"
    default = extract_pronunciations({"s": text})[0]
    hug = next(w for w in default.words if w.normalized == "huggable")
    assert hug.origin == "unresolved" and not default.eligible
    assert default.metadata["params"]["composition"] is None

    seq = extract_pronunciations(
        {"s": text}, compose=True, selection="dictionary_order"
    )[0]
    hug = next(w for w in seq.words if w.normalized == "huggable")
    assert hug.origin == "composed_estimated" and hug.reason is None
    assert hug.phones == ("HH", "AH1", "G", "AH0", "B", "AH0", "L")
    assert hug.alternative_labels == ("composed:hug+able",)
    assert seq.coverage["estimated_indices"] == [hug.index]
    assert not seq.eligible  # cerulean is still unresolved
    assert seq.metadata["params"]["composition"] == COMPOSITION_RULES

    supp = PronunciationLexicon.from_mapping(
        {"cerulean": [["S", "ER0", "UW1", "L", "IY0", "AH0", "N"]]}, name="study"
    )
    full = extract_pronunciations(
        {"s": text}, compose=True, supplemental=supp, selection="dictionary_order"
    )[0]
    assert full.eligible
    assert PronunciationSequence.from_dict(full.to_dict()) == full


def test_supplemental_entries_and_overrides_take_precedence():
    text = "huggable"
    supp = PronunciationLexicon.from_mapping(
        {"huggable": [["HH", "AH1", "G", "IH0", "B", "AH0", "L"]]}, name="study"
    )
    w = extract_pronunciations({"s": text}, compose=True, supplemental=supp)[0].words[0]
    assert w.origin == "supplemental_unverified" and w.phones[3] == "IH0"
    over = extract_pronunciations(
        {"s": text}, compose=True, overrides={"s": {0: ["HH", "AH1", "G", "Z"]}}
    )[0].words[0]
    assert over.origin == "item_override_unverified"
    assert over.phones == ("HH", "AH1", "G", "Z")
    assert over.alternative_labels[0] == "composed:hug+able"


def test_supplemental_words_can_serve_as_composition_bases():
    supp = PronunciationLexicon.from_mapping(
        {"zorp": [["Z", "AO1", "R", "P"]]}, name="study"
    )
    w = extract_pronunciations({"s": "zorpy"}, compose=True, supplemental=supp)[0]
    assert w.words[0].phones == ("Z", "AO1", "R", "P", "IY0")


def test_ambiguous_estimates_follow_selection_rule():
    seq = extract_pronunciations({"s": "rereading"}, compose=True)[0]
    word = seq.words[0]
    if len(word.alternatives) > 1:
        assert word.selected is None and word.reason == "alternatives_unselected"
        chosen = extract_pronunciations(
            {"s": "rereading"}, compose=True, selection="dictionary_order"
        )[0].words[0]
        assert chosen.selected == 0


def test_coverage_report_lists_estimates_separately():
    seqs = extract_pronunciations(
        {"a": "a huggable snowflake", "b": "a minty cerulean sky", "c": "a cat"},
        compose=True,
        selection="dictionary_order",
    )
    report = coverage_report(seqs, pairs=[("a", "b"), ("a", "c"), ("b", "c")])
    assert [r["token"] for r in report["unresolved"]] == ["cerulean"]
    estimated = {r["token"]: r for r in report["estimated"]}
    assert set(estimated) == {"huggable", "minty"}
    assert estimated["huggable"]["derivations"] == ["composed:hug+able"]
    assert estimated["minty"]["affected_pairs"] == 2
    assert report["eligible_items"] == 2 and report["estimated_items"] == 1


def test_bundled_dictionary_reproduces_held_out_regular_forms():
    entries = load_cmudict().entries
    for word in ("lovable", "stopped", "cities", "boxes", "walking", "sunflower"):
        alts, _ = compose_pronunciation(word, entries)
        assert alts and alts[0] in entries[word], word


def test_example_manifest_compose_flag_must_be_boolean(tmp_path, monkeypatch):
    import importlib.util
    import sys
    import json
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "phonology_comparison_compose_check",
        Path(__file__).resolve().parents[2] / "examples/phonology_comparison.py",
    )
    example = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, example)
    spec.loader.exec_module(example)
    items = [{"item_id": f"s{i}", "text": "cat", "audio": "x.wav"} for i in range(2)]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"items": items, "compose": "yes"}))
    with pytest.raises(ValueError, match="compose"):
        example.run_comparison(manifest, tmp_path / "out", selection="retain")
