from pathlib import Path

from natural_features.features.speech.phone_inventory import PhoneInventory, audit_decisions, load_phone_inventory


def _inventory():
    return PhoneInventory({"schema": "EnglishSingingPhoneInventory/v2", "mappings": {"a": {"phones": ["aa"], "relation": "coarse-equivalence", "reason": "test"}, "aɪə": {"phones": ["ay", "ah"], "relation": "sequence", "reason": "test"}}, "ambiguous": {"e": {"relation": "ambiguous", "alternatives": [["eh"], ["ey"]], "reason": "test", "source": "primary"}}})


def test_stress_normalization_and_compound_contract_are_explicit():
    decision = _inventory().decide("ˈaɪə")
    assert decision.normalized_token == "aɪə"
    assert decision.phones == ("ay", "ah")
    assert decision.relation == "sequence"


def test_unmapped_token_remains_visible_in_column_audit():
    result = audit_decisions(["<pad>", "a", "zz"], {"<pad>": "blank", "a": "phone", "zz": "phone"}, _inventory())
    assert result["roles"] == {"blank": 1, "phone": 2}
    assert result["phone_decisions"]["coarse-equivalence"] == 1
    assert result["unresolved_phone_columns"] == 1


def test_ambiguous_tokens_carry_alternatives_but_no_mapped_phone_mass():
    decision = _inventory().decide("e")
    assert decision.relation == "ambiguous"
    assert not decision.mapped
    assert decision.alternatives == (("eh",), ("ey",))


def test_v2_adds_only_the_two_source_backed_english_variants_to_legacy_scope():
    inventory = load_phone_inventory(Path(__file__).parents[2] / "src/natural_features/data/english_singing_phone_mapping_v2.json")
    assert inventory.decide("ɑ").phones == ("aa",)
    assert inventory.decide("əʊ").phones == ("ow",)
    assert inventory.decide("ɑ").relation == "coarse-equivalence"
    assert inventory.decide("əʊ").relation == "coarse-equivalence"
    assert inventory.decide("ɒ").relation == "unsupported"
