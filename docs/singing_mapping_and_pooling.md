# Singing-phone mapping and representation sensitivity

This v2 follow-up is a frozen, consumed-development comparison over all 32
retained v1 posterior artifacts. It does not rerun a recognizer, choose a
model, define a support threshold, or qualify singing recognition.

`english_singing_phone_mapping_v2.json` is deliberately English-only. Every
declared mapping says whether it is an `exact`, `coarse-equivalence`, or
`sequence` relation and gives a reason. Tokens not declared in the file are
`unsupported`; they remain in inventory, raw-mass, and greedy-event audits.
The mapping does not convert the multilingual model vocabulary into an English
inventory. Primary and secondary stress are removed before lookup. Compound
expansions are explicit. The four unqualified plain multilingual vowel symbols
(`a`, `e`, `o`, `u`) are recorded as ambiguous alternatives and receive no
mapped mass. The primary inputs are the pinned checkpoint vocabularies/model
cards and the NUS-48E annotation inventory; IPA symbol semantics only supports
the stated equivalence, not a multilingual inference. `kh`, `dx`, and `q` are
explicitly marked extended diagnostic labels, rather than claimed NUS labels.
Equal division of a compound's posterior mass is a compositional convention
only, never evidence that component phones occupied equal duration. A compound
event contributes one occurrence to each explicit component.

Two additional v2 decisions are semantic, pre-outcome additions: `ɑ → aa`
coarsens the PALM length contrast, consistently with `ɑː → aa`; `əʊ → ow`
coarsens the English GOAT diphthong. Their primary evidence is the
[eSpeak phoneme table](https://espeak.sourceforge.net/phonemes.html) (PALM
`[A:]`, GOAT `[oU]`) and the pinned upstream
[English phoneme source](https://raw.githubusercontent.com/espeak-ng/espeak-ng/master/phsource/ph_english)
(`A@ ipaɑː`, `oU ipaəʊ`). No comparable rule is added for `ɒ`, `ɜ`, `ʊə`, or
the ambiguous plain vowels.

The frozen protocol fixes all 32 cached artifacts through the archived v1
receipt, their source hashes, the v1 manifest and report, and the old and new
implementation source hashes. It crosses two mappings (the exact retained v1
legacy mapping and v2 English decisions), and each mapping uses all three
representations:

- raw posterior integral, retained as a sensitivity baseline;
- framewise phone-conditional posterior pooling, with blank, boundary,
  special, mapped-phone, and unsupported-phone coverage retained separately;
- greedy CTC token-event composition, collapsing raw IDs before removing
  blank/non-phone roles and resetting collapse across native-time gaps.

For each composition window and ordered bin, each representation is compared
separately with normalized reference **duration** and reference **occurrence**
compositions. A reference occurrence belongs to the bin containing its onset;
duration uses exact interval overlap. Event counts and posterior mass are not
durations. Zero-mass profiles stay `null`, including bins with no reference
phone occurrence or duration; no gaps are filled or imputed.

No F0 decision participates. This comparison introduces neither the unrelated
96/200 support rule nor a 5.49-second study window; it reuses each frozen
manifest item's four-second composition windows and two-second ordered bins.

First validate the immutable plan without recomputing the empirical rows:

```bash
PYTHONPATH=src python scripts/compare_singing_phone_representations.py \
  --protocol tests/benchmarks/thresholds/singing_representation_v2.json
```

After the protocol is frozen, run the predetermined cached arms:

```bash
PYTHONPATH=src python scripts/compare_singing_phone_representations.py \
  --protocol tests/benchmarks/thresholds/singing_representation_v2.json \
  --rescore --out-json /tmp/singing-representation-v2.json
```

The output reports each token's native-time posterior mass, greedy collapsed
raw token-event count, relation, source, alternatives, canonical-label kind,
and mapped status for both mappings. It reports coverage and missingness beside
both reference comparison families, plus a separate greedy CTC PER/S/D/I
record for each mapping.
