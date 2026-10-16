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

## Frozen comparison results

Code and protocol were committed at `b20a5d3` before rescoring. The
[complete report](../tests/benchmarks/reports/singing_representation_v2.json)
retains all 32 inputs, both inventories and all three poolings, including every
missing comparison. The v1 branch reproduced all archived recognition and raw
composition results. No model inference or fresh evaluation occurred.

The two new mappings reduce the large model's unresolved inventory columns
from 330 to 328, including four explicitly ambiguous columns. Inventory size
alone understates their effect: observed posterior mass depends on which
tokens a recording activates. Coverage below is total mapped phone-posterior
mass divided by total phone-posterior mass across the four retained recordings.
PER pools edit counts over 173 reference phones per singing condition, rather
than averaging recording-level rates.

| Large model condition | Mapping coverage v1 → v2 | PER v1 → v2 |
|---|---:|---:|
| Clean singing | 89.74% → 92.52% | 32.37% → 31.79% |
| 0dB mixture | 81.72% → 85.35% | 47.98% → 47.40% |
| Separated vocals | 88.50% → 91.65% | 33.53% → 32.95% |

Each singing condition has one fewer edit under the new mapping. These are
renderings of the same four source performances, not three independent
improvements. The base model's observed mapping and recognition results are
unchanged. The larger model's read-speech PER remains 20%. These small changes
do not establish accurate singing recognition.

Framewise phone-conditional renormalization worsens reference agreement in
every singing model/condition group in this pilot. Raw CTC mass and token
events remain different representations; neither measures duration. The next
table compares v2 duration composition on exactly the same eligible windows
within each row. Values are mean total variation (0 is exact agreement; 1 is
disjoint composition). Eligibility requires all three predicted profiles and
both reference targets to be present. It is descriptive coverage, not a library
support rule.

| Model / singing condition | Common windows / 16 | Raw integral | Framewise conditional | Token events |
|---|---:|---:|---:|---:|
| Base / clean | 13 | 0.5563 | 0.5794 | 0.5431 |
| Base / mixture | 8 | 0.5476 | 0.5940 | 0.5151 |
| Base / separated | 14 | 0.5556 | 0.6114 | 0.5344 |
| Large / clean | 14 | 0.5917 | 0.6551 | 0.5905 |
| Large / mixture | 14 | 0.6405 | 0.7224 | 0.6353 |
| Large / separated | 13 | 0.5717 | 0.6483 | 0.5542 |

For example, base-model mixture event profiles yield duration TV 0.5151 on
8/16 windows. Raw integration yields 0.6142 on its 14 reference-valid windows,
but 0.5476 on the same eight event-valid windows. Reporting only 0.6142 versus
0.5151 would mix representation quality with retained coverage. All missing
windows remain visible in the report.

The two-second ordered-bin comparison also keeps absolute temporal positions.
On common eligible bins, token events have lower occurrence-composition error
than raw integration in all six singing model/condition groups. Coverage ranges
from 13/32 bins (base mixtures) to 25/32 (large clean/mixed). This supports
further investigation of occurrence summaries; it does not validate phone
durations, exact phone timing or a downstream RSA predictor. Large remaining
duration-profile errors motivate a separately qualified framewise or alignment
model if duration estimates are required.

An [independent audit](../tests/benchmarks/reports/singing_representation_v2_review.json)
reconstructed 2,304 predicted profiles, 64 recognition records, 768 coverage
records and all 4,608 comparison slots using separate matrix/scalar calculations
and edit-distance code. It preserved 732 null comparison slots; maximum numeric
difference was `3.91e-14`. This verifies calculations, not scientific acceptance.
Replay it from the retained artifacts:

```bash
OPENBLAS_NUM_THREADS=1 python scripts/review_singing_representations.py \
  --out-json /private/tmp/singing-representations-review.json
```

The [reference reservation](singing_reference_reservation.md) remains unopened
for evaluation. Phones are still independent of F0, and no study-specific
window size or support threshold has become a library default.
