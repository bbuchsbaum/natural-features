# Singing-feature development evidence

These experiments extend the merged canonical PR #23 implementation. They do
not transfer qualification from the separate PR #24 primitives or the
downstream Spt report. All recordings here were already examined; no result
constitutes fresh qualification or automatic model adoption.

## Singing phones: completed 32-case pilot

The [frozen recipe](singing_phones.md) uses four NUS-48E singer/song pairs,
their first 16 seconds, two pinned CTC checkpoints and four conditions. The
[numerical report](../tests/benchmarks/reports/singing_phone_pilot_v1.json)
retains native-artifact identities, token and mapping coverage, missingness,
recognition counts, posterior profiles and temporal diagnostics. Original
audio/annotations and complete posterior artifacts remain outside Git.

| Condition | Base PER | Large PER | Base composition TV | Large composition TV |
|---|---:|---:|---:|---:|
| Read speech | 21.3% | 20.0% | 0.352 | 0.333 |
| Clean singing | 37.6% | 32.4% | 0.547 | 0.593 |
| Singing + 0 dB backing | 64.2% | 48.0% | 0.614 | 0.641 |
| Vocal stem from mixture | 37.6% | 33.5% | 0.556 | 0.601 |

PER is total edits divided by total reference phones: 450 for read speech,
173 for each singing condition. Equal-singer macro averages and item-level
counts are also in the report. TV is total variation between normalized raw
CTC posterior mass and duration-weighted reference profiles in four-second
windows; lower values indicate closer agreement with that particular
reference. It averages 16 comparable read windows and 14 of 16 singing
windows. Two singing windows have no reference-phone duration and remain
missing for comparison. This is not a model support threshold.

Separation restored recognition error close to clean singing in this pilot.
The larger model had lower observed canonical-inventory PER, but worse
duration-profile agreement in all singing conditions. The comparison does
not isolate acoustic recognition quality: conservative mappings cover
different token inventories, and the large model's mapped fraction of phone
mass ranges from about 82% in mixtures to 90% in clean singing. Read and sung
crops also contain different amounts of text, and read boundaries have
lower-grade automatic alignment.

The base model's share of posterior mass assigned to phones increased from
9.4% in clean singing to 25.1% in mixtures even while PER worsened. More phone posterior
mass therefore cannot serve as recognition ground truth. CTC evidence is
also not a phone-duration estimate. These results support evaluating the
desired representation directly, retaining coverage and missingness; they
do not establish accurate phoneme recognition or a generally superior model.

Independent rescoring checked all 32 native artifacts and 384 composition/
ordered-bin windows, including raw CTC decoding and edit paths. Separate
synthetic checks covered 961 short sequence pairs, irregular integration
cells, reference-only phones and missing native intervals. No phonetic
acceptance decision used F0.

The initial run could not load the large checkpoint because optional
tokenizer dependencies were absent. Its [failure receipt](../tests/benchmarks/reports/singing_phone_pilot_preflight_v1.json)
is preserved. Installing the missing dependencies allowed an identical
32-case rerun; model files, references, mapping and scoring were unchanged.

## Vocal pitch and activity

The [fixed comparison](vocal_backend_comparison.md) evaluates nine arms on
90 consumed Vocadito/control cases with the original strict scores and gates.
Complete empirical results and independent aggregate-gate rescoring are
pending. No candidate has been adopted.

The [independent synthetic parity receipt](../tests/benchmarks/reports/vocal_backend_parity_v1.json)
shows bitwise agreement with pinned upstream RMVPE and CPJKU computations.
The first canonical replay failed a numerical preflight because its CPU
thread count differed from the historical run. Restoring four threads
resolved the check without relaxing its tolerance or any scientific gate;
the [failed receipt](../tests/benchmarks/reports/vocal_backend_comparison_preflight_v1.json)
is retained.

## Qualification boundary

Neither experiment establishes general singing-domain qualification. Future
acceptance requires an audit of consumed recordings and model pretraining,
then untouched singer/song/accompaniment partitions and frozen acceptance
criteria. Existing strict pitch gates remain unchanged. The phone recipe
does not invent a universal support threshold.

Study-specific 5.49-second windows, scanner alignment, matched items/pairs,
nuisance controls and native rMVPA comparisons remain downstream. Preserved
phone-related geometry can motivate an RSA predictor; it does not establish
accurate recognition. Phonetic support remains independent of F0 acceptance.
