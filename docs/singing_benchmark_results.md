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

## Vocal pitch and activity: completed 90-case comparison

The [fixed comparison](vocal_backend_comparison.md) completed all 90 consumed
Vocadito/control cases with zero final runtime errors. The canonical output
passed its historical replay check on every case. **None of the nine arms
passed all original development gates.** No candidate was adopted.

The [complete numerical report](../tests/benchmarks/reports/vocal_backend_comparison_v1.json)
retains every case, artifact hash, activity score, group summary and failed
gate. The table shows extrema across the original three groups, each scored
as an equal-excerpt mean. Singing columns cover solo, 0 dB and −6 dB mixtures;
backing is a separate negative control. These extrema can come from different
groups/conditions. RPA is raw pitch accuracy and FA is voicing false-alarm
rate. Required bounds remain RPA/recall ≥80%, singing FA ≤10% and backing
FA ≤5%; the other original gates remain in the report.

| Pitch backend | Activity mask input | Min RPA | Min recall | Max singing FA | Max backing FA |
|---|---|---:|---:|---:|---:|
| Canonical Demucs–CREPE | none | 79.5% | 82.7% | 14.4% | 0.0% |
| Canonical Demucs–CREPE | mixture | 71.9% | 73.7% | 12.1% | 0.0% |
| Canonical Demucs–CREPE | stem | 79.4% | 82.6% | 14.2% | 0.0% |
| RMVPE mixture | none | 83.8% | 98.1% | 70.5% | 48.1% |
| RMVPE mixture | mixture | 73.8% | 86.6% | 48.0% | 1.8% |
| RMVPE mixture | stem | 83.7% | 97.9% | 59.4% | 23.7% |
| RMVPE vocal stem | none | 94.4% | 99.5% | 50.6% | 2.3% |
| RMVPE vocal stem | mixture | 83.8% | 87.5% | 38.4% | 0.0% |
| RMVPE vocal stem | stem | 94.4% | 99.2% | 46.1% | 0.6% |

RMVPE on canonical vocal stems gives substantially higher raw pitch accuracy
in this pilot, but its fixed upstream threshold accepts too many
reference-unvoiced frames. Adding mixture activity reduces these false
voicing decisions, while leaving them far above the original limit. For the
canonical backend, mixture activity reduces the worst FA from 14.4% to 12.1%
but also lowers the worst recall from 82.7% to 73.7%. Stem activity changes
little. Better pitch recovery therefore does not establish acceptable
pitched-voicing decisions.

The activity detector itself behaves differently on the two inputs. Its
backing-only false-activity rate is 0–3.1% on original mixtures versus
23.8–50.9% on separated vocal stems (group means). Its lowest group/condition
recall on known pitched-singing positives is 87.5% for mixtures and 99.2%
for stems. This supports retaining the original mixture as distinct evidence;
it does not establish full vocal-activity accuracy. Vocadito F0-unvoiced
frames remain unknown for presence because they can contain consonants.
Vocal presence is broader than pitched voicing.

Each pitch case has at most one unmatched reference timestamp. Activity
coverage is at least 99.87% per case on the known subset, and 100% on negative
controls; abstentions remain explicit. Every arm has zero silence FA.
RMVPE also fails the inherited analytic completeness check because its
100 Hz tone has no accepted interior F0. These synthetic tones are pitch-engine
controls, not labeled singing, and the empirical failures stand separately.

The [independent synthetic parity receipt](../tests/benchmarks/reports/vocal_backend_parity_v1.json)
shows bitwise agreement with pinned upstream RMVPE and CPJKU computations.
The first canonical replay failed a numerical preflight because its CPU
thread count differed from the historical run. Restoring four threads
resolved the check without relaxing its tolerance or any scientific gate;
the [failed receipt](../tests/benchmarks/reports/vocal_backend_comparison_preflight_v1.json)
is retained. A later [disk-full interruption](../tests/benchmarks/reports/vocal_backend_comparison_storage_failure_v1.json)
after 74 saved cases was resumed from the same verified cache. Neither
recovery changed models, references, thresholds or scoring.

## Independent reproduction

[`scripts/review_singing_empirical.py`](../scripts/review_singing_empirical.py)
recomputes pitch scores, activity confusion counts and masks, group gates,
CTC decoding/edit counts and posterior-profile integrals without importing
the production scorers or running models. It checks hashes of the retained
native artifacts, references, model snapshots and frozen source files.
Analytic pitch aggregates are checked from retained case summaries; this
auditor does not rerun those analytic models.

The [completed audit receipt](../tests/benchmarks/reports/singing_empirical_review_v1.json)
records 990 vocal artifact hashes, 810 pitch scores, 180 activity scores,
540 activity masks and 459 aggregate scalar gates, plus all 32 phone items
and 384 profile windows. Every check passed. The
[validation receipt](../tests/benchmarks/reports/singing_benchmark_validation_v1.json)
preserves runtime versions, process exits and the separate local test result:
647 passed, three skipped and one existing installed-R contract mismatch.
Static release checks passed for all 49 public feature contracts.

After running the two recipes and retaining their raw JSON reports:

```bash
python scripts/review_singing_empirical.py \
  --root "$PWD" \
  --vocal-report /tmp/vocal-comparison.json \
  --vocal-assets /path/to/preserved-vocal-assets \
  --phone-report /tmp/singing-pilot-report.json \
  --phone-manifest /tmp/singing-pilot-inputs/manifest.json \
  --complete
```

Use normal Python, not `python -O`, because assertions are the audit checks.
The numerical phone report committed here omits raw reference/decoded phone
sequences; the auditor needs the original generated report and its artifacts.
The independently authored oracle was formatted for inclusion in the
repository with an exactly equal Python syntax tree.

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
