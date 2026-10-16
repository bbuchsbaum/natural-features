# Joint pitched-voicing development

This bounded experiment compares threshold selection on RMVPE salience with
one fixed ridge-logistic model that combines RMVPE, original-mixture vocal
activity, and canonical source diagnostics. It uses all 90 previously consumed
cases from the canonical backend comparison. It does not change extractor
defaults or qualify a model on new recordings.

The [protocol](../tests/benchmarks/thresholds/vocal_evidence_calibration_v2.json)
pins sources, native-input report, unchanged acceptance limits, and the
[fresh-reference reservation](singing_reference_reservation.md). Both arms use
the same RMVPE candidates and reference grid. Candidate admissibility uses
availability, actual input bounds and 50–1000Hz range; the old 0.03 threshold is
not an irreversible filter. Original confidence and candidate pitch remain in
the output alongside a separately named pitched-voicing score.

The joint model has eight predetermined features: RMVPE salience, mixture
activity and CREPE confidence logits; log source RMS and energy share; pitch
disagreement in octaves; and two missing-evidence indicators. Missing activity
is not negative evidence. Training predicts pitched voicing, not vocal presence.
An unvoiced consonant remains a vocal event and phone support never depends on
this model's acceptance.

Five singer folds provide outer development predictions. Each outer training
partition uses three singer folds for threshold selection, then refits only on
its allowed training observations. Every fit excludes ancestor singers, exact
rendered audio hashes and held negative families. Weights and standardization
are recomputed after exclusion. Regularization is fixed at 0.01; no additional
model, feature or temporal-decoder search is performed. The threshold baseline
uses the same selector and admissibility without fitting a classifier.

Threshold selection exhausts attainable decision states and retains the
original strict gates. An infeasible selector returns a diagnostic 0.5
threshold with an explicit failure, not an accepted compromise. Every outer
selector, complete outer scores, final selector and inherited analytic controls
must pass for overall advancement. The retained RMVPE 100Hz analytic control
has no accepted interior estimates, so its incomplete evidence remains an
overall failure regardless of empirical improvement.

The saved fit trace identifies all training and forbidden keys, fitted scaling
and coefficients. Threshold-state arrays and all 180 outer output arrays are
retained outside Git. Public and optimized scorers must agree exactly on every
case; the input replay must also reproduce the previous fixed RMVPE scores.
Brier diagnostics include misses and state their benchmark weighting. They do
not establish calibration for a population of music recordings.

After freezing the protocol and code, run:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/calibrate_vocal_evidence.py \
  --assets /private/tmp/natural-features-vocal-assets \
  --artifact-dir /private/tmp/nf-joint-vocal-evidence-v2 \
  --out-json /private/tmp/nf-joint-vocal-evidence-v2.json
```

The asset paths describe this retained development run. Downloaded models and
native caches must exist and match their recorded hashes. Exit status zero
means execution completed; qualification is a separate, explicit report field.
Singer cross-validation on reused data is still development evidence. Global
song IDs are absent from the retained Vocadito metadata, and two shared backing
recordings cannot establish accompaniment generalization.

## Frozen run result

The experiment was committed at `00b37dc` before fitting. The
[complete report](../tests/benchmarks/reports/vocal_evidence_calibration_v2.json)
contains both predetermined arms, 24 fitted models, 180 outer case scores and
all threshold selections. Execution finished successfully, with exact public
scorer parity on every output. No fresh reserved reference was consumed.

The values below are the minimum accuracy/recall and maximum false-voicing
across the nine historical-group × singing-condition summaries, using their
original aggregation. They are not per-frame pooled estimates or independent
test-set performance.

| Arm | Minimum raw pitch accuracy | Minimum voiced recall | Worst false voicing | Backing / silence false voicing |
|---|---:|---:|---:|---:|
| RMVPE salience + trained-partition threshold | 87.51% | 88.60% | 10.14% | 0% / 0% |
| Fixed joint logistic evidence | 85.55% | 86.83% | 10.45% | 0% / 0% |

Every inner selector found an eligible threshold, and the final selectors were
eligible. Nevertheless, outer predictions in the consumed-v2 solo condition
exceeded the 10% false-voicing limit for both arms. Neither arm passes the
empirical procedure, and the inherited incomplete analytic control separately
prevents overall acceptance. No gate was rounded, relaxed or replaced.

Threshold calibration materially improves the fixed 0.03 RMVPE result. This
single joint model does not establish a benefit from added activity/source
evidence: it has lower minimum pitch accuracy and slightly worse worst-case
false voicing than the threshold-only baseline. Preserve this negative result;
do not expand the search on these repeatedly examined recordings. Defaults
remain unchanged and #21 remains scientifically unqualified.

An independent scalar audit reconstructed the input references, exclusions,
saved-model predictions, selector eligibility and final aggregation. It checked
90 input projections, 24 fits, 12 selectors and 180 outputs, confirming both
failures. The [receipt](../tests/benchmarks/reports/vocal_evidence_calibration_v2_review.json)
binds this finding to the frozen commit and report SHA-256. Replay it without
fitting or threshold search:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python scripts/review_vocal_evidence_calibration.py \
  --assets /private/tmp/natural-features-vocal-assets \
  --out-json /private/tmp/vocal-evidence-review.json
```
