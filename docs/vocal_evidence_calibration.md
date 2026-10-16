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
