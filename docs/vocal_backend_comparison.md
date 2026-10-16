# Vocal pitch and activity comparison

`scripts/compare_vocal_backends.py` compares the merged PR #23 implementation
with two experimental adapters. The original extractor and acceptance gates
are unchanged. This is a development experiment on 90 previously consumed
Vocadito/control inputs, not independent qualification.

The fixed arms are canonical Demucs–CREPE, RMVPE on the mixture, and RMVPE on
the same canonical Demucs vocal stem. Each is evaluated without an activity
mask and with CPJKU activity from the mixture or stem: nine pitch arms total.
All arms use identical rendered audio, annotations, strict scoring and gates.
The canonical output is checked against the preserved native evidence before
new results are accepted. No thresholds are selected from these outcomes.

The RMVPE adapter uses the pinned RVC implementation and compatible RVC
checkpoint, not the differently configured Dream-High implementation. Its
maximum salience is pitch confidence, not evidence that a human voice is
present. It retains the upstream strict salience threshold `> 0.03`, then
rejects candidates outside 50–1000 Hz. CREPE instead searches within its
configured range, so this difference remains part of the backend comparison.

The CPJKU/veracity detector supplies genuinely different singing-activity
evidence from a music-trained convolutional model. Its fixed upstream
diagnostic threshold is `> 0.51`. This score is not calibrated for this corpus.
Native centers, padding, real input bounds and computational context are
retained. Model files and source must match the hashes in the adapters; they
are not vendored or downloaded implicitly. The small compatibility changes
remove obsolete optional runtime hooks, not neural operations.

Vocal activity, pitched voicing and phonetic support remain separate. The
explicit `pitch_with_activity` operation intersects pitch acceptance with
activity; it cannot establish phonetic absence. On Vocadito, positive F0
provides only a subset of vocal-presence positives. F0-unvoiced frames are
unknown for activity, because they may contain unvoiced consonants. Backing
and silence supply known negative controls. Activity reports include class
coverage; unavailable negative predictions are not evidence of absence.

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python \
  scripts/compare_vocal_backends.py \
  --assets /path/to/preserved-vocal-assets \
  --rmvpe-assets /path/to/RVC \
  --activity-assets /path/to/veracity \
  --separator-checkpoint /path/to/955717e8-8726e21a.th \
  --artifact-dir /tmp/vocal-comparison-native \
  --out-json /tmp/vocal-comparison.json
```

The protocol records source/model revisions and download URLs. Native outputs
are retained with separate cache identities for every arm. Exit 0 means the
comparison finished; scientific gate failures remain explicit in the report
and never become qualification or automatic model adoption. Six original
analytic pitch controls remain secondary engine checks, not singing examples.

Independent synthetic review established native-rate parity against the
pinned upstream computations under the installed libraries. It does not
establish empirical accuracy, calibrated confidence, or exact reproduction
of the upstream projects' historical software environments.

Fresh singer, song and accompaniment partitions, plus a pretraining-overlap
audit, are still required for qualification. All 40 repeatedly examined
Vocadito clips are development evidence. The downstream report based on
PR #24 cannot qualify the canonical PR #23 pipeline.

Sources: [RMVPE paper](https://arxiv.org/abs/2306.15412),
[RVC source](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/tree/9f2f0559e6932c10c48642d404e7d2e771d9db43),
[CPJKU/veracity](https://github.com/CPJKU/veracity/tree/0983900f136173015f3c5d0b116be014edd33905).
