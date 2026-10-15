# Vocal F0 in mixed music

**Experimental and unqualified:** the v2 profile failed its singer-disjoint test on false voicing. The v6 hybrid passed the false-voicing limits on six fresh singers but failed pitch accuracy and voiced recall in the −6 dB mixtures. The v7 learned voicing experiment and v8 exhaustive nested threshold experiment also failed development. No fresh recordings were used for v7 or v8. None is a qualified vocal-pitch bank.

`audio.vocal_f0` selects the `vocals` stem of Demucs 4.0.1 `htdemucs`, then
estimates its dominant F0 with torchcrepe 0.0.24 `full`. It is a separate optional
extractor from the monophonic `audio.pitch` control. It cannot identify an
individual singer in a duet or choir. Separation leakage and octave errors
remain measurable failure modes.

Install `pip install 'natural-features[vocal_f0]'`. Extraction is offline by
default. An explicit `local_files_only=False` permits downloading the fixed
80 MB separator checkpoint; CREPE weights ship with the pinned package.

```python
from natural_features.features.audio import audio_vocal_f0

vocal = audio_vocal_f0(stimulus, local_files_only=False)
names = list(vocal.coords["feature"])
f0_hz = vocal.values[:, names.index("f0_hz")]
valid = vocal.values[:, names.index("voiced")].astype(bool)
```

Rejected F0 is `NaN`, with `missing=1`. It is never zero-valued phonology or a
silence label. Retain native rows and validity when pooling downstream.

| Column | Meaning |
|---|---|
| `f0_hz` | Accepted dominant pitch in the selected stem, or NaN |
| `confidence` | CREPE periodicity, not calibrated vocal probability |
| `source_support` | Vocal-stem energy divided by summed separated-stem energies in the pitch window |
| `source_supported` | Energy share at least 0.03 and vocal RMS at least 0.01 |
| `voiced`, `missing` | Acceptance additionally requires periodicity at least 0.65; missing is its complement |
| `range_edge` | Estimate within 20 cents of a search boundary; not proof the true pitch is out of range |
| `octave_jump` | Adjacent accepted estimates differ by 1200 ± 100 cents; genuine intervals can trigger it |
| `padded` | CREPE window extends outside supplied audio |
| `candidate_f0_hz` | Unaccepted raw pitch candidate, including instrumental/silent frames; not vocal F0 evidence |
| `source_rms` | Exact vocal-stem frame RMS used by the acceptance gate, retained at float64 precision |
| `framewise_voiced` | Acceptance before the optional temporal guard; retained so temporal rejection can be distinguished from confidence/source rejection |
| `local_periodicity`, `local_supported` | Candidate-period waveform agreement and availability of its full observed support; unavailable agreement is NaN |
| `local_energy_before`, `local_energy_after`, `local_cross_energy` | Float64 mean-square energies and cross-product of the two mean-centered waveform arms, retained for exact replay |

Confidence comparisons use float32; source support and RMS comparisons use float64, preserved in the diagnostic output.

These defaults are the development-selected v2 profile (0.65, 0.03, 0.01). They are engineering heuristics. Source support is not a
singing detector probability, and periodic instruments can leak into the stem.
The default search range is 50–1000 Hz. F0 outside that range is not recovered
by adding an octave correction. Changing thresholds or the range requires
qualification for that configuration.

`voicing_guard_s` defaults to zero, preserving the v2 framewise decision. A
positive value requires every neighboring frame out to
`ceil(voicing_guard_s / effective_hop_s)` rows on each side to pass the combined
framewise and optional local-waveform gate. Missing neighbors reject. For example, 0.02 seconds at the
default 0.01-second hop requires five consecutive accepted rows. It can only
remove accepted pitch; it never fills gaps or changes surviving pitch values.
The optional 20 ms configuration **failed development**, as detailed below.
It is not a recommended or qualified replacement for the default.

`local_periodicity_threshold=None` disables the optional local-waveform veto.
Its diagnostic samples two waveform arms one candidate period apart, with
320 symmetric pair midpoints spanning 20 ms. Each arm is mean-centered.
The score is `2*C/max(A+B, 2*min_source_rms**2)`, where A and B are mean-square
energies and C is their mean cross-product. The existing RMS floor penalizes
faint periodic tails. Full observed sample support is required, without zero
padding. The score lies in [-1,1] and is not a vocal probability. Periodic
instrument leakage and reverberation can still pass it.

`max_internal_gap_s=0` disables gap-aware neighborhood support. A positive
value permits bounded internal runs of rejected candidates, up to
`floor(max_internal_gap_s / effective_hop_s)` rows, in a temporary support
envelope before temporal erosion. It never bridges unavailable evidence or
clip endpoints. The final mask still intersects the original candidate mask:
rejected rows remain missing, and surviving pitches are unchanged. Short true
rests may also be bridged in the envelope, so this is not proof of continuous
singing. All three optional parameters remain disabled by default.

## Assets, clocks and context

The only supported separator is
`955717e8-8726e21a.th` (SHA256
`8726e21a993978c7ba086d3872e7608d7d5bfca646ca4aca459ffda844faa8b4`).
The CREPE full-weight SHA256 is
`133225604dedd2e4005f8bbd1bd0a2ec073ba8b7a6cd31ff6d5edbbfa3539986`.
Package versions and content identities are recorded. A wrong asset fails;
there is no fallback to mixture autocorrelation or guessed source labels.

Input stereo is preserved through separation; mono is duplicated. Explicit
SciPy polyphase resampling produces 44.1 kHz separator input and 16 kHz pitch
input. Metadata records original/effective lengths and any resampling tail.
Demucs uses whole-input normalization, zero random shifts and overlapping
segments. DC is removed from each analysis stem and never added back as source
energy. The amplitude threshold assumes floating audio with full scale 1.
Pitch uses an undithered local weighted decoder over nine bins,
avoiding upstream random pitch-bin dithering and batch-dependent Viterbi paths.

Native centers are integer multiples of the sample-rounded hop (160 samples by
default). CREPE has a centered 1024-sample window and 512 samples of zero padding
on each side. `time_bounds_s` records its frontend window intersected with the
real input duration. Centers at or beyond the real input end are retained but
marked missing. They are not stretched to fit the clip. Only the output clock
receives `start_offset_s`; the input clock and temporal context are preserved.
The optional temporal guard uses neighboring decisions as well. The frontend
bounds remain the central pitch window; the full computation context below
includes those neighboring dependencies.

These frontend bounds are not the full dependency. Separation and normalization
can use future audio, so `computation_context` conservatively records the entire
supplied clip as bidirectional input for every output row. There is no implicit
long-recording chunking by this wrapper. Processing a different clip can change
the stem and F0 at the same timestamp.

## Reproducing qualification

The frozen protocol is `tests/benchmarks/thresholds/vocal_f0_v1.json`.
It selects Vocadito recordings 1–6 numerically, using up to their first ten
seconds, before model inference. The reference annotations contain musician
labeled F0 and actual timestamps. Each excerpt is tested solo and with
deterministic instrumental accompaniment at 0 dB and −6 dB vocal-to-backing RMS,
along with backing-only and digital-silence negatives.
Backing-only negatives use each instrumental recording's original decoded gain.
Negative scores cover every native center within the real input duration.
Analytic pitch tests cover known tones, a glide, and a missing-fundamental harmonic complex; these
do not provide evidence of vocal identification.

```bash
python scripts/qualify_vocal_f0.py --download \
  --assets /tmp/vocal-f0-assets \
  --separator-checkpoint /path/to/955717e8-8726e21a.th \
  --artifact-dir /tmp/vocal-f0-native \
  --out-json /tmp/vocal-f0-report.json
```

The runner retains native predictions and input hashes, supports resuming exact
inputs, and reports each excerpt and condition. Accepted pitch within 50 cents
and accepted-voicing recall must both reach 0.80; unvoiced false alarms must be
at most 0.10, joint median error at most 50 cents, and octave errors at most
0.05. These are equal-weight macro averages across six excerpts, separately
for each singing condition. Backing-only false acceptance must be at most
0.05 and digital-silence acceptance must be zero. Missing assets, incomplete
inference and empty denominators cannot produce a passing qualification.
Unmatched or rejected reference-voiced rows remain misses in the denominators.
The original v1 thresholds and models were not tuned on those outcomes.

The retained v1 report failed its gates: solo/0 dB false-voicing rates were
0.380/0.228, and −6 dB voiced recall was 0.761. Its joint pitch error was about
6 cents, illustrating why pitch error and voice/source errors are separate.
That candidate is not qualified.

Version 2 uses IDs 1–6 as explicitly declared development data after that failed
evaluation. A single frozen 60-combination grid calibrates confidence, source
share and RMS gates; neural models, decoder, range and temporal processing stay
fixed. The original thresholds must first reproduce the retained v1 results.
Only a profile satisfying every original development gate may proceed to a
test on singer-disjoint IDs 7, 10, 11, 12, 13 and 14. Those IDs were selected
from singer metadata alone; the instrumental test windows start at ten seconds.
The grid is not expanded after outcomes, and the test set is not replaced.
Calibration passed: five profiles met all development gates, and the frozen ranking selected (0.65, 0.03, 0.01). The original threshold profile reproduced the retained v1 predictions and scores. The selected profile is frozen in `tests/benchmarks/thresholds/vocal_f0_holdout_v2.json`; the one-shot holdout evaluation failed. Calibration and holdout status are recorded in their separate reports.

The retained `tests/benchmarks/reports/vocal_f0_holdout_v2.json` contains all
30 completed conditions, no inference errors, and passing analytic controls.
It fails the unchanged 0.10 false-voicing limit in every singing condition:

| Holdout condition | Raw pitch accuracy | Voiced recall | False voicing |
|---|---:|---:|---:|
| Solo | 0.9599 | 0.9649 | 0.1171 |
| 0 dB mixture | 0.9444 | 0.9523 | 0.1443 |
| −6 dB mixture | 0.8881 | 0.9023 | 0.1273 |

Backing-only and silence false acceptance were zero. All remaining pitch,
recall, median-error and octave gates passed. The profile is nevertheless
unqualified. The thresholds and test set were not changed after this result;
the original failed v1 and successful development calibration remain retained.
The calibration report's `holdout_consumed=false` records its state before the
subsequent test, rather than claiming that the holdout is still unused.

Reproduce the frozen test with the runner above plus
`--protocol tests/benchmarks/thresholds/vocal_f0_holdout_v2.json`, retaining a
separate output report and native artifact directory.

## Version 3 temporal experiment: failed development

A new experiment after the v2 failure tested a fixed 20 ms symmetric temporal
guard, with the same models, pitch candidates, 0.65/0.03/0.01 thresholds,
scoring rules and acceptance limits. The original development data and the
consumed v2 test were both explicitly designated as development data. This
does not revise or replace the failed v2 qualification.

The protocol in `tests/benchmarks/thresholds/vocal_f0_development_v3.json` was
frozen before evaluating the new rule. Only this single duration was tested.
Both groups had to pass every original gate separately before any fresh test.
The raw cached predictions and original scores reproduced exactly before
replay. The new report retains all 60 condition results, the original artifact
identities, and counts of removed true and false positives.

| Development group | Condition | Raw pitch accuracy | Voiced recall | False voicing |
|---|---|---:|---:|---:|
| Original IDs 1–6 | Solo | 0.8363 | 0.8391 | 0.0231 |
| Original IDs 1–6 | 0 dB | 0.8108 | 0.8174 | 0.0391 |
| Original IDs 1–6 | −6 dB | 0.7528 | 0.7618 | 0.0169 |
| Consumed v2 test | Solo | 0.8846 | 0.8870 | 0.0421 |
| Consumed v2 test | 0 dB | 0.8603 | 0.8627 | 0.0752 |
| Consumed v2 test | −6 dB | 0.7867 | 0.7945 | 0.0712 |

False voicing now falls below 0.10 in every condition, but both groups fail
the 0.80 pitch-accuracy and recall gates at −6 dB. Backing and silence remain
at zero false acceptance. The experiment therefore stopped at development;
the metadata-selected prospective singers (IDs 15, 18, 19, 20, 21, 22) remain
untested. No duration adjustment, threshold search, or fresh qualification
was performed after this result. Defaults retain zero temporal guard.

The result shows the cost of rejecting every boundary and short interruption.
Further work needs evidence that separates false voicing from valid singing
in difficult mixtures; unconditional temporal rejection is insufficient.

Reproduce this development-only replay using the retained v2 native artifacts
and reference assets:

```bash
python scripts/evaluate_vocal_f0_temporal.py \
  --assets /tmp/vocal-f0-assets \
  --artifact-dir /tmp/vocal-f0-v3-replay \
  --out-json /tmp/vocal-f0-v3-development.json
```

The consumed reports record the native artifact paths and hashes. Restore
those files at the recorded paths when replaying on another machine. The
script checks report, artifact and annotation identities and exits nonzero
for incomplete input or failed development gates. A development pass would
still not establish qualification.

## Local waveform development: versions 4 and 5

Version 4 froze four local-agreement thresholds (0.25, 0.50, 0.75, 0.90)
before computing their scores. It held all v2 framewise gates fixed and
disabled temporal erosion. The separated stems were regenerated only for
the two consumed groups; their original RMS and source-share diagnostics
reproduced exactly. All four profiles failed, and no new singers were tested.
At 0.75, the consumed v2 group's 0 dB false-voicing rate was 0.1158. At 0.90,
both groups' −6 dB pitch accuracy fell below 0.80 (0.7869 and 0.7961).

Version 5 was separately registered as **adaptive development after the v4
failure**. It checked the complete threshold decision space for this fixed
scalar veto, keeping the v4 failure unchanged. Inclusive `q >= threshold`
decisions change immediately above each distinct float64 q; tied values are
removed together. Accuracy/recall and false-alarm constraints are monotone,
while conditional median and octave-error gates require separate checks.

There is no feasible threshold on these consumed groups: false-voicing gates
require a threshold at least 0.8899342361260348, while accuracy/recall require
at most 0.8849322848886569. The analysis covered 24,099 decision states and
found that these monotone constraints do not overlap. This is a limitation of
this fixed veto on these recordings, not proof that other voicing methods fail.
The prospective singer set was unused during versions 4 and 5.

The reports are `tests/benchmarks/reports/vocal_f0_development_v4.json` and
`vocal_f0_calibration_v5.json`; the corresponding protocols retain the
development groups, all original gates, selection rules and exact input hashes.
The independent v4 review verified all 60 artifacts and all 240 profile-row
scores, including waveform calculations from the retained stems.

## Gap-aware development: version 6

Version 6 is explicitly adaptive development following those failures. It
froze twelve combinations of local threshold {0.25, 0.50, 0.75}, temporal
guard {0.01, 0.02} seconds and maximum internal gap {0.02, 0.04} seconds.
Both consumed groups must separately pass every original gate. The frozen
ranking favors the greatest worst-condition pitch accuracy, then the lowest
worst-condition false voicing, followed by the specified parameter tie breaks.

Exactly one combination passed: local threshold 0.50, guard 0.01 seconds,
maximum gap 0.02 seconds. Across both groups and three singing conditions,
the lowest raw pitch accuracy was 0.8010634 and the highest false-voicing
rate was 0.0988746. These narrow development margins do not establish
qualification. The report retains all twelve profiles and all 720 scores.

The frozen development protocol retains a stale inherited sentence under
`method.acceptance` saying "no temporal erosion". That field describes only
the local candidate veto; `method.temporal_envelope`, the twelve-profile grid,
and the implementation define the complete v6 operation. The original file
is preserved unchanged. The prospective protocol states the complete rule.

Independent rescoring reproduced all 720 scores and verified all 60 selected
native artifacts. A fresh real-model run on development excerpt 1 at −6 dB
matched replay exactly across all 17 columns and native times. The selected
configuration, extractor, runner, scorer and prospective protocol were frozen
in commit `061a1c49c9eb6d7545d72365ade26f9b1e5c0f6b` before the new test.

The one-shot singer-disjoint test on IDs 15, 18, 19, 20, 21 and 22 **failed**.
These IDs were selected from metadata before outcomes, with accompaniment
starting at 20 seconds. All 30 conditions completed without inference errors;
all six analytic controls passed. The unchanged 0.80 raw-pitch-accuracy and
voiced-recall limits failed in the −6 dB mixtures:

| Fresh v6 condition | Raw pitch accuracy | Voiced recall | False voicing |
|---|---:|---:|---:|
| Solo | 0.8722 | 0.8803 | 0.0348 |
| 0 dB mixture | 0.8518 | 0.8659 | 0.0556 |
| −6 dB mixture | 0.7566 | 0.7808 | 0.0416 |

The remaining gates passed, including zero false acceptance on backing-only
and digital silence. The new and v2 held-out results use different singers
and backing windows; they do not establish a paired improvement estimate.
Independent rescoring reconstructed all 30 rendered inputs byte-for-byte,
checked 29,385 native prediction rows and reproduced every empirical score
and the two failed gates. The analytic-control review checked the retained
aggregate results; it did not independently regenerate their predictions.
No parameters were changed and no replacement test was run after the result.
The public defaults remain the unqualified v2 profile. These new singers are
now consumed data and cannot qualify a later method independently.

The exact protocol and result are
`tests/benchmarks/thresholds/vocal_f0_holdout_v6.json` and
`tests/benchmarks/reports/vocal_f0_holdout_v6.json`. Reproduction uses the
same qualification command with `--protocol` set to that manifest. The
runner verifies the frozen extractor, scorer and runner hashes before reading
test assets; historical source commit `061a1c4` retains those identities.
Further work must recover supported singing in difficult mixtures while
keeping false voicing controlled. A passing development profile alone was
insufficient, despite independent replay verification.

## Learned voicing and grouped validation: version 7

Version 7 implements a joint voicing decision and a two-state temporal decoder.
It uses all 18 previously consumed clips from 17 singers as adaptive development
data. It does not replace any earlier held-out result. The complete procedure
was committed as `c404d11fcf24729d17f269e87012f8a64191ac9e` before empirical
fitting, with protocol SHA256
`26099428ce8de69c703eb383ed48577f2af478bb7c69eba875ea247bbef85fc4`.

The fixed five predictors are transformed CREPE confidence, source energy
share, source RMS, local waveform agreement, and availability of that local
evidence. A ridge logistic model replaces the independent hard cutoffs.
Training-only scaling and singer-out-of-fold Platt calibration precede the
explicit voiced/unvoiced sequence decision. Voiced rows retain the original
raw pitch candidate; unvoiced rows are NaN. The method can recover candidates
rejected by the old gates. Only exact zero source energy, invalid pitch, or a
center outside real input bounds forces rejection independently of the model.

Five outer singer folds evaluate the procedure. Within each outer training
partition, three selector folds choose among probability thresholds
{0.25, 0.50, 0.75} and switch penalties {0, 1, 2}. Each selector-training fit
has its own three-fold probability calibration. Every level excludes held
singers, identical rendered inputs, and negative accompaniment families,
including inherited exclusions. Repeated backing/silence copies cannot
increase their training mass. Controls removed by ancestor exclusions are
explicitly untested in inner selection; all 90 outer results and every
original gate remain required.

**The frozen nine-profile family failed.** No outer selector or final selector
found an eligible profile. The predeclared diagnostic fallback (threshold
0.50, switch penalty 0) produced these outer-fold results:

| Consumed group | Condition | Raw pitch accuracy | Voiced recall | False voicing |
|---|---|---:|---:|---:|
| Original development | Solo | 0.9606 | 0.9723 | 0.1186 |
| Original development | 0 dB | 0.9328 | 0.9545 | 0.1365 |
| Original development | −6 dB | 0.9067 | 0.9342 | 0.0874 |
| Consumed v2 test | Solo | 0.9759 | 0.9837 | 0.1560 |
| Consumed v2 test | 0 dB | 0.9642 | 0.9791 | 0.1920 |
| Consumed v2 test | −6 dB | 0.9415 | 0.9649 | 0.1864 |
| Consumed v6 test | Solo | 0.9608 | 0.9791 | 0.1103 |
| Consumed v6 test | 0 dB | 0.9327 | 0.9620 | 0.1464 |
| Consumed v6 test | −6 dB | 0.8864 | 0.9354 | 0.1365 |

Backing-only and silence false acceptance were zero. Pitch, recall, median
and octave gates passed; false voicing failed in eight of nine singing
comparisons. These are out-of-fold predictions from five fitted models,
not independent qualification of the final all-data fit.

In the separate final three-fold selector, threshold 0.75 with no switch
penalty had the lowest worst-condition false voicing among the nine profiles:
0.1404, still above 0.10, with minimum pitch accuracy 0.8515. Positive switch
penalties did not rescue eligibility. No additional thresholds were tried.
This finite experiment does not establish that every possible threshold, or
the broader vocal-identification problem, is infeasible.

Outer-fold probability diagnostics are retained by condition. Before physical
admissibility, Platt calibration changed Brier scores from 0.03926 to 0.03798
for solo, 0.05403 to 0.05374 at 0 dB, and 0.06084 to 0.06147 at −6 dB.
It therefore did not improve every condition. Reliability bins and log loss
are also reported; the fitted probabilities describe the stated benchmark
weighting, not population prevalence or a temporal-state posterior.
Frames are correlated, development is adaptive, and only two accompaniment
recordings are represented. Analytic evidence is inherited from the unchanged
pitch engine; those controls bypass singing identification.

The experiment retains all 90 native outer predictions, 120 fit records,
fold membership, input/source hashes, every tested profile, and the final
unqualified diagnostic model in
`tests/benchmarks/reports/vocal_voicing_development_v7.json`.
Independent review reproduced all 90 outputs (88,990 native rows), all 54
selector-profile evaluations, and checked the 96 base/24 calibration fit records,
including training-only scaling and ancestor exclusions. It confirmed the
failure. The receipt is `tests/benchmarks/reports/vocal_voicing_review_v7.json`.
It did not refit the optimizer, rerun neural inference, or rerender the
previously audited input waveforms. The standalone reviewer is retained at
`/private/tmp/natural-features-vocal-assets/v7-independent-review/review.py`
(SHA256 `7afb1b39c9b7be2920e8acad48e6753f904dd56a3d25bd53593596ac3c1afe75`).
Reproduce the development experiment with the retained artifacts at their recorded paths:

```bash
python scripts/calibrate_vocal_voicing.py \
  --assets /tmp/vocal-f0-assets \
  --artifact-dir /tmp/vocal-voicing-v7 \
  --out-json /tmp/vocal-voicing-v7-report.json
```

The command exits nonzero on failed development or incomplete input. It
exports `final_profile.json` even when no eligible operating point exists,
marked as development only. Explicit experimental application is:

```python
import json
from pathlib import Path
from natural_features.features.audio.vocal_voicing import apply_voicing_model

profile = json.loads(Path("/tmp/vocal-voicing-v7/final_profile.json").read_text())
experimental = apply_voicing_model(vocal, profile)
```

Application requires the pinned models and a verified native frontend
identity with the original 50–1000 Hz range and 10 ms hop. Native times,
bounds, raw candidates, and input context are preserved. The output adds
`voicing_probability` and `voicing_admissible`; legacy gate diagnostics
remain available but no longer define acceptance. Public extraction defaults
remain unchanged. The failed bounded experiment stops here, without using
fresh qualification recordings or relaxing any acceptance limit.

## Exhaustive threshold selection and error audit: version 8

V8 asks whether v7 failed because its three probability thresholds were too
coarse. It reuses the exact stored v7 base models, calibration fits, and nested
singer partitions. No fitting, neural inference, new features, new recordings,
or annotation changes occur. The v8 protocol and implementation were frozen
in commit `2f9514dd36252db7c87dc95b04b866f1b5d75157` before empirical search.
The original v7 source, protocols, and reports remain byte-identical.

For zero switching cost, the intended decision is a direct comparison between
the clipped probability's logit and the threshold's logit, with equality
unvoiced. Pre-outcome synthetic review found that v7's cumulative dynamic
program could lose a tiny positive emission to floating-point rounding. V8
uses a separate direct comparator. An exact comparison of probabilities,
admissibility, timestamps, and candidates against all retained v7 outer
artifacts passed. At threshold 0.50, this numerical correction changed **zero**
empirical decisions, so it does not explain the new threshold results.

The search includes every distinct effective float64 score class among all
admissible native frames, including unscored frames. Ties move together.
One canonical probability threshold represents each class, plus an initial
all-admissible-accepted state; the last class gives all-rejected. Integer
annotation counts determine every accuracy, recall, and false-alarm curve.
Conditional median and octave gates are checked for every state surviving
those constraints. Canonical thresholds cover the possible decision masks;
their endpoints are not estimates of a robust operating interval.

Each outer threshold is selected using only its own inner-fold predictions.
The ranking and fallback remain explicit. V8 additionally requires every
outer selector, as well as the final selector, to find an eligible threshold.
The exhaustive searches covered **309,082 states** across six selectors:

| Selector | States | Eligible states | Selected threshold |
|---|---:|---:|---:|
| Outer fold 0 | 42,219 | 0 | Diagnostic fallback 0.50 |
| Outer fold 1 | 45,617 | 1,809 | 0.7950694704520352 |
| Outer fold 2 | 51,257 | 0 | Diagnostic fallback 0.50 |
| Outer fold 3 | 51,219 | 749 | 0.8269110753286072 |
| Outer fold 4 | 50,641 | 0 | Diagnostic fallback 0.50 |
| Final all-data selector | 68,129 | 613 | 0.8590826434171935 |

The final selector's chosen threshold passes its development gates: minimum
raw pitch accuracy 0.8129546 and maximum false voicing 0.0999259. Its false
voicing margin is only 0.0000741 below the 0.10 limit. These are inner
cross-validation selection scores, not a passing outer evaluation or
qualification of the final refitted model.

The three failed outer selectors have disjoint monotone requirements:

| Selector | First canonical threshold satisfying false-alarm limits | Last canonical threshold retaining accuracy/recall limits |
|---|---:|---:|
| Outer fold 0 | 0.8856594 | 0.8678928 |
| Outer fold 2 | 0.8711233 | 0.8510561 |
| Outer fold 4 | 0.8825440 | 0.8551883 |

Thus no threshold works for those fixed inner scores under the original
gates. This establishes a limitation of the fixed scores and development
design, not impossibility of vocal identification. The complete outer
evaluation also fails: five singing false-alarm gates fail, with the largest
rate 0.1577200 in the consumed v2 group's 0 dB condition. Every outer
pitch-accuracy, recall, median, octave, backing, and silence gate passes.
The original development group passes all outer gates. The algorithm used
the declared fallback in three folds; those results are diagnostic.

The prespecified error audit uses all 54 outer-held singing streams and fixed
0.50/0.75 diagnostic anchors. Counts below pool correlated annotation frames
and are not the macro-averaged qualification metrics:

| Condition | V8 false-positive annotations | Within 50 ms of observed voicing | More than 200 ms from observed voicing |
|---|---:|---:|---:|
| Solo | 834 | 708 (84.9%) | 24 |
| 0 dB mixture | 1,142 | 703 (61.6%) | 236 |
| −6 dB mixture | 996 | 589 (59.1%) | 192 |

Both mixture conditions also contain many false positives absent from their
paired solo predictions: 582 at 0 dB and 563 at −6 dB. There are six contiguous
false-positive runs lasting at least 200 ms, all in mixtures, spanning
approximately 238–302 ms. Hash-bound figures compare reference F0, accepted
pitch, probabilities, and the original solo waveform; paired solo predictions
stay low during these sustained mixture errors. This supports a descriptive
distinction between boundary-sensitive errors and mixture-associated errors.
It does not prove instrumental leakage or annotation mistakes. No auditory
assessment or relabeling was performed.

Rejected true singing is also weaker: at −6 dB its median source RMS is 0.0140,
versus 0.0494 for accepted true singing. These selected-frame distributions
describe the observed tradeoff; they are not causal estimates. Unknown labels,
unmatched timestamps, and missing previous/next opposite-state annotations
are recorded explicitly. Runs split across unknown/non-error annotations and
timestamp gaps. Clip-censored examples remain identified by their context.

Independent review verified all 309,082 canonical states, all 90 outputs
(88,990 native rows), and all 120 inherited fit/calibration traces. It also
recomputed the 162 error-audit rows, 108 paired comparisons, and six long runs,
checked figure hashes, and compared all six waveform snippets with the exact
original source samples. It confirmed both the feasible final selector and
the development failure. The receipt is
`tests/benchmarks/reports/vocal_voicing_review_v8.json`; its SHA256 is
`59c8e9f4cb4193029d3220c506b36933c6e4000741684aea9b58b4b10d28fa68`.
The runnable independent reviewer and its bundled common code are retained in
`/private/tmp/natural-features-vocal-assets/v8-independent-review/`:
`review.py` has SHA256
`a3ab3cdb2ec8b922bc322db4e8c0e98b35920b83ad91b3db421a359ccf4445f8`,
and `common.py` has SHA256
`7afb1b39c9b7be2920e8acad48e6753f904dd56a3d25bd53593596ac3c1afe75`.
The reviewer imports no production modules and performs no fits, inference,
or fresh-data evaluation. It verifies the recorded fixed-score experiment;
it does not independently qualify a vocal F0 model.

Reproduction requires the previously retained native assets and the optional
plotting dependency from the `docs` extra:

```bash
python scripts/calibrate_vocal_voicing_threshold.py \
  --assets /tmp/vocal-f0-assets \
  --artifact-dir /tmp/vocal-voicing-v8 \
  --out-json /tmp/vocal-voicing-v8-report.json \
  --audit-json /tmp/vocal-voicing-v8-errors.json
```

The committed reports are `tests/benchmarks/reports/vocal_voicing_development_v8.json`
and `vocal_voicing_errors_v8.json`. Complete threshold curves and masks, all 90
native outer outputs, figures, and source-waveform snippets are retained under
`/private/tmp/natural-features-vocal-assets/native-v8-development`, with hashes
in the reports. Use `apply_threshold_model` from
`natural_features.features.audio.vocal_voicing_threshold` for explicit
experimental application of its exported `final_profile.json`; the default
extractor is unchanged. The overall development result is **failed**, so no
fresh qualification follows. Threshold-only development for this fixed score
model stops here; further work needs better evidence for singing, informed by
the boundary and mixture error patterns.

Evaluation covers the named excerpts and controlled mixtures only.
It does not establish performance on commercial mixes, choirs, every singer,
calibrated source confidence, or independence from model training data.
Downstream item/scanner alignment remains study code's responsibility.

## References and attribution

- [Demucs model/source](https://github.com/facebookresearch/demucs/tree/v4.0.1), MIT.
- [CREPE implementation](https://github.com/maxrmorrison/torchcrepe), MIT.
- [McLeod and Wyvill, A Smarter Way to Find Pitch](https://www.cs.otago.ac.nz/graphics/Geoff/tartini/papers/A_Smarter_Way_to_Find_Pitch.pdf):
  source for symmetric energy normalization. The local veto here adds a fixed
  candidate period, separate arm centering and amplitude regularization; it is
  not an implementation of their complete pitch estimator.
- [Vocadito dataset and annotations](https://zenodo.org/records/5578807), CC BY 4.0;
  archive MD5 `dea40fd18f14d899643c4ba221b33a46`.
- Kevin MacLeod's [Vibe Ace attribution](https://librosa.org/data/audio/Kevin_MacLeod_-_Vibe_Ace.txt)
  and [Sugar Plum Fairy attribution](https://librosa.org/data/audio/Kevin_MacLeod_-_P_I_Tchaikovsky_Dance_of_the_Sugar_Plum_Fairy.txt),
  CC BY 3.0. Reference audio is fetched explicitly and not redistributed here.
