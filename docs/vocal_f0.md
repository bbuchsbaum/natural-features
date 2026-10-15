# Vocal F0 in mixed music

**Experimental and unqualified:** the frozen v2 profile failed its singer-disjoint test on false voicing. Pitch accuracy alone does not establish supported vocal F0. Do not treat this candidate as a qualified vocal-pitch bank.

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
`ceil(voicing_guard_s / effective_hop_s)` rows on each side to pass the same
framewise gate. Missing neighbors reject. For example, 0.02 seconds at the
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
The prospective singer set remains unused.

The reports are `tests/benchmarks/reports/vocal_f0_development_v4.json` and
`vocal_f0_calibration_v5.json`; the corresponding protocols retain the
development groups, all original gates, selection rules and exact input hashes.
The independent v4 review verified all 60 artifacts and all 240 profile-row
scores, including waveform calculations from the retained stems.

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
