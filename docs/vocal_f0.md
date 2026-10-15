# Vocal F0 in mixed music

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
| `source_supported` | Energy share at least 0.1 and vocal RMS at least 0.001 |
| `voiced`, `missing` | Acceptance additionally requires periodicity at least 0.21; missing is its complement |
| `range_edge` | Estimate within 20 cents of a search boundary; not proof the true pitch is out of range |
| `octave_jump` | Adjacent accepted estimates differ by 1200 ± 100 cents; genuine intervals can trigger it |
| `padded` | CREPE window extends outside supplied audio |

These thresholds are fixed engineering heuristics. Source support is not a
singing detector probability, and periodic instruments can leak into the stem.
The default search range is 50–1000 Hz. F0 outside that range is not recovered
by adding an octave correction. Changing thresholds or the range requires
qualification for that configuration.

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
Thresholds and models are not tuned on these outcomes.

Qualification applies to the named excerpts and controlled mixtures only.
It does not establish performance on commercial mixes, choirs, every singer,
calibrated source confidence, or independence from model training data.
Downstream item/scanner alignment remains study code's responsibility.

## References and attribution

- [Demucs model/source](https://github.com/facebookresearch/demucs/tree/v4.0.1), MIT.
- [CREPE implementation](https://github.com/maxrmorrison/torchcrepe), MIT.
- [Vocadito dataset and annotations](https://zenodo.org/records/5578807), CC BY 4.0;
  archive MD5 `dea40fd18f14d899643c4ba221b33a46`.
- Kevin MacLeod's [Vibe Ace attribution](https://librosa.org/data/audio/Kevin_MacLeod_-_Vibe_Ace.txt)
  and [Sugar Plum Fairy attribution](https://librosa.org/data/audio/Kevin_MacLeod_-_P_I_Tchaikovsky_Dance_of_the_Sugar_Plum_Fairy.txt),
  CC BY 3.0. Reference audio is fetched explicitly and not redistributed here.
