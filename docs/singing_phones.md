# Singing-phone posterior benchmark

`scripts/qualify_singing_phones.py` evaluates raw CTC posterior artifacts or a
local pinned model snapshot. It is descriptive infrastructure, not a claim
that a model recognizes singing generally. Unit tests use synthetic posterior
matrices and do not qualify a model or data set.

The bundled `singing_phones_v1.json` is a schema template and has no items, so
the runner intentionally rejects it until a pinned development manifest is
made concrete.

Each v1 manifest declares `state: "development"`, exact token
roles for every CTC column (`blank`, `boundary`, `special`, or `phone`), and a
canonical inventory. A phone token may expand to an ordered canonical sequence.
Unmapped phone tokens remain visible in mass diagnostics and decode as an
explicit unknown phone, so they worsen PER instead of disappearing.

For low-cost replay, place an `.npz` in a local pinned directory with
`times_s`, `values`, `labels`, and optional `bounds_s`, then use an item such
as:

```json
{
  "id": "singer01_song01_clean",
  "condition": "clean",
  "singer": "singer01",
  "song": "song01",
  "partition": "development",
  "posterior_npz": "artifacts/singer01.npz",
  "posterior_sha256": "...",
  "reference_intervals": [{"start_s": 0.0, "end_s": 0.12, "phone": "aa", "role": "phone"}],
  "window_bounds_s": [0.0, 0.5],
  "bin_bounds_s": [0.0, 0.1, 0.2, 0.5]
}
```

For fresh inference, replace `posterior_npz` with `audio_path`, `audio_sha256`,
`model_snapshot`, and `model_snapshot_sha256`. The runner calls
`ctc_phone_posteriors(..., drop_special_tokens=False, local_files_only=True)`;
it never downloads or substitutes a model. Run it with:

```bash
PYTHONPATH=src python scripts/qualify_singing_phones.py \
  --manifest tests/benchmarks/thresholds/singing_phones_v1.json \
  --root /path/to/pinned-data --artifact-dir /tmp/singing-artifacts \
  --out-json /tmp/singing-report.json
```

Profiles integrate disjoint native cells, retaining blank, boundary, special,
unmapped-phone, and mapped-phone mass separately. Zero mapped mass produces a
missing (`null`) profile. Every reference interval must explicitly declare
`role: "phone"` or `role: "nonphone"` (or use the narrow
`reference_nonphone_labels` policy). Positive-duration phone intervals must be
chronological and non-overlapping; the runner rejects overlap rather than
arbitrarily choosing a label. Reference-only phones remain in profile vectors
with predicted zero mass and lower reference-mapping coverage.

Reports include reference duration/coverage for each composition window and
ordered bin, both comparison families, and valid/missing profile-bin counts.
Recognition is independent free CTC greedy decoding with raw-ID collapse before
blank/boundary/special removal and true Levenshtein S/D/I/PER. Unmapped model
tokens use a reserved sentinel that reference labels cannot use. The report has
no invented acceptance gates. Version 1 accepts development manifests only;
qualification needs a separately verified protocol with untouched partitions.

## Pinned NUS-48E development pilot

`scripts/prepare_singing_phone_pilot.py` makes a concrete manifest from the
original [NUS laboratory distribution](https://smcnus.comp.nus.edu.sg/).
`singing_phone_pilot_v1.json` pins every source file and model artifact by URL
and hash. NUS permits research use; cite Duan et al., APSIPA ASC 2013. Raw
recordings and annotations are kept outside this repository.

The pilot uses the first 16 seconds of ADIZ/01, JLEE/05, JTAN/07 and KENN/04,
all previously examined downstream. It compares read speech, clean singing,
singing mixed at 0 dB with known instrumental backing, and the canonical
Demucs vocal stem from that mixture. Two fixed models yield 32 cases:
[base English CTC](https://huggingface.co/bobboyms/wav2vec2-base-en-phoneme-ctc-41h/tree/04111e5d2ee8030f715b057f076defa227827454)
and [large multilingual CTC](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft/tree/ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4).
No model is selected from these results. The [original corpus paper](https://apsipa.org/proceedings/2013/papers/364_NUS-corpus-Duan-2927261.pdf)
describes manual singing annotations and automatically aligned read-speech
boundaries; the latter provide a lower-grade diagnostic comparator.
The fixed crop is not matched for linguistic content
between spoken and sung versions, so condition differences are descriptive.

Four-second composition windows and two-second ordered bins are benchmark
choices, independent of study-specific 5.49-second windows. Reference `sil`
includes silence and inhalation; it is excluded from phone targets and cannot
establish vocal absence. Zero-duration `sp` is a word boundary. All other
reference phones, including unvoiced consonants, remain eligible regardless
of F0. The mixture's reference mask controls only its specified mixing level;
recognition never receives reference labels.

The explicit IPA mapping removes stress and maps known English phones to
broad NUS labels. Specified compound tokens expand to multiple phones;
their posterior mass is divided equally across those phones. Unmapped
multilingual tokens and `<unk>` count as unknown recognition tokens and
remain visible in coverage. Flaps and glottal stops are not silently merged
with reference stops. Blank, boundary, BOS and EOS are excluded from phone
profiles and decoding. Integrated CTC evidence is not a phone-duration
estimate: comparison with duration-weighted reference profiles is a
diagnostic of representation validity.

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python \
  scripts/prepare_singing_phone_pilot.py \
  --assets /path/to/nus-and-models --download \
  --backing-assets /path/to/preserved-vocal-assets \
  --separator-checkpoint /path/to/955717e8-8726e21a.th \
  --out-dir /tmp/singing-pilot-inputs
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python \
  scripts/qualify_singing_phones.py \
  --manifest /tmp/singing-pilot-inputs/manifest.json \
  --artifact-dir /tmp/singing-pilot-native \
  --out-json /tmp/singing-pilot-report.json
```

`--download` retrieves only missing pinned public assets (including about
1.7 GB of model weights). Omit it when assets already exist. Every file is
verified before use. Neither the 96/200 downstream support rule nor neural
fit quality supplies an upstream acceptance threshold. Fresh data and a
pretraining-overlap audit remain necessary for qualification.

The pinned large checkpoint's tokenizer also requires `phonemizer` and an
eSpeak shared library at load time, even though reference text is never sent
to it. The recorded run used `phonemizer==3.3.0` and `protobuf==6.33.5`, with
`PHONEMIZER_ESPEAK_LIBRARY` pointing to the installed library. See the
[upstream installation instructions](https://bootphon.github.io/phonemizer/install.html)
if automatic library discovery fails. A missing dependency is a runtime
failure, not a recognition score; retain that receipt before rerunning.

If retained posterior supports have native-time gaps, greedy CTC collapse resets
at each gap. Reports mark this as a partial path with `full_path_observed=false`
and give `n_discontinuities`; its PER retains all supplied reference phones and is not a
whole-recording recognition result. Each completed item records its provenance,
source hashes, artifact path and artifact SHA256. CLI reports are atomically
checkpointed after every item, including failures.

The conservative mapping covers 110 base-model phone columns but only 59 large-model columns; 330 large-model phone columns remain unmapped, including some English variants as well as other languages. Cross-model PER and profile differences therefore combine acoustic behavior with mapping coverage. They do not isolate recognizer quality or establish that one model is better.
