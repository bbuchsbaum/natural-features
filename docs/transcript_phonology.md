# Transcript phonology and acoustic comparison

This experimental workflow supplies interpretable sentence representations for
descriptive representational-similarity comparisons. It works offline with the
base installation (NumPy and PyYAML). It adds no runtime dependencies on
CMUdict's Python wrapper, Pronouncing, Pincelate, PanPhon, SciPy, or neural models.

## Pronunciations, not observed phones

```python
from natural_features.features.speech.pronunciation import extract_pronunciations
from natural_features.features.speech.phone_sequences import NgramSchema, pronunciation_counts

sequences = extract_pronunciations(
    {"s1": "The cat sat.", "s2": "The cat slept."},
    selection="dictionary_order",  # explicitly choose first listed variant
)
schema = NgramSchema(n=2, boundary="within", stress="strip", normalization="frequency")
bigrams = schema.transform(sequences)
counts = pronunciation_counts(sequences)
bigrams.save("sentence_bigrams")  # .npz data and .json schema/provenance
```

The default `selection="retain"` selects unique pronunciations and leaves
ambiguous words unselected. `dictionary_order` is deterministic, not an assertion
about the recording. Both retain all alternatives, original stress-bearing phone
labels, and dictionary variant labels. There are no estimated confidences.

`PronunciationSequence/v1` preserves item IDs, original text, source-character
spans, word indices, word boundaries and phone order. It has no times or durations.
`ItemFeatures/v1` has item coordinates, feature coordinates, values, a same-shaped
validity mask and provenance. These are documented experimental Python APIs,
not new IDs for `nf extract`, temporal `FeatureBundle` payloads, or R exports.
Existing temporal and stable root APIs are unchanged.

The bundled North American English CMUdict snapshot is identified by full Git
revision and SHA-256 in `features/speech/data/cmudict.json`. Its license travels
with the resource. Loading verifies the data hash. A maintainer can explicitly
refresh it with `python scripts/update_cmudict.py --revision <full-SHA>`; extraction
never downloads data. Resource updates require fixture review and new provenance.

### Text and coverage policy

- Lookup uses NFC, lowercase, curly-to-straight apostrophes and normalization of
  two common Unicode hyphens. Original text and spans are preserved.
- Paired single quotes enclosing a token are stripped for lookup. A leading or
  trailing apostrophe is first looked up as written (`'em`, `dogs'`); only if
  that form is absent and the unquoted form is present is it treated as a
  quotation mark (`‘Stop, he said` → `stop`). Internal apostrophes always remain
  part of the token. The span and original text still include the mark.
- Punctuation is excluded from pronunciation; it does not insert a silence phone.
  Cross-word bigrams may cross punctuation, exactly as other word boundaries.
- Contractions, possessives and compounds are looked up as whole tokens. By
  default no suffix rules, compound splitting, accent stripping, number expansion
  or G2P is applied. The opt-in composition step below is the only exception and
  is never silent. Numeric forms and unsupported symbols remain explicit
  unresolved tokens when absent from the lexicon. Underscore-containing strings
  are whole tokens. Word count means non-punctuation source tokens.
- Vowels require original stress labels `0`, `1` or `2`; consonants are unstressed.
  Unsupported phone symbols are errors in supplied pronunciations.
- Empty input remains an item, with undefined coverage and ineligible features.
  Unresolved or unselected words invalidate a sentence's pronunciation models.
  Diagnostic partial counts remain available under a false validity mask.
  Ngrams never bridge unresolved or unselected source words.

```python
from natural_features.features.speech.pronunciation import PronunciationLexicon, coverage_report

supplement = PronunciationLexicon.from_mapping(
    {"myterm": [["M", "AY1", "T", "ER2", "M"]]}, name="study-lexicon-v1"
)
sequences = extract_pronunciations(
    {"s1": "I read myterm."},
    selection="dictionary_order",
    supplemental=supplement,
    overrides={"s1": {1: ["R", "EH1", "D"]}},  # source-word index, zero-based
)
report = coverage_report(sequences)
```

Supplemental entries supersede dictionary entries for matching tokens. Item
overrides affect only the named source word and retain dictionary alternatives.
Both are labeled user-supplied/unverified: supplying phones does not establish
audio verification. Supplement hashes and the exact overrides are recorded.
`coverage_report(..., pairs=[("s1", "s2"), ...])` reports affected items and pairs
for each unresolved type. Affected-pair counts overlap and are not additive or
promises of recovered coverage. Unselected alternatives are reported separately.

### Estimating missing words from dictionary parts

`extract_pronunciations(..., compose=True)` estimates a word found in neither the
dictionary nor the supplement by rebuilding it from entries in both. The fixed
rules (`affix-compound-v1`, recorded in metadata) are:

- regular suffixes `-s/-es/-ies`, `-ed/-ied`, `-ing`, `-er`, `-est`, `-y`,
  `-ish`, `-able`, `-ly`, `-ness`, `-less`, `-ful` and `-ment`, with silent-`e`,
  doubled-consonant and `y`/`i` spelling changes (`huggable` = `hug` + `-able`);
- `'s` and plural `s'` possessives (`snowflake's`, `dogs'` = `dogs`);
- `-s` and `-ed` endings chosen from the base's final phone (`/IH0 Z/`, `/S/`,
  `/Z/`; `/IH0 D/`, `/T/`, `/D/`); the base keeps its own stress;
- closed compounds of two parts, each at least 3 and 4 letters, where the
  second part's primary stress becomes secondary (`mouth` + `watering`);
- hyphenated words whose parts all resolve, with stress unchanged.

At most two rules are stacked. Suffix analyses take precedence over compound
splits, and a suffix never attaches to a guessed compound. Analyses that
disagree (`hop`/`hope` cannot both explain a form) all become alternatives
under the usual `selection` rule, so ambiguity is never resolved silently.
Composed words have origin `composed_estimated` and derivation labels such as
`composed:hug+able`. Their indices are listed in `coverage["estimated_indices"]`,
and `coverage_report` lists them under `estimated`, separate from dictionary
words and from words that remain `unresolved`. A sentence that relies on an
estimate is eligible; studies should report how many items do (`estimated_items`)
and can rerun without `compose` as a sensitivity check. Supplemental entries and
item overrides always take precedence. Words that no rule explains, such as
`cerulean`, stay unresolved and need a supplement. No statistical model, weights
or external package is involved.

Held-out check against the bundled CMUdict
(`examples/validate_pronunciation_composition.py`; each word rebuilt without its own
entry; 125,938 alphabetic entries, 59,564 composable). Stress is ignored in
the phone error rate.

| Rule | Words | Exact | Exact ignoring stress |
| --- | ---: | ---: | ---: |
| `'s` possessive | 6,316 | 92.4% | 93.9% |
| `-s` | 14,066 | 87.8% | 89.6% |
| `-ed` | 4,659 | 87.1% | 89.3% |
| `-ing` | 4,545 | 86.6% | 88.9% |
| `-er` | 4,336 | 68.2% | 69.7% |
| `-y` | 3,171 | 53.0% | 55.9% |
| closed compound | 18,312 | 30.5% | 51.9% |

Overall phone error rate is 6.2%. CMUdict is dominated by surnames, which are
not regular derivations (`vin` + `cent`, `king` + `man` with a reduced vowel),
so the compound and `-y` rows understate accuracy for ordinary vocabulary: 34
of 36 common closed compounds (`sunflower`, `toothbrush`, `lighthouse`, ...)
are reproduced exactly, and the other two fall below the 4-letter limit.
These are canonical estimates, never audio verification.

## Representations and distances

`NgramSchema` supports unigrams and bigrams, `count` or L1 `frequency`, and
`strip` or `keep` stress. Bigram boundary modes are `within`, `cross` (only the
last/first phones of adjacent source words), and `all`. No boundary symbols are
inserted. The complete fixed inventory gives 39 unigram / 1,521 bigram coordinates
without stress; stress-preserving inventories have 69 / 4,761 coordinates.
Vocabulary, ordering, normalization and checksum are serialized by `to_dict()`;
`from_dict()` validates them. There is no fitting, unknown bucket, TF-IDF or PCA.
An all-zero vector has undefined cosine distance, even if pronunciation is complete.

Counts are separate: word count, selected phone count and vowel-nucleus count
(a canonical syllable-count approximation, not syllabification of the recording).

`articulatory_sequence(sequence)` returns an ordered bundle for each phone with
word index and original label. `phone_feature_bundles()` exposes the versioned
mapping. `PhoneEditDistance` and `sequence_distances` provide a second,
order-sensitive model. Insertion/deletion cost one; consonant substitution is the
weighted fraction of mismatched place, manner and voicing categories. Vowels use
height, backness, rounding and rhoticity, averaged over two ordered endpoints;
monophthongs repeat one endpoint. Cross-class substitution costs one.

Default group weights are equal. Stress is ignored by default; a positive
`stress_weight` requires original vowel stress and adds a separate stress term.
`weighted=False` gives ordinary phone edit distance; with `stress_weight > 0`
a vowel substitution costs `(identity mismatch + stress_weight * stress
mismatch) / (1 + stress_weight)`, the same convex form as the weighted path. `normalize=True` divides by the larger phone count;
empty/empty is zero at the numeric kernel, but empty sentence models are masked.
Word boundaries are preserved in the sequence objects but are not edit symbols.

These English feature categories and diphthong endpoints are explicit modeling
conventions. They are not articulator measurements, PanPhon output, or a claim
that normalized edit distance is a metric or perceptual distance. The mapping
builds on this package's phone categories and the feature-sequence idea illustrated
by [Pincelate](https://github.com/aparrish/pincelate/blob/main/pincelate/featurephone.py).
No Pincelate code, trained weights, or neural runtime is included. The small
dictionary interface follows the utility of [Pronouncing](https://github.com/aparrish/pronouncingpy),
without depending on that package.

## Acoustic blocks

`extract_sentence_acoustics({item_id: AudioStimulus, ...}, config=AcousticConfig())`
returns raw native blocks and per-item summaries. Default input is 16 kHz;
mismatches fail. Channel mixing is arithmetic averaging. Frequency limits are
fixed at 50–7,500 Hz across items. No resampling, trimming, speech synthesis,
speaker normalization or speech detection occurs.

- `spectral_raw`: mean/SD of 32 log10 mel-power bands, 25 ms windows/10 ms hops.
- `spectral_shape`: subtract each frame's across-band mean before aggregation.
  This removes common gain above the extractor's power floor; it is not an
  unconditional gain invariance. Raw mel remains available for comparison.
- `envelope`: mean/SD of the existing Hann-windowed RMS extractor.
- `modulation`: existing 2-D FFT kernel on a log-frequency representation of an
  ERB triangular STFT bank, not physiological gammatone filters or NSL aud2cor.
  Temporal bands are 1–2, 2–4, 4–8, 8–16 and 16–32 Hz; signed spectral bands
  are .25–.5, .5–1, 1–2 and 2–4 cycles/octave. FFT duration is 2 seconds,
  window hop .5 seconds, cochleagram hop 5 ms. The full audio support includes
  the final STFT window: 2.02 seconds at default settings.
- `controls`: clip duration and unwindowed whole-clip RMS. Optional pitch adds
  voiced-duration fraction and a separate pitch block. Pitch is the existing
  monophonic autocorrelation estimator, not source-specific vocal F0.

All silence is retained. All-zero clips have unavailable spectral/modulation
representations and valid zero level. Short/padded frames and clips without a
complete modulation window remain invalid; no synthetic padding is used to
invent modulation support. A modulation band requires at least two cycles at
its lower temporal edge and a resolved FFT cell; bands beyond Nyquist are invalid.
This is a fixed support rule, not an accuracy guarantee. Unvoiced pitch values
never enter F0 means/SDs. Pitch strength remains separately available.

Native bounds use rounded sample indices and preserve the source clock/context.
Overlapping windows receive weight `integral(1 / active_window_count)` over their
clipped audio support. Weights sum to covered union duration. Mean and population
SD use those weights and each column's validity; metadata records covered duration
and valid weight per column. Uncovered tails are reported through coverage.

`FeatureScaler.fit(features, reference_ids=[...])` requires a declared reference
set. It freezes per-column mean/population SD, excludes columns invalid on any
reference or numerically constant, and preserves target invalidity. Exported
scalers include coordinates, semantic schema, reference IDs and reference-data
hash. Transform never refits. `combine_scaled_blocks(..., weights=...)` applies
`sqrt(weight / (column_count * sum_weights))` to each standardized block so
Euclidean distance combines explicitly weighted block mean-square distances.

## Run the complete comparison

Prepare the six checksum-pinned, CC BY 4.0 LibriSpeech utterances:

```bash
python scripts/fetch_phonology_demo.py
PYTHONPATH=src python examples/phonology_comparison.py \
  artifacts/phonology-demo/manifest.json \
  --selection dictionary_order --fit-on-all --within-group \
  --out artifacts/phonology-comparison
```

The preparation script needs FFmpeg for lossless FLAC-to-PCM decoding. It streams
a bounded portion of the archive and checks individual resource and decoded-PCM
hashes; it does not download the full archive. Extraction from the resulting WAVs
uses only base dependencies. Binaries remain untracked under `artifacts/`.
`examples/phonology_demo.json` records attribution, license, transcripts and hashes.
The demo group label is an example grouping, not a real experimental run.

A study manifest is JSON with `items`, each containing `item_id`, `text`, `audio`
(relative to the manifest), and optionally `group`. Optional top-level
`supplemental`, `overrides` (word indices as JSON keys), `compose` (boolean,
default false), `pairs` (item-ID pairs),
and `acoustic_config` follow the APIs above. Duplicate IDs, unknown pair members,
self-pairs and duplicate unordered pairs fail. `--within-group` intersects the
requested pairs with within-group pairs; missing group labels fail.

Use `--reference-ids id1 id2 ...` for explicit scaling references. `--fit-on-all`
is an explicitly descriptive full-set audit, not held-out predictive evidence.
The default acoustic blocks are spectral shape and modulation, with equal block
weights. `--acoustic-blocks spectral_shape` is an explicit short-clip alternative;
the workflow never silently drops an unavailable block. All constant/invalid
reference coordinates produce an actionable error if an entire block disappears.

Existing text embeddings can be supplied with `--embeddings embeddings.npz
--embedding-metadata embeddings.json`. NPZ contains Unicode `item_ids` in exact
manifest order and a numeric `values` matrix. JSON must contain `model`, `revision`
and `text_sha256` mapping every item ID to the SHA-256 of its exact UTF-8 transcript.
The example verifies identity/order/hashes, records the embedding artifact hash,
and uses cosine distance. It neither generates embeddings nor substitutes a text
proxy. The public demo is acoustic/phonological unless embeddings are supplied.

Outputs include untimed pronunciation/ordered-articulatory records, coverage,
feature matrices with schemas and validity, frozen scalers, the edit-cost
specification, raw native acoustic values/bounds/masks/weights, RDMs, and
`overlap.json`. The latter contains one explicit candidate-pair table with raw
distances, controls, inclusion mask and exclusion reasons. Every comparison uses
the intersection of valid support across all selected representations and controls.

The audit reports Pearson/Spearman correlations, joint-design rank/conditioning,
and descriptive conditional correlations controlling word/phone/syllable count
differences, duration/level differences, lexical-set Jaccard distance and pair-group
indicators. Spearman ranks are computed on matched pairs, before residualizing
ranked distances and numeric controls; group indicators remain categorical.
Constant vectors or insufficient residual degrees of freedom yield JSON null.
The raw RDMs are preserved; residuals do not replace the original representation.

There are no ordinary pair-independent p-values. RDM regression for brain data,
searchlights, HRF convolution, cross-validation design and group inference remain
downstream. Six public utterances exercise reproducibility, not empirical validity
for the 120-sentence pilot. Neural G2P, trigrams, PanPhon and learned phonological
embeddings remain deferred until coverage or comparison evidence justifies them.
