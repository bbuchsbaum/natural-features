# Speech representational ladder

This library can emit time-aligned feature bands along

waveform → cochlear/envelope (A1) → spectrotemporal modulation (A2) →
acoustic-phonetic cues (A3) → phones and distinctive features (P) →
articulatory kinematics (G) → motor dynamics (M).

The bands are **gap extractors**: they are registered in the Python catalogue
and are not part of the R-public contract. Do not treat catalogue IDs as
interchangeable with paper names.

## Honesty

| ID | What it actually is |
|---|---|
| `audio.gammatone` | ERB triangular filterbank on STFT power, not a cochleagram |
| `audio.envelope` | Hilbert/RMS envelope plus derivative and onset |
| `audio.modulation.spectrotemporal` | STFT rate/scale energy (`backend=stft_rate_scale`), not NSL `aud2cor` |
| `audio.modulation.mps` | Windowed 2-D FFT of a log-frequency cochleagram; not the ladder A2 band |
| `audio.formants` | Default `lpc_autocorr` poles; optional Praat via `backend=parselmouth` |
| `audio.harmonicity` | Autocorr HNR for the A3 ladder band; not `audio.periodicity` |
| `audio.egemaps` | Vendor LLD dump, not the designed A3 band |
| `speech.phonemes` | Uniform split of word intervals, not acoustic phone alignment |
| `speech.phones.mfa` | MFA `phones` TextGrid tier |
| `speech.articulatory` | Orthographic letter-ratio proxy |
| `speech.articulatory.from_posteriors` | Static 22-D place/manner map from posteriors |
| `speech.phonology.distinctive_from_posteriors` | Broader English distinctive-feature occupancy; no `aspirated` (context-free) |
| `speech.phonology.distinctive_from_phoneme_events` | Same set plus contextual `aspirated` (voiceless stop released into vowel/approximant, not after /s/) |
| `speech.articulatory.sparc` | SPARC template EMA (12 channels, no velum/larynx) |
| `speech.articulatory.gestures` | Canonical overlapping activations from phone labels; with `stimulus` they are gain-modulated by measured intensity and voicing |
| `speech.articulatory.dynamics` | Finite-difference velocity/acceleration plus kinematic `effort` and co-activation `overlap` of a G series |
| `speech.syllables.onc` | Maximal-onset heuristic syllable onset/nucleus/coda occupancy, not lexicon syllabification |

Canonical gestures derived from phones alone are a function of P. Passing the
stimulus multiplies canonical targets by measured intensity and voicing gains,
which makes G a P-by-acoustics interaction (token-level reduction and
hyperarticulation) rather than a relabeling of P. Full unique variance of G
still requires waveform-conditioned inversion (`sparc`). `overlap` in the
dynamics band is meaningful only for activation-like G series, not signed EMA
positions. Predicted sensory consequence remains out of scope for band M.

SPARC is a speaker-agnostic template space trained for inversion/synthesis.
It is not measured EMA and it does not represent velum or larynx. Those
dimensions stay phonological until a later extra.

## Residual bands

`features.residualize` resamples a target and its predictors onto an explicit
analysis hop and returns the OLS residual of each target column.

That is a constructed feature space. It does **not** replace nested \(R^2\)
or commonality analysis. Full-stimulus OLS leaks if the same residuals are
later used as encoding features across the same timepoints. Residualize
inside cross-validation folds in the modeling library (fmrimod).

HRF convolution and TR grids stay downstream.

## Workflow

`extract_speech_ladder` and preset `fmri_speech_ladder` emit raw bands and
optional residual keys such as `a2|a1` and `g_ema|a+p`. Default phone
posteriors are PPGs when the `ppgs` extra is installed, otherwise CTC. When
phone events are supplied, gestures are acoustically gain-modulated and an
`m_syllables` onset/nucleus/coda band is emitted.
