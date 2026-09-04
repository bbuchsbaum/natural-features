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
| `audio.formants` | Default `lpc_autocorr` poles; optional Praat via `backend=parselmouth` |
| `audio.egemaps` | Vendor LLD dump, not the designed A3 band |
| `speech.phonemes` | Uniform split of word intervals, not acoustic phone alignment |
| `speech.phones.mfa` | MFA `phones` TextGrid tier |
| `speech.articulatory` | Orthographic letter-ratio proxy |
| `speech.articulatory.from_posteriors` | Static 22-D place/manner map from posteriors |
| `speech.phonology.distinctive_from_posteriors` | Broader English distinctive-feature occupancy |
| `speech.articulatory.sparc` | SPARC template EMA (12 channels, no velum/larynx) |
| `speech.articulatory.gestures` | Canonical overlapping activations from phone labels |
| `speech.articulatory.dynamics` | Finite-difference velocity/acceleration of a G series |

Canonical gestures derived from phones are almost a function of P. Unique
variance of G over P requires waveform-conditioned inversion (`sparc`), not
the gesture control layer.

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
posteriors are PPGs when the `ppgs` extra is installed, otherwise CTC.
