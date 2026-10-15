# Estimated vocal sources and source pitch

Install `natural-features[vocals]`. These optional primitives do not qualify a
recording as containing singing and do not replace a domain validation study.

`separate_vocals(audio, model_repo=...)` returns `SeparatedAudio`: named
`AudioStimulus` waveforms and provenance. Supply explicitly resampled 44.1 kHz
mono or stereo audio, and a dedicated local directory containing the prefetched
Demucs `.th` and bag `.yaml` files. Default `htdemucs` uses 7.8-second segments,
25% overlap and zero random shifts. All samples are processed; output retains
input length, clock, offset and clock mappings. Source labels are model
estimates. Segmentation and full-input normalization introduce future access.
Repository filenames and full hashes, backend version and input hashes are
recorded; record the natural-features Git revision in the analysis manifest.

`source_pitch_crepe(audio, source_role=..., source_provenance=...)` accepts
explicit 16 kHz mono audio. Use the returned separation provenance for an
estimated vocal stem, or describe the reference/source in a nonempty mapping.
The returned columns are raw F0 in Hz, uncalibrated network periodicity and RMS.
There is no implicit voiced threshold or silent-frame acceptance. Periodicity
can be high for instruments or silence. Vocal absence, invalid F0 and missing
support must be represented separately by the caller; never convert absence
into a measured zero-pitch event.

Frames contain exactly 1,024 samples (64 ms), without padding. At a 20 ms hop,
centers are 0.032, 0.052, ... seconds plus the input offset. Partial edge frames
are excluded. Native interval bounds and named clocks are retained. Weighted
argmax is frame independent, so batch size does not change Viterbi context.
A pitch row's local window is not the receptive field of an upstream separator.
The default model is full CREPE; tiny is an explicitly different diagnostic.

Before scientific use: test known contours, negative controls, existing labeled
singing references, missing/edge support and clean/mixed/separated conditions.
Report preprocessing and training-set overlap; MIR-1K was among CREPE's training
corpora, so an arbitrary MIR-1K subset is not independent validation. Evaluate
phone recognition on labeled clean singing as well as mixtures: successful
forced alignment with supplied phones is not evidence of correct recognition.
