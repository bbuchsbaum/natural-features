"""Curated, source-checked explanations for the public feature catalogue.

The live registry owns structural facts such as modalities, dependencies,
schemas, costs, and defaults.  This module adds the reader-facing meaning and
method references that cannot be inferred from those fields.  The docs build
fails if either side gains or loses a public feature ID.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from natural_features.workflows import FeatureCatalogEntry, available_features


@dataclass(frozen=True)
class FeatureGuide:
    meaning: str
    references: tuple[str, ...] = ()


REFERENCES: dict[str, tuple[str, str]] = {
    "adelson1985": (
        "Adelson & Bergen (1985), spatiotemporal energy",
        "https://doi.org/10.1364/JOSAA.2.000284",
    ),
    "ast": ("Gong et al. (2021), AST", "https://arxiv.org/abs/2104.01778"),
    "bert": ("Devlin et al. (2019), BERT", "https://aclanthology.org/N19-1423/"),
    "blazeface": (
        "Bazarevsky et al. (2019), BlazeFace",
        "https://arxiv.org/abs/1907.05047",
    ),
    "clap": (
        "Elizalde et al. (2022), CLAP",
        "https://arxiv.org/abs/2206.04769",
    ),
    "clip": (
        "Radford et al. (2021), CLIP",
        "https://proceedings.mlr.press/v139/radford21a.html",
    ),
    "ctc": (
        "Graves et al. (2006), CTC",
        "https://doi.org/10.1145/1143844.1143891",
    ),
    "dct": (
        "Ahmed, Natarajan & Rao (1974), discrete cosine transform",
        "https://doi.org/10.1109/T-C.1974.223784",
    ),
    "dinov2": (
        "Oquab et al. (2023), DINOv2",
        "https://arxiv.org/abs/2304.07193",
    ),
    "egemaps": (
        "Eyben et al. (2016), GeMAPS/eGeMAPS",
        "https://doi.org/10.1109/TAFFC.2015.2457417",
    ),
    "farneback": (
        "Farnebäck (2003), polynomial-expansion optical flow",
        "https://doi.org/10.1007/3-540-45103-X_50",
    ),
    "glover": (
        "Glover (1999), hemodynamic response model",
        "https://doi.org/10.1006/nimg.1998.0419",
    ),
    "hale": (
        "Hale (2001), probabilistic surprisal",
        "https://aclanthology.org/N01-1021/",
    ),
    "hubert": (
        "Hsu et al. (2021), HuBERT",
        "https://arxiv.org/abs/2106.07447",
    ),
    "mfcc": (
        "Davis & Mermelstein (1980), mel-frequency cepstra",
        "https://doi.org/10.1109/TASSP.1980.1163420",
    ),
    "pyannote": (
        "Bredin et al. (2020), pyannote.audio",
        "https://doi.org/10.1109/ICASSP40776.2020.9052974",
    ),
    "wavlm": (
        "Chen et al. (2022), WavLM",
        "https://arxiv.org/abs/2110.13900",
    ),
    "whisper": (
        "Radford et al. (2023), Whisper",
        "https://proceedings.mlr.press/v202/radford23a.html",
    ),
}


FEATURE_GUIDE: dict[str, FeatureGuide] = {
    "audio.ast": FeatureGuide(
        "Window-level Audio Spectrogram Transformer embeddings when the model is available; fallback mode emits a deterministic mel-spectral proxy and records that substitution.",
        ("ast",),
    ),
    "audio.clap": FeatureGuide(
        "Window-level embeddings from a contrastive language-audio model when available; fallback mode emits a deterministic mel-spectral proxy rather than CLAP representations.",
        ("clap",),
    ),
    "audio.egemaps": FeatureGuide(
        "Frame-level eGeMAPS low-level descriptors through openSMILE. Without openSMILE, fallback mode returns RMS plus five spectral summaries, not the eGeMAPS descriptor set.",
        ("egemaps",),
    ),
    "audio.gammatone": FeatureGuide(
        "Log energy in an ERB-spaced auditory filterbank. The implementation is a triangular frequency-domain approximation, not a time-domain gammatone filter cascade.",
    ),
    "audio.mel": FeatureGuide(
        "Short-time power pooled into triangular mel-frequency bands, returned as log10 energy by default.",
        ("mfcc",),
    ),
    "audio.mfcc": FeatureGuide(
        "An orthonormal cosine transform of log-mel energies, optionally followed by first and second temporal differences.",
        ("mfcc",),
    ),
    "audio.pitch": FeatureGuide(
        "Framewise fundamental-frequency estimate and normalized voicing strength from autocorrelation; unvoiced frames receive F0 = 0.",
    ),
    "audio.prosody": FeatureGuide(
        "Six framewise controls: RMS, log-RMS, F0, voicing strength, spectral centroid, and zero-crossing rate.",
    ),
    "audio.resample": FeatureGuide(
        "A new AudioStimulus sampled at the requested rate by linear interpolation, with the original start offset retained.",
    ),
    "audio.rms": FeatureGuide(
        "Hann-windowed root-mean-square amplitude for each audio frame; a compact measure of short-time signal energy.",
    ),
    "audio.spectral_stats": FeatureGuide(
        "Five framewise summaries: spectral centroid, bandwidth, 85% rolloff, flatness, and zero-crossing rate.",
    ),
    "audio.trim": FeatureGuide(
        "A time slice of an AudioStimulus; it preserves the source and advances the absolute start offset to the retained first sample.",
    ),
    "events.align": FeatureGuide(
        "A passthrough EventSeries node that preserves intervals and annotations while recording alignment-step provenance. It does not estimate a new alignment.",
    ),
    "features.hrf": FeatureGuide(
        "Columnwise convolution of a FeatureSeries with a sampled Glover-style hemodynamic response kernel.",
        ("glover",),
    ),
    "features.lag": FeatureGuide(
        "Causal, zero-padded lag copies of every feature column. Lags are row offsets, not seconds.",
    ),
    "features.resample": FeatureGuide(
        "A FeatureSeries placed on a regular time grid using bin means or linear interpolation.",
    ),
    "image.ocr": FeatureGuide(
        "Tesseract word boxes, confidence values, and relative image coordinates as events. Fallback mode returns an empty, provenance-marked EventSeries.",
    ),
    "language.bert": FeatureGuide(
        "BERT hidden-state vectors pooled over subwords for each word in isolation. The current extractor does not encode the full sentence jointly; fallback vectors are deterministic hashes.",
        ("bert",),
    ),
    "language.discourse": FeatureGuide(
        "Five deterministic word-level controls: normalized position, repetition flag, recurrence distance, local type-token ratio, and a heuristic content-word flag.",
    ),
    "language.hidden_states": FeatureGuide(
        "Selected causal-language-model hidden states pooled over subwords for each word in isolation. Fallback vectors are deterministic hashes rather than model activations.",
    ),
    "language.surface": FeatureGuide(
        "Public alias of language.discourse with the same five deterministic surface and recurrence controls.",
    ),
    "language.surprisal": FeatureGuide(
        "A deterministic token-length and character-diversity score named surprisal_proxy. It is not negative log probability from the configured language model.",
        ("hale",),
    ),
    "language.syntax": FeatureGuide(
        "Eight word-level indicators for length, function-word status, capitalization, punctuation, coarse part of speech, and sentence boundary. spaCy supplies POS labels when available; fallback mode uses suffix rules.",
    ),
    "speech.articulatory": FeatureGuide(
        "Five orthographic proxies computed from letters in each word label: vowel, labial, coronal, and dorsal ratios plus starts-with-vowel. This route does not infer articulatory features from aligned phones.",
    ),
    "speech.ctc": FeatureGuide(
        "Framewise phone-class probabilities from a phoneme CTC model. Fallback mode produces normalized mel-band acoustic phone proxies and records the substitution.",
        ("ctc",),
    ),
    "speech.diarization": FeatureGuide(
        "Speaker-activity tracks from pyannote.audio. Fallback mode returns one VAD-derived speaker track, not multi-speaker diarization.",
        ("pyannote",),
    ),
    "speech.emotion": FeatureGuide(
        "Whole-clip class probabilities from an audio-classification model. Fallback mode instead emits framewise arousal, valence, dominance, and voicing proxies derived from prosody.",
    ),
    "speech.hubert": FeatureGuide(
        "Selected framewise HuBERT hidden states. If the local model cannot run, deterministic acoustic projections are returned with fallback provenance.",
        ("hubert",),
    ),
    "speech.neural_vad": FeatureGuide(
        "Framewise speech probability from Silero VAD when available; fallback mode uses a deterministic energy-based probability proxy.",
    ),
    "speech.phonemes": FeatureGuide(
        "Phoneme events made by splitting each word interval uniformly across whitespace-separated phone labels already stored in that word. It is not grapheme-to-phoneme conversion.",
    ),
    "speech.vad": FeatureGuide(
        "Contiguous speech intervals obtained by thresholding a short-time energy probability series.",
    ),
    "speech.wavlm": FeatureGuide(
        "Selected framewise WavLM hidden states. If the local model cannot run, deterministic acoustic projections are returned with fallback provenance.",
        ("wavlm",),
    ),
    "speech.words": FeatureGuide(
        "Whisper segment and word events with timestamps, confidence, and alignment QC. A supplied transcript is distributed deterministically over the clip; other fallback paths return provenance-marked approximations.",
        ("whisper",),
    ),
    "text.tokenize": FeatureGuide(
        "Regex word tokens as EventSeries intervals. Existing TextStimulus timings are preserved; otherwise the requested duration is divided uniformly among tokens.",
    ),
    "video.audio.extract": FeatureGuide(
        "The mono audio stream decoded from a video file by ffmpeg, optionally clipped and resampled during decoding.",
    ),
    "video.frames.sample": FeatureGuide(
        "A VideoStimulus containing every nth frame, or frames nearest a requested target rate; absolute start time is preserved.",
    ),
    "video.ocr": FeatureGuide(
        "Tesseract word boxes and confidence values from sampled video frames, returned as frame-timed events with relative coordinates.",
    ),
    "video.trim": FeatureGuide(
        "Video frames whose absolute frame times fall inside a requested half-open interval.",
    ),
    "vision.clip": FeatureGuide(
        "Per-frame CLIP image embeddings when the model is available; fallback mode returns deterministic projected image summaries, not CLIP vectors.",
        ("clip",),
    ),
    "vision.dct": FeatureGuide(
        "Low-spatial-frequency two-dimensional cosine coefficients from resized grayscale frames, ordered from lower to higher combined frequency.",
        ("dct",),
    ),
    "vision.dino": FeatureGuide(
        "Selected per-frame DINOv2 representations when the model is available; fallback mode returns deterministic projected image summaries.",
        ("dinov2",),
    ),
    "vision.energy": FeatureGuide(
        "Framewise mean luminance, contrast, saturation, and gradient edge energy, with first differences by default.",
    ),
    "vision.face": FeatureGuide(
        "Per-frame face presence, count, total box area, and mean box center from MediaPipe. Fallback mode emits color/contrast proxies that must not be interpreted as detections.",
        ("blazeface",),
    ),
    "vision.frame_diffs": FeatureGuide(
        "Mean and 95th-percentile absolute grayscale change between successive frames.",
    ),
    "vision.motion": FeatureGuide(
        "Mean and 95th-percentile magnitude of a spatial-temporal gradient proxy. Despite the historical label, this route does not estimate optical flow.",
    ),
    "vision.motion_energy": FeatureGuide(
        "Spatiotemporal motion-energy pyramid responses through pymoten. Fallback mode returns the two-column gradient-motion proxy instead.",
        ("adelson1985",),
    ),
    "vision.optical_flow": FeatureGuide(
        "Mean horizontal flow, vertical flow, magnitude, and 95th-percentile magnitude from dense Farnebäck flow; fallback mode uses spatial-temporal gradients.",
        ("farneback",),
    ),
    "vision.semantic_views": FeatureGuide(
        "Frame-timed labels selected from a configurable scene vocabulary by CLIP zero-shot similarity. Fallback mode uses luminance, saturation, and edge heuristics.",
        ("clip",),
    ),
    "vision.social_proxies": FeatureGuide(
        "Three heuristic visual controls: luminance, a saturation-derived face-presence proxy, and a saturation-by-motion social-intensity proxy. These are not face or social-behavior detections.",
    ),
}


FAMILY_NOTES = {
    "audio": "Acoustic representations and audio preprocessing. Inputs are AudioStimulus objects or audio paths accepted by extract_features().",
    "events": "Operations on existing EventSeries objects.",
    "features": "Transforms on an existing FeatureSeries. Request the source feature first so the planner has a features token to route.",
    "image": "Still-image text extraction.",
    "language": "Word-aligned language controls and representations. Request text.tokenize first when starting from raw text.",
    "speech": "Speech activity, words, phones, speakers, affect, and learned audio representations.",
    "text": "Text-to-event preprocessing.",
    "video": "Video preprocessing and OCR.",
    "vision": "Framewise appearance, motion, semantic, and learned visual representations.",
}


def validate_feature_guide(entries: Iterable[FeatureCatalogEntry]) -> None:
    live = {entry.feature_id for entry in entries}
    described = set(FEATURE_GUIDE)
    missing = sorted(live - described)
    stale = sorted(described - live)
    if missing or stale:
        raise RuntimeError(
            "Feature catalog prose is out of sync with the public registry: "
            f"missing={missing}, stale={stale}"
        )
    unknown_refs = sorted(
        {
            ref
            for guide in FEATURE_GUIDE.values()
            for ref in guide.references
            if ref not in REFERENCES
        }
    )
    if unknown_refs:
        raise RuntimeError(f"Unknown feature reference keys: {unknown_refs}")


def reference_links(entry: FeatureCatalogEntry) -> str:
    refs = FEATURE_GUIDE[entry.feature_id].references
    if not refs:
        return "—"
    return "; ".join(f"[{REFERENCES[key][0]}]({REFERENCES[key][1]})" for key in refs)


def output_kind(entry: FeatureCatalogEntry) -> str:
    schemas = []
    for raw in entry.output_schema.split(","):
        name = raw.split("/", 1)[0]
        if name and name not in schemas and name != "dict":
            schemas.append(name)
    return " + ".join(schemas) or entry.output_schema


def access_label(entry: FeatureCatalogEntry) -> str:
    if entry.requires_opt_in:
        return f"opt-in · {entry.cost_class}"
    return f"default · {entry.cost_class}"


def markdown_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    entries = list(available_features(budget="all"))
    validate_feature_guide(entries)
    refs_used = {ref for guide in FEATURE_GUIDE.values() for ref in guide.references}
    print(
        f"Feature catalog guide: OK ({len(entries)} public IDs, "
        f"{len(refs_used)} primary reference groups)"
    )


if __name__ == "__main__":
    main()
