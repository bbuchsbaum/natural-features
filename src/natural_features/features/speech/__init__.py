"""Speech extractors."""

from .align import alignment_qc, whisperx_align
from .asr import whisper_transcribe, whisper_transcribe_chunked
from .backends import probe_alignment_backends, resolve_aligner_backend
from .benchmark import (
    BenchmarkConfig,
    benchmark_alignment_case,
    match_token_pairs,
    run_alignment_benchmark,
)
from .chunking import aggregate_chunk_qc, plan_audio_chunks, stitch_word_events
from .contracts import normalize_alignment_qc, validate_alignment_qc
from .diarization import speaker_diarization
from .doctor import build_alignment_doctor_report, run_alignment_doctor
from .emotion import speech_emotion
from .formats import read_ctm, read_textgrid, write_ctm, write_textgrid
from .phonology import (
    CTCModelRuntime,
    acoustic_phone_posteriors,
    articulatory_features,
    articulatory_from_phoneme_events,
    articulatory_from_posteriors,
    clear_ctc_runtime,
    ctc_phone_posteriors,
    load_ctc_runtime,
    phoneme_event_series,
    phoneme_events_from_words,
    phoneme_posteriorgrams,
)
from .runtime_pins import runtime_pin_metadata, runtime_version_snapshot
from .ssl import hubert_hidden_states, wavlm_hidden_states
from .vad import energy_vad, neural_vad, speech_vad
from .validation import validate_alignment_backends

__all__ = [
    "BenchmarkConfig",
    "CTCModelRuntime",
    "acoustic_phone_posteriors",
    "aggregate_chunk_qc",
    "alignment_qc",
    "articulatory_features",
    "articulatory_from_phoneme_events",
    "articulatory_from_posteriors",
    "benchmark_alignment_case",
    "build_alignment_doctor_report",
    "clear_ctc_runtime",
    "ctc_phone_posteriors",
    "energy_vad",
    "hubert_hidden_states",
    "load_ctc_runtime",
    "match_token_pairs",
    "neural_vad",
    "normalize_alignment_qc",
    "phoneme_event_series",
    "phoneme_events_from_words",
    "phoneme_posteriorgrams",
    "plan_audio_chunks",
    "probe_alignment_backends",
    "read_ctm",
    "read_textgrid",
    "resolve_aligner_backend",
    "run_alignment_benchmark",
    "run_alignment_doctor",
    "runtime_pin_metadata",
    "runtime_version_snapshot",
    "speaker_diarization",
    "speech_emotion",
    "speech_vad",
    "stitch_word_events",
    "validate_alignment_backends",
    "validate_alignment_qc",
    "wavlm_hidden_states",
    "whisper_transcribe",
    "whisper_transcribe_chunked",
    "whisperx_align",
    "write_ctm",
    "write_textgrid",
]
