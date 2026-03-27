# Alignment Benchmarking

This project provides corpus-level ASR+alignment benchmarking via:

- Python API: `natural_features.features.speech.run_alignment_benchmark`
- CLI: `nf speech-benchmark`

## Manifest Format

```json
{
  "items": [
    {
      "id": "clip_001",
      "audio_path": "clips/clip_001.wav",
      "reference_ctm": "refs/clip_001.ctm",
      "transcript": "optional transcript text",
      "language": "en",
      "backend": "auto"
    }
  ]
}
```

Required per item:

- `audio_path`
- `reference_ctm` or `reference_textgrid`

Optional:

- `transcript` or `transcript_path`
- per-item overrides (`language`, `backend`, `asr_model`, `execution_mode`)

## Run

```bash
nf speech-benchmark \
  --manifest /path/to/manifest.json \
  --backend auto \
  --execution-mode fallback \
  --out-json benchmark_report.json \
  --json
```

Built-in tiny benchmark manifest:

```bash
nf speech-benchmark \
  --manifest tests/benchmarks/manifests/tier_a_alignment_manifest.json \
  --backend auto \
  --json
```

Generate a larger SNL subset manifest (local data required):

```bash
python scripts/build_snl_alignment_manifest.py --snl-root data/snl_2023_task --limit 24
nf speech-benchmark --manifest tests/benchmarks/generated/snl/manifest.json --backend auto --json
```

## Metrics

Per-clip metrics include:

- `onset_mae_ms`, `offset_mae_ms`, `boundary_mae_ms`
- `boundary_jitter_ms_p50`, `boundary_jitter_ms_p95`
- token match metrics (`token_precision`, `token_recall`, `token_f1`)
- `fallback_used`, `align_mode`

Aggregate summary includes:

- success/failure counts
- fallback rate
- alignment mode distribution
- MAE/jitter distribution summaries

Benchmark reports also include `runtime_pin_metadata` for reproducibility:

- pinned backend version recommendations
- pinned default model IDs
- installed runtime package versions

## Quality Gates

Evaluate a benchmark report against soft/hard thresholds:

```bash
python scripts/check_alignment_benchmark_gate.py \
  --report benchmark_report.json \
  --thresholds tests/benchmarks/thresholds/alignment_quality_gate.json
```

- Soft threshold violations emit warnings.
- Hard threshold violations return non-zero status and can be used as release gates.
