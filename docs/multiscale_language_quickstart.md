# Multiscale Language Quickstart

`extract_multiscale_language` generates semantic language features at multiple
time scales from transcript text, word events, or audio (via ASR fallback).

```python
from natural_features.workflows import extract_multiscale_language

result = extract_multiscale_language(
    "data/speech.wav",
    scales_s=[2.0, 4.0, 16.0],
    feature_families=[
        "sentence_embeddings",
        "paragraph_embeddings",
        "surprisal",
        "lexical_controls",
    ],
    provider_config={
        "provider": "openai",
        "model": "text-embedding-3-large",
        "api_key_env_var": "OPENAI_API_KEY",
        "batch_size": 128,
    },
    aggregation="mean",
    window_policy="centered",
    as_dataframe=True,
)

X2 = result.by_scale[2.0].values
X4 = result.by_scale[4.0].values
X16 = result.by_scale[16.0].values
```

Notes:

- For local deterministic testing without API keys:
  - `provider_config={"provider": "local_bow", "dim": 1024}`
  - `provider_config={"provider": "local_hash", "dim": 256}` (legacy lightweight fallback)
- API-key and provider enforcement is strict by default. Set
  `execution_mode="fallback"` only to request the documented local substitute.
- Outputs are `FeatureSeries` objects keyed by scale.
- `result.qc` includes cache hit/miss metrics (`cache_unique_misses`) and unit counts.
