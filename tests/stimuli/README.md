# Tier A Stimuli

`tests/stimuli/tier_a/` contains tiny, deterministic synthetic fixtures used by CI and local tests.

Goals:

- Stable diagnostics for scene-cut detection, motion, energy/VAD, and transcript alignment.
- No redistribution/licensing risk (all files are generated).
- Fast execution (< 1s load time per file on typical dev machines).

## Files

- `video_scene_cut.npy` (6s, 10 fps, 64x64x3)
- `audio_speechlike.wav` (6s, 16kHz mono)
- `transcript_reference.txt`
- `reference_words.ctm` (deterministic word-timing reference for alignment benchmarks)
- `manifest.json` (hashes + expected diagnostic windows)

## Regeneration

```bash
python scripts/generate_tier_a_stimuli.py
python scripts/validate_tier_a_stimuli.py
```

## Tier B (on-demand)

Tier B references are declared in:

- `tests/stimuli/tier_b/manifest.json`

Acquire and prepare them with:

```bash
python scripts/fetch_tier_b_stimuli.py --allow-missing-sha
python scripts/prepare_tier_b_clips.py
```
