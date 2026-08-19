# Local naturalistic stimuli

Place large, non-redistributable recordings here. WAV and video files in this
directory are gitignored.

## Down the Rabbit Hole soundtrack

Expected filename:

```text
DownTheRabbitHoleFinal_mono_exp120_NR16_pad.wav
```

Copy from a local source, for example:

```bash
cp /path/to/DownTheRabbitHoleFinal_mono_exp120_NR16_pad.wav data/stimuli/
```

Override the path with `NF_RABBIT_HOLE_WAV` if the file lives elsewhere.

This file is used by:

- `examples/rms_energy_to_tr_grid.py`
- `tests/external/test_rabbit_hole_rms_tr.py` (skipped when the file is absent)

The committed tests and the cookbook page use a short synthetic clip with the
same TR and onset parameters so CI and the docs site do not depend on this
recording.
