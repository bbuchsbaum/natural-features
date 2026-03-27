from __future__ import annotations

import numpy as np
import pytest

from natural_features.core.stimulus import AudioStimulus, VideoStimulus
from natural_features.features.audio.lowlevel import rms
from natural_features.features.speech.vad import energy_vad
from natural_features.features.vision.lowlevel import visual_energy
from natural_features.features.vision.scene import scene_cuts
from natural_features.fmri.design import add_lags, concat_feature_series
from natural_features.fmri.hrf import hrf_convolve
from natural_features.fmri.render import render_events
from natural_features.fmri.resample import resample_feature_series


pytestmark = [pytest.mark.media, pytest.mark.smoke]


def test_tiny_clip_end_to_end_design_matrix() -> None:
    rng = np.random.default_rng(3)
    fps = 10.0
    frames = (rng.uniform(0, 255, size=(40, 24, 24, 3))).astype(np.uint8)
    frames[20:] = np.clip(frames[20:] + 40, 0, 255)
    video = VideoStimulus.from_array(frames, fps=fps)

    sr = 8000
    t = np.arange(int(sr * 4.0), dtype=np.float32) / sr
    audio = AudioStimulus.from_array((0.2 * np.sin(2 * np.pi * 180 * t)).astype(np.float32), sr_hz=sr)

    vis = visual_energy(video)
    cuts = scene_cuts(video)
    aud_rms = rms(audio, hop_s=0.02, win_s=0.03)
    vad = energy_vad(audio, hop_s=0.02, win_s=0.03)

    tr_s = 1.0
    vis_tr = resample_feature_series(vis, tr_s=tr_s, method="mean")
    rms_tr = resample_feature_series(aud_rms, tr_s=tr_s, method="mean", time_grid_s=vis_tr.times_s)
    vad_tr = resample_feature_series(vad, tr_s=tr_s, method="mean", time_grid_s=vis_tr.times_s)
    cuts_tr = render_events(cuts, vis_tr.times_s, mode="impulse", value="count")

    x = concat_feature_series([vis_tr, rms_tr, vad_tr, cuts_tr], standardize=True, add_intercept=True)
    x = hrf_convolve(x, tr_s=tr_s, kind="glover")
    x = add_lags(x, [0, 1, 2])

    assert x.values.shape[0] == len(vis_tr.times_s)
    assert x.values.shape[1] > 10
