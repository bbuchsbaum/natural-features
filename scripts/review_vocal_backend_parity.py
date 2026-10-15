#!/usr/bin/env python3
"""Independent review reproduction: original upstream computations on syntheses.

Authored by the independent reviewer; parent added CLI paths and hash guards.
Requires the pinned optional model assets. This is not empirical qualification.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import types

import numpy as np
import torch
import torchaudio
import librosa
from torchaudio.transforms import Spectrogram

from natural_features.core.stimulus import AudioStimulus
from natural_features.features.audio.vocal_evidence import (
    audio_rmvpe, audio_vocal_activity, RMVPE_CODE_SHA256,
    RMVPE_WEIGHTS_SHA256, ACTIVITY_CODE_SHA256, ACTIVITY_WEIGHTS_SHA256,
    ACTIVITY_CONFIG_SHA256,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rmvpe-assets', type=Path, required=True)
    ap.add_argument('--activity-assets', type=Path, required=True)
    ap.add_argument('--out-json', type=Path, required=True)
    args = ap.parse_args()
    root, activity_root = args.rmvpe_assets, args.activity_assets
    for path, expected in [
        (root / 'rmvpe.py', RMVPE_CODE_SHA256), (root / 'rmvpe.pt', RMVPE_WEIGHTS_SHA256),
        (activity_root / 'svd/core/model.py', ACTIVITY_CODE_SHA256),
        (activity_root / 'dataset/jamendo/models/model_log_0mean/model.pt', ACTIVITY_WEIGHTS_SHA256),
        (activity_root / 'dataset/jamendo/models/model_log_0mean/model.vars', ACTIVITY_CONFIG_SHA256),
    ]:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    torch.set_num_threads(1)
    infer, lib = types.ModuleType('infer'), types.ModuleType('infer.lib')
    lib.jit = types.ModuleType('infer.lib.jit')
    infer.lib = lib
    sys.modules.update({'infer': infer, 'infer.lib': lib, 'infer.lib.jit': lib.jit})
    rvc = {'__name__': 'independent_rvc'}
    exec(compile((root / 'rmvpe.py').read_text(), str(root / 'rmvpe.py'), 'exec'), rvc)
    rm = rvc['RMVPE'](str(root / 'rmvpe.pt'), is_half=False, device='cpu', use_jit=False)
    torchaudio.set_audio_backend = lambda name: None
    module = {'__name__': 'independent_cpjku'}
    exec(compile((activity_root / 'svd/core/model.py').read_text(), str(activity_root / 'svd/core/model.py'), 'exec'), module)
    config = {}
    for line in (activity_root / 'dataset/jamendo/models/model_log_0mean/model.vars').read_text().splitlines():
        key, value = map(str.strip, line.split('=', 1))
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
        config[key] = value
    namespace = {'torch': torch, 'librosa': librosa, 'np': np, 'Spectrogram': Spectrogram,
                 'warnings': __import__('warnings'), 'progress': lambda data, **kw: data}
    upstream_functions = {}
    for relative, name in [('svd/train.py', 'prep_net_config'), ('svd/predict.py', 'do_prediction'), ('svd/core/audio.py', 'load_spec')]:
        path = activity_root / relative
        upstream_functions[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        function = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    net = module['SingingVoiceDetector'](namespace['prep_net_config'](config, 372))
    net.load_state_dict(torch.load(activity_root / 'dataset/jamendo/models/model_log_0mean/model.pt', map_location='cpu', weights_only=True), strict=True)
    net.eval()
    receipt = []
    for sr, kind in [(16000, 'rmvpe'), (22050, 'activity')]:
        for silence in (False, True):
            n = int(sr * 1.2) + 1
            times = np.arange(n) / sr
            audio = (.11 * np.sin(2 * np.pi * (180 * times + 90 * times * times)) + .025 * np.sin(2 * np.pi * 660 * times)).astype(np.float32)
            audio[(times > .4) & (times < .6)] = 0
            if silence:
                audio[:] = 0
            stim = AudioStimulus.from_array(audio, sr, start_offset_s=3.25)
            if kind == 'rmvpe':
                with torch.inference_mode():
                    salience = rm.mel2hidden(rm.mel_extractor(torch.from_numpy(audio)[None], center=True))[0].numpy()
                raw = rm.decode(salience, thred=0)
                official = rm.infer_from_audio(audio, thred=.03)
                out = audio_rmvpe(stim, source_root=str(root))
                names = list(out.coords['feature'])
                assert np.array_equal(out.values[:, names.index('candidate_f0_hz')], raw, equal_nan=True)
                assert np.array_equal(out.values[:, names.index('confidence')], salience.max(1))
                valid = (np.arange(len(raw)) * .01 < n / sr) & (raw >= 50) & (raw <= 1000) & (salience.max(1) > .03)
                assert np.array_equal(out.values[:, names.index('voiced')].astype(bool), valid)
                assert np.array_equal(out.values[:, 0], np.where(valid, official, np.nan), equal_nan=True)
                hop = 160
            else:
                original_load = librosa.load
                librosa.load = lambda path, sr: (audio, sr)
                try:
                    spec = namespace['load_spec'](0, [], [activity_root / 'svd/core/model.py'], sr, Spectrogram(n_fft=1024, hop_length=315, power=1.))
                finally:
                    librosa.load = original_load
                direct = namespace['do_prediction']([(spec[None, None], None)], net, 'cpu', config)[0].numpy()
                out = audio_vocal_activity(stim, source_root=str(activity_root))
                assert np.array_equal(out.values[:, 0], direct)
                hop = 315
            np.testing.assert_allclose(out.times_s, 3.25 + np.arange(len(out.values)) * hop / sr, atol=1e-15, rtol=0)
            receipt.append({'kind': kind, 'silence': silence, 'samples': n, 'frames': len(out.values),
                'input_sha256': hashlib.sha256(audio.tobytes()).hexdigest(), 'parity': 'bitwise_exact'})
    source = Path(sys.modules[audio_rmvpe.__module__].__file__)
    result = {'checks': receipt, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'review_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'upstream_function_source_sha256': upstream_functions,
        'versions': {'torch': torch.__version__, 'torchaudio': torchaudio.__version__, 'numpy': np.__version__, 'librosa': librosa.__version__},
        'limits': 'Synthetic native-rate parity only; no historical/off-rate environment, calibration or empirical accuracy claim.'}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
