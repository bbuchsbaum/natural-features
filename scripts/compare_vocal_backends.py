#!/usr/bin/env python3
"""Frozen, development-only comparison of canonical F0, RMVPE and activity evidence."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from natural_features.core.feature_types import FeatureSeries  # noqa: E402
from natural_features.core.stimulus import AudioStimulus  # noqa: E402
from natural_features.core.timebase import TimebaseSpec, TemporalContext  # noqa: E402
from natural_features.features.audio import vocal_f0 as canonical  # noqa: E402
from natural_features.features.audio.vocal_evidence import audio_rmvpe, audio_vocal_activity, pitch_with_activity  # noqa: E402
from natural_features.features.audio.vocal_activity_benchmark import score_vocal_activity  # noqa: E402
from natural_features.features.audio.vocal_f0_benchmark import score_vocal_f0  # noqa: E402
from calibrate_vocal_f0 import _runner, replay  # noqa: E402
from calibrate_vocal_voicing import load_records, summarize  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render_record(record, assets, groups, runner):
    source_path = runner._find_one(assets, f'vocadito_{record.excerpt}.wav')
    annotation = runner._find_one(assets, f'vocadito_{record.excerpt}_f0.csv')
    source, sr = runner._read_audio(source_path)
    source = runner._to_16k_mono(source, sr, 10.0)
    reference = np.loadtxt(annotation, delimiter=',')
    reference = reference[reference[:, 0] < len(source) / 16000]
    backing_name = 'vibe' if record.excerpt % 2 else 'sugar'
    backing_path = assets / f'{backing_name}.ogg'
    expected = record.provenance['input_hashes']
    for path, key in [(source_path, 'source_audio_sha256'), (annotation, 'reference_sha256'), (backing_path, 'backing_sha256')]:
        if digest(path) != expected[key]:
            raise ValueError('input identity mismatch: ' + str(path))
    start = groups[record.group]['backing_start_s']
    backing, sr = runner._read_audio(backing_path)
    backing = runner._to_16k_mono(backing, sr, start + len(source) / 16000)[round(start * 16000):][:len(source)]
    if record.condition == 'solo':
        audio = source
    elif record.condition == 'silence':
        audio = np.zeros_like(source)
    elif record.condition == 'backing':
        audio = backing
    else:
        mask = runner._reference_mask(reference[:, 0], reference[:, 1], len(source))
        audio, _ = runner._mixture(source, backing, mask, 0.0 if record.condition == 'mixture_0db' else -6.0)
    if hashlib.sha256(np.ascontiguousarray(audio).tobytes()).hexdigest() != record.audio_hash:
        raise ValueError('rendered audio differs from preserved reference input')
    return audio


def canonical_and_stem(stimulus, checkpoint):
    """Observe the canonical separation call, without changing its computation."""
    runtime = canonical._load_runtime(checkpoint, None, True, 'cpu')
    original = runtime.separate
    captured = {}
    def capture(stereo):
        stems = original(stereo)
        captured['stems'] = stems
        return stems
    with patch.object(runtime, 'separate', capture):
        feature = canonical.audio_vocal_f0(stimulus, separator_checkpoint=checkpoint, device='cpu')
    stems = canonical._resample(captured['stems'], 44100, 16000)[:len(stimulus.samples)]
    stem = stems[:, list(runtime.sources).index('vocals')]
    return feature, stem


def load_saved(path, key):
    with np.load(path, allow_pickle=False) as a:
        if str(a['cache_key'].item()) != key:
            raise ValueError('cache identity mismatch: ' + str(path))
        return FeatureSeries(values=a['values'], times_s=a['times_s'], time_bounds_s=a['bounds'],
            coords={'feature': json.loads(str(a['feature_names_json'].item()))},
            metadata=json.loads(str(a['source_metadata_json'].item())),
            timebase=TimebaseSpec.from_dict(json.loads(str(a['timebase_json'].item())), default_kind='frames'),
            temporal_context=TemporalContext.from_dict(json.loads(str(a['temporal_context_json'].item()))))


def verify_protocol(protocol):
    if protocol['state'] != 'development' or protocol['qualification_allowed'] is not False:
        raise ValueError('this comparison permits consumed-data development only')
    if protocol['rmvpe_threshold'] != .03 or protocol['activity_threshold'] != .51:
        raise ValueError('fixed upstream diagnostic thresholds changed')
    for name, expected in protocol['frozen_sources'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('frozen source mismatch: ' + name)
    inherited = ROOT / protocol['consumed_protocol']['path']
    if digest(inherited) != protocol['consumed_protocol']['sha256']:
        raise ValueError('consumed input protocol changed')
    return json.loads(inherited.read_text())


def baseline_parity(old, current):
    expected = replay(old, .65, .03, .01)
    old_names, names = list(expected.coords['feature']), list(current.coords['feature'])
    np.testing.assert_array_equal(expected.times_s, current.times_s)
    for name in ('candidate_f0_hz', 'confidence', 'source_support', 'source_rms'):
        np.testing.assert_allclose(expected.values[:, old_names.index(name)], current.values[:, names.index(name)],
                                   rtol=2e-5, atol=1e-6, equal_nan=True)
    np.testing.assert_array_equal(expected.values[:, old_names.index('voiced')], current.values[:, names.index('voiced')])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', type=Path, default=ROOT / 'tests/benchmarks/thresholds/vocal_backend_comparison_v1.json')
    ap.add_argument('--assets', type=Path, required=True)
    ap.add_argument('--rmvpe-assets', type=Path, required=True)
    ap.add_argument('--activity-assets', type=Path, required=True)
    ap.add_argument('--separator-checkpoint', type=Path, required=True)
    ap.add_argument('--artifact-dir', type=Path, required=True)
    ap.add_argument('--out-json', type=Path, required=True)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(1)
    protocol = json.loads(args.protocol.read_text())
    inherited = verify_protocol(protocol)
    runner = _runner()
    records, analytic = load_records(inherited, args.assets, runner)
    protocol_hash = digest(args.protocol)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    report = {'schema': 'VocalBackendComparison/v1', 'state': 'development', 'qualified': False,
              'protocol_sha256': protocol_hash, 'protocol': protocol, 'errors': [], 'rows': [],
              'canonical_analytic': analytic, 'fresh_data_consumed': False,
              'activity_reference_limits': 'Known nonvocal controls and pitched singing positives only; F0-unvoiced frames remain unknown for vocal presence.',
              'torch_threads': 1}
    metrics_by_variant = {}
    for record in records:
        try:
            audio = render_record(record, args.assets, inherited['groups'], runner)
            stimulus = AudioStimulus.from_array(audio, 16000)
            key = hashlib.sha256((protocol_hash + record.key + record.audio_hash).encode()).hexdigest()
            folder = args.artifact_dir / record.key.replace(':', '_')
            folder.mkdir(exist_ok=True)
            stem_path, base_path = folder / 'stem.npy', folder / 'canonical.npz'
            if base_path.exists() and stem_path.exists():
                baseline = load_saved(base_path, key + ':canonical')
                stem = np.load(stem_path, allow_pickle=False)
                receipt = json.loads((folder / 'stem.json').read_text())
                if digest(stem_path) != receipt['sha256'] or receipt['key'] != key:
                    raise ValueError('stem cache identity mismatch')
            else:
                baseline, stem = canonical_and_stem(stimulus, str(args.separator_checkpoint))
                np.save(stem_path, stem, allow_pickle=False)
                (folder / 'stem.json').write_text(json.dumps({'key': key, 'sha256': digest(stem_path)}))
                runner._save_feature(base_path, key + ':canonical', baseline)
            baseline_parity(record.feature, baseline)
            features = {'canonical': baseline}
            activities = {}
            for source, waveform in [('mixture', audio), ('stem', stem)]:
                current = AudioStimulus.from_array(waveform, 16000)
                for kind in ('rmvpe', 'activity'):
                    name = kind + '_' + source
                    path = folder / (name + '.npz')
                    if path.exists():
                        feature = load_saved(path, key + ':' + name)
                    else:
                        feature = (audio_rmvpe(current, source_root=str(args.rmvpe_assets), threshold=protocol['rmvpe_threshold'])
                                   if kind == 'rmvpe' else audio_vocal_activity(current, source_root=str(args.activity_assets)))
                        if source == 'stem':
                            feature = replace(feature, metadata={**feature.metadata,
                                'input_source': 'canonical Demucs selected vocal stem',
                                'upstream_separation': baseline.metadata['computation_context'],
                                'upstream_model_revision': baseline.metadata['model_revision'],
                                'stem_sha256': digest(stem_path)})
                        runner._save_feature(path, key + ':' + name, feature)
                    (features if kind == 'rmvpe' else activities)[name] = feature
            unmasked = dict(features)
            for name, pitch in unmasked.items():
                for aname, activity in activities.items():
                    features[name + '__' + aname] = pitch_with_activity(pitch, activity, threshold=protocol['activity_threshold'])
            artifact_rows = {}
            for name, feature in {**features, **activities}.items():
                path = folder / (name + '.npz')
                if '__' in name:
                    runner._save_feature(path, key + ':' + name, feature)
                artifact_rows[name] = {'path': str(path), 'sha256': digest(path), 'key': key + ':' + name}
            scores = {name: score_vocal_f0(record.reference_times, record.reference_hz, feature) for name, feature in features.items()}
            for name, result in scores.items():
                metrics_by_variant.setdefault(name, {})[record.key] = result
            activity_scores = {}
            for name, feature in activities.items():
                if record.condition in ('backing', 'silence'):
                    times = feature.times_s[feature.times_s < len(audio) / 16000]
                    truth = np.zeros(len(times))
                else:
                    times = record.reference_times
                    truth = np.where(record.reference_hz > 0, 1., np.nan)
                activity_scores[name] = score_vocal_activity(times, truth, feature, threshold=protocol['activity_threshold'])
            report['rows'].append({'key': record.key, 'group': record.group, 'singer': record.singer,
                'excerpt': record.excerpt, 'condition': record.condition, 'audio_sha256': record.audio_hash,
                'baseline_replay_verified': True, 'metrics': scores, 'activity_partial_reference': activity_scores,
                'artifacts': artifact_rows, 'provenance': record.provenance})
            print(f'{len(report["rows"])}/90 {record.key}', flush=True)
        except Exception as exc:
            report['errors'].append({'key': record.key, 'type': type(exc).__name__, 'message': str(exc)})
            print(f'ERROR {record.key}: {exc}', flush=True)
            runner._atomic_json(args.out_json, report)
            return 2
        runner._atomic_json(args.out_json, report)
    # Reuse the original six analytic stimuli and original interior-median rule.
    # These pitch-engine controls do not label tones as singing activity.
    def rmvpe_analytic(audio, unused):
        return audio_rmvpe(AudioStimulus.from_array(audio, 16000), source_root=str(args.rmvpe_assets), threshold=.03)
    with patch.object(runner, '_analytic_feature', rmvpe_analytic):
        report['rmvpe_analytic'] = runner._analytic_cases(args)
    report['variants'] = {}
    for name, metrics in metrics_by_variant.items():
        groups = summarize(records, metrics, inherited['gates'], complete=True)
        control = (report['rmvpe_analytic'] if name.startswith('rmvpe') else next(iter(analytic.values()))['result'])
        analytic_pass = bool(control['complete'] and control['worst_median_abs_cents_error'] <= inherited['gates']['analytic_median_abs_cents_error_max'])
        report['variants'][name] = {'groups': groups, 'analytic_passed': analytic_pass,
            'development_gates_passed': all(g['gate']['passed'] for g in groups) and analytic_pass}
    report['completed_cases'] = len(report['rows'])
    report['adopted_model'] = None
    report['decision'] = 'No automatic adoption or fresh qualification; preserve every fixed arm for independent review.'
    runner._atomic_json(args.out_json, report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
