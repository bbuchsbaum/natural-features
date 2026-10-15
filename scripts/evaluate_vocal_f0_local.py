#!/usr/bin/env python3
"""Evaluate the frozen v4 local-period agreement grid on consumed data only."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from calibrate_vocal_f0 import CONDS, ROOT, _duration, _runner, _score, _summary, replay
from natural_features.features.audio.vocal_f0 import FEATURES, _frame_rms, _local_periodicity
from natural_features.features.common import extractor_metadata

PROTOCOL_SHA256 = '5ed1b041fdaf161fcd34074a3354319b10e707eb949b84522cf3df9668e5bc40'


def _metadata(feature, threshold, provenance):
    params = {'hop_s': .01, 'fmin': 50., 'fmax': 1000., 'confidence_threshold': .65,
              'source_threshold': .03, 'min_source_rms': .01, 'voicing_guard_s': 0.,
              'local_periodicity_threshold': threshold}
    return {**feature.metadata,
            **extractor_metadata('audio.vocal_f0', params, code_version='vocal-f0-v4-replay',
                                 model_revision=feature.metadata['model_revision']),
            'local_periodicity_threshold': threshold, 'voicing_guard_s': 0.,
            'local_periodicity_semantics': '2*C/max(A+B,2*min_source_rms**2); separate arm means removed; fixed 320 midpoints',
            'replay_provenance': provenance, 'qualification': 'consumed data; development only'}


def _save(runner, folder, feature, identity):
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = folder.resolve() / (key + '.npz')
    runner._save_feature(path, key, feature)
    return {'cache_key': key, 'artifact': str(path), 'artifact_sha256': runner._digest(path)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', type=Path, default=ROOT / 'tests/benchmarks/thresholds/vocal_f0_development_v4.json')
    ap.add_argument('--assets', type=Path, required=True)
    ap.add_argument('--stem-manifest', type=Path, required=True)
    ap.add_argument('--artifact-dir', type=Path, required=True)
    ap.add_argument('--out-json', type=Path, required=True)
    args = ap.parse_args()
    runner = _runner()
    if runner._digest(args.protocol) != PROTOCOL_SHA256:
        raise ValueError('protocol differs from the frozen v4 experiment')
    protocol = json.loads(args.protocol.read_text())
    stems = json.loads(args.stem_manifest.read_text())
    if stems['errors'] or len(stems['rows']) != 60:
        raise ValueError('stem collection must contain all60 consumed jobs without errors')
    stem_rows = {(r['group'], r['excerpt_id'], r['condition']): r for r in stems['rows']}
    report = {'schema': 'VocalF0LocalDevelopment/v4', 'qualified': False, 'prospective_test_consumed': False,
              'protocol_sha256': PROTOCOL_SHA256, 'stem_manifest_sha256': runner._digest(args.stem_manifest),
              'stem_manifest': str(args.stem_manifest.resolve()),
              'extractor_source_sha256': runner._digest(ROOT / 'src/natural_features/features/audio/vocal_f0.py'),
              'replay_source_sha256': runner._digest(ROOT / 'scripts/calibrate_vocal_f0.py'),
              'evaluation_source_sha256': runner._digest(Path(__file__)),
              'scorer_source_sha256': runner._digest(ROOT / 'src/natural_features/features/audio/vocal_f0_benchmark.py'),
              'raw_diagnostics': [], 'profiles': [], 'selected_threshold': None, 'development_passed': False,
              'errors': []}
    hashes = {k: v for k, v in report.items() if k.endswith('_sha256')}
    loaded, originals = {}, {}
    try:
        for group in protocol['development_sets']:
            old_path = ROOT / group['report']
            if runner._digest(old_path) != group['report_sha256']:
                raise ValueError('consumed report hash mismatch')
            old = json.loads(old_path.read_text())
            if old['errors'] or len(old['per_excerpt']) != 30 or {(r['excerpt_id'], r['condition']) for r in old['per_excerpt']} != {(i,c) for i in group['excerpt_ids'] for c in CONDS}:
                raise ValueError('consumed report is incomplete')
            originals[group['name']] = old
            for row in old['per_excerpt']:
                key = (group['name'], row['excerpt_id'], row['condition'])
                sr = stem_rows[key]
                if sr['original_artifact'] != row['artifact'] or not sr['legacy_diagnostics_exact']:
                    raise ValueError('stem provenance mismatch')
                artifact = row['artifact']
                path = Path(artifact['artifact'])
                if runner._digest(path) != artifact['artifact_sha256'] or runner._digest(Path(sr['stem_path'])) != sr['stem_sha256']:
                    raise ValueError('native artifact or stem hash mismatch')
                annotation = runner._find_one(args.assets, f"vocadito_{row['excerpt_id']}_f0.csv")
                if runner._digest(annotation) != row['input_hashes']['reference_sha256']:
                    raise ValueError('reference identity mismatch')
                raw = runner._load_feature(path, artifact['cache_key'])
                if raw is None:
                    raise ValueError('invalid raw artifact')
                params = old['extractor_parameters']
                restored = replay(raw, params['confidence_threshold'], params['source_threshold'], params['min_source_rms'])
                np.testing.assert_array_equal(restored.values[:, :11], raw.values[:, :11])
                if _score(runner, args.assets, row['excerpt_id'], row['condition'], restored) != row['metrics']:
                    raise ValueError('original scores no longer reproduce')
                with np.load(sr['stem_path'], allow_pickle=False) as z:
                    if z['input_hash'].item() != row['input_hashes']['rendered_16khz_mono_sha256'] or z['source_hash'].item() != stems['extractor_source_sha256']:
                        raise ValueError('cached stem input/source identity mismatch')
                    channels = json.loads(z['sources_json'].item())
                    wave = z['stems'][:, channels.index('vocals')]
                centers = np.arange(len(raw.values)) * 160
                np.testing.assert_array_equal(_frame_rms(wave, centers), raw.values[:, 10])
                local = _local_periodicity(wave, centers, raw.values[:, 9], .01, _duration(raw))
                base = replay(raw, .65, .03, .01)
                enriched = replace(base, values=np.column_stack([base.values, local]), coords={'feature': FEATURES},
                                   metadata=_metadata(base, None, sr))
                np.testing.assert_array_equal(replay(enriched, .65, .03, .01).values, enriched.values)
                identity = {**hashes, 'original_artifact': artifact, 'stem_sha256': sr['stem_sha256'],
                            'group': key[0], 'excerpt_id': key[1], 'condition': key[2], 'threshold': None}
                saved = _save(runner, args.artifact_dir, enriched, identity)
                report['raw_diagnostics'].append({'group': key[0], **row, 'original_artifact': artifact,
                                                  'artifact': saved, 'stem_provenance': sr})
                loaded[key] = enriched
        for threshold in protocol['calibration_grid']['local_periodicity_threshold']:
            groups = []
            for group in protocol['development_sets']:
                rows = []
                for row in originals[group['name']]['per_excerpt']:
                    key = (group['name'], row['excerpt_id'], row['condition'])
                    before = loaded[key]
                    after = replay(before, .65, .03, .01, local_periodicity_threshold=threshold)
                    if np.any((after.values[:, 4] > 0) & (before.values[:, 4] == 0)):
                        raise ValueError('local veto added accepted frames')
                    rows.append({'excerpt_id': key[1], 'condition': key[2],
                                 'metrics': _score(runner, args.assets, key[1], key[2], after)})
                summary, passed, failures = _summary(runner, rows, protocol['gates'], originals[group['name']]['analytic'])
                groups.append({'name': group['name'], 'summary': summary, 'gate': {'passed': passed, 'failures': failures},
                               'per_excerpt': rows, 'analytic_provenance': 'unchanged pitch engine; controls retained in consumed report'})
            eligible = all(g['gate']['passed'] for g in groups)
            report['profiles'].append({'threshold': threshold, 'eligible': eligible, 'development_sets': groups})
            if eligible and report['selected_threshold'] is None:
                report['selected_threshold'] = threshold
            runner._atomic_json(args.out_json, report)
        report['development_passed'] = report['selected_threshold'] is not None
        if report['development_passed']:
            report['selected_artifacts'] = []
            for key, before in loaded.items():
                threshold = report['selected_threshold']
                after = replay(before, .65, .03, .01, local_periodicity_threshold=threshold)
                after = replace(after, metadata=_metadata(before, threshold, stem_rows[key]))
                identity = {**hashes, 'group': key[0], 'excerpt_id': key[1], 'condition': key[2], 'threshold': threshold}
                saved = _save(runner, args.artifact_dir, after, identity)
                report['selected_artifacts'].append({'group': key[0], 'excerpt_id': key[1], 'condition': key[2], 'artifact': saved})
    except Exception as exc:
        report['errors'].append(runner._error_detail(exc))
        report['development_passed'] = False
    runner._atomic_json(args.out_json, report)
    print(json.dumps({k: report[k] for k in ('selected_threshold', 'development_passed', 'qualified', 'errors')}, indent=2))
    for profile in report['profiles']:
        print('threshold', profile['threshold'], 'eligible', profile['eligible'])
        for g in profile['development_sets']:
            print(g['name'], json.dumps({'summary': g['summary'], 'gate': g['gate']}))
    return 0 if report['development_passed'] and not report['errors'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
