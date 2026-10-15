#!/usr/bin/env python3
"""Frozen v6 adaptive development grid: local veto plus gap-aware support."""
from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import product
import json
from pathlib import Path

import numpy as np

from calibrate_vocal_f0 import CONDS, ROOT, _runner, _score, _summary, replay
from evaluate_vocal_f0_local import _metadata, _save
from natural_features.features.common import extractor_metadata

PROTOCOL_SHA256 = '53b78c2009c7e214449bae9b7cc3561b3d58e133340ef330ea02317afd508b67'


def select(profiles):
    eligible = [p for p in profiles if p['eligible']]
    return None if not eligible else max(eligible, key=lambda p: (
        p['objective']['minimum_pitch_accuracy'], -p['objective']['maximum_false_voicing'],
        p['parameters']['local_periodicity_threshold'], -p['parameters']['max_internal_gap_s'],
        -p['parameters']['voicing_guard_s']))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', type=Path, default=ROOT / 'tests/benchmarks/thresholds/vocal_f0_development_v6.json')
    ap.add_argument('--assets', type=Path, required=True)
    ap.add_argument('--artifact-dir', type=Path, required=True)
    ap.add_argument('--out-json', type=Path, required=True)
    args = ap.parse_args()
    runner = _runner()
    if runner._digest(args.protocol) != PROTOCOL_SHA256:
        raise ValueError('protocol differs from frozen v6 experiment')
    protocol = json.loads(args.protocol.read_text())
    input_path = ROOT / protocol['input_report']
    if runner._digest(input_path) != protocol['input_report_sha256']:
        raise ValueError('v4 diagnostic report identity mismatch')
    raw_report = json.loads(input_path.read_text())
    report = {'schema': 'VocalF0HybridDevelopment/v6', 'qualified': False, 'prospective_test_consumed': False,
              'protocol_sha256': PROTOCOL_SHA256, 'input_report_sha256': runner._digest(input_path),
              'extractor_source_sha256': runner._digest(ROOT / 'src/natural_features/features/audio/vocal_f0.py'),
              'replay_source_sha256': runner._digest(ROOT / 'scripts/calibrate_vocal_f0.py'),
              'calibration_source_sha256': runner._digest(Path(__file__)),
              'profiles': [], 'selected': None, 'development_passed': False, 'errors': []}
    try:
        loaded, analytics = {}, {}
        expected = {(g['name'], i, c) for g in protocol['development_sets'] for i in g['excerpt_ids'] for c in CONDS}
        if raw_report['errors'] or len(raw_report['raw_diagnostics']) != 60:
            raise ValueError('incomplete v4 diagnostics')
        for g in protocol['development_sets']:
            path = ROOT / g['report']
            if runner._digest(path) != g['report_sha256']:
                raise ValueError('consumed report identity mismatch')
            analytics[g['name']] = json.loads(path.read_text())['analytic']
        for row in raw_report['raw_diagnostics']:
            key = (row['group'], row['excerpt_id'], row['condition'])
            a = row['artifact']
            if runner._digest(Path(a['artifact'])) != a['artifact_sha256']:
                raise ValueError('diagnostic artifact hash mismatch')
            annotation = runner._find_one(args.assets, f"vocadito_{row['excerpt_id']}_f0.csv")
            if runner._digest(annotation) != row['input_hashes']['reference_sha256']:
                raise ValueError('reference identity mismatch')
            fs = runner._load_feature(Path(a['artifact']), a['cache_key'])
            if fs is None:
                raise ValueError('invalid diagnostic artifact')
            np.testing.assert_array_equal(replay(fs, .65, .03, .01).values, fs.values)
            loaded[key] = fs
        if set(loaded) != expected:
            raise ValueError('wrong groups or incomplete jobs')
        grid = protocol['calibration_grid']
        for threshold, guard, gap in product(grid['local_periodicity_threshold'], grid['voicing_guard_s'], grid['max_internal_gap_s']):
            params = dict(local_periodicity_threshold=threshold, voicing_guard_s=guard, max_internal_gap_s=gap)
            groups = []
            for g in protocol['development_sets']:
                rows = []
                for i in g['excerpt_ids']:
                    for c in CONDS:
                        fs = loaded[(g['name'], i, c)]
                        out = replay(fs, .65, .03, .01, **params)
                        if np.any((out.values[:, 4] > 0) & (fs.values[:, 4] == 0)):
                            raise ValueError('hybrid gate added accepted frames')
                        rows.append({'excerpt_id': i, 'condition': c, 'metrics': _score(runner, args.assets, i, c, out)})
                summary, passed, failures = _summary(runner, rows, protocol['gates'], analytics[g['name']])
                groups.append({'name': g['name'], 'summary': summary, 'gate': {'passed': passed, 'failures': failures}, 'per_excerpt': rows})
            positive = [g['summary'][c] for g in groups for c in ('solo', 'mixture_0db', 'mixture_minus6db')]
            report['profiles'].append({'parameters': params, 'eligible': all(g['gate']['passed'] for g in groups),
                                      'objective': {'minimum_pitch_accuracy': min(r['raw_pitch_accuracy'] for r in positive),
                                                    'maximum_false_voicing': max(r['voicing_false_alarm_rate'] for r in positive)},
                                      'development_sets': groups})
        chosen = select(report['profiles'])
        report['selected'] = chosen
        report['development_passed'] = chosen is not None
        if chosen is not None:
            report['selected_artifacts'] = []
            for key, fs in loaded.items():
                params = chosen['parameters']
                out = replay(fs, .65, .03, .01, **params)
                full_params = dict(hop_s=.01, fmin=50., fmax=1000., confidence_threshold=.65,
                                   source_threshold=.03, min_source_rms=.01, **params)
                md = {**_metadata(fs, params['local_periodicity_threshold'], fs.metadata['replay_provenance']),
                      **extractor_metadata('audio.vocal_f0', full_params, code_version='vocal-f0-v6-replay',
                                           model_revision=fs.metadata['model_revision']),
                      **params, 'voicing_rule': protocol['method']['temporal_envelope']}
                out = replace(out, metadata=md)
                identity = {k: v for k, v in report.items() if k.endswith('_sha256')}
                identity.update(group=key[0], excerpt_id=key[1], condition=key[2], parameters=params)
                saved = _save(runner, args.artifact_dir, out, identity)
                report['selected_artifacts'].append({'group': key[0], 'excerpt_id': key[1], 'condition': key[2], 'artifact': saved})
    except Exception as exc:
        report['errors'].append(runner._error_detail(exc))
        report['development_passed'] = False
    runner._atomic_json(args.out_json, report)
    print(json.dumps({'passed': report['development_passed'], 'selected': None if report['selected'] is None else report['selected']['parameters'], 'errors': report['errors']}, indent=2))
    for profile in report['profiles']:
        print(profile['parameters'], 'eligible=', profile['eligible'], profile['objective'])
    return 0 if report['development_passed'] and not report['errors'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
