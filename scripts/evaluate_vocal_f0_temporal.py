#!/usr/bin/env python3
"""Evaluate the single frozen v3 temporal hypothesis on consumed data only.

This reuses verified raw v2 diagnostics; it does not run neural inference or
consume the prospective test set. A development pass is never qualification.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from calibrate_vocal_f0 import CONDS, ROOT, _duration, _runner, _score, _summary, replay
from natural_features.features.common import extractor_metadata


def _replay_cache_key(artifact, report, group, row):
    # Identical audio can occur in several jobs (especially silence/backing).
    # Their row-specific provenance must not overwrite an earlier saved file.
    identity = {'original_artifact': artifact, 'group': group,
                'excerpt_id': row['excerpt_id'], 'condition': row['condition'],
                **{k: report[k] for k in ('protocol_sha256', 'extractor_source_sha256',
                                         'replay_source_sha256', 'evaluation_source_sha256')}}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _removed_frames(runner, assets, row, before, after):
    from calibrate_vocal_f0 import _reference
    from natural_features.features.audio.vocal_f0_benchmark import _nearest_index

    times, hz = _reference(assets, row['excerpt_id'], _duration(before))
    times, hz = runner._scoring_reference(row['condition'], times, hz, before, _duration(before))
    indices = [_nearest_index(before.times_s, float(t), .005001) for t in times]
    removed = np.array([False if i is None else before.values[i, 4] > 0 and after.values[i, 4] == 0
                        for i in indices])
    true_removed = removed & (hz > 0)
    false_removed = removed & (hz == 0)
    voiced_times = times[hz > 0]
    near = sum(min(abs(voiced_times - t)) <= .064 for t in times[false_removed]) if len(voiced_times) else 0
    return {'true_positive_frames_removed': int(true_removed.sum()),
            'false_positive_frames_removed': int(false_removed.sum()),
            'false_positive_frames_removed_within_64ms_of_reference_voicing': int(near)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', type=Path, default=ROOT / 'tests/benchmarks/thresholds/vocal_f0_development_v3.json')
    ap.add_argument('--assets', type=Path, required=True)
    ap.add_argument('--artifact-dir', type=Path, required=True)
    ap.add_argument('--out-json', type=Path, required=True)
    args = ap.parse_args()
    runner = _runner()
    protocol = json.loads(args.protocol.read_text())
    if protocol['method']['voicing_guard_s'] != .02 or protocol['extractor']['voicing_guard_s'] != .02:
        raise ValueError('only the frozen 20ms hypothesis is permitted')
    params = protocol['extractor']
    if tuple(params[k] for k in ('confidence_threshold', 'source_threshold', 'min_source_rms')) != (.65, .03, .01):
        raise ValueError('v3 must retain v2 framewise thresholds')
    report = {'schema': 'VocalF0TemporalDevelopment/v3', 'qualified': False,
              'prospective_test_consumed': False, 'protocol_sha256': runner._digest(args.protocol),
              'extractor_source_sha256': runner._digest(ROOT / 'src/natural_features/features/audio/vocal_f0.py'),
              'replay_source_sha256': runner._digest(ROOT / 'scripts/calibrate_vocal_f0.py'),
              'evaluation_source_sha256': runner._digest(Path(__file__)),
              'scorer_source_sha256': runner._digest(ROOT / 'src/natural_features/features/audio/vocal_f0_benchmark.py'),
              'development_sets': [], 'development_passed': False, 'errors': []}
    try:
        # These files must be exactly the consumed reports registered before replay.
        expected_groups = [[1, 2, 3, 4, 5, 6], [7, 10, 11, 12, 13, 14]]
        if [s['excerpt_ids'] for s in protocol['development_sets']] != expected_groups:
            raise ValueError('development groups differ from frozen v3 experiment')
        for group in protocol['development_sets']:
            input_path = ROOT / group['report']
            if runner._digest(input_path) != group['report_sha256']:
                raise ValueError('consumed report identity mismatch')
            old = json.loads(input_path.read_text())
            keys = {(r['excerpt_id'], r['condition']) for r in old['per_excerpt']}
            if keys != {(i, c) for i in group['excerpt_ids'] for c in CONDS} or len(old['per_excerpt']) != 30 or old['errors']:
                raise ValueError('consumed report is incomplete')
            rows, baseline_rows = [], []
            for row in old['per_excerpt']:
                artifact = row['artifact']
                path = Path(artifact['artifact'])
                if runner._digest(path) != artifact['artifact_sha256']:
                    raise ValueError('native artifact identity mismatch')
                annotation = runner._find_one(args.assets, f"vocadito_{row['excerpt_id']}_f0.csv")
                if runner._digest(annotation) != row['input_hashes']['reference_sha256']:
                    raise ValueError('reference identity mismatch')
                raw = runner._load_feature(path, artifact['cache_key'])
                if raw is None:
                    raise ValueError('invalid native artifact')
                original = old['extractor_parameters']
                restored = replay(raw, original['confidence_threshold'], original['source_threshold'], original['min_source_rms'])
                np.testing.assert_array_equal(restored.values[:, :11], raw.values[:, :11])
                if _score(runner, args.assets, row['excerpt_id'], row['condition'], restored) != row['metrics']:
                    raise ValueError('original scores no longer reproduce')
                before = replay(raw, .65, .03, .01)
                after = replay(raw, .65, .03, .01, voicing_guard_s=.02)
                keep = after.values[:, 4] > 0
                if np.any(keep & (before.values[:, 4] == 0)):
                    raise ValueError('temporal gate added accepted frames')
                np.testing.assert_array_equal(after.values[keep, 0], before.values[keep, 0])
                replay_params = {k: params[k] for k in ('hop_s', 'fmin', 'fmax', 'confidence_threshold',
                                                        'source_threshold', 'min_source_rms', 'voicing_guard_s')}
                after = replace(after, metadata={**raw.metadata,
                    **extractor_metadata('audio.vocal_f0', replay_params, code_version='vocal-f0-v3-replay',
                                         model_revision=raw.metadata['model_revision']),
                    'voicing_guard_s': .02, 'voicing_rule': protocol['method']['definition'],
                    'replay_provenance': artifact,
                    'qualification': 'consumed data; development only'})
                key = _replay_cache_key(artifact, report, group['name'], row)
                out = args.artifact_dir.resolve() / f'{key}.npz'
                runner._save_feature(out, key, after)
                metrics = _score(runner, args.assets, row['excerpt_id'], row['condition'], after)
                baseline_rows.append({**row, 'metrics': _score(runner, args.assets, row['excerpt_id'], row['condition'], before)})
                rows.append({**row, 'metrics': metrics, 'original_artifact': artifact,
                             'artifact': {'artifact': str(out), 'cache_key': key, 'artifact_sha256': runner._digest(out)},
                             'removed': _removed_frames(runner, args.assets, row, before, after)})
            summary, passed, failures = _summary(runner, rows, protocol['gates'], old['analytic'])
            baseline, _, _ = _summary(runner, baseline_rows, protocol['gates'], old['analytic'])
            report['development_sets'].append({'name': group['name'], 'input_report_sha256': group['report_sha256'],
                'summary': summary, 'baseline_v2_summary': baseline, 'gate': {'passed': passed, 'failures': failures},
                'completed_jobs': len(rows), 'original_predictions_and_scores_reproduced': True,
                'analytic': old['analytic'], 'analytic_provenance': 'unchanged pitch engine; retained consumed-report controls',
                'per_excerpt': rows})
            runner._atomic_json(args.out_json, report)
        report['development_passed'] = all(s['gate']['passed'] for s in report['development_sets'])
    except Exception as exc:
        report['errors'].append(runner._error_detail(exc))
    runner._atomic_json(args.out_json, report)
    print(json.dumps({k: report[k] for k in ('development_passed', 'qualified', 'prospective_test_consumed', 'errors')}, indent=2))
    for group in report['development_sets']:
        print(group['name'], json.dumps({'summary': group['summary'], 'gate': group['gate']}))
    return 0 if report['development_passed'] and not report['errors'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
