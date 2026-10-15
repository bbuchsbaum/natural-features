#!/usr/bin/env python3
"""Post-v4 adaptive development: exact decision intervals of one fixed veto."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from calibrate_vocal_f0 import CONDS, ROOT, _duration, _reference, _runner, _score, _summary, replay
from evaluate_vocal_f0_local import _metadata, _save
from natural_features.features.audio.vocal_f0_benchmark import _nearest_index

PROTOCOL_SHA256 = '10977e1e7b38d62d2779ccb15c54c126a81d73100f4aa33b4e68b5760d72fcb6'


def prepare(times, hz, feature):
    """Resolve reference matching once; keep missing rows in denominators."""
    indices = [_nearest_index(feature.times_s, float(t), .005001) for t in times]
    names = list(feature.coords['feature'])
    score = np.full(len(times), -np.inf)
    pitch = np.full(len(times), np.nan)
    for j, i in enumerate(indices):
        if i is not None and feature.values[i, names.index('voiced')] > 0 and feature.values[i, names.index('local_supported')] > 0:
            q = feature.values[i, names.index('local_periodicity')]
            if np.isfinite(q):
                score[j] = q
                pitch[j] = feature.values[i, names.index('candidate_f0_hz')]
    known = ~np.isnan(hz)
    positive = known & (hz > 0)
    negative = known & (hz == 0)
    cents = np.abs(1200 * np.log2(pitch[positive] / hz[positive]))
    return {'positive_scores': score[positive], 'negative_scores': score[negative], 'cents': cents,
            'n_reference': len(times), 'n_scored': int(known.sum()),
            'n_unmatched': sum(i is None and k for i, k in zip(indices, known, strict=True))}


def metrics(data, threshold):
    positive = data['positive_scores'] >= threshold
    errors = data['cents'][positive]
    n_positive, n_negative = len(positive), len(data['negative_scores'])
    def fraction(count, total):
        return None if total == 0 else float(count / total)
    octaves = np.rint(errors / 1200)
    return {'n_reference': data['n_reference'], 'n_scored': data['n_scored'],
            'n_reference_voiced': n_positive, 'n_reference_unvoiced': n_negative,
            'n_unmatched': int(data['n_unmatched']), 'n_joint_voiced': int(positive.sum()),
            'raw_pitch_accuracy': fraction(np.count_nonzero(errors <= 50), n_positive),
            'voicing_recall': fraction(positive.sum(), n_positive),
            'voicing_false_alarm_rate': fraction(np.count_nonzero(data['negative_scores'] >= threshold), n_negative),
            'median_abs_cents_error': None if not len(errors) else float(np.median(errors)),
            'octave_error_rate': fraction(np.count_nonzero((octaves >= 1) & (abs(errors - 1200 * octaves) <= 50)), len(errors)),
            'gross_pitch_error_rate': fraction(np.count_nonzero(errors > 50), len(errors))}


def decision_breakpoints(values):
    q = np.asarray(values, dtype=np.float64)
    q = q[np.isfinite(q) & (q >= 0) & (q < 1)]
    return np.unique(np.concatenate(([0., 1.], np.nextafter(q, np.inf))))


def _first_true(points, predicate):
    low, high = 0, len(points)
    while low < high:
        middle = (low + high) // 2
        if predicate(points[middle]):
            high = middle
        else:
            low = middle + 1
    return low


def feasible_intervals(points, false_alarm_ok, accuracy_ok, all_gates_ok):
    """Prune only monotone constraints; enumerate every remaining state."""
    first = _first_true(points, false_alarm_ok)
    last = _first_true(points, lambda t: not accuracy_ok(t))
    intervals, checked = [], 0
    for i in range(first, last):
        checked += 1
        lower = float(points[i])
        upper = float(np.nextafter(points[i + 1], -np.inf)) if i + 1 < len(points) else 1.
        if all_gates_ok(lower):
            if intervals and intervals[-1]['last_state_index'] == i - 1:
                intervals[-1].update(upper_inclusive=upper, last_state_index=i)
            else:
                intervals.append({'lower_inclusive': lower, 'upper_inclusive': upper,
                                  'first_state_index': i, 'last_state_index': i})
    return intervals, {'total_decision_states': len(points), 'first_false_alarm_feasible_index': first,
                       'first_accuracy_infeasible_index': last, 'fully_checked_states': checked,
                       'minimum_false_alarm_threshold': None if first == len(points) else float(points[first]),
                       'maximum_accuracy_threshold': None if last == 0 else 1. if last == len(points) else float(np.nextafter(points[last], -np.inf))}


def select_midpoint(intervals):
    if not intervals:
        return None
    chosen = max(intervals, key=lambda x: (x['upper_inclusive'] - x['lower_inclusive'], -x['lower_inclusive']))
    return chosen['lower_inclusive'] + (chosen['upper_inclusive'] - chosen['lower_inclusive']) / 2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', type=Path, default=ROOT / 'tests/benchmarks/thresholds/vocal_f0_calibration_v5.json')
    ap.add_argument('--assets', type=Path, required=True)
    ap.add_argument('--artifact-dir', type=Path, required=True)
    ap.add_argument('--out-json', type=Path, required=True)
    args = ap.parse_args()
    runner = _runner()
    if runner._digest(args.protocol) != PROTOCOL_SHA256:
        raise ValueError('protocol differs from frozen adaptive development experiment')
    protocol = json.loads(args.protocol.read_text())
    path = ROOT / protocol['input_report']
    if runner._digest(path) != protocol['input_report_sha256']:
        raise ValueError('input development report identity mismatch')
    old = json.loads(path.read_text())
    if old['errors'] or len(old['raw_diagnostics']) != 60:
        raise ValueError('development input is incomplete')
    report = {'schema': 'VocalF0AdaptiveCalibration/v5', 'qualified': False, 'prospective_test_consumed': False,
              'protocol_sha256': PROTOCOL_SHA256, 'input_report_sha256': runner._digest(path),
              'calibration_source_sha256': runner._digest(Path(__file__)),
              'extractor_source_sha256': runner._digest(ROOT / 'src/natural_features/features/audio/vocal_f0.py'),
              'replay_source_sha256': runner._digest(ROOT / 'scripts/calibrate_vocal_f0.py'),
              'development_passed': False, 'selected_threshold': None, 'errors': []}
    try:
        loaded, prepared, analytics = {}, {}, {}
        for group in protocol['development_sets']:
            prior_path = ROOT / group['report']
            if runner._digest(prior_path) != group['report_sha256']:
                raise ValueError('original consumed report identity mismatch')
            prior = json.loads(prior_path.read_text())
            analytics[group['name']] = prior['analytic']
        for row in old['raw_diagnostics']:
            a = row['artifact']
            artifact_path = Path(a['artifact'])
            if runner._digest(artifact_path) != a['artifact_sha256']:
                raise ValueError('diagnostic artifact identity mismatch')
            feature = runner._load_feature(artifact_path, a['cache_key'])
            if feature is None:
                raise ValueError('invalid diagnostic artifact')
            annotation = runner._find_one(args.assets, f"vocadito_{row['excerpt_id']}_f0.csv")
            if runner._digest(annotation) != row['input_hashes']['reference_sha256']:
                raise ValueError('reference identity mismatch')
            t, hz = _reference(args.assets, row['excerpt_id'], _duration(feature))
            t, hz = runner._scoring_reference(row['condition'], t, hz, feature, _duration(feature))
            key = (row['group'], row['excerpt_id'], row['condition'])
            loaded[key] = feature
            prepared[key] = prepare(t, hz, feature)
        expected = {(g['name'], i, c) for g in protocol['development_sets'] for i in g['excerpt_ids'] for c in CONDS}
        if set(loaded) != expected:
            raise ValueError('wrong development groups or missing jobs')
        # Verify the optimized score path against every already frozen v4 row.
        for profile in old['profiles']:
            for group in profile['development_sets']:
                for row in group['per_excerpt']:
                    key = (group['name'], row['excerpt_id'], row['condition'])
                    if metrics(prepared[key], profile['threshold']) != row['metrics']:
                        raise ValueError('prepared scoring disagrees with frozen v4 profile')
        report['verified_v4_rows'] = 240
        def summaries(threshold):
            groups = []
            for group in protocol['development_sets']:
                rows = [{'excerpt_id': i, 'condition': c, 'metrics': metrics(prepared[(group['name'], i, c)], threshold)}
                        for i in group['excerpt_ids'] for c in CONDS]
                summary, passed, failures = _summary(runner, rows, protocol['gates'], analytics[group['name']])
                groups.append({'name': group['name'], 'summary': summary, 'gate': {'passed': passed, 'failures': failures}, 'per_excerpt': rows})
            return groups
        gates = protocol['gates']
        def monotone(threshold, accuracy):
            for group in protocol['development_sets']:
                for c in CONDS:
                    if accuracy and c in ('backing', 'silence'):
                        continue
                    rows = [prepared[(group['name'], i, c)] for i in group['excerpt_ids']]
                    if accuracy:
                        rpa = np.mean([np.count_nonzero((r['positive_scores'] >= threshold) & (r['cents'] <= 50)) / len(r['positive_scores']) for r in rows])
                        recall = np.mean([np.count_nonzero(r['positive_scores'] >= threshold) / len(r['positive_scores']) for r in rows])
                        if rpa < gates['raw_pitch_accuracy_min'] or recall < gates['voicing_recall_min']:
                            return False
                    else:
                        fa = np.mean([np.count_nonzero(r['negative_scores'] >= threshold) / len(r['negative_scores']) for r in rows])
                        limit = gates['backing_false_alarm_rate_max'] if c == 'backing' else gates['silence_false_alarm_rate_max'] if c == 'silence' else gates['voicing_false_alarm_rate_max']
                        if fa > limit:
                            return False
            return True
        q = np.concatenate([v for r in prepared.values() for v in (r['positive_scores'], r['negative_scores'])])
        points = decision_breakpoints(q)
        intervals, search = feasible_intervals(points, lambda t: monotone(t, False), lambda t: monotone(t, True),
                                               lambda t: all(g['gate']['passed'] for g in summaries(t)))
        report['feasible_intervals'], report['search'] = intervals, search
        selected = select_midpoint(intervals)
        report['selected_threshold'] = selected
        if selected is not None:
            report['selected_development_sets'] = summaries(selected)
            if not all(g['gate']['passed'] for g in report['selected_development_sets']):
                raise ValueError('midpoint does not pass original gates')
            report['selected_artifacts'] = []
            for key, feature in loaded.items():
                out = replay(feature, .65, .03, .01, local_periodicity_threshold=selected)
                out = replace(out, metadata=_metadata(feature, selected, feature.metadata['replay_provenance']))
                if _score(runner, args.assets, key[1], key[2], out) != metrics(prepared[key], selected):
                    raise ValueError('selected direct scorer/replay disagrees with calibration')
                identity = {k: v for k, v in report.items() if k.endswith('_sha256')}
                identity.update(group=key[0], excerpt_id=key[1], condition=key[2], threshold=selected)
                artifact = _save(runner, args.artifact_dir, out, identity)
                report['selected_artifacts'].append({'group': key[0], 'excerpt_id': key[1], 'condition': key[2], 'artifact': artifact})
            report['development_passed'] = True
    except Exception as exc:
        report['errors'].append(runner._error_detail(exc))
    runner._atomic_json(args.out_json, report)
    print(json.dumps({k: report.get(k) for k in ('development_passed', 'selected_threshold', 'feasible_intervals', 'search', 'errors')}, indent=2))
    return 0 if report['development_passed'] and not report['errors'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
