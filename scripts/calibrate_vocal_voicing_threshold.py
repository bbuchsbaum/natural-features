#!/usr/bin/env python3
"""V8 exhaustive threshold selection inside the frozen V7 nested singer folds."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

import calibrate_vocal_voicing as v7
from natural_features.features.audio.vocal_voicing import predict_probability
from natural_features.features.audio.vocal_voicing_threshold import (
    apply_threshold_model, canonical_thresholds, effective_score, threshold_mask,
)

ROOT = v7.ROOT
PROTOCOL_SHA256 = 'eba6c0708a6b982b4abb04fb86e0130a49a6412d51b5e0d6e5ae87567d3aa395'


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _counts_above(scores, levels):
    ordered = np.sort(scores)
    return len(ordered) - np.searchsorted(ordered, levels, side='right')


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else np.full(len(numerator), np.nan)


def _blocks(mask, thresholds):
    changes = np.diff(np.r_[False, mask, False].astype(int))
    return [{'first_index': int(a), 'last_index': int(b-1), 'state_count': int(b-a),
             'first_canonical_threshold': float(thresholds[a]), 'last_canonical_threshold': float(thresholds[b-1])}
            for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))]


def exhaustive_search(records, probabilities, gates, state_path=None):
    """Integer annotation counts, exact macro denominators, then conditional gates."""
    if len({r.key for r in records}) != len(records) or set(probabilities) != {r.key for r in records}:
        raise ValueError('probability coverage must exactly match unique records')
    thresholds = canonical_thresholds([probabilities[r.key] for r in records], [r.admissible for r in records])
    levels = np.log(thresholds/(1-thresholds))
    per_record, positive_rows = {}, {}
    for r in records:
        scores = effective_score(probabilities[r.key])
        accepted = r.matched.copy()
        accepted[r.matched] &= r.admissible[r.indices[r.matched]]
        positive = accepted & r.positive
        negative = accepted & r.negative
        pos_scores = scores[r.indices[positive]]
        cents = r.cents[positive]
        joint = _counts_above(pos_scores, levels)
        correct = _counts_above(pos_scores[cents <= 50], levels)
        false = _counts_above(scores[r.indices[negative]], levels)
        multiple = np.rint(cents/1200)
        octave = _counts_above(pos_scores[(multiple >= 1) & (abs(cents-1200*multiple) <= 50)], levels)
        per_record[r.key] = {
            'raw_pitch_accuracy': _ratio(correct, int(r.positive.sum())),
            'voicing_recall': _ratio(joint, int(r.positive.sum())),
            'voicing_false_alarm_rate': _ratio(false, int(r.negative.sum())),
            'octave_error_rate': np.divide(octave, joint, out=np.full(len(levels), np.nan), where=joint > 0),
            'median_abs_cents_error': np.full(len(levels), np.nan),
            'gross_pitch_error_rate': np.divide(joint-correct, joint, out=np.full(len(levels), np.nan), where=joint > 0),
        }
        positive_rows[r.key] = (pos_scores, cents)
    groups, curves = [], {}
    recall_ok = np.ones(len(levels), dtype=bool)
    false_ok = np.ones(len(levels), dtype=bool)
    for group in sorted({r.group for r in records}):
        absent = []
        for condition in v7.CONDS:
            rows = [r for r in records if r.group == group and r.condition == condition]
            if not rows:
                absent.append(condition)
                if condition in v7.POSITIVE:
                    recall_ok[:] = False
                continue
            for metric in ('raw_pitch_accuracy', 'voicing_recall', 'voicing_false_alarm_rate'):
                curve = np.mean([per_record[r.key][metric] for r in rows], axis=0)
                curves[(group, condition, metric)] = curve
            for metric, op, bound in v7._limits(condition, gates):
                if metric not in ('raw_pitch_accuracy', 'voicing_recall', 'voicing_false_alarm_rate'):
                    continue
                curve = curves[(group, condition, metric)]
                passed = np.isfinite(curve) & (curve >= bound if op == '>=' else curve <= bound)
                if metric == 'voicing_false_alarm_rate':
                    false_ok &= passed
                else:
                    recall_ok &= passed
        groups.append({'name': group, 'absent_conditions': absent,
                       'untested_controls': {c: 'ancestor_family_purge' for c in absent if c not in v7.POSITIVE}})
    monotone = recall_ok & false_ok
    feasible = monotone.copy()
    # Compute each distinct accepted-positive set only once per record. Duplicate
    # reference annotations remain duplicate entries in this median calculation.
    for r in records:
        if r.condition not in v7.POSITIVE:
            continue
        scores, errors = positive_rows[r.key]
        order = np.argsort(scores, kind='stable')
        scores, errors = scores[order], errors[order]
        offsets = np.searchsorted(scores, levels[monotone], side='right')
        distinct, inverse = np.unique(offsets, return_inverse=True)
        medians = np.array([float(np.median(errors[i:])) if i < len(errors) else np.nan for i in distinct])
        per_record[r.key]['median_abs_cents_error'][monotone] = medians[inverse]
    for group in groups:
        for condition in v7.CONDS:
            rows = [r for r in records if r.group == group['name'] and r.condition == condition]
            if not rows:
                continue
            for metric in ('median_abs_cents_error', 'octave_error_rate', 'gross_pitch_error_rate'):
                curves[(group['name'], condition, metric)] = np.mean([per_record[r.key][metric] for r in rows], axis=0)
            if condition in v7.POSITIVE:
                for metric, bound in [('median_abs_cents_error', gates['median_abs_cents_error_max']),
                                      ('octave_error_rate', gates['octave_error_rate_max'])]:
                    curve = curves[(group['name'], condition, metric)]
                    feasible &= np.isfinite(curve) & (curve <= bound)
    singing = [(g['name'], c) for g in groups for c in v7.POSITIVE]
    minimum_rpa = np.min([curves.get((g, c, 'raw_pitch_accuracy'), np.full(len(levels), np.nan)) for g, c in singing], axis=0)
    maximum_fa = np.max([curves.get((g, c, 'voicing_false_alarm_rate'), np.full(len(levels), np.nan)) for g, c in singing], axis=0)
    eligible = np.flatnonzero(feasible)
    index = (max(eligible, key=lambda i: (minimum_rpa[i], -maximum_fa[i], thresholds[i])) if len(eligible) else None)
    threshold = .5 if index is None else float(thresholds[index])
    scores = {r.key: r.score(threshold_mask(probabilities[r.key], r.admissible, threshold)) for r in records}
    summaries = v7.summarize(records, scores, gates)
    if index is not None and not all(g['gate']['passed'] for g in summaries):
        raise ValueError('cumulative selection disagrees with scalar scoring')
    arrays = {'threshold': thresholds, 'threshold_score': levels, 'recall_constraints_pass': recall_ok,
              'false_alarm_constraints_pass': false_ok, 'monotone_pass': monotone, 'all_gates_pass': feasible,
              'minimum_rpa': minimum_rpa, 'maximum_false_alarm': maximum_fa}
    arrays.update({'::'.join(key): value for key, value in curves.items()})
    result = {'selected': {'threshold': threshold, 'switch_penalty': 0.}, 'eligible': index is not None,
              'fallback_used': index is None, 'selected_state_index': None if index is None else int(index),
              'state_count': len(thresholds), 'monotone_feasible_states': int(monotone.sum()),
              'all_gates_feasible_states': int(feasible.sum()), 'feasible_state_blocks': _blocks(feasible, thresholds),
              'first_false_alarm_feasible': None if not false_ok.any() else float(thresholds[np.flatnonzero(false_ok)[0]]),
              'last_recall_accuracy_feasible': None if not recall_ok.any() else float(thresholds[np.flatnonzero(recall_ok)[-1]]),
              'groups': summaries, 'structure': groups,
              'objective': None if index is None else {'minimum_rpa': float(minimum_rpa[index]), 'maximum_false_alarm': float(maximum_fa[index])}}
    if state_path is not None:
        state_path = Path(state_path)
        np.savez_compressed(state_path, **arrays)
        result['states_artifact'] = {'path': str(state_path.resolve()), 'sha256': hashlib.sha256(state_path.read_bytes()).hexdigest()}
    return result, arrays


class ReusedModels:
    """Validate every ancestry boundary before reusing the hash-bound fit archive."""
    def __init__(self, report):
        self.report = report
        self.traces = {(t['context'], t['kind']): t for t in report['fit_trace']}
        self.checked = set()
        if len(self.traces) != len(report['fit_trace']):
            raise ValueError('duplicate fit trace')

    def trace(self, records, held, context, kind):
        t = self.traces[(context, kind)]
        if ([r.key for r in records] != [r.key for r in v7.purged_training(records, held)]
                or t['training_keys'] != [r.key for r in records]
                or t['forbidden_keys'] != sorted({r.key for r in held})):
            raise ValueError(f'fit ancestry mismatch: {context}')
        self.checked.add((context, kind))
        return t

    def pipeline(self, records, held, context, expected):
        for fold, validation in enumerate(v7.grouped_folds(records, 3)):
            forbidden = held + validation
            self.trace(v7.purged_training(records, forbidden), forbidden, f'{context}/calibration-{fold}', 'base')
        base = self.trace(records, held, f'{context}/refit', 'base')['model']
        platt = self.trace(records, held, context, 'platt')['calibration']
        model = {**base, 'calibration': platt}
        if model != expected:
            raise ValueError(f'refit model mismatch: {context}')
        return model

    def selector(self, records, held, context, selection):
        probabilities = {}
        folds = v7.grouped_folds(records, 3)
        if len(selection['selector_models']) != len(folds):
            raise ValueError('selector model coverage mismatch')
        for i, (validation, saved) in enumerate(zip(folds, selection['selector_models'])):
            if saved['validation_keys'] != [r.key for r in validation]:
                raise ValueError('selector fold mismatch')
            forbidden = held + validation
            model = self.pipeline(v7.purged_training(records, forbidden), forbidden, f'{context}/selector-{i}', saved['model'])
            for r in validation:
                probabilities[r.key] = predict_probability(r.x, model)
        return probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT/'tests/benchmarks/thresholds/vocal_voicing_development_v8.json')
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--artifact-dir', type=Path, required=True)
    parser.add_argument('--out-json', type=Path, required=True)
    parser.add_argument('--audit-json', type=Path, required=True)
    args = parser.parse_args()
    runner = v7._runner()
    if runner._digest(args.protocol) != PROTOCOL_SHA256:
        raise ValueError('v8 protocol identity mismatch')
    protocol = json.loads(args.protocol.read_text())
    for relative, digest in protocol['frozen_inputs'].items():
        if runner._digest(ROOT/relative) != digest:
            raise ValueError(f'frozen input changed: {relative}')
    old = json.loads((ROOT/protocol['v7_report']).read_text())
    old_protocol = json.loads((ROOT/protocol['v7_protocol']).read_text())
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    report = {'schema': 'VocalVoicingDevelopment/v8', 'qualified': False, 'fresh_data_consumed': False,
              'environment': {'python': sys.version, 'numpy': np.__version__},
              'development_passed': False, 'protocol_sha256': PROTOCOL_SHA256,
              'v7_report_sha256': protocol['frozen_inputs'][protocol['v7_report']],
              'source_sha256': {p: runner._digest(ROOT/p) for p in ['scripts/calibrate_vocal_voicing_threshold.py',
                                 'src/natural_features/features/audio/vocal_voicing_threshold.py', 'scripts/audit_vocal_voicing_errors.py']},
              'outer_folds': [], 'outer_artifacts': [], 'errors': []}
    try:
        records, analytic = v7.load_records(old_protocol, args.assets, runner)
        if old['errors'] or old['fresh_data_consumed'] or protocol['gates'] != old_protocol['gates']:
            raise ValueError('invalid inherited experiment or changed gates')
        if len(old['dataset']) != len(records) or [d['key'] for d in old['dataset']] != [r.key for r in records]:
            raise ValueError('dataset identity mismatch')
        for r, saved in zip(records, old['dataset']):
            if (saved['singer'] != r.singer or saved['condition'] != r.condition or saved['group'] != r.group
                    or saved['excerpt_id'] != r.excerpt or saved['negative_family'] != r.family
                    or saved['rendered_audio_sha256'] != r.audio_hash or saved['provenance'] != r.provenance):
                raise ValueError('dataset provenance mismatch')
        report['dataset'] = old['dataset']
        reuse = ReusedModels(old)
        outer_scores, audit_inputs = {}, {}
        for fold, validation in enumerate(v7.grouped_folds(records, 5, outer=True)):
            before = old['outer_folds'][fold]
            training = v7.purged_training(records, validation)
            if before['validation_keys'] != [r.key for r in validation] or before['training_keys'] != [r.key for r in training]:
                raise ValueError('outer membership mismatch')
            probabilities = reuse.selector(training, validation, f'outer-{fold}', before['selection'])
            selection, _ = exhaustive_search(training, probabilities, protocol['gates'], args.artifact_dir/f'outer-{fold}-states.npz')
            selection['source_models_sha256'] = digest_json(before['selection']['selector_models'])
            model = reuse.pipeline(training, validation, f'outer-{fold}/selected', before['profile']['model'])
            profile = {**selection['selected'], 'model': model, 'protocol_sha256': PROTOCOL_SHA256,
                       'status': 'v8 nested development only; explicit direct zero-switch decoder'}
            result = {'fold': fold, 'held_singers': before['held_singers'], 'validation_keys': before['validation_keys'],
                      'training_keys': before['training_keys'], 'selection': selection, 'profile': profile, 'per_excerpt': []}
            for r in validation:
                out = apply_threshold_model(r.feature, profile)
                names = list(out.coords['feature'])
                voiced = out.values[:, names.index('voiced')] > 0
                metrics = r.score(voiced)
                if metrics != v7.score_vocal_f0(r.reference_times, r.reference_hz, out):
                    raise ValueError('v8 output/scorer mismatch')
                outer_scores[r.key] = metrics
                probability = out.values[:, names.index('voicing_probability')]
                baseline = threshold_mask(probability, r.admissible, .5)
                saved = next(a for a in old['outer_artifacts'] if a['key'] == r.key)
                if runner._digest(Path(saved['artifact'])) != saved['artifact_sha256']:
                    raise ValueError('v7 baseline artifact changed')
                previous = runner._load_feature(Path(saved['artifact']), saved['cache_key'])
                if previous is None or before['profile']['threshold'] != .5 or before['profile']['switch_penalty'] != 0:
                    raise ValueError('unexpected inherited baseline')
                old_names = list(previous.coords['feature'])
                for column in ('voicing_probability', 'voicing_admissible', 'candidate_f0_hz'):
                    np.testing.assert_array_equal(out.values[:, names.index(column)], previous.values[:, old_names.index(column)])
                np.testing.assert_array_equal(out.times_s, previous.times_s)
                previous_mask = previous.values[:, list(previous.coords['feature']).index('voiced')] > 0
                baseline_differences = int(np.count_nonzero(baseline != previous_mask))
                identity = {'protocol': PROTOCOL_SHA256, 'sources': report['source_sha256'], 'profile': profile,
                            'input': r.provenance['artifact']['artifact_sha256'], 'row': r.key}
                cache_key = digest_json(identity)
                path = args.artifact_dir/f'{cache_key}.npz'
                runner._save_feature(path, cache_key, out)
                report['outer_artifacts'].append({'key': r.key, 'fold': fold, 'artifact': str(path.resolve()),
                                                  'artifact_sha256': runner._digest(path), 'cache_key': cache_key})
                result['per_excerpt'].append({'key': r.key, 'metrics': metrics, 'comparator_only_v7_mask_changes': baseline_differences})
                audit_inputs[r.key] = {'probability': probability, 'voiced': voiced, 'threshold': profile['threshold'],
                                      'v7_baseline': previous_mask, 'comparator_baseline': baseline}
            report['outer_folds'].append(result)
            runner._atomic_json(args.out_json, report)
            print(f"outer {fold}: eligible={selection['eligible']} threshold={profile['threshold']:.17g} states={selection['state_count']}", flush=True)
        if set(outer_scores) != {r.key for r in records}:
            raise ValueError('incomplete outer scoring')
        report['outer_groups'] = v7.summarize(records, outer_scores, protocol['gates'], complete=True)
        probabilities = reuse.selector(records, [], 'final', old['final_selection'])
        final, _ = exhaustive_search(records, probabilities, protocol['gates'], args.artifact_dir/'final-states.npz')
        final['source_models_sha256'] = digest_json(old['final_selection']['selector_models'])
        report['final_selection'] = final
        model = reuse.pipeline(records, [], 'final/selected', old['final_profile']['profile']['model'])
        if reuse.checked != set(reuse.traces):
            raise ValueError('not every inherited fit was checked')
        report['verified_reused_fit_traces'] = len(reuse.checked)
        profile = {**final['selected'], 'model': model, 'protocol_sha256': PROTOCOL_SHA256,
                   'status': 'v8 adaptive development only; requires independent qualification; use apply_threshold_model'}
        profile_path = args.artifact_dir/'final_profile.json'
        runner._atomic_json(profile_path, profile)
        report['final_profile'] = {'path': str(profile_path.resolve()), 'sha256': runner._digest(profile_path), 'profile': profile}
        report['analytic'] = analytic
        analytic_passed = all(a['result']['complete'] and a['result']['worst_median_abs_cents_error'] is not None
                              and a['result']['worst_median_abs_cents_error'] <= protocol['gates']['analytic_median_abs_cents_error_max']
                              for a in analytic.values())
        report['analytic_passed'] = analytic_passed
        report['development_passed'] = (all(f['selection']['eligible'] for f in report['outer_folds']) and final['eligible']
                                         and all(g['gate']['passed'] for g in report['outer_groups']) and analytic_passed)
        from audit_vocal_voicing_errors import audit_errors
        audit = audit_errors(records, audit_inputs, args.assets, args.artifact_dir/'error-audit', protocol)
        runner._atomic_json(args.audit_json, audit)
        report['error_audit'] = {'path': str(args.audit_json.resolve()), 'sha256': runner._digest(args.audit_json)}
    except Exception as exc:
        report['errors'].append(runner._error_detail(exc))
        report['development_passed'] = False
    runner._atomic_json(args.out_json, report)
    print(json.dumps({'development_passed': report['development_passed'], 'errors': report['errors'],
                      'outer_groups': report.get('outer_groups'), 'final_selection': report.get('final_selection')}, indent=2))
    return 0 if report['development_passed'] and not report['errors'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
