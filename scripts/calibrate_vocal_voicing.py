#!/usr/bin/env python3
"""Nested singer-grouped development of a frozen learned voicing procedure."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from natural_features.features.audio.vocal_voicing import (  # noqa: E402
    FEATURE_NAMES, MODEL_SCHEMA, NEURAL_REVISION, apply_voicing_model, decode_states, evidence,
    predict_probability, sigmoid,
)
from natural_features.features.audio.vocal_f0_benchmark import _nearest_index, score_vocal_f0  # noqa: E402
from calibrate_vocal_f0 import CONDS, _duration, _reference, _runner  # noqa: E402

PROTOCOL_SHA256 = "26099428ce8de69c703eb383ed48577f2af478bb7c69eba875ea247bbef85fc4"
POSITIVE = CONDS[:3]
METRICS = ("raw_pitch_accuracy", "voicing_recall", "voicing_false_alarm_rate",
           "median_abs_cents_error", "octave_error_rate", "gross_pitch_error_rate")


@dataclass
class Record:
    key: str
    group: str
    singer: str
    excerpt: int
    condition: str
    audio_hash: str
    family: str | None
    feature: object
    reference_times: np.ndarray
    reference_hz: np.ndarray
    provenance: dict

    def __post_init__(self):
        self.x, self.admissible = evidence(self.feature)
        self.indices = np.array([(-1 if (i := _nearest_index(self.feature.times_s, float(t), .005001)) is None else i)
                                 for t in self.reference_times], dtype=int)
        self.known = ~np.isnan(self.reference_hz)
        self.positive = self.known & (self.reference_hz > 0)
        self.negative = self.known & (self.reference_hz == 0)
        self.matched = self.indices >= 0
        mask = self.known & self.matched
        n = len(self.x)
        self.counts = np.bincount(self.indices[mask], minlength=n).astype(float)
        voiced_counts = np.bincount(self.indices[self.positive & self.matched], minlength=n)
        self.labels = np.divide(voiced_counts, self.counts, out=np.zeros(n), where=self.counts > 0)
        self.training_counts = self.counts * self.admissible
        names = list(self.feature.coords["feature"])
        self.candidate = self.feature.values[:, names.index("candidate_f0_hz")]
        self.cents = np.full(len(self.reference_hz), np.inf)
        positive_matched = self.positive & self.matched
        self.cents[positive_matched] = abs(1200 * np.log2(self.candidate[self.indices[positive_matched]] / self.reference_hz[positive_matched]))

    def score(self, voiced):
        predicted = np.zeros(len(self.reference_hz), dtype=bool)
        predicted[self.matched] = np.asarray(voiced, dtype=bool)[self.indices[self.matched]]
        joint = predicted & self.positive
        error = self.cents[joint]
        def fraction(a, b):
            return None if not b else float(a/b)
        octave = np.rint(error/1200)
        return {"n_reference": len(predicted), "n_scored": int(self.known.sum()),
                "n_reference_voiced": int(self.positive.sum()), "n_reference_unvoiced": int(self.negative.sum()),
                "n_unmatched": int((self.known & ~self.matched).sum()), "n_joint_voiced": int(joint.sum()),
                "raw_pitch_accuracy": fraction(np.count_nonzero(error <= 50), self.positive.sum()),
                "voicing_recall": fraction(joint.sum(), self.positive.sum()),
                "voicing_false_alarm_rate": fraction((predicted & self.negative).sum(), self.negative.sum()),
                "median_abs_cents_error": None if not len(error) else float(np.median(error)),
                "octave_error_rate": fraction(np.count_nonzero((octave >= 1) & (abs(error-1200*octave) <= 50)), len(error)),
                "gross_pitch_error_rate": fraction(np.count_nonzero(error > 50), len(error))}


def singer_order(singer):
    return int(singer[1:])


def grouped_folds(records, count, outer=False):
    singers = sorted({r.singer for r in records}, key=singer_order)
    if len(singers) < count:
        raise ValueError("too few singers for frozen grouped folds")
    groups = [{s for i, s in enumerate(singers) if ((singer_order(s)-1) if outer else i) % count == fold}
              for fold in range(count)]
    return [[r for r in records if r.singer in group] for group in groups]


def purged_training(records, held):
    singers = {r.singer for r in held}
    hashes = {r.audio_hash for r in held}
    families = {r.family for r in held if r.family is not None}
    return [r for r in records if r.singer not in singers and r.audio_hash not in hashes
            and (r.family is None or r.family not in families)]


def stream_weights(records):
    """Annotation weights with singer and duplicate-negative mass accounting."""
    active = [r for r in records if r.training_counts.sum() > 0]
    mass = {}
    for singer in {r.singer for r in active if r.condition in POSITIVE}:
        singing = [r for r in active if r.singer == singer and r.condition in POSITIVE]
        excerpts = {r.excerpt for r in singing}
        for excerpt in excerpts:
            streams = [r for r in singing if r.excerpt == excerpt]
            for r in streams:
                mass[r.key] = 1 / len(excerpts) / len(streams)
    for family in {r.family for r in active if r.family is not None}:
        negatives = [r for r in active if r.family == family]
        hashes = {r.audio_hash for r in negatives}
        for digest in hashes:
            copies = [r for r in negatives if r.audio_hash == digest]
            for r in copies:
                mass[r.key] = 1 / len(hashes) / len(copies)
    total = sum(mass.values())
    if not total:
        raise ValueError("empty admissible training observations")
    return {r.key: r.training_counts * mass[r.key] / r.training_counts.sum() / total
            for r in active}


def training_arrays(records, inputs=None):
    weights = stream_weights(records)
    xs, ys, ws = [], [], []
    for r in records:
        if r.key not in weights:
            continue
        keep = weights[r.key] > 0
        xs.append((r.x if inputs is None else inputs[r.key])[keep])
        ys.append(r.labels[keep])
        ws.append(weights[r.key][keep])
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)


def binary_objective(theta, x, y, weights, l2):
    z = x @ theta[:-1] + theta[-1]
    residual = weights * (sigmoid(z)-y)
    return (float(weights @ (np.logaddexp(0., z)-y*z) + .5*l2*np.dot(theta[:-1], theta[:-1])),
            np.r_[x.T @ residual + l2*theta[:-1], residual.sum()])


def platt_objective(theta, logits, y, weights, l2):
    a, b = theta
    z = a * logits + b
    residual = weights * (sigmoid(z)-y)
    return (float(weights @ (np.logaddexp(0., z)-y*z) + .5*l2*(a-1)**2),
            np.array([residual @ logits + l2*(a-1), residual.sum()]))


def _check_fit(result, y, weights):
    prevalence = float(weights @ y)
    if not 0 < prevalence < 1:
        raise ValueError("fit requires both voicing classes")
    if not result.success or not np.isfinite(result.fun) or not np.all(np.isfinite(result.x)):
        raise ValueError(f"voicing fit did not converge: {result.message}")


class Trainer:
    def __init__(self, protocol, trace):
        self.protocol, self.trace = protocol, trace

    def base(self, records, held, context):
        expected = purged_training(records, held)
        if [r.key for r in expected] != [r.key for r in records]:
            raise ValueError("forbidden record entered base fit")
        x, y, weights = training_arrays(records)
        mean = weights @ x
        variance = weights @ ((x-mean)**2)
        scale = np.where(np.ptp(x, axis=0) == 0, 1., np.sqrt(variance))
        normalized = (x-mean)/scale
        fit = minimize(binary_objective, np.zeros(x.shape[1]+1), args=(normalized, y, weights, self.protocol['fit']['l2']),
                       method="L-BFGS-B", jac=True, options=self.protocol['fit']['optimizer_options'])
        _check_fit(fit, y, weights)
        revision = {r.feature.metadata['model_revision'] for r in records}
        if len(revision) != 1:
            raise ValueError("inconsistent neural model revisions")
        model = {'schema': MODEL_SCHEMA, 'feature_names': list(FEATURE_NAMES), 'mean': mean.tolist(), 'scale': scale.tolist(),
                 'coefficients': fit.x[:-1].tolist(), 'intercept': float(fit.x[-1]),
                 'calibration': {'slope': 1., 'intercept': 0.}, 'model_revision': next(iter(revision)),
                 'native_frontends': self.protocol['frontend']['accepted_native_identities']}
        self.trace.append({'context': context, 'kind': 'base', 'training_keys': [r.key for r in records],
                           'forbidden_keys': sorted({r.key for r in held}), 'model': model,
                           'loss': float(fit.fun), 'iterations': int(fit.nit), 'prevalence': float(weights@y)})
        return model

    @staticmethod
    def logits(record, model):
        return ((record.x-np.asarray(model['mean']))/np.asarray(model['scale'])) @ np.asarray(model['coefficients']) + model['intercept']

    def pipeline(self, records, held, context):
        logits = {}
        for fold, validation in enumerate(grouped_folds(records, 3)):
            forbidden = held + validation
            train = purged_training(records, forbidden)
            base = self.base(train, forbidden, f"{context}/calibration-{fold}")
            for r in validation:
                logits[r.key] = self.logits(r, base)
        z, y, weights = training_arrays(records, logits)
        fit = minimize(platt_objective, [1., 0.], args=(z, y, weights, self.protocol['fit']['platt_l2']),
                       method="L-BFGS-B", jac=True, bounds=[(0., None), (None, None)],
                       options=self.protocol['fit']['optimizer_options'])
        _check_fit(fit, y, weights)
        base = self.base(records, held, f"{context}/refit")
        model = {**base, 'calibration': {'slope': float(fit.x[0]), 'intercept': float(fit.x[1])}}
        self.trace.append({'context': context, 'kind': 'platt', 'training_keys': [r.key for r in records],
                           'forbidden_keys': sorted({r.key for r in held}), 'calibration': model['calibration'],
                           'loss': float(fit.fun), 'iterations': int(fit.nit)})
        return model


def _limits(condition, gates):
    if condition == 'backing':
        return [('voicing_false_alarm_rate', '<=', gates['backing_false_alarm_rate_max'])]
    if condition == 'silence':
        return [('voicing_false_alarm_rate', '<=', gates['silence_false_alarm_rate_max'])]
    return [('raw_pitch_accuracy', '>=', gates['raw_pitch_accuracy_min']),
            ('voicing_recall', '>=', gates['voicing_recall_min']),
            ('voicing_false_alarm_rate', '<=', gates['voicing_false_alarm_rate_max']),
            ('median_abs_cents_error', '<=', gates['median_abs_cents_error_max']),
            ('octave_error_rate', '<=', gates['octave_error_rate_max'])]


def summarize(records, metrics, gates, complete=False):
    groups = []
    for group in sorted({r.group for r in records}):
        summary, failures, absent = {}, [], []
        for condition in CONDS:
            rows = [r for r in records if r.group == group and r.condition == condition]
            if not rows:
                absent.append(condition)
                if complete or condition in POSITIVE:
                    failures.append(f"{condition}: absent")
                continue
            values = [metrics[r.key] for r in rows]
            summary[condition] = {key: None if any(row[key] is None for row in values) else float(np.mean([row[key] for row in values]))
                                  for key in METRICS}
            if complete and len(rows) != 6:
                failures.append(f"{condition}: expected six excerpts")
            for metric, op, bound in _limits(condition, gates):
                value = summary[condition][metric]
                if value is None or (value < bound if op == '>=' else value > bound):
                    failures.append(f"{condition}.{metric} {op} {bound}: {value}")
        groups.append({'name': group, 'summary': summary, 'absent_conditions': absent,
                       'untested_controls': {c: 'ancestor_family_purge' for c in absent if c not in POSITIVE and not complete},
                       'gate': {'passed': not failures, 'failures': failures}})
    return groups


def select_profiles(profiles):
    eligible = [p for p in profiles if p['eligible']]
    if not eligible:
        return {'threshold': .5, 'switch_penalty': 0.}, False
    best = max(eligible, key=lambda p: (p['objective']['minimum_rpa'], -p['objective']['maximum_false_alarm'],
                                      -p['parameters']['switch_penalty'], p['parameters']['threshold']))
    return best['parameters'], True


def select(trainer, records, held, context):
    probabilities, models = {}, []
    for fold, validation in enumerate(grouped_folds(records, 3)):
        forbidden = held + validation
        train = purged_training(records, forbidden)
        model = trainer.pipeline(train, forbidden, f"{context}/selector-{fold}")
        models.append({'validation_keys': [r.key for r in validation], 'model': model})
        for r in validation:
            probabilities[r.key] = predict_probability(r.x, model)
    profiles = []
    for threshold, penalty in product(trainer.protocol['grid']['threshold'], trainer.protocol['grid']['switch_penalty']):
        metrics = {r.key: r.score(decode_states(probabilities[r.key], r.admissible, threshold, penalty)) for r in records}
        groups = summarize(records, metrics, trainer.protocol['gates'])
        positive = [g['summary'][c] for g in groups for c in POSITIVE]
        profiles.append({'parameters': {'threshold': threshold, 'switch_penalty': penalty},
                         'eligible': all(g['gate']['passed'] for g in groups), 'groups': groups,
                         'objective': {'minimum_rpa': min(m['raw_pitch_accuracy'] for m in positive),
                                       'maximum_false_alarm': max(m['voicing_false_alarm_rate'] for m in positive)}})
    parameters, eligible = select_profiles(profiles)
    return {'selected': parameters, 'eligible': eligible, 'fallback_used': not eligible,
            'profiles': profiles, 'selector_models': models}


def probability_statistics(probability, labels, weights):
    p, y, w = (np.asarray(a, dtype=float) for a in (probability, labels, weights))
    w = w / w.sum()
    clipped = np.clip(p, 1e-12, 1-1e-12)
    bins = np.minimum((p*10).astype(int), 9)
    reliability = []
    for i in range(10):
        keep = bins == i
        mass = float(w[keep].sum())
        reliability.append({'bin': i, 'weight': mass, 'annotation_count': int(keep.sum()),
                            'mean_probability': None if not mass else float(w[keep]@p[keep]/mass),
                            'voiced_fraction': None if not mass else float(w[keep]@y[keep]/mass)})
    return {'brier': float(w@(y*(1-p)**2+(1-y)*p**2)),
            'log_loss': float(-w@(y*np.log(clipped)+(1-y)*np.log1p(-clipped))),
            'reliability': reliability,
            'ece_10_bins': sum(b['weight']*abs(b['mean_probability']-b['voiced_fraction']) for b in reliability if b['weight'])}


def probability_diagnostics(records, predictions):
    result = {}
    for condition in CONDS:
        rows = [r for r in records if r.condition == condition]
        pbase, pcal, labels, weights, admission = [], [], [], [], []
        units = {r.singer if condition in POSITIVE else r.audio_hash for r in rows}
        for r in rows:
            unit = r.singer if condition in POSITIVE else r.audio_hash
            copies = sum((x.singer if condition in POSITIVE else x.audio_hash) == unit for x in rows)
            mask = r.known & r.matched
            i = r.indices[mask]
            pbase.extend(predictions[r.key]['base'][i])
            pcal.extend(predictions[r.key]['calibrated'][i])
            admission.extend(r.admissible[i])
            labels.extend(r.positive[mask].astype(float))
            weights.extend(np.full(len(i), 1 / len(units) / copies / len(i)))
        weights = np.asarray(weights)
        admission = np.asarray(admission, dtype=bool)
        result[condition] = {'base_before_admissibility': probability_statistics(pbase, labels, weights),
                             'platt_before_admissibility': probability_statistics(pcal, labels, weights),
                             'platt_after_physical_admissibility': probability_statistics(np.asarray(pcal)*admission, labels, weights),
                             'admissible_fraction': float(weights@admission/weights.sum()),
                             'unmatched_annotations': int(sum((r.known & ~r.matched).sum() for r in rows)),
                             'weighting': 'equal singers and excerpts for singing; equal unique rendered inputs with duplicate copies split for controls; equal annotation mass within each stream'}
    return result


def load_records(protocol, assets, runner):
    metadata = runner._find_one(assets, 'vocadito_metadata.csv')
    if runner._digest(metadata) != protocol['singer_metadata_sha256']:
        raise ValueError('singer metadata identity mismatch')
    with metadata.open() as stream:
        singers = {int(r['track_id']): r['singer_id'] for r in csv.DictReader(stream)}
    records, analytic = [], {}
    for input_report in protocol['input_reports']:
        path = ROOT / input_report['path']
        if runner._digest(path) != input_report['sha256']:
            raise ValueError('input report identity mismatch')
        old = json.loads(path.read_text())
        if old['errors']:
            raise ValueError('input report contains errors')
        for row in old[input_report['rows']]:
            group = input_report.get('group', row.get('group'))
            excerpt, condition = row['excerpt_id'], row['condition']
            if group not in protocol['groups'] or excerpt not in protocol['groups'][group]['excerpt_ids'] or condition not in CONDS:
                raise ValueError('non-consumed input is not allowed')
            a = row['artifact']
            if runner._digest(Path(a['artifact'])) != a['artifact_sha256']:
                raise ValueError('native artifact identity mismatch')
            feature = runner._load_feature(Path(a['artifact']), a['cache_key'])
            if feature is None:
                raise ValueError('invalid native artifact')
            if feature.metadata.get('model_revision') != NEURAL_REVISION:
                raise ValueError('native neural model identity mismatch')
            identity = [feature.metadata.get('code_version'), feature.metadata.get('params_hash')]
            if identity not in protocol['frontend']['accepted_native_identities']:
                raise ValueError('unverified native frontend')
            annotation = runner._find_one(assets, f'vocadito_{excerpt}_f0.csv')
            if runner._digest(annotation) != row['input_hashes']['reference_sha256']:
                raise ValueError('reference annotation identity mismatch')
            times, hz = _reference(assets, excerpt, _duration(feature))
            times, hz = runner._scoring_reference(condition, times, hz, feature, _duration(feature))
            family = ('silence' if condition == 'silence' else
                      f"backing:{row['input_hashes']['backing_sha256']}:{protocol['groups'][group]['backing_start_s']}" if condition == 'backing' else None)
            key = f'{group}:{excerpt}:{condition}'
            record = Record(key, group, singers[excerpt], excerpt, condition,
                            row['input_hashes']['rendered_16khz_mono_sha256'], family, feature, times, hz,
                            {'input_report': input_report['path'], 'artifact': a, 'input_hashes': row['input_hashes'], 'native_identity': identity})
            # The optimized scorer must reproduce the independently tested public scorer.
            current = feature.values[:, list(feature.coords['feature']).index('voiced')] > 0
            if record.score(current) != score_vocal_f0(times, hz, feature):
                raise ValueError('prepared scoring differs from original scorer')
            records.append(record)
    expected = {f'{g}:{i}:{c}' for g, spec in protocol['groups'].items() for i in spec['excerpt_ids'] for c in CONDS}
    if len(records) != 90 or {r.key for r in records} != expected or len({r.singer for r in records}) != 17:
        raise ValueError('consumed data must contain exactly90 rows and17 singers')
    for group, source in protocol['analytic_reports'].items():
        path = ROOT / source['path']
        if runner._digest(path) != source['sha256']:
            raise ValueError('analytic input report identity mismatch')
        old = json.loads(path.read_text())
        analytic[group] = {'report': str(path), 'sha256': runner._digest(path), 'result': old['analytic'],
                           'scope': 'unchanged pitch-engine controls; no learned singing-identity claim from analytic tones'}
    return records, analytic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT/'tests/benchmarks/thresholds/vocal_voicing_development_v7.json')
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--artifact-dir', type=Path, required=True)
    parser.add_argument('--out-json', type=Path, required=True)
    args = parser.parse_args()
    runner = _runner()
    if runner._digest(args.protocol) != PROTOCOL_SHA256:
        raise ValueError('protocol differs from the frozen v7 procedure')
    protocol = json.loads(args.protocol.read_text())
    if runner._digest(ROOT/'src/natural_features/features/audio/vocal_f0.py') != protocol['frontend']['source_sha256']:
        raise ValueError('frozen native frontend source changed')
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    report = {'schema': 'VocalVoicingDevelopment/v7', 'protocol_sha256': PROTOCOL_SHA256,
              'training_source_sha256': runner._digest(Path(__file__)),
              'postprocessor_source_sha256': runner._digest(ROOT/'src/natural_features/features/audio/vocal_voicing.py'),
              'native_source_sha256': protocol['frontend']['source_sha256'],
              'qualified': False, 'fresh_data_consumed': False, 'development_passed': False,
              'outer_folds': [], 'fit_trace': [], 'errors': []}
    try:
        records, analytic = load_records(protocol, args.assets, runner)
        report['dataset'] = [{'key': r.key, 'singer': r.singer, 'group': r.group, 'condition': r.condition,
                              'excerpt_id': r.excerpt, 'negative_family': r.family, 'rendered_audio_sha256': r.audio_hash,
                              'provenance': r.provenance} for r in records]
        report['analytic'] = analytic
        trainer = Trainer(protocol, report['fit_trace'])
        outer_metrics, predictions = {}, {}
        report['outer_artifacts'] = []
        for fold, validation in enumerate(grouped_folds(records, 5, outer=True)):
            training = purged_training(records, validation)
            selection = select(trainer, training, validation, f'outer-{fold}')
            model = trainer.pipeline(training, validation, f'outer-{fold}/selected')
            profile = {**selection['selected'], 'model': model, 'status': 'outer-fold development only'}
            fold_result = {'fold': fold, 'held_singers': sorted({r.singer for r in validation}, key=singer_order),
                           'validation_keys': [r.key for r in validation], 'training_keys': [r.key for r in training],
                           'selection': selection, 'profile': profile, 'per_excerpt': []}
            for r in validation:
                out = apply_voicing_model(r.feature, profile)
                voiced = out.values[:, list(out.coords['feature']).index('voiced')] > 0
                metrics = r.score(voiced)
                if metrics != score_vocal_f0(r.reference_times, r.reference_hz, out):
                    raise ValueError('OOF postprocessor/scorer disagreement')
                outer_metrics[r.key] = metrics
                base = {**model, 'calibration': {'slope': 1., 'intercept': 0.}}
                predictions[r.key] = {'base': predict_probability(r.x, base), 'calibrated': predict_probability(r.x, model)}
                identity = {'protocol': PROTOCOL_SHA256, 'training_source': report['training_source_sha256'],
                            'postprocessor_source': report['postprocessor_source_sha256'], 'input': r.provenance['artifact']['artifact_sha256'],
                            'row': r.key, 'fold': fold, 'profile': profile}
                key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
                path = args.artifact_dir / f'{key}.npz'
                runner._save_feature(path, key, out)
                report['outer_artifacts'].append({'key': r.key, 'fold': fold, 'artifact': str(path.resolve()),
                                                  'artifact_sha256': runner._digest(path), 'cache_key': key})
                fold_result['per_excerpt'].append({'key': r.key, 'metrics': metrics})
            report['outer_folds'].append(fold_result)
            runner._atomic_json(args.out_json, report)
            print(f"outer fold {fold}: singers={fold_result['held_singers']}, eligible selector={selection['eligible']}, profile={selection['selected']}", flush=True)
        if set(outer_metrics) != {r.key for r in records}:
            raise ValueError('outer OOF coverage incomplete')
        report['outer_groups'] = summarize(records, outer_metrics, protocol['gates'], complete=True)
        report['probability_diagnostics'] = probability_diagnostics(records, predictions)
        report['final_selection'] = select(trainer, records, [], 'final')
        final_model = trainer.pipeline(records, [], 'final/selected')
        final_profile = {**report['final_selection']['selected'], 'model': final_model,
                         'status': 'development only; requires independent qualification', 'protocol_sha256': PROTOCOL_SHA256}
        final_path = args.artifact_dir / 'final_profile.json'
        runner._atomic_json(final_path, final_profile)
        report['final_profile'] = {'path': str(final_path.resolve()), 'sha256': runner._digest(final_path), 'profile': final_profile}
        analytic_passed = all(a['result']['complete'] and a['result']['worst_median_abs_cents_error'] is not None
                              and a['result']['worst_median_abs_cents_error'] <= protocol['gates']['analytic_median_abs_cents_error_max']
                              for a in analytic.values())
        report['development_passed'] = (all(g['gate']['passed'] for g in report['outer_groups'])
                                         and report['final_selection']['eligible'] and analytic_passed)
        report['analytic_passed'] = analytic_passed
    except Exception as exc:
        report['errors'].append(runner._error_detail(exc))
        report['development_passed'] = False
    runner._atomic_json(args.out_json, report)
    print(json.dumps({'development_passed': report['development_passed'], 'errors': report['errors'],
                      'outer_groups': report.get('outer_groups'),
                      'final_selection': report.get('final_selection', {}).get('selected')}, indent=2))
    return 0 if report['development_passed'] and not report['errors'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
