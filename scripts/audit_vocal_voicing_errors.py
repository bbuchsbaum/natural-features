#!/usr/bin/env python3
"""Descriptive, post-selection V8 error audit. Never changes labels or decisions."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from natural_features.features.audio.vocal_voicing_threshold import threshold_mask


BINS = ('within_50ms', '50_to_100ms', '100_to_200ms', 'beyond_200ms')


def nearest_opposite(times, reference_hz, target_voiced):
    """Distance to observed opposite-state annotations; edge censoring explicit."""
    times = np.asarray(times)
    reference_hz = np.asarray(reference_hz)
    opposite = reference_hz == 0 if target_voiced else reference_hz > 0
    neighbors = times[opposite]
    if not len(neighbors):
        return np.full(len(times), np.inf), np.ones(len(times), bool), np.ones(len(times), bool), np.ones(len(times), bool)
    pos = np.searchsorted(neighbors, times, side='left')
    missing_before, missing_after = pos == 0, pos == len(neighbors)
    before = np.where(missing_before, np.inf, times-neighbors[np.clip(pos-1, 0, len(neighbors)-1)])
    after = np.where(missing_after, np.inf, neighbors[np.clip(pos, 0, len(neighbors)-1)]-times)
    return np.minimum(before, after), before <= after, missing_before, missing_after


def annotation_mask(record, native_mask):
    result = np.zeros(len(record.reference_times), dtype=bool)
    result[record.matched] = native_mask[record.indices[record.matched]]
    return result


def runs(mask, times):
    """Split unknown states and gaps >1.5 median positive annotation steps."""
    mask, times = np.asarray(mask, bool), np.asarray(times, float)
    if mask.shape != times.shape or mask.ndim != 1 or np.any(np.diff(times) < 0):
        raise ValueError('run masks require matching ordered timestamps')
    gaps = np.diff(times)
    step = float(np.median(gaps[gaps > 0])) if np.any(gaps > 0) else 0.
    linked = (gaps <= 1.5*step) & mask[:-1] & mask[1:]
    starts = np.flatnonzero(mask & ~np.r_[False, linked])
    ends = np.flatnonzero(mask & ~np.r_[linked, False])+1
    return [(int(a), int(b), float(times[b-1]-times[a]+step))
            for a, b in zip(starts, ends)]


def quantiles(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return {'count': len(values), 'finite_count': len(finite),
            'q10': None if not len(finite) else float(np.quantile(finite, .1)),
            'median': None if not len(finite) else float(np.median(finite)),
            'q90': None if not len(finite) else float(np.quantile(finite, .9))}


def _details(record, prediction, probability):
    predicted = annotation_mask(record, prediction)
    fp = predicted & record.negative
    fn = ~predicted & record.positive
    tp = predicted & record.positive
    distance_fp, previous_voice, missing_prev_fp, missing_next_fp = nearest_opposite(record.reference_times, record.reference_hz, False)
    distance_fn, _, missing_prev_fn, missing_next_fn = nearest_opposite(record.reference_times, record.reference_hz, True)
    class_masks = {'true_positive': tp, 'false_positive': fp, 'false_negative': fn}
    near = fp & (distance_fp <= .2)
    bins_fp = np.searchsorted([.05, .1, .2], distance_fp, side='left')
    bins_fn = np.searchsorted([.05, .1, .2], distance_fn, side='left')
    p_reference = np.full(len(predicted), np.nan)
    p_reference[record.matched] = probability[record.indices[record.matched]]
    counts = {'false_positive': int(fp.sum()), 'false_negative': int(fn.sum()), 'true_positive': int(tp.sum()),
              'reference_positive': int(record.positive.sum()), 'reference_negative': int(record.negative.sum()),
              'unknown_annotations': int((~record.known).sum()), 'unmatched_known_annotations': int((record.known & ~record.matched).sum()),
              'fp_boundary_bins': {name: int(np.count_nonzero(fp & (bins_fp == b))) for b, name in enumerate(BINS)},
              'fn_boundary_bins': {name: int(np.count_nonzero(fn & (bins_fn == b))) for b, name in enumerate(BINS)},
              'fp_post_offset_near': int((near & previous_voice).sum()),
              'fp_pre_onset_near': int((near & ~previous_voice).sum()),
              'fp_interior_rest': int((fp & (distance_fp > .2)).sum()),
              'fp_interior_probability_ge_point9': int((fp & (distance_fp > .2) & (p_reference >= .9)).sum()),
              'fp_missing_previous_voiced_neighbor': int((fp & missing_prev_fp).sum()),
              'fp_missing_next_voiced_neighbor': int((fp & missing_next_fp).sum()),
              'fn_missing_previous_unvoiced_neighbor': int((fn & missing_prev_fn).sum()),
              'fn_missing_next_unvoiced_neighbor': int((fn & missing_next_fn).sum())}
    names = list(record.feature.coords['feature'])
    native_predictors = {'probability': probability}
    for name in ['confidence', 'source_support', 'source_rms', 'local_periodicity']:
        native_predictors[name] = record.feature.values[:, names.index(name)]
    distributions = {}
    for label, mask in class_masks.items():
        matched = mask & record.matched
        distributions[label] = {name: values[record.indices[matched]].tolist() for name, values in native_predictors.items()}
    return counts, distributions, fp


def _add_counts(target, source):
    for k, value in source.items():
        if isinstance(value, dict):
            _add_counts(target.setdefault(k, {}), value)
        else:
            target[k] = target.get(k, 0) + value


def audit_errors(records, inputs, assets, folder, protocol):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import soundfile as sf
    from calibrate_vocal_f0 import _runner
    runner = _runner()
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    singing = [r for r in records if r.condition in ('solo', 'mixture_0db', 'mixture_minus6db')]
    masks, fp_masks, rows, candidates = {}, {}, [], []
    summary, distributions = {}, {}
    for record in singing:
        data = inputs[record.key]
        probability = data['probability']
        variants = {'v8_selected_or_fallback': data['voiced'], 'fixed_point50': data['comparator_baseline'],
                    'fixed_point75': threshold_mask(probability, record.admissible, .75)}
        masks[record.key] = variants
        for variant, mask in variants.items():
            counts, values, fp = _details(record, mask, probability)
            fp_masks[(record.key, variant)] = fp
            rows.append({'key': record.key, 'singer': record.singer, 'group': record.group, 'condition': record.condition,
                         'variant': variant, 'threshold': data['threshold'] if variant == 'v8_selected_or_fallback' else .5 if variant == 'fixed_point50' else .75,
                         'counts': counts, 'metrics': record.score(mask),
                         'predictor_distributions': {label: {name: quantiles(v) for name, v in vs.items()} for label, vs in values.items()}})
            _add_counts(summary.setdefault(variant, {}).setdefault(record.condition, {}), counts)
            pooled = distributions.setdefault(variant, {}).setdefault(record.condition, {})
            for label, vs in values.items():
                for name, v in vs.items():
                    pooled.setdefault(label, {}).setdefault(name, []).extend(v)
            if variant == 'v8_selected_or_fallback':
                for start, end, duration in runs(fp, record.reference_times):
                    if duration >= .2:
                        candidates.append({'key': record.key, 'singer': record.singer, 'condition': record.condition,
                                           'start_s': float(record.reference_times[start]), 'end_s': float(record.reference_times[end-1]),
                                           'duration_s': duration, 'annotation_count': end-start})
    for by_condition in distributions.values():
        for by_class in by_condition.values():
            for by_predictor in by_class.values():
                for name, values in by_predictor.items():
                    by_predictor[name] = quantiles(values)
    paired = []
    for record in singing:
        if record.condition == 'solo':
            continue
        solo = next(r for r in singing if r.excerpt == record.excerpt and r.group == record.group and r.condition == 'solo')
        if not np.array_equal(solo.reference_times, record.reference_times) or not np.array_equal(solo.reference_hz, record.reference_hz, equal_nan=True):
            raise ValueError('paired conditions have different references')
        for variant in masks[record.key]:
            a, b = fp_masks[(solo.key, variant)], fp_masks[(record.key, variant)]
            paired.append({'key': record.key, 'variant': variant, 'both_solo_and_mixture_fp': int((a & b).sum()),
                           'mixture_only_fp': int((b & ~a).sum()), 'solo_only_fp': int((a & ~b).sum()),
                           'known_reference_unvoiced': int(record.negative.sum())})
    selected = sorted(candidates, key=lambda c: (-c['duration_s'], c['key'], c['start_s']))[:6]
    examples = []
    for number, event in enumerate(selected):
        record = next(r for r in singing if r.key == event['key'])
        companions = [r for r in singing if r.group == record.group and r.excerpt == record.excerpt]
        lo, hi = max(0., event['start_s']-.3), min(float(record.feature.metadata['computation_context']['input_bounds_s'][1]), event['end_s']+.3)
        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, layout='constrained')
        rt = record.reference_times
        ref_keep = (rt >= lo) & (rt <= hi)
        axes[0].plot(rt[ref_keep], np.where(record.reference_hz[ref_keep] > 0, record.reference_hz[ref_keep], np.nan), color='black', linewidth=2, label='Reference F0')
        for companion in companions:
            times = companion.feature.times_s
            keep = (times >= lo) & (times <= hi)
            probability = inputs[companion.key]['probability']
            voice = inputs[companion.key]['voiced']
            axes[0].plot(times[keep], np.where(voice[keep], companion.candidate[keep], np.nan), '.', markersize=3, label=companion.condition)
            axes[1].plot(times[keep], probability[keep], label=companion.condition)
        axes[1].axhline(inputs[record.key]['threshold'], color='black', linestyle='--', linewidth=.8, label='Selected threshold')
        source = runner._find_one(assets, f'vocadito_{record.excerpt}.wav')
        if runner._digest(source) != record.provenance['input_hashes']['source_audio_sha256']:
            raise ValueError('audit solo waveform identity mismatch')
        with sf.SoundFile(source) as stream:
            sample_rate = stream.samplerate
            a, b = int(lo*sample_rate), min(len(stream), int(hi*sample_rate))
            stream.seek(a)
            waveform = stream.read(b-a, always_2d=True)
        snippet = folder/f'example-{number+1}-solo.wav'
        sf.write(snippet, waveform, sample_rate, subtype='FLOAT')
        axes[2].plot(np.arange(a, b)/sample_rate, waveform.mean(axis=1), linewidth=.5, color='black')
        for axis in axes:
            axis.axvspan(event['start_s'], event['end_s'], color='red', alpha=.1)
        axes[0].set_ylabel('F0 (Hz)')
        axes[1].set_ylabel('Voicing probability')
        axes[2].set_ylabel('Original solo wave')
        axes[2].set_xlabel('Time (s)')
        axes[0].legend(fontsize=7, loc='upper right')
        axes[1].legend(fontsize=7, loc='upper right')
        axes[0].set_title(f"{record.key}: reference-unvoiced false acceptance")
        path = folder/f'example-{number+1}.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        examples.append({**event, 'context_start_s': lo, 'context_end_s': hi,
                         'figure': str(path.resolve()), 'figure_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'original_solo_snippet': str(snippet.resolve()), 'snippet_sha256': hashlib.sha256(snippet.read_bytes()).hexdigest(),
                         'source_audio_sha256': runner._digest(source), 'interpretation': 'Descriptive waveform/prediction context only; no auditory cause or annotation verdict assigned.'})
    return {'schema': 'VocalVoicingErrorAudit/v8', 'qualified': False, 'fresh_data_consumed': False,
            'audit_specification': protocol['error_audit'], 'singing_streams': len(singing),
            'summary_counts': summary, 'pooled_predictor_distributions': distributions,
            'per_excerpt': rows, 'paired_solo_mixture': paired, 'long_fp_runs_count': len(candidates), 'examples': examples,
            'aggregation': 'Counts pool correlated annotation frames descriptively; qualification gates remain equal-excerpt macro means.',
            'limitations': ['No auditory assessment or relabeling.', 'Distances are to observed opposite-state annotations, not exact acoustic onsets/offsets.',
                           'Missing previous/next opposite state explicitly records clip censoring.', 'Source persistence and paired errors do not establish instrumental leakage or reference error.']}
