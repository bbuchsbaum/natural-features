#!/usr/bin/env python3
"""Prepare a pinned, consumed NUS-48E development pilot without recognition."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import soundfile as sf

from natural_features.core.stimulus import AudioStimulus
from natural_features.features.audio._music_contract import asset_digest
from natural_features.features.speech.phonology import _normalize_ctc_token
from compare_vocal_backends import canonical_and_stem
from qualify_vocal_f0 import _read_audio, _to_16k_mono, _mixture

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def checked_download(assets, entry, download=False):
    path = assets / entry['path']
    if not path.exists() and download:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + '.partial')
        urllib.request.urlretrieve(entry['url'], temp)
        if sha(temp) != entry['sha256']:
            raise ValueError('download identity mismatch: ' + entry['path'])
        temp.replace(path)
    if not path.is_file() or sha(path) != entry['sha256']:
        raise ValueError('missing or changed asset: ' + entry['path'])
    return path


def references(path, duration):
    """Preserve original boundaries, clipping only to the frozen input interval."""
    result = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        lo, hi, phone = line.split()
        lo, hi = float(lo), float(hi)
        if lo >= duration or hi < 0:
            continue
        lo, hi = max(0., lo), min(duration, hi)
        role = 'nonphone' if phone in ('sil', 'sp') else 'phone'
        # Zero-length word markers carry no duration or recognition target.
        if hi == lo and role == 'nonphone':
            continue
        result.append({'start_s': lo, 'end_s': hi, 'phone': phone, 'role': role})
    return result


def token_contract(snapshot, mapping):
    """Derive every raw output-column role before observing model outputs."""
    config = json.loads((snapshot / 'config.json').read_text())
    tokenizer = json.loads((snapshot / 'tokenizer_config.json').read_text())
    vocab = json.loads((snapshot / 'vocab.json').read_text())
    added = snapshot / 'added_tokens.json'
    if added.exists():
        vocab.update(json.loads(added.read_text()))
    by_id = {i: token for token, i in vocab.items()}
    roles, inventory = {}, {}
    for i in range(config['vocab_size']):
        raw = by_id[i]
        token = _normalize_ctc_token(raw) or raw
        if token in roles:
            raise ValueError('ambiguous normalized vocabulary: ' + token)
        if i == config['pad_token_id']:
            role = 'blank'
        elif raw == tokenizer.get('word_delimiter_token') or token == 'word_boundary':
            role = 'boundary'
        elif raw in ('<s>', '</s>', '<pad>'):
            role = 'special'
        else:
            role = 'phone'
        roles[token] = role
        # Explicit stress removal only; unmatched multilingual tokens stay unknown.
        normalized = token.replace('ˈ', '').replace('ˌ', '')
        if role == 'phone' and normalized in mapping:
            inventory[token] = mapping[normalized]
    return roles, inventory


def prepare(protocol_path, assets, backing_assets, separator, out_dir, download=False):
    protocol = json.loads(protocol_path.read_text())
    if protocol['state'] != 'development' or protocol['qualification_allowed'] is not False:
        raise ValueError('this pilot is consumed-data development only')
    for name, expected in protocol['frozen_sources'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('frozen source changed: ' + name)
    for entry in protocol['assets']:
        checked_download(assets, entry, download)
    if sha(separator) != protocol['separator_sha256']:
        raise ValueError('separator checkpoint identity mismatch')
    for name, expected in protocol['backing_sha256'].items():
        if sha(backing_assets / name) != expected:
            raise ValueError('backing identity mismatch: ' + name)
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = protocol['duration_s']
    models = {}
    for name, model in protocol['models'].items():
        snapshot = assets / model['path']
        identity = asset_digest(snapshot)
        if identity != model['snapshot_digest']:
            raise ValueError('snapshot identity mismatch: ' + name)
        roles, mapping = token_contract(snapshot, protocol['ipa_to_reference'])
        models[name] = (snapshot, identity, roles, mapping)
    items, prepared = [], []
    for pair in protocol['pairs']:
        for mode in ('read', 'sing'):
            prefix = f"nus/{pair['singer']}/{mode}/{pair['song']}"
            wav_path, reference_path = assets / (prefix + '.wav'), assets / (prefix + '.txt')
            audio, sr = _read_audio(wav_path)
            audio = _to_16k_mono(audio, sr, duration)
            if len(audio) != round(duration * 16000):
                raise ValueError('input shorter than frozen duration')
            refs = references(reference_path, duration)
            provenance = {'source_audio_sha256': sha(wav_path), 'annotation_sha256': sha(reference_path),
                'protocol_sha256': sha(protocol_path), 'crop_s': [0, duration],
                'reference_quality': 'manually annotated singing' if mode == 'sing' else 'automatically aligned read speech; diagnostic comparator',
                'sil_policy': 'sil combines silence/inhalation; excluded from phone targets, not used as vocal absence',
                'sp_policy': 'word boundary; excluded', 'source_input': prefix,
                'no_pitch_gate': True}
            conditions = {mode + '_clean': audio}
            rendering = {}
            if mode == 'sing':
                backing_path = backing_assets / pair['backing']
                backing, sr = _read_audio(backing_path)
                backing = _to_16k_mono(backing, sr, duration)
                mask = np.zeros(len(audio), bool)
                for ref in refs:
                    if ref['role'] == 'phone':
                        mask[int(np.ceil(ref['start_s'] * 16000)):int(np.ceil(ref['end_s'] * 16000))] = True
                mixture, rendering = _mixture(audio, backing, mask, 0.)
                rendering.update({'mask': 'all positive-duration reference phones, including unvoiced consonants',
                                  'backing': pair['backing'], 'backing_sha256': sha(backing_path)})
                baseline, stem = canonical_and_stem(AudioStimulus.from_array(mixture, 16000), str(separator))
                conditions.update({'sing_mixture_0db': mixture, 'sing_stem_0db': stem})
                provenance['separation_metadata'] = baseline.metadata
            for condition, waveform in conditions.items():
                item_id = f"{pair['singer']}_{pair['song']}_{condition}"
                output = out_dir / (item_id + '.wav')
                # Float WAV preserves the computed waveform without PCM clipping.
                sf.write(output, waveform, 16000, subtype='FLOAT')
                prov = {**provenance, 'rendering': rendering if condition != 'sing_clean' else {}}
                prepared.append({'id': item_id, 'path': str(output), 'sha256': sha(output), 'provenance': prov})
                for model_name, (snapshot, identity, roles, mapping) in models.items():
                    items.append({'id': item_id + '_' + model_name, 'model': model_name,
                        'condition': condition, 'singer': pair['singer'], 'song': pair['song'],
                        'partition': 'consumed_development', 'audio_path': str(output), 'audio_sha256': sha(output),
                        'model_snapshot': str(snapshot), 'model_snapshot_sha256': identity, 'device': 'cpu',
                        'token_roles': roles, 'canonical_inventory': mapping, 'reference_intervals': refs,
                        'window_bounds_s': protocol['window_bounds_s'], 'bin_bounds_s': protocol['bin_bounds_s'],
                        'provenance': {**prov, 'model': protocol['models'][model_name]}})
            print(f"prepared {pair['singer']} {mode}", flush=True)
    manifest = {'manifest_version': 1, 'state': 'development', 'token_roles': {}, 'canonical_inventory': {},
                'protocol': protocol, 'protocol_sha256': sha(protocol_path), 'items': items}
    path = out_dir / 'manifest.json'
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    (out_dir / 'preparation-receipt.json').write_text(json.dumps(prepared, indent=2) + '\n')
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', type=Path, default=ROOT / 'tests/benchmarks/thresholds/singing_phone_pilot_v1.json')
    ap.add_argument('--assets', type=Path, required=True)
    ap.add_argument('--backing-assets', type=Path, required=True)
    ap.add_argument('--separator-checkpoint', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--download', action='store_true', help='Download missing pinned public files; NUS research-only terms apply.')
    args = ap.parse_args()
    import torch
    torch.set_num_threads(1)
    print(prepare(args.protocol, args.assets.resolve(), args.backing_assets.resolve(), args.separator_checkpoint.resolve(), args.out_dir.resolve(), args.download))


if __name__ == '__main__':
    main()
