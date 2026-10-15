import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('singing_pilot', SCRIPTS / 'prepare_singing_phone_pilot.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_roles_keep_blank_boundary_and_unknown_distinct(tmp_path):
    for name, data in {
        'config.json': {'vocab_size': 6, 'pad_token_id': 0},
        'tokenizer_config.json': {'word_delimiter_token': '|'},
        'vocab.json': {'<pad>': 0, '|': 1, '<unk>': 2, '<s>': 3, 'ˈaɪ': 4, 'foreign': 5},
    }.items():
        (tmp_path / name).write_text(json.dumps(data))
    roles, mapping = pilot.token_contract(tmp_path, {'aɪ': ['ay']})
    assert roles == {'<pad>': 'blank', 'word_boundary': 'boundary', '<unk>': 'phone', '<s>': 'special', 'ˈaɪ': 'phone', 'foreign': 'phone'}
    assert mapping == {'ˈaɪ': ['ay']}


def test_reference_crop_preserves_unvoiced_phones_and_roles(tmp_path):
    path = tmp_path / 'annotations.txt'
    path.write_text('0 1 sil\n1 2 s\n1 1 sp\n2 4 aa\n4 5 t\n')
    result = pilot.references(path, 3)
    assert result == [
        {'start_s': 0., 'end_s': 1., 'phone': 'sil', 'role': 'nonphone'},
        {'start_s': 1., 'end_s': 2., 'phone': 's', 'role': 'phone'},
        {'start_s': 2., 'end_s': 3., 'phone': 'aa', 'role': 'phone'},
    ]


def test_asset_hash_guard(tmp_path):
    p = tmp_path / 'file'
    p.write_text('known data')
    entry = {'path': 'file', 'sha256': pilot.sha(p), 'url': 'unused'}
    assert pilot.checked_download(tmp_path, entry) == p
    p.write_text('changed')
    with pytest.raises(ValueError, match='changed asset'):
        pilot.checked_download(tmp_path, entry)
