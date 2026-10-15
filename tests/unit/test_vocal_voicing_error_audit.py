import importlib
import hashlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'scripts'))
    return importlib.import_module('audit_vocal_voicing_errors')


def test_boundary_distance_ignores_unknowns_and_exposes_censoring(audit):
    times = np.arange(5)*.1
    hz = np.array([0., np.nan, 200., 0., 0.])
    distance, previous, missing_before, missing_after = audit.nearest_opposite(times, hz, False)
    np.testing.assert_allclose(distance, [.2, .1, 0., .1, .2])
    assert missing_before[:3].all() and missing_after[3:].all()
    assert not previous[:3].any() and previous[3:].all()
    no_voice = audit.nearest_opposite(times, np.zeros(5), False)
    assert np.isinf(no_voice[0]).all() and no_voice[2].all() and no_voice[3].all()


def test_run_duration_and_unknown_breaks(audit):
    actual = audit.runs(np.array([True, True, False, True, True]), np.arange(5)*.01)
    assert [(a, b) for a, b, _ in actual] == [(0, 2), (3, 5)]
    np.testing.assert_allclose([duration for _, _, duration in actual], [.02, .02])
    separated = audit.runs(np.ones(4, bool), np.array([0., .01, 1., 1.01]))
    assert [(a, b) for a, b, _ in separated] == [(0, 2), (2, 4)]
    np.testing.assert_allclose([duration for _, _, duration in separated], [.02, .02])


def test_quantiles_never_replace_missing_measurements_with_zero(audit):
    result = audit.quantiles([np.nan, 1., 3.])
    assert result['count'] == 3 and result['finite_count'] == 2 and result['median'] == 2.
    assert audit.quantiles([np.nan])['median'] is None


def test_audit_retains_paired_errors_and_hash_bound_context_artifacts(audit, monkeypatch, tmp_path):
    pytest.importorskip('matplotlib')
    sf = pytest.importorskip('soundfile')
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    native = importlib.import_module('test_vocal_voicing').native
    v7 = importlib.import_module('calibrate_vocal_voicing')
    source = tmp_path/'vocadito_1.wav'
    sf.write(source, np.sin(2*np.pi*200*np.arange(16000)/16000)*.1, 16000, subtype='FLOAT')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    records, inputs = [], {}
    for condition in v7.POSITIVE:
        feature = native(n=60, offset=0)
        hz = np.where(feature.times_s < .1, 200., 0.)
        r = v7.Record(condition, 'group', 'S1', 1, condition, condition, None, feature, feature.times_s, hz,
                      {'input_hashes': {'source_audio_sha256': digest}})
        records.append(r)
        voice = np.ones(60, bool)
        inputs[r.key] = {'probability': np.full(60, .99), 'voiced': voice, 'threshold': .8,
                         'comparator_baseline': voice, 'v7_baseline': voice}
    result = audit.audit_errors(records, inputs, tmp_path, tmp_path/'out', {'error_audit': {}})
    assert result['singing_streams'] == 3 and len(result['examples']) == 3
    assert result['summary_counts']['v8_selected_or_fallback']['solo']['false_positive'] == 50
    assert all(row['mixture_only_fp'] == 0 for row in result['paired_solo_mixture'])
    for example in result['examples']:
        assert Path(example['figure']).exists()
        assert hashlib.sha256(Path(example['original_solo_snippet']).read_bytes()).hexdigest() == example['snippet_sha256']
