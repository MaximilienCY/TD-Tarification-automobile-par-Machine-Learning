"""Protéger la sélection du modèle principal et le lien avec le fichier évalué."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
from pricing import main_model as m
from pricing.experiment import PROTECTED, PROTECTED_LF_HASH, text_digest


def test_main_entry_from_another_directory(tmp_path):
    result = subprocess.run([sys.executable, str(m.ROOT / 'run.py'), 'info'], cwd=tmp_path,
                            text=True, capture_output=True, check=True)
    recipe = json.loads(result.stdout)
    assert recipe['name'] == 'frequency_curves_1365_657'
    assert recipe['frequency']['correction']['panel'] == 'age_coverage'
    assert sum(part['weight'] for part in recipe['severity'].values()) == 1.


@pytest.mark.parametrize('newline', [b'\n', b'\r\n'])
def test_reported_submission_accepts_only_newline_changes(tmp_path, monkeypatch, newline):
    data = m.RELEASE_PATH.read_bytes().replace(b'\r\n', b'\n').replace(b'\n', newline)
    path = tmp_path / 'submitted.csv'
    path.write_bytes(data)
    monkeypatch.setattr(m, 'RELEASE_PATH', path)
    assert len(m.release()) == 50000
    # Altérer réellement la première prime doit être détecté.
    lines = data.splitlines()
    identifier, _ = lines[1].split(b',')
    lines[1] = identifier + b',99999999.0'
    path.write_bytes(newline.join(lines) + newline)
    with pytest.raises(ValueError, match='contenu'):
        m.release()


def test_historical_notebook_content_preserved():
    assert PROTECTED.parent == m.ROOT / 'archive/notebooks'
    assert text_digest(PROTECTED) == PROTECTED_LF_HASH


def test_primary_csv_is_the_latest_submitted_candidate():
    assert (m.ROOT / 'submission.csv').read_bytes() == m.RELEASE_PATH.read_bytes()
    assert m.lf_sha256(m.RELEASE_PATH) == 'f502ade3f4b8645c4591eafd82a065d941f5299b1cb8fbacdbf380cccc18d1b6'
