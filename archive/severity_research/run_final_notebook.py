"""Exécuter uniquement la synthèse publiable du modèle final en 80/20."""
from pathlib import Path
import os
import sys

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
for key, value in {
    'MPLCONFIGDIR': '/tmp/gravite_mpl',
    'OMP_NUM_THREADS': '2',
    'OPENBLAS_NUM_THREADS': '2',
    'JUPYTER_PLATFORM_DIRS': '1',
    'JUPYTER_RUNTIME_DIR': '/tmp/gravite_jupyter_runtime',
    'IPYTHONDIR': '/tmp/gravite_ipython',
}.items():
    os.environ.setdefault(key, value)
path = ROOT/'00_modele_final_gravite.ipynb'
nb = nbformat.read(path, as_version=4)
client = NotebookClient(nb, timeout=1200, kernel_name='python3',
                        resources={'metadata': {'path': str(ROOT)}})
client.create_kernel_manager()
client.km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
try:
    client.execute()
finally:
    nbformat.write(nb, path)
print('Synthèse finale exécutée :', path)
