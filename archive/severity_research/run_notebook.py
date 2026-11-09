"""Exécute le notebook et sauvegarde ses sorties, y compris en cas d'erreur."""
from pathlib import Path
import os
import sys
import argparse
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', '/tmp/gravite_mpl')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('JUPYTER_PLATFORM_DIRS', '1')
# Les caches du noyau sont temporaires; rien à écrire dans le profil utilisateur.
os.environ.setdefault('JUPYTER_RUNTIME_DIR', '/tmp/gravite_jupyter_runtime')
os.environ.setdefault('IPYTHONDIR', '/tmp/gravite_ipython')
PATH = ROOT / '01_gravite_xgboost_roadmap.ipynb'
nb = nbformat.read(PATH, as_version=4)
parser = argparse.ArgumentParser()
parser.add_argument('--section', type=int, choices=[9,10,11,12,13,14,15,16], help='Réexécuter seulement une section ajoutée, avec ses deux cellules de contexte.')
args = parser.parse_args()
if args.section is not None:
    start = next(i for i,c in enumerate(nb.cells) if c.cell_type == 'markdown' and c.source.startswith(f'## {args.section}.'))
    end = next((i for i in range(start+1,len(nb.cells)) if nb.cells[i].cell_type == 'markdown' and nb.cells[i].source.startswith('## ')), len(nb.cells))
    indices = [2,4] + list(range(start,end))
else:
    indices = list(range(len(nb.cells)))
selected_cells = set(indices)
for index,cell in enumerate(nb.cells):
    if index not in selected_cells:
        continue
    if cell.cell_type == 'code':
        cell.outputs = []
        cell.execution_count = None

def on_cell_start(cell, cell_index, **kwargs):
    if cell.cell_type == 'code':
        print(f'Cellule {cell_index+1}/{len(nb.cells)} : {cell.source.splitlines()[0][:100]}', flush=True)

def on_cell_executed(cell, cell_index, **kwargs):
    for output in cell.get('outputs', []):
        if output.get('output_type') == 'stream':
            lines = output.get('text', '').strip().splitlines()
            if lines:
                print('\n'.join(lines[-5:]), flush=True)

client = NotebookClient(nb, timeout=1800, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}},
                        on_cell_start=on_cell_start, on_cell_executed=on_cell_executed)
# Utiliser l'interpréteur courant, sans modifier de noyau global.
client.create_kernel_manager()
client.km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
try:
    if args.section is None:
        client.execute()
    else:
        with client.setup_kernel():
            for index in indices:
                if nb.cells[index].cell_type == 'code':
                    client.execute_cell(nb.cells[index], index)
finally:
    nbformat.write(nb, PATH)
print(f'Notebook exécuté : {PATH}', flush=True)
