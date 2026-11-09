"""Exécuter le notebook de gravité avec l'interpréteur courant."""
from pathlib import Path
import argparse
import os
import sys

for name, value in {
    'MPLCONFIGDIR': '/tmp/gravite_analyse_mpl',
    'JUPYTER_RUNTIME_DIR': '/tmp/gravite_analyse_jupyter',
    'IPYTHONDIR': '/tmp/gravite_analyse_ipython',
    'JUPYTER_PLATFORM_DIRS': '1',
    'OMP_NUM_THREADS': '2',
    'OPENBLAS_NUM_THREADS': '2',
}.items():
    os.environ.setdefault(name, value)

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
PATH = HERE / 'notebook.ipynb'
notebook = nbformat.read(PATH, as_version=4)
nbformat.validate(notebook)
parser = argparse.ArgumentParser()
parser.add_argument('--in-process', action='store_true',
                    help='Exécuter avec IPython sans socket de noyau Jupyter.')
args = parser.parse_args()


def started(cell, cell_index, **kwargs):
    if cell.cell_type == 'code':
        print(f'Cellule {cell_index + 1}/{len(notebook.cells)}', flush=True)


if args.in_process:
    # Alternative pour les environnements qui interdisent les sockets locales.
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    shell = InteractiveShell.instance()
    os.chdir(HERE)
    count = 0
    try:
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != 'code':
                continue
            started(cell, index)
            count += 1
            cell.outputs = []
            cell.execution_count = count
            with capture_output() as captured:
                result = shell.run_cell(cell.source, store_history=False)
            if captured.stdout:
                cell.outputs.append(nbformat.v4.new_output('stream', name='stdout', text=captured.stdout))
            if captured.stderr:
                cell.outputs.append(nbformat.v4.new_output('stream', name='stderr', text=captured.stderr))
            for output in captured.outputs:
                cell.outputs.append(nbformat.v4.new_output('display_data', data=output.data,
                                                          metadata=output.metadata))
            error = result.error_before_exec or result.error_in_exec
            if error is not None:
                cell.outputs.append(nbformat.v4.new_output('error', ename=type(error).__name__,
                                                          evalue=str(error), traceback=[]))
                raise error
    finally:
        nbformat.write(notebook, PATH)
else:
    client = NotebookClient(notebook, timeout=600, kernel_name='python3',
                            resources={'metadata': {'path': str(HERE)}}, on_cell_start=started)
    client.create_kernel_manager()
    client.km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    try:
        client.execute()
    finally:
        nbformat.write(notebook, PATH)
print(f'Notebook exécuté : {PATH}', flush=True)
