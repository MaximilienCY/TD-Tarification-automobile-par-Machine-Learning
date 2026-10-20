"""Run from any working directory: python /path/to/run.py search|nested|finalize|predict."""
from pathlib import Path
import os
import sys

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from pricing.experiment import main

if __name__ == '__main__':
    main()
