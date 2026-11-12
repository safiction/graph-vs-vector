"""Run from the repository root: python scripts/run_baselines.py --help."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from baseline_rag.__main__ import main

if __name__ == '__main__':
    main()
