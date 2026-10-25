"""Bounded CPU and native checks that require no historical weights or corpora."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
AZ = ROOT / 'alphazero'
MODULES = ['test_advantage_returns', 'test_advantage_behavior', 'test_fixed_rl',
           'test_open_v4', 'test_open_v4_runtime', 'test_open_v4_root_explore',
           'test_search_repairs', 'test_search_v3_dataset', 'test_search_v3_puct',
           'test_search_v3_recovery', 'test_search_v3_retention']


def main():
    checked = 0
    for path in ROOT.rglob('*.py'):
        if any(part in {'.git', '.venv', '__pycache__', 'build-native', 'runs'} for part in path.relative_to(ROOT).parts):
            continue
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        checked += 1
    for path in (AZ / 'config').rglob('*.json'):
        json.loads(path.read_text(encoding='utf-8-sig'))
    print(f'[source] {checked} Python files parsed; JSON configs parsed', flush=True)
    env = {**os.environ, 'PYTHONUTF8': '1', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
           'JUNQI_ROOT_EXPLORE_BUILD': 'build',
           'PYTHONPATH': os.pathsep.join(map(str, [ROOT, ROOT / 'build', AZ / 'build', AZ / 'python', AZ / 'tools', AZ / 'tests']))}
    subprocess.run([sys.executable, str(AZ / 'cpp/smoke_test.py')], cwd=AZ, env=env, check=True)
    subprocess.run([sys.executable, str(AZ / 'tests/native_conformance.py'), '--games', '12', '--plies', '80'], cwd=AZ, env=env, check=True)
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], cwd=ROOT, env=env, check=True)
    subprocess.run([sys.executable, '-m', 'unittest', '-v', *MODULES], cwd=AZ, env=env, check=True)
    print('[verified] source syntax, native rules, model/replay/state and search checks passed', flush=True)


if __name__ == '__main__':
    main()
