"""Build the source distribution with the selected Python, without training."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compiler', help='C++ compiler path/name, e.g. g++, clang++, or cl')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--skip-teachers', action='store_true')
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    projects = [('platform', ROOT / 'cpp', ROOT / 'build'),
                ('alphazero', ROOT / 'alphazero/cpp', ROOT / 'alphazero/build')]
    if not args.skip_teachers:
        projects.extend((f'teacher_{name}', ROOT / 'teachers' / name, ROOT / 'teachers' / name) for name in ['082', '084'])
    for name, source, output in projects:
        build = ROOT / 'build-native' / name
        command = ['cmake', '-S', str(source), '-B', str(build), '-G', 'Ninja',
                   '-DCMAKE_BUILD_TYPE=Release', f'-DPython_EXECUTABLE={sys.executable}',
                   f'-DJUNQI_OUTPUT_DIR={output}', f'-DJUNQI_AZ_OUTPUT_DIR={output}']
        if args.compiler:
            command.append(f'-DCMAKE_CXX_COMPILER={args.compiler}')
        print(f'[build] {name}', flush=True)
        subprocess.run(command, check=True)
        subprocess.run(['cmake', '--build', str(build), '--parallel', str(args.jobs)], check=True)


if __name__ == '__main__':
    main()
