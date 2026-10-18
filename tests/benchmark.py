#!/usr/bin/env python3
"""Reproducible native benchmark; fixture generation is excluded from timing.

Example: python3 tests/benchmark.py build/sci --bins 1000 --samples 25
Only the SCI subprocess is measured using wait4 (wall time and peak RSS).
"""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
from hic_fixture import v9


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('binary', type=Path)
    p.add_argument('--bins', type=int, default=512, help='bins per chromosome, maximum 32767')
    p.add_argument('--samples', type=float, default=1, help='millions of samples per order')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--order', choices=['1', '2', 'both'], default='1')
    p.add_argument('--keep', type=Path, help='retain fixture, outputs and timings in this directory')
    args = p.parse_args()
    assert 10 <= args.bins <= 32767
    with tempfile.TemporaryDirectory() as temp:
        root = args.keep or Path(temp)
        root.mkdir(parents=True, exist_ok=True)
        n = args.bins
        chroms = [(str(i+1), n*100000) for i in range(3)]
        # Every cell is present: exercises graph RAM and full pair sweeps.
        pairs = {(a, b): [(x, y, 1000 if x*5//n == y*5//n else 2)
                           for x in range(n) for y in range(n)]
                 for a in range(3) for b in range(a+1, 3)}
        hic = root/'benchmark.hic'
        v9(hic, chroms, pairs, resolution=100000)
        del pairs
        command = [str(args.binary.resolve()), '-f', str(hic), '-n', str(root/'benchmark'), '-r', '100000',
                   '--normalization', 'NONE', '-s', str(args.samples), '-t', str(args.threads), '-o', args.order]
        start = time.perf_counter()
        with (root/'timing.txt').open('w') as log:
            proc = subprocess.Popen(command, stdout=log, stderr=log)
            _, status, usage = os.wait4(proc.pid, 0)
            proc.returncode = os.waitstatus_to_exitcode(status)
        elapsed = time.perf_counter() - start
        assert proc.returncode == 0, (root/'timing.txt').read_text()
        summary = json.loads((root/'benchmark_summary.json').read_text())
        summary['subprocess_wall_seconds'] = elapsed
        summary['peak_rss_bytes'] = usage.ru_maxrss * (1 if platform.system() == 'Darwin' else 1024)
        summary['user_seconds'] = usage.ru_utime
        summary['system_seconds'] = usage.ru_stime
        (root/'benchmark_measurements.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(json.dumps(summary, indent=2))
        print((root/'timing.txt').read_text())


if __name__ == '__main__':
    main()
