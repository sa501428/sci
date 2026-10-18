#!/usr/bin/env python3
"""End-to-end .hic tests with a dense Python graph oracle and independent writers."""
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from itertools import permutations

sys.dont_write_bytecode = True
from hic_fixture import v9, v10


def call(binary, root, hic, name, *args, ok=True, norm='NONE', resolution=10, samples=2000):
    command = [str(binary), '-f', str(hic), '-n', str(root/name), '-r', str(resolution),
               '--sample-count', str(samples), '--dimensions', '8', '-k', '2', '--restarts', '2']
    if norm is not None:
        command += ['--normalization', norm]
    proc = subprocess.run(command + list(map(str, args)), text=True, capture_output=True, timeout=30)
    assert (proc.returncode == 0) == ok, (command, args, proc.stdout, proc.stderr)
    return proc


def graph_oracle(chroms, pairs, resolution=10, norms=None):
    offsets = []
    offset = 0
    for _, length in chroms:
        offsets.append(offset)
        offset += math.ceil(length / resolution)
    edges = {}
    for (a, b), points in pairs.items():
        # Deliberately aggregate into a dense matrix, as Python SCI does.
        matrix = [[0.] * math.ceil(chroms[b][1]/resolution)
                  for _ in range(math.ceil(chroms[a][1]/resolution))]
        for x, y, value in points:
            matrix[x][y] += value
        for x, row in enumerate(matrix):
            for y, value in enumerate(row):
                if norms:
                    nx, ny = norms[a][x], norms[b][y]
                    if not math.isfinite(nx) or not math.isfinite(ny) or nx <= 0 or ny <= 0:
                        continue
                    value /= nx * ny
                if value > 1 and math.isfinite(value):
                    edges[(offsets[a]+x, offsets[b]+y)] = math.log(value)
    return edges


def read_graph(path):
    return {(int(a), int(b)): float(w) for a, b, w in
            (line.split() for line in path.read_text().splitlines())}


def compare_edges(actual, expected):
    assert actual.keys() == expected.keys(), (actual, expected)
    for key in actual:
        assert math.isclose(actual[key], expected[key], rel_tol=2e-7), (key, actual[key], expected[key])


def inspect(root, name, chroms, dimensions):
    summary = json.loads((root/f'{name}_summary.json').read_text())
    assert summary['dimensions'] == dimensions
    assert math.isfinite(summary['inertia']) and summary['inertia'] >= 0
    bed = [line.split('\t') for line in (root/f'{name}_compartments.bed').read_text().splitlines()]
    assert len(bed) == sum(math.ceil(length/summary['resolution']) for _, length in chroms)
    lengths = dict(chroms)
    assert all(0 <= int(start) < int(end) <= lengths[chrom] for chrom, start, end, _ in bed)
    assert all(label in ('NA', '0', '1') for _, _, _, label in bed)
    assert sum(row[3] != 'NA' for row in bed) == summary['active_bins']
    table = (root/f'{name}_compartments.txt').read_text().splitlines()
    assert table[0] == 'chr\tstart\tend\tcompartment'
    assert table[1:] == (root/f'{name}_compartments.bed').read_text().splitlines()
    return summary, bed


def embeddings(path):
    lines = path.read_text().splitlines()
    n, dim = map(int, lines[0].split())
    rows = {int(p[0]): list(map(float, p[1:])) for p in map(str.split, lines[1:])}
    assert len(rows) == n and all(len(row) == dim for row in rows.values())
    assert all(math.isfinite(v) for row in rows.values() for v in row)
    return rows


def main():
    binary = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        chroms = [('chrA', 40), ('chrB', 35), ('chrC', 20)]
        pairs = {(0, 1): [(0, 0, 12), (0, 1, 1), (1, 0, 6), (1, 1, 24), (2, 2, 48), (3, 3, 1)],
                 (0, 2): [(0, 0, 8), (1, 0, 16)],
                 (1, 2): [(0, 0, 16), (1, 0, 8), (2, 0, 12)]}
        vectors = [[2, 2, float('nan'), 2], [3]*4, [2, 0]]
        hic = root/'test.hic'
        v9(hic, chroms, pairs, norms={'GW_KR': vectors})
        call(binary, root, hic, 'normalized', '--dump-graph', root/'normalized.graph', norm=None)
        compare_edges(read_graph(root/'normalized.graph'), graph_oracle(chroms, pairs, norms=vectors))
        summary, bed = inspect(root, 'normalized', chroms, 8)
        assert summary['normalization'] == 'GW_KR' and summary['active_bins'] == 6
        assert summary['total_bins'] == 10 and bed[3][3] == 'NA'
        for mode in ('1', '2', 'both'):
            call(binary, root, hic, mode, '-o', mode, '--dump-graph', root/f'{mode}.graph',
                 '--embeddings', root/f'{mode}.embedding')
            compare_edges(read_graph(root/f'{mode}.graph'), graph_oracle(chroms, pairs))
            inspect(root, mode, chroms, 16 if mode == 'both' else 8)
        first, second, both = [embeddings(root/f'{mode}.embedding') for mode in ('1', '2', 'both')]
        assert first.keys() == second.keys() == both.keys()
        for node in first:
            combined = first[node] + second[node]
            length = math.sqrt(sum(v*v for v in combined))
            assert all(abs(v/length-w) < 2e-7 for v, w in zip(combined, both[node]))
        call(binary, root, hic, 'repeat')
        assert (root/'repeat_compartments.bed').read_bytes() == (root/'1_compartments.bed').read_bytes()
        call(binary, root, hic, 'parallel', '--threads', '4', '-o', 'both')
        inspect(root, 'parallel', chroms, 16)
        # Reverse chromosome order and use aliases; exercise Straw transposition.
        sizes = root/'sizes'
        sizes.write_text('B\t35\nA\t40\n')
        call(binary, root, hic, 'subset', '-g', sizes, '--chromosomes', 'chrA,B',
             '--dump-graph', root/'subset.graph')
        subpairs = {(0, 1): [(y, x, w) for x, y, w in pairs[(0, 1)]]}
        compare_edges(read_graph(root/'subset.graph'), graph_oracle([('B', 35), ('A', 40)], subpairs))
        inspect(root, 'subset', [('B', 35), ('A', 40)], 8)
        ten = root/'ten.hic'
        v10(ten, chroms, pairs, derived=True)
        call(binary, root, ten, 'v10', '--dump-graph', root/'ten.graph')
        compare_edges(read_graph(root/'ten.graph'), graph_oracle(chroms, pairs))
        inspect(root, 'v10', chroms, 8)
        # Derived V10 resolution must aggregate before applying the logarithm.
        coarse = {}
        for pair, points in pairs.items():
            values = {}
            for x, y, w in points:
                values[x//2, y//2] = values.get((x//2, y//2), 0) + w
            coarse[pair] = [(x, y, w) for (x, y), w in values.items()]
        call(binary, root, ten, 'derived', '--dump-graph', root/'derived.graph', resolution=20)
        compare_edges(read_graph(root/'derived.graph'), graph_oracle(chroms, coarse, resolution=20))
        inspect(root, 'derived', chroms, 8)
        missing = root/'missing.hic'
        v9(missing, chroms, {(0, 1): pairs[(0, 1)]})
        call(binary, root, missing, 'missing')
        assert json.loads((root/'missing_summary.json').read_text())['missing_pairs'] == 2
        empty = root/'empty.hic'
        v9(empty, chroms, {(0, 1): [(0, 0, 1)]})
        assert 'no positive-weight' in call(binary, root, empty, 'empty', ok=False).stderr
        # Explicit diagnostics and strict numeric/option validation.
        for extra in [('--normalization', 'BAD'), ('--order', '3'), ('--clusters', '999'),
                      ('--threads', '0'), ('--dimensions', '-1'), ('--negative', '-1'),
                      ('--samples', 'nan'), ('--sample-count', '0'), ('--seed', '-1'),
                      ('--chromosomes', 'absent'), ('--resolution', '11'), ('--adj', 'x')]:
            call(binary, root, hic, 'bad', *extra, ok=False)
        call(binary, root, hic, 'bad', '--embeddings', hic, ok=False)
        # An overflowing SGD run must fail cleanly, including with several workers.
        call(binary, root, hic, 'unstable', '--learning-rate', '1e38', '--threads', '4', ok=False)
        garbage = root/'garbage.hic'
        garbage.write_bytes(b'not a hic file\n')
        call(binary, root, garbage, 'garbage', ok=False)
        # Scientific smoke test: learn three planted trans-contact communities.
        planted_chroms = [('1', 300), ('2', 300), ('3', 300)]
        planted_pairs = {(a, b): [(x, y, 1000 if x//10 == y//10 else 1)
                                  for x in range(30) for y in range(30)]
                         for a in range(3) for b in range(a+1, 3)}
        planted = root/'planted.hic'
        v9(planted, planted_chroms, planted_pairs)
        for mode in ('1', '2', 'both'):
            name = f'planted_{mode}'
            call(binary, root, planted, name, '-o', mode, '-k', '3', '--restarts', '5', samples=150000)
            labels = [int(line.split()[3]) for line in (root/f'{name}_compartments.bed').read_text().splitlines()]
            truth = [(i % 30)//10 for i in range(90)]
            accuracy = max(sum(p[label] == actual for label, actual in zip(labels, truth))/90
                           for p in permutations(range(3)))
            assert accuracy > .95, (mode, accuracy)
    print('v9/v10, normalization, graph oracle, all orders, outputs, determinism, parallel and failure cases passed')


if __name__ == '__main__':
    main()
