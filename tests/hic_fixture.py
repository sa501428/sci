"""Small independent binary .hic writers; standard library plus native libzstd.

Contact positions are bin indices. Tests never use Straw to write their inputs.
"""
import ctypes
import ctypes.util
import math
from pathlib import Path
import struct
import zlib


def pack(fmt, *v):
    return struct.pack('<' + fmt, *v)


def string(s):
    return s.encode() + b'\0'


def v9(path, chroms, pairs, resolution=10, norms=None):
    norms = norms or {}
    chromosomes = [('All', 1000), *chroms]
    data = bytearray(b'HIC\0' + pack('iQ', 9, 0) + string('synthetic'))
    nvi_patch = len(data)
    data += bytes(16) + pack('ii', 0, len(chromosomes))
    for name, length in chromosomes:
        data += string(name) + pack('Q', length)
    data += pack('iii', 1, resolution, 0)
    data += bytes(max(0, 131072 - len(data)))
    matrices = []
    for (a, b), points in pairs.items():
        rows = {}
        for x, y, count in points:
            rows.setdefault(y, []).append((x, count))
        raw = pack('iii4Bh', len(points), 0, 0, 1, 0, 0, 1, len(rows))
        for y, row in sorted(rows.items()):
            raw += pack('hh', y, len(row)) + b''.join(pack('hf', x, value) for x, value in sorted(row))
        block = zlib.compress(raw)
        block_at = len(data)
        data += block
        matrix_at = len(data)
        n = max(math.ceil(chroms[a][1] / resolution), math.ceil(chroms[b][1] / resolution))
        matrix = pack('iii', a + 1, b + 1, 1) + string('BP')
        matrix += pack('i4f4i', 0, sum(p[2] for p in points), len(points), 0, 0, resolution, n, 1, 1)
        matrix += pack('iQi', 0, block_at, len(block))
        data += matrix
        matrices.append((f'{a+1}_{b+1}', matrix_at, len(matrix)))
    master = len(data)
    struct.pack_into('<Q', data, 8, master)
    footer = pack('i', len(matrices))
    for name, at, length in matrices:
        footer += string(name) + pack('Qi', at, length)
    footer += pack('i', 0)  # no expected vectors
    data += pack('Q', len(footer)) + footer + pack('i', 0)
    nvi = len(data)
    index_size = 4 + sum(len(string(norm)) + 4 + 3 + 4 + 8 + 8 for norm in norms for _ in chroms)
    index, payload = bytearray(pack('i', len(norms) * len(chroms))), bytearray()
    for norm, vectors in norms.items():
        for c, vector in enumerate(vectors):
            chunk = pack('Q', len(vector)) + pack(f'{len(vector)}f', *vector)
            index += string(norm) + pack('i', c + 1) + string('BP')
            index += pack('iQQ', resolution, nvi + index_size + len(payload), len(chunk))
            payload += chunk
    data += index + payload + bytes(131072)
    struct.pack_into('<QQ', data, nvi_patch, nvi, len(index))
    Path(path).write_bytes(data)


def var(n):
    result = bytearray()
    while n >= 128:
        result.append((n & 127) | 128)
        n >>= 7
    result.append(n)
    return bytes(result)


def compress(data):
    lib = ctypes.CDLL(ctypes.util.find_library('zstd') or '/opt/homebrew/lib/libzstd.dylib')
    lib.ZSTD_compressBound.argtypes = [ctypes.c_size_t]
    lib.ZSTD_compressBound.restype = ctypes.c_size_t
    lib.ZSTD_compress.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    lib.ZSTD_compress.restype = ctypes.c_size_t
    lib.ZSTD_isError.argtypes = [ctypes.c_size_t]
    lib.ZSTD_isError.restype = ctypes.c_uint
    out = ctypes.create_string_buffer(lib.ZSTD_compressBound(len(data)))
    n = lib.ZSTD_compress(out, len(out), data, len(data), 1)
    assert not lib.ZSTD_isError(n)
    return out.raw[:n]


def v10(path, chroms, pairs, resolution=10, derived=False):
    variable = string('synthetic') + pack('II', 0, len(chroms))
    variable += b''.join(string(name) + pack('Q', length) for name, length in chroms)
    nr = 2 if derived else 1
    variable += pack('I', nr) + pack('IBBHI', resolution, 0, 1, 0, 0xffffffff)
    if derived:
        variable += pack('IBBHI', resolution * 2, 1, 1, 0, 0)
    variable += pack('II', 0, 0)  # no FRAG resolutions or normalizations
    data = bytearray(88) + variable
    entries = []
    for (a, b), points in pairs.items():
        nx = math.ceil(chroms[a][1] / resolution)
        ny = math.ceil(chroms[b][1] / resolution)
        points = sorted((y * nx + x, int(value)) for x, y, value in points)
        assert points and all(value > 0 for _, value in points)
        mp = len(data)
        data += b'H10M' + pack('5I', 1, a, b, nr, 0) + bytes(76 * nr)
        positions = var(points[0][0]) + b''.join(var(points[i][0] - points[i-1][0]) for i in range(1, len(points)))
        values = b''.join(var(value) for _, value in points)
        raw = pack('5B3x4IQ2I', 1, 0, 2, 0, 0, 0, 0, nx, ny, len(points), len(positions), len(values))
        raw += positions + values
        stored = b'H10B' + pack('BBHII', 1, 1, 0, len(raw), 0) + compress(raw)
        bp = len(data)
        data += stored
        ip = len(data)
        index = b'H10I' + pack('IQII', 2, 40, 1, 0) + pack('IIQ', 0, len(stored), bp)
        data += index
        descriptor = pack('4B3IB3xQQII2IQQ2I', 0, 0, 1, 0, 0, resolution, 0xffffffff,
                          0, sum(v for _, v in points), len(points), 0x7fc00000, 0x7fc00000,
                          max(nx, ny), 1, ip, len(index), 1, 0)
        if derived:
            coarse_n = len({((p % nx)//2, (p//nx)//2) for p, _ in points})
            descriptor += pack('4B3IB3xQQII2IQQ2I', 0, 1, 1, 0, 1, resolution*2, 0,
                               0, sum(v for _, v in points), coarse_n, 0x7fc00000, 0x7fc00000,
                               max(nx, ny), 1, 0, 0, 0, 0)
        data[mp+24:mp+24+len(descriptor)] = descriptor
        entries.append(pack('IIQQ', a, b, mp, 24+76*nr))
    fp = len(data)
    footer = b'H10F' + pack('IQII', 1, 24+len(entries)*24, len(entries), 0) + b''.join(entries)
    data += footer
    data[:88] = b'HIC\0' + pack('I10Q', 10, 88+len(variable), fp, len(footer), 0, 0, 0, 0, 0, 0, 0)
    Path(path).write_bytes(data)
