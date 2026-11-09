"""Surveys every texture in the game: size, format, mip count, and how much sits in the bundle's .stream
file. Writes internal/research/textures.csv and prints totals.

Usage: python tools/texture_survey.py
"""
import collections
import csv
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stingray as st  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "internal" / "research" / "textures.csv"
TEXTURE = st.murmur64("texture")


def entries_with_streams(path):
    raw = st.unpack(path)
    if raw is None:
        return
    (count,) = struct.unpack_from("<I", raw, 0)
    pos = 260 + count * 16
    for _ in range(count):
        kind, name, variants, _ = struct.unpack_from("<QQII", raw, pos)
        pos += 24
        heads = [struct.unpack_from("<III", raw, pos + 12 * i) for i in range(variants)]
        pos += 12 * variants
        first = None
        for _, size, stream in heads:
            if first is None:
                first = (raw[pos:pos + size], stream)
            pos += size
        yield kind, name, first


def describe(data):
    i = data.find(b"DDS ")
    if i < 0:
        return None
    h, w, mips = struct.unpack_from("<II", data, i + 12) + (struct.unpack_from("<I", data, i + 28)[0],)
    flags, fourcc = struct.unpack_from("<I4s", data, i + 80)
    fmt = fourcc.decode("latin-1").strip("\0") if flags & 4 else f"rgb{struct.unpack_from('<I', data, i + 88)[0]}"
    if fmt == "DX10":
        fmt = f"dxgi{struct.unpack_from('<I', data, i + 128)[0]}"
    return w, h, max(mips, 1), fmt, i


def main():
    rows = []
    for b in st.bundles():
        try:
            for kind, name, (data, stream) in entries_with_streams(b):
                if kind != TEXTURE:
                    continue
                d = describe(data)
                if d:
                    rows.append((b.name, f"{name:016x}", *d[:4], len(data), stream))
        except Exception as e:
            print("skip", b.name, e)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        csv.writer(f).writerows([("bundle", "name", "w", "h", "mips", "format", "bytes", "stream")] + rows)
    unique = {r[1]: r for r in rows}
    print(f"{len(rows)} texture entries, {len(unique)} distinct names")
    fmts = collections.Counter(r[5] for r in unique.values())
    print("formats:", fmts.most_common(8))
    sizes = collections.Counter(max(r[2], r[3]) for r in unique.values())
    print("largest side:", sorted(sizes.items()))
    total = sum(r[6] + r[7] for r in unique.values())
    streamed = sum(1 for r in unique.values() if r[7])
    print(f"distinct data {total / 2**30:.2f} GiB, {streamed} with a streamed part")


if __name__ == "__main__":
    main()
