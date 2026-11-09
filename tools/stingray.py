"""Reading HELLDIVERS (2015) bundles: the murmur64 name hash, bundle entries, and resource lookup.

The engine is Bitsquid (later Stingray). A bundle in contents/ is a header (0xF0000004, the unpacked size)
then zlib chunks of 64 KiB; a chunk of exactly 0x10000 bytes is stored raw. Unpacked: a count, a 256-byte
gap, then count x 16-byte (type, name) pairs, then each resource: type, name, variant count, a stream
offset, a 12-byte head per variant (language, size, stream size), then the variants' data.
"""
import os
import struct
import zlib
from pathlib import Path

CONTENTS = Path(os.environ.get("HD1_CONTENTS", r"C:\Program Files (x86)\Steam\steamapps\common\Helldivers\contents"))

TYPES = ["render_config", "shader", "shader_library", "shader_library_group", "shading_environment",
         "shading_environment_mapping", "texture", "material", "unit", "level", "lua", "package", "config",
         "particles", "font", "physics_properties", "surface_properties", "mouse_cursor", "bones", "state_machine",
         "animation", "network_config", "sound_environment", "wwise_bank", "wwise_stream", "wwise_metadata",
         "wwise_dep", "timpani_bank", "timpani_master", "strings", "data", "ivf", "bik", "flow", "vector_field",
         "apb", "entity", "component", "baked_lighting", "mod", "lua_ref", "level_data"]


def murmur64(data, seed=0):
    if isinstance(data, str):
        data = data.encode()
    m, r, mask = 0xc6a4a7935bd1e995, 47, (1 << 64) - 1
    h = (seed ^ (len(data) * m)) & mask
    tail = len(data) & ~7
    for (k,) in struct.iter_unpack("<Q", data[:tail]):
        k = (k * m) & mask
        k ^= k >> r
        k = (k * m) & mask
        h = ((h ^ k) * m) & mask
    rest = data[tail:]
    if rest:
        h = ((h ^ int.from_bytes(rest, "little")) * m) & mask
    h ^= h >> r
    h = (h * m) & mask
    return h ^ (h >> r)


def murmur32(data):
    """The 32-bit names materials and shading variables use: the top half of murmur64."""
    return murmur64(data) >> 32


TYPE_NAMES = {murmur64(t): t for t in TYPES}


def unpack(path):
    data = Path(path).read_bytes()
    if len(data) < 12 or struct.unpack_from("<I", data, 0)[0] != 0xF0000004:
        return None
    total = struct.unpack_from("<I", data, 4)[0]
    raw, pos = bytearray(), 12
    while pos < len(data) and len(raw) < total:
        (size,) = struct.unpack_from("<I", data, pos)
        pos += 4
        chunk = data[pos:pos + size]
        pos += size
        raw += chunk if size == 0x10000 else zlib.decompress(chunk)
    return bytes(raw)


def entries(path):
    """A bundle's resources: (type hash, name hash, [variant data])."""
    raw = unpack(path)
    if raw is None:
        return
    (count,) = struct.unpack_from("<I", raw, 0)
    pos = 260 + count * 16
    for _ in range(count):
        kind, name, variants, _ = struct.unpack_from("<QQII", raw, pos)
        pos += 24
        heads = [struct.unpack_from("<III", raw, pos + 12 * i) for i in range(variants)]
        pos += 12 * variants
        datas = []
        for _, size, _ in heads:
            datas.append(raw[pos:pos + size])
            pos += size
        yield kind, name, datas


def bundles():
    for p in sorted(CONTENTS.iterdir()):
        if p.is_file() and len(p.name) == 16 and "." not in p.name:
            yield p


def find(bundle, kind, name):
    kh = murmur64(kind)
    nh = name if isinstance(name, int) else murmur64(name)
    for k, n, datas in entries(CONTENTS / bundle):
        if k == kh and n == nh:
            return datas[0]
    raise KeyError(f"no {kind} {nh:016x} in {bundle}")
