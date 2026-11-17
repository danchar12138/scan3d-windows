"""Mark which way is up in the exported meshes.

The splat and the meshes share the trainer's coordinate frame, in which +Z is up. glTF viewers
and game engines expect +Y up, so a mesh opened as it is would lie on its side. The GLB files
get a node rotation that turns the scene's up direction to +Y while their vertex numbers stay
exactly equal to the splat's coordinates; the same rotation is written to report.json so it can
be applied to the splat.
"""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path


def rotate(matrix_3x3: list[list[float]], v: list[float]) -> list[float]:
    return [sum(matrix_3x3[r][c] * v[c] for c in range(3)) for r in range(3)]


def quaternion_to_y_up(up: list[float]) -> list[float]:
    """Unit quaternion (x, y, z, w) of the shortest rotation taking `up` to +Y."""
    ux, uy, uz = up
    # axis = up x (0, 1, 0); |axis| = sin(angle), up . y = cos(angle)
    axis = [-uz, 0.0, ux]
    sin_a = math.sqrt(axis[0] ** 2 + axis[2] ** 2)
    if sin_a < 1e-9:
        return [0.0, 0.0, 0.0, 1.0] if uy > 0 else [1.0, 0.0, 0.0, 0.0]
    half = math.atan2(sin_a, uy) / 2
    s = math.sin(half) / sin_a
    return [axis[0] * s, 0.0, axis[2] * s, math.cos(half)]


def set_glb_root_rotation(path: Path, quaternion_xyzw: list[float], note: dict) -> None:
    """Rewrite a GLB so its root node carries the rotation. Only the JSON chunk changes."""
    data = Path(path).read_bytes()
    magic, version, _length = struct.unpack("<III", data[:12])
    json_len, json_type = struct.unpack("<II", data[12:20])
    if magic != 0x46546C67 or json_type != 0x4E4F534A:
        raise ValueError(f"not a GLB file: {path}")
    doc = json.loads(data[20:20 + json_len])
    rest = data[20 + json_len:]
    for index in doc["scenes"][doc.get("scene", 0)]["nodes"]:
        doc["nodes"][index]["rotation"] = [round(q, 9) for q in quaternion_xyzw]
    doc["asset"].setdefault("extras", {}).update(note)
    body = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    body += b" " * ((-len(body)) % 4)
    out = struct.pack("<III", magic, version, 12 + 8 + len(body) + len(rest))
    out += struct.pack("<II", len(body), json_type) + body + rest
    tmp = Path(path).with_name(Path(path).name + ".tmp")
    tmp.write_bytes(out)
    tmp.replace(path)
