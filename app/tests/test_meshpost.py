"""Tests for app/meshpost.py. Standard library only.

    python -m unittest app.tests.test_meshpost          (from the project root)
    python app/tests/test_meshpost.py

The pipeline tests drive build_meshes end to end against a stand-in for colmap.exe (a small
Python script behind a .cmd launcher), so they need no GPU and take a few seconds each. Set
SCAN3D_MESH_ITEST=1 to also run the real tool on runs/sb-v2 (about 20 minutes, uses the GPU).
"""
import array
import json
import math
import os
import shutil
import struct
import sys
import tempfile
import unittest
import unittest.mock
import zlib
from collections import Counter
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
ROOT = APP.parent
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

import meshpost as mp  # noqa: E402
import proc  # noqa: E402


def cube(offset=(0.0, 0.0, 0.0), size=1.0):
    """8 vertices, 12 triangles."""
    ox, oy, oz = offset
    xyz = []
    for z in (0, 1):
        for y in (0, 1):
            for x in (0, 1):
                xyz += [ox + x * size, oy + y * size, oz + z * size]
    quads = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    tris = []
    for a, b, c, d in quads:
        tris += [a, b, c, a, c, d]
    return xyz, tris


def merged(*parts):
    xyz, tris = [], []
    for pxyz, ptris in parts:
        base = len(xyz) // 3
        xyz += pxyz
        tris += [i + base for i in ptris]
    return mp.Mesh(array.array("f", xyz), array.array("i", tris))


def grid(n, size=1.0, height=None):
    """n x n squares over [0, size]^2, two triangles each; z = height(x, y) or 0."""
    xyz, tris = [], []
    for j in range(n + 1):
        for i in range(n + 1):
            x, y = size * i / n, size * j / n
            xyz += [x, y, height(x, y) if height else 0.0]
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            tris += [a, a + 1, a + n + 2, a, a + n + 2, a + n + 1]
    return mp.Mesh(array.array("f", xyz), array.array("i", tris))


def two_chart_grid(n=8):
    """The unit grid as a texturer leaves it: cut into two charts along x = 0.5, the vertices
    on the cut present once per chart with different texture coordinates. The left chart
    fills u in [0.05, 0.45], the right one u in [0.55, 0.95]."""
    half = n // 2
    xyz, uv, tris = [], [], []
    for chart, (first, u0) in enumerate(((0, 0.05), (half, 0.55))):
        base = len(xyz) // 3
        for j in range(n + 1):
            for i in range(half + 1):
                xyz += [(first + i) / n, j / n, 0.0]
                uv += [u0 + 0.4 * i / half, 0.05 + 0.9 * j / n]
        for j in range(n):
            for i in range(half):
                a = base + j * (half + 1) + i
                tris += [a, a + 1, a + half + 2, a, a + half + 2, a + half + 1]
    return mp.Mesh(array.array("f", xyz), array.array("i", tris), array.array("f", uv))


def border_edges(mesh):
    """Edges with a single face, as sorted pairs of positions."""
    seen = Counter()
    point = [tuple(mesh.xyz[3 * i:3 * i + 3]) for i in range(mesh.vertex_count)]
    for f in range(mesh.face_count):
        a, b, c = (point[i] for i in mesh.tris[3 * f:3 * f + 3])
        for e in ((a, b), (b, c), (c, a)):
            seen[tuple(sorted(e))] += 1
    return [e for e, k in seen.items() if k == 1]


def tiny_png(width=4, height=2):
    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
    rows = b"".join(b"\x00" + b"".join(bytes([(40 * x) % 256, (90 * y) % 256, 200]) for x in range(width))
                    for y in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


class TempDirCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="meshpost_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)


class TopologyTests(unittest.TestCase):
    def test_components_and_area(self):
        m = merged(cube(), cube((5, 0, 0), 0.1), ([9, 9, 9, 9.01, 9, 9, 9, 9.01, 9], [0, 1, 2]))
        labels, faces, areas = mp.face_components(m)
        self.assertEqual(len(labels), 25)
        self.assertEqual(sorted(faces.values()), [1, 12, 12])
        self.assertEqual([round(a, 4) for a in sorted(areas.values())], [0.0001, 0.06, 6.0])

    def test_keep_large_components(self):
        m = merged(cube(), cube((5, 0, 0), 0.1), cube((0, 5, 0), 0.5))
        kept, stats = mp.keep_large_components(m, 0.05)      # 0.1 cube has 1% of the area
        self.assertEqual(stats["components_before"], 3)
        self.assertEqual(stats["components_after"], 2)
        self.assertEqual(kept.face_count, 24)
        self.assertEqual(kept.vertex_count, 16)
        self.assertLess(max(kept.tris), 16)
        self.assertAlmostEqual(stats["area_kept_fraction"], 7.5 / 7.56, places=5)
        one, _ = mp.keep_large_components(m, 0.5)
        self.assertEqual(one.face_count, 12)
        self.assertEqual(list(one.xyz), list(merged(cube()).xyz))
        same, stats = mp.keep_large_components(m, 0.0)
        self.assertIs(same, m)

    def test_duplicated_vertices_do_not_split_a_component(self):
        # Two triangles that share an edge by position only (four of six vertices coincide).
        m = mp.Mesh(array.array("f", [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]),
                    array.array("i", [0, 1, 2, 3, 5, 4]))
        _, faces, _ = mp.face_components(m)
        self.assertEqual(list(faces.values()), [2])
        # ...which is how the two charts of a textured mesh count as one piece
        _, faces, _ = mp.face_components(two_chart_grid())
        self.assertEqual(list(faces.values()), [128])

    def test_select_faces_keeps_texture_coordinates(self):
        m = two_chart_grid(4)
        keep = [f >= 16 for f in range(32)]                  # the right-hand chart only
        s = mp.select_faces(m, keep)
        self.assertEqual(s.face_count, 16)
        self.assertEqual(s.vertex_count, 15)
        self.assertEqual(len(s.uv), 30)
        self.assertGreaterEqual(min(s.uv[0::2]), 0.55 - 1e-6)
        # every face still points at the same position and coordinate as before
        for new_face, old_face in enumerate(range(16, 32)):
            for corner in range(3):
                a = s.tris[3 * new_face + corner]
                b = m.tris[3 * old_face + corner]
                self.assertEqual(list(s.xyz[3 * a:3 * a + 3]), list(m.xyz[3 * b:3 * b + 3]))
                self.assertEqual(list(s.uv[2 * a:2 * a + 2]), list(m.uv[2 * b:2 * b + 2]))

    def test_drop_degenerate_faces(self):
        m = mp.Mesh(array.array("f", [0, 0, 0, 1, 0, 0, 0, 1, 0, 5, 5, 5, 2, 0, 0]),
                    array.array("i", [0, 1, 2, 0, 0, 1, 2, 1, 2, 3, 3, 3, 0, 1, 4]))
        d = mp.drop_degenerate_faces(m)
        self.assertEqual(d.face_count, 2)                    # the collinear face 0-1-4 stays...
        z = mp.drop_degenerate_faces(m, zero_area=True)
        self.assertEqual(z.face_count, 1)                    # ...unless zero-area faces go too
        self.assertEqual(z.vertex_count, 3)
        clean = merged(cube())
        self.assertIs(mp.drop_degenerate_faces(clean, zero_area=True), clean)

    def test_weld_joins_the_charts(self):
        m = two_chart_grid(4)
        w = mp.weld(m)
        self.assertEqual(m.vertex_count, 30)
        self.assertEqual(w.vertex_count, 25)
        self.assertEqual(w.face_count, 32)
        self.assertIsNone(w.uv)
        self.assertEqual(len(border_edges(w)), 16)           # only the outline is open

    def test_vertex_normals(self):
        m = mp.Mesh(array.array("f", [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 7, 7, 7]),
                    array.array("i", [0, 1, 2, 0, 2, 3]))
        n = mp.vertex_normals(m)
        self.assertEqual(list(n[:12]), [0, 0, 1] * 4)
        self.assertEqual(list(n[12:]), [0, 0, 1])          # unreferenced vertex gets a unit normal
        c = mp.vertex_normals(merged(cube()))
        for i in range(8):
            x, y, z = c[3 * i:3 * i + 3]
            self.assertAlmostEqual(x * x + y * y + z * z, 1.0, places=5)

    def test_normals_agree_across_a_seam(self):
        # A ridge along the cut between the two charts: each side alone would tilt its normal.
        m = two_chart_grid(4)
        for i in range(m.vertex_count):
            m.xyz[3 * i + 2] = abs(m.xyz[3 * i] - 0.5)
        apart = mp.vertex_normals(m)
        together = mp.seamless_normals(m)

        def twins(normals):
            by_position = {}
            for i in range(m.vertex_count):
                by_position.setdefault(tuple(m.xyz[3 * i:3 * i + 3]), []).append(tuple(normals[3 * i:3 * i + 3]))
            return [v for v in by_position.values() if len(v) == 2]
        self.assertEqual(len(twins(apart)), 5)
        self.assertTrue(all(a != b for a, b in twins(apart)))        # a crease, chart by chart
        for a, b in twins(together):
            self.assertEqual(a, b)
            self.assertAlmostEqual(sum(c * c for c in a), 1.0, places=5)
        # away from the cut nothing changes
        self.assertEqual(list(together[:6]), list(apart[:6]))


class FrameTests(TempDirCase):
    def test_missing_file_is_identity(self):
        t = mp.read_scene_transform(self.tmp)
        self.assertTrue(t["identity"])

    def test_reads_matrices(self):
        fwd = [0.5, 0, 0, 1, 0, 0, -0.5, 2, 0, 0.5, 0, 3]
        inv = [2, 0, 0, -2, 0, 0, 2, -6, 0, -2, 0, 4]
        (self.tmp / "scene_transform.json").write_text(json.dumps({
            "train_from_world": {"matrix_3x4_flat_row_major": fwd},
            "world_from_train": {"matrix_3x4_flat_row_major": inv}}))
        t = mp.read_scene_transform(self.tmp)
        self.assertFalse(t["identity"])
        self.assertEqual(list(t["train_from_world"]), fwd)
        self.assertEqual(list(t["world_from_train"]), inv)

    def test_damaged_file_is_reported(self):
        (self.tmp / "scene_transform.json").write_text("{\"train_from_world\": {}}")
        with self.assertRaisesRegex(mp.MeshError, "scene transform"):
            mp.read_scene_transform(self.tmp)


class GlbTests(TempDirCase):
    def write_cube(self, name="c.glb", textured=False):
        m = merged(cube())
        kwargs = {}
        if textured:
            kwargs["uvs"] = array.array("f", [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0] * 2)
            kwargs["image"] = tiny_png()
        mp.write_glb(self.tmp / name, positions=m.xyz, indices=array.array("I", m.tris),
                     normals=mp.vertex_normals(m), name="cube", extras={"frame": "test"}, **kwargs)
        return self.tmp / name

    def test_untextured(self):
        info = mp.validate_glb(self.write_cube())
        self.assertEqual((info["faces"], info["vertices"]), (12, 8))
        self.assertTrue(info["has_normals"])
        self.assertFalse(info["has_uv"])
        self.assertEqual(info["texture_width"], 0)
        self.assertEqual(info["bbox_min"], [0.0, 0.0, 0.0])
        self.assertEqual(info["bbox_max"], [1.0, 1.0, 1.0])
        self.assertEqual(info["extras"], {"frame": "test"})

    def test_textured_and_layout(self):
        path = self.write_cube("t.glb", textured=True)
        info = mp.validate_glb(path)
        self.assertTrue(info["has_uv"])
        self.assertEqual((info["texture_width"], info["texture_height"]), (4, 2))
        raw = path.read_bytes()
        magic, version, total = struct.unpack_from("<III", raw, 0)
        self.assertEqual((magic, version, total), (0x46546C67, 2, len(raw)))
        n, kind = struct.unpack_from("<II", raw, 12)
        self.assertEqual(kind, 0x4E4F534A)
        self.assertEqual(n % 4, 0)
        doc = json.loads(raw[20:20 + n])
        self.assertEqual(doc["asset"]["version"], "2.0")
        self.assertNotIn("matrix", doc["nodes"][0])           # no node transform, ever
        self.assertNotIn("rotation", doc["nodes"][0])
        self.assertEqual(doc["images"][0]["mimeType"], "image/png")
        self.assertEqual(doc["accessors"][0]["count"], 36)
        for view in doc["bufferViews"]:
            self.assertEqual(view["byteOffset"] % 4, 0)
        self.assertFalse(list(self.tmp.glob("*.part")))

    def test_corruption_is_caught(self):
        path = self.write_cube()
        raw = bytearray(path.read_bytes())
        n = struct.unpack_from("<I", raw, 12)[0]
        doc = json.loads(raw[20:20 + n])
        bin_start = 20 + n + 8

        def variant(name, mutate_doc=None, mutate_raw=None):
            d = json.loads(json.dumps(doc))
            data = bytearray(raw)
            if mutate_doc:
                mutate_doc(d)
                text = json.dumps(d, separators=(",", ":")).encode()
                text += b" " * ((-len(text)) % 4)
                data = bytearray(struct.pack("<III", 0x46546C67, 2, 0) + struct.pack("<II", len(text), 0x4E4F534A)
                                 + text + bytes(raw[20 + n:]))
                struct.pack_into("<I", data, 8, len(data))
            if mutate_raw:
                mutate_raw(data)
            (self.tmp / name).write_bytes(data)
            return self.tmp / name

        cases = {
            "header says": variant("a.glb", mutate_raw=lambda d: d.__delitem__(slice(len(d) - 8, len(d)))),
            "magic": variant("b.glb", mutate_raw=lambda d: d.__setitem__(0, 0x41)),
            "index points past": variant("c.glb", mutate_raw=lambda d: struct.pack_into("<I", d, bin_start, 99)),
            "min/max": variant("d.glb", mutate_doc=lambda d: d["accessors"][1].__setitem__("min", [9, 9, 9])),
            "runs past": variant("e.glb", mutate_doc=lambda d: d["accessors"][1].__setitem__("count", 800)),
            "multiple of 3": variant("f.glb", mutate_doc=lambda d: d["accessors"][0].__setitem__("count", 35)),
            "outside the buffer": variant("g.glb", mutate_doc=lambda d: d["bufferViews"][1].__setitem__("byteLength", 10 ** 6)),
            "2.0": variant("h.glb", mutate_doc=lambda d: d["asset"].__setitem__("version", "1.0")),
            "does not exist": variant("i.glb", mutate_doc=lambda d: d["scenes"][0].__setitem__("nodes", [3])),
            "unit length": variant("j.glb", mutate_raw=lambda d: struct.pack_into(
                "<3f", d, bin_start + doc["bufferViews"][2]["byteOffset"], 3.0, 0.0, 0.0)),
        }
        for expected, path in cases.items():
            with self.subTest(expected):
                with self.assertRaisesRegex(mp.MeshError, expected):
                    mp.validate_glb(path)

    def test_empty_mesh_is_refused(self):
        with self.assertRaises(mp.MeshError):
            mp.write_glb(self.tmp / "e.glb", positions=array.array("f"), indices=array.array("I"))

    def test_image_size(self):
        self.assertEqual(mp.image_size(tiny_png(7, 5)), ("image/png", 7, 5))
        jpeg = (b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + bytes(11)
                + b"\xff\xc0\x00\x11\x08" + struct.pack(">HH", 30, 40) + bytes(12))
        self.assertEqual(mp.image_size(jpeg), ("image/jpeg", 40, 30))
        with self.assertRaises(mp.MeshError):
            mp.image_size(b"GIF89a" + bytes(40))


class ReadGlbTests(TempDirCase):
    def write(self, name="in.glb", textured=True, **kwargs):
        m = two_chart_grid(4)
        extra = {"uvs": m.uv, "image": tiny_png(8, 8)} if textured else {}
        mp.write_glb(self.tmp / name, positions=m.xyz, indices=array.array("I", m.tris),
                     normals=mp.vertex_normals(m), name="mesh", **extra, **kwargs)
        return m, self.tmp / name

    def test_round_trip(self):
        m, path = self.write()
        back, image = mp.read_glb(path)
        self.assertEqual(back.xyz, m.xyz)                    # bit for bit
        self.assertEqual(back.tris, m.tris)
        self.assertEqual(back.uv, m.uv)
        self.assertEqual(image, tiny_png(8, 8))

    def test_untextured(self):
        m, path = self.write(textured=False)
        back, image = mp.read_glb(path)
        self.assertIsNone(image)
        self.assertIsNone(back.uv)
        self.assertEqual(back.tris, m.tris)

    def rewrite(self, path, mutate):
        raw = path.read_bytes()
        n = struct.unpack_from("<I", raw, 12)[0]
        doc = json.loads(raw[20:20 + n])
        mutate(doc)
        text = json.dumps(doc, separators=(",", ":")).encode()
        text += b" " * ((-len(text)) % 4)
        data = bytearray(struct.pack("<III", 0x46546C67, 2, 0) + struct.pack("<II", len(text), 0x4E4F534A)
                         + text + raw[20 + n:])
        struct.pack_into("<I", data, 8, len(data))
        path.write_bytes(data)

    def test_node_transform_is_refused(self):
        # A rotated node would mean the numbers in the file are no longer splat coordinates.
        for key, value in (("rotation", [0.7071068, 0, 0, 0.7071068]), ("matrix", [1.0] * 16),
                           ("translation", [0, 0, 1]), ("scale", [2, 2, 2])):
            with self.subTest(key):
                _, path = self.write(key + ".glb")
                self.rewrite(path, lambda d: d["nodes"][0].__setitem__(key, value))
                with self.assertRaisesRegex(mp.MeshError, "transform"):
                    mp.read_glb(path)

    def test_16_bit_indices_and_interleaved_views(self):
        # Not what Spirula writes today, but valid glTF: short indices, and positions stored
        # with a stride (here: each position followed by four bytes of padding).
        m = merged(cube())
        idx = array.array("H", m.tris).tobytes()
        pos = b"".join(m.xyz[3 * i:3 * i + 3].tobytes() + b"\xff" * 4 for i in range(8))
        doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
               "nodes": [{"mesh": 0}],
               "meshes": [{"primitives": [{"attributes": {"POSITION": 1}, "indices": 0}]}],
               "accessors": [{"bufferView": 0, "componentType": 5123, "count": 36, "type": "SCALAR"},
                             {"bufferView": 1, "componentType": 5126, "count": 8, "type": "VEC3"}],
               "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 72},
                               {"buffer": 0, "byteOffset": 72, "byteLength": 128, "byteStride": 16}],
               "buffers": [{"byteLength": 200}]}
        text = json.dumps(doc).encode()
        text += b" " * ((-len(text)) % 4)
        body = idx + pos
        (self.tmp / "s.glb").write_bytes(
            struct.pack("<III", 0x46546C67, 2, 28 + len(text) + len(body))
            + struct.pack("<II", len(text), 0x4E4F534A) + text
            + struct.pack("<II", len(body), 0x004E4942) + body)
        back, image = mp.read_glb(self.tmp / "s.glb")
        self.assertEqual(back.xyz, m.xyz)
        self.assertEqual(back.tris, m.tris)
        self.assertIsNone(image)

    def test_rejects_what_it_cannot_use(self):
        (self.tmp / "junk.glb").write_bytes(b"not a mesh at all, just some bytes..")
        with self.assertRaisesRegex(mp.MeshError, "not a binary glTF"):
            mp.read_glb(self.tmp / "junk.glb")
        with self.assertRaisesRegex(mp.MeshError, "cannot read"):
            mp.read_glb(self.tmp / "missing.glb")
        _, path = self.write("cut.glb")
        path.write_bytes(path.read_bytes()[:-40])
        with self.assertRaisesRegex(mp.MeshError, "truncated"):
            mp.read_glb(path)
        _, path = self.write("index.glb")
        raw = bytearray(path.read_bytes())
        n = struct.unpack_from("<I", raw, 12)[0]
        struct.pack_into("<I", raw, 20 + n + 8, 99)          # first index: past the last vertex
        path.write_bytes(raw)
        with self.assertRaisesRegex(mp.MeshError, "does not exist"):
            mp.read_glb(path)
        _, path = self.write("two.glb")
        self.rewrite(path, lambda d: d["meshes"][0]["primitives"].append(d["meshes"][0]["primitives"][0]))
        with self.assertRaisesRegex(mp.MeshError, "exactly one mesh"):
            mp.read_glb(path)
        _, path = self.write("odd.glb")
        self.rewrite(path, lambda d: d.__delitem__("accessors"))
        with self.assertRaisesRegex(mp.MeshError, "structure"):
            mp.read_glb(path)


# ---------------------------------------------------------------------------------------------
# Helpers for the texturing pieces: COLMAP binary models, PNG files with every row filter
# ---------------------------------------------------------------------------------------------

def quat_from_matrix(R):
    """(w, x, y, z) for a row-major 3x3 rotation (no special cases needed for the test poses)."""
    r00, r01, r02, r10, r11, r12, r20, r21, r22 = R
    w = math.sqrt(max(0.0, 1.0 + r00 + r11 + r22)) / 2.0
    return (w, (r21 - r12) / (4 * w), (r02 - r20) / (4 * w), (r10 - r01) / (4 * w))


def write_model(folder, cameras, images):
    """cameras: {id: (model, w, h, params)}; images: [(id, name, R row-major, t, camera id)]."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    out = bytearray(struct.pack("<Q", len(cameras)))
    for cid, (model, w, h, params) in cameras.items():
        out += struct.pack("<iiQQ", cid, model, w, h) + struct.pack("<%dd" % len(params), *params)
    (folder / "cameras.bin").write_bytes(bytes(out))
    out = bytearray(struct.pack("<Q", len(images)))
    for iid, name, R, t, cid in images:
        out += struct.pack("<i4d3di", iid, *quat_from_matrix(R), *t, cid)
        out += name.encode() + b"\x00" + struct.pack("<Q", 0)
    (folder / "images.bin").write_bytes(bytes(out))


def png_bytes(width, height, rows, channels=1, filters=(0,)):
    """Encode 8-bit rows (bytes, width*channels each) as PNG, cycling through `filters`."""
    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
    bpp = channels
    prev = bytes(width * channels)
    raw = bytearray()
    for y, row in enumerate(rows):
        kind = filters[y % len(filters)]
        out = bytearray(len(row))
        for i, value in enumerate(row):
            a = row[i - bpp] if i >= bpp else 0
            b = prev[i]
            c = prev[i - bpp] if i >= bpp else 0
            if kind == 0:
                pred = 0
            elif kind == 1:
                pred = a
            elif kind == 2:
                pred = b
            elif kind == 3:
                pred = (a + b) >> 1
            else:
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
            out[i] = (value - pred) & 255
        raw += bytes([kind]) + out
        prev = row
    ctype = {1: 0, 2: 4, 3: 2, 4: 6}[channels]
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, ctype, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(raw))) + chunk(b"IEND", b""))


def decode_png(data):
    """(width, height, channels, rows) of an 8-bit non-interlaced PNG, via the module's own unfilter."""
    ihdr = None
    idat = b""
    for kind, body in mp._png_chunks(data):
        if kind == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat += body
    w, h, depth, ctype = ihdr[:4]
    channels = {0: 1, 2: 3, 4: 2, 6: 4}[ctype]
    return w, h, channels, mp._unfilter(zlib.decompress(idat), h, w * channels, channels)


# A pinhole camera at (2, 2, -3) looking along +z at the plane z = 0 (the stand-in's scene).
LOOK_R = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
LOOK_T = (-2.0, -2.0, 3.0)                     # t = -R C, C = (2, 2, -3)
LOOK_CAMERA = (1, 64, 48, (40.0, 40.0, 32.0, 24.0))
# The pipeline tests' camera: further back, it sees the whole stand-in scene, floaters included.
WIDE_T = (-21.0, -6.0, 30.0)                    # C = (21, 6, -30)


class CameraModelTests(TempDirCase):
    def test_model_round_trip(self):
        cams = {1: (4, 3072, 2304, (2562.3, 2566.2, 1546.9, 1177.9, -0.0335, 0.0317, 8.4e-05, 0.00064)),
                3: (1, 640, 480, (500.0, 501.0, 320.0, 240.0))}
        Rz = (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        write_model(self.tmp, cams, [(7, "photos/a.jpg", Rz, (1.0, 2.0, 3.0), 1),
                                     (9, "b.jpg", LOOK_R, LOOK_T, 3)])
        self.assertEqual(mp.read_cameras_bin(self.tmp / "cameras.bin"), cams)
        images = mp.read_images_bin(self.tmp / "images.bin")
        self.assertEqual([(i, n, c) for i, n, _, _, c in images], [(7, "photos/a.jpg", 1), (9, "b.jpg", 3)])
        for got, want in zip(images[0][2], Rz):
            self.assertAlmostEqual(got, want, places=12)
        self.assertEqual(images[1][3], LOOK_T)

    def test_undistort_inverts_distort(self):
        cases = [(0, (500.0, 320.0, 240.0)), (1, (500.0, 510.0, 320.0, 240.0)),
                 (2, (500.0, 320.0, 240.0, -0.08)), (3, (500.0, 320.0, 240.0, -0.08, 0.02)),
                 (4, (2562.3, 2566.2, 1546.9, 1177.9, -0.0335, 0.0317, 8.4e-05, 0.00064)),
                 (5, (500.0, 500.0, 320.0, 240.0, 0.01, -0.002, 0.001, 0.0)),
                 (6, (500.0, 500.0, 320.0, 240.0, -0.1, 0.02, 0.001, -0.001, 0.01, 0.0, 0.0, 0.0))]
        for model, params in cases:
            with self.subTest(model=model):
                for x, y in ((0.0, 0.0), (0.3, -0.2), (-0.55, 0.4)):
                    u, v = mp.distort(model, params, x, y)
                    bx, by = mp.undistort(model, params, u, v)
                    self.assertAlmostEqual(bx, x, places=7)
                    self.assertAlmostEqual(by, y, places=7)
        with self.assertRaises(mp.MeshError):
            mp.distort(9, (1.0,), 0.0, 0.0)


class PngTests(TempDirCase):
    def test_masks_with_every_filter(self):
        w, h = 23, 9
        rows = [bytes((255 if (x + 2 * y) % 7 else 0) for x in range(w)) for y in range(h)]
        want = [bytes(1 if v >= 128 else 0 for v in r) for r in rows]
        for filters in ((0,), (1,), (2,), (3,), (4,), (0, 1, 2, 3, 4)):
            with self.subTest(filters=filters):
                (self.tmp / "m.png").write_bytes(png_bytes(w, h, rows, 1, filters))
                self.assertEqual(mp.read_mask_png(self.tmp / "m.png"), (w, h, want))
        # RGB and grey+alpha masks are read by their first channel
        rgb = [bytes(v for x in r for v in (x, 7, 200)) for r in rows]
        (self.tmp / "c.png").write_bytes(png_bytes(w, h, rgb, 3, (4, 1)))
        self.assertEqual(mp.read_mask_png(self.tmp / "c.png"), (w, h, want))
        ga = [bytes(v for x in r for v in (x, 255)) for r in rows]
        (self.tmp / "g.png").write_bytes(png_bytes(w, h, ga, 2, (2, 3)))
        self.assertEqual(mp.read_mask_png(self.tmp / "g.png"), (w, h, want))

    def test_one_bit_mask(self):
        def chunk(kind, body):
            return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
        raw = b"\x00\xa0\x00" + b"\x00\xff\x80"             # 10 px wide: 1010 0000 00 / 1111 1111 10
        data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 10, 2, 1, 0, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
        (self.tmp / "b.png").write_bytes(data)
        self.assertEqual(mp.read_mask_png(self.tmp / "b.png"),
                         (10, 2, [bytes([1, 0, 1, 0, 0, 0, 0, 0, 0, 0]), bytes([1] * 9 + [0])]))

    def test_unsupported_masks_are_refused(self):
        (self.tmp / "x.png").write_bytes(b"GIF89a....")
        with self.assertRaises(mp.MeshError):
            mp.read_mask_png(self.tmp / "x.png")

    def test_padding_rows(self):
        rows = [bytes((x * 9 + y * 31) % 256 for x in range(3 * 5)) for y in range(3)]
        data = png_bytes(5, 3, rows, 3, (4, 1, 2))
        padded = mp.pad_png_rows(data, 8)
        self.assertEqual(mp.image_size(padded), ("image/png", 5, 8))
        w, h, channels, back = decode_png(padded)
        self.assertEqual((w, h, channels), (5, 8, 3))
        self.assertEqual(back[:3], rows)                      # the atlas is untouched
        self.assertEqual(back[3:], [bytes(15)] * 5)           # black below it
        self.assertEqual(padded[-12:], b"\x00\x00\x00\x00IEND\xaeB`\x82")
        self.assertIs(mp.pad_png_rows(data, 3), data)


class OccluderTests(TempDirCase):
    def test_cells_cover_the_masked_pixels(self):
        w, h = 40, 30
        rows = [bytes(0 if (10 <= x < 14 and 5 <= y < 9) else 1 for x in range(w)) for y in range(h)]
        cells = mp.masked_cells(w, h, rows, 4)
        # the 4x4 block sits in cells (2..3, 1..2), grown by one cell on every side
        self.assertEqual(cells, {(x, y) for x in range(1, 5) for y in range(0, 4)})
        self.assertEqual(mp.masked_cells(w, h, [bytes([1] * w)] * h, 4), set())
        corner = [bytes(0 if x == 0 and y == 0 else 1 for x in range(w)) for y in range(h)]
        self.assertEqual(mp.masked_cells(w, h, corner, 4), {(0, 0), (1, 0), (0, 1), (1, 1)})

    def test_triangles_hide_the_masked_area_from_its_camera(self):
        model, cw, ch, params = LOOK_CAMERA
        mw, mh = 32, 24                                      # the mask is at half the camera's size
        cells = {(3, 2), (4, 2), (5, 2), (3, 3)}
        cell = 4
        tris = mp.occluder_triangles(cells, cell, mw, mh, LOOK_CAMERA, LOOK_R, LOOK_T, 0.5)
        C = (2.0, 2.0, -3.0)
        covered = set()
        for tri in tris:
            p = [tri[0:3], tri[3:6], tri[6:9]]
            for q in p:
                self.assertAlmostEqual(q[2] - C[2], 0.5, places=6)    # all at the given depth
            u = [p[1][k] - p[0][k] for k in range(3)]
            v = [p[2][k] - p[0][k] for k in range(3)]
            n = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
            g = [C[k] - sum(q[k] for q in p) / 3 for k in range(3)]
            self.assertLess(sum(a * b for a, b in zip(n, g)), 0.0)    # the back faces the camera
            # project the corners into the mask and record the cells they span
            xs = [(params[0] * (q[0] - C[0]) / (q[2] - C[2]) + params[2]) * mw / cw for q in p]
            ys = [(params[1] * (q[1] - C[1]) / (q[2] - C[2]) + params[3]) * mh / ch for q in p]
            for cx in range(math.floor(min(xs) / cell + 1e-6), math.ceil(max(xs) / cell - 1e-6)):
                for cy in range(math.floor(min(ys) / cell + 1e-6), math.ceil(max(ys) / cell - 1e-6)):
                    covered.add((cx, cy))
        self.assertEqual(covered, cells)
        self.assertEqual(len(tris), 4)                         # two runs in row 2 and 3, two triangles each

    def test_occluders_only_for_photos_with_masked_pixels(self):
        ds = self.tmp / "ds"
        write_model(ds / "sparse" / "0", {1: LOOK_CAMERA},
                    [(1, "photos/a.jpg", LOOK_R, LOOK_T, 1), (2, "photos/b.jpg", LOOK_R, LOOK_T, 1),
                     (3, "photos/c.jpg", LOOK_R, LOOK_T, 1)])
        (ds / "masks" / "photos").mkdir(parents=True)
        person = [bytes(0 if (20 <= x < 30 and 10 <= y < 30) else 255 for x in range(64)) for y in range(48)]
        (ds / "masks" / "photos" / "a.png").write_bytes(png_bytes(64, 48, person))
        (ds / "masks" / "photos" / "b.png").write_bytes(png_bytes(64, 48, [bytes([255] * 64)] * 48))
        xyz = grid(8, 4.0).xyz                                 # the plane z = 0, 3 in front of the camera
        cams = mp.read_cameras_bin(ds / "sparse" / "0" / "cameras.bin")
        imgs = mp.read_images_bin(ds / "sparse" / "0" / "images.bin")
        tris, masked = mp.build_occluders(dataset_dir=ds, cameras=cams, images=imgs, xyz=xyz,
                                          log_path=self.tmp / "log")
        self.assertEqual(masked, 1)
        self.assertTrue(tris)
        diag = math.sqrt(32.0)
        for tri in tris:
            for k in (2, 5, 8):
                depth = tri[k] + 3.0
                self.assertGreater(depth, 0.0)
                self.assertLessEqual(depth, mp.OCCLUDER_MAX_DEPTH * diag + 1e-9)   # well before the wall
        self.assertEqual(mp.build_occluders(dataset_dir=self.tmp, cameras=cams, images=imgs, xyz=xyz,
                                            log_path=self.tmp / "log"), ([], 0))


class ShardTests(unittest.TestCase):
    def test_pixel_sizes_and_depths(self):
        # LOOK camera: f = 40 px, 3 units in front of the plane z = 0, centred on (2, 2).
        cams = {1: LOOK_CAMERA}
        images = [(1, "a.jpg", LOOK_R, LOOK_T, 1)]
        xyz = array.array("f", [2.0, 2.0, 0.0, 2.3, 2.0, 0.0, 2.0, 2.3, 0.0,       # 0.3 wide: 4 px edges
                                1.0, 1.0, 0.0, 3.0, 1.0, 0.0, 1.0, 3.0, 0.0,       # 2 wide: 40 px edges
                                2.0, 2.0, -4.0, 2.1, 2.0, -4.0, 2.0, 2.1, -4.0,    # behind the camera
                                9.0, 2.0, 0.0, 9.1, 2.0, 0.0, 9.0, 2.1, 0.0])      # outside the picture
        mesh = mp.Mesh(xyz, array.array("i", range(12)))
        sizes, depths = mp.face_pixel_sizes(mesh, cams, images)
        self.assertAlmostEqual(sizes[0], 40 * math.hypot(0.3, 0.3) / 3, places=4)
        self.assertAlmostEqual(sizes[1], 40 * math.hypot(2.0, 2.0) / 3, places=4)
        self.assertEqual(sizes[2:], [math.inf, math.inf])
        self.assertAlmostEqual(depths[1], 1.0, places=5)        # as far as the rest of the surface
        self.assertEqual(depths[2:], [0.0, 0.0])
        # a second view from further away sees the face smaller: the smallest size counts
        further = images + [(2, "b.jpg", LOOK_R, (-2.0, -2.0, 30.0), 1)]
        self.assertLess(mp.face_pixel_sizes(mesh, cams, further)[0][1], sizes[1])
        with self.assertRaises(mp.MeshError):
            mp.face_pixel_sizes(mesh, {1: (4, 64, 48, (40.0,) * 8)}, images)

    def test_only_large_faces_far_behind_the_surface_go(self):
        cams = {1: LOOK_CAMERA}
        images = [(1, "a.jpg", LOOK_R, LOOK_T, 1)]
        surface = grid(10, 4.0)                                  # the plane z = 0, 3 in front
        parts = [(list(surface.xyz), list(surface.tris)),
                 ([1.0, 1.0, 0.0, 3.0, 1.0, 0.0, 1.0, 3.0, 0.0], [0, 1, 2]),        # large, near: a wall
                 ([-8.0, -8.0, 9.0, 12.0, -8.0, 9.0, -8.0, 12.0, 9.0], [0, 1, 2]),  # large, far: a shard
                 ([2.0, 2.0, 9.0, 2.2, 2.0, 9.0, 2.0, 2.2, 9.0], [0, 1, 2])]        # small, far: surface
        mesh = merged(*parts)
        kept, removed = mp.drop_shards(mesh, cams, images, 30.0, 2.0)
        self.assertEqual(removed, 1)
        self.assertEqual(kept.face_count, mesh.face_count - 1)
        self.assertNotIn(12.0, set(kept.xyz))
        same, removed = mp.drop_shards(mesh, cams, images, 30.0, 10.0)
        self.assertEqual(removed, 0)
        self.assertIs(same, mesh)


class PlyTests(TempDirCase):
    def test_round_trip_with_extra_triangles(self):
        m = merged(cube())
        extra = [(9.0, 9.0, 9.0, 9.5, 9.0, 9.0, 9.0, 9.5, 9.0)]
        mp.write_ply(self.tmp / "m.ply", m.xyz, m.tris, extra)
        xyz, tris, uv = mp.read_ply(self.tmp / "m.ply")
        self.assertIsNone(uv)
        self.assertEqual(xyz[:24], m.xyz)
        self.assertEqual(list(xyz[24:]), list(extra[0]))
        self.assertEqual(tris[:36], m.tris)
        self.assertEqual(list(tris[36:]), [8, 9, 10])

    def test_reads_what_colmap_writes(self):
        # vertices with colour, CRLF header, faces with texture coordinates
        head = ("ply\r\nformat binary_little_endian 1.0\r\ncomment TextureFile texture.png\r\n"
                "element vertex 3\r\nproperty float x\r\nproperty float y\r\nproperty float z\r\n"
                "property uchar red\r\nproperty uchar green\r\nproperty uchar blue\r\n"
                "element face 1\r\nproperty list uchar int vertex_indices\r\n"
                "property list uchar float texcoord\r\nend_header\r\n").encode()
        body = b"".join(struct.pack("<3f3B", *p, 1, 2, 3) for p in ((0, 0, 0), (1, 0, 0), (0, 1, 0)))
        body += struct.pack("<B3iB6f", 3, 0, 1, 2, 6, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        (self.tmp / "t.ply").write_bytes(head + body)
        xyz, tris, uv = mp.read_ply(self.tmp / "t.ply")
        self.assertEqual(list(xyz), [0, 0, 0, 1, 0, 0, 0, 1, 0])
        self.assertEqual(list(tris), [0, 1, 2])
        self.assertEqual([round(v, 6) for v in uv], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    def test_rejects_what_it_cannot_use(self):
        (self.tmp / "a.ply").write_bytes(b"ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")
        with self.assertRaisesRegex(mp.MeshError, "binary"):
            mp.read_ply(self.tmp / "a.ply")
        head = (b"ply\nformat binary_little_endian 1.0\nelement vertex 4\nproperty float x\n"
                b"property float y\nproperty float z\nelement face 1\n"
                b"property list uchar int vertex_indices\nend_header\n")
        (self.tmp / "q.ply").write_bytes(head + bytes(48) + struct.pack("<B4i", 4, 0, 1, 2, 3))
        with self.assertRaisesRegex(mp.MeshError, "triangles"):
            mp.read_ply(self.tmp / "q.ply")
        (self.tmp / "r.ply").write_bytes(head.replace(b"vertex 4", b"vertex 2") + bytes(24)
                                         + struct.pack("<B3i", 3, 0, 1, 2))
        with self.assertRaisesRegex(mp.MeshError, "does not exist"):
            mp.read_ply(self.tmp / "r.ply")


class AssembleTests(unittest.TestCase):
    def test_coordinates_fill_and_drop(self):
        # A strip of four faces textured by COLMAP except faces 1 and 2, plus a separate face
        # that nothing textured. Atlas 8 x 4 in a padded 8 x 8 image.
        xyz = array.array("f", [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 2, 1, 0, 2, 0, 0, 3, 1, 0,
                                9, 9, 9, 9.5, 9, 9, 9, 9.5, 9])
        tris = array.array("i", [0, 1, 2, 2, 1, 3, 3, 1, 5, 3, 5, 4, 7, 8, 9])
        tris[12:15] = array.array("i", [7, 8, 9])
        xyz.extend([0.0, 0.0, 0.0])                          # vertex 10 unused
        face_uv = array.array("f", [0.1, 0.9, 0.2, 0.9, 0.1, 0.8] + [0.0] * 6 + [0.0] * 6
                              + [0.5, 0.5, 0.6, 0.5, 0.6, 0.6] + [0.0] * 6)
        out_xyz, out_tris, out_uv, stats = mp.assemble_textured(xyz, tris, face_uv, 8, 4, 8)
        self.assertEqual(stats, {"faces_flat_coloured": 2, "faces_unseen_dropped": 1})
        self.assertEqual(len(out_tris), 12)
        corners = [(tuple(out_xyz[3 * i:3 * i + 3]), tuple(round(c, 6) for c in out_uv[2 * i:2 * i + 2]))
                   for i in out_tris]
        # face 0 as COLMAP gave it: u kept, v flipped to glTF's top-left origin and squeezed
        # into the top half of the padded image
        self.assertEqual(corners[0], ((0.0, 0.0, 0.0), (0.1, 0.05)))
        self.assertEqual(corners[1], ((1.0, 0.0, 0.0), (0.2, 0.05)))
        self.assertEqual(corners[2], ((0.0, 1.0, 0.0), (0.1, 0.1)))
        # faces 1 and 2 are one flat colour taken from where they touch textured faces
        for f in (1, 2):
            self.assertEqual(len({c[1] for c in corners[3 * f:3 * f + 3]}), 1)
        self.assertEqual(corners[3][1], corners[2][1])         # face 1 starts at vertex 2 of face 0
        # nothing of the untextured island is left
        self.assertTrue(all(c[0][0] < 5 for c in corners))
        # one output vertex per (position, coordinate) pair
        self.assertEqual(len(out_xyz) // 3, len(set(corners)))


class ScaleSearchTests(unittest.TestCase):
    @staticmethod
    def colmap_like(area_at_one, patches, padding=5):
        """Atlas sizes the way COLMAP lays them out, for patches of total area s^2 * A plus a gutter."""
        calls = []

        def probe(scale):
            calls.append(scale)
            area = scale * scale * area_at_one + patches * padding * (padding + 2 * math.sqrt(
                scale * scale * area_at_one / patches))
            side = math.sqrt(1.3 * area)
            w = 1
            while w < side:
                w *= 2
            h = int(math.ceil(area / (0.9 * w)))
            while h > w:
                w *= 2
                h = int(math.ceil(area / (0.9 * w)))
            return w, h
        return probe, calls

    def test_finds_the_largest_scale_that_fits(self):
        probe, calls = self.colmap_like(186e6, 32000)
        scale, (w, h), probes = mp.fit_texture_scale(probe, 4096)
        self.assertEqual(w, 4096)
        self.assertLessEqual(probes, mp.ATLAS_PROBES)
        self.assertEqual(probe(scale), (w, h))
        # either the atlas is well filled, or a slightly larger scale would not have fitted
        self.assertTrue(h >= mp.ATLAS_GOOD_FILL * 4096
                        or probe(scale * (1 + mp.ATLAS_SCALE_TOLERANCE))[0] > 4096)
        self.assertGreaterEqual(h, 0.6 * 4096)
        self.assertTrue(all(c <= mp.TEXTURE_MAX_SCALE for c in calls))

    def test_never_magnifies_the_photos(self):
        probe, calls = self.colmap_like(1e6, 100)
        scale, (w, h), probes = mp.fit_texture_scale(probe, 8192)
        self.assertEqual(scale, mp.TEXTURE_MAX_SCALE)
        self.assertEqual(probes, 1)
        self.assertLessEqual(w, 8192)

    def test_impossible_budgets_are_explained(self):
        probe, _ = self.colmap_like(1e6, 10 ** 6)          # a million patches cannot fit in 256 x 256
        with self.assertRaisesRegex(mp.MeshError, "texture_size"):
            mp.fit_texture_scale(probe, 256)


class TransformTests(unittest.TestCase):
    def test_points(self):
        xyz = array.array("f", [1, 2, 3, -1, 0, 0.5])
        m = (0.0, -2.0, 0.0, 1.0, 2.0, 0.0, 0.0, -1.0, 0.0, 0.0, 2.0, 0.5)
        self.assertEqual(list(mp.transform_points(xyz, m)), [-3.0, 1.0, 6.5, 1.0, -3.0, 1.5])
        self.assertEqual(mp.transform_points(xyz, mp.IDENTITY), xyz)

    def test_photo_size_limit(self):
        cams = {1: (1, 3000, 2000, (1, 1, 1, 1))}
        few = [(i, "a", LOOK_R, LOOK_T, 1) for i in range(10)]
        self.assertEqual(mp.photo_size_limit(cams, few), -1)
        many = [(i, "a", LOOK_R, LOOK_T, 1) for i in range(1000)]
        limit = mp.photo_size_limit(cams, many)
        self.assertLess(limit, 3000)
        k = limit / 3000
        self.assertLessEqual(1000 * 3000 * 2000 * k * k, mp.PHOTO_PIXEL_BUDGET)
        self.assertGreater(1000 * 3000 * 2000 * k * k, 0.98 * mp.PHOTO_PIXEL_BUDGET)


# ---------------------------------------------------------------------------------------------
# End to end against a stand-in for colmap.exe
# ---------------------------------------------------------------------------------------------

FAKE_COLMAP = r'''
import array, math, os, shutil, struct, sys, time, zlib
from pathlib import Path
sys.path.insert(0, os.environ["FAKE_APP_DIR"])
import meshpost as mp

def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default

def log(text):
    print(text, file=sys.stderr, flush=True)

def fail_if(name, code=1):
    if os.environ.get("FAKE_FAIL") == name:
        log("E: %s failed" % name); sys.exit(code)
    if os.environ.get("FAKE_HANG") == name:
        log("%s: working" % name); time.sleep(600)

assert arg("--log_target") == "stderr"
command = sys.argv[1]
with open(os.environ["FAKE_CALLS"], "a", encoding="utf-8") as calls:
    calls.write(" ".join(sys.argv[1:]) + "\n")
fail_if(command)

if command == "image_undistorter":
    out = Path(arg("--output_path"))
    if not out.parent.is_dir():
        sys.exit(-1073740791)                      # what the real one does: 0xC0000409
    sparse = Path(arg("--input_path"))
    cams = mp.read_cameras_bin(sparse / "cameras.bin")
    images = mp.read_images_bin(sparse / "images.bin")
    wanted = None
    if arg("--image_list_path"):
        wanted = Path(arg("--image_list_path")).read_text(encoding="utf-8").split()
    (out / "sparse").mkdir(parents=True)
    body = bytearray(struct.pack("<Q", len(cams)))
    for cid, (model, w, h, p) in cams.items():
        body += struct.pack("<iiQQ4d", cid, 1, w, h, p[0], p[0], p[2], p[3])
    (out / "sparse" / "cameras.bin").write_bytes(bytes(body))
    shutil.copy(sparse / "images.bin", out / "sparse" / "images.bin")
    for _, name, _, _, _ in images:
        if wanted is None or name in wanted:
            target = out / "images" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(Path(arg("--image_path")) / name, target)
    (out / "stereo").mkdir()
    log("Undistorting image [1/1]")

elif command == "image_deleter":
    drop = Path(arg("--image_names_path")).read_text(encoding="utf-8").split()
    src, dst = Path(arg("--input_path")), Path(arg("--output_path"))
    assert dst.is_dir()
    images = [i for i in mp.read_images_bin(src / "images.bin") if i[1] not in drop]
    body = bytearray(struct.pack("<Q", len(images)))
    raw = (src / "images.bin").read_bytes()
    pos = 8
    for _ in range(struct.unpack_from("<Q", raw, 0)[0]):
        start = pos
        end = raw.index(b"\x00", pos + 64)
        name = raw[pos + 64:end].decode()
        pos = end + 1 + 8 + 24 * struct.unpack_from("<Q", raw, end + 1)[0]
        if name not in drop:
            body += raw[start:pos]
    (dst / "images.bin").write_bytes(bytes(body))
    shutil.copy(src / "cameras.bin", dst / "cameras.bin")

elif command == "patch_match_stereo":
    ws = Path(arg("--workspace_path"))
    cfg = (ws / "stereo" / "patch-match.cfg").read_text(encoding="utf-8").splitlines()
    names = cfg[0::2]
    known = [n for _, n, _, _, _ in mp.read_images_bin(ws / "sparse" / "images.bin")]
    for name in names:
        assert (ws / "images" / name).is_file() and name in known, name
    for kind in ("photometric", "geometric"):
        for k, name in enumerate(names):
            log("=== Processing view %d / %d for %s ===" % (k + 1, len(names), name))
            target = ws / "stereo" / "depth_maps" / (name + "." + kind + ".bin")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"depth")

elif command == "stereo_fusion":
    ws = Path(arg("--workspace_path"))
    names = (ws / "stereo" / "fusion.cfg").read_text(encoding="utf-8").split()
    for k, name in enumerate(names):
        assert (ws / "stereo" / "depth_maps" / (name + ".geometric.bin")).is_file()
        log("Fusing image [%d/%d] with index %d" % (k + 1, len(names), k))
    masks = arg("--StereoFusion.mask_path")
    if masks:
        # what the real tool reads: <mask folder>/<photo name>.png
        listing = sorted(str(f.relative_to(masks)).replace(os.sep, "/") for f in Path(masks).rglob("*.png"))
        with open(os.environ["FAKE_CALLS"], "a") as calls:
            calls.write("fusion-masks %s\n" % " ".join(listing))
    Path(arg("--output_path")).write_bytes(b"ply points")
    Path(arg("--output_path") + ".vis").write_bytes(b"vis")
    log("Number of fused points: 4321")

elif command == "delaunay_mesher":
    ws = Path(arg("--input_path"))
    if not (ws / "fused.ply").is_file() or not (ws / "fused.ply.vis").is_file():
        sys.exit(-1073740791)
    log("Triangulating points...")
    log("Integrating image [1/1]")
    # The scene, in the dataset frame: a 40 x 40 grid over [0, 4]^2 at z = 0 with a ripple at
    # x > 3 (3200 faces), and 30 floaters of 2 faces each, far to the side.
    n = 40
    xyz, tris = [], []
    for j in range(n + 1):
        for i in range(n + 1):
            x, y = 4.0 * i / n, 4.0 * j / n
            xyz += [x, y, 0.05 * (i % 2) * (x > 3.0)]
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            tris += [a, a + 1, a + n + 2, a, a + n + 2, a + n + 1]
    for k in range(30):
        base = len(xyz) // 3
        for dx, dy in ((0, 0), (1, 0), (1, 1), (0, 1)):
            xyz += [10.0 + k + 0.05 * dx, 10.0 + 0.05 * dy, 0.0]
        tris += [base, base + 1, base + 2, base, base + 2, base + 3]
    if os.environ.get("FAKE_SHARD") == "1":
        # what the graph cut leaves in the sky: one huge triangle far behind the scene, the
        # largest piece by area
        base = len(xyz) // 3
        xyz += [-100.0, -50.0, 60.0, 150.0, -50.0, 60.0, -100.0, 100.0, 60.0]
        tris += [base, base + 1, base + 2]
    mp.write_ply(arg("--output_path"), array.array("f", xyz), array.array("i", tris))
    log("Writing surface mesh...")

elif command == "mesh_simplifier":
    # Stand-in decimation: keep the first faces, as many as the ratio allows.
    xyz, tris, _ = mp.read_ply(arg("--input_path"))
    ratio = float(arg("--MeshSimplification.target_face_ratio"))
    keep = max(1, int(math.floor(len(tris) // 3 * ratio)))
    if os.environ.get("FAKE_SIMPLIFY_STUCK") == "1":
        keep = len(tris) // 3
    m = mp.select_faces(mp.Mesh(xyz, tris), [f < keep for f in range(len(tris) // 3)])
    mp.write_ply(arg("--output_path"), m.xyz, m.tris)

elif command == "mesh_texturer":
    scale = float(arg("--MeshTextureMapping.texture_scale_factor"))
    xyz, tris, _ = mp.read_ply(arg("--input_path"))
    n = len(tris) // 3
    with open(os.environ["FAKE_CALLS"], "a") as calls:
        calls.write("faces %d\n" % n)
    log("Loading 1 images...")
    log("Selecting views for %d faces from 1 images..." % n)
    # Faces whose first vertex lies in the band 1.0 <= x < 1.4 get no view; all other faces do.
    unseen = [1.0 <= xyz[3 * tris[3 * f]] < 1.4 for f in range(n)]
    if os.environ.get("FAKE_NO_VIEWS") == "1":
        unseen = [True] * n
    log("%d / %d faces assigned to views" % (unseen.count(False), n))
    log("Found %d regions" % max(1, n // 8))
    # Atlas: the patches take 4e6 * scale^2 px plus a gutter per patch; COLMAP's layout rule.
    area = 4e6 * scale * scale + (n // 8) * float(os.environ.get("FAKE_GUTTER", 25))
    w = 1
    while w < math.sqrt(1.3 * area):
        w *= 2
    h = int(math.ceil(area / (0.85 * w)))
    while h > w:
        w *= 2
        h = int(math.ceil(area / (0.85 * w)))
    h = int(os.environ.get("FAKE_ATLAS_HEIGHT", h))
    log("Atlas size: %d x %d" % (w, h))
    time.sleep(0.2)
    log("Baking texture...")
    out = Path(arg("--output_path"))
    out.mkdir(parents=True, exist_ok=True)
    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
    rows = b"".join(b"\x01" + bytes([200, 100, 50]) + bytes(3 * (w - 1)) for _ in range(h))
    (out / "texture.png").write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                                      + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
    # Each textured face gets its own small triangle in the atlas, laid out on a grid.
    per_row = max(1, int(math.sqrt(n)) + 1)
    head = ("ply\r\nformat binary_little_endian 1.0\r\ncomment TextureFile texture.png\r\nelement vertex %d\r\n"
            "property float x\r\nproperty float y\r\nproperty float z\r\nelement face %d\r\n"
            "property list uchar int vertex_indices\r\nproperty list uchar float texcoord\r\nend_header\r\n"
            % (len(xyz) // 3, n)).encode()
    body = bytearray(head) + xyz.tobytes()
    cell = 1.0 / per_row
    for f in range(n):
        if unseen[f]:
            uv = (0.0,) * 6
        else:
            cx, cy = (f % per_row) * cell, (f // per_row) * cell
            uv = (cx + 0.1 * cell, 1 - (cy + 0.1 * cell), cx + 0.9 * cell, 1 - (cy + 0.1 * cell),
                  cx + 0.1 * cell, 1 - (cy + 0.9 * cell))
        body += struct.pack("<B3iB6f", 3, *tris[3 * f:3 * f + 3], 6, *uv)
    (out / "mesh.ply").write_bytes(bytes(body))
    log("Writing textured mesh to mesh.ply")
else:
    sys.exit(3)
'''


@unittest.skipUnless(os.name == "nt", "the stand-in tool is a .cmd launcher")
class PipelineTests(TempDirCase):
    def setUp(self):
        super().setUp()
        (self.tmp / "fake_colmap.py").write_text(FAKE_COLMAP, encoding="utf-8")
        launcher = '@"%s" "%s" %%*\r\n' % (sys.executable, self.tmp / "fake_colmap.py")
        (self.tmp / "colmap.cmd").write_text(launcher, encoding="ascii")
        # spirula is passed by the pipeline but not used for meshing
        self.tools = {"spirula": str(self.tmp / "no_spirula.exe"), "colmap": str(self.tmp / "colmap.cmd")}
        self.dataset = self.tmp / "dataset"
        self.add_photos(["photos/a.jpg"])
        self.calls = self.tmp / "calls.txt"
        patch = unittest.mock.patch.dict(os.environ, {"FAKE_APP_DIR": str(APP), "FAKE_CALLS": str(self.calls)})
        patch.start()
        self.addCleanup(patch.stop)

    def add_photos(self, names):
        for name in names:
            (self.dataset / "images" / name).parent.mkdir(parents=True, exist_ok=True)
            (self.dataset / "images" / name).write_bytes(b"jpeg")
        write_model(self.dataset / "sparse" / "0", {1: LOOK_CAMERA},
                    [(k + 1, name, LOOK_R, WIDE_T, 1) for k, name in enumerate(names)])

    def make_run(self, name="run", transform=None):
        run = self.tmp / name
        (run / "step-000000100.ckpt").mkdir(parents=True)
        (run / "step-000000100.ckpt" / "splat.ply").write_bytes(b"ply\nend_header\n")
        (run / "config.json").write_text("{}")
        if transform:
            (run / "scene_transform.json").write_text(json.dumps(transform))
        return run

    def build(self, run, **kwargs):
        args = dict(run_dir=run, dataset_dir=self.dataset, out_dir=self.tmp / "out",
                    work_dir=self.tmp / "work", tools=self.tools, visual_faces=1000,
                    collision_faces=200, texture_size=1024, log_path=self.tmp / "mesh.log")
        args.update(kwargs)
        return mp.build_meshes(**args)

    def positions(self, path):
        mesh, _ = mp.read_glb(path)
        return {tuple(mesh.xyz[3 * i:3 * i + 3]) for i in range(mesh.vertex_count)}

    def calls_to(self, command):
        return [line for line in self.calls.read_text(encoding="utf-8").splitlines() if line.startswith(command)]

    def test_builds_both_files(self):
        seen = []
        result = self.build(self.make_run(), on_progress=lambda f, m: seen.append((f, m)))
        mesh = mp.validate_glb(result["mesh"])
        collision = mp.validate_glb(result["collision"])
        self.assertEqual(Path(result["mesh"]).name, "mesh.glb")
        self.assertEqual(Path(result["collision"]).name, "collision.glb")
        self.assertLessEqual(mesh["faces"], 1000)
        self.assertLessEqual(collision["faces"], 200)
        self.assertGreater(mesh["faces"], 900)              # close to the budget, not far under
        self.assertGreater(collision["faces"], 180)
        self.assertEqual(result["mesh_faces"], mesh["faces"])
        self.assertEqual(result["mesh_vertices"], mesh["vertices"])
        self.assertEqual(result["mesh_bytes"], mesh["bytes"])
        self.assertEqual(result["collision_faces"], collision["faces"])
        self.assertEqual(result["collision_vertices"], collision["vertices"])
        self.assertEqual(result["collision_bytes"], collision["bytes"])
        self.assertTrue(mesh["has_uv"] and mesh["has_normals"])
        self.assertTrue(collision["has_normals"])
        self.assertFalse(collision["has_uv"])
        # a square power-of-two texture no larger than asked for
        self.assertEqual((mesh["texture_width"], mesh["texture_height"]), (1024, 1024))
        self.assertEqual(result["texture_size"], 1024)
        self.assertEqual(result["texture_format"], "png")
        self.assertGreater(result["texture_scale"], 0.0)
        self.assertLessEqual(result["texture_scale"], mp.TEXTURE_MAX_SCALE)
        self.assertGreater(result["texture_patches"], 0)
        self.assertGreater(result["faces_flat_coloured"], 0)    # the band the stand-in leaves out
        self.assertEqual(result["faces_unseen_dropped"], 0)
        self.assertEqual(result["masked_photos"], 0)
        self.assertEqual(result["dense_masked_photos"], 0)
        self.assertEqual(self.calls_to("fusion-masks"), [])     # no masks: fusion gets no mask folder
        self.assertEqual(result["geometry"], "colmap-dense")
        self.assertEqual(result["dense_photos"], 1)
        self.assertEqual(result["dense_points"], 4321)
        self.assertEqual(result["raw_faces"], 3260)
        self.assertEqual(result["faces_shards_removed"], 0)
        self.assertEqual(result["raw_components"], 31)
        self.assertEqual(result["mesh_components_kept"], 1)   # the 30 floaters are gone
        self.assertEqual(result["collision_components_kept"], 1)
        self.assertGreaterEqual(result["seconds"], 0)
        self.assertEqual(set(result["timings"]), {"dense", "simplify", "texture", "visual_glb",
                                                  "collision_simplify", "collision_glb", "validate"})
        for gone in ("texture_charts", "faces_texture_stretched", "texture_area_kept", "chart_angle_deg",
                     "merge_factor", "gaussians"):
            self.assertNotIn(gone, result)
        # floaters sat at x >= 10; nothing of them may survive
        self.assertLess(mesh["bbox_max"][0], 5)
        self.assertLess(collision["bbox_max"][0], 5)
        # dense matching on the photo, with the configured number of source photos
        cfg = (self.tmp / "work" / "dense" / "stereo" / "patch-match.cfg").read_text().splitlines()
        self.assertEqual(cfg, ["photos/a.jpg", "__auto__, %d" % mp.MVS_SOURCE_IMAGES])
        self.assertIn("--max_image_size %d" % mp.MVS_IMAGE_SIZE, self.calls_to("image_undistorter")[0])
        self.assertEqual(len(self.calls_to("patch_match_stereo")), 1)
        self.assertEqual(self.calls_to("image_deleter"), [])   # every photo used: nothing to drop
        # the scale was found by probes that stopped early, then one full run
        texturer = self.calls_to("mesh_texturer")
        self.assertGreaterEqual(len(texturer), 2)
        self.assertIn("--MeshTextureMapping.texture_scale_factor %r" % result["texture_scale"], texturer[-1])
        fractions = [f for f, _ in seen]
        self.assertEqual(fractions, sorted(fractions))
        self.assertEqual(fractions[-1], 1.0)
        self.assertTrue(all(0.0 <= f <= 1.0 for f in fractions))
        self.assertLessEqual({"准备照片", "计算深度图", "融合点云", "重建表面", "清除零散碎片", "简化网格",
                              "处理遮罩", "估算贴图尺寸", "生成贴图", "整理贴图", "写出带贴图网格",
                              "简化碰撞网格", "写出碰撞网格", "检查文件", "网格完成"},
                             {m for _, m in seen})
        log = (self.tmp / "mesh.log").read_text(encoding="utf-8", errors="replace")
        for word in ("patch_match_stereo", "stereo_fusion", "delaunay_mesher", "mesh_simplifier",
                     "mesh_texturer", "[meshpost] texture scale probes", "[meshpost] wrote mesh.glb"):
            self.assertIn(word, log)
        self.assertNotIn("spirula", log.lower().replace("spirula-train", ""))

    def test_dense_matching_uses_a_spread_of_the_photos(self):
        names = ["photos/p_%03d.jpg" % k for k in range(9)] + ["v1/f_%03d.jpg" % k for k in range(3)]
        self.add_photos(names)
        with unittest.mock.patch.object(mp, "MVS_MAX_IMAGES", 4):
            result = self.build(self.make_run())
        chosen = mp.mvs_images([(0, n, None, None, 1) for n in names], 4)
        self.assertEqual(len(chosen), 4)
        self.assertEqual(len(set(chosen)), 4)
        self.assertEqual(result["dense_photos"], 4)
        cfg = (self.tmp / "work" / "dense" / "stereo" / "patch-match.cfg").read_text().splitlines()
        self.assertEqual(cfg[0::2], chosen)
        listed = (self.tmp / "work" / "dense_images.txt").read_text().split()
        self.assertEqual(listed, chosen)
        self.assertEqual(len(self.calls_to("image_deleter")), 1)
        kept = [n for _, n, _, _, _ in mp.read_images_bin(self.tmp / "work" / "dense" / "sparse" / "images.bin")]
        self.assertEqual(sorted(kept), sorted(chosen))
        # ...while the texture is made from every photo
        textured = [n for _, n, _, _, _ in mp.read_images_bin(self.tmp / "work" / "undistorted" / "sparse" / "images.bin")]
        self.assertEqual(len(textured), 12)
        self.assertEqual(mp.mvs_images([(0, n, None, None, 1) for n in names], 50), sorted(names))

    def test_shards_are_removed(self):
        with unittest.mock.patch.dict(os.environ, {"FAKE_SHARD": "1"}):
            result = self.build(self.make_run())
        self.assertEqual(result["faces_shards_removed"], 1)
        self.assertEqual(result["raw_faces"], 3261)
        self.assertEqual(result["mesh_components_kept"], 1)
        mesh = mp.validate_glb(result["mesh"])
        self.assertLess(mesh["bbox_max"][2], 0.1)               # the grid, not the shard above it
        self.assertGreater(mesh["bbox_min"][0], -0.01)
        self.assertGreater(mesh["faces"], 900)

    def test_texture_coordinates_point_into_the_atlas(self):
        result = self.build(self.make_run(), visual_faces=5000, collision_faces=5000)
        mesh, image = mp.read_glb(result["mesh"])
        self.assertEqual(result["mesh_faces"], 3200)          # all of the scene, none of the floaters
        self.assertEqual(mp.image_size(image), ("image/png", 1024, 1024))
        w, h, channels, rows = decode_png(image)
        atlas_h = max(i for i, r in enumerate(rows) if r[:3] == bytes([200, 100, 50])) + 1
        self.assertLess(atlas_h, 1024)                        # COLMAP's atlas, padded below
        flat = 0
        for f in range(mesh.face_count):
            corners = [tuple(mesh.uv[2 * i:2 * i + 2]) for i in mesh.tris[3 * f:3 * f + 3]]
            for u, v in corners:
                self.assertLessEqual(v * 1024, atlas_h + 1e-3)   # never into the padding
            if len(set(corners)) == 1:
                flat += 1
            else:
                (u0, v0), (u1, v1), (u2, v2) = corners
                self.assertNotEqual((u1 - u0) * (v2 - v0) - (v1 - v0) * (u2 - u0), 0.0)
        self.assertEqual(flat, result["faces_flat_coloured"])
        self.assertGreater(flat, 100)                         # the band 1.0 <= x < 1.4 had no view
        self.assertEqual(result["collision_faces"], 3200)
        self.assertEqual(result["collision_vertices"], 41 * 41)   # one vertex per position for collision
        self.assertEqual(self.calls_to("mesh_simplifier"), [])   # nothing over budget, nothing simplified

    def test_masked_people_become_occluders(self):
        (self.dataset / "masks" / "photos").mkdir(parents=True)
        person = [bytes(0 if (20 <= x < 30 and 10 <= y < 30) else 255 for x in range(64)) for y in range(48)]
        (self.dataset / "masks" / "photos" / "a.png").write_bytes(png_bytes(64, 48, person, 1, (0, 2, 4)))
        result = self.build(self.make_run())
        self.assertEqual(result["masked_photos"], 1)
        # dense geometry leaves the masked pixels out of the points too: the mask of photo
        # photos/a.jpg is where fusion looks for it, photos/a.jpg.png
        self.assertEqual(result["dense_masked_photos"], 1)
        self.assertEqual(self.calls_to("fusion-masks"), ["fusion-masks photos/a.jpg.png"])
        faces = [int(line.split()[1]) for line in self.calls_to("faces ")]
        _, tris, _ = mp.read_ply(self.tmp / "work" / "texture_in.ply")
        self.assertGreater(len(tris) // 3, result["mesh_faces"])          # occluders handed to COLMAP...
        self.assertEqual(set(faces), {len(tris) // 3})
        mesh = mp.validate_glb(result["mesh"])
        self.assertLessEqual(mesh["faces"], 1000)                           # ...and none of them kept
        self.assertGreater(mesh["bbox_min"][2], -0.01)                      # nothing near the camera

    def test_texture_size_is_rounded_down_to_a_power_of_two(self):
        result = self.build(self.make_run(), texture_size=1500)
        self.assertEqual(result["texture_size"], 1024)

    def test_small_scenes_are_not_magnified(self):
        # at scale 1 the stand-in's atlas is 4096 wide: an 8192 budget is not filled up
        result = self.build(self.make_run(), texture_size=8192)
        self.assertEqual(result["texture_scale"], 1.0)
        self.assertEqual(result["texture_size"], 4096)
        self.assertEqual(len(self.calls_to("mesh_texturer")), 2)    # one probe, one run

    def test_outputs_are_in_the_splat_frame_when_the_run_was_recentred(self):
        fwd = [0.5, 0, 0, 1, 0, 0, -0.5, 2, 0, 0.5, 0, 3]
        inv = [2, 0, 0, -2, 0, 0, 2, -6, 0, -2, 0, 4]
        run = self.make_run("run_t", {"train_from_world": {"matrix_3x4_flat_row_major": fwd},
                                      "world_from_train": {"matrix_3x4_flat_row_major": inv}})
        moved = self.build(run, out_dir=self.tmp / "out_t", work_dir=self.tmp / "work_t")
        plain = self.build(self.make_run("run_p"), out_dir=self.tmp / "out_p", work_dir=self.tmp / "work_p")
        self.assertFalse(moved["scene_transform_is_identity"])
        self.assertTrue(plain["scene_transform_is_identity"])
        self.assertEqual(moved["frame"], "spirula-train")
        # COLMAP reconstructs in the dataset frame; the files hold the same points in the splat's
        world = self.positions(plain["mesh"])
        train = self.positions(moved["mesh"])
        self.assertEqual(len(world), len(train))
        want = {tuple(array.array("f", [fwd[0] * x + fwd[1] * y + fwd[2] * z + fwd[3],
                                        fwd[4] * x + fwd[5] * y + fwd[6] * z + fwd[7],
                                        fwd[8] * x + fwd[9] * y + fwd[10] * z + fwd[11]]))
                for x, y, z in world}
        self.assertEqual(train, want)
        doc_raw = Path(moved["mesh"]).read_bytes()
        doc = json.loads(doc_raw[20:20 + struct.unpack_from("<I", doc_raw, 12)[0]])
        self.assertEqual(doc["nodes"], [{"mesh": 0, "name": "mesh"}])     # no transform on the node
        self.assertEqual(mp.validate_glb(moved["mesh"])["extras"]["frame"],
                         "spirula-train (same numbers as splat.ply)")
        # ...and the texturer was given them back in the dataset frame, where the cameras are
        a, _, _ = mp.read_ply(self.tmp / "work_p" / "texture_in.ply")
        b, _, _ = mp.read_ply(self.tmp / "work_t" / "texture_in.ply")
        self.assertEqual(len(a), len(b))
        self.assertLess(max(abs(p - q) for p, q in zip(a, b)), 1e-5)

    def test_patches_that_do_not_fit_the_texture_get_a_larger_one(self):
        run = self.make_run()
        base = self.build(run)
        self.assertFalse(base["texture_enlarged"])
        self.assertEqual(base["texture_size"], 1024)
        # a gutter so large that the patches cannot fit 1024 x 1024 at any scale
        with unittest.mock.patch.dict(os.environ, {"FAKE_GUTTER": "20000"}):
            result = self.build(run, work_dir=self.tmp / "work2", out_dir=self.tmp / "out2")
        self.assertTrue(result["texture_enlarged"])
        self.assertEqual(result["texture_size"], 2048)
        self.assertEqual(mp.validate_glb(result["mesh"])["texture_width"], 2048)
        # and when not even the largest allowed texture holds them the mesh step says so
        with unittest.mock.patch.dict(os.environ, {"FAKE_GUTTER": "1000000"}):
            with self.assertRaisesRegex(mp.MeshError, "too many separate texture patches"):
                self.build(run, work_dir=self.tmp / "work3", out_dir=self.tmp / "out3")

    def test_an_atlas_other_than_planned_is_refused(self):
        with unittest.mock.patch.dict(os.environ, {"FAKE_ATLAS_HEIGHT": "4096"}):
            with self.assertRaisesRegex(mp.MeshError, "texture"):
                self.build(self.make_run())
        self.assertFalse((self.tmp / "out" / "mesh.glb").exists())

    def test_nothing_textured_is_refused(self):
        with unittest.mock.patch.dict(os.environ, {"FAKE_NO_VIEWS": "1"}):
            with self.assertRaisesRegex(mp.MeshError, "No camera saw"):
                self.build(self.make_run())

    def test_a_simplifier_that_stalls_is_reported(self):
        with unittest.mock.patch.dict(os.environ, {"FAKE_SIMPLIFY_STUCK": "1"}):
            with self.assertRaisesRegex(mp.MeshError, "could not be simplified"):
                self.build(self.make_run())
        self.assertEqual(len(self.calls_to("mesh_simplifier")), mp.SIMPLIFY_PASSES)

    def test_tool_failures_are_explained(self):
        for command, words in (("image_undistorter", "image_undistorter"),
                               ("patch_match_stereo", "NVIDIA GPU"),
                               ("stereo_fusion", "stereo_fusion"),
                               ("delaunay_mesher", "delaunay_mesher"),
                               ("mesh_simplifier", "mesh_simplifier"),
                               ("mesh_texturer", "mesh_texturer")):
            with self.subTest(command), unittest.mock.patch.dict(os.environ, {"FAKE_FAIL": command}):
                with self.assertRaisesRegex(mp.MeshError, words):
                    self.build(self.make_run("run_" + command))
                self.assertFalse((self.tmp / "out" / "mesh.glb").exists())

    def test_stale_outputs_do_not_survive_a_failed_run(self):
        out = self.tmp / "out"
        out.mkdir()
        (out / "mesh.glb").write_bytes(b"old")
        (out / "collision.glb").write_bytes(b"old")
        (self.tmp / "work" / "dense").mkdir(parents=True)
        (self.tmp / "work" / "dense_mesh.ply").write_bytes(b"old")
        with unittest.mock.patch.dict(os.environ, {"FAKE_FAIL": "stereo_fusion"}):
            with self.assertRaises(mp.MeshError):
                self.build(self.make_run())
        self.assertFalse((out / "mesh.glb").exists())
        self.assertFalse((out / "collision.glb").exists())
        self.assertFalse((self.tmp / "work" / "dense_mesh.ply").exists())

    def test_cancel_stops_the_tool(self):
        cancel = self.tmp / "cancel"
        cancel.write_text("x")
        with self.assertRaises(proc.Cancelled):
            self.build(self.make_run(), cancel_file=cancel)
        self.assertFalse((self.tmp / "out" / "mesh.glb").exists())

    def test_cancel_between_steps_is_honoured(self):
        # The cancel file appears while this module, not a tool, is at work.
        cancel = self.tmp / "cancel"

        def on_progress(fraction, message):
            if message == "清除零散碎片":
                cancel.write_text("x")
        with self.assertRaises(proc.Cancelled):
            self.build(self.make_run(), cancel_file=cancel, on_progress=on_progress)
        self.assertFalse((self.tmp / "out" / "mesh.glb").exists())
        self.assertFalse((self.tmp / "out" / "collision.glb").exists())
        self.assertEqual(self.calls_to("mesh_simplifier") + self.calls_to("mesh_texturer"), [])

    def test_hung_tools_are_stopped(self):
        for command in ("patch_match_stereo", "mesh_texturer"):
            with self.subTest(command), unittest.mock.patch.object(mp, "COLMAP_STALL", 2), \
                    unittest.mock.patch.dict(os.environ, {"FAKE_HANG": command}):
                with self.assertRaisesRegex(mp.MeshError, "stopped making progress"):
                    self.build(self.make_run("run_" + command))

    def test_bad_arguments(self):
        run = self.make_run()
        with self.assertRaisesRegex(mp.MeshError, "colmap"):
            self.build(run, tools={"spirula": self.tools["spirula"]})
        with self.assertRaisesRegex(mp.MeshError, "colmap"):
            self.build(run, tools={"colmap": str(self.tmp / "nope.exe")})
        with self.assertRaisesRegex(mp.MeshError, "config.json"):
            self.build(self.tmp / "no_such_run")
        empty = self.tmp / "empty_run"
        empty.mkdir()
        (empty / "config.json").write_text("{}")
        with self.assertRaisesRegex(mp.MeshError, "splat.ply"):
            self.build(empty)
        with self.assertRaisesRegex(mp.MeshError, "images/"):
            self.build(run, dataset_dir=self.tmp)
        bare = self.tmp / "bare"
        (bare / "images").mkdir(parents=True)
        (bare / "sparse").mkdir()
        with self.assertRaisesRegex(mp.MeshError, "camera model"):
            self.build(run, dataset_dir=bare)
        with self.assertRaisesRegex(mp.MeshError, "visual_faces"):
            self.build(run, visual_faces=3)
        with self.assertRaisesRegex(mp.MeshError, "collision_faces"):
            self.build(run, collision_faces=20.5)
        with self.assertRaisesRegex(mp.MeshError, "texture_size"):
            self.build(run, texture_size=10)
        self.assertFalse(self.calls.exists())                 # nothing was started


@unittest.skipUnless(os.environ.get("SCAN3D_MESH_ITEST") == "1", "set SCAN3D_MESH_ITEST=1 to run the real tools")
class RealToolsTest(TempDirCase):
    """Runs COLMAP for real (uses the GPU, about 20 minutes). SCAN3D_MESH_RUN and
    SCAN3D_MESH_DATASET choose the input; the default is the south-building scan (runs/sb-v2)."""

    def test_default_budgets(self):
        run = Path(os.environ.get("SCAN3D_MESH_RUN", ROOT / "runs" / "sb-v2" / "train" / "run"))
        dataset = Path(os.environ.get("SCAN3D_MESH_DATASET", ROOT / "runs" / "sb-v2" / "dataset"))
        if not run.is_dir() or not dataset.is_dir():
            self.skipTest(f"no training run at {run}")
        tools = {"spirula": str(ROOT / "tools" / "spirula" / "spirula.exe"),
                 "colmap": str(ROOT / "tools" / "colmap" / "bin" / "colmap.exe")}
        result = mp.build_meshes(run_dir=run, dataset_dir=dataset, out_dir=self.tmp / "out",
                                 work_dir=self.tmp / "work", tools=tools, log_path=self.tmp / "mesh.log")
        self.assertLessEqual(result["mesh_faces"], 300_000)
        self.assertGreater(result["mesh_faces"], 200_000)
        self.assertLessEqual(result["collision_faces"], 20_000)
        self.assertGreater(result["collision_faces"], 10_000)
        self.assertEqual(result["texture_size"], 4096)
        self.assertTrue(mp.validate_glb(result["mesh"])["has_uv"])
        self.assertFalse(mp.validate_glb(result["collision"])["has_uv"])


if __name__ == "__main__":
    unittest.main()
