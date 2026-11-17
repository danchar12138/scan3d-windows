"""Tests for app/meshpost.py. Standard library only.

    python -m unittest app.tests.test_meshpost          (from the project root)
    python app/tests/test_meshpost.py

The pipeline test drives build_meshes end to end against a stand-in for spirula.exe (a small
Python script behind a .cmd launcher), so it needs no GPU and takes a few seconds. Set
SCAN3D_MESH_ITEST=1 to also run the real tool on runs/sb-std (about four minutes, uses the GPU).
"""
import array
import json
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


class CollapseTests(unittest.TestCase):
    def test_reaches_the_budget_without_moving_a_vertex(self):
        m = grid(20, height=lambda x, y: 0.2 * x * x + 0.1 * y)
        out, stats = mp.collapse_to(m, 200)
        self.assertLessEqual(out.face_count, 200)
        self.assertGreaterEqual(out.face_count, 198)
        self.assertEqual(stats["faces_before"], 800)
        self.assertEqual(stats["faces_after"], out.face_count)
        before = {tuple(m.xyz[3 * i:3 * i + 3]) for i in range(m.vertex_count)}
        after = {tuple(out.xyz[3 * i:3 * i + 3]) for i in range(out.vertex_count)}
        self.assertLessEqual(after, before)                  # a subset: nothing new, nothing moved
        self.assertEqual(max(out.tris), out.vertex_count - 1)
        self.assertIs(mp.drop_degenerate_faces(out, zero_area=True), out)
        self.assertEqual(len(mp.face_components(out)[2]), 1)

    def test_under_budget_is_left_alone(self):
        m = grid(4)
        out, stats = mp.collapse_to(m, 32)
        self.assertEqual(list(out.tris), list(m.tris))
        self.assertEqual(stats["collapses"], 0)
        with self.assertRaises(mp.MeshError):
            mp.collapse_to(m, 2)

    def test_flat_outline_is_kept(self):
        # Everything inside a flat square is free to go; its outline is not.
        out, _ = mp.collapse_to(grid(12), 40)
        self.assertLessEqual(out.face_count, 40)
        self.assertEqual([min(out.xyz[k::3]) for k in range(3)], [0.0, 0.0, 0.0])
        self.assertEqual([max(out.xyz[k::3]) for k in range(3)], [1.0, 1.0, 0.0])
        corners = {(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)}
        self.assertLessEqual(corners, {tuple(out.xyz[3 * i:3 * i + 3]) for i in range(out.vertex_count)})
        area = sum(mp.face_components(out)[2].values())
        self.assertAlmostEqual(area, 1.0, places=5)
        for a, b in border_edges(out):                       # still a square: every open edge on a side
            self.assertTrue(any(a[k] == b[k] and a[k] in (0.0, 1.0) for k in (0, 1)))

    def test_shape_decides_what_goes_first(self):
        # Flat on the left, corrugated on the right: the faces that are left are on the right.
        m = grid(24, height=lambda x, y: 0.0 if x <= 0.5 else 0.05 * ((round(x * 24) + round(y * 24)) % 2))
        out, _ = mp.collapse_to(m, 400)
        left = sum(1 for f in range(out.face_count)
                   if max(out.xyz[3 * i] for i in out.tris[3 * f:3 * f + 3]) <= 0.5)
        self.assertLess(left, 0.2 * out.face_count)

    def test_texture_coordinates_are_kept_and_the_seam_stays_shut(self):
        m = two_chart_grid(16)
        out, stats = mp.collapse_to(m, 150, texels=256.0 * 256.0)
        self.assertLessEqual(out.face_count, 150)
        self.assertEqual(len(out.uv), 2 * out.vertex_count)
        pairs = {(tuple(m.xyz[3 * i:3 * i + 3]), tuple(m.uv[2 * i:2 * i + 2])) for i in range(m.vertex_count)}
        reach = mp.TEXTURE_REACH_TEXELS / 256.0 + 1e-6
        untouched = 0
        for f in range(out.face_count):
            idx = out.tris[3 * f:3 * f + 3]
            coords = [tuple(out.uv[2 * i:2 * i + 2]) for i in idx]
            # the three corners are in one chart, never across the gap between the two
            self.assertEqual(len({c[0] < 0.5 for c in coords}), 1)
            # and the triangle is the same way round in the atlas as in space
            (u0, v0), (u1, v1), (u2, v2) = coords
            self.assertGreater((u1 - u0) * (v2 - v0) - (v1 - v0) * (u2 - u0), 0.0)
            # a corner is a vertex the input had with the coordinates the input gave it, or a
            # coordinate made for it no further outside its chart than the gutter allows
            if all((tuple(out.xyz[3 * i:3 * i + 3]), c) in pairs for i, c in zip(idx, coords)):
                untouched += 1
            for u, v in coords:
                self.assertTrue(0.05 - reach <= u <= 0.45 + reach or 0.55 - reach <= u <= 0.95 + reach)
                self.assertTrue(0.05 - reach <= v <= 0.95 + reach)
        self.assertGreater(untouched, 0.8 * out.face_count)
        self.assertLess(stats["faces_stretched"], 0.1 * out.face_count)
        self.assertGreater(stats["atlas_shown"], 0.9)
        # no crack along the cut: once welded, only the outline of the square is open
        for a, b in border_edges(mp.weld(out)):
            self.assertTrue(any(a[k] == b[k] and a[k] in (0.0, 1.0) for k in (0, 1)))

    def test_a_face_that_leaves_its_chart_keeps_a_patch_of_it(self):
        # Eight faces in two charts, down to four or fewer: far too few to keep every corner
        # inside its own chart. Whatever cannot keep its patch must go on showing a piece of
        # its own chart, pulled out of shape: never one flat colour, and never a triangle
        # stretched across the gap between two charts.
        m = two_chart_grid(2)
        out, stats = mp.collapse_to(m, 4, texels=64.0 * 64.0)
        self.assertLessEqual(out.face_count, 4)
        reach = mp.TEXTURE_REACH_TEXELS / 64.0 + 1e-6
        for f in range(out.face_count):
            coords = [tuple(out.uv[2 * i:2 * i + 2]) for i in out.tris[3 * f:3 * f + 3]]
            self.assertEqual(len({c[0] < 0.5 for c in coords}), 1)
            (u0, v0), (u1, v1), (u2, v2) = coords
            self.assertGreater((u1 - u0) * (v2 - v0) - (v1 - v0) * (u2 - u0), 0.0)
            for u, v in coords:
                self.assertTrue(0.05 - reach <= u <= 0.45 + reach or 0.55 - reach <= u <= 0.95 + reach)
        self.assertGreaterEqual(min(out.uv), 0.0)
        self.assertLessEqual(max(out.uv), 1.0)
        self.assertLessEqual(stats["faces_stretched"], out.face_count)

    def strip_of_charts(self, n=12):
        """A flat strip of n squares along x, each square a chart of its own (two faces), laid
        out in the atlas with a wide gap between neighbours: every vertex is on a chart border."""
        xyz, uv, tris = [], [], []
        for i in range(n):
            base = len(xyz) // 3
            for dx, dy in ((0, 0), (1, 0), (1, 1), (0, 1)):
                xyz += [float(i + dx), float(dy), 0.0]
                uv += [(i + 0.25 + 0.5 * dx) / n, 0.25 + 0.5 * dy]
            tris += [base, base + 1, base + 2, base, base + 2, base + 3]
        return mp.Mesh(array.array("f", xyz), array.array("i", tris), array.array("f", uv))

    def test_coordinates_outside_a_chart_stay_within_reach_of_it(self):
        # Two faces per chart and a third of the faces to go: charts have to be left. With no
        # reach, a corner that leaves keeps the coordinate it had; with some, new coordinates
        # appear, each no further from its chart than the reach.
        n = 12
        m = self.strip_of_charts(n)
        texels = 240.0                                   # so a chart is 10 texels wide, 20 apart
        given = {tuple(m.uv[2 * i:2 * i + 2]) for i in range(m.vertex_count)}
        with unittest.mock.patch.object(mp, "TEXTURE_REACH_TEXELS", 0.0):
            out, stats = mp.collapse_to(m, 16, texels=texels * texels)
        self.assertLessEqual(out.face_count, 16)
        self.assertLessEqual({tuple(out.uv[2 * i:2 * i + 2]) for i in range(out.vertex_count)}, given)
        self.assertGreater(stats["faces_stretched"], 0)
        with unittest.mock.patch.object(mp, "TEXTURE_REACH_TEXELS", 3.0):
            out, stats = mp.collapse_to(m, 16, texels=texels * texels)
        self.assertLessEqual(out.face_count, 16)
        made = {tuple(out.uv[2 * i:2 * i + 2]) for i in range(out.vertex_count)} - given
        self.assertTrue(made)
        for u, v in made:
            chart = min(range(n), key=lambda i: abs(u - (i + 0.5) / n))
            lo, hi = (chart + 0.25) / n, (chart + 0.75) / n
            du = max(lo - u, 0.0, u - hi)
            dv = max(0.25 - v, 0.0, v - 0.75)
            self.assertLessEqual((du * du + dv * dv) ** 0.5 * texels, 3.0 + 1e-3)
        for f in range(out.face_count):                    # and no face lies across two charts
            us = [out.uv[2 * i] for i in out.tris[3 * f:3 * f + 3]]
            self.assertLess(max(us) - min(us), 0.5 / n + 2 * 3.0 / texels + 1e-6)

    def closed_box(self, n=6, bumps=0.0):
        """A cube with n x n squares per side, welded shut; `bumps` pushes every vertex in or out."""
        parts = []
        for axis in range(3):
            for side in (0.0, 1.0):
                g = grid(n)
                xyz = []
                for i in range(g.vertex_count):
                    p = [g.xyz[3 * i], g.xyz[3 * i + 1]]
                    p.insert(axis, side)
                    xyz += p
                tris = list(g.tris)
                if (axis + int(side)) % 2:                  # keep every face pointing outwards
                    tris = [t for f in range(0, len(tris), 3) for t in (tris[f], tris[f + 2], tris[f + 1])]
                parts.append((xyz, tris))
        m = mp.weld(merged(*parts))
        if bumps:
            for i in range(m.vertex_count):
                scale = 1.0 + bumps * (((i * 7919) % 11) / 10.0 - 0.5)
                for k in range(3):
                    m.xyz[3 * i + k] = 0.5 + (m.xyz[3 * i + k] - 0.5) * scale
        return m

    def test_closed_surface_collapses_cleanly(self):
        # Down to a handful of faces: no face twice, nothing torn open.
        m = self.closed_box()
        self.assertEqual(m.vertex_count, 6 * 7 * 7 - 12 * 7 + 8)
        self.assertEqual(border_edges(m), [])
        out, _ = mp.collapse_to(m, 24)
        self.assertLessEqual(out.face_count, 24)
        faces = Counter(tuple(sorted(out.tris[3 * f:3 * f + 3])) for f in range(out.face_count))
        self.assertEqual(max(faces.values()), 1)
        self.assertEqual(border_edges(out), [])
        self.assertEqual([min(out.xyz[k::3]) for k in range(3)], [0.0] * 3)   # still the whole cube
        self.assertEqual([max(out.xyz[k::3]) for k in range(3)], [1.0] * 3)

    def test_budget_is_met_even_where_every_collapse_bends_the_surface(self):
        # A lumpy closed surface, with the turn limit set so tight that no collapse passes it:
        # the first round runs dry at once (one round alone stops at all 432 faces); the second,
        # which only refuses outright flips, gets to the budget.
        lumpy = self.closed_box(bumps=0.3)
        with unittest.mock.patch.object(mp, "COLLAPSE_MAX_TURN_COS", 0.99999):
            out, stats = mp.collapse_to(lumpy, 100)
        self.assertEqual(stats["faces_before"], 432)
        self.assertLessEqual(out.face_count, 100)
        self.assertGreater(out.face_count, 90)
        self.assertIs(mp.drop_degenerate_faces(out), out)

    def test_tick_reports_and_can_stop_the_work(self):
        m = grid(80)                                         # 12,800 faces: several thousand collapses
        seen = []
        mp.collapse_to(m, 2000, tick=seen.append)
        self.assertTrue(seen)
        self.assertEqual(seen, sorted(seen, reverse=True))
        self.assertTrue(all(2000 <= left < 12800 for left in seen))

        def stop(_left):
            raise proc.Cancelled("stop")
        with self.assertRaises(proc.Cancelled):
            mp.collapse_to(m, 2000, tick=stop)


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
# End to end against a stand-in for spirula.exe
# ---------------------------------------------------------------------------------------------

FAKE_TOOL = r'''
import array, os, struct, sys, time, zlib
from pathlib import Path
sys.path.insert(0, os.environ["FAKE_APP_DIR"])
import meshpost as mp

def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default

def png(width, height):
    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
    rows = (b"\x00" + bytes(3 * width)) * height
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))

if sys.argv[1] != "mesh":
    sys.exit(2)
run = Path(sys.argv[2]).name
if "--boom" in run:
    print("[meshing] error: out of memory"); sys.exit(1)
if "--nocam" in run:
    print("[meshing] loading: WARNING: No camera dataset in use: meshing from Gaussian densities only.")
if "--hang" in run:
    print("[meshing] loading: ...", flush=True); time.sleep(600)
assert arg("--lang") == "en" and arg("--format") == "glb+png" and arg("--color") == "texture"
assert arg("--floater-min-faces") == "100" and int(arg("--chart-angle-deg")) == mp.CHART_ANGLE_DEG
assert float(arg("--merge-factor")) in (mp.MERGE_FACTOR, mp.LARGE_SPLAT_MERGE_FACTOR)
assert Path(arg("--data"), "images").is_dir()
size = int(arg("--texture-size"))
assert size & (size - 1) == 0
print("[meshing] loading: Gaussians: 5", flush=True)
# The scene: a 40 x 40 grid over [0, 4]^2 with a ripple, in two charts with the cut along
# x = 2 (3200 faces), and 30 floaters of 2 faces each, a chart apiece, far to the side.
n, half = 40, 20
xyz, uv, tris = [], [], []
for first, u0 in ((0, 0.02), (half, 0.42)):
    base = len(xyz) // 3
    for j in range(n + 1):
        for i in range(half + 1):
            x, y = 4.0 * (first + i) / n, 4.0 * j / n
            xyz += [x, y, 0.05 * ((first + i) % 2) * (x > 3.0)]
            uv += [u0 + 0.36 * i / half, 0.02 + 0.96 * j / n]
    for j in range(n):
        for i in range(half):
            a = base + j * (half + 1) + i
            tris += [a, a + 1, a + half + 2, a, a + half + 2, a + half + 1]
for k in range(30):
    base = len(xyz) // 3
    for dx, dy in ((0, 0), (1, 0), (1, 1), (0, 1)):
        xyz += [10.0 + k + 0.05 * dx, 10.0 + 0.05 * dy, 0.0]
        uv += [0.82 + 0.005 * k + 0.004 * dx, 0.02 + 0.004 * dy]
    tris += [base, base + 1, base + 2, base, base + 2, base + 3]
for stage in ("point cloud", "Delaunay", "occupancy", "cut edges", "bisection", "marching tets",
              "merge", "cull unseen", "merge", "cleanup", "cull unseen", "merge", "cleanup",
              "quality", "orient"):
    print("[meshing] %s: 0.01s" % stage)
print("[meshing] stats: - vertices: %d, faces: %d, components: 31;" % (len(xyz) // 3, len(tris) // 3))
print("[meshing] texel density: cameras rendered: 2/2")
print("[meshing] UV: charts: 32 over faces: %d (0.01s)" % (len(tris) // 3))
print("[meshing] UV: charts parameterized: 32 (split retries: 0, parked: 0) (0.01s)")
print("[meshing] UV: charts packed: 32, efficiency 60%")
size = int(os.environ.get("FAKE_TEXTURE_SIZE", size))
textured = os.environ.get("FAKE_NO_TEXTURE") != "1"
m = mp.Mesh(array.array("f", xyz), array.array("i", tris))
mp.write_glb(arg("--output") + ".glb", positions=m.xyz, indices=array.array("I", tris),
             normals=mp.vertex_normals(m), uvs=array.array("f", uv) if textured else None,
             image=png(size, size) if textured else None)
Path(arg("--output") + ".positions").write_bytes(m.xyz.tobytes())
print("[meshing] bake: the %dx%d texture is finished" % (size, size))
print("[meshing] done: vertices: %d, faces: %d (total 0.1s)" % (len(xyz) // 3, len(tris) // 3))
'''


@unittest.skipUnless(os.name == "nt", "the stand-in tool is a .cmd launcher")
class PipelineTests(TempDirCase):
    def setUp(self):
        super().setUp()
        (self.tmp / "fake_tool.py").write_text(FAKE_TOOL, encoding="utf-8")
        launcher = '@"%s" "%s" %%*\r\n' % (sys.executable, self.tmp / "fake_tool.py")
        (self.tmp / "spirula.cmd").write_text(launcher, encoding="ascii")
        self.tools = {"spirula": str(self.tmp / "spirula.cmd")}     # no other tool exists any more
        self.dataset = self.tmp / "dataset"
        (self.dataset / "images").mkdir(parents=True)
        (self.dataset / "sparse").mkdir()
        patch = unittest.mock.patch.dict(os.environ, {"FAKE_APP_DIR": str(APP)})
        patch.start()
        self.addCleanup(patch.stop)

    def make_run(self, name="run", transform=None, gaussians=5):
        run = self.tmp / name
        (run / "step-000000100.ckpt").mkdir(parents=True)
        (run / "step-000000100.ckpt" / "splat.ply").write_bytes(
            b"ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nend_header\n"
            % gaussians)
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
        return {bytes(mesh.xyz[3 * i:3 * i + 3]) for i in range(mesh.vertex_count)}

    def extracted_positions(self, work="work"):
        raw = (self.tmp / work / "spirula" / "mesh.positions").read_bytes()
        return {raw[i:i + 12] for i in range(0, len(raw), 12)}

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
        self.assertTrue(mesh["has_uv"] and mesh["has_normals"])
        self.assertTrue(collision["has_normals"])
        self.assertFalse(collision["has_uv"])
        self.assertEqual(result["texture_size"], 1024)
        self.assertEqual((mesh["texture_width"], mesh["texture_height"]), (1024, 1024))
        self.assertEqual(result["texture_format"], "png")
        self.assertEqual(result["texture_charts"], 32)
        self.assertEqual(result["raw_faces"], 3260)
        self.assertEqual(result["raw_components"], 31)
        self.assertEqual(result["mesh_components_kept"], 1)   # the 30 floaters are gone
        self.assertEqual(result["collision_components_kept"], 1)
        self.assertEqual(result["merge_factor"], mp.MERGE_FACTOR)
        self.assertEqual(result["merge_factor"], 1.0)
        self.assertEqual(result["chart_angle_deg"], mp.CHART_ANGLE_DEG)
        self.assertEqual(result["gaussians"], 5)
        self.assertGreater(result["texture_area_kept"], 0.9)
        self.assertLess(result["faces_texture_stretched"], 0.1 * mesh["faces"])
        self.assertNotIn("faces_flat_coloured", result)
        self.assertGreaterEqual(result["seconds"], 0)
        # floaters sat at x >= 10; nothing of them may survive
        self.assertLess(mesh["bbox_max"][0], 5)
        self.assertLess(collision["bbox_max"][0], 5)
        # every vertex of either file is a vertex Spirula wrote, at the numbers it wrote
        source = self.extracted_positions()
        self.assertLessEqual(self.positions(result["mesh"]), source)
        self.assertLessEqual(self.positions(result["collision"]), source)
        fractions = [f for f, _ in seen]
        self.assertEqual(fractions, sorted(fractions))
        self.assertEqual(fractions[-1], 1.0)
        self.assertTrue(all(0.0 <= f <= 1.0 for f in fractions))
        self.assertLessEqual({"提取表面", "生成贴图", "清除零散碎片", "写出带贴图网格", "简化碰撞网格",
                              "写出碰撞网格", "检查文件", "网格完成"}, {m for _, m in seen})
        log = (self.tmp / "mesh.log").read_text(encoding="utf-8", errors="replace")
        for word in ("--data", "--color texture", "--merge-factor 1.0", "--chart-angle-deg",
                     "[meshpost] wrote mesh.glb"):
            self.assertIn(word, log)
        self.assertNotIn("colmap", log.lower())

    def test_no_face_of_the_result_is_one_flat_colour(self):
        result = self.build(self.make_run(), visual_faces=400)
        mesh, _ = mp.read_glb(result["mesh"])
        self.assertLessEqual(mesh.face_count, 400)
        for f in range(mesh.face_count):
            (u0, v0), (u1, v1), (u2, v2) = (tuple(mesh.uv[2 * i:2 * i + 2]) for i in mesh.tris[3 * f:3 * f + 3])
            self.assertNotEqual((u1 - u0) * (v2 - v0) - (v1 - v0) * (u2 - u0), 0.0)
        self.assertGreaterEqual(result["faces_texture_stretched"], 0)
        self.assertLessEqual(result["faces_texture_stretched"], mesh.face_count)

    def test_a_very_large_splat_is_extracted_coarser(self):
        result = self.build(self.make_run("run_big", gaussians=mp.LARGE_SPLAT_GAUSSIANS + 1))
        self.assertEqual(result["merge_factor"], mp.LARGE_SPLAT_MERGE_FACTOR)
        self.assertEqual(result["gaussians"], mp.LARGE_SPLAT_GAUSSIANS + 1)
        log = (self.tmp / "mesh.log").read_text(encoding="utf-8", errors="replace")
        self.assertIn("--merge-factor %r" % mp.LARGE_SPLAT_MERGE_FACTOR, log)

    def test_gaussian_count_and_merge_factor(self):
        run = self.make_run("run_n", gaussians=123456)
        self.assertEqual(mp.count_gaussians(run / "step-000000100.ckpt" / "splat.ply"), 123456)
        self.assertEqual(mp.count_gaussians(self.tmp / "no_such.ply"), 0)
        (self.tmp / "odd.ply").write_bytes(b"ply\n")
        self.assertEqual(mp.count_gaussians(self.tmp / "odd.ply"), 0)
        self.assertEqual(mp.merge_factor_for(0), 1.0)                # unknown size: the full surface
        self.assertEqual(mp.merge_factor_for(1_000_000), 1.0)
        self.assertEqual(mp.merge_factor_for(2_000_000), 1.0)        # the high preset's cap
        self.assertEqual(mp.merge_factor_for(mp.LARGE_SPLAT_GAUSSIANS), mp.MERGE_FACTOR)
        self.assertEqual(mp.merge_factor_for(mp.LARGE_SPLAT_GAUSSIANS + 1), mp.LARGE_SPLAT_MERGE_FACTOR)
        self.assertGreater(mp.LARGE_SPLAT_MERGE_FACTOR, mp.MERGE_FACTOR)

    def test_a_mesh_inside_its_budget_is_passed_through_untouched(self):
        result = self.build(self.make_run(), visual_faces=5000, collision_faces=5000)
        self.assertEqual(result["mesh_faces"], 3200)          # all of the scene, none of the floaters
        self.assertEqual(result["faces_texture_stretched"], 0)
        self.assertEqual(result["texture_area_kept"], 1.0)
        mesh, image = mp.read_glb(result["mesh"])
        self.assertEqual(mesh.vertex_count, 2 * 21 * 41)      # seam vertices still present twice
        source = self.extracted_positions()
        self.assertEqual(len(self.positions(result["mesh"])), 41 * 41)
        self.assertLessEqual(self.positions(result["mesh"]), source)
        self.assertEqual(mp.image_size(image), ("image/png", 1024, 1024))
        self.assertGreaterEqual(min(mesh.uv), 0.02 - 1e-6)
        self.assertLessEqual(max(mesh.uv[0::2]), 0.78 + 1e-6)
        self.assertEqual(result["collision_faces"], 3200)
        self.assertEqual(result["collision_vertices"], 41 * 41)   # ...and joined for collision

    def test_texture_size_is_rounded_down_to_a_power_of_two(self):
        result = self.build(self.make_run(), texture_size=1500)
        self.assertEqual(result["texture_size"], 1024)

    def test_outputs_stay_in_the_splat_frame_when_the_run_was_recentred(self):
        fwd = [0.5, 0, 0, 1, 0, 0, -0.5, 2, 0, 0.5, 0, 3]
        inv = [2, 0, 0, -2, 0, 0, 2, -6, 0, -2, 0, 4]
        run = self.make_run("run_t", {"train_from_world": {"matrix_3x4_flat_row_major": fwd},
                                      "world_from_train": {"matrix_3x4_flat_row_major": inv}})
        moved = self.build(run, out_dir=self.tmp / "out_t", work_dir=self.tmp / "work_t")
        plain = self.build(self.make_run("run_p"), out_dir=self.tmp / "out_p", work_dir=self.tmp / "work_p")
        self.assertFalse(moved["scene_transform_is_identity"])
        self.assertTrue(plain["scene_transform_is_identity"])
        self.assertEqual(moved["frame"], "spirula-train")
        # Spirula writes the mesh in the frame of the splat whatever the dataset frame is, so
        # the transform must change nothing: same numbers out as Spirula put in.
        self.assertEqual(self.positions(moved["mesh"]), self.positions(plain["mesh"]))
        self.assertLessEqual(self.positions(moved["mesh"]), self.extracted_positions("work_t"))
        doc_raw = Path(moved["mesh"]).read_bytes()
        doc = json.loads(doc_raw[20:20 + struct.unpack_from("<I", doc_raw, 12)[0]])
        self.assertEqual(doc["nodes"], [{"mesh": 0, "name": "mesh"}])     # no transform on the node
        self.assertEqual(mp.validate_glb(moved["mesh"])["extras"]["frame"],
                         "spirula-train (same numbers as splat.ply)")

    def test_oversized_texture_is_refused(self):
        with unittest.mock.patch.dict(os.environ, {"FAKE_TEXTURE_SIZE": "2048"}):
            with self.assertRaisesRegex(mp.MeshError, "2048 x 2048 texture"):
                self.build(self.make_run())
        self.assertFalse((self.tmp / "out" / "mesh.glb").exists())

    def test_mesh_without_texture_is_refused(self):
        with unittest.mock.patch.dict(os.environ, {"FAKE_NO_TEXTURE": "1"}):
            with self.assertRaisesRegex(mp.MeshError, "without a texture"):
                self.build(self.make_run())

    def test_stale_outputs_do_not_survive_a_failed_run(self):
        out = self.tmp / "out"
        out.mkdir()
        (out / "mesh.glb").write_bytes(b"old")
        (out / "collision.glb").write_bytes(b"old")
        with self.assertRaises(mp.MeshError):
            self.build(self.make_run("run--boom"))
        self.assertFalse((out / "mesh.glb").exists())
        self.assertFalse((out / "collision.glb").exists())

    def test_tool_failure_is_explained(self):
        with self.assertRaisesRegex(mp.MeshError, "out of memory"):
            self.build(self.make_run("run--boom"))

    def test_extraction_without_cameras_is_refused(self):
        with self.assertRaisesRegex(mp.MeshError, "could not read the cameras"):
            self.build(self.make_run("run--nocam"))

    def test_cancel_stops_the_tool(self):
        cancel = self.tmp / "cancel"
        cancel.write_text("x")
        with self.assertRaises(proc.Cancelled):
            self.build(self.make_run(), cancel_file=cancel)
        self.assertFalse((self.tmp / "out" / "mesh.glb").exists())

    def test_cancel_between_steps_is_honoured(self):
        # The cancel file appears while this module, not the tool, is at work.
        cancel = self.tmp / "cancel"

        def on_progress(fraction, message):
            if message == "清除零散碎片":
                cancel.write_text("x")
        with self.assertRaises(proc.Cancelled):
            self.build(self.make_run(), cancel_file=cancel, on_progress=on_progress)
        self.assertFalse((self.tmp / "out" / "mesh.glb").exists())
        self.assertFalse((self.tmp / "out" / "collision.glb").exists())

    def test_hung_tool_is_stopped(self):
        with unittest.mock.patch.object(mp, "SPIRULA_STALL", 2):
            with self.assertRaisesRegex(mp.MeshError, "stopped making progress"):
                self.build(self.make_run("run--hang"))

    def test_bad_arguments(self):
        run = self.make_run()
        with self.assertRaisesRegex(mp.MeshError, "tools"):
            self.build(run, tools={"spirula": "nope.exe"})
        with self.assertRaisesRegex(mp.MeshError, "tools"):
            self.build(run, tools={})
        with self.assertRaisesRegex(mp.MeshError, "config.json"):
            self.build(self.tmp / "no_such_run")
        empty = self.tmp / "empty_run"
        empty.mkdir()
        (empty / "config.json").write_text("{}")
        with self.assertRaisesRegex(mp.MeshError, "splat.ply"):
            self.build(empty)
        with self.assertRaisesRegex(mp.MeshError, "images/"):
            self.build(run, dataset_dir=self.tmp)
        with self.assertRaisesRegex(mp.MeshError, "visual_faces"):
            self.build(run, visual_faces=3)
        with self.assertRaisesRegex(mp.MeshError, "collision_faces"):
            self.build(run, collision_faces=20.5)
        with self.assertRaisesRegex(mp.MeshError, "texture_size"):
            self.build(run, texture_size=10)


@unittest.skipUnless(os.environ.get("SCAN3D_MESH_ITEST") == "1", "set SCAN3D_MESH_ITEST=1 to run the real tool")
class RealToolsTest(TempDirCase):
    """Runs Spirula for real. SCAN3D_MESH_RUN and SCAN3D_MESH_DATASET choose the input; the
    default is the south-building scan made by the pipeline itself (runs/sb-std)."""

    def test_default_budgets(self):
        run = Path(os.environ.get("SCAN3D_MESH_RUN", ROOT / "runs" / "sb-std" / "train" / "run"))
        dataset = Path(os.environ.get("SCAN3D_MESH_DATASET", ROOT / "runs" / "sb-std" / "dataset"))
        if not run.is_dir() or not dataset.is_dir():
            self.skipTest(f"no training run at {run}")
        tools = {"spirula": str(ROOT / "tools" / "spirula" / "spirula.exe")}
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
