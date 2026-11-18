"""Tests for app/coverage.py. Standard library only.

    tools/python/python.exe app/tests/test_coverage.py

The scenes are synthetic: a long front wall photographed from an arc of cameras, and a small
back wall hidden behind it. What the real scans showed is in coverage.py's docstring."""
from __future__ import annotations

import base64
import contextlib
import io
import json
import math
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import coverage  # noqa: E402


def write_ply(path: Path, points, opacity: float = 3.0) -> None:
    """A binary PLY with the properties coverage.py reads (and a colour, as a real one has)."""
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"]
    header = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % len(points)
    header += "".join("property float %s\n" % n for n in names) + "end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for x, y, z in points:
            f.write(struct.pack("<7f", x, y, z, 0.5, 0.5, 0.5, opacity))


def quaternion(rows):
    """(w, x, y, z) of a rotation matrix given by rows."""
    (m00, m01, m02), (m10, m11, m12), (m20, m21, m22) = rows
    w = math.sqrt(max(0.0, 1 + m00 + m11 + m22)) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + m00 - m11 - m22)) / 2, m21 - m12)
    y = math.copysign(math.sqrt(max(0.0, 1 - m00 + m11 - m22)) / 2, m02 - m20)
    z = math.copysign(math.sqrt(max(0.0, 1 - m00 - m11 + m22)) / 2, m10 - m01)
    return w, x, y, z


def arc_cameras(count=15, radius=2.5, degrees=75.0):
    """Cameras on an arc in front of the origin (at negative z), each looking at the origin;
    image y (down) is world +y, so the scene's up is -y. -> [(name, rows, t)]"""
    out = []
    for i in range(count):
        a = math.radians(-degrees + 2 * degrees * i / (count - 1))
        centre = (radius * math.sin(a), 0.0, -radius * math.cos(a))
        forward = (-math.sin(a), 0.0, math.cos(a))
        down = (0.0, 1.0, 0.0)
        right = (math.cos(a), 0.0, math.sin(a))          # down x forward
        rows = (right, down, forward)
        t = tuple(-sum(rows[r][c] * centre[c] for c in range(3)) for r in range(3))
        out.append(("cam_%02d.jpg" % i, rows, t))
    return out


def write_images_bin(path: Path, cameras, camera_id=1) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for index, (name, rows, t) in enumerate(cameras):
            f.write(struct.pack("<I7dI", index + 1, *quaternion(rows), *t, camera_id))
            f.write(name.encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", 1))
            f.write(struct.pack("<ddq", 1.0, 2.0, -1))


def write_cameras_bin(path: Path, model=1, params=(500.0, 500.0, 320.0, 240.0), size=(640, 480)) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<IiQQ", 1, model, size[0], size[1]))
        f.write(struct.pack("<%dd" % len(params), *params))


def grid(x0, x1, y0, y1, z, step=0.04):
    nx, ny = int(round((x1 - x0) / step)) + 1, int(round((y1 - y0) / step)) + 1
    return [(x0 + i * step, y0 + j * step, z) for i in range(nx) for j in range(ny)]


FRONT = grid(-3.0, 3.0, -1.0, 1.0, 0.0)          # long, faces the cameras
BACK = grid(-0.3, 0.3, -0.4, 0.4, 1.0, 0.02)       # behind it: no camera can see it
UP = (0.0, -1.0, 0.0)


def decode(text, code):
    raw = base64.b64decode(text)
    return list(struct.unpack("<%d%s" % (len(raw) // struct.calcsize(code), code), raw))


class Scene(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="scan3d-coverage-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.ply = self.tmp / "splat.ply"
        self.images = self.tmp / "images.bin"
        self.cameras = self.tmp / "cameras.bin"
        write_ply(self.ply, FRONT + BACK)
        write_images_bin(self.images, arc_cameras())
        write_cameras_bin(self.cameras)

    def levels_by_wall(self, result):
        """{front: {level: n}, back: {level: n}} by voxel, plus the views of each wall."""
        g = result["grid"]
        keys = decode(result["voxels"]["keys"], "I")
        level = list(base64.b64decode(result["voxels"]["level"]))
        views = list(base64.b64decode(result["voxels"]["views"]))
        nx, ny, _nz = g["dims"]
        out = {"front": {}, "back": {}, "front_views": [], "back_views": []}
        for k, lv, v in zip(keys, level, views):
            x = g["origin"][0] + (k % nx + 0.5) * g["voxel"]
            z = g["origin"][2] + (k // (nx * ny) + 0.5) * g["voxel"]
            if z < 0.5 and abs(x) > 1.0:
                continue        # the ends of the long wall: only the outer photos see them
            wall = "back" if z > 0.5 else "front"
            out[wall][lv] = out[wall].get(lv, 0) + 1
            out[wall + "_views"].append(v)
        return out

    def compute(self, **kwargs):
        kwargs.setdefault("up", UP)
        return coverage.compute(self.ply, self.images, self.cameras, **kwargs)


class MeasureTests(Scene):
    def test_the_hidden_wall_is_weak_and_the_photographed_one_strong(self):
        result = self.compute()
        walls = self.levels_by_wall(result)
        front, back = walls["front"], walls["back"]
        self.assertGreater(front.get(2, 0), 0.6 * sum(front.values()), front)     # mostly strong
        self.assertEqual(set(back), {0}, back)                                    # all weak
        # no photo sees through the wall (one may graze a voxel at its very edge)
        self.assertEqual(sorted(walls["back_views"])[len(walls["back_views"]) // 2], 0)
        self.assertLessEqual(max(walls["back_views"]), 1)
        # the middle of the front wall is seen by (nearly) every photo
        self.assertGreaterEqual(sorted(walls["front_views"])[len(walls["front_views"]) // 2], 10)
        self.assertEqual(result["levels"]["weak"] + result["levels"]["medium"] + result["levels"]["strong"],
                         result["voxels"]["count"])

    def test_without_the_depth_test_the_back_of_the_wall_would_count_as_seen(self):
        """This is why occlusion matters: every camera has the hidden wall in its picture."""
        walls = self.levels_by_wall(self.compute(occlusion=False))
        self.assertGreaterEqual(min(walls["back_views"]), 10)
        self.assertNotIn(0, walls["back"])

    def test_a_place_photographed_from_one_side_only_is_weak(self):
        # every camera bunched within a few degrees: plenty of photos, one direction
        write_images_bin(self.images, arc_cameras(count=12, degrees=4))
        walls = self.levels_by_wall(self.compute())
        middle = sorted(walls["front_views"])[len(walls["front_views"]) // 2]
        self.assertGreaterEqual(middle, 8)                  # many photos ...
        self.assertNotIn(2, walls["front"])                 # ... yet nowhere strong
        self.assertGreater(walls["front"].get(0, 0), 0)

    def test_regions_sentence_and_a_place_to_stand(self):
        result = self.compute()
        regions = result["regions"]
        self.assertEqual([r["number"] for r in regions], list(range(1, len(regions) + 1)))
        # the hidden wall (the far ends of the long wall lie outside what the photos aimed at)
        reasons = sorted(r["reason"] for r in regions)
        self.assertEqual(reasons.count("unseen"), 1)
        self.assertTrue(set(reasons) <= {"few", "one_side", "unseen"}, reasons)
        first = [r for r in regions if r["reason"] == "unseen"][0]
        self.assertGreater(first["center"][2], 0.9)          # the hidden wall
        # it is in the pictures, only hidden: the sentence says so, not "not photographed"
        self.assertGreater(first["attention"], 0)
        self.assertIn("挡住了", first["text"])
        self.assertNotIn("没有照片拍到", first["text"])
        self.assertTrue(first["nearest_photo"].startswith("cam_"))
        suggest = first["suggest"]
        self.assertIsNotNone(suggest)
        # it stands behind the front wall (z > 0), where the hidden wall can be seen
        self.assertGreater(suggest["position"][2], 0.05)
        self.assertEqual(suggest["target"], first["center"])
        self.assertFalse(suggest["elevated"])
        self.assertEqual(set(suggest), {"position", "target", "elevated"})
        # the voxels of the region carry its number
        region = list(base64.b64decode(result["voxels"]["region"]))
        for r in regions:
            self.assertEqual(region.count(r["number"]), r["voxels"])
        self.assertTrue(set(region) <= set(range(len(regions) + 1)))
        self.assertLessEqual(len(regions), coverage.MAX_REGIONS)

    def test_output_shape(self):
        result = self.compute()
        self.assertEqual((result["format"], result["version"]), ("scan3d-coverage", coverage.FORMAT_VERSION))
        n = result["voxels"]["count"]
        keys = decode(result["voxels"]["keys"], "I")
        self.assertEqual(len(keys), n)
        self.assertEqual(keys, sorted(keys))
        for name in ("level", "views", "bins", "region"):
            self.assertEqual(len(base64.b64decode(result["voxels"][name])), n, name)
        self.assertEqual(len(result["cameras"]), 15)
        self.assertEqual(set(result["cameras"][0]), {"position", "forward", "name"})
        self.assertAlmostEqual(result["cameras"][7]["position"][2], -2.5, places=4)   # the middle one
        for a, b in zip(result["cameras"][7]["forward"], (0, 0, 1)):
            self.assertAlmostEqual(a, b, places=4)
        self.assertEqual(result["up"], [0.0, -1.0, 0.0])
        self.assertGreater(result["typical_distance"], 1.5)
        self.assertEqual(result["gaussians"], len(FRONT) + len(BACK))
        self.assertIn("z-buffer", result["method"])
        json.dumps(result)          # plain JSON

    def test_up_is_guessed_from_the_cameras(self):
        result = coverage.compute(self.ply, self.images, self.cameras)
        for a, b in zip(result["up"], UP):
            self.assertAlmostEqual(a, b, places=5)

    def test_the_training_frame_may_differ_from_the_dataset_frame(self):
        """splat.ply in the training frame p' = s R p + t, cameras in the dataset frame: the
        result is the same scene, in the training frame's numbers."""
        reference = self.compute()
        s, rows, t = 2.0, [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [0.5, -1.0, 3.0]

        def move(p):
            return tuple(s * sum(rows[r][c] * p[c] for c in range(3)) + t[r] for r in range(3))
        write_ply(self.ply, [move(p) for p in FRONT + BACK])
        up = tuple(sum(rows[r][c] * UP[c] for c in range(3)) for r in range(3))
        moved = coverage.compute(self.ply, self.images, self.cameras, transform=(s, rows, t), up=up)
        # the same up to voxels cut differently by the turned grid
        for name in ("weak", "medium", "strong"):
            self.assertAlmostEqual(moved["levels"][name], reference["levels"][name],
                                   delta=0.02 * reference["voxels"]["count"])
        self.assertEqual(len(moved["regions"]), len(reference["regions"]))
        for got, want in zip(moved["cameras"][3]["position"], move(reference["cameras"][3]["position"])):
            self.assertAlmostEqual(got, want, places=3)
        for got, want in zip(moved["regions"][0]["center"], move(reference["regions"][0]["center"])):
            self.assertAlmostEqual(got, want, delta=0.2)
        self.assertAlmostEqual(moved["typical_distance"], 2 * reference["typical_distance"], delta=0.05)

    def test_broken_values_are_skipped(self):
        nan, inf = float("nan"), float("inf")
        clean = self.compute()
        write_ply(self.ply, FRONT + BACK + [(nan, 0, 0), (0, inf, 0), (0, 0, -inf)] * 10)
        result = self.compute()
        self.assertEqual(result["gaussians_used"], clean["gaussians_used"])
        self.assertEqual(result["levels"], clean["levels"])
        # faint Gaussians do not make surfaces
        write_ply(self.ply, FRONT + BACK, opacity=-6.0)
        with self.assertRaises(coverage.CoverageError):
            self.compute()


def look_at(centre, target, up=(0.0, 0.0, 1.0)):
    """COLMAP rows (right, down, forward) and t of a camera at centre looking at target."""
    f = [target[a] - centre[a] for a in range(3)]
    n = math.sqrt(sum(v * v for v in f))
    f = [v / n for v in f]
    r = [f[1] * up[2] - f[2] * up[1], f[2] * up[0] - f[0] * up[2], f[0] * up[1] - f[1] * up[0]]
    n = math.sqrt(sum(v * v for v in r))
    r = [v / n for v in r]
    d = [f[1] * r[2] - f[2] * r[1], f[2] * r[0] - f[0] * r[2], f[0] * r[1] - f[1] * r[0]]
    rows = (tuple(r), tuple(d), tuple(f))
    t = tuple(-sum(rows[i][c] * centre[c] for c in range(3)) for i in range(3))
    return rows, t


class ObjectInABigPlaceTests(unittest.TestCase):
    """A small object walked around closely, in a street ten times its size (the case of a
    truck filmed in a square): the object must be judged at its own scale, the far
    surroundings left out, and the verdict must be about the object."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="scan3d-coverage-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        step = 0.025
        box = []
        k = int(round(1.0 / step))
        for i in range(k + 1):
            for j in range(int(round(0.6 / step)) + 1):
                a, h = -0.5 + i * step, j * step
                box += [(a, -0.5, h), (a, 0.5, h), (-0.5, a, h), (0.5, a, h)]
            for j in range(k + 1):
                box.append((-0.5 + i * step, -0.5 + j * step, 0.6))
        street = [(-10 + i * 0.2, -10 + j * 0.2, 0.0) for i in range(101) for j in range(101)]
        walls = [(x, -10 + j * 0.2, h * 0.2) for x in (-10.0, 10.0) for j in range(101) for h in range(15)]
        write_ply(self.tmp / "splat.ply", box + street + walls)
        cams = []
        for i in range(24):
            a = 2 * math.pi * i / 24
            rows, t = look_at((1.6 * math.cos(a), 1.6 * math.sin(a), 1.0), (0.0, 0.0, 0.3))
            cams.append(("ring_%02d.jpg" % i, rows, t))
        write_images_bin(self.tmp / "images.bin", cams)
        write_cameras_bin(self.tmp / "cameras.bin")
        self.result = coverage.compute(self.tmp / "splat.ply", self.tmp / "images.bin",
                                       self.tmp / "cameras.bin", up=(0, 0, 1))

    def test_the_object_is_judged_at_its_own_scale_and_comes_out_well_covered(self):
        result = self.result
        self.assertLess(result["typical_distance"], 2.0)       # the object, not the street
        self.assertLess(result["grid"]["voxel"], 0.06)         # >= 16 voxels across the object
        g = result["grid"]
        nx, ny, _nz = g["dims"]
        keys = decode(result["voxels"]["keys"], "I")
        level = list(base64.b64decode(result["voxels"]["level"]))
        on_object, far = [], 0
        for k, lv in zip(keys, level):
            x = g["origin"][0] + (k % nx + 0.5) * g["voxel"]
            y = g["origin"][1] + ((k // nx) % ny + 0.5) * g["voxel"]
            z = g["origin"][2] + (k // (nx * ny) + 0.5) * g["voxel"]
            if abs(x) <= 0.55 and abs(y) <= 0.55 and z > 0.05:
                on_object.append(lv)
            if math.hypot(x, y) > 4:
                far += 1
        self.assertEqual(far, 0)                               # the far street is not judged
        self.assertGreater(on_object.count(2), 0.8 * len(on_object))
        self.assertGreater(result["surface_share"]["strong"], 0.5)
        self.assertGreater(result["surface_share"]["strong"], result["surface_share"]["weak"])


class ReadingTests(Scene):
    def test_camera_models(self):
        for model, params, fx, fy in ((0, (400.0, 320.0, 240.0), 400, 400),          # SIMPLE_PINHOLE
                                      (1, (400.0, 410.0, 320.0, 240.0), 400, 410),   # PINHOLE
                                      (2, (400.0, 320.0, 240.0, 0.01), 400, 400),    # SIMPLE_RADIAL
                                      (4, (400.0, 410.0, 320.0, 240.0, 0.01, 0.0, 0.0, 0.0), 400, 410),  # OPENCV
                                      (5, (400.0, 410.0, 320.0, 240.0, 0.1, 0.0, 0.0, 0.0), 400, 410)):  # OPENCV_FISHEYE
            write_cameras_bin(self.cameras, model, params)
            intrinsics = coverage.read_intrinsics(self.cameras)
            self.assertEqual(intrinsics[1], (640, 480, fx, fy, 320.0, 240.0), model)
        write_cameras_bin(self.cameras, 99, (1.0,))
        with self.assertRaises(coverage.CoverageError):
            coverage.read_intrinsics(self.cameras)

    def test_not_a_splat(self):
        for content in (b"", b"not a ply", b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nend_header\n",
                        b"ply\nformat binary_little_endian 1.0\nelement vertex 5\nproperty float x\n"
                        b"property float y\nproperty float z\nproperty float opacity\nend_header\n" + b"\0" * 7,
                        b"ply\nformat binary_little_endian 1.0\nelement vertex 1\n"):
            self.ply.write_bytes(content)
            with self.assertRaises(coverage.CoverageError, msg=content[:40]):
                coverage.read_splat(self.ply)

    def test_poses_with_broken_numbers_are_skipped(self):
        cams = arc_cameras(count=3)
        cams[1] = (cams[1][0], cams[1][1], (float("nan"), 0.0, 0.0))
        write_images_bin(self.images, cams)
        self.assertEqual([p[0] for p in coverage.read_poses(self.images)], ["cam_00.jpg", "cam_02.jpg"])


class CommandLineTests(Scene):
    def run_main(self, args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = coverage.main([str(a) for a in args])
        lines = out.getvalue().strip().splitlines()
        return code, json.loads(lines[-1]) if lines else None

    def test_writes_the_map_and_says_how_long_it_took(self):
        target = self.tmp / "coverage.json"
        transform = self.tmp / "scene_transform.json"
        transform.write_text(json.dumps({"train_from_world": {"scale": 1, "translation": [0, 0, 0],
                                         "rotation": {"matrix_3x3": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}}}))
        code, said = self.run_main([self.ply, self.images, self.cameras, target,
                                    "--transform", transform, "--up", "0,-1,0"])
        self.assertEqual(code, 0, said)
        self.assertTrue(said["ok"])
        self.assertIn("total", said["seconds"])
        written = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(written["format"], "scan3d-coverage")
        self.assertEqual(written["levels"], said["levels"])
        self.assertEqual([p.name for p in self.tmp.iterdir() if p.name.endswith(".part")], [])

    def test_failures_are_one_json_line_and_exit_code_1(self):
        target = self.tmp / "coverage.json"
        self.ply.write_bytes(b"junk")
        code, said = self.run_main([self.ply, self.images, self.cameras, target])
        self.assertEqual(code, 1)
        self.assertIn("PLY", said["error"])
        self.assertFalse(target.exists())
        write_ply(self.ply, FRONT)
        for extra in (["--up", "0,0"], ["--up", "nan,0,1"], ["--up", "0,0,0"]):
            code, said = self.run_main([self.ply, self.images, self.cameras, target] + extra)
            self.assertEqual(code, 1, extra)
        bad = self.tmp / "bad.json"
        bad.write_text('{"train_from_world": {"rotation": 5}}')
        code, said = self.run_main([self.ply, self.images, self.cameras, target, "--transform", bad])
        self.assertEqual(code, 1)
        code, said = self.run_main([self.ply, self.tmp / "missing.bin", self.cameras, target])
        self.assertEqual(code, 1)
        self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
