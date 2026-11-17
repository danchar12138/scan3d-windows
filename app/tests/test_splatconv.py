"""Tests for app/splatconv.py. Standard library only.

    tools/python/python.exe app/tests/test_splatconv.py

A small synthetic 3DGS PLY (the same layout the trainer writes: 62 float32 per Gaussian) is
converted and decoded again; every value must come back within the quantization step.
"""
from __future__ import annotations

import gzip
import json
import math
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import splatconv  # noqa: E402

PROPERTIES = (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
              + ["f_rest_%d" % i for i in range(45)]
              + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"])


def make_gaussians(count: int, seed: int = 7, extent: float = 3.0) -> list:
    """`count` random Gaussians as dicts of the PLY's own (unactivated) values."""
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        g = {name: rng.uniform(-0.2, 0.2) for name in PROPERTIES}
        g.update(x=rng.uniform(-extent, extent), y=rng.uniform(-extent, extent),
                 z=rng.uniform(-extent, extent),
                 f_dc_0=rng.uniform(-2.5, 2.5), f_dc_1=rng.uniform(-2.5, 2.5),
                 f_dc_2=rng.uniform(-2.5, 2.5), opacity=rng.uniform(-6.0, 8.0),
                 scale_0=rng.uniform(-8.0, 0.5), scale_1=rng.uniform(-8.0, 0.5),
                 scale_2=rng.uniform(-8.0, 0.5),
                 rot_0=rng.uniform(-1, 1), rot_1=rng.uniform(-1, 1),
                 rot_2=rng.uniform(-1, 1), rot_3=rng.uniform(-1, 1))
        out.append(g)
    return out


def write_ply(path: Path, gaussians: list, properties=PROPERTIES, extra_header: str = "",
              types: dict = None) -> None:
    types = types or {}
    header = "ply\nformat binary_little_endian 1.0\ncomment made by a test\n"
    header += "element vertex %d\n" % len(gaussians)
    for name in properties:
        header += "property %s %s\n" % (types.get(name, "float"), name)
    header += extra_header + "end_header\n"
    codes = {"float": "f", "double": "d", "uchar": "B", "int": "i"}
    layout = "<" + "".join(codes[types.get(name, "float")] for name in properties)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for g in gaussians:
            f.write(struct.pack(layout, *[g.get(name, 0) for name in properties]))


def f32(value: float) -> float:
    """What the PLY really holds: the value rounded to float32."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def importance(g: dict) -> float:
    scales = sorted(f32(g["scale_%d" % i]) for i in range(3))
    return sigmoid(f32(g["opacity"])) * math.exp(scales[1] + scales[2])


class SplatConvCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="scan3d-splatconv-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def assert_matches(self, decoded: dict, gaussians: list, sh_step: float = 1 / 128) -> None:
        """Every decoded Gaussian equals its source within the quantization tolerance."""
        self.assertEqual(decoded["count"], len(gaussians))
        step = 2.0 ** -decoded["fractional_bits"]
        for i, g in enumerate(gaussians):
            for axis, name in enumerate("xyz"):
                self.assertAlmostEqual(decoded["positions"][i][axis], f32(g[name]),
                                       delta=step * 0.5 + 1e-7, msg="position %d" % i)
            self.assertAlmostEqual(decoded["opacities"][i], sigmoid(f32(g["opacity"])),
                                   delta=0.5 / 255 + 1e-6, msg="opacity %d" % i)
            for channel in range(3):
                self.assertAlmostEqual(decoded["colours"][i][channel],
                                       f32(g["f_dc_%d" % channel]),
                                       delta=0.5 / (255 * splatconv.COLOUR_SCALE) + 1e-6,
                                       msg="colour %d" % i)
                self.assertAlmostEqual(decoded["scales"][i][channel],
                                       f32(g["scale_%d" % channel]), delta=1 / 32 + 1e-6,
                                       msg="scale %d" % i)
            q = [f32(g["rot_%d" % k]) for k in range(4)]
            norm = math.sqrt(sum(v * v for v in q))
            q = [v / norm for v in q]
            got = decoded["rotations"][i]
            self.assertAlmostEqual(sum(v * v for v in got), 1.0, delta=1e-9)
            # q and -q are one rotation; |<q, got>| is the cosine of half the angle between
            # the two. "Smallest three" with 9-bit magnitudes stays within 0.2 degrees.
            dot = min(1.0, abs(sum(a * b for a, b in zip(q, got))))
            self.assertLess(math.degrees(2 * math.acos(dot)), 0.2, "rotation %d" % i)
            # SH: coefficient k of channel c is f_rest_(c * K + k), K per channel in the PLY
            per_channel = sum(1 for name in g if name.startswith("f_rest_")) // 3
            for k, rgb in enumerate(decoded["sh"][i]):
                for channel in range(3):
                    self.assertAlmostEqual(rgb[channel],
                                           f32(g["f_rest_%d" % (channel * per_channel + k)]),
                                           delta=sh_step / 2 + 1e-6,
                                           msg="SH %d of Gaussian %d" % (k, i))


class RoundTripTests(SplatConvCase):
    def test_round_trip_within_quantization_tolerance(self):
        gaussians = make_gaussians(500)
        ply = self.tmp / "splat.ply"
        out = self.tmp / "out" / "preview.spz"       # the folder is created
        write_ply(ply, gaussians)
        result = splatconv.write_preview(ply, out)
        self.assertEqual(result["gaussians_in"], 500)
        self.assertEqual(result["gaussians_out"], 500)
        self.assertEqual(result["bytes_in"], ply.stat().st_size)
        self.assertEqual(result["bytes_out"], out.stat().st_size)
        self.assertEqual((result["format"], result["spz_version"], result["sh_degree"],
                          result["sh_degree_in"], result["sh_bits"]), ("spz", 3, 3, 3, 8))
        self.assertLess(result["bytes_out"], result["bytes_in"] / 3)    # 65 of 248 bytes, gzipped
        self.assertEqual(list(out.parent.iterdir()), [out])   # no temporary file left behind
        # extent 3 -> 21 fractional bits fit into 24-bit fixed point
        self.assertEqual(result["fractional_bits"], 21)
        decoded = splatconv.read_preview(out)
        self.assertEqual((decoded["sh_degree"], len(decoded["sh"]), len(decoded["sh"][0])),
                         (3, 500, 15))
        self.assert_matches(decoded, gaussians)

    def test_file_is_a_valid_spz_container(self):
        gaussians = make_gaussians(10)
        ply, out = self.tmp / "splat.ply", self.tmp / "preview.spz"
        write_ply(ply, gaussians)
        splatconv.write_preview(ply, out)
        raw = out.read_bytes()
        self.assertEqual(raw[:2], b"\x1f\x8b")                  # gzip
        data = gzip.decompress(raw)
        magic, version, count, sh_degree, fractional_bits, flags, reserved = struct.unpack_from(
            "<IIIBBBB", data)
        self.assertEqual((magic, version, count, sh_degree, flags, reserved),
                         (0x5053474E, 3, 10, 3, 0, 0))
        # header + positions 9 + alpha 1 + colour 3 + scale 3 + rotation 4 bytes per Gaussian,
        # then 15 SH coefficients x 3 channels, one byte each
        self.assertEqual(len(data), 16 + 10 * (20 + 45))
        # the SH section by hand: Gaussian 0, coefficient 0 as red, green, blue; then coefficient 1
        sh = data[16 + 10 * 20:]
        for slot, name in enumerate(("f_rest_0", "f_rest_15", "f_rest_30", "f_rest_1")):
            self.assertEqual(sh[slot], int(f32(gaussians[0][name]) * 128 + 128.5), name)
        splatconv.write_preview(ply, out, sh_degree=0)
        self.assertEqual(len(gzip.decompress(out.read_bytes())), 16 + 10 * 20)
        # first position, by hand: 24-bit little-endian two's complement fixed point
        x = int.from_bytes(data[16:19], "little", signed=False)
        x = x - (1 << 24) if x & 0x800000 else x
        self.assertAlmostEqual(x / (1 << fractional_bits), f32(gaussians[0]["x"]),
                               delta=2.0 ** -fractional_bits)

    def test_known_values(self):
        g = {name: 0.0 for name in PROPERTIES}
        g.update(x=1.0, y=-2.0, z=0.5, opacity=0.0, f_dc_0=0.0, f_dc_1=1.0, f_dc_2=-1.0,
                 scale_0=-10.0, scale_1=0.0, scale_2=-4.0,
                 rot_0=-2.0, rot_1=0.0, rot_2=0.0, rot_3=0.0)       # w < 0, not normalized
        ply, out = self.tmp / "splat.ply", self.tmp / "preview.spz"
        write_ply(ply, [g])
        splatconv.write_preview(ply, out)
        decoded = splatconv.read_preview(out)
        self.assertEqual(decoded["positions"][0], (1.0, -2.0, 0.5))    # exact in fixed point
        self.assertAlmostEqual(decoded["opacities"][0], 0.5, delta=0.5 / 255)
        self.assertEqual(decoded["scales"][0], (-10.0, 0.0, -4.0))
        self.assertAlmostEqual(decoded["colours"][0][0], 0.0, delta=0.014)
        self.assertAlmostEqual(decoded["colours"][0][1], 1.0, delta=0.014)
        self.assertAlmostEqual(decoded["colours"][0][2], -1.0, delta=0.014)
        self.assertEqual(decoded["rotations"][0], (1.0, 0.0, 0.0, 0.0))

    def test_rotations_with_each_component_largest(self):
        """The encoding leaves out the largest component; every case and sign must decode."""
        g = {name: 0.0 for name in PROPERTIES}
        cases = []
        for largest in range(4):
            for sign in (1.0, -1.0):
                q = [0.3, -0.2, 0.1, -0.25]
                q[largest] = 2.0 * sign
                cases.append(dict(g, rot_0=q[0], rot_1=q[1], rot_2=q[2], rot_3=q[3]))
        cases.append(dict(g, rot_0=0.5, rot_1=0.5, rot_2=0.5, rot_3=0.5))       # a four-way tie
        cases.append(dict(g, rot_0=0.0, rot_1=1.0, rot_2=-1.0, rot_3=0.0))      # magnitude at its limit
        ply, out = self.tmp / "splat.ply", self.tmp / "preview.spz"
        write_ply(ply, cases)
        splatconv.write_preview(ply, out)
        self.assert_matches(splatconv.read_preview(out), cases)

    def test_values_outside_the_ranges_are_clamped_not_wrapped(self):
        g = {name: 0.0 for name in PROPERTIES}
        g.update(rot_0=1.0, opacity=50.0, f_dc_0=100.0, f_dc_1=-100.0, scale_0=-40.0,
                 scale_1=30.0, x=5000.0, y=-5000.0)
        h = dict(g, opacity=-50.0, x=0.0, y=0.0)
        ply, out = self.tmp / "splat.ply", self.tmp / "preview.spz"
        write_ply(ply, [g, h])
        result = splatconv.write_preview(ply, out)
        self.assertEqual(result["fractional_bits"], splatconv.MIN_FRACTIONAL_BITS)
        decoded = splatconv.read_preview(out)
        limit = (1 << 23) / (1 << splatconv.MIN_FRACTIONAL_BITS)        # 2048 units
        self.assertAlmostEqual(decoded["positions"][0][0], limit, delta=0.001)
        self.assertAlmostEqual(decoded["positions"][0][1], -limit, delta=0.001)
        self.assertEqual(decoded["opacities"], [1.0, 0.0])
        self.assertAlmostEqual(decoded["colours"][0][0], 0.5 / 0.15, delta=1e-6)
        self.assertAlmostEqual(decoded["colours"][0][1], -0.5 / 0.15, delta=1e-6)
        self.assertEqual(decoded["scales"][0][:2], (-10.0, 255 / 16.0 - 10.0))

    def test_nan_and_infinity_do_not_break_the_conversion(self):
        gaussians = make_gaussians(40, seed=3)
        gaussians[5].update(x=float("nan"), opacity=float("inf"))
        gaussians[9].update(scale_1=float("-inf"), rot_2=float("nan"), f_dc_0=float("nan"))
        gaussians[11].update(rot_0=0.0, rot_1=0.0, rot_2=0.0, rot_3=0.0)   # no rotation at all
        ply, out = self.tmp / "splat.ply", self.tmp / "preview.spz"
        write_ply(ply, gaussians)
        result = splatconv.write_preview(ply, out)
        self.assertEqual(result["gaussians_out"], 40)
        decoded = splatconv.read_preview(out)
        self.assertEqual(decoded["positions"][5][0], 0.0)
        clean = [i for i in range(40) if i not in (5, 9, 11)]
        self.assert_matches({k: ([v[i] for i in clean] if isinstance(v, list) else v)
                             for k, v in dict(decoded, count=len(clean)).items()},
                            [gaussians[i] for i in clean])
        # ... and with a reduction, where the scores see the same bad values
        result = splatconv.write_preview(ply, out, max_gaussians=10)
        self.assertEqual(splatconv.read_preview(out)["count"], 10)

    def test_more_than_one_block(self):
        real = splatconv.BLOCK
        splatconv.BLOCK = 64
        self.addCleanup(setattr, splatconv, "BLOCK", real)
        gaussians = make_gaussians(300, seed=11)
        ply, out = self.tmp / "splat.ply", self.tmp / "preview.spz"
        write_ply(ply, gaussians)
        splatconv.write_preview(ply, out)
        self.assert_matches(splatconv.read_preview(out), gaussians)
        splatconv.write_preview(ply, out, max_gaussians=77)
        keep = sorted(sorted(range(300), key=lambda i: -importance(gaussians[i]))[:77])
        self.assert_matches(splatconv.read_preview(out), [gaussians[i] for i in keep])


class ShTests(SplatConvCase):
    """The view-dependent colour (SH bands f_rest_*)."""

    def test_each_degree(self):
        gaussians = make_gaussians(60, seed=21)
        ply, out = self.tmp / "splat.ply", self.tmp / "preview.spz"
        write_ply(ply, gaussians)
        for degree, coefficients in ((0, 0), (1, 3), (2, 8), (3, 15)):
            result = splatconv.write_preview(ply, out, sh_degree=degree)
            self.assertEqual((result["sh_degree"], result["sh_degree_in"]), (degree, 3))
            self.assertEqual(result["sh_bits"], 8 if degree else None)
            decoded = splatconv.read_preview(out)
            self.assertEqual((decoded["sh_degree"], len(decoded["sh"][7])), (degree, coefficients))
            self.assert_matches(decoded, gaussians)
            header = splatconv.read_preview_header(out)
            self.assertEqual((header["count"], header["sh_degree"], header["spz_version"],
                              header["fractional_bits"]), (60, degree, 3, result["fractional_bits"]))
            self.assertEqual(header["sh_bits"], 8 if degree else None)

    def test_defaults_full_keeps_everything_and_reduced_keeps_none(self):
        """The two calls the pipeline makes, with no SH argument."""
        gaussians = make_gaussians(50, seed=22)
        ply = self.tmp / "splat.ply"
        write_ply(ply, gaussians)
        full = splatconv.write_preview(ply, self.tmp / splatconv.PREVIEW_NAME)
        lite = splatconv.write_preview(ply, self.tmp / splatconv.PREVIEW_LITE_NAME,
                                       max_gaussians=splatconv.LITE_MAX_GAUSSIANS)
        self.assertEqual((full["sh_degree"], full["sh_bits"]),
                         (splatconv.DEFAULT_SH_DEGREE, splatconv.DEFAULT_SH_BITS))
        self.assertEqual((splatconv.DEFAULT_SH_DEGREE, splatconv.DEFAULT_SH_BITS), (3, 8))
        self.assertEqual((lite["sh_degree"], lite["sh_bits"], lite["gaussians_out"]), (0, None, 50))
        self.assertEqual(splatconv.default_sh_degree(None), 3)
        self.assertEqual(splatconv.default_sh_degree(300000), 0)
        # a reduced preview may still be given SH explicitly
        both = splatconv.write_preview(ply, self.tmp / "x.spz", max_gaussians=20, sh_degree=2)
        self.assertEqual((both["sh_degree"], both["gaussians_out"]), (2, 20))
        ranked = sorted(range(50), key=lambda i: -importance(gaussians[i]))
        self.assert_matches(splatconv.read_preview(self.tmp / "x.spz"),
                            [gaussians[i] for i in sorted(ranked[:20])])

    def test_a_splat_with_fewer_bands_gives_what_it_has(self):
        gaussians = make_gaussians(12, seed=23)
        base = [name for name in PROPERTIES if not name.startswith("f_rest_")]
        ply, out = self.tmp / "splat.ply", self.tmp / "preview.spz"
        for degree, per_channel in ((0, 0), (1, 3), (2, 8)):
            properties = base + ["f_rest_%d" % i for i in range(3 * per_channel)]
            write_ply(ply, gaussians, properties)
            self.assertEqual(splatconv.splat_info(ply), {"gaussians": 12, "sh_degree": degree})
            result = splatconv.write_preview(ply, out)           # asks for degree 3
            self.assertEqual((result["sh_degree"], result["sh_degree_in"]), (degree, degree))
            decoded = splatconv.read_preview(out)
            self.assertEqual(len(decoded["sh"][0]), per_channel)
            self.assert_matches(decoded, [{k: g[k] for k in properties} for g in gaussians])
        write_ply(ply, gaussians)
        self.assertEqual(splatconv.splat_info(ply), {"gaussians": 12, "sh_degree": 3})
        (self.tmp / "junk.ply").write_bytes(b"junk")
        with self.assertRaises(splatconv.PreviewError):
            splatconv.splat_info(self.tmp / "junk.ply")

    def test_fewer_bits_round_to_coarser_steps_and_can_be_told_apart(self):
        gaussians = make_gaussians(200, seed=24)
        ply, out = self.tmp / "splat.ply", self.tmp / "preview.spz"
        write_ply(ply, gaussians)
        for bits in (4, 5, 6, 7, 8):
            result = splatconv.write_preview(ply, out, sh_bits=bits)
            self.assertEqual(result["sh_bits"], bits)
            step = (1 << (8 - bits)) / 128.0
            decoded = splatconv.read_preview(out)
            # the coarse step, plus the half byte of the first rounding
            self.assert_matches(decoded, gaussians, sh_step=step + 1 / 128)
            raw = gzip.decompress(out.read_bytes())[16 + 200 * 20:]
            self.assertEqual({b % (1 << (8 - bits)) for b in raw if b != 255}, {0})
            self.assertEqual(splatconv.read_preview_header(out)["sh_bits"], bits)

    def test_sh_values_outside_the_range_are_clamped(self):
        g = {name: 0.0 for name in PROPERTIES}
        g.update(rot_0=1.0, f_rest_0=5.0, f_rest_15=-5.0, f_rest_30=0.5, f_rest_44=-0.25)
        ply, out = self.tmp / "splat.ply", self.tmp / "preview.spz"
        write_ply(ply, [g])
        splatconv.write_preview(ply, out)
        sh = splatconv.read_preview(out)["sh"][0]
        self.assertEqual(sh[0], (127 / 128, -1.0, 0.5))
        self.assertEqual(sh[14], (0.0, 0.0, -0.25))
        self.assertEqual(sh[1], (0.0, 0.0, 0.0))

    def test_header_of_a_file_without_any_view_dependence(self):
        g = {name: 0.0 for name in PROPERTIES}
        g.update(rot_0=1.0)
        ply, out = self.tmp / "splat.ply", self.tmp / "preview.spz"
        write_ply(ply, [g] * 5)
        splatconv.write_preview(ply, out)
        header = splatconv.read_preview_header(out)
        self.assertEqual((header["sh_degree"], header["sh_bits"]), (3, None))   # nothing to measure

    def test_bad_sh_arguments(self):
        ply = self.tmp / "splat.ply"
        write_ply(ply, make_gaussians(3))
        for bad in (-1, 4, 1.0, "3", True):
            with self.assertRaises(ValueError, msg=repr(bad)):
                splatconv.write_preview(ply, self.tmp / "x.spz", sh_degree=bad)
        for bad in (0, 3, 9, 8.0, "8", True):
            with self.assertRaises(ValueError, msg=repr(bad)):
                splatconv.write_preview(ply, self.tmp / "x.spz", sh_bits=bad)
        self.assertFalse((self.tmp / "x.spz").exists())

    def test_read_preview_header_refuses_other_files(self):
        path = self.tmp / "x.spz"
        for content in (b"", b"plain text, not gzip", gzip.compress(bytes(64)),
                        gzip.compress(b"short"), bytes([0x1f, 0x8b]) + b" pipeline full"):
            path.write_bytes(content)
            with self.assertRaises(splatconv.PreviewError, msg=repr(content[:12])):
                splatconv.read_preview_header(path)
        with self.assertRaises(OSError):
            splatconv.read_preview_header(self.tmp / "missing.spz")
        # a file cut short after its header still tells what it claims to be
        ply = self.tmp / "splat.ply"
        write_ply(ply, make_gaussians(30))
        splatconv.write_preview(ply, path)
        whole = gzip.decompress(path.read_bytes())
        path.write_bytes(gzip.compress(whole[:16 + 100]))
        header = splatconv.read_preview_header(path)
        self.assertEqual((header["count"], header["sh_degree"], header["sh_bits"]), (30, 3, None))


class ReductionTests(SplatConvCase):
    def test_max_gaussians_keeps_the_most_visible_in_file_order(self):
        gaussians = make_gaussians(400, seed=5)
        ply, out = self.tmp / "splat.ply", self.tmp / "lite.spz"
        write_ply(ply, gaussians)
        result = splatconv.write_preview(ply, out, max_gaussians=120)
        self.assertEqual((result["gaussians_in"], result["gaussians_out"]), (400, 120))
        ranked = sorted(range(400), key=lambda i: -importance(gaussians[i]))
        keep = sorted(ranked[:120])
        decoded = splatconv.read_preview(out)
        self.assert_matches(decoded, [gaussians[i] for i in keep])
        # the weakest survivor is at least as visible as the strongest one dropped
        self.assertGreaterEqual(min(importance(gaussians[i]) for i in keep),
                                max(importance(gaussians[i]) for i in ranked[120:]))

    def test_limit_at_or_above_the_count_keeps_everything(self):
        gaussians = make_gaussians(30)
        ply, out = self.tmp / "splat.ply", self.tmp / "lite.spz"
        write_ply(ply, gaussians)
        for limit in (30, 31, splatconv.LITE_MAX_GAUSSIANS):
            result = splatconv.write_preview(ply, out, max_gaussians=limit)
            self.assertEqual(result["gaussians_out"], 30)
            self.assert_matches(splatconv.read_preview(out), gaussians)

    def test_ties_keep_file_order(self):
        g = {name: 0.0 for name in PROPERTIES}
        g.update(rot_0=1.0)
        gaussians = [dict(g, x=float(i)) for i in range(10)]     # all equally important
        ply, out = self.tmp / "splat.ply", self.tmp / "lite.spz"
        write_ply(ply, gaussians)
        splatconv.write_preview(ply, out, max_gaussians=4)
        decoded = splatconv.read_preview(out)
        self.assertEqual([p[0] for p in decoded["positions"]], [0.0, 1.0, 2.0, 3.0])

    def test_bad_limits(self):
        ply = self.tmp / "splat.ply"
        write_ply(ply, make_gaussians(3))
        for bad in (0, -1, 1.5, "10", True):
            with self.assertRaises(ValueError, msg=repr(bad)):
                splatconv.write_preview(ply, self.tmp / "x.spz", max_gaussians=bad)
        self.assertFalse((self.tmp / "x.spz").exists())


class InputTests(SplatConvCase):
    def test_other_layouts_are_read_by_property_name(self):
        gaussians = make_gaussians(25, seed=2)
        # no normals, no SH bands, another order, an extra double and an extra byte in between
        properties = ["rot_0", "rot_1", "rot_2", "rot_3", "weight", "x", "y", "z", "flag",
                      "opacity", "scale_0", "scale_1", "scale_2", "f_dc_0", "f_dc_1", "f_dc_2"]
        ply, out = self.tmp / "splat.ply", self.tmp / "preview.spz"
        write_ply(ply, gaussians, properties, types={"weight": "double", "flag": "uchar"},
                  extra_header="element face 0\nproperty list uchar int vertex_indices\n")
        splatconv.write_preview(ply, out)
        self.assert_matches(splatconv.read_preview(out), gaussians)

    def test_empty_splat(self):
        ply, out = self.tmp / "splat.ply", self.tmp / "preview.spz"
        write_ply(ply, [])
        result = splatconv.write_preview(ply, out, max_gaussians=5)
        self.assertEqual(result["gaussians_out"], 0)
        self.assertEqual(splatconv.read_preview(out)["count"], 0)

    def test_unusable_input_is_refused_and_leaves_no_output(self):
        out = self.tmp / "preview.spz"
        cases = {
            "not a ply": b"hello world\n" * 20,
            "ascii": b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nend_header\n0\n",
            "big endian": b"ply\nformat binary_big_endian 1.0\nelement vertex 0\n"
                          b"property float x\nend_header\n",
            "no end": b"ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\n",
            "empty": b"",
        }
        for label, content in cases.items():
            ply = self.tmp / "bad.ply"
            ply.write_bytes(content)
            with self.assertRaises(splatconv.PreviewError, msg=label):
                splatconv.write_preview(ply, out)
        # a point cloud that is not a Gaussian splat
        ply = self.tmp / "points.ply"
        write_ply(ply, make_gaussians(3), ["x", "y", "z"])
        with self.assertRaises(splatconv.PreviewError) as caught:
            splatconv.write_preview(ply, out)
        self.assertIn("opacity", str(caught.exception))
        # a double where a float is needed
        write_ply(ply, make_gaussians(3), types={"opacity": "double"})
        with self.assertRaises(splatconv.PreviewError):
            splatconv.write_preview(ply, out)
        # shorter than the header promises
        write_ply(ply, make_gaussians(8))
        ply.write_bytes(ply.read_bytes()[:-100])
        with self.assertRaises(splatconv.PreviewError) as caught:
            splatconv.write_preview(ply, out)
        self.assertIn("shorter", str(caught.exception))
        with self.assertRaises(OSError):
            splatconv.write_preview(self.tmp / "missing.ply", out)
        self.assertEqual(sorted(p.name for p in self.tmp.iterdir()), ["bad.ply", "points.ply"])

    def test_existing_output_survives_a_failed_conversion(self):
        ply, out = self.tmp / "splat.ply", self.tmp / "preview.spz"
        write_ply(ply, make_gaussians(5))
        splatconv.write_preview(ply, out)
        good = out.read_bytes()
        ply.write_bytes(b"not a ply any more")
        with self.assertRaises(splatconv.PreviewError):
            splatconv.write_preview(ply, out)
        self.assertEqual(out.read_bytes(), good)

    def test_read_preview_refuses_other_files(self):
        path = self.tmp / "x.spz"
        path.write_bytes(gzip.compress(b"\x00" * 64))
        with self.assertRaises(splatconv.PreviewError):
            splatconv.read_preview(path)


class CommandLineTests(SplatConvCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(APP_DIR / "splatconv.py")] + list(args),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)

    def test_command_line(self):
        gaussians = make_gaussians(50)
        ply, out = self.tmp / "splat.ply", self.tmp / "preview-lite.spz"
        write_ply(ply, gaussians)
        done = self.run_cli(str(ply), str(out), "--max", "20")
        self.assertEqual(done.returncode, 0, done.stdout)
        result = json.loads(done.stdout.decode("utf-8").strip().splitlines()[-1])
        self.assertEqual((result["gaussians_in"], result["gaussians_out"]), (50, 20))
        self.assertEqual((result["sh_degree"], result["sh_bits"]), (0, None))
        self.assertEqual(splatconv.read_preview(out)["count"], 20)
        done = self.run_cli(str(ply), str(out))
        result = json.loads(done.stdout.decode("utf-8").strip().splitlines()[-1])
        self.assertEqual((result["gaussians_out"], result["sh_degree"], result["sh_bits"]),
                         (50, 3, 8))
        done = self.run_cli(str(ply), str(out), "--sh", "1", "--sh-bits", "6")
        self.assertEqual(done.returncode, 0, done.stdout)
        result = json.loads(done.stdout.decode("utf-8").strip().splitlines()[-1])
        self.assertEqual((result["sh_degree"], result["sh_bits"]), (1, 6))
        self.assertEqual(len(splatconv.read_preview(out)["sh"][0]), 3)
        self.assertEqual(self.run_cli(str(ply), str(out), "--sh", "4").returncode, 2)

        ply.write_bytes(b"junk")
        failed = self.run_cli(str(ply), str(self.tmp / "other.spz"))
        self.assertEqual(failed.returncode, 1)
        self.assertIn("error", json.loads(failed.stdout.decode("utf-8").strip().splitlines()[-1]))
        self.assertFalse((self.tmp / "other.spz").exists())

    def test_documented_names(self):
        self.assertEqual(splatconv.PREVIEW_NAME, "preview.spz")
        self.assertEqual(splatconv.PREVIEW_LITE_NAME, "preview-lite.spz")
        self.assertEqual(splatconv.LITE_MAX_GAUSSIANS, 300_000)
        for text in (splatconv.PREVIEW_NAME, splatconv.PREVIEW_LITE_NAME, "300,000"):
            self.assertIn(text, splatconv.__doc__)


class SpeedTests(SplatConvCase):
    def test_bulk_speed(self):
        """200,000 Gaussians must take well under the budget of a minute for 2,000,000."""
        one = struct.pack("<62f", *[0.25] * 54, 1.0, -3.0, -4.0, -5.0, 0.5, 0.5, 0.5, 0.5)
        ply = self.tmp / "big.ply"
        header = "ply\nformat binary_little_endian 1.0\nelement vertex 200000\n"
        header += "".join("property float %s\n" % name for name in PROPERTIES) + "end_header\n"
        with open(ply, "wb") as f:
            f.write(header.encode("ascii"))
            block = one * 10000
            for _ in range(20):
                f.write(block)
        result = splatconv.write_preview(ply, self.tmp / "big.spz", max_gaussians=60000)
        self.assertEqual(result["gaussians_out"], 60000)
        self.assertLess(result["seconds"], 6.0)
        # ... and with all 45 SH coefficients of every Gaussian
        result = splatconv.write_preview(ply, self.tmp / "big.spz")
        self.assertEqual((result["gaussians_out"], result["sh_degree"]), (200000, 3))
        self.assertLess(result["seconds"], 12.0)


if __name__ == "__main__":
    unittest.main(verbosity=1)
