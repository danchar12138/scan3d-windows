"""How well the photos of a finished scan saw each part of the scene. Standard library only.

Nothing can recover what the camera never saw, so the useful answer to "this part looks bad" is
"these parts were photographed too little; stand here and take a few more". This module works
out the first half from the training cameras and the trained splat, and suggests the second.

    compute(splat_ply, images_bin, cameras_bin, *, transform=None, up=None) -> dict
        transform: (scale, 3x3 rotation, translation) of train_from_world (scene_transform.json),
                   None for none; up: the scene's up direction in splat coordinates
                   (report.json orientation.up_in_file_coordinates), None to guess it from the
                   cameras. Raises CoverageError (a ValueError) for unusable input, OSError for
                   I/O problems.
    python coverage.py <splat.ply> <images.bin> <cameras.bin> <out.json> [--transform FILE]
                       [--up x,y,z]
        the same, written atomically to out.json; prints one JSON line with the timings.

The method
    1. What is judged. A coarse pass bins the Gaussian centres (opacity at least MIN_ALPHA)
       inside the box of their 1st..99th percentile per axis into cubic voxels, GRID_CELLS
       along the longest side. From it, what each photo aimed at: the nearest surface in the
       middle 40 % of the picture (median over an 8x8 grid of cells), that far along the
       camera's axis (_aim_points). The median of those depths is the subject distance
       ("typical_distance"). Only the box around the aim points (2nd..98th percentile along
       east, north and up), widened by AIM_MARGIN subject distances, is judged; the grid is
       laid again over that box at GRID_CELLS along its longest side. So a truck walked
       around in a big square gets voxels of a few centimetres and the square around it is
       left out (未判断, grey in the viewer), while a room or a building walked around keeps
       nearly all of itself. A voxel keeps its total opacity ("mass") and the opacity-
       weighted mean of its centres. Voxels with less mass than MIN_MASS are dropped (stray
       floaters, not surfaces); voxels with at least OCCLUDER_MASS, or holding a Gaussian at
       least OPAQUE_ALPHA opaque, can hide what is behind them.
    2. What each photo saw. Every camera of the training dataset (all of images.bin, with the
       intrinsics of cameras.bin; lens distortion ignored, which moves points by a few pixels
       at most at this resolution) projects every voxel. A voxel is a candidate when it is in
       front of the camera, inside the picture, and closer than FAR_FACTOR times the scene
       depth (the median depth of everything the cameras see). Occlusion: the
       candidates that can hide things are drawn into a small depth image (a z-buffer: the
       picture shrunk so that one voxel at the camera's median depth covers about one pixel;
       each voxel fills the square of pixels it covers, at least one), drawn far to near so
       each pixel keeps the nearest surface. A candidate counts as seen only when it lies
       less than max(pixel, voxel) / tan(MIN_GRAZING_DEG) + one voxel behind the surface its
       pixel holds, so that a surface seen at a flat angle does not hide itself (one seen
       flatter than MIN_GRAZING_DEG may). This is a shadow-map test:
       one pass per camera over a few tens of thousands of voxels instead of a ray through the
       volume for every (voxel, camera) pair, which is what keeps it inside the time budget
       in plain Python. The back of a wall is "in view" of every camera that faces the front;
       the depth test removes those views.
    3. What is kept per voxel: the number of photos that saw it, the set of directions they saw
       it from (24 bins: 8 compass sectors x 3 elevation bands, in a frame with `up` as up),
       the mean viewing direction, and the finest pixel footprint (depth / focal length).
    4. Levels. strong (拍得充分): at least STRONG_VIEWS photos from at least STRONG_BINS
       direction bins; weak (不足): fewer than WEAK_VIEWS photos, or all of them from one
       bin; medium (一般): the rest.
    5. Weak regions: weak voxels joined by touching (26 neighbours), at least
       MIN_REGION_VOXELS each, the MAX_REGIONS with the most solid surface first (opacity
       mass, each voxel counted up to OCCLUDER_MASS, so faint haze ranks below real
       surfaces). For each, a place to re-shoot from (see _suggest) and a plain sentence; a
       region no photo saw only because something nearer was in the way is described as
       such ("在 N 张照片的画面里，但都被前面的东西挡住了"), not as "not photographed".
    6. Summary: the share of each level by voxels ("share") and by amount of surface
       (opacity mass, "surface_share"); the page shows the latter, which weighs a dense,
       well-reconstructed object above the same volume of sparse, smeared clutter.

What it does not see (measured on three real scans, see the evidence below)
    - Holes. Where the photos saw nothing there are usually no Gaussians at all, so there is
      nothing to colour; the viewer shows the background there.
    - Sky and other things without parallax: the trainer puts their Gaussians at some
      arbitrary depth, every camera "sees" them, so they count as well covered, yet from a
      new viewpoint they float as haze in front of the scene.
    - Blur from standing closer than any photo: coverage belongs to a place, closeness to a
      viewpoint.
    - Lens distortion is ignored, and fisheye cameras are treated as pinhole ones.
    - See-through parts (slatted rails, foliage) hide what is behind them as if solid: on the
      truck scan 16 % of the photo-to-point matches were called hidden.

Evidence (2026-11-17; runs/_covlab/ holds the scripts and pictures)
    sb-v2 (a building walked around, 128 photos), playroom-std (a room, 225 photos), both
    1,000,000 Gaussians, and accept-std (a truck orbited closely in a big square, 100 video
    frames, 2,000,000 Gaussians).
    - Occlusion model against the SfM tracks (which photos really matched each sparse point):
      of the (point, photo) pairs in the tracks, the model calls 96.0 % (building), 97.7 %
      (room) and 84 % (truck) seen; without the depth test 99.6 % / 99.7 %, but then a
      sparse point of the building is "seen" by 104 photos on average instead of 28, and
      almost nothing is weak.
    - Gaussians in weak voxels are larger (blurrier) than in strong voxels at the same
      distance from the cameras, in all four distance quartiles of every scan: 1.4-3.6 times
      on the building, 2.2-2.6 on the room, 1.2-2.0 on the truck.
    - 100x100-pixel tiles of renders away from the capture path (camera lifted, orbited
      45-180 degrees, moved closer), shuffled and rated broken or not without seeing their
      level (one rater): building 20 of 30 weak tiles broken against 3 of 30 strong ones;
      truck 10 of 11 weak against 10 of 30 strong. In the room 1 % of the surface is weak,
      too few for a sample; medium and strong tiles looked alike there.
    - Whole verdict (share of surface): truck 78 % strong, 15 % medium, 7 % weak (before the
      judged part followed the aim points: 98 % weak, the square around it counted);
      building 90 / 6 / 4 %; room 95 / 4 / 0.3 %.

Output (JSON, small: a few hundred KB)
    {"format": "scan3d-coverage", "version": FORMAT_VERSION, "method": "<one line>",
     "grid": {"origin": [x, y, z], "voxel": size, "dims": [nx, ny, nz]},
         voxel (i, j, k) covers origin + voxel * [i, i+1) ...; its key is i + nx * (j + ny * k)
     "voxels": {"count": n, "keys": base64 of uint32 LE (sorted), "level": base64 of uint8
                (0 weak, 1 medium, 2 strong), "views": base64 of uint8 (photos, at most 255),
                "bins": base64 of uint8 (direction bins seen), "region": base64 of uint8
                (0, or the number of the listed weak region the voxel belongs to)},
     "levels": {"weak": n, "medium": n, "strong": n}   judged voxels per level (shares in
         "share"), "surface_share": {"weak", "medium", "strong"} by opacity mass,
     "not_judged": voxels inside the grid but outside the judged box (not in "voxels"),
     "cameras": [{"position", "forward", "name"}]   every training camera, splat coordinates,
     "up": [x, y, z], "typical_distance": subject distance, "scene_depth": d,
     "regions": [{"number" (1..), "center" (a voxel of the region near its middle), "size"
                  [dx, dy, dz], "voxels", "share", "views" (median photos), "bins" (median
                  direction bins), "bins_any", "height" ("low" | "middle" | "high"),
                  "nearest_photo" (name of the photo taken nearest to it while facing it),
                  "attention" (photos that had it in the picture, ignoring what is in
                  front), "solid", "reason" ("unseen" | "few" | "one_side"), "text" (one
                  Chinese sentence), "suggest": {"position", "target", "elevated": bool} or
                  null}],
     "thresholds": {...}, "seconds": {...}, "gaussians": n, "gaussians_used": n,
     "candidates": n}
"""
from __future__ import annotations

import base64
import json
import math
import os
import struct
import sys
import time
from array import array
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

FORMAT_VERSION = 2
MIN_ALPHA = 0.05            # Gaussians fainter than this are ignored
MIN_MASS = 0.25             # a voxel with less total opacity is not a surface
OCCLUDER_MASS = 1.0         # ... with at least this much it hides what is behind it,
OPAQUE_ALPHA = 0.7          # ... as does one holding a Gaussian at least this opaque (small voxels)
GRID_CELLS = 110            # voxels along the longest side of the scene box
FAR_FACTOR = 3.0            # views from further than this x the scene depth do not count
AIM_MARGIN = 0.5            # judged: the box around the aim points, widened by this x the subject distance
NEAR_FACTOR = 0.02          # ... nor from nearer than this (inside the voxel)
MIN_GRAZING_DEG = 10.0      # surfaces seen at a flatter angle may hide themselves (see below)
ZBUF_MAX_SIDE = 320         # pixels on the long side of a depth image
ZBUF_MIN_SIDE = 24
ZBUF_FILL = 0.5            # half the side of the square a voxel fills in the depth image, in voxels
STRONG_VIEWS = 8
STRONG_BINS = 3
WEAK_VIEWS = 3
MAX_REGIONS = 5
MIN_REGION_VOXELS = 30
MAX_CAMERAS = 5000          # more are thinned out evenly
MAX_GAUSSIANS = 20_000_000

_PLY_TYPES = {"char": 1, "int8": 1, "uchar": 1, "uint8": 1, "short": 2, "int16": 2,
              "ushort": 2, "uint16": 2, "int": 4, "int32": 4, "uint": 4, "uint32": 4,
              "float": 4, "float32": 4, "double": 8, "float64": 8}
_BLOCK = 1 << 16
# COLMAP camera model id -> number of parameters (SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL,
# RADIAL, OPENCV, OPENCV_FISHEYE, FULL_OPENCV, FOV, SIMPLE_RADIAL_FISHEYE, RADIAL_FISHEYE,
# THIN_PRISM_FISHEYE); the models with one focal length store f, cx, cy first.
_CAMERA_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}
_ONE_FOCAL = (0, 2, 3, 8, 9)


METHOD = ("per voxel of the splat's surfaces around what the photos aimed at: photos that saw "
          "it (in the picture, in front, within %g x the scene depth, not hidden: per-camera z-buffer of the voxels, "
          "surfaces flatter than %g degrees may hide themselves) and from how many of 24 "
          "direction bins; strong >= %d photos and >= %d bins, weak < %d photos or one bin"
          % (FAR_FACTOR, MIN_GRAZING_DEG, STRONG_VIEWS, STRONG_BINS, WEAK_VIEWS))


class CoverageError(ValueError):
    """The input cannot be used to work out the coverage."""


# ---------------------------------------------------------------------------- reading

def read_splat(path: os.PathLike) -> Tuple[int, array, array, array, array]:
    """(count, x, y, z, alpha) of a 3DGS PLY: centres and sigmoid(opacity), array('f')."""
    with open(path, "rb") as f:
        if f.readline(256).strip() != b"ply":
            raise CoverageError("not a PLY file")
        count, stride, offsets, in_vertex, binary = None, 0, {}, False, False
        for _ in range(4096):
            raw = f.readline(1024)
            if not raw:
                raise CoverageError("the PLY header is not terminated")
            words = raw.decode("ascii", "replace").split()
            if not words:
                continue
            if words[0] == "end_header":
                break
            if words[0] == "format":
                binary = len(words) > 1 and words[1] == "binary_little_endian"
            elif words[0] == "element" and len(words) == 3:
                in_vertex = count is None and words[1] == "vertex"
                if in_vertex:
                    count = int(words[2]) if words[2].isdigit() else None
            elif words[0] == "property" and in_vertex and len(words) == 3:
                if words[1] not in _PLY_TYPES:
                    raise CoverageError("unsupported PLY property type")
                if words[1] in ("float", "float32"):
                    offsets[words[2]] = stride
                stride += _PLY_TYPES[words[1]]
        else:
            raise CoverageError("the PLY header is too long")
        if not binary or count is None or not all(k in offsets for k in ("x", "y", "z", "opacity")):
            raise CoverageError("not a binary 3D Gaussian splat PLY")
        if count > MAX_GAUSSIANS:
            raise CoverageError("too many Gaussians: %d" % count)
        cols = {k: array("f") for k in ("x", "y", "z", "opacity")}
        done = 0
        while done < count:
            n = min(_BLOCK, count - done)
            block = f.read(n * stride)
            if len(block) != n * stride:
                raise CoverageError("the PLY file is shorter than its header says")
            for name, column in cols.items():
                raw = bytearray(4 * n)
                at = offsets[name]
                for byte in range(4):
                    raw[byte::4] = block[at + byte::stride]
                part = array("f")
                part.frombytes(raw)
                if sys.byteorder != "little":
                    part.byteswap()
                column.extend(part)
            done += n
    exp = math.exp
    alpha = array("f", [1.0 / (1.0 + exp(-o)) if -60.0 < o < 60.0 else (1.0 if o > 0 else 0.0)
                        for o in cols["opacity"]])
    return count, cols["x"], cols["y"], cols["z"], alpha


def read_intrinsics(cameras_bin: os.PathLike) -> Dict[int, Tuple[int, int, float, float, float, float]]:
    """camera id -> (width, height, fx, fy, cx, cy) from a COLMAP cameras.bin."""
    out = {}
    with open(cameras_bin, "rb") as f:
        (count,) = struct.unpack("<Q", f.read(8))
        if count > 100000:
            raise CoverageError("implausible camera count")
        for _ in range(count):
            cid, model, width, height = struct.unpack("<IiQQ", f.read(24))
            if model not in _CAMERA_PARAMS:
                raise CoverageError("unknown camera model %d" % model)
            p = struct.unpack("<%dd" % _CAMERA_PARAMS[model], f.read(8 * _CAMERA_PARAMS[model]))
            if model in _ONE_FOCAL:
                fx = fy = p[0]
                cx, cy = p[1], p[2]
            else:
                fx, fy, cx, cy = p[0], p[1], p[2], p[3]
            if not (0 < width < 1e6 and 0 < height < 1e6 and 0 < fx < 1e9 and 0 < fy < 1e9
                    and math.isfinite(cx) and math.isfinite(cy)):
                raise CoverageError("unusable camera intrinsics")
            out[cid] = (int(width), int(height), fx, fy, cx, cy)
    return out


def read_poses(images_bin: os.PathLike) -> List[Tuple[str, int, Tuple[float, ...], Tuple[float, float, float]]]:
    """[(name, camera id, world-to-camera rotation row-major (9), translation (3))]."""
    out = []
    with open(images_bin, "rb") as f:
        (count,) = struct.unpack("<Q", f.read(8))
        if count > 1_000_000:
            raise CoverageError("implausible image count")
        for _ in range(count):
            _iid, qw, qx, qy, qz, tx, ty, tz, cid = struct.unpack("<I7dI", f.read(64))
            name = bytearray()
            for _ in range(4096):
                byte = f.read(1)
                if byte in (b"\x00", b""):
                    break
                name += byte
            (points,) = struct.unpack("<Q", f.read(8))
            f.seek(points * 24, 1)
            norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
            if not norm > 1e-12:
                continue
            qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
            rot = (1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw),
                   2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw),
                   2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy))
            if not all(math.isfinite(v) for v in rot + (tx, ty, tz)):
                continue
            out.append((name.decode("utf-8", "replace"), cid, rot, (tx, ty, tz)))
    return out


def read_transform(path: os.PathLike) -> Optional[Tuple[float, List[List[float]], List[float]]]:
    """train_from_world of a scene_transform.json: (scale, rotation 3x3, translation)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    try:
        part = data["train_from_world"]
        scale = float(part.get("scale", 1))
        matrix = [[float(v) for v in row] for row in part["rotation"]["matrix_3x3"]]
        translation = [float(v) for v in part.get("translation", [0, 0, 0])]
    except (KeyError, TypeError, ValueError, AttributeError):
        raise CoverageError("unusable scene transform") from None
    if len(matrix) != 3 or any(len(r) != 3 for r in matrix) or len(translation) != 3 or not scale > 0:
        raise CoverageError("unusable scene transform")
    return scale, matrix, translation


class Camera:
    __slots__ = ("name", "rot", "t", "center", "fx", "fy", "cx", "cy", "width", "height")

    def __init__(self, name, rot, t, intr):
        self.name = name
        self.rot, self.t = rot, t
        r = rot
        self.center = tuple(-(r[0 + i] * t[0] + r[3 + i] * t[1] + r[6 + i] * t[2]) for i in range(3))
        self.width, self.height, self.fx, self.fy, self.cx, self.cy = intr

    @property
    def forward(self):
        return (self.rot[6], self.rot[7], self.rot[8])


def load_cameras(images_bin, cameras_bin, transform=None) -> List[Camera]:
    intrinsics = read_intrinsics(cameras_bin)
    cams = []
    for name, cid, rot, t in read_poses(images_bin):
        if cid not in intrinsics:
            continue
        if transform is not None:
            # p_train = s M p_world + T, so p_cam * s = (R M^T) p_train + (s t - R M^T T)
            s, m, tt = transform
            r = [rot[0:3], rot[3:6], rot[6:9]]
            rm = [[sum(r[i][k] * m[j][k] for k in range(3)) for j in range(3)] for i in range(3)]
            nt = [s * t[i] - sum(rm[i][j] * tt[j] for j in range(3)) for i in range(3)]
            rot = tuple(v for row in rm for v in row)
            t = tuple(nt)
        cams.append(Camera(name, rot, t, intrinsics[cid]))
    if len(cams) > MAX_CAMERAS:
        step = len(cams) / MAX_CAMERAS
        cams = [cams[int(i * step)] for i in range(MAX_CAMERAS)]
    return cams


# ---------------------------------------------------------------------------- geometry helpers

def _percentile(values: Sequence[float], q: float) -> float:
    v = sorted(values)
    return v[min(len(v) - 1, max(0, int(q * (len(v) - 1) + 0.5)))]


def _normalize(v):
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return (v[0] / n, v[1] / n, v[2] / n) if n > 1e-12 else (0.0, 0.0, 1.0)


def _frame(up):
    """Two unit vectors (east, north) that with `up` make a right-handed frame."""
    up = _normalize(up)
    a = (1.0, 0.0, 0.0) if abs(up[0]) < 0.9 else (0.0, 1.0, 0.0)
    e = _normalize((a[1] * up[2] - a[2] * up[1], a[2] * up[0] - a[0] * up[2], a[0] * up[1] - a[1] * up[0]))
    n = (up[1] * e[2] - up[2] * e[1], up[2] * e[0] - up[0] * e[2], up[0] * e[1] - up[1] * e[0])
    return e, n, up


def _b64(a: array) -> str:
    if sys.byteorder != "little" and a.itemsize > 1:
        a = array(a.typecode, a)
        a.byteswap()
    return base64.b64encode(a.tobytes()).decode("ascii")


# ---------------------------------------------------------------------------- the measure

def compute(splat_ply, images_bin, cameras_bin, *, transform=None, up=None, debug=False,
            grid_cells: int = GRID_CELLS, occlusion: bool = True) -> dict:
    timings = {}
    t0 = time.perf_counter()
    cams = load_cameras(images_bin, cameras_bin, transform)
    if len(cams) < 1:
        raise CoverageError("no usable training cameras")
    count, xs, ys, zs, alpha = read_splat(splat_ply)
    timings["read"] = time.perf_counter() - t0

    # ---- 1. voxels: a coarse pass over the whole splat finds what the cameras look at; the
    # grid is then laid over the part near the cameras only, so a small subject in a big
    # street gets voxels of its own size and the far background is left out (未判断)
    t = time.perf_counter()
    if up is None:      # cameras' mean "up" (COLMAP: -y of the camera)
        s = [0.0, 0.0, 0.0]
        for c in cams:
            for i in range(3):
                s[i] -= c.rot[3 + i]
        up = s
    up = _normalize(up)
    step = max(1, count // 200000)
    finite = math.isfinite
    sample = [i for i in range(0, count, step)
              if alpha[i] >= 0.2 and finite(xs[i]) and finite(ys[i]) and finite(zs[i])]
    if len(sample) < 10:
        raise CoverageError("the splat is (almost) empty")
    lo, hi = [], []
    for col in (xs, ys, zs):
        vals = [col[i] for i in sample]
        a, b = _percentile(vals, 0.01), _percentile(vals, 0.99)
        pad = 0.05 * (b - a) + 1e-6
        lo.append(a - pad)
        hi.append(b + pad)
    coarse = _bin(xs, ys, zs, alpha, lo, hi, grid_cells)
    if not coarse["keys"]:
        raise CoverageError("no surfaces found in the splat")
    east, north, upv = _frame(up)
    aims = _aim_points(cams, coarse)
    depth = _median_depth(cams, coarse)                 # how far everything in the photos was
    if aims:
        typical = _percentile([d for _p, d in aims], 0.5)   # how far the subject was
        # the judged part: the box (in the frame of `up`) around the points the photos
        # aimed at, widened by AIM_MARGIN subject distances
        margin = AIM_MARGIN * typical
        frame = (east, north, upv)
        span = []
        for axis in frame:
            vals = [sum(p[a] * axis[a] for a in range(3)) for p, _d in aims]
            span.append((_percentile(vals, 0.02) - margin, _percentile(vals, 0.98) + margin))
        corners = [[sum(frame[k][a] * (span[k][0] if (m >> k) & 1 else span[k][1]) for k in range(3))
                    for a in range(3)] for m in range(8)]
        lo = [max(lo[a], min(c[a] for c in corners)) for a in range(3)]
        hi = [min(hi[a], max(c[a] for c in corners)) for a in range(3)]
        if not all(hi[a] > lo[a] for a in range(3)):
            raise CoverageError("the cameras aim outside the splat")
    else:
        typical = depth or max(hi[a] - lo[a] for a in range(3))
        span = None
    depth = depth or typical
    del coarse
    grid = _bin(xs, ys, zs, alpha, lo, hi, grid_cells)
    del xs, ys, zs, alpha
    keys = grid["keys"]
    if not keys:
        raise CoverageError("no surfaces found where the photos looked")
    VX, VY, VZ, vmass, opaque = grid["VX"], grid["VY"], grid["VZ"], grid["mass"], grid["opaque"]
    vs, dims, used = grid["voxel"], grid["dims"], grid["used"]
    nx, ny, nz = dims
    ox, oy, oz = grid["origin"]
    inv = 1.0 / vs
    del grid
    n = len(keys)
    occluder = bytearray(1 if o else 0 for o in opaque)
    index_of = {k: i for i, k in enumerate(keys)}
    timings["voxels"] = time.perf_counter() - t
    t = time.perf_counter()
    far, near = FAR_FACTOR * depth, max(NEAR_FACTOR * typical, 0.5 * vs)
    if span is None:
        judged = bytearray(b"\x01" * n)
    else:
        judged = bytearray(1 if all(span[k][0] <= x * f[0] + y * f[1] + z * f[2] <= span[k][1]
                                    for k, f in enumerate((east, north, upv))) else 0
                           for x, y, z in zip(VX, VY, VZ))

    # ---- 2. per camera: candidates, z-buffer, visibility
    views = [0] * n
    bins = [0] * n
    dsx = [0.0] * n
    dsy = [0.0] * n
    dsz = [0.0] * n
    gsd = [1e30] * n
    seen_by = [0] * n if debug else None
    candidates_total = 0
    # How far behind the surface in its pixel a voxel may lie and still count as seen. A
    # pixel of the depth image spans w = depth * shrink / fx, a voxel fills at least one, and
    # a surface seen at a grazing angle g spans s / tan(g) of depth across a width s; so a
    # tolerance of max(w, voxel) / tan(MIN_GRAZING_DEG) plus one voxel keeps a surface from
    # hiding itself unless it is seen flatter than that.
    grazing = 1.0 / math.tan(math.radians(MIN_GRAZING_DEG))
    atan2, sqrt, pi = math.atan2, math.sqrt, math.pi
    sector_scale = 8 / (2 * pi)
    for ci, c in enumerate(cams):
        r0, r1, r2, r3, r4, r5, r6, r7, r8 = c.rot
        t0_, t1_, t2_ = c.t
        fx, fy, cx, cy, W, H = c.fx, c.fy, c.cx, c.cy, c.width, c.height
        zc_all = [r6 * x + r7 * y + r8 * z + t2_ for x, y, z in zip(VX, VY, VZ)]
        cand = []
        for i, zc in enumerate(zc_all):
            if near < zc < far:
                x, y, z = VX[i], VY[i], VZ[i]
                u = (r0 * x + r1 * y + r2 * z + t0_) / zc * fx + cx
                if 0 <= u < W:
                    v = (r3 * x + r4 * y + r5 * z + t1_) / zc * fy + cy
                    if 0 <= v < H:
                        cand.append((zc, u, v, i))
        if not cand:
            continue
        candidates_total += len(cand)
        if occlusion:
            # depth image: one voxel at the median candidate depth covers about one pixel
            zmed = sorted(cand)[len(cand) // 2][0]
            shrink = max(fx * vs / zmed, max(W, H) / ZBUF_MAX_SIDE, 1e-9)
            shrink = min(shrink, max(W, H) / ZBUF_MIN_SIDE)
            ws, hs = int(W / shrink) + 1, int(H / shrink) + 1
            inf = float("inf")
            zb = [inf] * (ws * hs)
            # A voxel fills the pixels its square (side: its size, a little more so that
            # neighbours leave no cracks) covers at its depth.
            reach = ZBUF_FILL * fx * vs / shrink      # half the side in pixels, times depth
            occl = [e for e in cand if occluder[e[3]]]
            occl.sort(reverse=True)             # far to near: nearer ones overwrite
            for zc, u, v, i in occl:
                rad = reach / zc
                pu, pv = u / shrink, v / shrink
                front = zc - 0.5 * vs
                a, b = max(0, int(pu - rad)), min(ws - 1, int(pu + rad))
                if rad < 0.5 and a == b and int(pv - rad) == int(pv + rad):
                    zb[int(pv) * ws + a] = front
                    continue
                fill = [front] * (b - a + 1)
                for row in range(max(0, int(pv - rad)), min(hs - 1, int(pv + rad)) + 1):
                    zb[row * ws + a: row * ws + b + 1] = fill
        ccx, ccy, ccz = c.center
        for zc, u, v, i in cand:
            if occlusion and zc - zb[int(v / shrink) * ws + int(u / shrink)] > grazing * max(zc * shrink / fx, vs) + vs:
                continue
            views[i] += 1
            dx, dy, dz = ccx - VX[i], ccy - VY[i], ccz - VZ[i]
            d = sqrt(dx * dx + dy * dy + dz * dz)
            dx, dy, dz = dx / d, dy / d, dz / d
            dsx[i] += dx
            dsy[i] += dy
            dsz[i] += dz
            e = dx * east[0] + dy * east[1] + dz * east[2]
            nn = dx * north[0] + dy * north[1] + dz * north[2]
            h = dx * upv[0] + dy * upv[1] + dz * upv[2]
            sector = int((atan2(nn, e) + pi) * sector_scale) % 8
            band = 0 if h < -0.34 else (1 if h < 0.5 else 2)
            bins[i] |= 1 << (band * 8 + sector)
            g = zc / fx
            if g < gsd[i]:
                gsd[i] = g
            if seen_by is not None:
                seen_by[i] |= 1 << ci
    timings["visibility"] = time.perf_counter() - t

    # ---- 4. levels
    t = time.perf_counter()
    nbins = [bin(b).count("1") for b in bins]
    level = bytearray(n)
    for i in range(n):
        if views[i] >= STRONG_VIEWS and nbins[i] >= STRONG_BINS:
            level[i] = 2
        elif views[i] < WEAK_VIEWS or nbins[i] <= 1:
            level[i] = 0
        else:
            level[i] = 1
    counts = {name: sum(1 for i in range(n) if judged[i] and level[i] == lv)
              for lv, name in enumerate(("weak", "medium", "strong"))}
    # The same by amount of surface (opacity mass): a dense, well-reconstructed object weighs
    # more than the same volume of sparse, smeared clutter. This is what the page reports.
    weights = [0.0, 0.0, 0.0]
    for i in range(n):
        if judged[i]:
            weights[level[i]] += vmass[i]
    total_mass = sum(weights) or 1.0
    surface = {name: round(weights[lv] / total_mass, 4)
               for lv, name in enumerate(("weak", "medium", "strong"))}

    # ---- 5. weak regions and where to re-shoot them from
    heights = [x * upv[0] + y * upv[1] + z * upv[2] for x, y, z in zip(VX, VY, VZ)]
    h_lo, h_hi = _percentile(heights, 0.02), _percentile(heights, 0.98)
    cam_h = [c.center[0] * upv[0] + c.center[1] * upv[1] + c.center[2] * upv[2] for c in cams]
    occupied = {keys[i] for i in range(n) if occluder[i]}
    neighbours = [dx + nx * (dy + ny * dz) for dz in (-1, 0, 1) for dy in (-1, 0, 1)
                  for dx in (-1, 0, 1) if dx or dy or dz]
    done = bytearray(n)
    components = []
    for start in range(n):
        if level[start] != 0 or done[start] or not judged[start]:
            continue
        done[start] = 1
        stack, members = [start], []
        while stack:
            i = stack.pop()
            members.append(i)
            k = keys[i]
            ki, kj, kk = k % nx, (k // nx) % ny, k // (nx * ny)
            for off in neighbours:
                j = index_of.get(k + off)
                if j is None or done[j] or level[j] != 0 or not judged[j]:
                    continue
                k2 = keys[j]
                if abs(k2 % nx - ki) > 1 or abs((k2 // nx) % ny - kj) > 1 or abs(k2 // (nx * ny) - kk) > 1:
                    continue
                done[j] = 1
                stack.append(j)
        if len(members) >= MIN_REGION_VOXELS:
            components.append(members)

    def blocked(p, q) -> bool:
        """Does the straight line from p to q pass through an occluding voxel (ignoring the
        voxel at p and the last three at q, the surface being looked at)?"""
        dx, dy, dz = q[0] - p[0], q[1] - p[1], q[2] - p[2]
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        steps = int(length / (0.5 * vs))
        for s in range(2, steps - 6):
            f = s / steps
            x, y, z = p[0] + dx * f, p[1] + dy * f, p[2] + dz * f
            i, j, k = int((x - ox) * inv), int((y - oy) * inv), int((z - oz) * inv)
            if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz and (i + nx * (j + ny * k)) in occupied:
                return True
        return False

    total_vox = float(max(1, sum(judged)))
    cam_hmed = _percentile(cam_h, 0.5)
    region_of = bytearray(n)

    def solid(members):
        return sum(min(vmass[i], OCCLUDER_MASS) for i in members) / OCCLUDER_MASS

    def attention(p):
        """How many cameras had p in the picture (ignoring what is in front of it)."""
        hits = 0
        for c in cams:
            r = c.rot
            zc = r[6] * p[0] + r[7] * p[1] + r[8] * p[2] + c.t[2]
            if near < zc < far:
                u = (r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + c.t[0]) / zc * c.fx + c.cx
                v = (r[3] * p[0] + r[4] * p[1] + r[5] * p[2] + c.t[1]) / zc * c.fy + c.cy
                if 0 <= u < c.width and 0 <= v < c.height:
                    hits += 1
        return hits

    ranked = sorted(components, key=solid, reverse=True)
    regions = []
    for number, members in enumerate(ranked[:MAX_REGIONS]):
        m = len(members)
        for i in members:
            region_of[i] = number + 1
        cxr = sum(VX[i] for i in members) / m
        cyr = sum(VY[i] for i in members) / m
        czr = sum(VZ[i] for i in members) / m
        anchor = min(members, key=lambda i: (VX[i] - cxr) ** 2 + (VY[i] - cyr) ** 2 + (VZ[i] - czr) ** 2)
        cxr, cyr, czr = centre = (VX[anchor], VY[anchor], VZ[anchor])
        size = [max(VX[i] for i in members) - min(VX[i] for i in members) + vs,
                max(VY[i] for i in members) - min(VY[i] for i in members) + vs,
                max(VZ[i] for i in members) - min(VZ[i] for i in members) + vs]
        mviews = sorted(views[i] for i in members)[m // 2]
        allbins = 0
        for i in members:
            allbins |= bins[i]
        mbins = sorted(nbins[i] for i in members)[m // 2]
        dmean = _normalize((sum(dsx[i] for i in members), sum(dsy[i] for i in members),
                            sum(dsz[i] for i in members)))
        seen_any = sum(1 for i in members if views[i] > 0) / m
        reason = "unseen" if seen_any < 0.5 or mviews == 0 else (
            "few" if mviews < WEAK_VIEWS else "one_side")
        hc = cxr * upv[0] + cyr * upv[1] + czr * upv[2]
        rel = (hc - h_lo) / max(h_hi - h_lo, 1e-9)
        height = "low" if rel < 0.33 else ("high" if rel > 0.66 else "middle")
        # the photo taken nearest to it while facing it (else simply the nearest)
        def facing(c):
            d = _normalize((cxr - c.center[0], cyr - c.center[1], czr - c.center[2]))
            return d[0] * c.forward[0] + d[1] * c.forward[1] + d[2] * c.forward[2] > 0.5
        near_cams = [c for c in cams if facing(c)] or cams
        best = min(near_cams, key=lambda c: (c.center[0] - cxr) ** 2 + (c.center[1] - cyr) ** 2 + (c.center[2] - czr) ** 2)
        suggest = _suggest(members, centre, dmean, reason, VX, VY, VZ, blocked, occupied,
                           (ox, oy, oz), vs, dims, upv, east, north, cam_hmed, typical)
        in_pictures = attention(centre)
        text = _sentence(reason, mviews, height, best.name, suggest, in_pictures)
        regions.append({
            "number": number + 1,
            "center": [round(v, 5) for v in centre], "size": [round(v, 5) for v in size],
            "voxels": m, "share": round(m / total_vox, 4), "views": mviews, "bins": mbins,
            "bins_any": bin(allbins).count("1"), "height": height, "nearest_photo": best.name,
            "attention": in_pictures, "solid": round(solid(members), 1),
            "reason": reason, "text": text, "suggest": suggest})
    timings["regions"] = time.perf_counter() - t
    timings["total"] = time.perf_counter() - t0

    out = [i for i in range(n) if judged[i]]
    result = {
        "format": "scan3d-coverage", "version": FORMAT_VERSION, "method": METHOD,
        "grid": {"origin": [ox, oy, oz], "voxel": vs, "dims": dims},
        "voxels": {"count": len(out), "keys": _b64(array("I", [keys[i] for i in out])),
                   "level": _b64(array("B", [level[i] for i in out])),
                   "views": _b64(array("B", [min(255, views[i]) for i in out])),
                   "bins": _b64(array("B", [nbins[i] for i in out])),
                   "region": _b64(array("B", [region_of[i] for i in out]))},
        "not_judged": n - len(out),
        "levels": counts,
        "share": {k: round(v / total_vox, 4) for k, v in counts.items()},
        "surface_share": surface,
        "cameras": [{"position": [round(v, 5) for v in c.center],
                     "forward": [round(v, 5) for v in c.forward], "name": c.name} for c in cams],
        "up": [round(v, 6) for v in upv], "typical_distance": round(typical, 6),
        "scene_depth": round(depth, 6),
        "regions": regions,
        "thresholds": {"strong_views": STRONG_VIEWS, "strong_bins": STRONG_BINS,
                       "weak_views": WEAK_VIEWS, "far_factor": FAR_FACTOR,
                       "min_grazing_deg": MIN_GRAZING_DEG, "occlusion": occlusion},
        "gaussians": count, "gaussians_used": used, "candidates": candidates_total,
        "seconds": {k: round(v, 2) for k, v in timings.items()},
    }
    if debug:
        result["_debug"] = {"keys": keys, "views": views, "nbins": nbins, "level": list(level),
                            "judged": list(judged),
                            "seen_by": seen_by, "gsd": gsd, "mass": vmass,
                            "centroids": list(zip(VX, VY, VZ)), "cams": cams}
    return result


def _bin(xs, ys, zs, alpha, lo, hi, grid_cells) -> dict:
    """Gaussians inside the box lo..hi into cubic voxels, grid_cells along its longest side.
    -> {"voxel", "dims", "origin", "used", "keys" (sorted, of voxels with at least MIN_MASS),
    "VX", "VY", "VZ" (opacity-weighted centres), "mass"}"""
    extent = max(hi[i] - lo[i] for i in range(3))
    vs = extent / grid_cells
    dims = [max(1, int(math.ceil((hi[i] - lo[i]) / vs))) for i in range(3)]
    nx, ny, nz = dims
    ox, oy, oz = lo
    ex, ey, ez = ox + nx * vs, oy + ny * vs, oz + nz * vs
    inv = 1.0 / vs
    mass: Dict[int, float] = {}
    sx: Dict[int, float] = {}
    sy: Dict[int, float] = {}
    sz: Dict[int, float] = {}
    peak: Dict[int, float] = {}
    used = 0
    get = mass.get
    for x, y, z, a in zip(xs, ys, zs, alpha):
        # comparisons are false for NaN, so broken values drop out here
        if not (a >= MIN_ALPHA and ox <= x < ex and oy <= y < ey and oz <= z < ez):
            continue
        i = min(int((x - ox) * inv), nx - 1)
        j = min(int((y - oy) * inv), ny - 1)
        k = min(int((z - oz) * inv), nz - 1)
        key = i + nx * (j + ny * k)
        m = get(key)
        if m is None:
            mass[key] = a
            sx[key] = a * x
            sy[key] = a * y
            sz[key] = a * z
            peak[key] = a
        else:
            mass[key] = m + a
            sx[key] += a * x
            sy[key] += a * y
            sz[key] += a * z
            if a > peak[key]:
                peak[key] = a
        used += 1
    keys = sorted(k for k, m in mass.items() if m >= MIN_MASS)
    return {"voxel": vs, "dims": dims, "origin": [ox, oy, oz], "used": used, "keys": keys,
            "VX": [sx[k] / mass[k] for k in keys], "VY": [sy[k] / mass[k] for k in keys],
            "VZ": [sz[k] / mass[k] for k in keys], "mass": [mass[k] for k in keys],
            "opaque": [mass[k] >= OCCLUDER_MASS or peak[k] >= OPAQUE_ALPHA for k in keys]}


def _project_all(c, X, Y, Z, floor, top):
    """(index, depth, u, v) of the points in front of camera c (floor < depth < top) that
    fall inside its picture."""
    r0, r1, r2, r3, r4, r5, r6, r7, r8 = c.rot
    t0, t1, t2 = c.t
    out = []
    for i, (x, y, z) in enumerate(zip(X, Y, Z)):
        zc = r6 * x + r7 * y + r8 * z + t2
        if not floor < zc < top:
            continue
        u = (r0 * x + r1 * y + r2 * z + t0) / zc * c.fx + c.cx
        if not 0 <= u < c.width:
            continue
        v = (r3 * x + r4 * y + r5 * z + t1) / zc * c.fy + c.cy
        if 0 <= v < c.height:
            out.append((i, zc, u, v))
    return out


def _median_depth(cams, grid) -> Optional[float]:
    """The median depth of (a sample of) the voxels in the pictures of (some of) the
    cameras: how far, typically, everything the photos show is. Views from further than
    FAR_FACTOR times this do not count."""
    VX, VY, VZ = grid["VX"], grid["VY"], grid["VZ"]
    step = max(1, len(VX) // 2000)
    X, Y, Z = VX[::step], VY[::step], VZ[::step]
    depths = []
    for c in cams[:: max(1, len(cams) // 60)]:
        depths.extend(zc for _i, zc, _u, _v in _project_all(c, X, Y, Z, 0.0, float("inf")))
    return _percentile(depths, 0.5) if depths else None


def _aim_points(cams, grid) -> List[Tuple[Tuple[float, float, float], float]]:
    """What the photos aimed at: for each camera (up to 60 of them), the nearest solid
    surface in each cell of an 8x8 grid over the middle 40 % of the picture (each voxel
    covering the cells its square covers), and the median of those depths; the point that far along the camera's axis. -> [(point, depth)].
    Unlike the median depth of everything in the pictures, this stays at a truck that was
    walked around, not at the street behind it."""
    VX, VY, VZ, opaque = grid["VX"], grid["VY"], grid["VZ"], grid["opaque"]
    solid = [i for i, o in enumerate(opaque) if o] or list(range(len(opaque)))
    PX, PY, PZ = [VX[i] for i in solid], [VY[i] for i in solid], [VZ[i] for i in solid]
    out = []
    for c in cams[:: max(1, len(cams) // 60)]:
        u0, u1, v0, v1 = 0.3 * c.width, 0.7 * c.width, 0.3 * c.height, 0.7 * c.height
        cu, cv = 8.0 / (u1 - u0), 8.0 / (v1 - v0)
        cells: Dict[int, float] = {}
        half = 0.5 * grid["voxel"] * c.fx      # half a voxel in pixels, times depth
        for _i, zc, u, v in _project_all(c, PX, PY, PZ, 0.5 * grid["voxel"], float("inf")):
            # each voxel covers the cells its square covers (a coarse voxel can span several)
            ra, rb = (u - half / zc - u0) * cu, (u + half / zc - u0) * cu
            ca, cb = (v - half / zc - v0) * cv, (v + half / zc - v0) * cv
            if rb < 0 or ra >= 8 or cb < 0 or ca >= 8:
                continue
            for row in range(max(0, int(ca)), min(7, int(cb)) + 1):
                for col in range(max(0, int(ra)), min(7, int(rb)) + 1):
                    cell = col + 8 * row
                    if zc < cells.get(cell, float("inf")):
                        cells[cell] = zc
        if len(cells) >= 16:
            d = _percentile(list(cells.values()), 0.5)
            f = c.forward
            out.append((tuple(c.center[a] + d * f[a] for a in range(3)), d))
    return out


def _suggest(members, centre, dmean, reason, VX, VY, VZ, blocked, occupied, origin, vs, dims, upv,
             east, north, cam_height, typical):
    """A place to photograph a region from: {"position", "target", "elevated"} or None.

    Candidates stand on rings around the region at the height the photos were taken from
    (the median camera height), at 0.25, 0.5, 0.8 and 1.2 times the typical viewing distance,
    every 15 degrees. A candidate must lie inside the scene box, at least a voxel away from
    any surface (so not behind a wall it would otherwise look through), and see at least
    half of a sample of the region's voxels along unblocked straight lines. The best one sees
    most of the region, at about the usual distance, not lengthwise along it when it is long
    and thin, and: for a region seen from one side only, from a direction unlike the ones it
    was photographed from; for one seen by too few photos, from the side those photos were
    taken on (that is where there is room to stand). Only when no candidate at that height works are two higher
    rings tried (0.5 and 1.0 typical distances above), marked elevated."""
    ox, oy, oz = origin
    nx, ny, nz = dims
    inv = 1.0 / vs
    sample = members[:: max(1, len(members) // 12)][:12]
    h_centre = centre[0] * upv[0] + centre[1] * upv[1] + centre[2] * upv[2]
    along = _long_axis(members, VX, VY, VZ)
    for lift in (0.0, 0.5, 1.0):
        height = cam_height + lift * typical
        best = None
        for factor in (0.25, 0.5, 0.8, 1.2):
            radius = factor * typical
            for step in range(24):
                ang = step * math.pi / 12
                ca, sa = math.cos(ang), math.sin(ang)
                p = [centre[a] + radius * (ca * east[a] + sa * north[a]) + (height - h_centre) * upv[a]
                     for a in range(3)]
                i, j, k = int((p[0] - ox) * inv), int((p[1] - oy) * inv), int((p[2] - oz) * inv)
                if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz) or p[0] < ox or p[1] < oy or p[2] < oz:
                    continue                    # outside the part of the scene that is known
                if any((i + a) + nx * ((j + b) + ny * (k + c)) in occupied
                       for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)):
                    continue                    # inside something, or right against it
                seen = sum(0 if blocked(p, (VX[m], VY[m], VZ[m])) else 1 for m in sample) / len(sample)
                if seen < 0.5:
                    continue
                d = _normalize((p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]))
                same = d[0] * dmean[0] + d[1] * dmean[1] + d[2] * dmean[2]
                if reason == "one_side":
                    bonus = 0.6 * (1.0 - same)         # a direction it was not seen from
                elif reason == "few":
                    bonus = 0.4 * max(0.0, same)       # the open side it was seen from
                else:
                    bonus = 0.0
                if along is not None:                  # not lengthwise along a strip
                    bonus -= 0.5 * abs(d[0] * along[0] + d[1] * along[1] + d[2] * along[2])
                score = seen + bonus - 0.2 * abs(factor - 0.8)
                if best is None or score > best[0]:
                    best = (score, p)
        if best is not None:
            return {"position": [round(v, 5) for v in best[1]], "target": [round(v, 5) for v in centre],
                    "elevated": lift > 0}
    return None


def _long_axis(members, VX, VY, VZ):
    """The direction a region is longest in, when it is clearly longer than wide; else None."""
    m = len(members)
    c = [sum(V[i] for i in members) / m for V in (VX, VY, VZ)]
    cov = [[0.0] * 3 for _ in range(3)]
    for i in members:
        d = (VX[i] - c[0], VY[i] - c[1], VZ[i] - c[2])
        for a in range(3):
            for b in range(3):
                cov[a][b] += d[a] * d[b]
    v = (1.0, 0.7, 0.4)
    for _ in range(30):         # power iteration: the largest eigenvector
        v = _normalize([sum(cov[a][b] * v[b] for b in range(3)) for a in range(3)])
    big = sum(v[a] * sum(cov[a][b] * v[b] for b in range(3)) for a in range(3))
    rest = cov[0][0] + cov[1][1] + cov[2][2] - big
    return v if big > 3.0 * max(rest, 1e-12) else None


def _sentence(reason: str, views: int, height: str, photo: str, suggest, in_pictures: int = 0) -> str:
    where = {"low": "靠近地面的", "middle": "", "high": "高处的"}[height]
    if reason == "unseen" and in_pictures > 0:
        # in the picture, but always behind something nearer: say that, not "not photographed"
        text = "这片%s区域在 %d 张照片的画面里，但都被前面的东西挡住了，看到的样子是猜出来的。" % (where, in_pictures)
    elif reason == "unseen":
        text = "这片%s区域没有照片拍到，看到的样子是猜出来的。" % where
    elif reason == "few":
        text = "这片%s区域只有 %d 张照片拍到。" % (where, views)
    else:
        text = "这片%s区域的照片都是从差不多同一个方向拍的，换个角度看会变形。" % where
    text += "离它最近的照片是 %s。" % photo
    if suggest and suggest["elevated"]:
        text += "从地面拍不到它，需要站到更高的地方（比如楼上或用自拍杆）。"
    return text


# ---------------------------------------------------------------------------- command line

def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    transform, up = None, None
    try:
        if "--transform" in args:
            at = args.index("--transform")
            transform = read_transform(args[at + 1])
            del args[at:at + 2]
        if "--up" in args:
            at = args.index("--up")
            up = [float(v) for v in args[at + 1].split(",")]
            if len(up) != 3 or not all(math.isfinite(v) for v in up) or not any(up):
                raise CoverageError("--up needs x,y,z")
            del args[at:at + 2]
        if len(args) != 4:
            print(__doc__)
            return 2
        splat, images, cameras, out = args
        result = compute(splat, images, cameras, transform=transform, up=up)
        out = Path(out)
        tmp = out.with_name(out.name + ".part")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp, out)
    except (CoverageError, OSError, ValueError, struct.error) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({"ok": True, "voxels": result["voxels"]["count"], "levels": result["levels"],
                      "regions": len(result["regions"]), "seconds": result["seconds"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
