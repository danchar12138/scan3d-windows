"""Game-engine meshes from a finished Spirula training run, textured from the photos.

`build_meshes` writes two files next to each other:

    mesh.glb        textured visual mesh, at most `visual_faces` triangles, one square PNG texture
    collision.glb   untextured mesh for physics and shadow receiving, at most `collision_faces`

Only two things run: `colmap.exe` (the CUDA build) and this file, which uses the standard
library alone (it runs on the embeddable Python). The splat itself is not meshed; the run folder
only says which frame the files are written in. How the meshes are made, with the times of the
south-building scan (runs/sb-v2: 128 photos of 3072 x 2304, masks for passers-by) at the
default budgets on an RTX 5070 Ti Laptop:

1. Geometry by COLMAP's dense reconstruction, in the dataset frame:
   `image_undistorter` (pinhole photos of at most MVS_IMAGE_SIZE px, at most MVS_MAX_IMAGES of
   them), `patch_match_stereo` (depth and normal maps on the GPU, a photometric and then a
   geometric pass, MVS_SOURCE_IMAGES photos matched against each), `stereo_fusion` (points
   with the cameras that saw them) and `delaunay_mesher` (a surface carved out of the
   tetrahedralised points by the rays that saw them, by graph cut). 28 min on sb-v2 (24 of them dense matching), 1.24 M faces.
2. Here: faces wider than SHARD_MAX_PIXELS in every photo that sees them and far behind what
   the photos show are dropped (the graph cut's shards, see below), the mesh is moved into the
   splat's frame, components smaller than
   VISUAL_MIN_AREA_FRACTION of the largest by area are dropped, and `colmap mesh_simplifier`
   (quadric edge collapse with optimal vertex placement) brings it to the visual budget.
3. Texture from the photos: `colmap mesh_texturer` on undistorted photos at full resolution
   (made smaller only beyond PHOTO_PIXEL_BUDGET). It gives each face the photo that sees it
   largest, unoccluded by the mesh, smooths that choice over neighbours, levels colour across
   patch seams and packs the patches into an atlas. Its atlas size cannot be set, only the
   scale of the patches, so the scale is found by short probe runs that stop as soon as COLMAP
   prints the atlas size (3 probes on sb-v2); the atlas is then padded to a square. Faces no
   photo could texture take the colour of a textured neighbour; pieces no camera saw from the
   front are left out. Masked pixels (masks/<image path>.png, black = leave out) are kept off
   the texture by occluder triangles: before texturing, the masked area of each photo is
   covered, as seen from its camera, by small triangles between the camera and the scene,
   which the texturer's occlusion test then respects; they are removed afterwards.
4. The collision mesh is the visual geometry, welded, without its smaller pieces, brought to
   its budget by the same simplifier.
5. Both files are packed as binary glTF 2.0 and checked by `validate_glb`.

After the dense step the rest takes about a minute (texturing 45 s at 4096, 73 s at 8192).

Why it is done this way. Measured on sb-v2 by rendering the meshes from the dataset's cameras
next to the photos ("solid cover": share of the photo's non-sky pixels the mesh covers; "sky
cover": share of its sky pixels, all of them junk; colour error: mean absolute difference per
channel over 8 x 8 px blocks, on pixels every compared mesh covers; sharpness: mean gradient of
the render over that of the photo; flatness: RMS distance from a fitted plane of the surface
inside three window-free patches of brick wall, in thousandths of the camera spread):

* Geometry: COLMAP's dense reconstruction, not the splat. The surface Spirula extracts from the
  splat (`spirula mesh`, merge factor 1, 1.16 M faces, then decimated) covers 88.9 % of the
  solid pixels of 16 views and 80.5 % of the sky, with walls shaded as lumps, thin structures
  fused and slabs of junk hanging in the sky; the dense mesh covers 94.1 % of the solid pixels
  and 21 % of the sky (tree crowns and roof edges against it), with windows, cornices, columns,
  steps and the ground in their place, and 10 pieces instead of 29. On three patches of brick
  wall the dense surface is not flatter in the large (plane-fit RMS 10.1 / 4.3 / 3.8 against
  8.1 / 3.5 / 3.4 thousandths of the camera spread; the first patch straddles a step in the
  wall) and is rougher in the small: neighbouring faces meet at 30 / 19 / 22 degrees on
  average against 17 / 14 / 14, because each Delaunay facet is tilted a little by depth noise
  where Spirula relaxes its mesh. That shows under flat shading only; with the smooth vertex
  normals engines use, the dense walls shade evenly and the splat surface shows its lumps. A splat trained with Spirula's `meshing` preset
  (30,000 steps, 16 more minutes) meshed worse than the standard splat: walls broken into
  holes (frame coverage 72-88 % against 85-98 %), 68 components against 29. Poisson meshing of
  the fused points (23 M faces, 6 min, plus 6 min to simplify) gave blobs and an 82 % cover;
  `delaunay_mesher` (3 M faces from 2000 px depth maps, 7 min) the sharpest surface; it is used.
* Dense matching cost and settings: COLMAP's defaults at 2000 px took 56 min for the 128 photos
  (GPU partly shared), at 1200 px 30 min. MVS_IMAGE_SIZE 1600 with MVS_SOURCE_IMAGES 10 and 3
  iterations instead of 20 and 5 took 24 min and is not to be told apart from the 2000 px
  default in the renders (colour error 16.62 against 16.75, sharpness 0.910 against 0.908);
  1200 px with the defaults gives slightly rounder windows. Fusion and meshing then take
  3.5 min. Scratch space: about 9 GB of depth and normal maps for 128 photos.
* Texture: COLMAP's texturer on the photos, not Spirula's bake. Spirula bakes each surface
  point's colour from a render of the splat, which is visibly softer than the photos. Same
  300 k-face mesh, 4096 atlas, 16 views whose photos the texturer was NOT given (the splat had
  been trained on them): colour error 18.13 (photos) against 18.12 (bake), sharpness 0.95
  against 0.56; brick joints and window bars show in the photo texture and not in the bake.
* Atlas scale: COLMAP samples the photos bilinearly at the chosen scale. Shrinking the photos
  first (with a proper filter, by giving the texturer smaller cameras) gave a softer texture
  (sharpness 0.59 against 0.87) for the same atlas and was dropped.
* Colour levelling across patch seams (COLMAP's colour correction) lowers the colour error
  from 16.33 to 15.99; regularisation 0.1 and 1.0 are the same.
* Faces without a photo: COLMAP leaves them at texture coordinate (0, 0), which renders as black
  specks wherever the mesh hides a vertex from every camera (6 % of the faces on sb-v2, mostly
  in window reveals). Taking the colour of a textured neighbour removes the specks and brings
  the colour error from 17.50 to 16.99.
* Masks: COLMAP's texturer has no notion of masks; it chooses photos by geometry alone. A test
  photo with a magenta block painted over a wall and masked put 1,150 magenta texels into the
  atlas without the occluders and none with them.
* Shards: where there are no points the Delaunay mesher's graph cut closes the volume with
  triangles tens to hundreds of pixels long. In the sky and the far background (seen past the
  edge of an object) these are junk that the texturer paints with whatever lies behind; on plain
  walls and ceilings they are the real surface. Faces of the dense surface measure 1.9 px
  (median), 5.5 px (90 %) and 22.6 px (99 %) along their longest edge in the photo that sees them
  best (truck video, runs/accept-std, 100 frames of 978 x 546; edges measured as projected, so
  ground seen at a grazing angle is not penalised). Of the faces over 30 px, those on the truck
  lie at a median 2.9 times the median distance of what that photo shows (90 % beyond 1.35);
  those in the room (white walls) at 1.0, none beyond 2. Dropping faces over 30 px that are also
  beyond 2 times took the truck's sky cover from 0.586 to 0.283 (4 views; 1,315 of 281 k faces,
  every large slab in the sky and over the street) and removed nothing in the room and 23 of
  1.24 M faces on sb-v2. Without the distance condition the room lost half its walls (cover 0.999
  to 0.540). Shards near the camera stay. COLMAP's own limit (max_side_length_factor, relative to
  the 95th percentile side) removed only 579 of 930 k faces on sb-v2 when lowered from 25 to 10.
* Decimation: `colmap mesh_simplifier` and the earlier pure-Python collapse give renders that
  cannot be told apart (to 300 k and to 20 k faces); COLMAP's places vertices optimally (one
  wall patch 2.1 against 8.1), and takes 7 s instead of 25 s for 1 M to 300 k faces.

Coordinate frame
----------------
Both GLBs are written in the frame of the splat file (`step-*.ckpt/splat.ply`), which is
Spirula's training frame: no node transform, no axis swap, no unit change. A vertex (x, y, z)
in either GLB is the same point as a Gaussian centre (x, y, z) in splat.ply, so the splat and
the meshes overlay exactly in any engine whose glTF importer and splat loader agree on how to
treat raw coordinates. glTF nominally means "+Y up, metres"; a photo reconstruction has no
metric scale and no gravity direction, so that reading does not hold here and whatever rotation
and scale the engine applies to the splat must be applied to the two meshes as well.

COLMAP works in the dataset frame, where the cameras are. The training frame can differ from it
by `scene_transform.json` in the run folder: the dense mesh is moved into the training frame
with `train_from_world` as soon as it is read, and everything handed back to COLMAP for
texturing is moved into the dataset frame with `world_from_train`; only texture coordinates come
back from that, never positions.
"""
from __future__ import annotations

import array
import heapq
import itertools
import json
import math
import operator
import os
import re
import shutil
import struct
import sys
import time
import zlib
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

try:  # imported as part of the `app` package, or as a top-level module from inside app/
    from . import proc
except ImportError:  # pragma: no cover - depends on how the caller set up sys.path
    import proc

__all__ = ["build_meshes", "validate_glb", "MeshError"]

# --- tuning, settled by experiment (see the module docstring) --------------------------------

# Components smaller than this fraction of the largest component's surface area are dropped.
# Area, not face count: it does not depend on how finely a region happens to be triangulated.
VISUAL_MIN_AREA_FRACTION = 0.001
COLLISION_MIN_AREA_FRACTION = 0.01
# --- geometry (COLMAP dense reconstruction) ---
# Dense matching runs on at most MVS_MAX_IMAGES photos (spread evenly in name order), each made
# at most MVS_IMAGE_SIZE pixels on its long side, with MVS_SOURCE_IMAGES photos matched against
# each. Its cost grows with all three; these values are settled in the module docstring.
MVS_MAX_IMAGES = 200
MVS_IMAGE_SIZE = 1600
MVS_SOURCE_IMAGES = 10
PATCH_MATCH_OPTIONS = ("--PatchMatchStereo.geom_consistency", "1",
                       "--PatchMatchStereo.num_iterations", "3")
FUSION_OPTIONS = ()
DELAUNAY_OPTIONS = ()
# A face of the dense surface wider than this (longest edge, in pixels of the photos dense
# matching ran on) in every photo that sees it is a shard of the graph cut, not surface, when it
# is also far behind what the photos show (see `drop_shards` and the module docstring)...
SHARD_MAX_PIXELS = 30.0
# ...and further away, in the photo where it is smallest, than this many times the median
# distance of the surface that photo shows (estimated from this many vertices).
SHARD_MIN_DEPTH = 2.0
SHARD_DEPTH_SAMPLES = 20_000
# colmap mesh_simplifier usually lands on the budget in one pass; it is run again on what it
# returns, at most this often in all, when its collapse queue runs dry above the budget.
SIMPLIFY_PASSES = 3

# --- texturing (colmap mesh_texturer) ---
# Options passed to the texturer besides the scale; the rest are COLMAP's defaults (one photo
# per face, chosen by projected area and an occlusion test against the mesh, three rounds of
# neighbour smoothing, per-vertex colour levelling across patch seams, 2 px gutters).
TEXTURER_OPTIONS = ("--MeshTextureMapping.apply_color_correction", "1",
                    "--MeshTextureMapping.color_correction_regularization", "0.1")
# Atlas texels per photo pixel: never more than 1 (magnifying the photos adds nothing).
TEXTURE_MAX_SCALE = 1.0
MIN_TEXTURE_SCALE = 0.005
# COLMAP sizes its atlas itself: a power of two wide, as tall as its shelf packing needs, from
# the patches' size at the given scale. The scale is found by probing (a probe stops as soon
# as COLMAP prints the atlas size). The search aims at an atlas this full, stops once one is
# at least ATLAS_GOOD_FILL full or the bracket is this tight, and makes at most ATLAS_PROBES
# probes. COLMAP only picks a width when the patches fill at most 1/1.3 of the square.
ATLAS_FILL = 0.76
ATLAS_GOOD_FILL = 0.70
ATLAS_SCALE_TOLERANCE = 0.03
ATLAS_PROBES = 8
# The texturer holds every photo in memory at once: above this many pixels in all, the photos
# are made smaller when they are undistorted (a 4096 atlas uses a fifth of the photos'
# resolution on a typical scan, so this costs no texture detail).
PHOTO_PIXEL_BUDGET = 1_500_000_000
# Masks: the masked area of a photo is covered with occluder triangles on a grid of this many
# cells across the mask's short side, grown by one cell. They stand at half the distance to
# the nearest surface the camera sees (a sample of OCCLUDER_SAMPLE_VERTICES vertices), but no
# nearer than OCCLUDER_MIN_DEPTH and no further than OCCLUDER_MAX_DEPTH of the scene's size.
MASK_CELLS_ACROSS = 96
OCCLUDER_SAMPLE_VERTICES = 20_000
OCCLUDER_MIN_DEPTH = 0.001
OCCLUDER_MAX_DEPTH = 0.02
OCCLUDER_FOV_SLOPE = 10.0

# Dense matching: 2 x 200 photos at about 7 s each on an RTX 5070 Ti Laptop is under an hour.
MVS_TIMEOUT = 4 * 3600
COLMAP_TIMEOUT = 3600
# The longest silences: the texturer loading the photos and choosing views, and the Delaunay
# mesher's triangulation and graph cut (a minute or two each on large scans).
COLMAP_STALL = 1200

_F32 = array.array("f")
_I32 = array.array("i")
if _F32.itemsize != 4 or _I32.itemsize != 4 or sys.byteorder != "little":  # pragma: no cover
    raise ImportError("meshpost needs a little-endian platform with 32-bit C int and float")


class MeshError(RuntimeError):
    """A mesh step failed. The message says what to do about it."""


# =============================================================================================
# Mesh container
# =============================================================================================

class Mesh:
    """Indexed triangle mesh held in flat typed arrays (never one Python object per element)."""

    __slots__ = ("xyz", "tris", "uv")

    def __init__(self, xyz: array.array, tris: array.array, uv: Optional[array.array] = None):
        self.xyz = xyz            # array('f'), 3 per vertex
        self.tris = tris          # array('i'), 3 per face
        self.uv = uv              # array('f'), 2 per vertex (glTF convention), or None

    @property
    def vertex_count(self) -> int:
        return len(self.xyz) // 3

    @property
    def face_count(self) -> int:
        return len(self.tris) // 3


def _gather(data, start: int, stride: int, count: int, offset: int, width: int) -> bytes:
    """Pull `width` bytes at `offset` out of each of `count` records of `stride` bytes."""
    if count == 0:
        return b""
    end = start + stride * (count - 1) + offset + width
    if stride == width and offset == 0:
        return bytes(data[start:end])
    out = bytearray(count * width)
    for k in range(width):
        out[k::width] = data[start + offset + k:end:stride]
    return bytes(out)


def _from_bytes(typecode: str, raw) -> array.array:
    a = array.array(typecode)
    a.frombytes(raw)
    return a


# =============================================================================================
# Topology: components, filtering, cleanup
# =============================================================================================

def _vertex_roots(n_vertices: int, tris: array.array) -> list:
    """Union-find over the faces. Returns, per vertex, the smallest vertex index of its component."""
    parent = list(range(n_vertices))
    for a, b, c in zip(tris[0::3], tris[1::3], tris[2::3]):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        while parent[b] != b:
            parent[b] = parent[parent[b]]
            b = parent[b]
        if a != b:
            if a < b:
                parent[b] = a
            else:
                parent[a] = b
                a = b
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        if a != c:
            if a < c:
                parent[c] = a
            else:
                parent[a] = c
    # Links always point to a smaller index, so one ascending pass flattens every chain.
    for i in range(n_vertices):
        parent[i] = parent[parent[i]]
    return parent


def _weld_map(mesh: Mesh) -> Optional[list]:
    """Map each vertex to the first vertex at exactly the same position, or None if all differ.

    Meshes duplicate vertices along texture seams and where non-manifold fans were split;
    without welding, the two sides of such a split would be counted as separate components.
    """
    raw = mesh.xyz.tobytes()
    first = {}
    out = []
    append = out.append
    setdefault = first.setdefault
    for i in range(mesh.vertex_count):
        append(setdefault(raw[12 * i:12 * i + 12], i))
    return out if len(first) != len(out) else None


def face_components(mesh: Mesh):
    """Return (labels, faces, areas): a component label per face, and per label its face count
    and surface area."""
    tris = mesh.tris
    weld = _weld_map(mesh)
    if weld is not None:
        tris = array.array("i", map(weld.__getitem__, tris))
    roots = _vertex_roots(mesh.vertex_count, tris)
    labels = list(map(roots.__getitem__, tris[0::3]))
    faces = Counter(labels)
    areas = dict.fromkeys(faces, 0.0)
    x = mesh.xyz
    t = mesh.tris
    sqrt = math.sqrt
    j = 0
    for lab in labels:
        a = 3 * t[j]
        b = 3 * t[j + 1]
        c = 3 * t[j + 2]
        j += 3
        ax = x[a]
        ay = x[a + 1]
        az = x[a + 2]
        ux = x[b] - ax
        uy = x[b + 1] - ay
        uz = x[b + 2] - az
        vx = x[c] - ax
        vy = x[c + 1] - ay
        vz = x[c + 2] - az
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        areas[lab] += sqrt(nx * nx + ny * ny + nz * nz)
    for lab in areas:
        areas[lab] *= 0.5
    return labels, faces, areas


def select_faces(mesh: Mesh, keep) -> Mesh:
    """Sub-mesh of the faces whose entry in `keep` is true; unreferenced vertices are dropped."""
    n_vertices = mesh.vertex_count
    cols = [array.array("i", itertools.compress(mesh.tris[k::3], keep)) for k in range(3)]
    used = bytearray(n_vertices)
    one = itertools.repeat(1)
    for col in cols:
        # bytearray has no fancy indexing; a dict-free way to mark many indices at C speed.
        for _ in map(used.__setitem__, col, one):
            pass
    new_index = list(itertools.accumulate(used))       # 1-based index of each kept vertex
    n_kept = new_index[-1] if new_index else 0
    tris = array.array("i", bytes(12 * len(cols[0])))
    shift = (-1).__add__
    for k in range(3):
        tris[k::3] = array.array("i", map(shift, map(new_index.__getitem__, cols[k])))
    xyz = array.array("f", bytes(12 * n_kept))
    for k in range(3):
        xyz[k::3] = array.array("f", itertools.compress(mesh.xyz[k::3], used))
    uv = None
    if mesh.uv is not None:
        uv = array.array("f", bytes(8 * n_kept))
        for k in range(2):
            uv[k::2] = array.array("f", itertools.compress(mesh.uv[k::2], used))
    return Mesh(xyz, tris, uv)


def keep_large_components(mesh: Mesh, min_area_fraction: float, max_components: int = 0):
    """Drop every component whose area is below `min_area_fraction` of the largest one's.

    Returns (mesh, stats). Area, not face count, decides: face count depends on how finely the
    extractor happened to triangulate a region, area is what is seen and collided with.
    """
    if mesh.face_count == 0:
        raise MeshError("the extracted mesh has no faces")
    labels, faces, areas = face_components(mesh)
    largest = max(areas.values())
    ranked = sorted(areas, key=areas.get, reverse=True)
    kept = [lab for lab in ranked if areas[lab] >= min_area_fraction * largest]
    if max_components and len(kept) > max_components:
        kept = kept[:max_components]
    kept_set = set(kept)
    total_area = sum(areas.values())
    stats = {
        "components_before": len(areas),
        "components_after": len(kept),
        "faces_before": mesh.face_count,
        "faces_after": sum(faces[lab] for lab in kept),
        "area_kept_fraction": (sum(areas[lab] for lab in kept) / total_area) if total_area else 0.0,
    }
    if len(kept) == len(areas):
        return mesh, stats
    return select_faces(mesh, list(map(kept_set.__contains__, labels))), stats


def drop_degenerate_faces(mesh: Mesh, zero_area: bool = False) -> Mesh:
    """Remove faces that use the same vertex twice and, with `zero_area`, faces whose three
    corners lie on one line (both are left behind by welding and by edge collapses)."""
    a, b, c = mesh.tris[0::3], mesh.tris[1::3], mesh.tris[2::3]
    ne = operator.ne
    keep = [x and y and z for x, y, z in zip(map(ne, a, b), map(ne, b, c), map(ne, a, c))]
    if zero_area and mesh.face_count:
        x = mesh.xyz
        # Twice the area, squared, against the square of the longest edge: a sliver thinner
        # than a millionth of its length has no normal worth the name.
        for i, (p, q, r) in enumerate(zip(a, b, c)):
            if not keep[i]:
                continue
            p *= 3
            q *= 3
            r *= 3
            ux = x[q] - x[p]
            uy = x[q + 1] - x[p + 1]
            uz = x[q + 2] - x[p + 2]
            vx = x[r] - x[p]
            vy = x[r + 1] - x[p + 1]
            vz = x[r + 2] - x[p + 2]
            nx = uy * vz - uz * vy
            ny = uz * vx - ux * vz
            nz = ux * vy - uy * vx
            wx = vx - ux
            wy = vy - uy
            wz = vz - uz
            longest = max(ux * ux + uy * uy + uz * uz, vx * vx + vy * vy + vz * vz,
                          wx * wx + wy * wy + wz * wz)
            if nx * nx + ny * ny + nz * nz <= 1e-12 * longest * longest:
                keep[i] = False
    if all(keep):
        return mesh
    return select_faces(mesh, keep)


def weld(mesh: Mesh) -> Mesh:
    """Positions and triangles only, with vertices at exactly the same position made one."""
    same = _weld_map(mesh)
    if same is None:
        return Mesh(mesh.xyz, mesh.tris)
    tris = array.array("i", map(same.__getitem__, mesh.tris))
    return select_faces(Mesh(mesh.xyz, tris), [True] * mesh.face_count)


def vertex_normals(mesh: Mesh) -> array.array:
    """Area-weighted smooth vertex normals, unit length, as array('f') with 3 per vertex."""
    n = mesh.vertex_count
    nx = [0.0] * n
    ny = [0.0] * n
    nz = [0.0] * n
    x = mesh.xyz
    for a, b, c in zip(mesh.tris[0::3], mesh.tris[1::3], mesh.tris[2::3]):
        i = 3 * a
        j = 3 * b
        k = 3 * c
        ax = x[i]
        ay = x[i + 1]
        az = x[i + 2]
        ux = x[j] - ax
        uy = x[j + 1] - ay
        uz = x[j + 2] - az
        vx = x[k] - ax
        vy = x[k + 1] - ay
        vz = x[k + 2] - az
        fx = uy * vz - uz * vy
        fy = uz * vx - ux * vz
        fz = ux * vy - uy * vx
        nx[a] += fx
        ny[a] += fy
        nz[a] += fz
        nx[b] += fx
        ny[b] += fy
        nz[b] += fz
        nx[c] += fx
        ny[c] += fy
        nz[c] += fz
    out = array.array("f", bytes(12 * n))
    sqrt = math.sqrt
    for i in range(n):
        fx = nx[i]
        fy = ny[i]
        fz = nz[i]
        length = sqrt(fx * fx + fy * fy + fz * fz)
        if length > 1e-30:
            nx[i] = fx / length
            ny[i] = fy / length
            nz[i] = fz / length
        else:  # an unreferenced or fully degenerate vertex still needs a unit normal
            nx[i] = 0.0
            ny[i] = 0.0
            nz[i] = 1.0
    out[0::3] = array.array("f", nx)
    out[1::3] = array.array("f", ny)
    out[2::3] = array.array("f", nz)
    # Rounding to float32 can leave a normal a hair off unit length; that is within glTF's tolerance.
    return out


def _take3(values: array.array, source) -> array.array:
    """values[3*s : 3*s+3] for each s in source, as a flat array."""
    out = array.array("f", bytes(12 * len(source)))
    for k in range(3):
        column = values[k::3]
        out[k::3] = array.array("f", map(column.__getitem__, source))
    return out


def seamless_normals(mesh: Mesh) -> array.array:
    """Smooth vertex normals that agree on both sides of a texture seam.

    Seam vertices exist twice, once per chart; averaged separately they would shade as a
    crease along every chart border.
    """
    same = _weld_map(mesh)
    if same is None:
        return vertex_normals(mesh)
    welded = Mesh(mesh.xyz, array.array("i", map(same.__getitem__, mesh.tris)))
    return _take3(vertex_normals(welded), same)


# =============================================================================================
# Frames
# =============================================================================================

IDENTITY = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def read_scene_transform(run_dir) -> dict:
    """Read Spirula's scene_transform.json. Returns row-major 3x4 matrices for both directions.

    A run without the file was trained in the dataset frame, so both are the identity. The
    meshes are made and written in the training frame (the frame of the splat); only what is
    handed to COLMAP, which works in the dataset frame where the cameras are, is moved there
    with `world_from_train`.
    """
    path = Path(run_dir) / "scene_transform.json"
    if not path.exists():
        return {"train_from_world": IDENTITY, "world_from_train": IDENTITY, "identity": True}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        out = {}
        for key in ("train_from_world", "world_from_train"):
            m = [float(v) for v in doc[key]["matrix_3x4_flat_row_major"]]
            if len(m) != 12 or not all(math.isfinite(v) for v in m):
                raise ValueError(f"{key} is not a finite 3x4 matrix")
            out[key] = tuple(m)
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise MeshError(
            f"cannot read the scene transform {path} ({e}). It is written by Spirula when "
            f"training finishes; train the splat again if the file is damaged.") from e
    out["identity"] = all(abs(a - b) < 1e-9 for a, b in zip(out["train_from_world"], IDENTITY))
    return out


def transform_points(xyz: array.array, m) -> array.array:
    """Apply a row-major 3x4 matrix to every point (3 floats each)."""
    a, b, c, d, e, f, g, h, i, j, k, l = m
    xs = xyz[0::3]
    ys = xyz[1::3]
    zs = xyz[2::3]
    out = array.array("f", bytes(4 * len(xyz)))
    out[0::3] = array.array("f", [a * x + b * y + c * z + d for x, y, z in zip(xs, ys, zs)])
    out[1::3] = array.array("f", [e * x + f * y + g * z + h for x, y, z in zip(xs, ys, zs)])
    out[2::3] = array.array("f", [i * x + j * y + k * z + l for x, y, z in zip(xs, ys, zs)])
    return out


# =============================================================================================
# glTF binary
# =============================================================================================

_GLB_MAGIC = 0x46546C67
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN = 0x004E4942
_COMPONENT = {5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2), 5123: ("H", 2), 5125: ("I", 4),
              5126: ("f", 4)}
_TYPE_SIZE = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


def image_size(data) -> tuple:
    """(mime type, width, height) of a PNG or JPEG held in memory."""
    data = bytes(data[:26]) if len(data) < 26 else data
    if data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        w, h = struct.unpack_from(">II", data, 16)
        return "image/png", w, h
    if data[:2] == b"\xff\xd8":
        pos = 2
        while pos + 9 < len(data):
            if data[pos] != 0xFF:
                pos += 1
                continue
            kind = data[pos + 1]
            if kind in (0xC0, 0xC1, 0xC2):
                h, w = struct.unpack_from(">HH", data, pos + 5)
                return "image/jpeg", w, h
            if kind in (0xD8, 0x01) or 0xD0 <= kind <= 0xD7:
                pos += 2
                continue
            pos += 2 + struct.unpack_from(">H", data, pos + 2)[0]
        raise MeshError("the texture is a JPEG without a size header")
    raise MeshError("the texture is neither PNG nor JPEG")


def write_glb(path, *, positions: array.array, indices: array.array, normals=None, uvs=None,
              image: Optional[bytes] = None, name: str = "mesh", extras: Optional[dict] = None) -> None:
    """Write one indexed triangle primitive as a self-contained GLB.

    positions/normals: array('f') with 3 per vertex; uvs: array('f') with 2 per vertex, origin at
    the top-left of the image as glTF defines it; indices: 3 per triangle; image: PNG/JPEG bytes.
    """
    n_vertices = len(positions) // 3
    if n_vertices == 0 or len(indices) == 0:
        raise MeshError(f"refusing to write an empty mesh to {path}")
    chunks = []
    views = []
    accessors = []
    offset = 0

    def add_view(raw: bytes, target: Optional[int]) -> int:
        nonlocal offset
        pad = (-offset) % 4
        if pad:
            chunks.append(b"\x00" * pad)
            offset += pad
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(raw)}
        if target:
            view["target"] = target
        views.append(view)
        chunks.append(raw)
        offset += len(raw)
        return len(views) - 1

    accessors.append({"bufferView": add_view(indices.tobytes(), 34963), "componentType": 5125,
                      "count": len(indices), "type": "SCALAR"})
    attributes = {"POSITION": len(accessors)}
    accessors.append({
        "bufferView": add_view(positions.tobytes(), 34962), "componentType": 5126,
        "count": n_vertices, "type": "VEC3",
        "min": [min(positions[k::3]) for k in range(3)],
        "max": [max(positions[k::3]) for k in range(3)]})
    if normals is not None:
        attributes["NORMAL"] = len(accessors)
        accessors.append({"bufferView": add_view(normals.tobytes(), 34962), "componentType": 5126,
                          "count": n_vertices, "type": "VEC3"})
    if uvs is not None:
        attributes["TEXCOORD_0"] = len(accessors)
        accessors.append({"bufferView": add_view(uvs.tobytes(), 34962), "componentType": 5126,
                          "count": n_vertices, "type": "VEC2"})

    # Scans are open surfaces seen from both sides, so nothing is back-face culled.
    material = {"name": name, "pbrMetallicRoughness": {"metallicFactor": 0, "roughnessFactor": 1},
                "doubleSided": True}
    doc = {
        "asset": {"version": "2.0", "generator": "scan3d meshpost"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{"name": name, "primitives": [
            {"attributes": attributes, "indices": 0, "material": 0, "mode": 4}]}],
        "materials": [material],
        "accessors": accessors,
    }
    if image is not None and uvs is not None:
        mime, _, _ = image_size(image)
        material["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}
        doc["samplers"] = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}]
        doc["textures"] = [{"sampler": 0, "source": 0}]
        doc["images"] = [{"bufferView": add_view(bytes(image), None), "mimeType": mime}]
    if extras:
        doc["asset"]["extras"] = extras
    pad = (-offset) % 4
    if pad:
        chunks.append(b"\x00" * pad)
        offset += pad
    doc["bufferViews"] = views
    doc["buffers"] = [{"byteLength": offset}]

    text = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    text += b" " * ((-len(text)) % 4)
    total = 12 + 8 + len(text) + 8 + offset
    if total >= 2 ** 32:
        raise MeshError("the mesh is larger than a GLB file can hold (4 GB)")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<III", _GLB_MAGIC, 2, total))
        f.write(struct.pack("<II", len(text), _CHUNK_JSON))
        f.write(text)
        f.write(struct.pack("<II", offset, _CHUNK_BIN))
        for chunk in chunks:
            f.write(chunk)
    os.replace(tmp, path)  # readers never see a half-written file


def validate_glb(path) -> dict:
    """Check that `path` is structurally valid binary glTF 2.0 and return what it holds.

    Verified: header, chunk layout and padding, buffer and bufferView bounds, accessor bounds and
    alignment, POSITION min/max against the real data, equal attribute counts, index range,
    triangle count, finite values, unit normals, texture coordinates inside [0, 1], image
    signature against its mimeType, and every scene/node/mesh/material/texture reference.
    Raises MeshError naming the first problem found.
    """
    path = Path(path)

    def fail(msg: str):
        raise MeshError(f"{path.name} is not a valid GLB: {msg}")

    try:
        raw = path.read_bytes()
    except OSError as e:
        raise MeshError(f"cannot read {path}: {e}") from e
    if len(raw) < 20:
        fail("the file is too short to hold a header")
    magic, version, total = struct.unpack_from("<III", raw, 0)
    if magic != _GLB_MAGIC:
        fail("wrong magic number")
    if version != 2:
        fail(f"container version {version}, expected 2")
    if total != len(raw):
        fail(f"header says {total} bytes but the file has {len(raw)}")
    doc = None
    binary = None
    pos = 12
    index = 0
    while pos < total:
        if pos + 8 > total:
            fail("truncated chunk header")
        length, kind = struct.unpack_from("<II", raw, pos)
        if length % 4 or pos % 4:
            fail("a chunk is not aligned to 4 bytes")
        if pos + 8 + length > total:
            fail("a chunk runs past the end of the file")
        if index == 0:
            if kind != _CHUNK_JSON:
                fail("the first chunk is not JSON")
            try:
                doc = json.loads(raw[pos + 8:pos + 8 + length].decode("utf-8"))
            except ValueError as e:
                fail(f"the JSON chunk does not parse ({e})")
        elif index == 1:
            if kind != _CHUNK_BIN:
                fail("the second chunk is not BIN")
            binary = memoryview(raw)[pos + 8:pos + 8 + length]
        index += 1
        pos += 8 + length
    if not isinstance(doc, dict):
        fail("no JSON chunk")
    if doc.get("asset", {}).get("version") != "2.0":
        fail("asset.version is not 2.0")

    buffers = doc.get("buffers", [])
    if len(buffers) != 1 or "uri" in buffers[0]:
        fail("expected exactly one buffer, stored in the BIN chunk")
    if binary is None:
        fail("no BIN chunk")
    buffer_length = buffers[0].get("byteLength", -1)
    if not (0 < buffer_length <= len(binary) <= buffer_length + 3):
        fail(f"buffer.byteLength {buffer_length} does not match the BIN chunk ({len(binary)} bytes)")

    views = doc.get("bufferViews", [])
    for i, view in enumerate(views):
        start = view.get("byteOffset", 0)
        if view.get("buffer") != 0 or start < 0 or view.get("byteLength", 0) <= 0 \
                or start + view["byteLength"] > buffer_length:
            fail(f"bufferView {i} lies outside the buffer")

    accessors = doc.get("accessors", [])

    def read_accessor(i: int, want_type: str, want_components: tuple) -> array.array:
        if not isinstance(i, int) or not 0 <= i < len(accessors):
            fail(f"accessor index {i} does not exist")
        acc = accessors[i]
        if acc.get("type") != want_type or acc.get("componentType") not in want_components:
            fail(f"accessor {i} should be {want_type} of component type {want_components}")
        if "sparse" in acc or not isinstance(acc.get("bufferView"), int) \
                or not 0 <= acc["bufferView"] < len(views):
            fail(f"accessor {i} has no usable bufferView")
        view = views[acc["bufferView"]]
        code, size = _COMPONENT[acc["componentType"]]
        width = size * _TYPE_SIZE[want_type]
        start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        count = acc.get("count", 0)
        stride = view.get("byteStride", width)
        if count <= 0:
            fail(f"accessor {i} is empty")
        if start % size:
            fail(f"accessor {i} is not aligned to its component size")
        if stride != width:
            fail(f"accessor {i} uses an interleaved bufferView, which this checker does not read")
        if acc.get("byteOffset", 0) + count * width > view["byteLength"]:
            fail(f"accessor {i} runs past its bufferView")
        return _from_bytes(code, binary[start:start + count * width])

    meshes = doc.get("meshes", [])
    materials = doc.get("materials", [])
    textures = doc.get("textures", [])
    images = doc.get("images", [])
    samplers = doc.get("samplers", [])
    nodes = doc.get("nodes", [])
    scenes = doc.get("scenes", [])
    if not scenes or not 0 <= doc.get("scene", 0) < len(scenes):
        fail("no default scene")
    used_meshes = set()
    for scene in scenes:
        stack = list(scene.get("nodes", []))
        seen = set()
        while stack:
            n = stack.pop()
            if not isinstance(n, int) or not 0 <= n < len(nodes):
                fail(f"node index {n} does not exist")
            if n in seen:
                fail("the node graph has a cycle or a shared child")
            seen.add(n)
            node = nodes[n]
            if "mesh" in node:
                if not 0 <= node["mesh"] < len(meshes):
                    fail(f"node {n} refers to a mesh that does not exist")
                used_meshes.add(node["mesh"])
            stack.extend(node.get("children", []))
    if not used_meshes:
        fail("the scene draws no mesh")

    n_faces = 0
    n_vertices = 0
    has_uv = False
    has_normals = False
    texture = None
    bbox_min = [math.inf] * 3
    bbox_max = [-math.inf] * 3
    for mi in sorted(used_meshes):
        for prim in meshes[mi].get("primitives", []):
            if prim.get("mode", 4) != 4:
                fail("a primitive is not a triangle list")
            attrs = prim.get("attributes", {})
            if "POSITION" not in attrs:
                fail("a primitive has no POSITION")
            acc = accessors[attrs["POSITION"]] if isinstance(attrs["POSITION"], int) \
                and 0 <= attrs["POSITION"] < len(accessors) else fail("POSITION accessor missing")
            p = read_accessor(attrs["POSITION"], "VEC3", (5126,))
            count = len(p) // 3
            lo = [min(p[k::3]) for k in range(3)]
            hi = [max(p[k::3]) for k in range(3)]
            if not all(math.isfinite(v) for v in lo + hi):
                fail("POSITION holds NaN or infinity")
            if acc.get("min") != lo or acc.get("max") != hi:
                fail(f"POSITION min/max {acc.get('min')}..{acc.get('max')} do not match the data "
                     f"{lo}..{hi}")
            bbox_min = [min(a, b) for a, b in zip(bbox_min, lo)]
            bbox_max = [max(a, b) for a, b in zip(bbox_max, hi)]
            if "NORMAL" in attrs:
                nrm = read_accessor(attrs["NORMAL"], "VEC3", (5126,))
                if len(nrm) != len(p):
                    fail("NORMAL and POSITION have different counts")
                for x, y, z in zip(nrm[0::3], nrm[1::3], nrm[2::3]):
                    if not 0.99 < x * x + y * y + z * z < 1.01:  # also false for NaN
                        fail("a normal is not unit length")
                has_normals = True
            if "TEXCOORD_0" in attrs:
                uv = read_accessor(attrs["TEXCOORD_0"], "VEC2", (5126,))
                if len(uv) // 2 != count:
                    fail("TEXCOORD_0 and POSITION have different counts")
                if not (min(uv) >= 0.0 and max(uv) <= 1.0):  # also false for NaN
                    fail("texture coordinates fall outside [0, 1]")
                has_uv = True
            if "indices" in prim:
                idx = read_accessor(prim["indices"], "SCALAR", (5121, 5123, 5125))
                if len(idx) % 3:
                    fail("the index count is not a multiple of 3")
                if max(idx) >= count:
                    fail("an index points past the last vertex")
                n_faces += len(idx) // 3
            else:
                if count % 3:
                    fail("the vertex count of a non-indexed primitive is not a multiple of 3")
                n_faces += count // 3
            n_vertices += count
            if "material" in prim:
                mat = prim["material"]
                if not isinstance(mat, int) or not 0 <= mat < len(materials):
                    fail("a primitive refers to a material that does not exist")
                base = materials[mat].get("pbrMetallicRoughness", {}).get("baseColorTexture")
                if base is not None:
                    if "TEXCOORD_0" not in attrs:
                        fail("a textured primitive has no TEXCOORD_0")
                    ti = base.get("index")
                    if not isinstance(ti, int) or not 0 <= ti < len(textures):
                        fail("the material refers to a texture that does not exist")
                    tex = textures[ti]
                    if "sampler" in tex and not 0 <= tex["sampler"] < len(samplers):
                        fail("the texture refers to a sampler that does not exist")
                    src = tex.get("source")
                    if not isinstance(src, int) or not 0 <= src < len(images):
                        fail("the texture refers to an image that does not exist")
                    img = images[src]
                    vi = img.get("bufferView")
                    if not isinstance(vi, int) or not 0 <= vi < len(views):
                        fail("the image is not embedded in the file")
                    view = views[vi]
                    blob = binary[view.get("byteOffset", 0):view.get("byteOffset", 0) + view["byteLength"]]
                    try:
                        mime, w, h = image_size(bytes(blob[:65536]))
                    except MeshError as e:
                        fail(str(e))
                    if mime != img.get("mimeType"):
                        fail(f"the image is {mime} but is declared as {img.get('mimeType')}")
                    if mime == "image/png" and bytes(blob[-8:]) != b"IEND\xaeB`\x82":
                        fail("the embedded PNG is truncated")
                    texture = (w, h)
    if n_faces == 0:
        fail("the mesh has no triangles")
    return {"path": str(path), "bytes": len(raw), "faces": n_faces, "vertices": n_vertices,
            "has_normals": has_normals, "has_uv": has_uv,
            "texture_width": texture[0] if texture else 0,
            "texture_height": texture[1] if texture else 0,
            "bbox_min": bbox_min, "bbox_max": bbox_max,
            "extras": doc.get("asset", {}).get("extras")}


def read_glb(path):
    """Read the one triangle mesh of a GLB such as `write_glb` writes (for checks and tests).

    Returns (mesh, image): positions, triangles and texture coordinates exactly as stored, and
    the embedded texture file (PNG or JPEG bytes), or None when the mesh has no texture.
    Normals are not read; they are recomputed. A file whose node carries a transform is
    refused: its vertex numbers would no longer be splat coordinates.
    """
    path = Path(path)

    def fail(msg: str):
        raise MeshError(f"cannot use the mesh in {path}: {msg}")

    try:
        raw = path.read_bytes()
    except OSError as e:
        raise MeshError(f"cannot read mesh file {path}: {e}") from e
    if len(raw) < 28 or struct.unpack_from("<II", raw, 0) != (_GLB_MAGIC, 2):
        fail("it is not a binary glTF 2.0 file")
    json_length, json_kind = struct.unpack_from("<II", raw, 12)
    start = 20 + json_length
    if json_kind != _CHUNK_JSON or start + 8 > len(raw):
        fail("it is truncated")
    try:
        doc = json.loads(raw[20:start].decode("utf-8"))
    except ValueError as e:
        fail(f"its JSON chunk does not parse ({e})")
    bin_length, bin_kind = struct.unpack_from("<II", raw, start)
    if bin_kind != _CHUNK_BIN or start + 8 + bin_length > len(raw):
        fail("it is truncated")
    binary = memoryview(raw)[start + 8:start + 8 + bin_length]

    try:
        nodes = [n for n in doc["nodes"] if "mesh" in n]
        if len(nodes) != 1 or len(doc["meshes"]) != 1 or len(doc["meshes"][0]["primitives"]) != 1:
            fail("expected exactly one mesh with one primitive")
        if any(k in n for n in doc["nodes"] for k in ("matrix", "rotation", "translation", "scale")):
            fail("a node carries a transform, so the vertex numbers are not splat coordinates")
        prim = doc["meshes"][0]["primitives"][0]
        if prim.get("mode", 4) != 4 or "indices" not in prim:
            fail("the primitive is not an indexed triangle list")

        def column(index: int, kind: str, components: tuple) -> array.array:
            acc = doc["accessors"][index]
            if acc["type"] != kind or acc["componentType"] not in components or "sparse" in acc:
                fail(f"accessor {index} is not plain {kind} data of type {components}")
            view = doc["bufferViews"][acc["bufferView"]]
            code, size = _COMPONENT[acc["componentType"]]
            width = size * _TYPE_SIZE[kind]
            first = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
            stride = view.get("byteStride") or width
            count = acc["count"]
            if count <= 0 or first + stride * (count - 1) + width > len(binary):
                fail(f"accessor {index} runs past the end of the file")
            return _from_bytes(code, _gather(binary, first, stride, count, 0, width))

        attributes = prim["attributes"]
        xyz = column(attributes["POSITION"], "VEC3", (5126,))
        index = column(prim["indices"], "SCALAR", (5121, 5123, 5125))
        uv = column(attributes["TEXCOORD_0"], "VEC2", (5126,)) if "TEXCOORD_0" in attributes else None
        image = None
        material = doc["materials"][prim["material"]] if "material" in prim else {}
        base = material.get("pbrMetallicRoughness", {}).get("baseColorTexture")
        if base is not None and uv is not None:
            source = doc["images"][doc["textures"][base["index"]]["source"]]
            view = doc["bufferViews"][source["bufferView"]]
            first = view.get("byteOffset", 0)
            image = bytes(binary[first:first + view["byteLength"]])
            if len(image) != view["byteLength"]:
                fail("the texture runs past the end of the file")
    except (KeyError, IndexError, TypeError) as e:
        fail(f"its structure is not a single indexed triangle mesh ({e!r})")
    if len(index) % 3 or not len(index):
        fail("the index count is not a positive multiple of 3")
    if max(index) >= len(xyz) // 3:
        fail("a face refers to a vertex that does not exist")
    if uv is not None and len(uv) // 2 != len(xyz) // 3:
        fail("positions and texture coordinates have different counts")
    if not all(math.isfinite(v) for v in (min(xyz), max(xyz))):
        fail("a vertex position is not a finite number")
    return Mesh(xyz, array.array("i", index), uv if image is not None else None), image


# =============================================================================================
# COLMAP camera model, PNG masks, PLY
# =============================================================================================

# model id -> (name, number of parameters)
CAMERA_MODELS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5),
                 4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8), 6: ("FULL_OPENCV", 12)}


def read_cameras_bin(path) -> dict:
    data = Path(path).read_bytes()
    n = struct.unpack_from("<Q", data, 0)[0]
    pos = 8
    out = {}
    for _ in range(n):
        cid, model, w, h = struct.unpack_from("<iiQQ", data, pos)
        pos += 24
        if model not in CAMERA_MODELS:
            raise MeshError(f"camera model {model} in {path} is not supported")
        k = CAMERA_MODELS[model][1]
        params = struct.unpack_from("<%dd" % k, data, pos)
        pos += 8 * k
        out[cid] = (model, w, h, params)
    return out


def read_images_bin(path) -> list:
    """[(image_id, name, R row-major 9-tuple (cam_from_world), t, camera_id)]."""
    data = Path(path).read_bytes()
    n = struct.unpack_from("<Q", data, 0)[0]
    pos = 8
    out = []
    for _ in range(n):
        iid = struct.unpack_from("<i", data, pos)[0]
        qw, qx, qy, qz = struct.unpack_from("<4d", data, pos + 4)
        t = struct.unpack_from("<3d", data, pos + 36)
        cid = struct.unpack_from("<i", data, pos + 60)[0]
        pos += 64
        end = data.index(b"\x00", pos)
        name = data[pos:end].decode("utf-8")
        pos = end + 1
        m = struct.unpack_from("<Q", data, pos)[0]
        pos += 8 + 24 * m
        s = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz) or 1.0
        qw, qx, qy, qz = qw / s, qx / s, qy / s, qz / s
        R = (1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy),
             2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx),
             2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy))
        out.append((iid, name, R, t, cid))
    return out


def distort(model: int, p, x: float, y: float):
    """Normalised undistorted (x, y) -> pixel (u, v) in the distorted image."""
    if model == 0:
        f, cx, cy = p
        return f * x + cx, f * y + cy
    if model == 1:
        fx, fy, cx, cy = p
        return fx * x + cx, fy * y + cy
    if model in (2, 3):
        f, cx, cy, k1 = p[:4]
        k2 = p[4] if model == 3 else 0.0
        r2 = x * x + y * y
        d = 1 + k1 * r2 + k2 * r2 * r2
        return f * x * d + cx, f * y * d + cy
    if model in (4, 6):
        fx, fy, cx, cy, k1, k2, p1, p2 = p[:8]
        r2 = x * x + y * y
        if model == 4:
            d = 1 + k1 * r2 + k2 * r2 * r2
        else:
            k3, k4, k5, k6 = p[8:12]
            d = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1 + r2 * (k4 + r2 * (k5 + r2 * k6)))
        xd = x * d + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * d + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        return fx * xd + cx, fy * yd + cy
    if model == 5:
        fx, fy, cx, cy, k1, k2, k3, k4 = p
        r = math.sqrt(x * x + y * y)
        if r < 1e-12:
            return fx * x + cx, fy * y + cy
        th = math.atan(r)
        t2 = th * th
        thd = th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
        return fx * x * thd / r + cx, fy * y * thd / r + cy
    raise MeshError(f"camera model {model} is not supported")


def undistort(model: int, p, u: float, v: float):
    """Pixel (u, v) in the distorted image -> normalised undistorted (x, y), by Newton steps."""
    if model == 0:
        f, cx, cy = p
        fx = fy = f
    elif model == 2 or model == 3:
        f, cx, cy = p[:3]
        fx = fy = f
    else:
        fx, fy, cx, cy = p[:4]
    xd = (u - cx) / fx
    yd = (v - cy) / fy
    if model in (0, 1):
        return xd, yd
    x, y = xd, yd
    for _ in range(30):
        uu, vv = distort(model, p, x, y)
        ex = (uu - u) / fx
        ey = (vv - v) / fy
        if ex * ex + ey * ey < 1e-24:
            break
        h = 1e-7
        ux, vx = distort(model, p, x + h, y)
        uy, vy = distort(model, p, x, y + h)
        a = (ux - uu) / (fx * h)
        b = (uy - uu) / (fx * h)
        c = (vx - vv) / (fy * h)
        d = (vy - vv) / (fy * h)
        det = a * d - b * c
        if abs(det) < 1e-12:
            break
        x -= (d * ex - b * ey) / det
        y -= (-c * ex + a * ey) / det
    return x, y


def _png_chunks(data: bytes):
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise MeshError("not a PNG file")
    pos = 8
    while pos + 8 <= len(data):
        length, kind = struct.unpack_from(">I4s", data, pos)
        yield kind, data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if kind == b"IEND":
            break


def _unfilter(raw: bytes, height: int, row_bytes: int, bpp: int) -> list:
    """Undo PNG row filters. Returns a list of `height` bytes objects."""
    rows = []
    prev = bytes(row_bytes)
    stride = row_bytes + 1
    if len(raw) < height * stride:
        raise MeshError("the PNG image data is truncated")
    for y in range(height):
        kind = raw[y * stride]
        line = raw[y * stride + 1:(y + 1) * stride]
        if kind == 0:
            cur = line
        elif kind == 2:
            if not any(line):
                cur = prev
            else:
                cur = bytes((a + b) & 255 for a, b in zip(line, prev))
        elif kind == 1:
            out = bytearray(line)
            for i in range(bpp, row_bytes):
                out[i] = (out[i] + out[i - bpp]) & 255
            cur = bytes(out)
        elif kind == 3:
            out = bytearray(line)
            for i in range(row_bytes):
                left = out[i - bpp] if i >= bpp else 0
                out[i] = (out[i] + ((left + prev[i]) >> 1)) & 255
            cur = bytes(out)
        elif kind == 4:
            out = bytearray(line)
            for i in range(row_bytes):
                a = out[i - bpp] if i >= bpp else 0
                b = prev[i]
                c = prev[i - bpp] if i >= bpp else 0
                pa = abs(b - c)
                pb = abs(a - c)
                pc = abs(a + b - 2 * c)
                pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                out[i] = (out[i] + pred) & 255
            cur = bytes(out)
        else:
            raise MeshError(f"unknown PNG filter {kind}")
        rows.append(cur)
        prev = cur
    return rows


def read_mask_png(path):
    """A mask PNG as (width, height, rows): one bytes object per row, one byte per pixel, where
    0 means masked (the pixel value was below half) and 1 means kept."""
    data = Path(path).read_bytes()
    ihdr = None
    idat = []
    for kind, body in _png_chunks(data):
        if kind == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat.append(body)
    if ihdr is None:
        raise MeshError(f"{path} has no PNG header")
    w, h, depth, ctype, _, _, interlace = ihdr
    if interlace:
        raise MeshError(f"{path} is an interlaced PNG, which is not supported for masks")
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(ctype)
    if channels is None or depth not in (1, 2, 4, 8, 16) or (depth < 8 and ctype not in (0, 3)):
        raise MeshError(f"{path}: unsupported PNG format (bit depth {depth}, colour type {ctype})")
    if ctype == 3:
        raise MeshError(f"{path}: palette masks are not supported; write it as greyscale")
    bits = depth * channels
    row_bytes = (w * bits + 7) // 8
    rows = _unfilter(zlib.decompress(b"".join(idat)), h, row_bytes, max(1, bits // 8))
    out = []
    if depth == 8:
        step = channels
        table = bytes(1 if v >= 128 else 0 for v in range(256))
        for r in rows:
            out.append(r[0::step].translate(table) if step > 1 else r.translate(table))
    elif depth == 16:
        step = 2 * channels
        table = bytes(1 if v >= 128 else 0 for v in range(256))
        for r in rows:
            out.append(r[0::step].translate(table))
    else:
        per = 8 // depth
        half = 1 << (depth - 1)
        mask = (1 << depth) - 1
        for r in rows:
            vals = bytearray(w)
            for x in range(w):
                byte = r[x // per]
                shift = 8 - depth * (x % per + 1)
                vals[x] = 1 if ((byte >> shift) & mask) >= half else 0
            out.append(bytes(vals))
    return w, h, out


def pad_png_rows(data: bytes, new_height: int) -> bytes:
    """Return the PNG with rows of zeros added at the bottom up to `new_height` (8-bit RGB/RGBA/grey)."""
    ihdr = None
    idat = []
    other_before = []
    for kind, body in _png_chunks(data):
        if kind == b"IHDR":
            ihdr = body
        elif kind == b"IDAT":
            idat.append(body)
    w, h, depth, ctype, comp, filt, interlace = struct.unpack(">IIBBBBB", ihdr)
    if interlace or depth != 8 or ctype not in (0, 2, 4, 6):
        raise MeshError("the texture atlas is not a plain 8-bit PNG")
    if new_height <= h:
        return data
    channels = {0: 1, 2: 3, 4: 2, 6: 4}[ctype]
    raw = zlib.decompress(b"".join(idat))
    raw += bytes((w * channels + 1) * (new_height - h))
    body = zlib.compress(raw, 6)

    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))
    head = struct.pack(">IIBBBBB", w, new_height, depth, ctype, comp, filt, 0)
    out = [b"\x89PNG\r\n\x1a\n", chunk(b"IHDR", head)]
    step = 1 << 24
    for i in range(0, len(body), step):
        out.append(chunk(b"IDAT", body[i:i + step]))
    out.append(chunk(b"IEND", b""))
    return b"".join(out)


def masked_cells(mw: int, mh: int, rows: list, cell: int) -> set:
    """Grid cells (cx, cy) of `cell` mask pixels that hold at least one masked pixel, grown by one cell."""
    found = set()
    for y, r in enumerate(rows):
        if 0 not in r:
            continue
        cy = y // cell
        x = r.find(0)
        while x >= 0:
            cx = x // cell
            found.add((cx, cy))
            nxt = (cx + 1) * cell
            x = r.find(0, nxt) if nxt < mw else -1
    grown = set()
    for cx, cy in found:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                grown.add((cx + dx, cy + dy))
    nx = (mw + cell - 1) // cell
    ny = (mh + cell - 1) // cell
    return {(cx, cy) for cx, cy in grown if 0 <= cx < nx and 0 <= cy < ny}


def occluder_triangles(cells: set, cell: int, mw: int, mh: int, camera, R, t, depth: float,
                       max_run: int = 8) -> list:
    """Triangles (9 floats each, world frame) just in front of the camera that cover the masked
    cells as seen from it, turned so that their front faces away from the camera."""
    model, cw, ch, params = camera
    sx = cw / mw
    sy = ch / mh
    # camera centre and the camera-to-world rotation (R transposed)
    C = (-(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]),
         -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]),
         -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]))
    corner_cache = {}

    def world(gx: int, gy: int):
        key = (gx, gy)
        p = corner_cache.get(key)
        if p is None:
            u = min(gx * cell, mw) * sx
            v = min(gy * cell, mh) * sy
            x, y = undistort(model, params, u, v)
            dx = R[0] * x + R[3] * y + R[6]
            dy = R[1] * x + R[4] * y + R[7]
            dz = R[2] * x + R[5] * y + R[8]
            p = (C[0] + depth * dx, C[1] + depth * dy, C[2] + depth * dz)
            corner_cache[key] = p
        return p

    rows = {}
    for cx, cy in cells:
        rows.setdefault(cy, []).append(cx)
    tris = []
    for cy, xs in rows.items():
        xs.sort()
        start = xs[0]
        prev = start
        runs = []
        for x in xs[1:] + [None]:
            if x is not None and x == prev + 1 and x - start < max_run:
                prev = x
                continue
            runs.append((start, prev + 1))
            if x is not None:
                start = prev = x
        for x0, x1 in runs:
            a = world(x0, cy)
            b = world(x1, cy)
            c = world(x1, cy + 1)
            d = world(x0, cy + 1)
            for p, q, r in ((a, b, c), (a, c, d)):
                ux, uy, uz = q[0] - p[0], q[1] - p[1], q[2] - p[2]
                vx, vy, vz = r[0] - p[0], r[1] - p[1], r[2] - p[2]
                nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
                gx = C[0] - (p[0] + q[0] + r[0]) / 3
                gy = C[1] - (p[1] + q[1] + r[1]) / 3
                gz = C[2] - (p[2] + q[2] + r[2]) / 3
                if nx * gx + ny * gy + nz * gz > 0:
                    q, r = r, q
                tris.append(p + q + r)
    return tris


_PLY_TYPES = {"char": ("b", 1), "int8": ("b", 1), "uchar": ("B", 1), "uint8": ("B", 1),
              "short": ("h", 2), "int16": ("h", 2), "ushort": ("H", 2), "uint16": ("H", 2),
              "int": ("i", 4), "int32": ("i", 4), "uint": ("I", 4), "uint32": ("I", 4),
              "float": ("f", 4), "float32": ("f", 4), "double": ("d", 8), "float64": ("d", 8)}


def write_ply(path, xyz: array.array, tris: array.array, extra_tris: list = ()) -> None:
    """Binary PLY: the mesh, then `extra_tris` (9 floats each) as separate faces with their own vertices."""
    nv = len(xyz) // 3
    ne = len(extra_tris)
    head = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\n"
            "property float z\nelement face %d\nproperty list uchar int vertex_indices\nend_header\n"
            % (nv + 3 * ne, len(tris) // 3 + ne)).encode("ascii")
    with open(path, "wb") as f:
        f.write(head)
        f.write(xyz.tobytes())
        if ne:
            f.write(array.array("f", [v for tri in extra_tris for v in tri]).tobytes())
        rec = struct.Struct("<Biii")
        body = bytearray()
        for a, b, c in zip(tris[0::3], tris[1::3], tris[2::3]):
            body += rec.pack(3, a, b, c)
        for k in range(ne):
            base = nv + 3 * k
            body += rec.pack(3, base, base + 1, base + 2)
        f.write(body)


def read_ply(path):
    """A binary little-endian triangle PLY as (xyz array('f'), tris array('i'), face_uv array('f') or None).

    Vertex properties other than x, y, z are skipped; faces must be triangles, optionally with a
    six-entry `texcoord` list (as `colmap mesh_texturer` writes)."""
    raw = Path(path).read_bytes()
    i = raw.find(b"end_header")
    if not raw.startswith(b"ply") or i < 0:
        raise MeshError(f"{path} is not a PLY file")
    start = raw.index(b"\n", i) + 1
    elements = []
    for line in raw[:start].decode("ascii", "replace").replace("\r", "").split("\n"):
        p = line.split()
        if not p:
            continue
        if p[0] == "format" and p[1] != "binary_little_endian":
            raise MeshError(f"{path} is not a binary little-endian PLY")
        if p[0] == "element":
            elements.append((p[1], int(p[2]), []))
        elif p[0] == "property":
            elements[-1][2].append(p[1:])
    data = memoryview(raw)
    pos = start
    xyz = tris = uv = None
    for name, count, props in elements:
        if name == "vertex":
            offsets = {}
            size = 0
            for p in props:
                if p[0] == "list":
                    raise MeshError(f"{path}: list property on vertices")
                code, width = _PLY_TYPES[p[0]]
                offsets[p[1]] = (size, code, width)
                size += width
            if any(offsets.get(k, (0, "f"))[1] != "f" for k in "xyz") or not all(k in offsets for k in "xyz"):
                raise MeshError(f"{path}: vertices need float x, y, z")
            xyz = array.array("f", bytes(12 * count))
            for k, axis in enumerate("xyz"):
                col = array.array("f")
                col.frombytes(_gather(data, pos, size, count, offsets[axis][0], 4))
                xyz[k::3] = col
            pos += size * count
        elif name == "face":
            # fixed layout: every list holds the same number of entries as the first record
            layout = []
            q = pos
            for p in props:
                if p[0] == "list":
                    ccode, cw = _PLY_TYPES[p[1]]
                    icode, iw = _PLY_TYPES[p[2]]
                    n = struct.unpack_from("<" + ccode, raw, q)[0] if count else 0
                    layout.append((p[3], q - pos, cw, icode, iw, n))
                    q += cw + iw * n
                else:
                    code, w = _PLY_TYPES[p[0]]
                    layout.append((p[1], q - pos, 0, code, w, 1))
                    q += w
            rec = q - pos
            for name_, off, cw, icode, iw, n in layout:
                if cw:
                    counts = bytes(data[pos + off:pos + off + rec * count:rec]) if cw == 1 else None
                    if counts is not None and counts.count(n) != count:
                        raise MeshError(f"{path}: faces of mixed size; only triangles are supported")
                if name_ in ("vertex_indices", "vertex_index"):
                    if n != 3 or icode not in ("i", "I"):
                        raise MeshError(f"{path}: faces are not triangles with int indices")
                    tris = array.array("i")
                    tris.frombytes(_gather(data, pos, rec, count, off + cw, 12))
                elif name_ == "texcoord" and n == 6 and icode == "f":
                    uv = array.array("f")
                    uv.frombytes(_gather(data, pos, rec, count, off + cw, 24))
            pos += rec * count
        else:
            raise MeshError(f"{path}: unexpected element {name}")
    if xyz is None or tris is None:
        raise MeshError(f"{path} holds no triangle mesh")
    if pos > len(raw):
        raise MeshError(f"{path} is truncated")
    if len(tris) and (max(tris) >= len(xyz) // 3 or min(tris) < 0):
        raise MeshError(f"{path}: a face refers to a vertex that does not exist")
    return xyz, tris, uv


def assemble_textured(xyz: array.array, tris: array.array, face_uv: array.array, atlas_w: int,
                      atlas_h: int, edge: int):
    """Turn per-face texture coordinates from COLMAP into a vertex-indexed mesh for glTF.

    Faces COLMAP could not give a view (all six coordinates 0) take one flat colour: the texel at
    a vertex they share with a textured face, spreading ring by ring. Faces that no such ring
    reaches (pieces that no camera saw from the front) are left out.
    Returns (xyz, tris, uv, stats) with uv in glTF convention for an `edge` x `edge` image whose
    top-left `atlas_w` x `atlas_h` holds the atlas.
    """
    n = len(tris) // 3
    su = atlas_w / edge
    sv = atlas_h / edge
    # corner coordinates in glTF convention
    U = [0.0] * (3 * n)
    V = [0.0] * (3 * n)
    textured = bytearray(n)
    at_vertex = {}
    for f in range(n):
        o = 6 * f
        c = face_uv[o:o + 6]
        if not any(c):
            continue
        textured[f] = 1
        for k in range(3):
            u = c[2 * k] * su
            v = (1.0 - c[2 * k + 1]) * sv
            U[3 * f + k] = u
            V[3 * f + k] = v
            vert = tris[3 * f + k]
            if vert not in at_vertex:
                at_vertex[vert] = (u, v)
    pending = [f for f in range(n) if not textured[f]]
    filled = 0
    while pending:
        later = []
        reached = []
        for f in pending:
            a, b, c = tris[3 * f], tris[3 * f + 1], tris[3 * f + 2]
            uv = at_vertex.get(a) or at_vertex.get(b) or at_vertex.get(c)
            if uv is None:
                later.append(f)
            else:
                reached.append((f, uv))
        if not reached:
            break
        for f, (u, v) in reached:
            for k in range(3):
                U[3 * f + k] = u
                V[3 * f + k] = v
                vert = tris[3 * f + k]
                if vert not in at_vertex:
                    at_vertex[vert] = (u, v)
            textured[f] = 2
        filled += len(reached)
        pending = later
    dropped = len(pending)
    # one output vertex per (vertex, coordinate) pair
    pairs = {}
    number = pairs.setdefault
    out_tris = array.array("i")
    for f in range(n):
        if not textured[f]:
            continue
        for k in range(3):
            j = 3 * f + k
            out_tris.append(number((tris[j], U[j], V[j]), len(pairs)))
    out_xyz = array.array("f", bytes(12 * len(pairs)))
    out_uv = array.array("f", bytes(8 * len(pairs)))
    for i, (vert, u, v) in enumerate(pairs):
        out_xyz[3 * i:3 * i + 3] = xyz[3 * vert:3 * vert + 3]
        out_uv[2 * i] = min(1.0, max(0.0, u))
        out_uv[2 * i + 1] = min(1.0, max(0.0, v))
    stats = {"faces_flat_coloured": filled, "faces_unseen_dropped": dropped}
    return out_xyz, out_tris, out_uv, stats


_PLY_TYPES = {"char": ("b", 1), "int8": ("b", 1), "uchar": ("B", 1), "uint8": ("B", 1),
              "short": ("h", 2), "int16": ("h", 2), "ushort": ("H", 2), "uint16": ("H", 2),
              "int": ("i", 4), "int32": ("i", 4), "uint": ("I", 4), "uint32": ("I", 4),
              "float": ("f", 4), "float32": ("f", 4), "double": ("d", 8), "float64": ("d", 8)}


# =============================================================================================
# Tools
# =============================================================================================

def _note(log_path, text: str) -> None:
    """Add a line of our own to the tool log, so one file tells the whole story of a run."""
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n[meshpost] {text}\n")
    except OSError:
        pass  # the log is a convenience; never fail a run over it


def _check_cancel(cancel_file) -> None:
    if cancel_file is not None and Path(cancel_file).exists():
        raise proc.Cancelled("cancelled by user")


def _find_checkpoint(run_dir: Path) -> Path:
    if not (run_dir / "config.json").exists():
        raise MeshError(
            f"{run_dir} is not a Spirula training run (no config.json). Pass the folder that "
            f"training wrote, the one holding config.json and step-*.ckpt.")
    found = sorted(p for p in run_dir.glob("step-*.ckpt") if (p / "splat.ply").exists())
    if not found:
        raise MeshError(
            f"{run_dir} has no step-*.ckpt/splat.ply. Training has not finished or was "
            f"interrupted before its first checkpoint; train the splat first.")
    return found[-1]


def _sparse_dir(dataset_dir: Path) -> Path:
    """The folder holding the dataset's cameras.bin and images.bin (sparse/0 or sparse)."""
    for candidate in (dataset_dir / "sparse" / "0", dataset_dir / "sparse"):
        if (candidate / "cameras.bin").is_file() and (candidate / "images.bin").is_file():
            return candidate
    raise MeshError(
        f"{dataset_dir} has no camera model (sparse/0/cameras.bin and images.bin). Pass the "
        f"dataset folder the splat was trained on.")


def _atlas_edge(texture_size: int) -> int:
    """Engines want power-of-two textures; the largest one that fits the requested size."""
    edge = 1
    while edge * 2 <= texture_size:
        edge *= 2
    return edge


def _tool_failure(e: Exception, what: str, log_path, detail: str = "") -> MeshError:
    if isinstance(e, proc.ProcTimeout):
        return MeshError(f"{what} did not finish in time and was stopped. The computer is "
                         f"probably busy with another job; wait for it and run the mesh step "
                         f"again. Log: {log_path}")
    if isinstance(e, proc.ProcStalled):
        return MeshError(f"{what} stopped making progress and was stopped. Close other "
                         f"programs and run the mesh step again. Log: {log_path}")
    return MeshError(f"{what} failed{detail}. Log: {log_path}")


def _colmap_cmd(colmap, command: str, *args) -> list:
    # COLMAP logs to stderr and, by default, also into files in the system temp folder.
    return [str(colmap), command, "--log_target", "stderr", *[str(a) for a in args]]


def _undistort(*, colmap, dataset_dir: Path, sparse: Path, out: Path, max_image_size: int,
               log_path, cancel_file, image_list: Optional[Path] = None) -> None:
    """`colmap image_undistorter`: the photos as pinhole images, the workspace the texturer reads."""
    if out.exists():
        shutil.rmtree(out)
    # The undistorter crashes (0xC0000409) when the output folder's parent does not exist.
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = _colmap_cmd(colmap, "image_undistorter", "--image_path", dataset_dir / "images",
                      "--input_path", sparse, "--output_path", out, "--output_type", "COLMAP",
                      "--max_image_size", max_image_size,
                      *(("--image_list_path", image_list) if image_list else ()))
    try:
        proc.run(cmd, log_path=log_path, timeout=COLMAP_TIMEOUT, stall_timeout=COLMAP_STALL,
                 cancel_file=cancel_file)
    except proc.ProcError as e:
        raise _tool_failure(e, "Preparing the photos for texturing (colmap image_undistorter)",
                            log_path, f" on {dataset_dir / 'images'}; check that the photos the "
                            f"splat was trained on are still there") from e
    if not (out / "sparse" / "cameras.bin").is_file():
        raise MeshError(f"colmap image_undistorter wrote no workspace in {out}. Log: {log_path}")


def photo_size_limit(cameras: dict, images: list) -> int:
    """`--max_image_size` for the undistorter: -1, or the long edge that keeps all photos together
    within PHOTO_PIXEL_BUDGET pixels (the texturer holds every photo in memory at once)."""
    total = 0
    longest = 0
    for _, _, _, _, cid in images:
        _, w, h, _ = cameras[cid]
        total += w * h
        longest = max(longest, w, h)
    if total <= PHOTO_PIXEL_BUDGET:
        return -1
    return max(256, int(longest * math.sqrt(PHOTO_PIXEL_BUDGET / total)))


_ATLAS_LINE = re.compile(r"Atlas size: (\d+) x (\d+)")
_TEXTURER_STAGES = (("Loading", 0.02), ("Selecting views", 0.25), ("faces assigned", 0.40),
                    ("Packing", 0.45), ("Baking texture", 0.50), ("color correction", 0.65),
                    ("Inpainting", 0.75), ("Writing texture atlas", 0.80), ("Writing textured", 0.97))


def _run_texturer(*, colmap, workspace: Path, mesh_ply: Path, out: Path, scale: float, probe: bool,
                  log_path, cancel_file, progress=None) -> dict:
    """One `colmap mesh_texturer` run. A probe stops as soon as the atlas size is known."""
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = _colmap_cmd(colmap, "mesh_texturer", "--workspace_path", workspace,
                      "--input_path", mesh_ply, "--output_path", out,
                      "--MeshTextureMapping.texture_scale_factor", repr(float(scale)),
                      *TEXTURER_OPTIONS)
    state = {"atlas": None, "regions": 0, "assigned": None, "at": -1}

    def on_line(line: str) -> None:
        m = _ATLAS_LINE.search(line)
        if m:
            state["atlas"] = (int(m.group(1)), int(m.group(2)))
        m = re.search(r"Found (\d+) regions", line)
        if m:
            state["regions"] = int(m.group(1))
        m = re.search(r"(\d+) / (\d+) faces assigned", line)
        if m:
            state["assigned"] = (int(m.group(1)), int(m.group(2)))
        if progress is not None:
            for j in range(state["at"] + 1, len(_TEXTURER_STAGES)):
                if _TEXTURER_STAGES[j][0] in line:
                    state["at"] = j
                    progress(_TEXTURER_STAGES[j][1], "生成贴图")
                    break

    try:
        proc.run(cmd, log_path=log_path, timeout=COLMAP_TIMEOUT, stall_timeout=COLMAP_STALL,
                 on_line=on_line, cancel_file=cancel_file,
                 done_marker=_ATLAS_LINE.pattern if probe else None, done_grace=0)
    except proc.ProcError as e:
        raise _tool_failure(e, "Texturing from the photos (colmap mesh_texturer)", log_path) from e
    if state["atlas"] is None:
        raise MeshError(f"colmap mesh_texturer did not lay out a texture for the mesh. "
                        f"Log: {log_path}")
    if not probe and not ((out / "mesh.ply").is_file() and (out / "texture.png").is_file()):
        raise MeshError(f"colmap mesh_texturer finished but wrote no textured mesh in {out}. "
                        f"Log: {log_path}")
    return state


def fit_texture_scale(probe: Callable[[float], tuple], edge: int) -> tuple:
    """The largest texture scale (atlas texels per photo pixel, at most TEXTURE_MAX_SCALE) whose
    atlas is no wider than `edge`. `probe(scale)` returns the atlas (width, height) COLMAP lays
    out at that scale: a power of two wide, as tall as the packed patches need.

    Returns (scale, (width, height), probes). The patches' area grows with the square of the
    scale plus a fixed gutter per patch, so each step aims at an atlas filled to ATLAS_FILL and
    the search keeps a bracket around the answer.
    """
    lo = None                 # (scale, size) known to fit
    hi = None                 # scale known not to fit
    scale = TEXTURE_MAX_SCALE
    probes = 0
    while probes < ATLAS_PROBES:
        w, h = probe(scale)
        probes += 1
        if w <= edge:
            if lo is None or scale > lo[0]:
                lo = (scale, (w, h))
            if scale >= TEXTURE_MAX_SCALE or h >= ATLAS_GOOD_FILL * edge:
                break
        else:
            hi = scale if hi is None else min(hi, scale)
        if lo is not None and hi is not None and hi / lo[0] < 1.0 + ATLAS_SCALE_TOLERANCE:
            break
        aim = scale * math.sqrt(ATLAS_FILL * edge * edge / max(1, w * h))
        low = lo[0] if lo is not None else 0.0
        high = hi if hi is not None else TEXTURE_MAX_SCALE * 1.000001
        if not low < aim < high:
            aim = math.sqrt(low * high) if low > 0.0 else high / 4.0
        if aim < MIN_TEXTURE_SCALE:
            break
        scale = min(TEXTURE_MAX_SCALE, aim)
    if lo is None:
        raise MeshError(
            f"The mesh has too many separate texture patches to fit a {edge} x {edge} texture. "
            f"Raise texture_size or lower visual_faces.")
    return lo[0], lo[1], probes


def build_occluders(*, dataset_dir: Path, cameras: dict, images: list, xyz: array.array,
                    log_path) -> tuple:
    """Triangles that hide what the masks cover from the camera that took each masked photo.

    `masks/<image path>.png` (black: a person or anything else to leave out) is matched to each
    photo. For every photo with masked pixels, the masked area is covered, as seen from that
    camera, by small triangles placed between the camera and the nearest surface it sees; the
    texturer's occlusion test then refuses that photo for everything behind them. Returns
    (triangles, number of masked photos).
    """
    masks_dir = dataset_dir / "masks"
    if not masks_dir.is_dir():
        return [], 0
    n = len(xyz) // 3
    if n == 0:
        return [], 0
    lo = [min(xyz[k::3]) for k in range(3)]
    hi = [max(xyz[k::3]) for k in range(3)]
    diag = math.sqrt(sum((b - a) ** 2 for a, b in zip(lo, hi))) or 1.0
    # A sample of the surface is enough to know how near the nearest of it comes to a camera.
    step = max(1, n // OCCLUDER_SAMPLE_VERTICES)
    xs = xyz[0::3][::step]
    ys = xyz[1::3][::step]
    zs = xyz[2::3][::step]
    tris = []
    masked = 0
    for _, name, R, t, cid in images:
        path = (masks_dir / name).with_suffix(".png")
        if not path.is_file():
            continue
        try:
            mw, mh, rows = read_mask_png(path)
        except (OSError, zlib.error) as e:
            raise MeshError(f"cannot read the mask {path}: {e}") from e
        cell = max(4, round(min(mw, mh) / MASK_CELLS_ACROSS))
        cells = masked_cells(mw, mh, rows, cell)
        if not cells:
            continue
        masked += 1
        model, w, h, params = cameras[cid]
        # nearest surface sample in front of the camera and inside its picture
        near = math.inf
        r0, r1, r2, r3, r4, r5, r6, r7, r8 = R
        for x, y, z in zip(xs, ys, zs):
            cz = r6 * x + r7 * y + r8 * z + t[2]
            if cz <= 0.0 or cz >= near:
                continue
            cx = (r0 * x + r1 * y + r2 * z + t[0]) / cz
            cy = (r3 * x + r4 * y + r5 * z + t[1]) / cz
            if abs(cx) < OCCLUDER_FOV_SLOPE and abs(cy) < OCCLUDER_FOV_SLOPE:
                u, v = distort(model, params, cx, cy)
                if 0.0 <= u <= w and 0.0 <= v <= h:
                    near = cz
        depth = max(OCCLUDER_MIN_DEPTH * diag, min(OCCLUDER_MAX_DEPTH * diag, 0.5 * near))
        tris.extend(occluder_triangles(cells, cell, mw, mh, cameras[cid], R, t, depth))
    return tris, masked


def mvs_images(images: list, limit: int) -> list:
    """Names of the photos dense matching runs on: all of them, or `limit` spread evenly over the
    photos in name order (consecutive video frames carry nearly the same information)."""
    names = sorted(name for _, name, _, _, _ in images)
    if len(names) <= limit:
        return names
    return [names[(2 * k + 1) * len(names) // (2 * limit)] for k in range(limit)]


def _write_list(path: Path, names) -> Path:
    path.write_text("".join(name + "\n" for name in names), encoding="utf-8")
    return path


def _mvs_workspace(*, colmap, dataset_dir: Path, sparse: Path, out: Path, names: list,
                   all_names: list, log_path, cancel_file) -> None:
    """An undistorted workspace holding only `names`, at MVS_IMAGE_SIZE, with its own stereo
    configuration (MVS_SOURCE_IMAGES source photos per photo, chosen by COLMAP)."""
    listing = _write_list(out.parent / (out.name + "_images.txt"), names)
    _undistort(colmap=colmap, dataset_dir=dataset_dir, sparse=sparse, out=out,
               max_image_size=MVS_IMAGE_SIZE, log_path=log_path, cancel_file=cancel_file,
               image_list=listing)
    dropped = sorted(set(all_names) - set(names))
    if dropped:
        # The undistorter writes only the listed photos but keeps every camera in the model;
        # dense matching would look for the others' files.
        kept = out / "sparse_kept"
        kept.mkdir()
        cmd = _colmap_cmd(colmap, "image_deleter", "--input_path", out / "sparse",
                          "--output_path", kept, "--image_names_path",
                          _write_list(out.parent / (out.name + "_dropped.txt"), dropped))
        try:
            proc.run(cmd, log_path=log_path, timeout=COLMAP_TIMEOUT, stall_timeout=COLMAP_STALL,
                     cancel_file=cancel_file)
        except proc.ProcError as e:
            raise _tool_failure(e, "Preparing the photos for dense matching (colmap image_deleter)",
                                log_path) from e
        shutil.rmtree(out / "sparse")
        kept.rename(out / "sparse")
    stereo = out / "stereo"
    stereo.mkdir(exist_ok=True)
    (stereo / "patch-match.cfg").write_text(
        "".join(f"{name}\n__auto__, {MVS_SOURCE_IMAGES}\n" for name in names), encoding="utf-8")
    _write_list(stereo / "fusion.cfg", names)


# The Delaunay mesher's phases and the share of the dense step that is over when each starts
# (measured on 128 photos: triangulation, ray tracer, integration and graph cut are the long ones).
_DELAUNAY_PHASES = (("Triangulating points", 0.79), ("Initializing ray tracer", 0.81),
                    ("Setting up optimization", 0.91), ("Running graph-cut", 0.93),
                    ("Extracting surface", 0.95), ("Writing surface mesh", 0.97))


def _dense_geometry(*, colmap, dataset_dir: Path, work_dir: Path, log_path, cancel_file,
                    progress) -> dict:
    """Dense matching, fusion and meshing with COLMAP. Returns {"path" (PLY, dataset frame),
    "images", "points"}."""
    sparse = _sparse_dir(dataset_dir)
    images = read_images_bin(sparse / "images.bin")
    all_names = [name for _, name, _, _, _ in images]
    names = mvs_images(images, MVS_MAX_IMAGES)
    workspace = work_dir / "dense"
    progress(0.0, "准备照片")
    _mvs_workspace(colmap=colmap, dataset_dir=dataset_dir, sparse=sparse, out=workspace,
                   names=names, all_names=all_names, log_path=log_path, cancel_file=cancel_file)

    # 1. Depth and normal maps per photo (GPU), photometric pass then geometric pass.
    _check_cancel(cancel_file)
    progress(0.04, "计算深度图")
    done = {"views": 0}
    total = 2 * len(names)

    def on_view(line: str) -> None:
        if "Processing view" in line:
            done["views"] += 1
            progress(0.04 + 0.66 * min(1.0, done["views"] / total), "计算深度图")

    cmd = _colmap_cmd(colmap, "patch_match_stereo", "--workspace_path", workspace,
                      "--workspace_format", "COLMAP", *PATCH_MATCH_OPTIONS)
    try:
        proc.run(cmd, log_path=log_path, timeout=MVS_TIMEOUT, stall_timeout=COLMAP_STALL,
                 on_line=on_view, cancel_file=cancel_file)
    except proc.ProcError as e:
        raise _tool_failure(
            e, "Dense matching (colmap patch_match_stereo)", log_path,
            " (it needs an NVIDIA GPU with free memory; close other GPU programs and run the "
            "mesh step again)") from e
    maps = list((workspace / "stereo" / "depth_maps").rglob("*.geometric.bin"))
    if len(maps) < len(names):
        raise MeshError(f"Dense matching produced depth maps for {len(maps)} of {len(names)} "
                        f"photos. Log: {log_path}")

    # 2. Fusion of the depth maps into points with visibility (CPU).
    _check_cancel(cancel_file)
    progress(0.70, "融合点云")
    fused = workspace / "fused.ply"
    cmd = _colmap_cmd(colmap, "stereo_fusion", "--workspace_path", workspace,
                      "--workspace_format", "COLMAP", "--input_type", "geometric",
                      "--output_path", fused, *FUSION_OPTIONS)
    state = {"points": 0}

    def on_fused(line: str) -> None:
        m = re.search(r"Number of fused points: (\d+)", line)
        if m:
            state["points"] = int(m.group(1))
        m = re.search(r"Fusing image \[(\d+)/(\d+)\]", line)
        if m:
            progress(0.70 + 0.08 * int(m.group(1)) / max(1, int(m.group(2))), "融合点云")

    try:
        proc.run(cmd, log_path=log_path, timeout=COLMAP_TIMEOUT, stall_timeout=COLMAP_STALL,
                 on_line=on_fused, cancel_file=cancel_file)
    except proc.ProcError as e:
        raise _tool_failure(e, "Fusing the depth maps (colmap stereo_fusion)", log_path) from e
    if not fused.is_file() or not (workspace / "fused.ply.vis").is_file():
        raise MeshError(f"colmap stereo_fusion wrote no points. Log: {log_path}")

    # 3. Surface from the points and the rays that saw them (CPU).
    _check_cancel(cancel_file)
    progress(0.78, "重建表面")
    meshed = work_dir / "dense_mesh.ply"
    cmd = _colmap_cmd(colmap, "delaunay_mesher", "--input_path", workspace,
                      "--input_type", "dense", "--output_path", meshed, *DELAUNAY_OPTIONS)

    def on_mesh(line: str) -> None:
        m = re.search(r"Integrating image \[(\d+)/(\d+)\]", line)
        if m:
            progress(0.82 + 0.08 * int(m.group(1)) / max(1, int(m.group(2))), "重建表面")
            return
        for phase, at in _DELAUNAY_PHASES:
            if phase in line:
                progress(at, "重建表面")

    try:
        proc.run(cmd, log_path=log_path, timeout=COLMAP_TIMEOUT, stall_timeout=COLMAP_STALL,
                 on_line=on_mesh, cancel_file=cancel_file)
    except proc.ProcError as e:
        raise _tool_failure(e, "Building the surface (colmap delaunay_mesher)", log_path) from e
    if not meshed.is_file():
        raise MeshError(f"colmap delaunay_mesher wrote no mesh. Log: {log_path}")
    progress(1.0, "重建表面")
    return {"path": meshed, "images": len(names), "points": state["points"],
            "sparse": workspace / "sparse"}


def face_pixel_sizes(mesh: Mesh, cameras: dict, images: list, settle: float = 0.0):
    """How large each face appears, and how far away it is, in the photo where it is smallest.

    Returns (sizes, depths). sizes[k]: face k's longest edge in pixels, in the photo where that
    is shortest, among the photos (pinhole cameras) that have all three corners in front and the
    face's centre in the picture; infinity where no photo sees the face. depths[k]: the face's
    distance in that photo over the median distance of the surface the photo shows (estimated
    from a sample of the vertices); 0 where no photo sees the face.

    With `settle` > 0 a face is no longer looked at once one photo shows it at most that large
    (its values then come from that photo, not necessarily the one where it is smallest): most
    faces are settled by the first few photos, which makes the pass many times faster.
    """
    n = mesh.face_count
    X = mesh.xyz[0::3]
    Y = mesh.xyz[1::3]
    Z = mesh.xyz[2::3]
    A = mesh.tris[0::3]
    B = mesh.tris[1::3]
    C = mesh.tris[2::3]
    best = [math.inf] * n
    depths = [0.0] * n
    limit = settle * settle
    pending = list(range(n))
    step = max(1, mesh.vertex_count // SHARD_DEPTH_SAMPLES)
    for _, _, R, t, cid in images:
        if not pending:
            break
        model, w, h, p = cameras[cid]
        if model != 1:
            raise MeshError(f"the dense workspace has a camera of model {model}, expected PINHOLE")
        fx, fy, cx, cy = p
        r0, r1, r2, r3, r4, r5, r6, r7, r8 = R
        t0, t1, t2 = t
        shown = []
        for x, y, z in zip(X[::step], Y[::step], Z[::step]):
            d = r6 * x + r7 * y + r8 * z + t2
            if d > 0.0:
                u = fx * (r0 * x + r1 * y + r2 * z + t0) / d + cx
                v = fy * (r3 * x + r4 * y + r5 * z + t1) / d + cy
                if 0.0 <= u <= w and 0.0 <= v <= h:
                    shown.append(d)
        shown.sort()
        median = shown[len(shown) // 2] if shown else 0.0
        seen = {}

        def project(i: int):
            q = seen.get(i)
            if q is None:
                x = X[i]
                y = Y[i]
                z = Z[i]
                d = r6 * x + r7 * y + r8 * z + t2
                q = (fx * (r0 * x + r1 * y + r2 * z + t0) / d + cx,
                     fy * (r3 * x + r4 * y + r5 * z + t1) / d + cy, d) if d > 0.0 else None
                seen[i] = q
            return q

        still = []
        keep = still.append
        for k in pending:
            pi = project(A[k])
            pj = project(B[k])
            pl = project(C[k])
            if pi is not None and pj is not None and pl is not None:
                ui, vi, di = pi
                uj, vj, dj = pj
                ul, vl, dl = pl
                if 0.0 <= (ui + uj + ul) / 3.0 <= w and 0.0 <= (vi + vj + vl) / 3.0 <= h:
                    e = max((ui - uj) ** 2 + (vi - vj) ** 2, (uj - ul) ** 2 + (vj - vl) ** 2,
                            (ul - ui) ** 2 + (vl - vi) ** 2)
                    if e < best[k]:
                        best[k] = e
                        depths[k] = (di + dj + dl) / (3.0 * median) if median > 0.0 else math.inf
                    if e <= limit:
                        continue
            keep(k)
        pending = still
    return [math.sqrt(e) for e in best], depths


def drop_shards(mesh: Mesh, cameras: dict, images: list, max_pixels: float, min_depth: float):
    """Remove the shards the Delaunay mesher's graph cut leaves where there are no points.

    Fused depth maps put a point every pixel or two, so a face of the dense surface measures a
    few pixels in the photos that saw it. The graph cut also closes off regions without points:
    the sky and the far background seen past an object's edge, but also plain walls and
    ceilings, which are real. The shards lie far behind what the photos show; the walls do not.
    A face goes when it is larger than `max_pixels` (longest edge) in every photo that sees it
    AND, in the photo where it is smallest, further away than `min_depth` times the median
    distance of the surface that photo shows. Faces no photo sees are kept.
    Returns (mesh, number of faces removed)."""
    sizes, depths = face_pixel_sizes(mesh, cameras, images, settle=max_pixels)
    keep = [not (max_pixels < s < math.inf and d > min_depth) for s, d in zip(sizes, depths)]
    removed = keep.count(False)
    return (select_faces(mesh, keep) if removed else mesh), removed


def simplify(*, colmap, mesh: Mesh, target_faces: int, work_dir: Path, name: str, log_path,
             cancel_file) -> Mesh:
    """`colmap mesh_simplifier` down to at most `target_faces` (quadric edge collapse with
    optimal vertex placement). Returns the mesh unchanged when it is already within budget."""
    path_in = work_dir / f"{name}_in.ply"
    path_out = work_dir / f"{name}_out.ply"
    for _ in range(SIMPLIFY_PASSES):
        if mesh.face_count <= target_faces:
            return mesh
        write_ply(path_in, mesh.xyz, mesh.tris)
        ratio = target_faces / mesh.face_count
        cmd = _colmap_cmd(colmap, "mesh_simplifier", "--input_path", path_in,
                          "--output_path", path_out,
                          "--MeshSimplification.target_face_ratio", repr(ratio))
        try:
            proc.run(cmd, log_path=log_path, timeout=COLMAP_TIMEOUT, stall_timeout=COLMAP_STALL,
                     cancel_file=cancel_file)
        except proc.ProcError as e:
            raise _tool_failure(e, "Simplifying the mesh (colmap mesh_simplifier)", log_path) from e
        xyz, tris, _ = read_ply(path_out)
        mesh = drop_degenerate_faces(Mesh(xyz, tris))
    if mesh.face_count > target_faces:
        raise MeshError(f"The mesh could not be simplified to {target_faces} faces (it still "
                        f"has {mesh.face_count}). Raise the face budget. Log: {log_path}")
    return mesh


def texture_mesh(*, colmap, dataset_dir: Path, work_dir: Path, mesh: Mesh, world_from_train,
                 edge: int, log_path, cancel_file, progress) -> dict:
    """Texture `mesh` (training frame) from the dataset's photos with `colmap mesh_texturer`.

    Returns {"mesh": textured Mesh with uv (glTF convention), "image": PNG bytes (square),
    "edge", "scale", "probes", "regions", "masked_photos", "occluders", "faces_flat_coloured",
    "faces_unseen_dropped", "photo_size"}. Positions are `mesh`'s own numbers, never COLMAP's.
    """
    sparse = _sparse_dir(dataset_dir)
    cameras = read_cameras_bin(sparse / "cameras.bin")
    images = read_images_bin(sparse / "images.bin")
    missing = [cid for *_, cid in images if cid not in cameras]
    if missing or not images:
        raise MeshError(f"the camera model in {sparse} is incomplete (no images, or an image "
                        f"refers to a camera that does not exist)")

    progress(0.0, "准备照片")
    limit = photo_size_limit(cameras, images)
    workspace = work_dir / "undistorted"
    _undistort(colmap=colmap, dataset_dir=dataset_dir, sparse=sparse, out=workspace,
               max_image_size=limit, log_path=log_path, cancel_file=cancel_file)

    # COLMAP works in the dataset frame; the mesh is in the training frame.
    xyz = mesh.xyz if world_from_train is None else transform_points(mesh.xyz, world_from_train)
    _check_cancel(cancel_file)
    progress(0.08, "处理遮罩")
    occluders, masked = build_occluders(dataset_dir=dataset_dir, cameras=cameras, images=images,
                                        xyz=xyz, log_path=log_path)
    mesh_ply = work_dir / "texture_in.ply"
    write_ply(mesh_ply, xyz, mesh.tris, occluders)

    runs = []

    def probe(scale: float) -> tuple:
        _check_cancel(cancel_file)
        progress(0.12 + 0.03 * min(len(runs), 6), "估算贴图尺寸")
        state = _run_texturer(colmap=colmap, workspace=workspace, mesh_ply=mesh_ply,
                              out=work_dir / "texture_probe", scale=scale, probe=True,
                              log_path=log_path, cancel_file=cancel_file)
        runs.append((scale, state["atlas"]))
        return state["atlas"]

    scale, planned, probes = fit_texture_scale(probe, edge)
    _note(log_path, "texture scale probes: " + ", ".join(f"{s:.4f} -> {w}x{h}" for s, (w, h) in runs)
          + f"; chosen {scale:.4f}")
    _check_cancel(cancel_file)
    out = work_dir / "texture"
    state = _run_texturer(colmap=colmap, workspace=workspace, mesh_ply=mesh_ply, out=out,
                          scale=scale, probe=False, log_path=log_path, cancel_file=cancel_file,
                          progress=lambda f, m: progress(0.30 + 0.62 * f, m))
    atlas_w, atlas_h = state["atlas"]
    if atlas_w > edge or atlas_h > atlas_w:
        raise MeshError(f"colmap mesh_texturer laid out a {atlas_w} x {atlas_h} texture where "
                        f"{planned[0]} x {planned[1]} was planned. Log: {log_path}")
    _check_cancel(cancel_file)
    progress(0.93, "整理贴图")
    _, tris, face_uv = read_ply(out / "mesh.ply")
    n_faces = mesh.face_count
    if face_uv is None or len(tris) != len(mesh.tris) + 3 * len(occluders) \
            or tris[:len(mesh.tris)] != mesh.tris:
        raise MeshError(f"colmap mesh_texturer returned a different mesh from the one it was "
                        f"given ({out / 'mesh.ply'}). Log: {log_path}")
    png = (out / "texture.png").read_bytes()
    mime, png_w, png_h = image_size(png)
    if (mime, png_w, png_h) != ("image/png", atlas_w, atlas_h):
        raise MeshError(f"the texture COLMAP wrote ({png_w} x {png_h}) is not the atlas it "
                        f"reported ({atlas_w} x {atlas_h}). Log: {log_path}")
    xyz_out, tris_out, uv_out, fill = assemble_textured(
        mesh.xyz, mesh.tris, face_uv[:6 * n_faces], atlas_w, atlas_h, atlas_w)
    if not tris_out:
        raise MeshError(f"No camera saw any part of the mesh from the front, so nothing could be "
                        f"textured. Log: {log_path}")
    image = pad_png_rows(png, atlas_w)
    return {"mesh": Mesh(xyz_out, tris_out, uv_out), "image": image, "edge": atlas_w,
            "scale": scale, "probes": probes, "regions": state["regions"],
            "masked_photos": masked, "occluders": len(occluders),
            "photo_size": limit, **fill}


# =============================================================================================
# Entry point
# =============================================================================================

def build_meshes(*, run_dir, dataset_dir, out_dir, work_dir, tools, visual_faces=300_000,
                 collision_faces=20_000, texture_size=4096, log_path, cancel_file=None,
                 on_progress=None) -> dict:
    """Write `out_dir/mesh.glb` and `out_dir/collision.glb` for a Spirula training run.

    run_dir       Spirula run folder (config.json, scene_transform.json, step-*.ckpt/): the
                  meshes are written in the frame of its splat.
    dataset_dir   the dataset the run was trained on: images/, sparse/0 (or sparse/) with any
                  COLMAP camera model, and optionally masks/ (black = leave out of the texture).
    work_dir      scratch folder for intermediate files (several GB on large scans); safe to
                  delete afterwards.
    tools         {"colmap": path to colmap.exe (the CUDA build)}; other entries are ignored.
    visual_faces, collision_faces
                  upper bounds on the triangle counts of the two meshes.
    texture_size  upper bound on the texture's edge, in pixels. The texture is square, with a
                  power-of-two edge.
    log_path      the tools' output and a summary of each step are appended here.
    cancel_file   if this path appears the work stops and proc.Cancelled is raised.
    on_progress   called as on_progress(fraction 0..1, message), never going backwards.

    Raises MeshError with an actionable message when a step fails.
    """
    started = time.monotonic()
    run_dir = Path(run_dir).resolve()
    dataset_dir = Path(dataset_dir).resolve()
    out_dir = Path(out_dir).resolve()
    work_dir = Path(work_dir).resolve()
    log_path = Path(log_path).resolve()
    if "colmap" not in tools or not Path(tools["colmap"]).is_file():
        raise MeshError(f"tools['colmap'] must be the path to colmap.exe; got "
                        f"{tools.get('colmap')!r}. Run setup to install the tools.")
    colmap = Path(tools["colmap"]).resolve()
    for name, value in (("visual_faces", visual_faces), ("collision_faces", collision_faces)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 100:
            raise MeshError(f"{name} must be a whole number of at least 100, got {value!r}")
    if not isinstance(texture_size, int) or not 256 <= texture_size <= 16384:
        raise MeshError(f"texture_size must be between 256 and 16384, got {texture_size!r}")
    _find_checkpoint(run_dir)
    if not (dataset_dir / "images").is_dir() or not (dataset_dir / "sparse").is_dir():
        raise MeshError(
            f"{dataset_dir} is not the dataset of this run: it needs an images/ folder and a "
            f"sparse/ folder. Pass the dataset folder the splat was trained on.")
    _sparse_dir(dataset_dir)
    frames = read_scene_transform(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if work_dir.exists():
        shutil.rmtree(work_dir)   # leftovers of an earlier run must not be mistaken for results
    work_dir.mkdir(parents=True)
    mesh_path = out_dir / "mesh.glb"
    collision_path = out_dir / "collision.glb"
    for old in (mesh_path, collision_path):
        if old.exists():
            old.unlink()   # a failed run must not leave last time's files looking like its result

    last = [0.0]

    def report(lo: float, hi: float) -> Callable[[float, str], None]:
        def inner(fraction: float, message: str) -> None:
            last[0] = value = max(last[0], lo + (hi - lo) * max(0.0, min(1.0, fraction)))
            if on_progress is not None:
                try:
                    on_progress(value, message)
                except Exception:
                    pass  # a broken progress display must not stop the meshing
        return inner

    timings = {}

    def timed(key: str, since: float) -> float:
        timings[key] = round(time.monotonic() - since, 2)
        return time.monotonic()

    # 1. Dense geometry from the photos (GPU, then CPU), in the dataset frame.
    _check_cancel(cancel_file)
    t = time.monotonic()
    dense = _dense_geometry(colmap=colmap, dataset_dir=dataset_dir, work_dir=work_dir,
                            log_path=log_path, cancel_file=cancel_file, progress=report(0.0, 0.70))
    t = timed("dense", t)

    # 2. Into the splat's frame, without the fragments, down to the visual budget.
    _check_cancel(cancel_file)
    report(0.70, 0.70)(0.0, "清除零散碎片")
    xyz, tris, _ = read_ply(dense["path"])
    raw = drop_degenerate_faces(Mesh(xyz, tris), zero_area=True)
    del xyz, tris
    raw_faces = raw.face_count
    if raw_faces == 0:
        raise MeshError(f"Dense reconstruction found no surface in the photos of {dataset_dir}. "
                        f"Check that the photos overlap and are sharp. Log: {log_path}")
    # Shards are judged in the dataset frame, with the cameras dense matching used.
    raw, shards = drop_shards(raw, read_cameras_bin(dense["sparse"] / "cameras.bin"),
                              read_images_bin(dense["sparse"] / "images.bin"), SHARD_MAX_PIXELS,
                              SHARD_MIN_DEPTH)
    if raw.face_count == 0:
        raise MeshError(f"Dense reconstruction found no surface in the photos of {dataset_dir} "
                        f"that the photos resolve. Log: {log_path}")
    _check_cancel(cancel_file)
    if not frames["identity"]:
        raw = Mesh(transform_points(raw.xyz, frames["train_from_world"]), raw.tris)
    surface, cut = keep_large_components(raw, VISUAL_MIN_AREA_FRACTION)
    del raw
    _check_cancel(cancel_file)
    report(0.72, 0.72)(0.0, "简化网格")
    surface = simplify(colmap=colmap, mesh=surface, target_faces=visual_faces, work_dir=work_dir,
                       name="visual", log_path=log_path, cancel_file=cancel_file)
    # A collapse can cut a sliver loose; the fragment rule applies to those too.
    surface, _ = keep_large_components(surface, VISUAL_MIN_AREA_FRACTION)
    surface = drop_degenerate_faces(surface, zero_area=True)
    t = timed("simplify", t)

    # 3. Texture from the photos.
    _check_cancel(cancel_file)
    textured = texture_mesh(colmap=colmap, dataset_dir=dataset_dir, work_dir=work_dir,
                            mesh=surface, world_from_train=None if frames["identity"]
                            else frames["world_from_train"], edge=_atlas_edge(texture_size),
                            log_path=log_path, cancel_file=cancel_file,
                            progress=report(0.75, 0.92))
    t = timed("texture", t)

    _check_cancel(cancel_file)
    report(0.92, 0.92)(0.0, "写出带贴图网格")
    visual = textured["mesh"]
    frame_note = {"frame": "spirula-train (same numbers as splat.ply)", "source": "scan3d"}
    write_glb(mesh_path, positions=visual.xyz, indices=array.array("I", visual.tris),
              normals=seamless_normals(visual), uvs=visual.uv, image=textured["image"],
              name="mesh", extras=frame_note)
    del visual
    textured["image"] = None
    t = timed("visual_glb", t)

    # 4. Collision mesh: the same surface, larger pieces only, far fewer faces.
    _check_cancel(cancel_file)
    report(0.94, 0.94)(0.0, "简化碰撞网格")
    collision, first_cut = keep_large_components(weld(surface), COLLISION_MIN_AREA_FRACTION)
    del surface
    collision = drop_degenerate_faces(collision, zero_area=True)
    collision = simplify(colmap=colmap, mesh=collision, target_faces=collision_faces,
                         work_dir=work_dir, name="collision", log_path=log_path,
                         cancel_file=cancel_file)
    collision, _ = keep_large_components(collision, COLLISION_MIN_AREA_FRACTION)
    collision = drop_degenerate_faces(collision, zero_area=True)
    collision_components = len(face_components(collision)[2])
    t = timed("collision_simplify", t)

    _check_cancel(cancel_file)
    report(0.96, 0.96)(0.0, "写出碰撞网格")
    write_glb(collision_path, positions=collision.xyz, indices=array.array("I", collision.tris),
              normals=vertex_normals(collision), name="collision", extras=frame_note)
    del collision
    t = timed("collision_glb", t)

    # 5. Check both files the way a loader would read them.
    report(0.97, 0.97)(0.0, "检查文件")
    mesh_info = validate_glb(mesh_path)
    collision_info = validate_glb(collision_path)
    if mesh_info["faces"] > visual_faces or collision_info["faces"] > collision_faces:
        raise MeshError(
            f"The meshes could not be brought down to their budgets (visual "
            f"{mesh_info['faces']} of {visual_faces}, collision {collision_info['faces']} of "
            f"{collision_faces}). Raise the face budgets and run the mesh step again.")
    timed("validate", t)
    _note(log_path, f"dense reconstruction from {dense['images']} photos: {dense['points']} "
                    f"points, {raw_faces} faces, {shards} of them far shards wider than "
                    f"{SHARD_MAX_PIXELS:g} px in every photo; {cut['components_before']} "
                    f"components, kept "
                    f"{cut['components_after']} ({cut['area_kept_fraction']:.1%} of the area); "
                    f"scene transform {'identity' if frames['identity'] else 'not identity'}")
    _note(log_path, f"texture: scale {textured['scale']:.4f} after {textured['probes']} probes, "
                    f"{textured['regions']} patches, {textured['edge']} px, "
                    f"{textured['masked_photos']} masked photos ({textured['occluders']} "
                    f"occluder triangles), {textured['faces_flat_coloured']} faces flat-coloured, "
                    f"{textured['faces_unseen_dropped']} unseen faces dropped")
    _note(log_path, f"wrote {mesh_path.name}: {mesh_info['faces']} faces, "
                    f"{mesh_info['vertices']} vertices, {mesh_info['bytes']} bytes; "
                    f"{collision_path.name}: {collision_info['faces']} faces, "
                    f"{collision_info['vertices']} vertices, {collision_info['bytes']} bytes; "
                    f"{time.monotonic() - started:.1f} s")
    report(1.0, 1.0)(1.0, "网格完成")
    return {
        "mesh": str(mesh_path),
        "collision": str(collision_path),
        "mesh_faces": mesh_info["faces"],
        "mesh_vertices": mesh_info["vertices"],
        "mesh_bytes": mesh_info["bytes"],
        "collision_faces": collision_info["faces"],
        "collision_vertices": collision_info["vertices"],
        "collision_bytes": collision_info["bytes"],
        "texture_size": mesh_info["texture_width"],
        "texture_width": mesh_info["texture_width"],
        "texture_height": mesh_info["texture_height"],
        "texture_format": "png",
        "texture_scale": round(textured["scale"], 4),
        "texture_patches": textured["regions"],
        "faces_flat_coloured": textured["faces_flat_coloured"],
        "faces_unseen_dropped": textured["faces_unseen_dropped"],
        "masked_photos": textured["masked_photos"],
        "geometry": "colmap-dense",
        "dense_photos": dense["images"],
        "dense_points": dense["points"],
        "raw_faces": raw_faces,
        "faces_shards_removed": shards,
        "raw_components": cut["components_before"],
        "mesh_components_kept": cut["components_after"],
        "mesh_area_kept": round(cut["area_kept_fraction"], 4),
        "collision_components_kept": collision_components,
        "collision_area_kept": round(first_cut["area_kept_fraction"], 4),
        "frame": "spirula-train",
        "scene_transform_is_identity": frames["identity"],
        "bbox_min": mesh_info["bbox_min"],
        "bbox_max": mesh_info["bbox_max"],
        "timings": timings,
        "seconds": round(time.monotonic() - started, 1),
    }
