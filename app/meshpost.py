"""Game-engine meshes from a finished Spirula training run.

`build_meshes` writes two files next to each other:

    mesh.glb        textured visual mesh, at most `visual_faces` triangles
    collision.glb   untextured mesh for physics and shadow receiving, at most `collision_faces`

Only two things run: `spirula.exe` and this file, which uses the standard library alone (it
runs on the embeddable Python). How the meshes are made; the numbers are from the
south-building scene (runs/sb-std: 1 M Gaussians, 128 photos) on an RTX 5070 Ti Laptop:

1. `spirula mesh --format glb --color texture --merge-factor 1 --chart-angle-deg 10` extracts
   the surface from the splat, carved by the dataset cameras, and textures it in the same run:
   2.06 M faces with a 4096 atlas of 133 k charts in 142 s. Spirula can only texture the mesh
   it has just extracted, so everything after this step has to keep that atlas.
2. Here, in plain Python: the GLB is read back, connected components are found (vertices are
   welded by position first: Spirula splits them along texture seams) and the small ones are
   dropped, by surface area relative to the largest component (4 s).
3. `collapse_to` brings the visual mesh down to its budget by quadric edge collapse without
   touching the atlas: a face keeps pointing into the chart it came from. A corner that moves
   takes the coordinates its new vertex has in that chart; where the vertex is not part of
   the chart, the chart's own mapping is carried on to it, as far as the gutter around the
   chart goes; beyond that the corner stays where it was in the atlas and the face shows its
   patch pulled out of shape (2.04 M to 300 k faces: 56 s, 12 % of the faces stretched like
   that, 86 % of the atlas area still shown).
4. The collision mesh is the visual mesh without its seams and its smaller pieces, collapsed
   further by the same routine (300 k to 20 k faces: 6 s), then cleaned of the slivers and
   zero-area faces a collapse can leave behind.
5. Both files are packed as binary glTF 2.0 and checked by `validate_glb`.

That is 3.5 minutes for the default budgets, of which 2.4 are Spirula's; this module peaks at
2.3 GB of memory for 2 M extracted faces.

Why it is done this way (measured; "kept" is the share of the well-sampled surface of the
earlier COLMAP-based output that a mesh still covers in six of the training views):

* Spirula cannot be made to extract at a face budget, and trying costs surface.
  `--merge-factor` levels off (2.06 M faces at 1, 0.89 M at 1.5, 0.70 M at 2, 0.63 M at 4)
  and, because its merge runs before the pass that drops what no camera saw, thin walls are
  fused with their unseen back and go with it: kept 98.8 % at factor 1, 95.1 % at 1.5,
  91.3 % at 2, 88.7 % at 4. `--cull-unseen 0` is no way round it (the result is a closed
  shell around the cameras), nor are `--merge-max-flip-deg` (420 k faces at factor 4 with the
  guard off, a visibly worse surface), `--grid-cell-factor` and `--max-grid-res` (an
  acceleration grid only) or `--bisection-iters 0` (the surface stays where the first guess
  put it). So the extraction runs at factor 1 and the way down is made here, where it costs
  0.5 %: kept 98.3 % at 300 k faces, 98.1 % at 150 k.
* `--chart-angle-deg` is worth changing. Spirula's default of 60 gives 150 k charts at factor
  1, a texturing step of 128 s and 0.058 of the texels the photos could fill; 30 gives 140 k,
  104 s, 0.067; 10 gives 133 k, 60 s, 0.088, with no difference to be seen in the renders
  (at factor 1.5: 85 k / 77 k / 69 k / 65 k charts at 60 / 30 / 15 / 8 degrees).
  `--tex-gutter-px 1` instead of 4 would add another 11 % of texels; it is left alone
  because the gutter is what step 3 reaches into.
* A collapse that keeps every face strictly inside its chart is not possible: the charts are
  a dozen faces each and nearly every vertex lies on a chart border. Giving a face that
  leaves its chart one flat colour, which is what this module did first, showed as a mosaic
  of plain triangles over a third of the faces (2 M to 300 k); carrying the chart on into
  its gutter and stretching the patch beyond that leaves 12 % of the faces affected, and a
  render that is hard to tell from the undecimated mesh.
* The texture is as sharp as Spirula bakes it, which is visibly softer than a projection of
  the photos (the COLMAP texturer this module used before), and in the renders no sharper at
  8192 than at 4096, although the larger atlas holds four times the texels.
* Vertex colours instead of a texture were not needed.

Coordinate frame
----------------
Both GLBs are written in the frame of the splat file (`step-*.ckpt/splat.ply`), which is
Spirula's training frame, with the numbers untouched: no node transform, no axis swap, no unit
change. A vertex (x, y, z) in either GLB is the same point as a Gaussian centre (x, y, z) in
splat.ply, so the splat and the meshes overlay exactly in any engine whose glTF importer and
splat loader agree on how to treat raw coordinates. glTF nominally means "+Y up, metres"; a
photo reconstruction has no metric scale and no gravity direction, so that reading does not hold
here and whatever rotation and scale the engine applies to the splat must be applied to the two
meshes as well, as one rigid group.

`spirula mesh` writes its GLB in the frame of the splat.ply it reads, without a node transform
(measured: the vertices of mesh.glb lie a median 0.06 % of the scene size from the nearest
Gaussian centre, and 0.4 % once the mesh is shifted by 1 %; and when a splat was moved by a
known scale, rotation and shift, its mesh moved with it, to a median 0.04 % of the scene
size). Nothing here moves a vertex: both meshes are made by removing vertices
from what Spirula wrote, so every coordinate in either file is one Spirula wrote, bit for bit.
A GLB from Spirula that carried a node transform would break this, so `read_glb` refuses one.
The dataset frame (where the cameras are) can differ from the training frame by
`scene_transform.json`; Spirula deals with that itself, and nothing here needs it.
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
import struct
import sys
import time
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
# Spirula's own floater filter, in faces. The help text gives 100 as the default, but the flag
# has to be passed to get it: without it the same scene came out with 11,782 components, with
# it 937. It costs no time and saves the Python pass most of its work.
SPIRULA_FLOATER_MIN_FACES = 100

# `--merge-factor` for the extraction: how far Spirula merges short edges. 1 is its default and
# the only value that keeps the whole surface: its merge runs before it throws away what no
# camera saw, and takes real surface along (see the module docstring). The way down to a face
# budget is done here, by `collapse_to`, which loses next to nothing.
MERGE_FACTOR = 1.0
# The one exception, for time and memory: a splat with more Gaussians than this is extracted
# at the coarser factor. Factor 1 gives about 2.1 faces per Gaussian, and every million faces
# costs this module about 1.1 GB and half a minute.
LARGE_SPLAT_GAUSSIANS = 2_500_000
LARGE_SPLAT_MERGE_FACTOR = 1.5
# `--chart-angle-deg`: Spirula's default is 60. Smaller is better on every count that was
# measured (see the module docstring): fewer charts, more texels, a much shorter run.
CHART_ANGLE_DEG = 10
# How Spirula stores the atlas in its GLB: "png" (lossless) or "jpg" (JPEG, quality 95).
TEXTURE_ENCODING = "png"

# Collapsing an edge may not turn a face by more than this (cosine), and may not eat into an
# open border more cheaply than this weight allows (relative to a face of the same size).
COLLAPSE_MAX_TURN_COS = 0.2
COLLAPSE_BORDER_WEIGHT = 1.0
# Price of pulling a face's patch of the atlas out of shape (a corner left where it was in the
# atlas while it moves in space): as if a face of its size had been moved by
# sqrt(TEXTURE_LOSS_WEIGHT) times its own size; less, in proportion, for a face with fewer than
# TEXTURE_LOSS_TEXELS texels, which has little to lose.
TEXTURE_LOSS_WEIGHT = 1.0
TEXTURE_LOSS_TEXELS = 8.0
# How far outside its chart a texture coordinate may be put, in texels. Spirula leaves a gutter
# of 4 texels between charts and fills it with the colours of the nearest chart edge, so half
# of that belongs to each chart.
TEXTURE_REACH_TEXELS = 2.0

SPIRULA_TIMEOUT = 2400
# The longest silence of a healthy run is the chart parameterisation: 50 s for 2 M faces.
SPIRULA_STALL = 900

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

    Spirula duplicates vertices where it splits non-manifold fans; without welding, the two
    sides of such a split would be counted as separate components.
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
# Simplification
# =============================================================================================

def collapse_to(mesh: Mesh, target_faces: int, *, texels: float = 0.0,
                tick: Optional[Callable[[int], None]] = None):
    """Bring a mesh down to at most `target_faces` triangles by quadric-error edge collapse.

    Returns (mesh, stats). Only half-edge collapses are made: a vertex is merged into one of
    its neighbours, cheapest first by the summed squared distance to the planes of the faces the
    vertex has stood for (Garland and Heckbert's measure). No vertex is ever moved, so every
    vertex of the result is a vertex of the input at its original position.

    A scan is not a clean manifold: the extracted surface is full of handles and tunnels a few
    faces across, and a collapse that respected the topology stalls on them (measured: a
    strictly topology-preserving version stopped at 528 k of 1.75 M faces, with a third of the
    remaining vertices unable to move). So a collapse may pinch such a handle shut. When it
    folds two faces onto each other, both are removed; a collapse that would turn a face over
    is refused, and a vertex on an open border only moves along the border.

    Vertices at the same position are treated as one, so the two sides of a texture seam never
    come apart. When the mesh has texture coordinates (`mesh.uv`, with `texels` the number of
    texels in the atlas) the atlas is kept as it is and every face keeps pointing into the
    chart it came from. A corner that moves takes the coordinates its new vertex has in that
    chart. Where the new vertex is not part of the chart, it is given coordinates there: the
    place the faces' own mapping puts it when carried on past the edge of the chart, which is
    in the gutter Spirula fills with the colours of the chart's edge. Such a coordinate may
    lie at most TEXTURE_REACH_TEXELS outside the chart. What does not fit is cut short, or, if
    the triangle would turn over in the atlas, the corner keeps the coordinate it had: the
    face then shows its patch of the texture pulled out of shape, which costs detail in
    proportion to the texels the face had and is priced into the collapse accordingly
    (TEXTURE_LOSS_WEIGHT), so collapses inside a chart and along a seam go first. No face is
    ever reduced to a single colour.

    stats: faces_before, faces_after, collapses, faces_stretched (faces showing a patch that
    was cut short or left behind) and atlas_shown (area of the atlas the faces still show,
    relative to the start).

    `tick(faces_left)` is called every few thousand collapses and may raise to stop the work.
    """
    if target_faces < 4:
        raise MeshError(f"a face budget must be at least 4, got {target_faces}")
    xyz = mesh.xyz
    n = mesh.vertex_count
    textured = mesh.uv is not None
    same = _weld_map(mesh)
    # The quadrics square the coordinates, and a scan can sit far from the origin: centre them.
    centre = [(min(xyz[k::3]) + max(xyz[k::3])) / 2.0 for k in range(3)]
    X = [v - centre[0] for v in xyz[0::3]]
    Y = [v - centre[1] for v in xyz[1::3]]
    Z = [v - centre[2] for v in xyz[2::3]]
    # Faces by position (F: the first vertex at each position stands for all of them) and, for
    # a textured mesh, by texture coordinate (T: an index into U and V).
    T0 = mesh.tris[0::3].tolist()
    T1 = mesh.tris[1::3].tolist()
    T2 = mesh.tris[2::3].tolist()
    if same is None:
        F0, F1, F2 = list(T0), list(T1), list(T2)
    else:
        F0 = list(map(same.__getitem__, T0))
        F1 = list(map(same.__getitem__, T1))
        F2 = list(map(same.__getitem__, T2))
    alive = 0
    VF = [[] for _ in range(n)]          # faces around each position
    for f, (a, b, c) in enumerate(zip(F0, F1, F2)):
        if a == b or b == c or a == c:
            F0[f] = -1                   # two corners at one position: no face at all
            continue
        alive += 1
        VF[a].append(f)
        VF[b].append(f)
        VF[c].append(f)
    stats = {"faces_before": alive, "collapses": 0, "faces_stretched": 0, "atlas_shown": 1.0}
    if textured:
        U = mesh.uv[0::2].tolist()
        V = mesh.uv[1::2].tolist()
        chart = _vertex_roots(n, mesh.tris)      # per texture coordinate: the chart it is in
        stray = [0.0] * n                        # per texture coordinate: how far outside its chart
        stretched = bytearray(len(T0))           # faces whose patch is no longer the true one
        reach = TEXTURE_REACH_TEXELS / math.sqrt(texels) if texels > 0.0 else 0.0
    else:
        del T0, T1, T2

    # One quadric per face (its plane, weighted by its area), summed into its three corners.
    sqrt = math.sqrt
    zero = (0.0,) * 10
    planes = []
    add = planes.append
    total_area = 0.0
    for a, b, c in zip(F0, F1, F2):
        if a < 0:
            add(zero)
            continue
        ax = X[a]
        ay = Y[a]
        az = Z[a]
        ux = X[b] - ax
        uy = Y[b] - ay
        uz = Z[b] - az
        vx = X[c] - ax
        vy = Y[c] - ay
        vz = Z[c] - az
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        l2 = nx * nx + ny * ny + nz * nz
        if l2 < 1e-60:
            add(zero)
            continue
        w = 0.5 / sqrt(l2)                # area / |n|^2, for the un-normalised n
        total_area += l2 * w
        d = -(nx * ax + ny * ay + nz * az)
        wx = w * nx
        wy = w * ny
        wz = w * nz
        add((wx * nx, wx * ny, wx * nz, wx * d, wy * ny, wy * nz, wy * d, wz * nz, wz * d, w * d * d))
    Q = [list(map(sum, zip(*[planes[f] for f in fs]))) if fs else None for fs in VF]
    del planes

    def ring_of(u: int) -> dict:
        """Neighbours of u, each with the number of faces on the edge to it."""
        ring = {}
        get = ring.get
        for f in VF[u]:
            a = F0[f]
            b = F1[f]
            c = F2[f]
            if a != u:
                ring[a] = get(a, 0) + 1
            if b != u:
                ring[b] = get(b, 0) + 1
            if c != u:
                ring[c] = get(c, 0) + 1
        return ring

    # Open borders get a plane of their own through each border edge, at right angles to its
    # face, so that sliding a border vertex along a straight border is free and cutting a
    # corner off the outline is not.
    for u in range(n):
        if not VF[u]:
            continue
        for w, faces_on_edge in ring_of(u).items():
            if faces_on_edge != 1 or w < u:
                continue
            for f in VF[u]:
                if F0[f] == w or F1[f] == w or F2[f] == w:
                    break
            a = F0[f]
            b = F1[f]
            c = F2[f]
            ux = X[b] - X[a]
            uy = Y[b] - Y[a]
            uz = Z[b] - Z[a]
            vx = X[c] - X[a]
            vy = Y[c] - Y[a]
            vz = Z[c] - Z[a]
            nx = uy * vz - uz * vy
            ny = uz * vx - ux * vz
            nz = ux * vy - uy * vx
            ex = X[w] - X[u]
            ey = Y[w] - Y[u]
            ez = Z[w] - Z[u]
            mx = ey * nz - ez * ny
            my = ez * nx - ex * nz
            mz = ex * ny - ey * nx
            l2 = mx * mx + my * my + mz * mz
            if l2 < 1e-60:
                continue
            wgt = COLLAPSE_BORDER_WEIGHT * (ex * ex + ey * ey + ez * ez) / l2
            d = -(mx * X[u] + my * Y[u] + mz * Z[u])
            wx = wgt * mx
            wy = wgt * my
            wz = wgt * mz
            plane = (wx * mx, wx * my, wx * mz, wx * d, wy * my, wy * mz, wy * d, wz * mz, wz * d,
                     wgt * d * d)
            Q[u] = [s + t for s, t in zip(Q[u], plane)]
            Q[w] = [s + t for s, t in zip(Q[w], plane)]

    INF = math.inf
    FROZEN = 255
    spent = [0.0] * n            # error already accumulated at each vertex
    queued = [INF] * n           # cost of the live queue entry of each vertex; INF = not queued
    tries = bytearray(n)         # failed attempts; FROZEN = never ask again
    turn_limit = COLLAPSE_MAX_TURN_COS * COLLAPSE_MAX_TURN_COS
    if textured:
        # A face's size in texels is twice its atlas area times this; its size in space is
        # that over the atlas's average density. Both go into the price of flattening it.
        atlas_area = 0.0
        for f, (a, b, c) in enumerate(zip(T0, T1, T2)):
            if F0[f] >= 0:
                atlas_area += abs((U[b] - U[a]) * (V[c] - V[a]) - (V[b] - V[a]) * (U[c] - U[a]))
        half_texels = 0.5 * texels
        space_per_texel = total_area / (half_texels * atlas_area) if half_texels * atlas_area > 0.0 else 0.0

    def moves(u: int):
        """Possible targets for u as [(cost, target)], cheapest first; None when u may not move."""
        ring = ring_of(u)
        border = [w for w, faces_on_edge in ring.items() if faces_on_edge == 1]
        if border:
            if len(border) != 2:
                return None              # several borders meet here: leave it alone
            targets = border             # a border vertex only slides along its border
        else:
            targets = ring
        q0, q1, q2, q3, q4, q5, q6, q7, q8, q9 = Q[u]
        out = []
        for v in targets:
            x = X[v]
            y = Y[v]
            z = Z[v]
            out.append((x * (q0 * x + 2.0 * (q1 * y + q2 * z + q3))
                        + y * (q4 * y + 2.0 * (q5 * z + q6)) + z * (q7 * z + 2.0 * q8) + q9
                        + spent[v], v))
        out.sort()
        return out

    def allowed(u: int, v: int) -> bool:
        """True unless merging u into v would turn one of the faces around u over."""
        x = X[v]
        y = Y[v]
        z = Z[v]
        ox = X[u]
        oy = Y[u]
        oz = Z[u]
        for f in VF[u]:
            a = F0[f]
            b = F1[f]
            c = F2[f]
            if a == v or b == v or c == v:
                continue                 # this face disappears
            if a == u:
                p = b
                q = c
            elif b == u:
                p = c
                q = a
            else:
                p = a
                q = b
            px = X[p]
            py = Y[p]
            pz = Z[p]
            ex = X[q] - px
            ey = Y[q] - py
            ez = Z[q] - pz
            ax = ox - px
            ay = oy - py
            az = oz - pz
            n0x = ey * az - ez * ay
            n0y = ez * ax - ex * az
            n0z = ex * ay - ey * ax
            ax = x - px
            ay = y - py
            az = z - pz
            n1x = ey * az - ez * ay
            n1y = ez * ax - ex * az
            n1z = ex * ay - ey * ax
            dot = n0x * n1x + n0y * n1y + n0z * n1z
            if dot <= 0.0 or dot * dot < turn_limit * (n0x * n0x + n0y * n0y + n0z * n0z) \
                    * (n1x * n1x + n1y * n1y + n1z * n1z):
                return False
        return True

    def retexture(u: int, v: int):
        """What merging u into v does to the texture: (price, plan, fresh).

        `plan` lists (face, coordinate, stretched) for the faces around u that survive: the
        texture coordinate their corner at u gets at v. A coordinate >= 0 is one that exists; a
        negative one, -1 - k, is `fresh[k]`, a new (u, v, chart, stray) to be made when the merge
        goes ahead.
        """
        at_v = {}                        # chart -> the coordinate v has in it
        for g in VF[v]:
            t = T0[g] if F0[g] == v else (T1[g] if F1[g] == v else T2[g])
            at_v[chart[t]] = t
        price = 0.0
        plan = []
        fresh = []
        fans = None                      # coordinate at u -> the faces that have to leave their chart
        for f in VF[u]:
            a = F0[f]
            b = F1[f]
            c = F2[f]
            if a == v or b == v or c == v:
                continue
            if a == u:
                tu = T0[f]
                tp = T1[f]
                tq = T2[f]
                p = b
                q = c
            elif b == u:
                tu = T1[f]
                tp = T2[f]
                tq = T0[f]
                p = c
                q = a
            else:
                tu = T2[f]
                tp = T0[f]
                tq = T1[f]
                p = a
                q = b
            pu = U[tp]
            pv = V[tp]
            eu = U[tq] - pu
            ev = V[tq] - pv
            before = eu * (V[tu] - pv) - ev * (U[tu] - pu)
            tv = at_v.get(chart[tu], -1)
            if tv >= 0:
                after = eu * (V[tv] - pv) - ev * (U[tv] - pu)
                # Same way round in the atlas as before, and not squeezed to a sliver of it.
                if before * after > 0.0 and 16.0 * abs(after) > abs(before):
                    plan.append((f, tv, 0))
                    continue
            if fans is None:
                fans = {}
            fans.setdefault(tu, []).append((f, p, q, tp, tq, before, tv < 0))
        if fans is None:
            return 0.0, plan, fresh
        ox = X[u]
        oy = Y[u]
        oz = Z[u]
        wx = X[v] - ox
        wy = Y[v] - oy
        wz = Z[v] - oz
        for tu, fan in fans.items():
            # v is not part of this chart. Carry the chart's own mapping on to where v is: each
            # face of the fan says where v would lie in the atlas if its patch went on beyond
            # its edge, and the fan takes the mean, weighted by the size of the face.
            uu = U[tu]
            vv = V[tu]
            su = 0.0
            sv = 0.0
            sw = 0.0
            for f, p, q, tp, tq, before, outside in fan:
                if not outside:
                    continue             # v has a place in this chart, only not one this face can use
                e1x = X[p] - ox
                e1y = Y[p] - oy
                e1z = Z[p] - oz
                e2x = X[q] - ox
                e2y = Y[q] - oy
                e2z = Z[q] - oz
                d11 = e1x * e1x + e1y * e1y + e1z * e1z
                d12 = e1x * e2x + e1y * e2y + e1z * e2z
                d22 = e2x * e2x + e2y * e2y + e2z * e2z
                det = d11 * d22 - d12 * d12
                if det <= 1e-12 * d11 * d22:
                    continue
                w1 = wx * e1x + wy * e1y + wz * e1z
                w2 = wx * e2x + wy * e2y + wz * e2z
                ca = (d22 * w1 - d12 * w2) / det
                cb = (d11 * w2 - d12 * w1) / det
                weight = sqrt(det)
                su += weight * (ca * (U[tp] - uu) + cb * (U[tq] - uu))
                sv += weight * (ca * (V[tp] - vv) + cb * (V[tq] - vv))
                sw += weight
            kept = 0.0                   # the share of that step the atlas has room for
            k = 0
            if sw > 0.0:
                su /= sw
                sv /= sw
                step = sqrt(su * su + sv * sv)
                room = reach - stray[tu]
                if step <= room:
                    kept = 1.0
                    off = stray[tu] + step
                elif room > 0.0:
                    kept = room / step
                    su *= kept
                    sv *= kept
                    off = reach
                if kept > 0.0:
                    nu = uu + su
                    nv = vv + sv
                    nu = 0.0 if nu < 0.0 else 1.0 if nu > 1.0 else nu
                    nv = 0.0 if nv < 0.0 else 1.0 if nv > 1.0 else nv
                    fresh.append((nu, nv, chart[tu], off))
                    k = -len(fresh)
            for f, p, q, tp, tq, before, outside in fan:
                size = abs(before) * half_texels
                space = size * space_per_texel
                loss = TEXTURE_LOSS_WEIGHT * space * space * (size / TEXTURE_LOSS_TEXELS
                                                           if size < TEXTURE_LOSS_TEXELS else 1.0)
                if k < 0:
                    pu = U[tp]
                    pv = V[tp]
                    after = (U[tq] - pu) * (nv - pv) - (V[tq] - pv) * (nu - pu)
                    if before * after > 0.0 and 16.0 * abs(after) > abs(before):
                        plan.append((f, k, kept < 1.0))
                        price += loss * (1.0 - kept)
                        continue
                # The corner keeps the coordinate it had: the face goes on showing the patch
                # it showed, pulled over to where it now lies.
                plan.append((f, tu, 1))
                price += loss
        return price, plan, fresh

    heap = []
    pop = heapq.heappop
    push = heapq.heappush
    steps = 0
    for turn_limit in (turn_limit, -1.0):
        # Queue every vertex at the cost of its cheapest move; whether that move is allowed,
        # and what it does to the texture, is only worked out when the entry reaches the top.
        # The second round, entered only when the first one runs dry above the budget, lifts
        # the limit on how far a face may turn (it still may not flip): the budget is a
        # promise, a smooth fold is not.
        if alive <= target_faces:
            break
        heap.clear()
        for u in range(n):
            if not VF[u]:
                continue
            found = moves(u)
            if not found:
                queued[u] = INF
                tries[u] = FROZEN
                continue
            tries[u] = 0
            queued[u] = found[0][0]
            heap.append((found[0][0], u))
        heapq.heapify(heap)
        while alive > target_faces and heap:
            cost, u = pop(heap)
            if cost != queued[u]:
                continue                 # an older entry of a vertex that was queued again
            found = moves(u)
            v = -1
            now = INF
            plan = None
            if found:
                looked = 0
                for geometric, w in found:
                    if geometric >= now:
                        break            # the rest can only be dearer
                    if not allowed(u, w):
                        continue
                    if textured:
                        price, changes, made = retexture(u, w)
                        if geometric + price < now:
                            now = geometric + price
                            v = w
                            plan = changes
                            fresh = made
                    else:
                        now = geometric
                        v = w
                    looked += 1
                    if looked == 3:
                        break
            if v < 0:
                # Nothing it may do for now. A collapse next to it puts it back in the queue.
                queued[u] = INF
                tries[u] = tries[u] + 1 if tries[u] < 3 else FROZEN
                continue
            if now > cost and heap and now > heap[0][0]:
                queued[u] = now          # dearer than when it was queued: not its turn yet
                push(heap, (now, u))
                continue
            # --- merge u into v
            if plan:
                if fresh:
                    base = len(U) + len(fresh)       # fresh[k] becomes coordinate base - 1 - k
                    for nu, nv, ch, off in reversed(fresh):
                        U.append(nu)
                        V.append(nv)
                        chart.append(ch)
                        stray.append(off)
                for f, t, pulled in plan:
                    if t < 0:
                        t += base
                    if pulled:
                        stretched[f] = 1
                    if F0[f] == u:
                        T0[f] = t
                    elif F1[f] == u:
                        T1[f] = t
                    else:
                        T2[f] = t
            around_v = VF[v]
            ring_v = ring_of(v)          # as it was before the merge; only used as a quick filter
            for f in VF[u]:
                a = F0[f]
                b = F1[f]
                c = F2[f]
                if a == v or b == v or c == v:
                    alive -= 1
                    F0[f] = -1
                    around_v.remove(f)
                    VF[a if (a != u and a != v) else (b if (b != u and b != v) else c)].remove(f)
                    continue
                if a == u:
                    p = b
                    q = c
                elif b == u:
                    p = c
                    q = a
                else:
                    p = a
                    q = b
                if p in ring_v and q in ring_v:
                    twin = -1
                    for g in around_v:
                        ga = F0[g]
                        gb = F1[g]
                        gc = F2[g]
                        if (ga == p or gb == p or gc == p) and (ga == q or gb == q or gc == q):
                            twin = g
                            break
                    if twin >= 0:
                        # The moved face lands on one v already has: a flap with nothing
                        # inside, the two walls of a pinched handle. Both go. (Keeping one
                        # of them was tried, for the sake of thin two-sided shells: the mesh
                        # covered the same pixels of six test views to within 0.2 %, and one
                        # collapse in two left such a fin behind.)
                        alive -= 2
                        F0[f] = -1
                        F0[twin] = -1
                        around_v.remove(twin)
                        VF[p].remove(f)
                        VF[p].remove(twin)
                        VF[q].remove(f)
                        VF[q].remove(twin)
                        continue
                if a == u:
                    F0[f] = v
                elif b == u:
                    F1[f] = v
                else:
                    F2[f] = v
                around_v.append(f)
            VF[u] = ()
            queued[u] = INF
            Q[v] = [s + t for s, t in zip(Q[v], Q[u])]
            Q[u] = None
            spent[v] = now
            for f in around_v:
                for w in (F0[f], F1[f], F2[f]):
                    if queued[w] == INF and tries[w] != FROZEN:
                        queued[w] = now
                        push(heap, (now, w))
            steps += 1
            if tick is not None and not steps & 0xFFF:
                tick(alive)
    stats["collapses"] = steps
    stats["faces_after"] = alive

    # --- pack what is left
    keep = [a >= 0 for a in F0]
    corners = []
    for column in (F0, F1, F2):
        corners.append(list(itertools.compress(column, keep)))
    if not textured:
        tris = array.array("i", bytes(12 * alive))
        for k in range(3):
            tris[k::3] = array.array("i", corners[k])
        return select_faces(Mesh(xyz, tris), [True] * alive), stats
    # One output vertex per (position, texture coordinate) pair in use.
    pairs = {}
    number = pairs.setdefault
    tris = array.array("i", bytes(12 * alive))
    span = len(U)
    for k, column in enumerate((T0, T1, T2)):
        tris[k::3] = array.array("i", [number(p * span + t, len(pairs))
                                       for p, t in zip(corners[k], itertools.compress(column, keep))])
    source = [key // span for key in pairs]
    coords = [key % span for key in pairs]
    shown = 0.0
    for a, b, c in zip(*(itertools.compress(column, keep) for column in (T0, T1, T2))):
        shown += abs((U[b] - U[a]) * (V[c] - V[a]) - (V[b] - V[a]) * (U[c] - U[a]))
    stats["faces_stretched"] = sum(itertools.compress(stretched, keep))
    stats["atlas_shown"] = shown / atlas_area if atlas_area > 0.0 else 0.0
    uv = array.array("f", bytes(8 * len(pairs)))
    uv[0::2] = array.array("f", map(U.__getitem__, coords))
    uv[1::2] = array.array("f", map(V.__getitem__, coords))
    return Mesh(_take3(xyz, source), tris, uv), stats


# =============================================================================================
# Frames
# =============================================================================================

IDENTITY = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def read_scene_transform(run_dir) -> dict:
    """Read Spirula's scene_transform.json. Returns row-major 3x4 matrices for both directions.

    A run without the file was trained in the dataset frame, so both are the identity. Nothing
    here applies these matrices: the meshes are written in the training frame, which is the
    frame of the splat, whatever its relation to the dataset frame. The result only tells the
    caller whether the two frames differ.
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
    """Read the one textured triangle mesh of a GLB written by `spirula mesh`.

    Returns (mesh, image): positions, triangles and texture coordinates exactly as stored, and
    the embedded texture file (PNG or JPEG bytes), or None when the mesh has no texture.
    Normals are not read; they are recomputed. A file whose node carries a transform is
    refused: its vertex numbers would no longer be splat coordinates.
    """
    path = Path(path)

    def fail(msg: str):
        raise MeshError(f"cannot use the mesh Spirula wrote ({path}): {msg}")

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
        fail(f"its structure is not what `spirula mesh` is known to write ({e!r})")
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
# Spirula
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


def _atlas_edge(texture_size: int) -> int:
    """Engines want power-of-two textures; the largest one that fits the requested size."""
    edge = 1
    while edge * 2 <= texture_size:
        edge *= 2
    return edge


# `spirula mesh` names the stage it is in at the start of every line. The stages in the order
# their first line appears, with the share of the run that is over by then (measured, 1 M
# splats, merge factor 1, chart angle 10, 4096 texture: 142 s in all).
_SPIRULA_STAGES = (("loading", 0.0), ("point cloud", 0.02), ("Delaunay", 0.02),
                   ("occupancy", 0.045), ("cut edges", 0.06), ("bisection", 0.07),
                   ("marching tets", 0.11), ("merge", 0.12), ("cull unseen", 0.20),
                   ("merge", 0.24), ("cleanup", 0.26), ("cull unseen", 0.43), ("merge", 0.44),
                   ("cleanup", 0.45), ("quality", 0.50), ("orient", 0.56), ("stats", 0.57),
                   ("texel density", 0.57), ("UV", 0.60), ("bake", 0.97), ("texture", 0.98),
                   ("wrote", 0.99))
_FIRST_TEXTURE_STAGE = 17
_REPEATED_STAGES = ("merge", "cleanup", "cull unseen")
# The UV stage is a third of the run and prints four lines; the one that ends its long part.
_UV_PARAMETERIZED = 0.95


def count_gaussians(splat_path) -> int:
    """Number of Gaussians a splat.ply declares in its header, or 0 if it cannot be read."""
    try:
        with open(splat_path, "rb") as f:
            head = f.read(4096)
    except OSError:
        return 0
    found = re.search(rb"element vertex (\d+)", head)
    return int(found.group(1)) if found else 0


def merge_factor_for(gaussians: int) -> float:
    """The `--merge-factor` a splat of this many Gaussians is extracted at."""
    return LARGE_SPLAT_MERGE_FACTOR if gaussians > LARGE_SPLAT_GAUSSIANS else MERGE_FACTOR


def _run_spirula(*, spirula, run_dir, dataset_dir, out_base: Path, texture_size: int,
                 merge_factor: float, log_path, cancel_file, progress) -> dict:
    """Extract and texture in one `spirula mesh` run. Returns {"path", "faces", "charts"}."""
    encoding = {"png": "glb+png", "jpg": "glb+jpg"}[TEXTURE_ENCODING]
    cmd = [str(spirula), "mesh", str(run_dir), "--lang", "en",
           "--data", str(dataset_dir),
           "--output", str(out_base),
           "--format", encoding, "--color", "texture",
           "--texture-size", str(texture_size),
           "--chart-angle-deg", str(CHART_ANGLE_DEG),
           "--merge-factor", repr(float(merge_factor)),
           "--floater-min-faces", str(SPIRULA_FLOATER_MIN_FACES)]
    state = {"at": 0, "last_error": "", "no_cameras": False, "faces": 0, "charts": 0}

    def on_line(line: str) -> None:
        m = re.match(r"\[meshing\] ([A-Za-z ]+?):", line)
        if m:
            # Only ever move forward. Stage names repeat (merge, cleanup and cull unseen run
            # several times): those only count when they are at most a few steps ahead.
            at = state["at"]
            reach = 4 if m.group(1) in _REPEATED_STAGES else len(_SPIRULA_STAGES)
            for j in range(at, min(at + reach, len(_SPIRULA_STAGES))):
                if _SPIRULA_STAGES[j][0] == m.group(1):
                    if j > at:
                        state["at"] = j
                        progress(_SPIRULA_STAGES[j][1],
                                 "提取表面" if j < _FIRST_TEXTURE_STAGE else "生成贴图")
                    break
            found = re.search(r"stats:.*?faces: (\d+)", line)
            if found:
                state["faces"] = int(found.group(1))
            found = re.search(r"UV: charts packed: (\d+)", line)
            if found:
                state["charts"] = int(found.group(1))
            if "UV: charts parameterized" in line:
                progress(_UV_PARAMETERIZED, "生成贴图")
        if "error" in line.lower():
            state["last_error"] = line
        if "No camera dataset in use" in line:
            state["no_cameras"] = True

    out_base.parent.mkdir(parents=True, exist_ok=True)
    for stale in out_base.parent.glob(out_base.name + "*.glb"):
        stale.unlink()
    try:
        proc.run(cmd, log_path=log_path, timeout=SPIRULA_TIMEOUT, stall_timeout=SPIRULA_STALL,
                 on_line=on_line, cancel_file=cancel_file)
    except proc.ProcTimeout as e:
        raise MeshError(
            f"Surface extraction did not finish in {SPIRULA_TIMEOUT // 60} minutes and was "
            f"stopped. The GPU is probably busy with another job; wait for it and run the mesh "
            f"step again. Log: {log_path}") from e
    except proc.ProcStalled as e:
        raise MeshError(
            f"Surface extraction stopped making progress and was stopped. Close other GPU "
            f"programs and run the mesh step again. Log: {log_path}") from e
    except proc.ProcError as e:
        detail = f" ({state['last_error']})" if state["last_error"] else ""
        raise MeshError(
            f"Spirula could not extract a mesh from {run_dir}{detail}. Check that the dataset "
            f"folder {dataset_dir} is the one the splat was trained on and that the GPU has "
            f"free memory. Log: {log_path}") from e
    if state["no_cameras"]:
        # Without cameras Spirula still writes a mesh, from the Gaussian densities alone: about
        # twice the faces, with the sky and every distant floater turned into geometry, and
        # nothing to texture it from.
        raise MeshError(
            f"Spirula could not read the cameras in {dataset_dir} and would have meshed without "
            f"them, which gives a much worse surface. Check that the folder holds the dataset "
            f"the splat was trained on (images/ and sparse/). Log: {log_path}")
    out = out_base.with_name(out_base.name + ".glb")
    if not out.exists():
        candidates = sorted(out_base.parent.glob(out_base.name + "*.glb"))
        if not candidates:
            raise MeshError(f"Spirula finished but wrote no mesh under {out_base.parent}. "
                            f"Log: {log_path}")
        out = candidates[0]
    return {"path": out, "faces": state["faces"], "charts": state["charts"]}


# =============================================================================================
# Entry point
# =============================================================================================

def build_meshes(*, run_dir, dataset_dir, out_dir, work_dir, tools, visual_faces=300_000,
                 collision_faces=20_000, texture_size=4096, log_path, cancel_file=None,
                 on_progress=None) -> dict:
    """Write `out_dir/mesh.glb` and `out_dir/collision.glb` from a Spirula training run.

    run_dir       Spirula run folder (config.json, scene_transform.json, step-*.ckpt/).
    dataset_dir   the dataset the run was trained on (images/ and sparse/), as Spirula's own
                  pose estimation writes it or undistorted; lens distortion may be left in the
                  camera model.
    work_dir      scratch folder for intermediate files; safe to delete afterwards.
    tools         {"spirula": path to spirula.exe}; other entries are ignored.
    visual_faces, collision_faces
                  upper bounds on the triangle counts of the two meshes.
    texture_size  upper bound on the texture's edge, in pixels. The atlas is square, with the
                  largest power-of-two edge that fits.
    log_path      the tool's output and a summary of each step are appended here.
    cancel_file   if this path appears the work stops and proc.Cancelled is raised.
    on_progress   called as on_progress(fraction 0..1, message).

    Both files are in the frame of the run's splat.ply (see the module docstring). Raises
    MeshError with an actionable message when a step fails.
    """
    started = time.monotonic()
    run_dir = Path(run_dir).resolve()
    dataset_dir = Path(dataset_dir).resolve()
    out_dir = Path(out_dir).resolve()
    work_dir = Path(work_dir).resolve()
    log_path = Path(log_path).resolve()
    if "spirula" not in tools or not Path(tools["spirula"]).is_file():
        raise MeshError(f"tools['spirula'] must be the path to spirula.exe; "
                        f"got {tools.get('spirula')!r}. Run setup to install the tools.")
    spirula = Path(tools["spirula"]).resolve()
    for name, value in (("visual_faces", visual_faces), ("collision_faces", collision_faces)):
        if not isinstance(value, int) or value < 100:
            raise MeshError(f"{name} must be a whole number of at least 100, got {value!r}")
    if not isinstance(texture_size, int) or not 256 <= texture_size <= 16384:
        raise MeshError(f"texture_size must be between 256 and 16384, got {texture_size!r}")
    checkpoint = _find_checkpoint(run_dir)
    if not (dataset_dir / "images").is_dir() or not (dataset_dir / "sparse").is_dir():
        raise MeshError(
            f"{dataset_dir} is not the dataset of this run: it needs an images/ folder and a "
            f"sparse/ folder. Pass the dataset folder the splat was trained on.")
    frames = read_scene_transform(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = out_dir / "mesh.glb"
    collision_path = out_dir / "collision.glb"
    for old in (mesh_path, collision_path):
        if old.exists():
            old.unlink()   # a failed run must not leave last time's files looking like its result

    def report(lo: float, hi: float) -> Callable[[float, str], None]:
        def inner(fraction: float, message: str) -> None:
            if on_progress is not None:
                try:
                    on_progress(lo + (hi - lo) * max(0.0, min(1.0, fraction)), message)
                except Exception:
                    pass  # a broken progress display must not stop the meshing
        return inner

    def ticker(lo: float, hi: float, first: int, last: int, message: str):
        """Progress and cancellation for a collapse running from `first` faces down to `last`."""
        say = report(lo, hi)

        def tick(faces_left: int) -> None:
            _check_cancel(cancel_file)
            say((first - faces_left) / max(1, first - last), message)
        return tick

    timings = {}

    def timed(key: str, since: float) -> float:
        timings[key] = round(time.monotonic() - since, 2)
        return time.monotonic()

    # 1. Surface extraction and texturing (GPU).
    _check_cancel(cancel_file)
    t = time.monotonic()
    report(0.0, 0.0)(0.0, "提取表面")
    edge = _atlas_edge(texture_size)
    gaussians = count_gaussians(checkpoint / "splat.ply")
    merge_factor = merge_factor_for(gaussians)
    extracted = _run_spirula(spirula=spirula, run_dir=run_dir, dataset_dir=dataset_dir,
                             out_base=work_dir / "spirula" / "mesh", texture_size=edge,
                             merge_factor=merge_factor, log_path=log_path,
                             cancel_file=cancel_file, progress=report(0.0, 0.62))
    t = timed("extract", t)

    # 2. Read it back and drop the fragments.
    _check_cancel(cancel_file)
    report(0.62, 0.62)(0.0, "清除零散碎片")
    raw, image = read_glb(extracted["path"])
    raw_faces = raw.face_count
    if image is None:
        raise MeshError(
            f"Spirula wrote a mesh without a texture for {run_dir}. It textures from the photos "
            f"in {dataset_dir / 'images'}: check that they are still there. Log: {log_path}")
    _, tex_w, tex_h = image_size(image)
    if max(tex_w, tex_h) > texture_size:
        raise MeshError(f"Spirula wrote a {tex_w} x {tex_h} texture where at most {texture_size} "
                        f"px was asked for. Log: {log_path}")
    labels, _, comp_areas = face_components(raw)
    raw_components = len(comp_areas)
    largest = max(comp_areas.values())
    total_area = sum(comp_areas.values())
    if not largest > 0.0:
        raise MeshError(
            f"Spirula extracted a mesh without any surface from {run_dir}. The splat may not "
            f"have converged; check the training result before meshing.")
    kept = {lab for lab, a in comp_areas.items() if a >= VISUAL_MIN_AREA_FRACTION * largest}
    visual_components = len(kept)
    visual_area = sum(comp_areas[lab] for lab in kept) / total_area
    visual = raw if len(kept) == raw_components else \
        select_faces(raw, list(map(kept.__contains__, labels)))
    del raw, labels
    t = timed("filter", t)

    # 3. Visual mesh: down to its budget, keeping the atlas.
    _check_cancel(cancel_file)
    filtered_faces = visual.face_count
    stretched = 0
    atlas_shown = 1.0
    if filtered_faces > visual_faces:
        report(0.65, 0.65)(0.0, "简化带贴图网格")
        visual, collapse_stats = collapse_to(
            visual, visual_faces, texels=float(tex_w * tex_h),
            tick=ticker(0.68, 0.90, filtered_faces, visual_faces, "简化带贴图网格"))
        stretched = collapse_stats["faces_stretched"]
        atlas_shown = collapse_stats["atlas_shown"]
        # Pinching a handle shut can cut a sliver loose; the fragment rule applies to those too.
        visual, _ = keep_large_components(visual, VISUAL_MIN_AREA_FRACTION)
    t = timed("visual_simplify", t)

    _check_cancel(cancel_file)
    report(0.90, 0.90)(0.0, "写出带贴图网格")
    uvs = visual.uv
    if min(uvs) < 0.0 or max(uvs) > 1.0:
        # Bilinear filtering never needs coordinates outside the image; clamp away float noise.
        uvs = array.array("f", [0.0 if c < 0.0 else 1.0 if c > 1.0 else c for c in uvs])
    frame_note = {"frame": "spirula-train (same numbers as splat.ply)", "source": "scan3d"}
    write_glb(mesh_path, positions=visual.xyz, indices=array.array("I", visual.tris),
              normals=seamless_normals(visual), uvs=uvs, image=image, name="mesh",
              extras=frame_note)
    del image
    t = timed("visual_glb", t)

    # 4. Collision mesh: the same surface, larger pieces only, no seams, far fewer faces.
    _check_cancel(cancel_file)
    report(0.91, 0.91)(0.0, "简化碰撞网格")
    collision, first_cut = keep_large_components(weld(visual), COLLISION_MIN_AREA_FRACTION)
    del visual
    collision = drop_degenerate_faces(collision, zero_area=True)
    if collision.face_count > collision_faces:
        collision, _ = collapse_to(
            collision, collision_faces,
            tick=ticker(0.91, 0.96, collision.face_count, collision_faces, "简化碰撞网格"))
        collision, _ = keep_large_components(collision, COLLISION_MIN_AREA_FRACTION)
        collision = drop_degenerate_faces(collision, zero_area=True)
    if collision.face_count == 0:
        raise MeshError("simplifying the collision mesh left nothing; raise collision_faces")
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
    _note(log_path, f"spirula mesh of {gaussians} Gaussians at merge factor {merge_factor}, chart "
                    f"angle {CHART_ANGLE_DEG}: {raw_faces} faces in "
                    f"{raw_components} components, {extracted['charts']} texture charts; kept "
                    f"{visual_components} components ({filtered_faces} faces, {visual_area:.1%} of "
                    f"the area); scene transform "
                    f"{'identity' if frames['identity'] else 'not identity'}")
    _note(log_path, f"wrote {mesh_path.name}: {mesh_info['faces']} faces ({stretched} of them "
                    f"with a stretched patch of the texture, {atlas_shown:.1%} of the atlas still "
                    f"shown), {mesh_info['vertices']} vertices, {mesh_info['bytes']} "
                    f"bytes; {collision_path.name}: {collision_info['faces']} faces, "
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
        "texture_size": max(tex_w, tex_h),
        "texture_width": tex_w,
        "texture_height": tex_h,
        "texture_format": TEXTURE_ENCODING,
        "texture_charts": extracted["charts"],
        "faces_texture_stretched": stretched,
        "texture_area_kept": round(atlas_shown, 4),
        "merge_factor": merge_factor,
        "chart_angle_deg": CHART_ANGLE_DEG,
        "gaussians": gaussians,
        "raw_faces": raw_faces,
        "raw_components": raw_components,
        "mesh_components_kept": visual_components,
        "mesh_area_kept": round(visual_area, 4),
        "collision_components_kept": collision_components,
        "collision_area_kept": round(first_cut["area_kept_fraction"], 4),
        "frame": "spirula-train",
        "scene_transform_is_identity": frames["identity"],
        "bbox_min": mesh_info["bbox_min"],
        "bbox_max": mesh_info["bbox_max"],
        "timings": timings,
        "seconds": round(time.monotonic() - started, 1),
    }
