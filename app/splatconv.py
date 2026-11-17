"""Compact browser previews of a trained Gaussian splat. Standard library only.

The training result (`output/splat.ply`, 248 bytes per Gaussian with SH degree 3) is too big for
a phone and slow to fetch anywhere. This module rewrites it as an SPZ file: the open,
gzip-compressed splat format (MIT licensed specification, https://github.com/nianticlabs/spz)
that the web viewer's renderer (Spark, `app/web/vendor/spark/`) loads directly. Measured on a
999,861-Gaussian scan (236.5 MB): 40.2 MB with every Gaussian and all view-dependent colour,
5.1 MB for the phone variant.

File names (the web viewer and `app/server.py` look for exactly these, next to splat.ply)

    output/preview.spz        every Gaussian, SH degree 3   write_preview(ply, out / PREVIEW_NAME)
    output/preview-lite.spz   the 300,000 most important,   write_preview(ply, out / PREVIEW_LITE_NAME,
                              SH degree 0                       max_gaussians=LITE_MAX_GAUSSIANS)

The pipeline's export stage should write both. When a job has none (it predates them, or the
export was interrupted), or has a preview.spz from before previews kept view-dependent colour,
the server builds them on demand, one at a time, in a child process:
`python splatconv.py <splat.ply> <out.spz> [--max N] [--sh D] [--sh-bits B]`, cached under
`runs/<job>/web-preview/`.

    write_preview(splat_ply, out_path, *, max_gaussians=None, sh_degree=None, sh_bits=None) -> dict
        sh_degree   0..3: how many bands of view-dependent colour to keep (never more than the
                    PLY has). None: DEFAULT_SH_DEGREE (3) for a full preview, 0 when
                    `max_gaussians` is given, so the two calls above need no extra argument.
        sh_bits     4..8: bits kept of each SH coefficient. None: DEFAULT_SH_BITS (8).
        Returns {"format": "spz", "spz_version": 3, "sh_degree": d, "sh_degree_in": d,
                 "sh_bits": b or None, "fractional_bits": n, "gaussians_in": n,
                 "gaussians_out": n, "bytes_in": n, "bytes_out": n, "seconds": s}.
        Raises PreviewError (a ValueError) for a file that is not a usable 3DGS PLY, OSError for
        I/O problems. The output is written to a temporary file next to `out_path` and moved
        into place only when complete, so a reader never sees half a file.

    splat_info(splat_ply) -> {"gaussians": n, "sh_degree": d}     from the PLY header alone
    read_preview_header(path) -> {"spz_version", "count", "sh_degree", "sh_bits", ...}
                                     what an SPZ file holds, without decoding all of it
    read_preview(path) -> dict       decodes an SPZ written here (tests, diagnostics)

What is kept, what is lost
    Position   24-bit fixed point per axis. `fractional_bits` is chosen from the scene's extent
               (never below 12): the error is at most 2**-(fractional_bits+1) scene units.
               A Gaussian further out than the 24 bits reach is clamped to the edge.
    Scale      the log scale, in steps of 1/16 (about 6 % in size), range -10 .. +5.94.
    Rotation   unit quaternion as "smallest three", 10 bits each: within 0.2 degrees.
    Colour     the SH DC term, 8 bits per channel, plus the view-dependent SH bands
               (f_rest_*) up to `sh_degree`, one byte per coefficient: round(v * 128) + 128,
               so steps of 1/128 and a range of -1 .. +0.992 (clamped).
    Opacity    sigmoid(opacity), 8 bits.
    Coordinates are the same numbers as in splat.ply (no axis flip), so the rotation in
    report.json `orientation.rotation_to_y_up_xyzw` aligns the preview with the GLB files
    exactly as it does the PLY.

Why SH degree 3 with 8 bits
    See SH_EVIDENCE below the imports.

Which Gaussians `max_gaussians` keeps
    The ones with the largest  opacity x area,  where opacity = sigmoid(opacity) and area is
    the product of the two longest axes of the ellipsoid (its largest silhouette). That is a
    cheap stand-in for "how much of the picture this Gaussian paints": small or nearly
    transparent ones go first. Ties keep file order. The survivors are not enlarged or
    re-fitted, so a reduced preview is slightly more see-through in finely detailed areas.

Speed
    The file is read in blocks of 65,536 Gaussians; each property is pulled out of a block with
    strided slices and quantized with one list comprehension (only the rotations take a plain
    loop). Measured on the embeddable Python 3.12, 999,861 Gaussians, machine idle: SH degree 0
    in 2.1 s, degree 1 in 3.1 s, degree 2 in 4.8 s, degree 3 (the full preview) in 7.0 s;
    reduced to 300,000 (degree 0) in 1.5 s. With the machine busy training, up to twice that
    (degree 3: 9-14 s). 1,999,722 Gaussians, idle: the full preview in 14.6 s (80.4 MB),
    reduced to 300,000 in 2.4 s.
"""
from __future__ import annotations

import gzip
import itertools
import math
import os
import struct
import sys
import time
import zlib
from array import array
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PREVIEW_NAME = "preview.spz"
PREVIEW_LITE_NAME = "preview-lite.spz"
LITE_MAX_GAUSSIANS = 300_000
DEFAULT_SH_DEGREE = 3           # of the full variant; see SH_EVIDENCE
DEFAULT_SH_BITS = 8             # bits kept of each SH coefficient; see SH_EVIDENCE
MIN_SH_BITS = 4

SH_EVIDENCE = """
Why the full preview keeps SH degree 3 at 8 bits per coefficient (measured 2026-11, scripts in
runs/_viewerlab/scripts: make_eval.py, eval_matrix.mjs, eval_metrics.py, sh_numeric.py).

Two scans of 1,000,000 Gaussians (outdoor "sb", indoor "playroom") were rendered by the web
viewer's renderer (Spark 2.3.1, headless Edge) from three training cameras each, 960 px wide,
and compared (a) with the PLY itself loaded at Spark's extended precision and (b) with the
training photo after fitting gain and offset per channel. PSNR in dB, outdoor / indoor:

    file                         size MB        against the PLY    against the photo
    SH degree 0                  16.9 / 17.1    31.2 / 32.7        24.33 / 29.16
    degree 1, 8 bits             21.3 / 21.0    33.1 / 34.4        24.62 / 29.74
    degree 2, 8 bits             28.9 / 27.9    36.7 / 38.5        24.97 / 31.04
    degree 3, 8 bits             40.2 / 38.3    38.3 / 40.6        25.17 / 32.13
    degree 3, 8 bits, extended   (same file)    42.8 / 42.5        25.15 / 32.17
    degree 3, 7 bits             34.8 / 33.1    38.2 / 40.1        25.18 / 32.09
    degree 3, 6 bits             29.5 / 27.8    37.1 / 39.0        25.14 / 32.19
    degree 3, 5 and 4 bits       21.6 / 19.5    34.7 / 35.2        25.09 / 30.36
    the PLY, Spark's defaults    236 / 236      38.1 / 40.4        25.11 / 32.26
    the PLY, extended            236 / 236      reference          25.13 / 32.05

("extended": loaded with Spark's `extSplats`, which keeps float positions and near-exact SH;
by default Spark packs positions into 16-bit floats and SH into 7, 8 and 6 bits.)

- Every band counts: indoors the photo PSNR climbs 29.2 -> 29.7 -> 31.0 -> 32.1 dB from degree
  0 to 3, and the per-Gaussian colour seen from a training camera is off by 12 (of 255) levels
  on average without SH, 11 with degree 1, 7-9 with degree 2.
- The 5 and 4 bits that SPZ tools usually keep (of the degree-1 and the higher coefficients)
  are too few for these splats: their coefficients are small (half below 0.015, 99 % below
  0.14), so 77-98 % of them round to zero and a degree-3 file renders hardly better than
  degree 1. All 8 bits of the byte that the format stores anyway give back what the PLY
  shows; 6 or 7 bits save 5-10 MB and lose a little against the PLY, nothing measurable
  against the photo. A local viewer has no reason to save those megabytes.
- Any SPZ reader decodes these files: the layout is the standard one, only the rounding done
  by the encoder is finer.
"""

SPZ_MAGIC = 0x5053474E          # "NGSP"
SPZ_VERSION = 3                 # 3: rotations as "smallest three" (2 loses precision near w = 0)
COLOUR_SCALE = 0.15             # SPZ stores dc * 0.15 + 0.5 in 8 bits
MIN_FRACTIONAL_BITS = 12
MAX_FRACTIONAL_BITS = 24
MAX_GAUSSIANS = 10_000_000      # the SPZ readers refuse more
BLOCK = 1 << 16                 # Gaussians handled at a time
GZIP_LEVEL = 6

_PLY_TYPES = {
    "char": 1, "int8": 1, "uchar": 1, "uint8": 1, "short": 2, "int16": 2, "ushort": 2,
    "uint16": 2, "int": 4, "int32": 4, "uint": 4, "uint32": 4, "float": 4, "float32": 4,
    "double": 8, "float64": 8,
}
_NEEDED = ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
           "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")
_SECTIONS = ("positions", "alphas", "colours", "scales", "rotations", "sh")   # SPZ file order
MAX_SH_DEGREE = 3
_SH_COEFFICIENTS = (0, 3, 8, 15)        # per colour channel, without the DC term, by degree


def _bucket_table(bits: int) -> Optional[bytes]:
    """Keeping fewer than 8 bits of an SH coefficient: its byte is rounded to the nearest
    multiple of 2**(8 - bits) (and still stored as a byte, which gzip then packs tightly).
    None for all 8 bits."""
    if bits >= 8:
        return None
    bucket = 1 << (8 - bits)
    return bytes(min(255, (q + bucket // 2) // bucket * bucket) for q in range(256))
_XOR_80 = bytes(b ^ 0x80 for b in range(256))
_LITTLE = sys.byteorder == "little"


class PreviewError(ValueError):
    """The input is not a 3D Gaussian splat PLY this module can read."""


# ---------------------------------------------------------------------------- reading the PLY

def read_ply_header(f) -> Tuple[int, int, Dict[str, int], int]:
    """Parse the header of a binary PLY open at its start.

    Returns (vertex count, bytes per vertex, {float32 property: byte offset}, data offset).
    """
    first = f.readline(256)
    if first.strip() != b"ply":
        raise PreviewError("not a PLY file")
    count = None
    stride = 0
    offsets: Dict[str, int] = {}
    in_vertex = False
    seen_format = False
    for _ in range(4096):
        raw = f.readline(1024)
        if not raw:
            raise PreviewError("the PLY header is not terminated")
        words = raw.decode("ascii", "replace").split()
        if not words or words[0] in ("comment", "obj_info"):
            continue
        if words[0] == "end_header":
            break
        if words[0] == "format":
            if len(words) < 2 or words[1] != "binary_little_endian":
                raise PreviewError("only binary little-endian PLY files are supported")
            seen_format = True
        elif words[0] == "element":
            if len(words) != 3:
                raise PreviewError("malformed PLY element line")
            if count is None:
                if words[1] != "vertex":
                    raise PreviewError("the first PLY element must be 'vertex'")
                try:
                    count = int(words[2])
                except ValueError:
                    raise PreviewError("malformed vertex count") from None
                in_vertex = True
            else:
                in_vertex = False       # later elements (faces ...) follow the vertex data
        elif words[0] == "property" and in_vertex:
            if len(words) != 3 or words[1] not in _PLY_TYPES:
                raise PreviewError("unsupported PLY vertex property: %s" % " ".join(words[1:]))
            if words[1] in ("float", "float32"):
                offsets[words[2]] = stride
            stride += _PLY_TYPES[words[1]]
    else:
        raise PreviewError("the PLY header is too long")
    if not seen_format or count is None or stride == 0:
        raise PreviewError("the PLY header has no binary vertex element")
    if count < 0 or count > MAX_GAUSSIANS:
        raise PreviewError("unsupported number of Gaussians: %d" % count)
    missing = [name for name in _NEEDED if name not in offsets]
    if missing:
        raise PreviewError("not a 3D Gaussian splat PLY (missing float property %s)"
                           % ", ".join(missing))
    return count, stride, offsets, f.tell()


def _sh_layout(offsets: Dict[str, int]) -> Tuple[int, int]:
    """(SH degree the PLY carries, f_rest coefficients per colour channel).

    A 3DGS PLY stores f_rest_0 .. f_rest_(3K-1): K coefficients for red, then K for green,
    then K for blue; K is 3, 8 or 15 for degree 1, 2 or 3."""
    total = 0
    while "f_rest_%d" % total in offsets:
        total += 1
    per_channel = total // 3
    degree = max(d for d in range(MAX_SH_DEGREE + 1) if _SH_COEFFICIENTS[d] <= per_channel)
    return degree, per_channel


def splat_info(splat_ply: os.PathLike) -> dict:
    """{"gaussians": count, "sh_degree": 0..3} of a 3DGS PLY, from its header alone."""
    with open(splat_ply, "rb") as f:
        count, _stride, offsets, _data_offset = read_ply_header(f)
    return {"gaussians": count, "sh_degree": _sh_layout(offsets)[0]}


def _column(block, offset: int, stride: int, n: int) -> array:
    """One float32 property of `n` interleaved vertices, as array('f')."""
    raw = bytearray(4 * n)
    for byte in range(4):
        raw[byte::4] = block[offset + byte::stride]
    column = array("f")
    column.frombytes(raw)
    if not _LITTLE:
        column.byteswap()
    return column


def _finite(column) -> array:
    """The column with NaN and infinities replaced by 0 (the slow path, rarely needed)."""
    isfinite = math.isfinite
    return array("f", [v if isfinite(v) else 0.0 for v in column])


def _blocks(f, data_offset: int, count: int, stride: int):
    """Yield (first index, n, bytes) for each block of vertices."""
    f.seek(data_offset)
    done = 0
    while done < count:
        n = min(BLOCK, count - done)
        block = f.read(n * stride)
        if len(block) != n * stride:
            raise PreviewError("the PLY file is shorter than its header says")
        yield done, n, block
        done += n


# ---------------------------------------------------------------------------- quantization

def _quantize_positions(column, scale: float) -> array:
    """24-bit fixed point, biased by 2**23 so every value is a non-negative int."""
    lo = -8388608.0 / scale
    hi = 8388607.0 / scale
    return array("I", [0 if v <= lo else 16777215 if v >= hi else int(v * scale + 8388608.5)
                       for v in column])


def _quantize_linear(column, scale: float, bias: float) -> bytes:
    """round(v * scale + bias) clamped to a byte."""
    lo = -bias / scale
    hi = (255.0 - bias) / scale
    bias += 0.5
    return bytes([0 if v <= lo else 255 if v >= hi else int(v * scale + bias) for v in column])


def _quantize_alpha(column) -> bytes:
    exp = math.exp
    return bytes([0 if v < -30.0 else int(255.0 / (1.0 + exp(-v)) + 0.5) for v in column])


_ROTATION_SCALE = 511.0 / math.sqrt(0.5)    # 9 bits for a magnitude of at most sqrt(1/2)


def _quantize_rotations(w, x, y, z) -> bytes:
    """SPZ 3 "smallest three": 4 bytes per unit quaternion. The largest of x, y, z, w is left
    out (it follows from the other three); each of the others is a sign bit and 9 bits of
    magnitude; the top two bits say which one was left out. PLY order is w, x, y, z."""
    packed = array("I")
    append = packed.append
    scale = _ROTATION_SCALE
    for qw, qx, qy, qz, norm in zip(w, x, y, z, map(math.hypot, w, x, y, z)):
        if not norm > 1e-30:            # no rotation given (or NaN): identity
            append(0xC0000000)
            continue
        k = scale / norm
        ax, ay, az, aw = abs(qx), abs(qy), abs(qz), abs(qw)
        # e_* = magnitude | sign bit, for the case that the left-out component is positive
        ex = min(int(ax * k + 0.5), 511) | (512 if qx < 0.0 else 0)
        ey = min(int(ay * k + 0.5), 511) | (512 if qy < 0.0 else 0)
        ez = min(int(az * k + 0.5), 511) | (512 if qz < 0.0 else 0)
        ew = min(int(aw * k + 0.5), 511) | (512 if qw < 0.0 else 0)
        if aw >= ax and aw >= ay and aw >= az:
            value = 0xC0000000 | ex << 20 | ey << 10 | ez
            if qw < 0.0:
                value ^= 0x20080200     # q and -q are the same rotation: flip the three signs
        elif ax >= ay and ax >= az:
            value = ey << 20 | ez << 10 | ew
            if qx < 0.0:
                value ^= 0x20080200
        elif ay >= az:
            value = 0x40000000 | ex << 20 | ez << 10 | ew
            if qy < 0.0:
                value ^= 0x20080200
        else:
            value = 0x80000000 | ex << 20 | ey << 10 | ew
            if qz < 0.0:
                value ^= 0x20080200
        append(value)
    if packed.itemsize != 4:            # a platform whose C unsigned int is not 32 bits
        return struct.pack("<%dI" % len(packed), *packed)
    if not _LITTLE:
        packed.byteswap()
    return packed.tobytes()


def _encode_block(columns: Dict[str, array], n: int, position_scale: float,
                  sh_names: List[str], sh_table: Optional[bytes]
                  ) -> Tuple[bytes, bytes, bytes, bytes, bytes, bytes]:
    """The six SPZ sections (positions, alphas, colours, scales, rotations, SH) of one block.
    `sh_names` lists the f_rest properties in SPZ order: coefficient by coefficient, each as
    red, green, blue.

    Raises ValueError or OverflowError when a value is NaN or infinite."""
    positions = bytearray(9 * n)
    for axis, name in enumerate("xyz"):
        fixed = _quantize_positions(columns[name], position_scale)
        if fixed.itemsize == 4:
            if not _LITTLE:
                fixed.byteswap()
            raw = fixed.tobytes()
        else:                       # a platform whose C unsigned int is not 32 bits
            raw = struct.pack("<%dI" % n, *fixed)
        positions[3 * axis::9] = raw[0::4]
        positions[3 * axis + 1::9] = raw[1::4]
        positions[3 * axis + 2::9] = raw[2::4].translate(_XOR_80)   # bias -> two's complement

    alphas = _quantize_alpha(columns["opacity"])

    colours = bytearray(3 * n)
    for channel in range(3):
        colours[channel::3] = _quantize_linear(
            columns["f_dc_%d" % channel], COLOUR_SCALE * 255.0, 0.5 * 255.0)

    scales = bytearray(3 * n)
    for axis in range(3):
        scales[axis::3] = _quantize_linear(columns["scale_%d" % axis], 16.0, 160.0)

    rotations = _quantize_rotations(columns["rot_0"], columns["rot_1"], columns["rot_2"],
                                    columns["rot_3"])
    width = len(sh_names)
    sh = bytearray(width * n)
    for slot, name in enumerate(sh_names):
        quantized = _quantize_linear(columns[name], 128.0, 128.0)   # round(v * 128) + 128
        sh[slot::width] = quantized if sh_table is None else quantized.translate(sh_table)
    return bytes(positions), alphas, bytes(colours), bytes(scales), rotations, bytes(sh)


def _importance(columns: Dict[str, array]) -> List[float]:
    """log(opacity x largest silhouette area) for each Gaussian of a block."""
    log1p, exp = math.log1p, math.exp
    return [a + b + c - min(a, b, c) - (log1p(exp(-o)) if o > -700.0 else -o)
            for o, a, b, c in zip(columns["opacity"], columns["scale_0"], columns["scale_1"],
                                  columns["scale_2"])]


def _selection_mask(scores: array, keep: int) -> bytearray:
    """1 for the `keep` highest scores (ties: earlier in the file first), else 0."""
    threshold = sorted(scores)[len(scores) - keep]
    mask = bytearray([s > threshold for s in scores])
    spare = keep - sum(mask)
    if spare > 0:
        for index, score in enumerate(scores):
            if score == threshold:
                mask[index] = 1
                spare -= 1
                if spare == 0:
                    break
    return mask


# ---------------------------------------------------------------------------- the conversion

def default_sh_degree(max_gaussians: Optional[int]) -> int:
    """The SH degree write_preview uses when none is given: the full variant keeps
    view-dependent colour, a reduced (phone) variant does not."""
    return DEFAULT_SH_DEGREE if max_gaussians is None else 0


def write_preview(splat_ply: Path, out_path: Path, *, max_gaussians: Optional[int] = None,
                  sh_degree: Optional[int] = None, sh_bits: Optional[int] = None) -> dict:
    """Write an SPZ preview of a 3DGS training PLY. See the module comment for the details."""
    started = time.perf_counter()
    splat_ply = Path(splat_ply)
    out_path = Path(out_path)
    if max_gaussians is not None and (isinstance(max_gaussians, bool)
                                      or not isinstance(max_gaussians, int)
                                      or max_gaussians < 1):
        raise ValueError("max_gaussians must be a positive integer or None")
    if sh_degree is None:
        sh_degree = default_sh_degree(max_gaussians)
    elif isinstance(sh_degree, bool) or not isinstance(sh_degree, int) \
            or not 0 <= sh_degree <= MAX_SH_DEGREE:
        raise ValueError("sh_degree must be 0, 1, 2, 3 or None")
    if sh_bits is None:
        sh_bits = DEFAULT_SH_BITS
    elif isinstance(sh_bits, bool) or not isinstance(sh_bits, int) \
            or not MIN_SH_BITS <= sh_bits <= 8:
        raise ValueError("sh_bits must be %d .. 8 or None" % MIN_SH_BITS)
    sh_table = _bucket_table(sh_bits)

    with open(splat_ply, "rb") as f:
        count, stride, offsets, data_offset = read_ply_header(f)
        bytes_in = os.fstat(f.fileno()).st_size
        if bytes_in < data_offset + count * stride:
            raise PreviewError("the PLY file is shorter than its header says")
        reduce = max_gaussians is not None and max_gaussians < count
        sh_in, per_channel = _sh_layout(offsets)
        sh_degree = min(sh_degree, sh_in)       # a PLY cannot give more than it has
        sh_names = ["f_rest_%d" % (channel * per_channel + coefficient)
                    for coefficient in range(_SH_COEFFICIENTS[sh_degree]) for channel in range(3)]
        wanted = _NEEDED + tuple(sh_names)

        # Pass 1: the extent of the scene (it decides the position precision) and, when the
        # count is being reduced, the importance of every Gaussian.
        largest = 0.0
        scores = array("d")
        for _first, n, block in _blocks(f, data_offset, count, stride):
            for name in "xyz":
                column = _column(block, offsets[name], stride, n)
                low, high = min(column), max(column)
                if not (math.isfinite(low) and math.isfinite(high)):   # NaN compares False
                    column = _finite(column)
                    low, high = min(column), max(column)
                largest = max(largest, -low, high)
            if reduce:
                columns = {name: _column(block, offsets[name], stride, n)
                           for name in ("opacity", "scale_0", "scale_1", "scale_2")}
                scores.extend(_importance(columns))
        if reduce:
            # NaN scores (NaN inputs survive `min`) would poison the sort: rank them last.
            if any(s != s for s in scores):
                scores = array("d", [s if s == s else -math.inf for s in scores])
            mask: Optional[bytearray] = _selection_mask(scores, max_gaussians)
            kept = max_gaussians
        else:
            mask = None
            kept = count
        del scores

        fractional_bits = MAX_FRACTIONAL_BITS
        while fractional_bits > MIN_FRACTIONAL_BITS \
                and largest * (1 << fractional_bits) > 8388607.0:
            fractional_bits -= 1
        position_scale = float(1 << fractional_bits)

        # Pass 2: quantize block by block into the SPZ sections.
        out = {name: bytearray() for name in _SECTIONS}
        for first, n, block in _blocks(f, data_offset, count, stride):
            columns = {name: _column(block, offsets[name], stride, n) for name in wanted}
            if mask is not None:
                selector = mask[first:first + n]
                n = sum(selector)
                if n == 0:
                    continue
                columns = {name: array("f", itertools.compress(column, selector))
                           for name, column in columns.items()}
            try:
                sections = _encode_block(columns, n, position_scale, sh_names, sh_table)
            except (ValueError, OverflowError):     # NaN or infinity somewhere in this block
                sections = _encode_block({k: _finite(v) for k, v in columns.items()}, n,
                                         position_scale, sh_names, sh_table)
            for name, section in zip(_SECTIONS, sections):
                out[name] += section

    header = struct.pack("<IIIBBBB", SPZ_MAGIC, SPZ_VERSION, kept, sh_degree, fractional_bits,
                         0, 0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name("%s.%d.tmp" % (out_path.name, os.getpid()))
    try:
        with open(tmp, "wb") as sink:
            packer = zlib.compressobj(GZIP_LEVEL, zlib.DEFLATED, 31)    # 31: gzip container
            sink.write(packer.compress(header))
            for name in _SECTIONS:
                section = out[name]
                for start in range(0, len(section), 1 << 22):
                    sink.write(packer.compress(bytes(section[start:start + (1 << 22)])))
                out[name] = bytearray()
            sink.write(packer.flush())
        os.replace(tmp, out_path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return {"format": "spz", "spz_version": SPZ_VERSION, "sh_degree": sh_degree,
            "sh_degree_in": sh_in, "sh_bits": sh_bits if sh_degree else None,
            "fractional_bits": fractional_bits, "gaussians_in": count, "gaussians_out": kept,
            "bytes_in": bytes_in, "bytes_out": out_path.stat().st_size,
            "seconds": round(time.perf_counter() - started, 2)}


# ---------------------------------------------------------------------------- reading it back

_SH_SAMPLE_BYTES = 1 << 16


def read_preview_header(path: os.PathLike) -> dict:
    """What an SPZ file holds: {"spz_version", "count", "sh_degree", "fractional_bits",
    "sh_bits"}, without decoding all of it.

    "sh_bits" is measured, as the file does not say it: the bits that the first SH
    coefficients (about 1,500 Gaussians of a degree-3 file) actually use, 1..8; None when
    there is no SH or all of those coefficients are zero. A file written with the 5 and 4 bits
    usual for SPZ gives 5; one written by write_preview gives 8. Finding the SH section means
    unpacking what precedes it (20 bytes per Gaussian: about 0.1 s per 1,000,000).

    Raises PreviewError for anything that is not an SPZ file, OSError when it cannot be read."""
    try:
        with gzip.open(path, "rb") as f:
            head = f.read(16)
            if len(head) < 16:
                raise PreviewError("not an SPZ file")
            magic, version, count, sh_degree, fractional_bits, _flags, _reserved = \
                struct.unpack("<IIIBBBB", head)
            if magic != SPZ_MAGIC:
                raise PreviewError("not an SPZ file")
            sample = b""
            if 0 < sh_degree <= MAX_SH_DEGREE and 0 < count <= MAX_GAUSSIANS:
                left = 20 * count
                while left > 0:
                    skipped = len(f.read(min(left, 1 << 20)))
                    if not skipped:
                        break
                    left -= skipped
                sample = f.read(_SH_SAMPLE_BYTES) if left == 0 else b""
    except (EOFError, zlib.error, gzip.BadGzipFile) as exc:
        raise PreviewError("not an SPZ file") from exc
    used = 0
    for value in set(sample) - {128, 255}:      # 128 is zero, 255 may be a clamped value
        used |= value
    sh_bits = None
    if used:
        sh_bits = 8
        while not used & 1:
            used >>= 1
            sh_bits -= 1
    return {"spz_version": version, "count": count, "sh_degree": sh_degree,
            "fractional_bits": fractional_bits, "sh_bits": sh_bits}


def read_preview(path: Path) -> dict:
    """Decode an SPZ file written by write_preview (SPZ version 3).

    Returns {"count", "sh_degree", "fractional_bits", "positions": [(x, y, z)],
    "opacities": [0..1], "colours": [(dc0, dc1, dc2)], "scales": [(log s0, s1, s2)],
    "rotations": [(w, x, y, z)], "sh": [[(r, g, b) for each coefficient]]}
    (the rotation is a unit quaternion; it may be the negative of the one in the PLY, which is
    the same rotation; "sh" has 0, 3, 8 or 15 coefficients per Gaussian).
    Meant for tests and diagnostics: it builds one tuple per Gaussian.
    """
    with gzip.open(path, "rb") as f:
        data = f.read()
    if len(data) < 16:
        raise PreviewError("not an SPZ file")
    magic, version, count, sh_degree, fractional_bits, _flags, _reserved = struct.unpack_from(
        "<IIIBBBB", data, 0)
    if magic != SPZ_MAGIC:
        raise PreviewError("not an SPZ file")
    if version != SPZ_VERSION or sh_degree > MAX_SH_DEGREE:
        raise PreviewError("unsupported SPZ variant (version %d, SH degree %d)"
                           % (version, sh_degree))
    coefficients = _SH_COEFFICIENTS[sh_degree]
    if len(data) != 16 + count * (20 + 3 * coefficients):
        raise PreviewError("the SPZ file has the wrong length")
    at = 16
    raw = data[at:at + 9 * count]
    at += 9 * count
    scale = 1.0 / (1 << fractional_bits)
    axes = []
    for axis in range(3):
        low, mid, high = raw[3 * axis::9], raw[3 * axis + 1::9], raw[3 * axis + 2::9]
        axes.append([((l | m << 8 | h << 16) - (16777216 if h & 0x80 else 0)) * scale
                     for l, m, h in zip(low, mid, high)])
    positions = list(zip(*axes))
    opacities = [v / 255.0 for v in data[at:at + count]]
    at += count
    raw = data[at:at + 3 * count]
    at += 3 * count
    colours = [tuple((v / 255.0 - 0.5) / COLOUR_SCALE for v in raw[i:i + 3])
               for i in range(0, 3 * count, 3)]
    raw = data[at:at + 3 * count]
    at += 3 * count
    scales = [tuple(v / 16.0 - 10.0 for v in raw[i:i + 3]) for i in range(0, 3 * count, 3)]
    rotations = []
    unit = math.sqrt(0.5) / 511.0
    for (value,) in struct.iter_unpack("<I", data[at:at + 4 * count]):
        largest = value >> 30
        q = [0.0, 0.0, 0.0, 0.0]            # x, y, z, w
        total = 0.0
        for index in (3, 2, 1, 0):
            if index != largest:
                q[index] = (value & 511) * unit * (-1.0 if value & 512 else 1.0)
                total += q[index] * q[index]
                value >>= 10
        q[largest] = math.sqrt(max(0.0, 1.0 - total))
        rotations.append((q[3], q[0], q[1], q[2]))
    at += 4 * count
    width = 3 * coefficients
    sh = [[tuple((v - 128) / 128.0 for v in data[start + 3 * k:start + 3 * k + 3])
           for k in range(coefficients)]
          for start in range(at, at + width * count, width)] if width else [[] for _ in range(count)]
    return {"count": count, "sh_degree": sh_degree, "fractional_bits": fractional_bits,
            "positions": positions, "opacities": opacities, "colours": colours,
            "scales": scales, "rotations": rotations, "sh": sh}


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Write a compact SPZ browser preview of a 3D Gaussian splat PLY.")
    parser.add_argument("splat_ply")
    parser.add_argument("out_path")
    parser.add_argument("--max", type=int, default=None, dest="max_gaussians",
                        help="keep only this many Gaussians (the most visible ones)")
    parser.add_argument("--sh", type=int, default=None, dest="sh_degree", choices=(0, 1, 2, 3),
                        help="SH degree to keep (default: %d, or 0 together with --max)"
                             % DEFAULT_SH_DEGREE)
    parser.add_argument("--sh-bits", type=int, default=None, dest="sh_bits",
                        choices=tuple(range(MIN_SH_BITS, 9)),
                        help="bits kept of each SH coefficient (default: %d)" % DEFAULT_SH_BITS)
    args = parser.parse_args(argv)
    try:
        result = write_preview(Path(args.splat_ply), Path(args.out_path),
                               max_gaussians=args.max_gaussians, sh_degree=args.sh_degree,
                               sh_bits=args.sh_bits)
    except (PreviewError, OSError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), flush=True)
        return 1
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
