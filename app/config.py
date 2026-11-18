"""Paths and settings shared by the pipeline, the CLI and the server."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app"
TOOLS = ROOT / "tools"
RUNS = ROOT / "runs"
# Network weights the tools need, kept inside the bundle so nothing is fetched while scanning.
MODELS = TOOLS / "models"

# Spirula Studio: frame extraction, masking, camera poses, priors, splat training and surface
# extraction. COLMAP: the undistorted image workspace and texturing the mesh from the photos.
TOOL_PATHS = {
    "spirula": TOOLS / "spirula" / "spirula.exe",
    "colmap": TOOLS / "colmap" / "bin" / "colmap.exe",
}

QUALITIES = ("preview", "standard", "high")
STAGES = ("ingest", "sfm", "dataset", "train", "mesh", "export")

# What Spirula decodes. Other image types are reported to the user and skipped.
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
UNSUPPORTED_IMAGE_EXT = {".webp", ".heic", ".heif", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"}


def load_settings() -> dict:
    """Read settings.json. Every tunable lives there so behaviour is changed by editing one
    file, not by code paths that guess."""
    with open(APP / "settings.json", encoding="utf-8") as f:
        return json.load(f)


def missing_tools() -> list[str]:
    """Names of the programs that are not installed."""
    return [name for name, path in TOOL_PATHS.items() if not path.exists()]
