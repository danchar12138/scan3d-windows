"""The network weights the pipeline uses: where they come from, which licence covers them, and
which ones the current settings need.

Weights live in tools/models so scanning never touches the network. setup.ps1 and the web UI
both call `download_missing`; nothing else knows a download address.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable, Optional

import config
import proc

# Third-party licences a person has to accept before the matching weights are used. `flag` is
# the name Spirula's --accept-license option knows the licence by; None where Spirula loads a
# local file without asking.
LICENCES = {
    "sam3": {
        "name": "SAM License (Meta)",
        "url": "https://github.com/facebookresearch/sam3/blob/main/LICENSE",
        "used_for": "按文字抠除人、车等移动物体（SAM 3）",
        "flag": "sam3",
    },
    "sam2": {
        "name": "Apache License 2.0 (SAM 2.1, Meta)",
        "url": "https://github.com/facebookresearch/sam2/blob/main/LICENSE",
        "used_for": "抠除移动物体的备选模型（SAM 2.1，配合 Grounding DINO）",
        "flag": "sam2",
    },
    "gdino": {
        "name": "Apache License 2.0 (Grounding DINO, IDEA Research)",
        "url": "https://github.com/IDEA-Research/GroundingDINO/blob/main/LICENSE",
        "used_for": "SAM 2.1 按文字找物体时用的检测器（Grounding DINO）",
        "flag": "gdino",
    },
    "dinov3": {
        "name": "DINOv3 License (Meta)，以及 RoMa v2 (MIT)",
        "url": "https://github.com/facebookresearch/dinov3/blob/adc254450203739c8149213a7a69d8d905b4fcfa/LICENSE.md",
        "used_for": "稠密点云初始化（RoMa v2）",
        "flag": None,
    },
    "birefnet": {
        "name": "MIT License (BiRefNet)",
        "url": "https://github.com/ZhengPeng7/BiRefNet/blob/main/LICENSE",
        "used_for": "物体扫描：只保留主体，抠掉背景（BiRefNet）",
        "flag": "birefnet",
    },
}

# file name -> where it comes from. `licence` is a key of LICENCES, or None when the weights are
# under a permissive licence that needs no acceptance step.
CATALOGUE = {
    "moge2-vitb-normal.onnx": {
        "url": "https://huggingface.co/Ruicheng/moge-2-vitb-normal-onnx/resolve/main/model.onnx",
        "size_mb": 400, "licence": None, "what": "MoGe-2 表面法线估计"},
    "moge2-vitl-normal.onnx": {
        "url": "https://huggingface.co/Ruicheng/moge-2-vitl-normal-onnx/resolve/main/model.onnx",
        "size_mb": 1263, "licence": None, "what": "MoGe-2 表面法线估计（大模型）"},
    "romav2.0.1.pt": {
        "url": "https://github.com/Parskatt/RoMaV2/releases/download/v2.0.1/romav2.0.1.pt",
        "size_mb": 1045, "licence": "dinov3", "what": "RoMa v2 稠密匹配"},
    "sam3-f16.ggml": {
        "url": "https://huggingface.co/PABannier/sam3.cpp/resolve/main/sam3-f16.ggml",
        "size_mb": 1753, "licence": "sam3", "what": "SAM 3 分割"},
    "sam2.1_hiera_large_f16.ggml": {
        "url": "https://huggingface.co/PABannier/sam3.cpp/resolve/main/sam2.1_hiera_large_f16.ggml",
        "size_mb": 430, "licence": "sam2", "what": "SAM 2.1 分割"},
    "grounding-dino-base.safetensors": {
        "url": "https://huggingface.co/IDEA-Research/grounding-dino-base/resolve/main/model.safetensors",
        "size_mb": 890, "licence": "gdino", "what": "Grounding DINO 文字检测"},
    "birefnet-general.safetensors": {
        "url": "https://huggingface.co/ZhengPeng7/BiRefNet/resolve/main/model.safetensors",
        "size_mb": 424, "licence": "birefnet", "what": "BiRefNet 主体分割"},
    "aliked-n16rot.onnx": {
        "url": "https://github.com/colmap/colmap/releases/download/3.13.0/aliked-n16rot.onnx",
        "size_mb": 3, "licence": None, "what": "ALIKED 特征点"},
    "aliked-lightglue.onnx": {
        "url": "https://github.com/colmap/colmap/releases/download/3.13.0/aliked-lightglue.onnx",
        "size_mb": 44, "licence": None, "what": "LightGlue 匹配"},
}

_ACCEPTED_FILE = config.TOOLS / "accepted-licences.json"
_CURL = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "curl.exe"


def accepted_licences() -> set[str]:
    """Licence ids the person running this installation has accepted."""
    try:
        data = json.loads(_ACCEPTED_FILE.read_text(encoding="utf-8"))
        return {k for k in data.get("accepted", {}) if k in LICENCES}
    except (OSError, json.JSONDecodeError, AttributeError):
        return set()


def accept_licences(ids: list[str]) -> set[str]:
    """Record that the person at this machine accepted these licences."""
    unknown = [i for i in ids if i not in LICENCES]
    if unknown:
        raise ValueError(f"unknown licence id(s): {unknown}")
    try:
        data = json.loads(_ACCEPTED_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = {}
    data.setdefault("note", "Third-party model licences accepted by the person running this "
                            "installation. Not part of the source repository.")
    accepted = data.setdefault("accepted", {})
    for i in ids:
        accepted.setdefault(i, {"licence": LICENCES[i]["name"], "url": LICENCES[i]["url"],
                                "accepted_at": time.strftime("%Y-%m-%d")})
    _ACCEPTED_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _ACCEPTED_FILE.with_name(_ACCEPTED_FILE.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, _ACCEPTED_FILE)
    return accepted_licences()


def licence_of(file_name: str) -> Optional[str]:
    return CATALOGUE[file_name]["licence"]


def feature_models(settings: dict) -> dict[str, dict]:
    """Every feature that needs weights: {"files": [...], "enabled": bool}. A feature is enabled
    when some quality level in settings.json turns it on; the others stay available as
    switches and their weights are fetched only when asked for."""
    levels = list(settings["quality"].values())

    def on(key: str) -> bool:
        return any(q.get(key) for q in levels)

    used = {
        "masks": {"files": [f for f in (settings["masks"]["model_file"],
                                        settings["masks"].get("detector_file")) if f],
                  "enabled": on("masks")},
        "subject_only": {"files": [settings["subject"]["model_file"]],
                         "enabled": on("subject_only")},
        "normals": {"files": [settings["normals"]["model_file"]], "enabled": on("normals")},
        "dense_seed": {"files": [settings["dense_seed"]["model_file"]],
                       "enabled": on("dense_seed")},
        "sfm": {"files": [a[k] for a in settings["sfm"]["attempts"]
                          for k in ("aliked_model", "lightglue_model") if a.get(k)],
                "enabled": True},
    }
    for feature in used.values():
        for f in feature["files"]:
            if f not in CATALOGUE:
                raise ValueError(f"settings.json names a model file this version does not know: {f}")
    return {k: v for k, v in used.items() if v["files"]}


def licence_args(file_name: str) -> list[str]:
    """Command-line arguments that tell Spirula the licence of these weights was accepted.
    Empty when the person has not accepted it, so the tool refuses instead of this program
    answering for them."""
    licence = licence_of(file_name)
    flag = LICENCES[licence]["flag"] if licence else None
    if flag and licence in accepted_licences():
        return ["--accept-license", f"{flag}=yes"]
    return []


def usable(file_name: str, accepted: Optional[set[str]] = None) -> bool:
    """True when the weights are installed and their licence, if any, has been accepted."""
    accepted = accepted_licences() if accepted is None else accepted
    licence = licence_of(file_name)
    return (config.MODELS / file_name).exists() and (licence is None or licence in accepted)


def status(settings: dict) -> dict:
    """What the settings need, what is installed and which licences are still open."""
    accepted = accepted_licences()
    features = {}
    pending: set[str] = set()
    missing: list[str] = []
    for feature, info in feature_models(settings).items():
        files = info["files"]
        needs = sorted({licence_of(f) for f in files if licence_of(f)} - accepted)
        absent = [f for f in files if not (config.MODELS / f).exists()
                  and (licence_of(f) is None or licence_of(f) in accepted)]
        features[feature] = {"files": files, "enabled": info["enabled"],
                             "ready": not needs and not absent,
                             "licences_needed": needs, "files_missing": absent,
                             "download_mb": sum(CATALOGUE[f]["size_mb"] for f in files
                                                if not (config.MODELS / f).exists())}
        if info["enabled"]:
            pending.update(needs)
            missing.extend(f for f in absent if f not in missing)
    return {
        "features": features,
        "licences": {k: {**{a: b for a, b in v.items() if a != "flag"},
                         "accepted": k in accepted, "needed": k in pending}
                     for k, v in LICENCES.items()},
        # Weights of the enabled features that can be fetched right now.
        "files_missing": missing,
        "download_mb": sum(CATALOGUE[f]["size_mb"] for f in missing),
    }


def download_missing(settings: dict, *, log_path: Path, features: Optional[list[str]] = None,
                     echo: Optional[Callable[[str], None]] = print,
                     cancel_file: Optional[Path] = None) -> list[str]:
    """Fetch the weights of the enabled features, or of the features named, whose licence is
    accepted. Returns the names downloaded. Interrupted downloads resume."""
    state = status(settings)
    if features is None:
        wanted = state["files_missing"]
    else:
        unknown = [f for f in features if f not in state["features"]]
        if unknown:
            raise ValueError(f"unknown feature(s): {unknown}")
        wanted = []
        for feature in features:
            wanted.extend(f for f in state["features"][feature]["files_missing"] if f not in wanted)
    config.MODELS.mkdir(parents=True, exist_ok=True)
    part_dir = config.MODELS / "_part"
    part_dir.mkdir(exist_ok=True)
    done: list[str] = []
    for name in wanted:
        entry = CATALOGUE[name]
        if echo:
            echo(f"[download] {name} ({entry['size_mb']} MB) - {entry['what']}")
        part = part_dir / name
        # The CDNs drop long transfers on some networks: retry on any error, resuming each time.
        proc.run([str(_CURL), "-L", "--fail", "--retry", "30", "--retry-delay", "3",
                  "--retry-all-errors", "-C", "-", "-sS", "-o", str(part), entry["url"]],
                 log_path=log_path, timeout=6 * 3600, stall_timeout=None, cancel_file=cancel_file)
        os.replace(part, config.MODELS / name)
        done.append(name)
        if echo:
            echo(f"[ok] {name}")
    return done
