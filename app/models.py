"""The network weights the pipeline uses: where they come from, which licence covers them, which
stage needs which, and how they get onto this machine.

The person's only choice is whether to use a feature. Enabling a stage that needs weights is
enough: when that stage starts, the pipeline records the licence that covers the weights
(`ensure_stage_models`; `tools/accepted-licences.json`, `"via": "feature-enabled"`, with the date),
downloads what is missing, and tells the tool about the licence (`--accept-license <flag>=yes`,
see `licence_args`). Weights without a licence step (MoGe-2, ALIKED, LightGlue) are only
downloaded. Nothing is asked and nothing is skipped; each licence has a link so the page can say
"using this means agreeing to ...".

Features (the unit of the download table; a stage needs one or more, depending on its parameters)

    id            stage       weights
    sam3          masks       SAM 3                        SAM License (Meta)
    sam2          masks       SAM 2.1 + Grounding DINO     Apache-2.0 x 2
    subject       masks       BiRefNet                     MIT
    normals_vitb  normals     MoGe-2 ViT-B                 -
    normals_vitl  normals     MoGe-2 ViT-L                 -
    dense_seed    dense_seed  RoMa v2                      DINOv3 licence
    aliked        sfm         ALIKED + LightGlue           -

File names come from settings.json (`models`), where each feature names its weights.

Public functions (everything is JSON-able; `settings` defaults to `config.load_settings()`)

    features_table()                      the `GET /api/models` payload:
        {"features": [{"id", "label", "stage", "files": [{"name", "mb", "installed"}],
                       "licence": {"id", "label", "url", "consent": "none"|"auto", "accepted": bool}
                                  | null, "licences": [that, for every licence of the feature],
                       "ready": bool}], "disk_mb": installed weights in MB}
    needs(workflow)                       what the enabled stages of a workflow use:
        {"licences": [{"id", "label", "url", "consent", "accepted", "stage"}],
         "models_needed": [{"file", "mb", "stage"}]}   (weights not installed yet)
    stage_features(stage, params)         ids of the features a stage needs with these parameters
    stage_files(stage, params)            ... and their file names
    accept_licences(ids, via="user")      record licences (unknown id: ValueError); returns the set
    accepted_licences()                   ids recorded on this machine
    ensure_stage_models(stage, params, ...)   the start-of-stage step: record, download, report
    download_features(features, ...)      download these features' weights (CLI `models --download`)
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import config
import proc

# Third-party licences that cover the weights. `flag` is the name Spirula's --accept-license
# option knows the licence by; None where Spirula loads a local file without asking. Every one of
# them is recorded by the pipeline when a stage that needs the weights runs ("consent": "auto");
# weights with no licence step have no entry here ("consent": "none").
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

# file name -> where it comes from. `licence` is a key of LICENCES, or None when the weights have
# no licence step.
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

# feature id -> label and stage; the files come from settings.json ("models"; "aliked" from the
# learned attempt of "sfm").
FEATURES = {
    "sam3": {"label": "SAM 3 抠图", "stage": "masks"},
    "sam2": {"label": "SAM 2.1 + Grounding DINO 抠图", "stage": "masks"},
    "subject": {"label": "BiRefNet 主体分割", "stage": "masks"},
    "normals_vitb": {"label": "MoGe-2 法线（ViT-B）", "stage": "normals"},
    "normals_vitl": {"label": "MoGe-2 法线（ViT-L）", "stage": "normals"},
    "dense_seed": {"label": "RoMa v2 稠密点云", "stage": "dense_seed"},
    "aliked": {"label": "ALIKED + LightGlue 特征匹配", "stage": "sfm"},
}

_ACCEPTED_FILE = config.TOOLS / "accepted-licences.json"
_CURL = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "curl.exe"


# ---------------------------------------------------------------------------------------------
# Licences
# ---------------------------------------------------------------------------------------------

def accepted_licences() -> set:
    """Licence ids recorded on this installation."""
    try:
        data = json.loads(_ACCEPTED_FILE.read_text(encoding="utf-8"))
        return {k for k in data.get("accepted", {}) if k in LICENCES}
    except (OSError, json.JSONDecodeError, AttributeError):
        return set()


def accept_licences(ids: list, via: str = "user") -> set:
    """Record licences as accepted by the person using this installation (the date is kept from
    the first time). Unknown ids: ValueError, and nothing is written."""
    unknown = [i for i in ids if i not in LICENCES]
    if unknown:
        raise ValueError(f"unknown licence id(s): {unknown}")
    try:
        data = json.loads(_ACCEPTED_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
    except (OSError, json.JSONDecodeError):
        data = {}
    data.setdefault("note", "Third-party model licences that apply to the weights this "
                            "installation uses. Recorded when a feature that needs them is "
                            "enabled and runs. Not part of the source repository.")
    accepted = data.setdefault("accepted", {})
    if not isinstance(accepted, dict):
        accepted = data["accepted"] = {}
    changed = False
    for i in ids:
        if i not in accepted:
            accepted[i] = {"licence": LICENCES[i]["name"], "url": LICENCES[i]["url"],
                           "accepted_at": time.strftime("%Y-%m-%d"), "via": via}
            changed = True
    if changed:
        _ACCEPTED_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _ACCEPTED_FILE.with_name(_ACCEPTED_FILE.name + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, _ACCEPTED_FILE)
    return accepted_licences()


def licence_of(file_name: str) -> Optional[str]:
    return CATALOGUE[file_name]["licence"]


def licence_args(file_name: str) -> list:
    """Command-line arguments that tell Spirula the licence of these weights was recorded. Empty
    for weights without a licence step and for licences the tool does not ask about."""
    licence = licence_of(file_name)
    flag = LICENCES[licence]["flag"] if licence else None
    if flag and licence in accepted_licences():
        return ["--accept-license", f"{flag}=yes"]
    return []


def installed(file_name: str) -> bool:
    return (config.MODELS / file_name).exists()


# ---------------------------------------------------------------------------------------------
# Features and stages
# ---------------------------------------------------------------------------------------------

def _settings(settings: Optional[dict]) -> dict:
    return config.load_settings() if settings is None else settings


def feature_files(feature: str, settings: Optional[dict] = None) -> list:
    """The weight files of a feature, as settings.json names them."""
    if feature not in FEATURES:
        raise ValueError(f"unknown feature: {feature}")
    settings = _settings(settings)
    if feature == "aliked":
        files = [a[k] for a in settings["sfm"]["attempts"] if a.get("learned")
                 for k in ("aliked_model", "lightglue_model") if a.get(k)]
    else:
        files = [f for f in settings["models"][feature].values() if f]
    for f in files:
        if f not in CATALOGUE:
            raise ValueError(f"settings.json names a model file this version does not know: {f}")
    return files


def feature_licences(feature: str, settings: Optional[dict] = None) -> list:
    """Licence ids that cover a feature's files, in file order."""
    out: list = []
    for f in feature_files(feature, settings):
        lic = licence_of(f)
        if lic and lic not in out:
            out.append(lic)
    return out


def stage_features(stage: str, params: Optional[dict] = None) -> list:
    """Ids of the features `stage` needs when run with `params` (the stage's own parameter
    dict from the workflow). Stages without weights: []."""
    params = params or {}
    if stage == "masks":
        if params.get("kind") == "subject":
            return ["subject"]
        return ["sam2"] if params.get("engine") == "sam2" else ["sam3"]
    if stage == "normals":
        return ["normals_vitl" if params.get("model") == "vitl" else "normals_vitb"]
    if stage == "dense_seed":
        return ["dense_seed"]
    if stage == "sfm":
        return ["aliked"] if params.get("learned_fallback", True) else []
    return []


def stage_files(stage: str, params: Optional[dict] = None, settings: Optional[dict] = None) -> list:
    files: list = []
    for feature in stage_features(stage, params):
        files.extend(f for f in feature_files(feature, settings) if f not in files)
    return files


def _licence_view(lic: str, accepted: set) -> dict:
    return {"id": lic, "label": LICENCES[lic]["name"], "url": LICENCES[lic]["url"],
            "consent": "auto", "accepted": lic in accepted}


def features_table(settings: Optional[dict] = None) -> dict:
    """The `GET /api/models` payload (see the module docstring)."""
    settings = _settings(settings)
    accepted = accepted_licences()
    rows = []
    for fid, info in FEATURES.items():
        files = feature_files(fid, settings)
        lics = [_licence_view(l, accepted) for l in feature_licences(fid, settings)]
        rows.append({
            "id": fid, "label": info["label"], "stage": info["stage"],
            "files": [{"name": f, "mb": CATALOGUE[f]["size_mb"], "installed": installed(f)}
                      for f in files],
            "licence": lics[0] if lics else None, "licences": lics,
            "ready": all(installed(f) for f in files)})
    disk = 0
    for name in CATALOGUE:
        try:
            disk += (config.MODELS / name).stat().st_size
        except OSError:
            pass
    return {"features": rows, "disk_mb": round(disk / 1048576)}


def needs(workflow: dict, settings: Optional[dict] = None) -> dict:
    """What the stages that will run (switched on, plus their dependencies) use: the licences
    that apply (for the page to show; recording is automatic) and the weights still to download."""
    import workflow as wf

    settings = _settings(settings)
    accepted = accepted_licences()
    licences: list = []
    seen_licences: set = set()
    missing: list = []
    seen_files: set = set()
    for stage in wf.effective_enabled(workflow):
        params = workflow["stages"][stage]["params"]
        for f in stage_files(stage, params, settings):
            lic = licence_of(f)
            if lic and lic not in seen_licences:
                seen_licences.add(lic)
                licences.append({**_licence_view(lic, accepted), "stage": stage})
            if not installed(f) and f not in seen_files:
                seen_files.add(f)
                missing.append({"file": f, "mb": CATALOGUE[f]["size_mb"], "stage": stage})
    return {"licences": licences, "models_needed": missing}


def default_features(settings: Optional[dict] = None) -> list:
    """The features the default template's enabled stages need (what `models --download` with no
    names fetches)."""
    import workflow as wf

    flow = wf.resolve()
    out: list = []
    for stage in wf.effective_enabled(flow):
        for fid in stage_features(stage, flow["stages"][stage]["params"]):
            if fid not in out:
                out.append(fid)
    return out


# ---------------------------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------------------------

def _download_one(name: str, *, log_path: Path, cancel_file: Optional[Path],
                  on_progress: Optional[Callable[[float], None]]) -> None:
    entry = CATALOGUE[name]
    part_dir = config.MODELS / "_part"
    part_dir.mkdir(parents=True, exist_ok=True)
    part = part_dir / name
    stop = threading.Event()

    def watch() -> None:
        total = entry["size_mb"] * 1048576
        while not stop.wait(1.0):
            try:
                on_progress(min(part.stat().st_size / total, 0.99))
            except OSError:
                pass

    watcher = None
    if on_progress is not None:
        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
    try:
        # The CDNs drop long transfers on some networks: retry on any error, resuming each time.
        proc.run([str(_CURL), "-L", "--fail", "--retry", "30", "--retry-delay", "3",
                  "--retry-all-errors", "-C", "-", "-sS", "-o", str(part), entry["url"]],
                 log_path=log_path, timeout=6 * 3600, stall_timeout=None, cancel_file=cancel_file)
    finally:
        stop.set()
        if watcher is not None:
            watcher.join(timeout=2)
    os.replace(part, config.MODELS / name)


def download_files(files: list, *, log_path: Path, echo: Optional[Callable[[str], None]] = print,
                   cancel_file: Optional[Path] = None,
                   on_progress: Optional[Callable[[float, str], None]] = None) -> list:
    """Fetch the files that are not installed. Interrupted downloads resume. `on_progress(
    fraction of the whole, message)`. Returns the names downloaded."""
    todo = [f for f in dict.fromkeys(files) if not installed(f)]
    config.MODELS.mkdir(parents=True, exist_ok=True)
    total = sum(CATALOGUE[f]["size_mb"] for f in todo) or 1
    done_mb = 0
    done: list = []
    for index, name in enumerate(todo, start=1):
        entry = CATALOGUE[name]
        label = f"下载模型 {entry['what']}（{entry['size_mb']} MB，{index}/{len(todo)}）"
        if echo:
            echo(f"[download] {name} ({entry['size_mb']} MB) - {entry['what']}")
        if on_progress:
            on_progress(done_mb / total, label)

        def inner(fraction: float, base=done_mb, size=entry["size_mb"], text=label) -> None:
            on_progress((base + fraction * size) / total, text)

        _download_one(name, log_path=log_path, cancel_file=cancel_file,
                      on_progress=inner if on_progress else None)
        done_mb += entry["size_mb"]
        done.append(name)
        if echo:
            echo(f"[ok] {name}")
    return done


def ensure_stage_models(stage: str, params: Optional[dict], *, log_path: Path,
                        settings: Optional[dict] = None,
                        echo: Optional[Callable[[str], None]] = None,
                        cancel_file: Optional[Path] = None,
                        on_progress: Optional[Callable[[float, str], None]] = None) -> dict:
    """What a stage does before it starts: record the licences of the weights it uses (the stage
    being enabled is the person's choice to use them) and download the weights that are missing.
    Returns {"files": [...], "licences": [ids recorded now], "downloaded": [...]}."""
    settings = _settings(settings)
    files = stage_files(stage, params, settings)
    lics = [l for l in dict.fromkeys(licence_of(f) for f in files) if l]
    fresh = [l for l in lics if l not in accepted_licences()]
    if fresh:
        accept_licences(fresh, via="feature-enabled")
    downloaded = download_files(files, log_path=log_path, echo=echo, cancel_file=cancel_file,
                                on_progress=on_progress)
    return {"files": files, "licences": fresh, "downloaded": downloaded}


def download_features(features: Optional[list] = None, *, log_path: Path,
                      settings: Optional[dict] = None,
                      echo: Optional[Callable[[str], None]] = print,
                      cancel_file: Optional[Path] = None,
                      on_progress: Optional[Callable[[float, str], None]] = None) -> list:
    """Download the weights of the named features (None: those of the default template's
    enabled stages). Licences are not recorded here; that happens when a stage uses the weights.
    Returns the names downloaded. Unknown feature: ValueError."""
    settings = _settings(settings)
    if features is None:
        features = default_features(settings)
    unknown = [f for f in features if f not in FEATURES]
    if unknown:
        raise ValueError(f"unknown feature(s): {unknown}")
    files: list = []
    for fid in features:
        files.extend(f for f in feature_files(fid, settings) if f not in files)
    return download_files(files, log_path=log_path, echo=echo, cancel_file=cancel_file,
                          on_progress=on_progress)
