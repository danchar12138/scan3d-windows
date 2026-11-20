"""The scan workflow: its stages, their parameters, how they depend on each other, what is stale.

This module is the single description of what a scan job *is*. The pipeline runs it, the server
exposes it (`schema()` drives the forms in the UI, so the UI has no list of parameters of its
own) and the CLI edits it. Standard library only; no file is touched except the user's own
templates file (see "User settings").

Stages (order = execution order; `deps` must be done before a stage runs)

    id          what it does                                        deps          optional
    ingest      photos in order / frames cut from a video           -             no
    masks       paint out people (SAM) or keep the subject          ingest        yes
    sfm         camera poses                                        ingest        no
    normals     surface-normal priors (MoGe-2)                      sfm           yes
    dense_seed  dense point-cloud seed (RoMa)                       sfm           yes
    train       Gaussian splat                                      sfm           yes
    dense       COLMAP dense reconstruction (GPU, slow)             sfm           yes
    mesh        shard filter, simplify, photo texture, collision    dense         yes
    coverage    where the photos are too few                        train         yes
    export      previews, report.json, output/                      -             no

`deps` are hard: a stage that is switched on pulls them in (see `effective_enabled`). `soft`
dependencies are used when they are switched on and ignored otherwise: masks and priors for
sfm/train/dense/dense_seed, masks and the splat for mesh (the mesh is written in the splat's frame
when there is one). `export` takes whatever exists.

A workflow is a plain dict, stored per job in `runs/<job>/workflow.json`:

    {"template": "standard",
     "stages": {"ingest": {"enabled": true, "params": {...every parameter...}}, ...}}

`resolve()` builds it from a template plus overrides, `validate()` checks one, and both give
back complete dicts: every stage, every parameter. An override is the partial form
`{stage: {"enabled"?: bool, "params"?: {key: value}}}`.

Staleness. Each stage has a fingerprint: a hash of its parameters, its `version`, the input of
the job and the fingerprints of the stages it consumed. A stage whose saved record carries the
same fingerprint and whose consumed stages are all done is `done`; with another fingerprint it
is `stale`; with no record `pending`; switched off `disabled`. Changing a parameter therefore
invalidates that stage and everything after it, and nothing else.

`plan()` turns a request (run up to a stage, run only some stages, force some to run again) into
the ordered list of stages to run, or raises PlanError with a machine-readable `code`.

User settings. Templates the user saved in the UI live in `<root>/user-settings.json`
(ignored by git):  {"default_template": id, "templates": {id: {"label": str, "base": id,
"stages": {override}}}}. A user template with the id of a built-in one replaces it.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Optional

import config

STAGE_ORDER = ("ingest", "masks", "sfm", "normals", "dense_seed", "train", "dense", "mesh",
               "coverage", "export")


class WorkflowError(ValueError):
    """A workflow or an override that is not valid. `problems` lists every reason, in Chinese."""

    def __init__(self, problems: list):
        super().__init__("；".join(problems))
        self.problems = list(problems)


class PlanError(ValueError):
    """A run request that cannot be carried out. `code` is stable (the API returns it)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _param(key, label, kind, default, *, help="", minimum=None, maximum=None, choices=None,
           advanced=False):
    """One parameter. kind: bool | int | float | choice | text. A choice is
    [{"value", "label"}]; its value may be a number (0 usually meaning "no limit")."""
    entry = {"key": key, "label": label, "type": kind, "default": default, "help": help,
             "advanced": advanced}
    if minimum is not None:
        entry["min"] = minimum
    if maximum is not None:
        entry["max"] = maximum
    if choices is not None:
        entry["choices"] = [{"value": v, "label": l} for v, l in choices]
    return entry


# `features`: the keys of the network-weight features (models.py) a stage may need; which of
# them it really needs can depend on its parameters (models.stage_features decides).
STAGES = {
    "ingest": {
        "label": "整理素材", "optional": False, "gpu": False, "deps": [], "soft": [], "version": 1,
        "desc": "照片按顺序整理；视频按清晰度和分布挑出画面。",
        "params": [
            _param("min_images", "最少照片数", "int", 20, minimum=3, maximum=200,
                   help="少于这个数任务直接停止，位姿在照片太少时容易失败。"),
            _param("video_frames", "视频取帧数", "int", 480, minimum=30, maximum=1000,
                   help="视频最多挑出多少张画面（实际数量还取决于画面变化和清晰度）。照片不受影响。"),
            _param("blur_reject_px", "丢弃模糊画面（像素）", "float", 2.0, minimum=0, maximum=20,
                   help="运动模糊超过这个像素数的视频帧会被丢掉；0 表示不筛。"),
            _param("max_long_side", "图片长边上限", "choice", 3840,
                   choices=[(1920, "1920"), (2560, "2560"), (3840, "3840"), (0, "不缩小")],
                   help="更大的照片会缩小到这个长边，节省后面所有步骤的时间。"),
        ],
    },
    "masks": {
        "label": "抠图", "optional": True, "gpu": True, "deps": ["ingest"], "soft": [],
        "version": 1, "features": ["sam3", "sam2", "subject"],
        "desc": "抠除路过的人，或只保留拍摄的物体、背景不重建。位姿、训练和网格都会避开被抠掉的区域。",
        "params": [
            _param("kind", "抠什么", "choice", "moving",
                   choices=[("moving", "抠掉走动的人（场景）"), ("subject", "只留主体（物体）")]),
            _param("prompt", "要抠掉的东西", "text", "person",
                   help="英文提示词，多个用分号隔开，例如 person; dog。只对「抠掉走动的人」有效。"),
            _param("engine", "抠图模型", "choice", "sam3",
                   choices=[("sam3", "SAM 3"), ("sam2", "SAM 2.1 + Grounding DINO")],
                   help="只对「抠掉走动的人」有效。", advanced=True),
        ],
    },
    "sfm": {
        "label": "相机位姿", "optional": False, "gpu": True, "deps": ["ingest"], "soft": ["masks"],
        "version": 1, "features": ["aliked"],
        "desc": "算出每张照片的拍摄位置和镜头参数，是后面所有步骤的基础。",
        "params": [
            _param("quality", "精度", "choice", "high",
                   choices=[("extreme", "极高（很慢）"), ("high", "高"), ("medium", "中"),
                            ("low", "低（快）")]),
            _param("use_gps", "用照片里的 GPS 定比例", "bool", False,
                   help="照片带 GPS 时，据此把模型换算成米。手持拍摄的物体或室内场景的定位误差远大于拍摄范围，"
                        "会得到荒谬的比例，后面的稠密重建会失败；默认不用。", advanced=True),
            _param("learned_fallback", "失败时换学习型特征重试", "bool", True,
                   help="常规特征注册的照片太少时，改用 ALIKED + LightGlue 再算一次。",
                   advanced=True),
            _param("min_registered_fraction", "最低注册比例", "float", 0.6, minimum=0.1,
                   maximum=1.0, help="低于这个比例的照片注册成功就算失败。", advanced=True),
        ],
    },
    "normals": {
        "label": "法线先验", "optional": True, "gpu": True, "deps": ["sfm"], "soft": [],
        "version": 1, "features": ["normals_vitb", "normals_vitl"],
        "desc": "用 MoGe-2 估计每张图的表面法线，让高斯训练的表面更平整。",
        "params": [
            _param("model", "模型", "choice", "vitb",
                   choices=[("vitb", "ViT-B（400 MB，快）"), ("vitl", "ViT-L（1.3 GB，更准）")]),
        ],
    },
    "dense_seed": {
        "label": "稠密点云种子", "optional": True, "gpu": True, "deps": ["sfm"], "soft": ["masks"],
        "version": 1, "features": ["dense_seed"],
        "desc": "RoMa v2 匹配出稠密点云，作为高斯训练的初始点。实测在室外场景会变差，默认关闭。",
        "params": [
            _param("preset", "精度", "choice", "balanced",
                   choices=[("fast", "快"), ("balanced", "均衡"), ("high", "精细")]),
        ],
    },
    "train": {
        "label": "高斯训练", "optional": True, "gpu": True, "deps": ["sfm"],
        "soft": ["masks", "normals", "dense_seed"], "version": 1,
        "desc": "训练高斯泼溅，画面质量主要看这一步。",
        "params": [
            _param("max_image_size", "训练图像长边", "choice", 1920,
                   choices=[(1280, "1280"), (1600, "1600"), (1920, "1920"), (2560, "2560"),
                            (3072, "3072"), (0, "原尺寸")]),
            _param("iterations", "训练步数", "int", 30000, minimum=1000, maximum=200000,
                   help="越多越清晰，用时线性增加。"),
            _param("cap_max", "高斯数量上限", "int", 2000000, minimum=100000, maximum=10000000,
                   help="越多细节越多，显存和文件大小也越大。"),
            _param("primitive", "高斯类型", "choice", "mip",
                   choices=[("mip", "mip（抗锯齿）"), ("3dgs", "标准 3DGS")]),
            _param("eval_holdout", "留出八分之一照片评测", "bool", False,
                   help="用来比较参数，会少用八分之一的照片训练。", advanced=True),
            _param("extra_args", "追加给训练器的参数", "text", "",
                   help="空格分隔，原样传给 spirula train，例如 --floater-suppression mild。",
                   advanced=True),
        ],
    },
    "dense": {
        "label": "稠密重建", "optional": True, "gpu": True, "deps": ["sfm"], "soft": ["masks"],
        "version": 1,
        "desc": "COLMAP 多视角稠密匹配，生成网格的几何。最慢的一步，约半小时。",
        "params": [
            _param("image_size", "匹配图像长边", "choice", 1600,
                   choices=[(800, "800（快）"), (1024, "1024"), (1280, "1280"), (1600, "1600"),
                            (2000, "2000（慢）")]),
            _param("max_images", "最多用多少张照片", "int", 200, minimum=20, maximum=600,
                   help="超过的照片会均匀抽取。"),
            _param("source_images", "每张图参考几张邻图", "int", 10, minimum=4, maximum=20),
            _param("geom_consistency", "几何一致性检查", "bool", True,
                   help="多一遍几何一致性匹配，更干净，更慢。", advanced=True),
        ],
    },
    "mesh": {
        "label": "网格", "optional": True, "gpu": False, "deps": ["dense"],
        "soft": ["masks", "train"],
        "version": 1,
        "desc": "去碎片、减面、用原始照片贴图，并生成碰撞网格。改面数和贴图不需要重做稠密重建。",
        "params": [
            _param("visual_faces", "网格面数", "int", 300000, minimum=10000, maximum=2000000),
            _param("collision_faces", "碰撞网格面数", "int", 20000, minimum=1000, maximum=200000),
            _param("texture_size", "贴图尺寸", "choice", 4096,
                   choices=[(1024, "1024"), (2048, "2048"), (4096, "4096"), (8192, "8192")],
                   help="碎块太多放不下时会自动放大一档。"),
            _param("drop_shards", "去掉远处的碎片面", "bool", True, advanced=True),
        ],
    },
    "coverage": {
        "label": "拍摄覆盖", "optional": True, "gpu": False, "deps": ["train"], "soft": [],
        "version": 1,
        "desc": "标出哪些地方照片太少，并给出补拍位置。几秒钟。",
        "params": [],
    },
    "export": {
        "label": "导出", "optional": False, "gpu": False, "deps": [],
        "soft": ["train", "mesh", "coverage"], "version": 1,
        "desc": "整理输出文件，生成网页预览和报告。",
        "params": [
            _param("previews", "生成网页预览", "bool", True,
                   help="关掉可以省几十秒，网页查看器会在需要时再生成。", advanced=True),
        ],
    },
}
assert tuple(STAGES) == STAGE_ORDER

# ---------------------------------------------------------------------------------------------
# Templates (starting points; everything can be changed per job)
# ---------------------------------------------------------------------------------------------

BASE_ENABLED = {"ingest": True, "sfm": True, "export": True, "train": True, "masks": False,
                "normals": False, "dense_seed": False, "dense": False, "mesh": False,
                "coverage": False}

TEMPLATES = {
    "preview": {
        "label": "快速预览", "desc": "只出高斯，几分钟。确认素材能不能用。",
        "stages": {
            "train": {"params": {"max_image_size": 1280, "iterations": 7000, "cap_max": 500000}},
        },
    },
    "preview_mesh": {
        "label": "快速预览（含网格）", "desc": "快速预览的高斯，再用低配的稠密重建做一个小网格，用来先看网格能不能用。",
        "stages": {
            "train": {"params": {"max_image_size": 1280, "iterations": 7000, "cap_max": 500000}},
            "dense": {"enabled": True, "params": {"image_size": 800, "max_images": 60,
                                                  "source_images": 6, "geom_consistency": False}},
            "mesh": {"enabled": True, "params": {"visual_faces": 100000, "collision_faces": 10000,
                                                 "texture_size": 2048}},
        },
    },
    "standard": {
        "label": "标准（高斯）", "desc": "法线先验，200 万高斯、3 万步。不做网格。",
        "stages": {
            "normals": {"enabled": True}, "coverage": {"enabled": True},
        },
    },
    "high": {
        "label": "高质量（高斯）", "desc": "300 万高斯、5 万步，3072 像素训练。不做网格。",
        "stages": {
            "normals": {"enabled": True}, "coverage": {"enabled": True},
            "train": {"params": {"max_image_size": 3072, "iterations": 50000,
                                 "cap_max": 3000000}},
        },
    },
    "full": {
        "label": "完整（高斯 + 网格）", "desc": "标准高斯，再做稠密重建和带贴图的网格，要一个多小时。",
        "stages": {
            "normals": {"enabled": True}, "coverage": {"enabled": True},
            "dense": {"enabled": True}, "mesh": {"enabled": True},
        },
    },
    "mesh_only": {
        "label": "只要网格", "desc": "不训练高斯，只做稠密重建和网格（网格坐标是照片坐标系）。",
        "stages": {
            "train": {"enabled": False}, "dense": {"enabled": True}, "mesh": {"enabled": True},
        },
    },
}
DEFAULT_TEMPLATE = "standard"


# ---------------------------------------------------------------------------------------------
# Schema, templates, resolving, validation
# ---------------------------------------------------------------------------------------------

def schema() -> dict:
    """Everything a client needs to draw the workflow editor, JSON-able."""
    stages = []
    for sid in STAGE_ORDER:
        d = STAGES[sid]
        stages.append({"id": sid, "label": d["label"], "desc": d["desc"], "optional": d["optional"],
                       "gpu": d["gpu"], "deps": list(d["deps"]), "soft": list(d["soft"]),
                       "features": list(d.get("features", [])), "params": copy.deepcopy(d["params"])})
    return {"stages": stages, "default_template": DEFAULT_TEMPLATE}


def _defaults() -> dict:
    return {sid: {"enabled": BASE_ENABLED[sid] if sid in BASE_ENABLED else True,
                  "params": {p["key"]: p["default"] for p in STAGES[sid]["params"]}}
            for sid in STAGE_ORDER}


def _merge(workflow: dict, override: Optional[dict]) -> None:
    for sid, change in (override or {}).items():
        target = workflow["stages"].get(sid)
        if target is None or not isinstance(change, dict):
            continue   # validate() reports it
        if "enabled" in change:
            target["enabled"] = change["enabled"]
        target["params"].update(change.get("params") or {})


def load_user() -> dict:
    """The user's saved templates; {} when there is none or the file is damaged."""
    try:
        with open(config.ROOT / "user-settings.json", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_user(data: dict) -> None:
    path = config.ROOT / "user-settings.json"
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def templates(user: Optional[dict] = None) -> list:
    """[{id, label, desc, builtin, base?}] the built-in templates, then the user's; a user
    template with a built-in id replaces it."""
    user = load_user() if user is None else user
    out = {tid: {"id": tid, "label": t["label"], "desc": t["desc"], "builtin": True}
           for tid, t in TEMPLATES.items()}
    for tid, t in (user.get("templates") or {}).items():
        out[tid] = {"id": tid, "label": t.get("label") or tid, "desc": t.get("desc", ""),
                    "builtin": False, "base": t.get("base", "standard")}
    return list(out.values())


def resolve(template: Optional[str] = None, overrides: Optional[dict] = None, *,
            user: Optional[dict] = None) -> dict:
    """The complete workflow for a template (default: the user's default or `standard`) with
    `overrides` applied. Raises WorkflowError."""
    user = load_user() if user is None else user
    template = template or user.get("default_template") or DEFAULT_TEMPLATE
    saved = (user.get("templates") or {}).get(template)
    if template not in TEMPLATES and saved is None:
        raise WorkflowError([f"没有这个模板：{template}"])
    workflow = {"template": template, "stages": _defaults()}
    chain = []
    seen = set()
    cur = template
    while cur is not None and cur not in seen:       # user template -> its base -> ... -> built-in
        seen.add(cur)
        entry = (user.get("templates") or {}).get(cur)
        if entry is not None:
            chain.append(entry.get("stages"))
            cur = entry.get("base") if entry.get("base") != cur else None
            if cur is not None and cur not in TEMPLATES and cur not in (user.get("templates") or {}):
                cur = "standard"
        else:
            chain.append(TEMPLATES[cur]["stages"])
            cur = None
    for stages in reversed(chain):
        _merge(workflow, stages)
    _merge(workflow, overrides)
    return validate(workflow, overrides=overrides)


def validate(workflow: dict, *, overrides: Optional[dict] = None) -> dict:
    """Check types, ranges and choices of a workflow (and of the override it was built from,
    for unknown names); returns it with values coerced (an int given as 3.0 becomes 3).
    Raises WorkflowError listing every problem."""
    problems = []
    for sid, change in (overrides or {}).items():
        if sid not in STAGES:
            problems.append(f"没有这个阶段：{sid}")
        elif not isinstance(change, dict):
            problems.append(f"{STAGES[sid]['label']}：设置必须是对象")
        else:
            for key in change.get("params") or {}:
                if key not in {p["key"] for p in STAGES[sid]["params"]}:
                    problems.append(f"{STAGES[sid]['label']}：没有这个参数 {key}")
            for key in change:
                if key not in ("enabled", "params"):
                    problems.append(f"{STAGES[sid]['label']}：不认识 {key}")
    stages = workflow.get("stages")
    if not isinstance(stages, dict) or set(stages) != set(STAGE_ORDER):
        raise WorkflowError(problems + ["流程里的阶段不完整"])
    for sid in STAGE_ORDER:
        st, d = stages[sid], STAGES[sid]
        if not isinstance(st.get("enabled"), bool):
            problems.append(f"{d['label']}：开关必须是 true 或 false")
        if not d["optional"] and st.get("enabled") is False:
            problems.append(f"{d['label']}不能关闭")
        params = st.get("params")
        if not isinstance(params, dict):
            problems.append(f"{d['label']}：参数缺失")
            continue
        for p in d["params"]:
            key, label = p["key"], f"{d['label']}·{p['label']}"
            if key not in params:
                problems.append(f"{label}：缺少")
                continue
            value, kind = params[key], p["type"]
            if kind == "bool":
                if not isinstance(value, bool):
                    problems.append(f"{label}：必须是开或关")
            elif kind == "choice":
                if isinstance(value, bool) or value not in [c["value"] for c in p["choices"]]:
                    problems.append(f"{label}：不在可选项里")
            elif kind in ("int", "float"):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
                    problems.append(f"{label}：必须是数字")
                    continue
                if kind == "int":
                    if value != int(value):
                        problems.append(f"{label}：必须是整数")
                        continue
                    params[key] = value = int(value)
                if ("min" in p and value < p["min"]) or ("max" in p and value > p["max"]):
                    problems.append(f"{label}：要在 {p.get('min')} 到 {p.get('max')} 之间")
            elif kind == "text" and not isinstance(value, str):
                problems.append(f"{label}：必须是文字")
    if problems:
        raise WorkflowError(problems)
    return workflow


def fill_defaults(workflow: dict) -> dict:
    """A copy of a stored workflow with the stages and parameters it lacks set to their
    defaults (a workflow saved before a parameter was added stays usable; its fingerprints
    change only if the new parameter does not hold its default meaning). Anything else is
    left for `validate` to judge."""
    out = copy.deepcopy(workflow)
    stages = out.get("stages")
    if not isinstance(stages, dict):
        return out
    for sid, base in _defaults().items():
        st = stages.get(sid)
        if st is None:
            stages[sid] = base
        elif isinstance(st, dict) and isinstance(st.get("params"), dict):
            for key, value in base["params"].items():
                st["params"].setdefault(key, value)
    return out


def apply_overrides(workflow: dict, overrides: dict) -> dict:
    """A copy of `workflow` with the partial `overrides` applied and checked."""
    out = copy.deepcopy(workflow)
    _merge(out, overrides)
    return validate(out, overrides=overrides)


def changed_from_template(workflow: dict) -> dict:
    """The partial override that turns the job's template into this workflow (for display and
    for saving as a new template)."""
    base = resolve(workflow.get("template"))
    diff = {}
    for sid in STAGE_ORDER:
        a, b = workflow["stages"][sid], base["stages"][sid]
        entry = {}
        if a["enabled"] != b["enabled"]:
            entry["enabled"] = a["enabled"]
        params = {k: v for k, v in a["params"].items() if v != b["params"].get(k)}
        if params:
            entry["params"] = params
        if entry:
            diff[sid] = entry
    return diff


# ---------------------------------------------------------------------------------------------
# Dependencies, fingerprints, states, plans
# ---------------------------------------------------------------------------------------------

def effective_enabled(workflow: dict) -> list:
    """Stage ids that will really run when everything is requested: the switched-on ones, the
    stages that cannot be switched off, and the hard dependencies of all of those. In order."""
    on = {sid for sid in STAGE_ORDER
          if workflow["stages"][sid]["enabled"] or not STAGES[sid]["optional"]}
    changed = True
    while changed:
        changed = False
        for sid in list(on):
            for dep in STAGES[sid]["deps"]:
                if dep not in on:
                    on.add(dep)
                    changed = True
    return [sid for sid in STAGE_ORDER if sid in on]


def consumed(workflow: dict, stage: str) -> list:
    """The stages `stage` reads from: its hard dependencies and the soft ones that are on."""
    on = set(effective_enabled(workflow))
    return [d for d in STAGES[stage]["deps"] + STAGES[stage]["soft"] if d in on]


def downstream(workflow: dict, stage: str) -> list:
    """`stage` and every effective-enabled stage that (directly or not) consumes it."""
    out = {stage}
    for sid in effective_enabled(workflow):
        if any(d in out for d in consumed(workflow, sid)):
            out.add(sid)
    return [sid for sid in STAGE_ORDER if sid in out]


def fingerprints(workflow: dict, input_signature: str = "") -> dict:
    """{stage: hex} for the effective-enabled stages. `input_signature` identifies the job's
    input (the pipeline passes names, sizes and times of the source files)."""
    result = {}
    for sid in effective_enabled(workflow):
        material = {"stage": sid, "version": STAGES[sid]["version"],
                    "params": workflow["stages"][sid]["params"],
                    "input": input_signature if sid == "ingest" else "",
                    "from": {d: result[d] for d in consumed(workflow, sid)}}
        result[sid] = hashlib.sha1(json.dumps(material, sort_keys=True,
                                              ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    return result


def stage_states(workflow: dict, records: dict, input_signature: str = "") -> dict:
    """{stage: "disabled" | "pending" | "done" | "stale"} from the saved stage records
    ({stage: {"fingerprint": ...}}). A stage is done only when its record is current and every
    stage it consumed is done. `export` takes whatever exists, so it is done when its record is
    current and the stages that were done when it ran (record["inputs"], a list of ids) are
    exactly the ones that are done now."""
    fps = fingerprints(workflow, input_signature)
    states = {}
    for sid in STAGE_ORDER:
        if sid not in fps:
            states[sid] = "disabled"
            continue
        record = records.get(sid)
        used = consumed(workflow, sid)
        if not record:
            states[sid] = "pending"
        elif record.get("fingerprint") != fps[sid]:
            states[sid] = "stale"
        elif sid == "export":
            now = sorted(d for d in used if states[d] == "done")
            states[sid] = "done" if sorted(record.get("inputs") or []) == now else "stale"
        elif any(states[d] != "done" for d in used):
            states[sid] = "stale"
        else:
            states[sid] = "done"
    return states


def plan(workflow: dict, states: dict, *, to: Optional[str] = None,
         only: Optional[Iterable[str]] = None, force: Iterable[str] = ()) -> list:
    """The stages to run, in order, as [{"stage", "reason"}]; reason is "requested", "needed"
    (an upstream stage of the target that is not done), "forced" or "downstream" (of a forced
    stage).

    to      run this stage and whatever it needs that is not done, then `export`; None =
            everything switched on that is not done. A stage that is not switched on cannot be
            a target.
    only    run exactly these stages (again, even if done); what they consume (hard
            dependencies and the soft ones that are switched on) must be done already or be
            among them.
    force   run these again even when done, and every stage that consumes them if that stage
            is part of the run anyway.
    Raises PlanError: unknown_stage, stage_disabled, needs_upstream, nothing_to_do.
    """
    enabled = effective_enabled(workflow)
    force = list(force or ())
    only = list(only) if only else []
    for sid in force + ([to] if to else []) + only:
        if sid not in STAGES:
            raise PlanError("unknown_stage", f"没有这个阶段：{sid}")
        if sid not in enabled:
            raise PlanError("stage_disabled", f"「{STAGES[sid]['label']}」没有打开，先把它打开")
    reasons: dict = {}
    if only:
        for sid in only:
            reasons[sid] = "requested"
    elif to:
        todo, closure = [to], set()
        while todo:
            sid = todo.pop()
            if sid not in closure:
                closure.add(sid)
                todo.extend(consumed(workflow, sid))
        for sid in enabled:
            if sid in closure:
                reasons[sid] = "requested" if sid == to else "needed"
        if "export" in enabled:
            reasons.setdefault("export", "needed")
    else:
        for sid in enabled:
            reasons[sid] = "requested"
    for sid in force:
        reasons[sid] = "forced"
    for sid in force:
        for dn in downstream(workflow, sid):
            if dn in reasons and reasons[dn] != "forced":
                reasons[dn] = "downstream"
    steps = []
    for sid in STAGE_ORDER:
        if sid not in reasons:
            continue
        reason = reasons[sid]
        if states.get(sid) == "done" and reason in ("requested", "needed") and sid not in only:
            # `export` takes what exists: when a stage it lists runs now, it is out of date by
            # the end of this run even though it is current at this moment.
            if not (sid == "export" and not only
                    and any(s["stage"] in consumed(workflow, "export") for s in steps)):
                continue
        steps.append({"stage": sid, "reason": reason})
    ahead = {step["stage"] for step in steps}
    for step in steps:
        if step["stage"] == "export":
            continue        # takes whatever exists
        # What a stage consumes (hard dependencies and the soft ones that are switched on) has
        # to be done, or run before it in this plan: a stage computed from a missing or
        # out-of-date input would claim to be current once that input shows up.
        for dep in consumed(workflow, step["stage"]):
            if states.get(dep) != "done" and dep not in ahead:
                raise PlanError("needs_upstream",
                                f"「{STAGES[step['stage']]['label']}」需要先完成「{STAGES[dep]['label']}」")
    if not steps:
        raise PlanError("nothing_to_do", "要求的阶段都已经完成了")
    return steps


def stage_label(stage: str) -> str:
    return STAGES[stage]["label"]
