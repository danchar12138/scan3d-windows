"""The scan pipeline: photos or video in, a Gaussian splat and/or meshes out, as separate stages.

The stages, their parameters, dependencies and fingerprints are described in workflow.py. This
module runs them. Every stage is a function that reads only what it declares (files in the job
folder and the records of the stages it consumes) and writes only its own outputs; no stage calls
another. `run_job` asks `workflow.plan` which stages to run, and runs them in order.

Job folder `runs/<job>/`

    workflow.json        the resolved workflow (what to run and with which parameters)
    input.json           where the photos/video are (written when a run starts)
    stages/<id>.json     {"fingerprint", "result", "finished_at", "seconds", "warnings", "inputs"?}
                         written when a stage finishes; a stage that runs again deletes its own
                         record first; `export` also records "inputs", the stages that were done
    status.json          state (created/queued/running/done/failed/cancelled), stage (running),
                         message, stage_progress, error, warnings, outputs, run {"to", "only",
                         "force", "plan"} and stages {id: {"state", "progress", "message",
                         "seconds", "error", "summary"}} for every stage
    logs/                pipeline.log and one log per stage that wrote one (masks.log, sfm.log, ...)
    dataset/             images/ names.json (ingest), masks/ (masks), sparse/ features/ matches.bin
                         (sfm), normals/ (normals), dense/roma.ply (dense_seed)
    train/run/           the trainer's run (train)
    dense/               dense_mesh.ply + the cameras it ran on (dense)
    output/              mesh.glb collision.glb (mesh), coverage.json (coverage), splat.ply,
                         preview*.spz, report.json, scene_transform.json (export)

Public helpers (used by the server; every one works from the job folder alone)

    job_view(job_dir) -> {"workflow", "template", "stages", "run"}
        The stored workflow (workflow.json; when it is missing or damaged, `workflow.resolve()`
        without writing anything), and the per-stage map: status.json's while a run is active,
        otherwise computed from the records with `workflow.stage_states` (a stage that failed in
        the last run keeps showing "failed" with its error until it runs again or is reset).
    write_workflow(job_dir, workflow)
        Validate and store the workflow atomically (workflow.WorkflowError when invalid).
    reset_stage(job_dir, stage)
        Delete the stage's record and what only it produced; stages after it then show "stale".
        ValueError for an unknown stage.
    build_workflow(job_dir, template, enable, disable, sets) -> workflow
        What the command line does with a job's workflow (see `cmd_run`); saves it.
    run_job(input_path, name, *, to, only, force, fresh, workflow, runs_dir, echo) -> exit code
        0 done, 1 failed, 2 cancelled. workflow.WorkflowError / workflow.PlanError (bad request)
        are raised before the job's status is touched.
    stage_table(job_dir) -> [(id, label, state, summary text)] for `scan3d.py stages`.
"""
from __future__ import annotations

import ctypes
import datetime as _dt
import hashlib
import json
import os
import re
import shlex
import shutil
import struct
import time
import traceback
from pathlib import Path
from typing import Callable, Optional

import config
import models
import proc
import workflow as wf
from config import IMAGE_EXT, TOOL_PATHS, UNSUPPORTED_IMAGE_EXT, VIDEO_EXT


class PipelineError(RuntimeError):
    """A failure with a message meant for the person using the product."""


# --------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------

def _now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _natural_key(path: Path):
    """Sort img2 before img10, the order the photos were taken in."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def _write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    # A reader may hold the file open for an instant; Windows then refuses the replace.
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))
    os.replace(tmp, path)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = ctypes.c_void_p
    handle = kernel32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    code = ctypes.c_ulong()
    kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle(handle)
    return bool(ok) and code.value == 259  # STILL_ACTIVE


def image_size(path: Path) -> Optional[tuple]:
    """Width and height from a JPEG or PNG header, or None for other formats."""
    try:
        with open(path, "rb") as f:
            head = f.read(26)
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                return struct.unpack(">II", head[16:24])
            if head[:2] != b"\xff\xd8":
                return None
            f.seek(2)
            while True:
                marker = f.read(4)
                if len(marker) < 4 or marker[0] != 0xFF:
                    return None
                kind, length = marker[1], struct.unpack(">H", marker[2:4])[0]
                # Start-of-frame markers carry the dimensions; C4/C8/CC are not frames.
                if 0xC0 <= kind <= 0xCF and kind not in (0xC4, 0xC8, 0xCC):
                    data = f.read(5)
                    height, width = struct.unpack(">HH", data[1:5])
                    return width, height
                f.seek(length - 2, os.SEEK_CUR)
    except (OSError, struct.error):
        return None


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)  # instant and takes no extra space on the same drive
    except OSError:
        shutil.copy2(src, dst)


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


# --------------------------------------------------------------------------------------------
# What a job folder holds: workflow, input, records, outputs
# --------------------------------------------------------------------------------------------

def _export_names() -> list:
    import splatconv

    return ["splat.ply", splatconv.PREVIEW_NAME, splatconv.PREVIEW_LITE_NAME, "report.json",
            "scene_transform.json"]


def stage_outputs(job_dir: Path, stage: str) -> list:
    """The paths only this stage produces (what resetting or re-running it deletes)."""
    d, out = job_dir / "dataset", job_dir / "output"
    return {
        "ingest": [d / "images", d / "names.json"],
        "masks": [d / "masks", d / "masks_raw"],
        "sfm": [d / "sparse", d / "features", d / "matches.bin"],
        "normals": [d / "normals"],
        "dense_seed": [d / "dense"],
        "train": [job_dir / "train"],
        "dense": [job_dir / "dense", job_dir / "dense.part", job_dir / "dense_work"],
        "mesh": [out / "mesh.glb", out / "collision.glb", job_dir / "mesh"],
        "coverage": [out / "coverage.json"],
        "export": [out / n for n in _export_names()],
    }[stage]


def read_records(job_dir: Path) -> dict:
    """{stage: record} of the stages that finished (damaged or foreign files are ignored)."""
    out = {}
    for sid in wf.STAGE_ORDER:
        rec = _read_json(Path(job_dir) / "stages" / f"{sid}.json")
        if isinstance(rec, dict) and isinstance(rec.get("fingerprint"), str) \
                and isinstance(rec.get("result"), dict):
            out[sid] = rec
    return out


def find_input(job_dir: Path) -> Optional[Path]:
    """Where the job's photos or video are: what the last run was given, else what the server
    recorded for the job, else its upload folder."""
    job_dir = Path(job_dir)
    for source in (_read_json(job_dir / "input.json"), _read_json(job_dir / "server-job.json")):
        if isinstance(source, dict):
            value = source.get("path") or source.get("input_path")
            if isinstance(value, str) and value:
                return Path(value)
    upload = job_dir / "upload"
    return upload if upload.exists() else None


def input_signature(path: Optional[Path]) -> str:
    """Identifies the job's input: the names, sizes and times of its files."""
    if path is None or not Path(path).exists():
        return "missing"
    path = Path(path)
    items = []
    files = [path] if path.is_file() else [p for p in path.rglob("*")
                                           if p.is_file() and not p.name.startswith(".")]
    for p in sorted(files, key=lambda p: p.as_posix().lower()):
        try:
            st = p.stat()
        except OSError:
            continue
        rel = p.name if path.is_file() else p.relative_to(path).as_posix()
        items.append(f"{rel}|{st.st_size}|{st.st_mtime_ns}")
    return hashlib.sha1("\n".join(items).encode("utf-8")).hexdigest()[:16]


def load_workflow(job_dir: Path) -> dict:
    """The job's workflow.json, checked; `workflow.resolve()` (not saved) when there is none or
    it is damaged."""
    saved = _read_json(Path(job_dir) / "workflow.json")
    if isinstance(saved, dict):
        try:
            return wf.validate(wf.fill_defaults(saved))
        except wf.WorkflowError:
            pass
    return wf.resolve()


def write_workflow(job_dir, workflow: dict) -> None:
    """Validate and store a workflow (atomic). Raises workflow.WorkflowError."""
    _write_json_atomic(Path(job_dir) / "workflow.json", wf.validate(workflow))


def _summary(stage: str, result: dict) -> dict:
    """The few facts of a result the UI shows on the stage's card."""
    keys = {
        "ingest": ("images", "videos"), "masks": ("masked_percent", "mask_kind"),
        "sfm": ("registered", "images"), "normals": ("images",), "dense_seed": ("dense_points",),
        "train": ("gaussians", "iterations"), "dense": ("dense_points", "photos", "dense_photos_dropped"),
        "mesh": ("mesh_faces", "collision_faces"), "coverage": ("regions",), "export": (),
    }[stage]
    return {k: result[k] for k in keys if result.get(k) is not None}


def stage_map(job_dir: Path, workflow: dict, records: Optional[dict] = None,
              signature: Optional[str] = None, previous: Optional[dict] = None) -> dict:
    """The per-stage map of status.json computed from the records. `previous` is an earlier map:
    a stage that failed there keeps its failure until it has a current record."""
    job_dir = Path(job_dir)
    records = read_records(job_dir) if records is None else records
    if signature is None:
        signature = input_signature(find_input(job_dir))
    states = wf.stage_states(workflow, records, signature)
    out = {}
    for sid in wf.STAGE_ORDER:
        rec, state = records.get(sid), states[sid]
        entry = {"state": state, "progress": None, "message": None,
                 "seconds": rec.get("seconds") if rec and state in ("done", "stale") else None,
                 "error": None,
                 "summary": _summary(sid, rec["result"]) if rec and state in ("done", "stale")
                 else {}}
        old = (previous or {}).get(sid)
        if isinstance(old, dict) and old.get("state") == "failed" \
                and state in ("pending", "stale"):
            entry.update(state="failed", error=old.get("error"), message=old.get("message"))
        out[sid] = entry
    return out


def job_view(job_dir) -> dict:
    """{"workflow", "template", "stages", "run"} of a job (see the module docstring)."""
    job_dir = Path(job_dir)
    flow = load_workflow(job_dir)
    status = _read_json(job_dir / "status.json")
    status = status if isinstance(status, dict) else {}
    state = status.get("state")
    active = state in ("queued", "running") and isinstance(status.get("stages"), dict)
    if active and state == "running" and isinstance(status.get("pid"), int) \
            and not _pid_alive(status["pid"]):
        active = False                      # the run died; the server marks it interrupted
    previous = status.get("stages") if isinstance(status.get("stages"), dict) else None
    stages = status["stages"] if active else stage_map(job_dir, flow, previous=previous)
    run = status.get("run")
    if not isinstance(run, dict):
        run = {"to": None, "only": [], "force": [], "plan": []}
    return {"workflow": flow, "template": flow.get("template"), "stages": stages, "run": run}


def outputs_of(job_dir: Path) -> dict:
    """The viewable files of a job: {"splat", "mesh", "collision", "preview", "preview_lite",
    "report"} for those that exist."""
    import splatconv

    out = Path(job_dir) / "output"
    wanted = {"splat": "splat.ply", "mesh": "mesh.glb", "collision": "collision.glb",
              "preview": splatconv.PREVIEW_NAME, "preview_lite": splatconv.PREVIEW_LITE_NAME,
              "report": "report.json"}
    return {k: f"output/{v}" for k, v in wanted.items() if (out / v).is_file()}


def reset_stage(job_dir, stage: str) -> None:
    """Delete a stage's record and the files only it produced. The stages after it become stale
    on their own (their fingerprints chain through this one)."""
    if stage not in wf.STAGES:
        raise ValueError(f"unknown stage: {stage}")
    job_dir = Path(job_dir)
    (job_dir / "stages" / f"{stage}.json").unlink(missing_ok=True)
    for path in stage_outputs(job_dir, stage):
        _remove(path)
    status = _read_json(job_dir / "status.json")
    if isinstance(status, dict):
        status["outputs"] = outputs_of(job_dir)
        if isinstance(status.get("stages"), dict) and stage in status["stages"]:
            previous = {k: v for k, v in status["stages"].items() if k != stage}
            status["stages"] = stage_map(job_dir, load_workflow(job_dir), previous=previous)
        _write_json_atomic(job_dir / "status.json", status)


def build_workflow(job_dir, template: Optional[str] = None, enable=(), disable=(),
                   sets=()) -> dict:
    """The workflow a command line asks for, saved in the job folder.

    With no workflow.json the template (default: the user's default or `standard`) is resolved;
    with one it is read, unless `template` is given, which starts again from that template.
    `enable` / `disable` are stage ids; `sets` are "stage.param=value" (value as JSON, else text).
    Raises workflow.WorkflowError."""
    job_dir = Path(job_dir)
    path = job_dir / "workflow.json"
    saved = _read_json(path)
    if template or not isinstance(saved, dict):
        flow = wf.resolve(template)
    else:
        flow = wf.validate(wf.fill_defaults(saved))
    over: dict = {}
    problems = []
    for sid in enable:
        over.setdefault(sid, {})["enabled"] = True
    for sid in disable:
        over.setdefault(sid, {})["enabled"] = False
    for item in sets:
        key, sep, value = item.partition("=")
        stage, dot, param = key.partition(".")
        if not sep or not dot or not stage or not param:
            problems.append(f"--set 要写成 阶段.参数=值，收到 {item!r}")
            continue
        try:
            parsed = json.loads(value)
        except ValueError:
            parsed = value
        over.setdefault(stage, {}).setdefault("params", {})[param] = parsed
    if problems:
        raise wf.WorkflowError(problems)
    if over:
        flow = wf.apply_overrides(flow, over)
    changed = over or template or not isinstance(saved, dict) or saved != flow
    if changed:
        write_workflow(job_dir, flow)
    return flow


# --------------------------------------------------------------------------------------------
# Job state
# --------------------------------------------------------------------------------------------

class Job:
    """One run of a job: its workflow, its status.json and the stage records."""

    def __init__(self, name: str, *, runs_dir: Optional[Path] = None,
                 echo: Optional[Callable[[str], None]] = print, workflow: Optional[dict] = None,
                 input_path: Optional[Path] = None):
        self.name = name
        self.dir = Path(runs_dir if runs_dir is not None else config.RUNS) / name
        self.logs = self.dir / "logs"
        self.stages_dir = self.dir / "stages"
        self.cancel_file = self.dir / "cancel"
        self.status_path = self.dir / "status.json"
        self.echo = echo
        for d in (self.dir, self.logs, self.stages_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.workflow = workflow if workflow is not None else load_workflow(self.dir)
        self.settings = config.load_settings()
        self.input_path = Path(input_path) if input_path is not None else find_input(self.dir)
        self.signature = input_signature(self.input_path)
        self.fingerprints = wf.fingerprints(self.workflow, self.signature)
        self.current: Optional[str] = None
        # Warnings belong to the stage that gave them and live in its record.
        self.stage_warnings: dict = {sid: list(rec.get("warnings") or [])
                                     for sid, rec in read_records(self.dir).items()}
        old = _read_json(self.status_path)
        self.status = dict(old) if isinstance(old, dict) else {}
        self.status.update({"job": name, "state": "running", "stage": None,
                            "stage_progress": None, "message": "", "started_at": _now(),
                            "updated_at": _now(), "finished_at": None, "error": None,
                            "pid": os.getpid()})
        for gone in ("stage_index", "stage_count", "quality"):
            self.status.pop(gone, None)
        self.overlay: dict = {}
        self._last_echo_at = 0.0

    # -- log and status

    @property
    def warnings(self) -> list:
        return [w for sid in wf.STAGE_ORDER for w in self.stage_warnings.get(sid, [])]

    def log(self, text: str) -> None:
        line = f"[{_dt.datetime.now().strftime('%H:%M:%S')}] {text}"
        with open(self.logs / "pipeline.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if self.echo:
            self.echo(line)

    def update(self, **fields) -> None:
        self.status.update(fields)
        self.status["updated_at"] = _now()
        self.status["warnings"] = self.warnings
        _write_json_atomic(self.status_path, self.status)

    def publish_stages(self) -> None:
        """Recompute the per-stage map from the records, with the run's queued / running /
        failed stages on top of it, and write status.json."""
        records = read_records(self.dir)
        stages = stage_map(self.dir, self.workflow, records, self.signature)
        for sid, over in self.overlay.items():
            stages[sid].update(over)
        self.status["stages"] = stages
        self.update()

    def set_overlay(self, stage: str, **fields) -> None:
        self.overlay[stage] = fields

    def progress(self, fraction: Optional[float], message: Optional[str] = None) -> None:
        value = None if fraction is None else round(min(max(fraction, 0), 1), 4)
        fields = {"stage_progress": value}
        if message is not None:
            fields["message"] = message
        entry = self.overlay.get(self.current) if self.current else None
        if entry is not None:
            entry["progress"] = value
            if message is not None:
                entry["message"] = message
            stage_entry = (self.status.get("stages") or {}).get(self.current)
            if stage_entry is not None:
                stage_entry.update(entry)
        self.update(**fields)
        # Console feedback for command-line runs, at most one line every 20 seconds.
        text = self.status.get("message") or ""
        if self.echo and text and time.monotonic() - self._last_echo_at > 20:
            self._last_echo_at = time.monotonic()
            self.echo(f"    {text}")

    def warn(self, text: str) -> None:
        self.stage_warnings.setdefault(self.current or "ingest", []).append(text)
        self.log("警告: " + text)

    def check_cancel(self) -> None:
        if self.cancel_file.exists():
            raise proc.Cancelled("cancelled by user")

    # -- records

    def record(self, stage: str) -> Optional[dict]:
        return read_records(self.dir).get(stage)

    def result(self, stage: str) -> dict:
        """The result of a finished stage this one consumes; PipelineError when it has none."""
        rec = self.record(stage)
        if rec is None:
            raise PipelineError(f"「{wf.stage_label(stage)}」还没有完成，这一步需要它的结果")
        return rec["result"]

    def uses(self, stage: str, dep: str) -> bool:
        """True when `stage` consumes `dep` in this workflow (a soft dependency that is on)."""
        return dep in wf.consumed(self.workflow, stage)

    def params(self, stage: str) -> dict:
        return self.workflow["stages"][stage]["params"]

    def states(self) -> dict:
        return wf.stage_states(self.workflow, read_records(self.dir), self.signature)

    def save_record(self, stage: str, result: dict, seconds: float,
                    inputs: Optional[list] = None) -> None:
        rec = {"fingerprint": self.fingerprints[stage], "result": result,
               "finished_at": _now(), "seconds": round(seconds, 1),
               "warnings": self.stage_warnings.get(stage, [])}
        if inputs is not None:
            rec["inputs"] = inputs
        _write_json_atomic(self.stages_dir / f"{stage}.json", rec)

    def delete_record(self, stage: str) -> None:
        (self.stages_dir / f"{stage}.json").unlink(missing_ok=True)

    def clean_outputs(self, stage: str) -> None:
        for path in stage_outputs(self.dir, stage):
            _remove(path)

    # -- tools

    def tool(self, name: str, args: list, *, log: str, timeout: float, **kw) -> proc.RunResult:
        cmd = [str(TOOL_PATHS[name])] + [str(a) for a in args]
        return proc.run(cmd, log_path=self.logs / log, timeout=timeout,
                        cancel_file=self.cancel_file, **kw)


class GpuLock:
    """One GPU job at a time across every scan3d process on this machine."""

    def __init__(self, job: Job, wait_seconds: float):
        self.job = job
        self.path = job.dir.parent / ".gpu.lock"
        self.queue = job.dir.parent / ".gpu.queue"
        self.wait_seconds = wait_seconds
        self.held = False
        self.ticket: Optional[Path] = None

    def _first_in_line(self) -> bool:
        """Waiters take a numbered ticket and are served oldest first. Without this, a process
        that submits one job after another re-takes the lock every time and starves the rest."""
        tickets = []
        for t in self.queue.iterdir():
            try:
                pid = int(t.name.rsplit("-", 1)[1])
            except (IndexError, ValueError):
                continue
            if _pid_alive(pid):
                tickets.append(t.name)
            else:
                t.unlink(missing_ok=True)  # its owner died while waiting
        return not tickets or min(tickets) == self.ticket.name

    def __enter__(self):
        self.queue.mkdir(parents=True, exist_ok=True)
        self.ticket = self.queue / f"{time.time_ns():020d}-{os.getpid()}"
        self.ticket.touch()
        deadline = time.monotonic() + self.wait_seconds
        announced = False
        try:
            while True:
                if self._first_in_line():
                    try:
                        fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                        os.write(fd, str(os.getpid()).encode())
                        os.close(fd)
                        self.held = True
                        return self
                    except FileExistsError:
                        try:
                            owner = int(self.path.read_text().strip() or "0")
                        except (OSError, ValueError):
                            owner = 0
                        if not owner or not _pid_alive(owner):
                            # The previous owner died without cleaning up.
                            self.path.unlink(missing_ok=True)
                            continue
                if not announced:
                    self.job.log("另一个扫描任务正在使用显卡，排队等待中")
                    self.job.progress(None, "等待显卡空闲")
                    announced = True
                self.job.check_cancel()
                if time.monotonic() > deadline:
                    raise PipelineError("等待显卡空闲超时：另一个扫描任务一直没有结束")
                time.sleep(2)
        finally:
            self.ticket.unlink(missing_ok=True)

    def __exit__(self, *exc):
        if self.held:
            self.path.unlink(missing_ok=True)


# --------------------------------------------------------------------------------------------
# ingest: photos in order, frames cut from videos
# --------------------------------------------------------------------------------------------

def _video_info(job: Job, video: Path) -> dict:
    """Frame count, size and codec of a video."""
    info: dict = {}

    def on_line(line: str) -> None:
        key, _, value = line.partition("\t")
        if value:
            info[key.strip()] = value.strip()

    try:
        job.tool("spirula", ["sam", "video", "--info", video, "--lang", "en"],
                 log="ingest.log", timeout=120, on_line=on_line)
        frames, width, height = int(info["frames"]), int(info["width"]), int(info["height"])
    except (proc.ProcError, KeyError, ValueError):
        raise PipelineError(f"无法读取视频文件（文件可能损坏、没有传完，或编码不受支持）：{video.name}")
    if frames <= 0:
        raise PipelineError(f"视频没有可用的画面：{video.name}")
    return {"frames": frames, "width": width, "height": height,
            "codec": info.get("codec", "?")}


def _extract_frames(job: Job, video: Path, out_dir: Path, target: int, params: dict,
                    timeout: float) -> int:
    """Pull about `target` sharp, well-spread frames out of a video.

    Handheld video has motion blur in a good share of its frames. The extractor keeps the
    sharpest frame of each window, spaces frames by how much the view changes instead of by
    time, leaves out frames that are still blurred and frames where the camera stood still.
    """
    info = _video_info(job, video)
    skip = max(1, round(info["frames"] / max(target, 1)))
    args = ["sam", "extract", video, "-o", out_dir, "-s", skip, "-k", "-1", "--adaptive",
            "--similar", job.settings["video"]["similar_bits"], "--max-frames", int(target * 1.5),
            "--decoder", "builtin", "--lang", "en"]
    # Frames whose edges are smeared wider than this (in pixels, beyond the clip's sharpest
    # frames) are left out; the tool never drops more than two in a row. On the sample clip
    # 2 px removed 20 of 120 frames, all visibly motion-blurred.
    if params["blur_reject_px"]:
        args += ["--blur-reject", params["blur_reject_px"]]
    long_side, limit = max(info["width"], info["height"]), params["max_long_side"]
    if limit and long_side > limit:
        args += ["--scale", f"{limit / long_side:.4f}"]
    try:
        job.tool("spirula", args, log="ingest.log", timeout=timeout, stall_timeout=600)
    except proc.ProcError:
        raise PipelineError(
            f"无法解码视频 {video.name}（编码 {info['codec']}）。请在手机相机设置里把视频编码"
            f"改为 H.264（兼容性最好的那一项）后重拍，或者改用照片")
    kept = len(list(out_dir.glob("*.jpg")))
    if kept == 0:
        raise PipelineError(f"未能从视频中提取画面：{video.name}")
    job.log(f"{video.name}: 共 {info['frames']} 帧，保留 {kept} 帧清晰画面")
    return kept


def stage_ingest(job: Job, params: dict) -> dict:
    limits, timeouts = job.settings["limits"], job.settings["timeouts_seconds"]
    input_path = job.input_path
    if input_path is None or not input_path.exists():
        raise PipelineError(f"找不到输入：{input_path or '（任务没有记录素材位置）'}")
    if input_path.is_file():
        files = [input_path]
    else:
        files = [p for p in input_path.rglob("*") if p.is_file() and not p.name.startswith(".")]
    photos = sorted((p for p in files if p.suffix.lower() in IMAGE_EXT), key=_natural_key)
    unreadable = sorted((p for p in files if p.suffix.lower() in UNSUPPORTED_IMAGE_EXT),
                        key=_natural_key)
    videos = sorted((p for p in files if p.suffix.lower() in VIDEO_EXT), key=_natural_key)
    if unreadable:
        job.warn(f"{len(unreadable)} 张图片的格式不受支持，已跳过（例如 {unreadable[0].name}）。"
                 f"支持 JPG、PNG、BMP；请在手机相机设置里把照片格式设为 JPG")
    if not (photos or videos):
        raise PipelineError("输入里没有找到可用的照片或视频（支持 JPG、PNG、BMP 图片和 MP4、MOV 等视频）")

    dataset = job.dir / "dataset"
    images_dir = dataset / "images"
    images_dir.mkdir(parents=True)

    # Photos are linked, not re-encoded, so their EXIF survives: the pose estimation reads the
    # focal length and the rotation tag from it. Names become plain ASCII because phone
    # galleries are full of names the tools downstream cannot be relied on to handle.
    names: dict = {}
    if photos:
        (images_dir / "photos").mkdir()
        for i, src in enumerate(photos, start=1):
            dst = images_dir / "photos" / f"p_{i:05d}{src.suffix.lower()}"
            _link_or_copy(src, dst)
            names[f"photos/{dst.name}"] = str(src)

    # Each video gets its own folder: the pose estimation gives every folder its own camera
    # and treats a folder named as a sequence as consecutive frames.
    sequences: list = []
    video_frames = 0
    if videos:
        infos = [_video_info(job, v) for v in videos]
        total_frames = sum(i["frames"] for i in infos)
        target_total = int(params["video_frames"])
        for i, (video, info) in enumerate(zip(videos, infos), start=1):
            job.check_cancel()
            job.progress((i - 1) / len(videos), f"从视频中挑选清晰画面 {i}/{len(videos)}")
            target = max(params["min_images"], round(target_total * info["frames"] / total_frames))
            folder = f"v{i}"
            video_frames += _extract_frames(job, video, images_dir / folder, target, params,
                                            timeouts["extract"])
            sequences.append(folder)
            names[folder + "/"] = str(video)

    all_images = sorted((p for p in images_dir.rglob("*") if p.is_file()),
                        key=lambda p: (p.parent.name, _natural_key(p)))
    if len(all_images) > limits["max_images"]:
        # Keep an even spread so coverage of the scene is preserved.
        keep = limits["max_images"]
        chosen = {round(i * (len(all_images) - 1) / (keep - 1)) for i in range(keep)}
        for i, p in enumerate(all_images):
            if i not in chosen:
                p.unlink()
        job.warn(f"共 {len(all_images)} 张图片，超过上限 {keep}，已均匀抽取 {keep} 张")
        all_images = [p for i, p in enumerate(all_images) if i in chosen]
    if len(all_images) < params["min_images"]:
        raise PipelineError(
            f"只有 {len(all_images)} 张可用图片，至少需要 {params['min_images']} 张。"
            f"请围绕场景多拍一些，相邻照片之间保持大部分画面重叠；"
            f"想试照片很少的素材，可以在「整理素材」里调低「最少照片数」")

    sizes = [s for s in (image_size(p) for p in all_images[:: max(len(all_images) // 50, 1)]) if s]
    _write_json_atomic(dataset / "names.json", names)
    result = {
        "images": len(all_images), "photos": len(photos),
        "video_frames": video_frames, "videos": len(videos), "sequences": sequences,
        "skipped_unsupported": len(unreadable),
        "long_side": max((max(s) for s in sizes), default=0),
    }
    job.log(f"输入就绪：{result['images']} 张图片"
            + (f"（其中 {video_frames} 张来自视频）" if video_frames else ""))
    return result


# --------------------------------------------------------------------------------------------
# masks: paint out people, or keep only the subject
# --------------------------------------------------------------------------------------------

def _model_path(file_name: str) -> Path:
    path = config.MODELS / file_name
    if not path.exists():
        raise PipelineError(f"缺少模型文件 {file_name}，下载没有成功。详见 logs 里这一步的日志，"
                            f"或运行 setup.cmd")
    return path


def stage_masks(job: Job, params: dict) -> dict:
    """Write dataset/masks/<folder>/<image stem>.png, black where the image must be ignored.

    Two uses: paint out what moves between photos (people by default, named by a text prompt),
    or, for a scan of a single object, keep the subject and paint out everything else. Later
    stages use the folder only when they consume this stage.
    """
    ingest = job.result("ingest")
    dataset = job.dir / "dataset"
    folders = (["photos"] if ingest.get("photos") else []) + list(ingest.get("sequences", []))
    subject = params["kind"] == "subject"
    settings = job.settings["models"]
    if subject:
        model, detector = _model_path(settings["subject"]["model"]), None
    elif params["engine"] == "sam2":
        model = _model_path(settings["sam2"]["model"])
        detector = _model_path(settings["sam2"]["detector"])
    else:
        model, detector = _model_path(settings["sam3"]["model"]), None

    total = sum(len(list((dataset / "images" / f).iterdir())) for f in folders)
    done_before = 0
    kept: list = []
    label = "分离主体" if subject else "抠除移动物体"
    for folder in folders:
        job.check_cancel()
        frames = dataset / "images" / folder
        raw = dataset / "masks_raw" / folder
        shutil.rmtree(raw, ignore_errors=True)
        raw.mkdir(parents=True)
        names: dict = {}

        def on_line(line: str) -> None:
            # "frame 12: p_00013.jpg -- kept 97.3%"
            m = re.match(r"frame (\d+): (.+?) -- kept ([\d.]+)%", line)
            if m:
                names[int(m.group(1))] = m.group(2)
                kept.append(float(m.group(3)))
                job.progress((done_before + len(names)) / max(total, 1),
                             f"{label} {done_before + len(names)}/{total}")

        args = ["sam", "track", "--model", model, *models.licence_args(model.name),
                "--frames", frames, "--out", raw, "--lang", "en",
                # Every image is segmented by itself, video frames included. The tool's
                # tracking mode was tried on video and drifted from the people onto the
                # scene (it painted out the truck in the sample video, down to 9% of a
                # frame kept), while per-image masks covered the people and nothing else.
                "--mask-mode", "image"]
        if not subject:
            args += ["--text", params["prompt"]]
            if detector is not None:
                args += ["--detector", detector, *models.licence_args(detector.name)]
        job.tool("spirula", args, log="masks.log", timeout=job.settings["timeouts_seconds"]["masks"],
                 stall_timeout=900, on_line=on_line)

        # The tool numbers its output by reading order; the other tools look masks up by
        # image name.
        out = dataset / "masks" / folder
        out.mkdir(parents=True, exist_ok=True)
        for index, name in names.items():
            src = raw / f"frame_{index:05d}.png"
            if src.exists():
                os.replace(src, out / (Path(name).stem + ".png"))
        done_before += len(names)
    shutil.rmtree(dataset / "masks_raw", ignore_errors=True)

    written = sum(1 for _ in (dataset / "masks").rglob("*.png"))
    if written < total:
        raise PipelineError(f"遮罩只生成了 {written}/{total} 张，详见 logs/masks.log")
    touched = sum(1 for k in kept if k < 99.95)
    share = 100 - sum(kept) / len(kept) if kept else 0.0
    result = {"masks": True, "mask_kind": "subject" if subject else "moving", "images": total,
              "masked_images": touched, "masked_percent": round(share, 2)}
    if subject:
        job.log(f"主体分离：平均保留画面的 {100 - share:.0f}%")
    else:
        job.log(f"抠除移动物体（{params['prompt']}）：{touched}/{total} 张图片里有，"
                f"平均遮住画面的 {share:.1f}%")
        if share > 30:
            job.warn(f"抠除的区域很大（平均 {share:.0f}% 的画面）。如果被抠掉的是想保留的内容，"
                     f"请在「抠图」里修改要抠掉的东西，或关闭抠图")
    return result


# --------------------------------------------------------------------------------------------
# sfm: camera poses
# --------------------------------------------------------------------------------------------

def _run_sfm(job: Job, dataset: Path, ingest: dict, attempt: dict, params: dict) -> dict:
    """One pose-estimation run with the given feature detector and matcher."""
    for leftover in ("sparse", "features"):
        shutil.rmtree(dataset / leftover, ignore_errors=True)
    (dataset / "matches.bin").unlink(missing_ok=True)
    n = ingest["images"]
    found_models: dict = {}
    found: dict = {}

    def on_line(line: str) -> None:
        m = re.match(r"\[extract\]\s+(\d+)/(\d+)", line)
        if m:
            job.progress(0.25 * int(m.group(1)) / int(m.group(2)),
                         f"提取特征点 {m.group(1)}/{m.group(2)}")
            return
        m = re.match(r"\[match\]\s+pairs scored: (\d+)/(\d+)", line)
        if m:
            job.progress(0.25 + 0.15 * int(m.group(1)) / int(m.group(2)), "筛选可匹配的照片")
            return
        m = re.match(r"\[match\]\s+(\d+)/(\d+) pairs matched", line)
        if m:
            job.progress(0.4 + 0.2 * int(m.group(1)) / int(m.group(2)), "匹配照片")
            return
        m = re.search(r"images in the model: (\d+)", line)
        if m:
            k = int(m.group(1))
            job.progress(0.6 + 0.4 * min(k / n, 1), f"计算相机位姿 {k}/{n}")
            return
        m = re.search(r"Wrote model (\d+) \((\d+) images, (\d+) points\)", line)
        if m:
            found_models[int(m.group(1))] = (int(m.group(2)), int(m.group(3)))
            return
        m = re.search(r"([\d.]+) px mean reprojection", line)
        if m:
            found["reprojection"] = float(m.group(1))

    # The image folder is named explicitly: given the dataset folder, the tool would read the
    # masks beside it as photos too.
    args = ["sfm", "auto", dataset / "images", "-o", dataset,
            "--quality", params["quality"],
            "--features", attempt["features"], "--matcher", attempt["matcher"], "--lang", "en"]
    # Photos carry GPS that is metres off: scaling a handheld scan of a flower by it gave a model
    # 1e-9 units across, which crashes COLMAP's dense matching. The scale of a photo scan is
    # arbitrary anyway, so GPS is used only when the workflow asks for it.
    args += ["--metric-gps", "auto" if params["use_gps"] else "none"]
    # The tool adopts a `masks` folder beside the images by itself; when masks are switched
    # off the folder (left from an earlier run) must not be seen.
    if not job.uses("sfm", "masks"):
        args.append("--no-masks")
    # Learned detectors and matchers read their weights from the bundle, never the network.
    for flag, key in (("--aliked-model", "aliked_model"), ("--lightglue-model", "lightglue_model")):
        if attempt.get(key):
            args += [flag, _model_path(attempt[key])]
    for folder in ingest["sequences"]:
        args += ["--sequence", folder]
    job.progress(0, "提取特征点")
    job.tool("spirula", args, log="sfm.log", timeout=job.settings["timeouts_seconds"]["sfm"],
             stall_timeout=1800, on_line=on_line)
    registered, points = found_models.get(0, (0, 0))
    if not (dataset / "sparse" / "0").is_dir():
        registered = 0
    return {"registered": registered, "points": points, "models": len(found_models),
            "mean_reprojection_error_px": found.get("reprojection")}


def stage_sfm(job: Job, params: dict) -> dict:
    ingest = job.result("ingest")
    dataset = job.dir / "dataset"
    n = ingest["images"]
    need = max(job.params("ingest")["min_images"] // 2, int(n * params["min_registered_fraction"]))
    failure = ("相机位姿计算失败。常见原因：相邻照片重叠太少、画面模糊、场景缺少纹理"
               "（白墙、玻璃、反光面）或拍摄时场景里有大面积移动的物体")

    # The attempts in settings.json run in order until one registers enough images. The first
    # is the classical detector, which is the most precise on textured scenes; the learned
    # detector and matcher that follow (when switched on) hold up better on low texture and
    # wide viewpoint changes.
    attempts = [a for a in job.settings["sfm"]["attempts"]
                if params["learned_fallback"] or not a.get("learned")]
    result: dict = {"registered": 0}
    used = None
    for index, attempt in enumerate(attempts):
        job.check_cancel()
        label = f"{attempt['features']} + {attempt['matcher']}"
        try:
            result = _run_sfm(job, dataset, ingest, attempt, params)
        except (proc.ProcTimeout, proc.ProcStalled):
            raise
        except proc.ProcError:
            job.log(f"位姿计算（{label}）没有得到结果")
            result = {"registered": 0}
        used = label
        if result["registered"] >= need:
            break
        if index + 1 < len(attempts):
            job.log(f"位姿计算（{label}）只对上 {result['registered']}/{n} 张，换下一种特征重试")

    registered = result["registered"]
    if registered == 0:
        raise PipelineError(failure + "。详见 logs/sfm.log")
    if registered < need:
        raise PipelineError(f"{failure}（{n} 张图片里只有 {registered} 张能对上）")
    # Model 0 is the largest connected part of the capture; the others are discarded.
    if result["models"] > 1:
        job.warn(f"素材分成了 {result['models']} 个互不相连的部分，只使用了最大的一部分（{registered} 张）")
    elif registered < n:
        job.warn(f"{n - registered} 张图片未能参与重建（{registered}/{n}）")
    reprojection = result["mean_reprojection_error_px"]
    job.log(f"位姿（{used}）：注册 {registered}/{n} 张"
            + (f"，重投影误差 {reprojection:.2f} 像素" if reprojection is not None else ""))
    return {"model": "sparse/0", "images": n, "method": used, **result}


# --------------------------------------------------------------------------------------------
# normals and dense_seed: what the trainer can use beyond the photos
# --------------------------------------------------------------------------------------------

def stage_normals(job: Job, params: dict) -> dict:
    """One surface-normal map per image from a monocular network. The trainer uses them to keep
    flat, textureless surfaces (walls, floors) flat."""
    dataset = job.dir / "dataset"
    model = _model_path(job.settings["models"]["normals_" + params["model"]]["model"])

    def on_line(line: str) -> None:
        m = re.match(r"(\d+) / (\d+) images", line)
        if m:
            job.progress(int(m.group(1)) / int(m.group(2)),
                         f"估计表面法线 {m.group(1)}/{m.group(2)}")

    job.progress(0, "估计表面法线")
    job.tool("spirula", ["geometry", dataset, "--model", model, "--lang", "en"],
             log="normals.log", timeout=job.settings["timeouts_seconds"]["normals"],
             stall_timeout=900, on_line=on_line)
    if not (dataset / "normals").is_dir():
        raise PipelineError("法线图没有生成，详见 logs/normals.log")
    return {"normals": True, "model": params["model"],
            "images": sum(1 for _ in (dataset / "normals").rglob("*") if _.is_file())}


def stage_dense_seed(job: Job, params: dict) -> dict:
    """Millions of matched points triangulated with the known camera poses. The trainer starts
    from this cloud instead of the sparse pose-estimation points, so detail is in place from the
    first step."""
    dataset = job.dir / "dataset"
    model = _model_path(job.settings["models"]["dense_seed"]["model"])
    result: dict = {"dense_seed": False, "preset": params["preset"]}
    # The steps the tool reports, with the share of the time each takes.
    steps = (("Matching image pairs", 0.0, 0.75, "稠密匹配"),
             ("Refining points", 0.75, 0.85, "优化点位置"),
             ("Fusing points", 0.85, 0.95, "融合点云"),
             ("Exporting the point cloud", 0.95, 1.0, "写出点云"))

    def on_dense(line: str) -> None:
        for prefix, lo, hi, label in steps:
            if not line.startswith(prefix):
                continue
            m = re.search(r"(\d+)/(\d+)$", line) or re.search(r"(\d+)%$", line)
            if not m:
                return
            done = (int(m.group(1)) / int(m.group(2))) if m.lastindex == 2 \
                else int(m.group(1)) / 100
            job.progress(lo + (hi - lo) * done, f"稠密点云：{label}")
            return
        m = re.search(r"points: (\d+)", line)
        if m and line.startswith("Dense cloud completed"):
            result["dense_points"] = int(m.group(1))

    job.progress(0, "稠密点云：加载模型")
    job.tool("spirula", ["dense", dataset, "--checkpoint", model, "--preset", params["preset"],
                         "--use-masks", "1" if job.uses("dense_seed", "masks") else "0",
                         "--lang", "en"],
             log="dense_seed.log", timeout=job.settings["timeouts_seconds"]["dense_seed"],
             stall_timeout=1200, on_line=on_dense)
    if not (dataset / "dense" / "roma.ply").exists():
        raise PipelineError("稠密点云没有生成，详见 logs/dense_seed.log")
    result["dense_seed"] = True
    return result


# --------------------------------------------------------------------------------------------
# train: Gaussian splat
# --------------------------------------------------------------------------------------------

def ply_vertex_count(path: Path) -> Optional[int]:
    with open(path, "rb") as f:
        head = f.read(4096).split(b"end_header")[0].decode("ascii", "replace")
    m = re.search(r"element vertex (\d+)", head)
    return int(m.group(1)) if m else None


def train_splat(job_dir: Path) -> Optional[Path]:
    """The newest splat the trainer wrote, or None."""
    run = Path(job_dir) / "train" / "run"
    checkpoints = sorted(run.glob("step-*.ckpt"))
    splat = checkpoints[-1] / "splat.ply" if checkpoints else None
    return splat if splat is not None and splat.exists() else None


def stage_train(job: Job, params: dict) -> dict:
    timeouts = job.settings["timeouts_seconds"]
    dataset = job.dir / "dataset"
    train_dir = job.dir / "train"
    ingest = job.result("ingest")
    iterations = int(params["iterations"])
    masks = job.result("masks") if job.uses("train", "masks") else None
    # A scan masked down to its subject is trained with the trainer's preset for that case:
    # the masked background is trained as empty space instead of being skipped.
    train_preset = "centered-object" if masks and masks.get("mask_kind") == "subject" else "3dgs"
    args = ["train", train_preset, "--lang", "en", "--data", dataset,
            "--output-dir-prefix", train_dir, "--output-dir-name", "run",
            "--num-iterations", iterations, "--cap-max", params["cap_max"],
            # Without these the trainer keeps a web viewer open after it finishes and never
            # exits.
            "--disable-viewer", "1", "--keep-viewer-alive", "0"]
    # The working resolution is the longest image side. The trainer takes that as a divisor of
    # the source size (it need not be a whole number).
    long_side, limit = ingest.get("long_side", 0), params["max_image_size"]
    if long_side and limit and long_side > limit:
        args += ["--train-resolution-divisor", f"{long_side / limit:.4f}"]
    # The trainer would use whatever it finds in the dataset folder; state it instead.
    args += ["--load-normals", "1" if job.uses("train", "normals") else "0",
             "--load-depths", "0",
             "--load-masks", "1" if masks else "0",
             "--seed-pointcloud", "dense/roma.ply" if job.uses("train", "dense_seed") else "sparse"]
    holdout = bool(params["eval_holdout"])
    if holdout:
        # Hold out every 8th image and score the result on them. Used for comparing settings;
        # normal runs train on every image.
        args += ["--eval-mode", "interval", "--save-eval-images", "1"]
    args += ["--primitive", params["primitive"]]
    try:
        args += shlex.split(params["extra_args"])
    except ValueError as e:
        raise PipelineError(f"「追加给训练器的参数」写得不对：{e}")

    started = time.monotonic()
    state = {"step": 0}

    def on_line(line: str) -> None:
        m = re.match(r"step\s+(\d+)/(\d+)", line)
        if not m:
            return
        step, total = int(m.group(1)), int(m.group(2))
        if step - state["step"] < max(total // 200, 1) and step < total:
            return
        state["step"] = step
        elapsed = time.monotonic() - started
        remaining = elapsed / step * (total - step) if step else 0
        job.progress(step / total,
                     f"训练高斯 {step}/{total} 步，预计还需 {max(int(remaining // 60), 0) + 1} 分钟")

    job.progress(0, "加载训练数据")
    result = job.tool("spirula", args, log="train.log", timeout=timeouts["train"],
                      stall_timeout=timeouts["train_stall"], on_line=on_line,
                      done_marker=None if holdout else r"^Training complete\. Steps",
                      done_grace=120)
    splat = train_splat(job.dir)
    if splat is None:
        raise PipelineError("训练没有产出高斯文件，详见 logs/train.log")
    out = {"gaussians": ply_vertex_count(splat), "iterations": iterations,
           "train_seconds": round(result.seconds, 1),
           "size_mb": round(splat.stat().st_size / 1048576, 1)}
    metrics = train_dir / "run" / "metrics.json"
    if metrics.exists():
        try:
            out["metrics"] = json.loads(metrics.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return out


# --------------------------------------------------------------------------------------------
# dense and mesh
# --------------------------------------------------------------------------------------------

def stage_dense(job: Job, params: dict) -> dict:
    """COLMAP's dense reconstruction (the half hour), kept in dense/ for the mesh stage."""
    import meshpost

    job.progress(0, "稠密重建")
    work = job.dir / "dense_work"
    try:
        result = meshpost.build_dense(
            dataset_dir=job.dir / "dataset", out_dir=job.dir / "dense", work_dir=work,
            tools={"colmap": TOOL_PATHS["colmap"]}, log_path=job.logs / "dense.log",
            cancel_file=job.cancel_file, max_images=params["max_images"],
            image_size=params["image_size"], source_images=params["source_images"],
            geom_consistency=params["geom_consistency"],
            use_masks=job.uses("dense", "masks"),
            on_progress=lambda f, m: job.progress(f, f"稠密重建：{m}"))
    except meshpost.MeshError as e:
        raise PipelineError(str(e))
    finally:
        # Depth maps and fused points are several GB and are not needed once the surface is kept.
        shutil.rmtree(work, ignore_errors=True)
    return {"photos": result["images"], "dense_points": result["points"],
            "faces": result["faces"], "masked_photos": result["masked_photos"],
            "dense_photos_dropped": result.get("photos_dropped", 0),
            "image_size": params["image_size"]}


def _scene_up(job_dir: Path) -> list:
    """The scene's up direction in the frame of the exported files."""
    import orient

    # The pose estimation levels its model with +Z up (from the photos' EXIF rotation, the
    # camera poses and the ground plane).
    up = [0.0, 0.0, 1.0]
    transform = Path(job_dir) / "train" / "run" / "scene_transform.json"
    try:
        matrix = json.loads(transform.read_text(encoding="utf-8"))[
            "train_from_world"]["rotation"]["matrix_3x3"]
        up = orient.rotate(matrix, up)
    except (OSError, KeyError, json.JSONDecodeError):
        pass  # no transform recorded: the training frame is the dataset frame
    return up


def stage_mesh(job: Job, params: dict) -> dict:
    """From the kept dense surface: shard filter, simplify, photo texture, collision mesh, GLB.
    Written in the splat's frame when the splat is part of this workflow, else in the photos'."""
    import meshpost
    import orient

    out = job.dir / "output"
    out.mkdir(exist_ok=True)
    run_dir = job.dir / "train" / "run" if job.uses("mesh", "train") else None
    job.progress(0, "生成网格")
    try:
        result = meshpost.build_meshes(
            run_dir=run_dir, dataset_dir=job.dir / "dataset", out_dir=out,
            work_dir=job.dir / "mesh", dense=job.dir / "dense",
            tools={k: TOOL_PATHS[k] for k in ("spirula", "colmap")},
            visual_faces=params["visual_faces"], collision_faces=params["collision_faces"],
            texture_size=params["texture_size"], log_path=job.logs / "mesh.log",
            cancel_file=job.cancel_file, drop_shards_filter=params["drop_shards"],
            use_masks=job.uses("mesh", "masks"),
            on_progress=lambda f, m: job.progress(f, f"生成网格：{m}"))
    except meshpost.MeshError as e:
        raise PipelineError(str(e))
    finally:
        # The intermediate meshes are several hundred MB and are not needed once the GLBs exist.
        shutil.rmtree(job.dir / "mesh", ignore_errors=True)
    # glTF viewers and engines expect +Y up: the root node carries the rotation, the vertex
    # numbers stay equal to the splat's.
    quaternion = orient.quaternion_to_y_up(_scene_up(job.dir))
    for name in ("mesh.glb", "collision.glb"):
        orient.set_glb_root_rotation(out / name, quaternion,
                                     {"scan3d_root_rotation": "turns the scan's up direction to +Y"})
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in result.items()}


# --------------------------------------------------------------------------------------------
# coverage and export
# --------------------------------------------------------------------------------------------

def stage_coverage(job: Job, params: dict) -> dict:
    """Where the photos saw too little of the scene: output/coverage.json (see coverage.py)."""
    import coverage

    splat = train_splat(job.dir)
    if splat is None:
        raise PipelineError("找不到高斯文件，无法计算拍摄覆盖")
    sparse = job.dir / "dataset" / "sparse" / "0"
    if not (sparse / "images.bin").is_file() or not (sparse / "cameras.bin").is_file():
        raise PipelineError("找不到相机位姿文件，无法计算拍摄覆盖")
    transform = job.dir / "train" / "run" / "scene_transform.json"
    job.progress(None, "计算拍摄覆盖")
    try:
        result = coverage.compute(
            splat, sparse / "images.bin", sparse / "cameras.bin",
            transform=coverage.read_transform(transform) if transform.exists() else None,
            up=_scene_up(job.dir))
    except (coverage.CoverageError, OSError, ValueError, struct.error) as e:
        raise PipelineError(f"无法计算拍摄覆盖：{e}")
    out = job.dir / "output" / "coverage.json"
    out.parent.mkdir(exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, out)
    summary = {"regions": len(result.get("regions", []))}
    try:
        summary["voxels"] = result["voxels"]["count"]
    except (KeyError, TypeError):
        pass
    return summary


def stage_export(job: Job, params: dict, states: dict) -> dict:
    """Gather what exists: the splat, previews for the web viewer, the meshes (already in
    output/), and report.json with every stage's result. Works with any subset."""
    import orient
    import splatconv

    out = job.dir / "output"
    out.mkdir(exist_ok=True)
    job.progress(None, "导出结果")
    splat = train_splat(job.dir) if states.get("train") == "done" else None
    if splat is not None:
        splat_dst = out / "splat.ply"
        _link_or_copy(splat, splat_dst)
        transform = job.dir / "train" / "run" / "scene_transform.json"
        if transform.exists():
            shutil.copy2(transform, out / "scene_transform.json")
        # Compact copies of the splat for the in-browser viewer: the training PLY is far too
        # large to send to a phone. Losing them only costs the viewer a rebuild on demand.
        if params["previews"]:
            job.progress(None, "生成网页预览")
            try:
                splatconv.write_preview(splat_dst, out / splatconv.PREVIEW_NAME)
                splatconv.write_preview(splat_dst, out / splatconv.PREVIEW_LITE_NAME,
                                        max_gaussians=splatconv.LITE_MAX_GAUSSIANS)
            except (splatconv.PreviewError, OSError) as e:
                (out / splatconv.PREVIEW_NAME).unlink(missing_ok=True)
                (out / splatconv.PREVIEW_LITE_NAME).unlink(missing_ok=True)
                job.warn(f"网页预览文件生成失败，查看时会重新生成：{e}")

    up = _scene_up(job.dir)
    quaternion = orient.quaternion_to_y_up(up)
    orientation = {
        "up_in_file_coordinates": [round(c, 6) for c in up],
        "rotation_to_y_up_xyzw": [round(c, 9) for c in quaternion],
        "note": "splat.ply and the vertex data of both GLB files share one coordinate frame. "
                "The GLB files carry this rotation on their root node so they stand upright "
                "in a glTF viewer (+Y up). Apply the same rotation to the splat to keep it "
                "aligned with the meshes. The scale is arbitrary: photos carry no real-world "
                "size.",
    }
    records = read_records(job.dir)
    files = {k: {"path": v, "size_mb": round((job.dir / v).stat().st_size / 1048576, 1)}
             for k, v in outputs_of(job.dir).items() if k != "report"}
    if (out / "coverage.json").is_file() and states.get("coverage") == "done":
        files["coverage"] = {"path": "output/coverage.json",
                             "size_mb": round((out / "coverage.json").stat().st_size / 1048576, 1)}
    report = {
        "job": job.name, "template": job.workflow.get("template"), "finished_at": _now(),
        "warnings": job.warnings, "orientation": orientation,
        "workflow": {sid: {"enabled": job.workflow["stages"][sid]["enabled"],
                           "params": job.workflow["stages"][sid]["params"]}
                     for sid in wf.STAGE_ORDER},
        "stages": {sid: {**rec["result"], "stage_seconds": rec.get("seconds")}
                   for sid, rec in records.items() if states.get(sid) == "done"},
        "files": files,
    }
    _write_json_atomic(out / "report.json", report)
    return {"outputs": outputs_of(job.dir)}


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------

STAGE_FUNCTIONS = {
    "ingest": stage_ingest, "masks": stage_masks, "sfm": stage_sfm, "normals": stage_normals,
    "dense_seed": stage_dense_seed, "train": stage_train, "dense": stage_dense,
    "mesh": stage_mesh, "coverage": stage_coverage,
}


def _prepare_models(job: Job, stage: str) -> None:
    """Before a stage starts: record the licences of the weights it uses and download what is
    missing (no GPU needed, so not under the lock)."""
    params = job.params(stage)
    if not models.stage_files(stage, params, job.settings):
        return
    job.progress(None, "检查模型文件")
    try:
        done = models.ensure_stage_models(
            stage, params, log_path=job.logs / f"{stage}.log", settings=job.settings,
            echo=None, cancel_file=job.cancel_file, on_progress=job.progress)
    except proc.Cancelled:
        raise
    except proc.ProcError as e:
        raise PipelineError(f"下载模型失败：{e}。网络恢复后重新运行即可继续下载")
    for lic in done["licences"]:
        job.log(f"使用 {models.LICENCES[lic]['name']} 覆盖的模型，已记录该许可：{models.LICENCES[lic]['url']}")
    for name in done["downloaded"]:
        job.log(f"已下载模型 {name}")


def _run_one(job: Job, stage: str) -> None:
    """Run one stage and write its record. Raises PipelineError / proc errors."""
    job.check_cancel()
    job.current = stage
    job.stage_warnings[stage] = []
    job.delete_record(stage)           # it is not finished any more; downstream becomes stale
    job.clean_outputs(stage)
    job.set_overlay(stage, state="running", progress=None, message="开始", error=None)
    job.update(stage=stage, stage_progress=None, message=f"{wf.stage_label(stage)}")
    job.publish_stages()
    job.log(f"阶段 {stage}（{wf.stage_label(stage)}）开始")
    started = time.monotonic()
    _prepare_models(job, stage)
    timeouts = job.settings["timeouts_seconds"]
    inputs = None
    if wf.STAGES[stage]["gpu"]:
        with GpuLock(job, timeouts["gpu_lock_wait"]):
            job.progress(None, wf.stage_label(stage))
            result = STAGE_FUNCTIONS[stage](job, job.params(stage))
    elif stage == "export":
        states = job.states()
        result = stage_export(job, job.params(stage), states)
        inputs = sorted(d for d in wf.consumed(job.workflow, "export") if states[d] == "done")
    else:
        result = STAGE_FUNCTIONS[stage](job, job.params(stage))
    seconds = time.monotonic() - started
    job.save_record(stage, result, seconds, inputs)
    job.overlay.pop(stage, None)
    job.log(f"阶段 {stage} 完成，用时 {seconds:.0f} 秒")
    job.publish_stages()


def run_job(input_path: Optional[Path], name: str, *, to: Optional[str] = None,
            only=None, force=None, fresh: bool = False, workflow: Optional[dict] = None,
            runs_dir: Optional[Path] = None,
            echo: Optional[Callable[[str], None]] = print) -> int:
    """Run (or resume) a job's workflow. Returns a process exit code: 0 done, 1 failed,
    2 cancelled. A bad request raises workflow.WorkflowError / workflow.PlanError before the
    job's status is touched.

    input_path   the photos or video (None: what the job already knows, see `find_input`)
    workflow     replaces the job's workflow.json; default: read it, else the default template
    to / only / force   as in `workflow.plan`
    fresh        delete the job's records and outputs first (not its input, upload or workflow)
    """
    runs_dir = Path(runs_dir if runs_dir is not None else config.RUNS)
    job_dir = runs_dir / name
    if fresh and job_dir.exists():
        for sub in ("stages", "dataset", "train", "dense", "dense.part", "dense_work", "mesh",
                    "output", "logs"):
            _remove(job_dir / sub)
    job_dir.mkdir(parents=True, exist_ok=True)
    if input_path is not None:
        _write_json_atomic(job_dir / "input.json", {"path": str(Path(input_path).resolve())})
    if workflow is not None:
        write_workflow(job_dir, workflow)
    elif not (job_dir / "workflow.json").exists():
        write_workflow(job_dir, wf.resolve())
    job = Job(name, runs_dir=runs_dir, echo=echo, input_path=input_path)
    flow = job.workflow
    states = job.states()
    only, force = list(only or []), list(force or [])
    try:
        steps = wf.plan(flow, states, to=to, only=only, force=force)
    except wf.PlanError as e:
        if e.code != "nothing_to_do":
            raise
        steps = []
    run = {"to": to, "only": only, "force": force, "plan": [s["stage"] for s in steps]}
    job.cancel_file.unlink(missing_ok=True)
    job.status["run"] = run
    job.log(f"任务 {name} 开始，模板 {flow.get('template')}，要运行：{'、'.join(run['plan']) or '（无）'}")
    for step in steps:
        job.set_overlay(step["stage"], state="queued", progress=None, message=None, error=None)
    job.update(state="running", message="开始")
    job.publish_stages()
    stage = None
    try:
        missing = config.missing_tools()
        if missing:
            raise PipelineError("缺少组件：" + "、".join(missing) + "。请先运行 setup.cmd")
        for step in steps:
            stage = step["stage"]
            _run_one(job, stage)
        stage = None
        job.overlay.clear()
        job.update(state="done", stage=None, stage_progress=1.0,
                   message="完成" if steps else "已经是最新", finished_at=_now(),
                   outputs=outputs_of(job_dir))
        job.publish_stages()
        job.log("任务完成，结果在 " + str(job.dir / "output"))
        return 0
    except proc.Cancelled:
        job.overlay.clear()
        job.update(state="cancelled", stage=None, message="已取消", finished_at=_now(),
                   outputs=outputs_of(job_dir))
        job.publish_stages()
        job.log("任务已取消")
        return 2
    except (PipelineError, proc.ProcError) as e:
        _fail(job, stage, str(e))
        return 1
    except Exception as e:  # a bug in this program: keep the traceback for diagnosis
        job.log(traceback.format_exc())
        _fail(job, stage, f"内部错误：{e}")
        return 1


def _fail(job: Job, stage: Optional[str], message: str) -> None:
    job.overlay = {sid: o for sid, o in job.overlay.items() if o.get("state") == "failed"}
    if stage is not None:
        job.overlay[stage] = {"state": "failed", "progress": None, "message": "失败",
                              "error": message}
    job.update(state="failed", stage=None, message="失败", error=message, finished_at=_now(),
               outputs=outputs_of(job.dir))
    job.publish_stages()
    job.log(f"任务失败：{message}")


def stage_table(job_dir) -> list:
    """[(id, label, state, summary text)] for the command line."""
    view = job_view(job_dir)
    rows = []
    for sid in wf.STAGE_ORDER:
        entry = view["stages"].get(sid) or {}
        facts = "  ".join(f"{k}={v}" for k, v in (entry.get("summary") or {}).items())
        if entry.get("seconds") is not None:
            facts = (facts + f"  {entry['seconds']:.0f}s").strip()
        if entry.get("error"):
            facts = (facts + "  " + entry["error"]).strip()
        rows.append((sid, wf.stage_label(sid), entry.get("state", "?"), facts))
    return rows
