"""The scan pipeline: photos or video in, Gaussian splat and meshes out.

Stages run in order and each records its result in `runs/<job>/stages/<stage>.json`, so a job
that is run again continues from the first unfinished stage. Progress is published in
`runs/<job>/status.json`, which is what the CLI and the web UI read.
"""
from __future__ import annotations

import ctypes
import datetime as _dt
import json
import os
import re
import shutil
import struct
import time
import traceback
from pathlib import Path
from typing import Callable, Optional

import config
import proc
from config import IMAGE_EXT, STAGES, TOOL_PATHS, UNSUPPORTED_IMAGE_EXT, VIDEO_EXT


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


def image_size(path: Path) -> Optional[tuple[int, int]]:
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


# --------------------------------------------------------------------------------------------
# Job state
# --------------------------------------------------------------------------------------------

class Job:
    def __init__(self, name: str, quality: str, *, runs_dir: Path = config.RUNS,
                 echo: Optional[Callable[[str], None]] = print):
        self.name = name
        self.quality = quality
        self.dir = runs_dir / name
        self.logs = self.dir / "logs"
        self.stages_dir = self.dir / "stages"
        self.cancel_file = self.dir / "cancel"
        self.status_path = self.dir / "status.json"
        self.echo = echo
        self.warnings: list[str] = []
        for d in (self.dir, self.logs, self.stages_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.status = {
            "job": name, "state": "running", "stage": STAGES[0], "stage_index": 1,
            "stage_count": len(STAGES), "stage_progress": None, "message": "",
            "quality": quality, "started_at": _now(), "updated_at": _now(),
            "finished_at": None, "error": None, "pid": os.getpid(), "warnings": [],
        }
        self._last_echo_at = 0.0

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

    def progress(self, fraction: Optional[float], message: Optional[str] = None) -> None:
        fields = {"stage_progress": None if fraction is None else round(min(max(fraction, 0), 1), 4)}
        if message is not None:
            fields["message"] = message
        self.update(**fields)
        # Console feedback for command-line runs, at most one line every 20 seconds.
        text = self.status.get("message") or ""
        if self.echo and text and time.monotonic() - self._last_echo_at > 20:
            self._last_echo_at = time.monotonic()
            self.echo(f"    {text}")

    def warn(self, text: str) -> None:
        self.warnings.append(text)
        self.log("警告: " + text)

    def check_cancel(self) -> None:
        if self.cancel_file.exists():
            raise proc.Cancelled("cancelled by user")

    def stage_result(self, stage: str) -> Optional[dict]:
        path = self.stages_dir / f"{stage}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def save_stage(self, stage: str, result: dict) -> None:
        _write_json_atomic(self.stages_dir / f"{stage}.json", result)

    def tool(self, name: str, args: list, *, log: str, timeout: float, **kw) -> proc.RunResult:
        cmd = [str(TOOL_PATHS[name])] + [str(a) for a in args]
        return proc.run(cmd, log_path=self.logs / log, timeout=timeout,
                        cancel_file=self.cancel_file, **kw)


class GpuLock:
    """One GPU job at a time across every scan3d process on this machine."""

    def __init__(self, job: Job, wait_seconds: float):
        self.job = job
        self.path = config.RUNS / ".gpu.lock"
        self.queue = config.RUNS / ".gpu.queue"
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
                    self.job.update(message="等待显卡空闲")
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
# Stage 1: ingest
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


def _extract_frames(job: Job, video: Path, out_dir: Path, target: int, settings: dict,
                    timeout: float) -> int:
    """Pull about `target` sharp, well-spread frames out of a video.

    Handheld video has motion blur in a good share of its frames. The extractor keeps the
    sharpest frame of each window, spaces frames by how much the view changes instead of by
    time, leaves out frames that are still blurred and frames where the camera stood still.
    """
    info = _video_info(job, video)
    skip = max(1, round(info["frames"] / max(target, 1)))
    args = ["sam", "extract", video, "-o", out_dir, "-s", skip, "-k", "-1", "--adaptive",
            "--similar", settings["similar_bits"], "--max-frames", int(target * 1.5),
            "--decoder", "builtin", "--lang", "en"]
    # Frames whose edges are smeared wider than this (in pixels, beyond the clip's sharpest
    # frames) are left out; the tool never drops more than two in a row. On the sample clip
    # 2 px removed 20 of 120 frames, all visibly motion-blurred.
    if settings.get("blur_reject_px"):
        args += ["--blur-reject", settings["blur_reject_px"]]
    long_side = max(info["width"], info["height"])
    if long_side > settings["max_long_side"]:
        args += ["--scale", f"{settings['max_long_side'] / long_side:.4f}"]
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


def _weights(job: Job, feature: str, file_name: str) -> Optional[Path]:
    """Path of a weights file, or None when its licence has not been accepted on this machine
    (the feature is then left out, with a warning: nobody but the person running the scan can
    accept a licence). A file that is simply not installed is an error."""
    import models

    licence = models.licence_of(file_name)
    if licence and licence not in models.accepted_licences():
        job.warn(f"{feature}已跳过：尚未同意 {models.LICENCES[licence]['name']}。"
                 f"在页面的「模型与许可」里同意后即可启用")
        return None
    return _model_path(file_name)


def _write_masks(job: Job, dataset: Path, folders: list[str], preset: dict,
                 settings: dict) -> dict:
    """Write dataset/masks/<folder>/<image stem>.png, black where the image must be ignored.

    Pose estimation, the dense cloud and training all pick the folder up by themselves.
    Two uses: paint out what moves between photos (people by default, named by a text prompt),
    or, for a scan of a single object, keep the subject and paint out everything else.
    """
    import models

    subject = bool(preset.get("subject_only"))
    if subject:
        model = _weights(job, "只保留主体", settings["subject"]["model_file"])
        detector = None
    else:
        model = _weights(job, "抠除移动物体", settings["masks"]["model_file"])
        detector_file = settings["masks"].get("detector_file")
        detector = _weights(job, "抠除移动物体", detector_file) if detector_file else None
        if detector_file and detector is None:
            model = None
    if model is None:
        return {"masks": False}

    total = sum(len(list((dataset / "images" / f).iterdir())) for f in folders)
    done_before = 0
    kept: list[float] = []
    label = "分离主体" if subject else "抠除移动物体"
    with GpuLock(job, settings["timeouts_seconds"]["gpu_lock_wait"]):
        for folder in folders:
            job.check_cancel()
            frames = dataset / "images" / folder
            raw = dataset / "masks_raw" / folder
            shutil.rmtree(raw, ignore_errors=True)
            raw.mkdir(parents=True)
            names: dict[int, str] = {}

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
                args += ["--text", settings["masks"]["prompt"]]
                if detector is not None:
                    args += ["--detector", detector, *models.licence_args(detector.name)]
            job.tool("spirula", args, log="masks.log", timeout=settings["timeouts_seconds"]["masks"],
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
    result = {"masks": True, "mask_kind": "subject" if subject else "moving",
              "masked_images": touched, "masked_percent": round(share, 2)}
    if subject:
        job.log(f"主体分离：平均保留画面的 {100 - share:.0f}%")
    else:
        job.log(f"抠除移动物体（{settings['masks']['prompt']}）：{touched}/{total} 张图片里有，"
                f"平均遮住画面的 {share:.1f}%")
        if share > 30:
            job.warn(f"抠除的区域很大（平均 {share:.0f}% 的画面）。如果被抠掉的是想保留的内容，"
                     f"请修改 settings.json 里 masks.prompt 的提示词，或关闭抠除移动物体")
    return result


def stage_ingest(job: Job, input_path: Path, preset: dict, settings: dict) -> dict:
    limits, timeouts = settings["limits"], settings["timeouts_seconds"]
    if not input_path.exists():
        raise PipelineError(f"找不到输入：{input_path}")
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
    if dataset.exists():
        shutil.rmtree(dataset)
    images_dir = dataset / "images"
    images_dir.mkdir(parents=True)

    # Photos are linked, not re-encoded, so their EXIF survives: the pose estimation reads the
    # focal length and the rotation tag from it. Names become plain ASCII because phone
    # galleries are full of names the tools downstream cannot be relied on to handle.
    names: dict[str, str] = {}
    if photos:
        (images_dir / "photos").mkdir()
        for i, src in enumerate(photos, start=1):
            dst = images_dir / "photos" / f"p_{i:05d}{src.suffix.lower()}"
            _link_or_copy(src, dst)
            names[f"photos/{dst.name}"] = str(src)

    # Each video gets its own folder: the pose estimation gives every folder its own camera
    # and treats a folder named as a sequence as consecutive frames.
    sequences: list[str] = []
    video_frames = 0
    if videos:
        infos = [_video_info(job, v) for v in videos]
        total_frames = sum(i["frames"] for i in infos)
        target_total = int(settings["video"]["target_frames"])
        for i, (video, info) in enumerate(zip(videos, infos), start=1):
            job.check_cancel()
            job.progress((i - 1) / len(videos), f"从视频中挑选清晰画面 {i}/{len(videos)}")
            target = max(limits["min_images"], round(target_total * info["frames"] / total_frames))
            folder = f"v{i}"
            video_frames += _extract_frames(job, video, images_dir / folder, target,
                                            settings["video"], timeouts["extract"])
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
    if len(all_images) < limits["min_images"]:
        raise PipelineError(
            f"只有 {len(all_images)} 张可用图片，至少需要 {limits['min_images']} 张。"
            f"请围绕场景多拍一些，相邻照片之间保持大部分画面重叠")

    sizes = [s for s in (image_size(p) for p in all_images[:: max(len(all_images) // 50, 1)]) if s]
    _write_json_atomic(dataset / "names.json", names)
    result = {
        "dataset": str(dataset), "images": len(all_images), "photos": len(photos),
        "video_frames": video_frames, "videos": len(videos), "sequences": sequences,
        "skipped_unsupported": len(unreadable),
        "long_side": max((max(s) for s in sizes), default=0),
    }
    job.log(f"输入就绪：{result['images']} 张图片"
            + (f"（其中 {video_frames} 张来自视频）" if video_frames else ""))
    result["masks"] = False
    if preset.get("masks") or preset.get("subject_only"):
        folders = (["photos"] if photos else []) + sequences
        result.update(_write_masks(job, dataset, folders, preset, settings))
    return result


# --------------------------------------------------------------------------------------------
# Stage 2: camera poses
# --------------------------------------------------------------------------------------------

def _run_sfm(job: Job, dataset: Path, ingest: dict, attempt: dict, settings: dict) -> dict:
    """One pose-estimation run with the given feature detector and matcher."""
    for leftover in ("sparse", "features"):
        shutil.rmtree(dataset / leftover, ignore_errors=True)
    (dataset / "matches.bin").unlink(missing_ok=True)
    n = ingest["images"]
    models: dict[int, tuple[int, int]] = {}
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
            models[int(m.group(1))] = (int(m.group(2)), int(m.group(3)))
            return
        m = re.search(r"([\d.]+) px mean reprojection", line)
        if m:
            found["reprojection"] = float(m.group(1))

    # The image folder is named explicitly: given the dataset folder, the tool would read the
    # masks beside it as photos too.
    args = ["sfm", "auto", dataset / "images", "-o", dataset,
            "--quality", settings["sfm"]["quality"],
            "--features", attempt["features"], "--matcher", attempt["matcher"], "--lang", "en"]
    # Learned detectors and matchers read their weights from the bundle, never the network.
    for flag, key in (("--aliked-model", "aliked_model"), ("--lightglue-model", "lightglue_model")):
        if attempt.get(key):
            args += [flag, _model_path(attempt[key])]
    for folder in ingest["sequences"]:
        args += ["--sequence", folder]
    job.progress(0, "提取特征点")
    with GpuLock(job, settings["timeouts_seconds"]["gpu_lock_wait"]):
        job.tool("spirula", args, log="sfm.log", timeout=settings["timeouts_seconds"]["sfm"],
                 stall_timeout=1800, on_line=on_line)
    registered, points = models.get(0, (0, 0))
    if not (dataset / "sparse" / "0").is_dir():
        registered = 0
    return {"registered": registered, "points": points, "models": len(models),
            "mean_reprojection_error_px": found.get("reprojection")}


def stage_sfm(job: Job, ingest: dict, settings: dict) -> dict:
    cfg = settings["sfm"]
    dataset = Path(ingest["dataset"])
    n = ingest["images"]
    need = max(settings["limits"]["min_images"] // 2, int(n * cfg["min_registered_fraction"]))
    failure = ("相机位姿计算失败。常见原因：相邻照片重叠太少、画面模糊、场景缺少纹理"
               "（白墙、玻璃、反光面）或拍摄时场景里有大面积移动的物体")

    # The attempts in settings.json run in order until one registers enough images. The first
    # is the classical detector, which is the most precise on textured scenes; the learned
    # detector and matcher that follow hold up better on low texture and wide viewpoint changes.
    result: dict = {"registered": 0}
    used = None
    for index, attempt in enumerate(cfg["attempts"]):
        job.check_cancel()
        label = f"{attempt['features']} + {attempt['matcher']}"
        try:
            result = _run_sfm(job, dataset, ingest, attempt, settings)
        except (proc.ProcTimeout, proc.ProcStalled):
            raise
        except proc.ProcError:
            job.log(f"位姿计算（{label}）没有得到结果")
            result = {"registered": 0}
        used = label
        if result["registered"] >= need:
            break
        if index + 1 < len(cfg["attempts"]):
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
# Stage 3: geometric priors (surface normals and a dense seed cloud)
# --------------------------------------------------------------------------------------------

def _model_path(file_name: str) -> Path:
    path = config.MODELS / file_name
    if not path.exists():
        raise PipelineError(f"缺少模型文件 {file_name}。请在页面的「模型与许可」里下载，"
                            f"或运行 setup.cmd")
    return path


def stage_dataset(job: Job, ingest: dict, preset: dict, settings: dict) -> dict:
    """Compute what the trainer can use beyond the photos themselves.

    Surface normals: one map per image from a monocular network. The trainer uses them to keep
    flat, textureless surfaces (walls, floors) flat.
    Dense seed cloud: millions of matched points triangulated with the known camera poses. The
    trainer starts from it instead of the sparse pose-estimation points, so detail is in place
    from the first step.
    """
    dataset = Path(ingest["dataset"])
    timeouts = settings["timeouts_seconds"]
    shutil.rmtree(dataset / "normals", ignore_errors=True)
    shutil.rmtree(dataset / "dense", ignore_errors=True)
    both = bool(preset.get("normals")) and bool(preset.get("dense_seed"))
    result: dict = {"normals": False, "dense_seed": False}

    if preset.get("normals"):
        model = _model_path(settings["normals"]["model_file"])
        span = 0.3 if both else 1.0

        def on_normals(line: str) -> None:
            m = re.match(r"(\d+) / (\d+) images", line)
            if m:
                job.progress(span * int(m.group(1)) / int(m.group(2)),
                             f"估计表面法线 {m.group(1)}/{m.group(2)}")

        job.progress(0, "估计表面法线")
        with GpuLock(job, timeouts["gpu_lock_wait"]):
            job.tool("spirula", ["geometry", dataset, "--model", model, "--lang", "en"],
                     log="normals.log", timeout=timeouts["normals"], stall_timeout=900,
                     on_line=on_normals)
        result["normals"] = True

    dense_model = (_weights(job, "稠密点云初始化", settings["dense_seed"]["model_file"])
                   if preset.get("dense_seed") else None)
    if dense_model is not None:
        model = dense_model
        base = 0.3 if both else 0.0
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
                job.progress(base + (1 - base) * (lo + (hi - lo) * done), f"稠密点云：{label}")
                return
            m = re.search(r"points: (\d+)", line)
            if m and line.startswith("Dense cloud completed"):
                result["dense_points"] = int(m.group(1))

        job.progress(base, "稠密点云：加载模型")
        with GpuLock(job, timeouts["gpu_lock_wait"]):
            job.tool("spirula", ["dense", dataset, "--checkpoint", model,
                                 "--preset", settings["dense_seed"]["preset"], "--lang", "en"],
                     log="dense.log", timeout=timeouts["dense"], stall_timeout=1200,
                     on_line=on_dense)
        if not (dataset / "dense" / "roma.ply").exists():
            raise PipelineError("稠密点云没有生成，详见 logs/dense.log")
        result["dense_seed"] = True
    return result


# --------------------------------------------------------------------------------------------
# Stage 4: Gaussian splat training
# --------------------------------------------------------------------------------------------

def ply_vertex_count(path: Path) -> Optional[int]:
    with open(path, "rb") as f:
        head = f.read(4096).split(b"end_header")[0].decode("ascii", "replace")
    m = re.search(r"element vertex (\d+)", head)
    return int(m.group(1)) if m else None


def stage_train(job: Job, dataset: dict, preset: dict, settings: dict, holdout: bool,
                priors: Optional[dict] = None) -> dict:
    """`priors` is the result of the dataset stage. Left out (comparison scripts do that), the
    trainer decides by what it finds in the dataset folder and by `train_args`."""
    timeouts = settings["timeouts_seconds"]
    train_dir = job.dir / "train"
    if train_dir.exists():
        shutil.rmtree(train_dir)
    iterations = int(preset["iterations"])
    # A scan masked down to its subject is trained with the trainer's preset for that case:
    # the masked background is trained as empty space instead of being skipped.
    train_preset = ("centered-object" if dataset.get("mask_kind") == "subject"
                    else preset["train_preset"])
    args = ["train", train_preset, "--lang", "en", "--data", dataset["dataset"],
            "--output-dir-prefix", train_dir, "--output-dir-name", "run",
            "--num-iterations", iterations, "--cap-max", preset["cap_max"],
            # Without these the trainer keeps a web viewer open after it finishes and never
            # exits.
            "--disable-viewer", "1", "--keep-viewer-alive", "0"]
    # The working resolution of a quality level is its longest image side. The trainer takes
    # that as a divisor of the source size (it need not be a whole number).
    long_side, limit = dataset.get("long_side", 0), preset.get("max_image_size", 0)
    if long_side and limit and long_side > limit:
        args += ["--train-resolution-divisor", f"{long_side / limit:.4f}"]
    if priors is not None:
        # The trainer would use whatever it finds in the dataset folder; state it instead.
        args += ["--load-normals", "1" if priors.get("normals") else "0",
                 "--load-depths", "0",
                 "--load-masks", "1" if dataset.get("masks") else "0",
                 "--seed-pointcloud", "dense/roma.ply" if priors.get("dense_seed") else "sparse"]
    if holdout:
        # Hold out every 8th image and score the result on them. Used for comparing settings;
        # normal runs train on every image.
        args += ["--eval-mode", "interval", "--save-eval-images", "1"]
    args += list(preset.get("train_args", []))

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
    with GpuLock(job, timeouts["gpu_lock_wait"]):
        result = job.tool("spirula", args, log="train.log", timeout=timeouts["train"],
                          stall_timeout=timeouts["train_stall"], on_line=on_line,
                          done_marker=None if holdout else r"^Training complete\. Steps",
                          done_grace=120)

    run = train_dir / "run"
    checkpoints = sorted(run.glob("step-*.ckpt"))
    splat = checkpoints[-1] / "splat.ply" if checkpoints else None
    if splat is None or not splat.exists():
        raise PipelineError("训练没有产出高斯文件，详见 logs/train.log")
    out = {"run": str(run), "splat": str(splat), "gaussians": ply_vertex_count(splat),
           "iterations": iterations, "seconds": round(result.seconds, 1),
           "size_mb": round(splat.stat().st_size / 1048576, 1)}
    metrics = run / "metrics.json"
    if metrics.exists():
        try:
            out["metrics"] = json.loads(metrics.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return out


# --------------------------------------------------------------------------------------------
# Stage 5 and 6: meshes and export
# --------------------------------------------------------------------------------------------

def stage_mesh(job: Job, dataset: dict, train: dict, preset: dict, settings: dict) -> dict:
    import meshpost  # imported here so the splat-only path works without it

    out = job.dir / "output"
    out.mkdir(exist_ok=True)
    mesh_cfg = preset["mesh"]
    job.progress(0, "生成网格")
    try:
        with GpuLock(job, settings["timeouts_seconds"]["gpu_lock_wait"]):
            result = meshpost.build_meshes(
                run_dir=Path(train["run"]), dataset_dir=Path(dataset["dataset"]), out_dir=out,
                work_dir=job.dir / "mesh",
                tools={k: TOOL_PATHS[k] for k in ("spirula", "colmap")},
                visual_faces=mesh_cfg["visual_faces"],
                collision_faces=mesh_cfg["collision_faces"],
                texture_size=mesh_cfg["texture_size"], log_path=job.logs / "mesh.log",
                cancel_file=job.cancel_file,
                on_progress=lambda f, m: job.progress(f, f"生成网格：{m}"))
    except meshpost.MeshError as e:
        raise PipelineError(str(e))
    # The intermediate meshes are several hundred MB and are not needed once the GLBs exist.
    shutil.rmtree(job.dir / "mesh", ignore_errors=True)
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in result.items()}


def _scene_up(job: Job, results: dict) -> Optional[list[float]]:
    """The scene's up direction in the frame of the exported files."""
    import orient

    # The pose estimation levels its model with +Z up (from the photos' EXIF rotation, the
    # camera poses and the ground plane).
    up = [0.0, 0.0, 1.0]
    transform = Path(results["train"]["run"]) / "scene_transform.json"
    try:
        matrix = json.loads(transform.read_text(encoding="utf-8"))[
            "train_from_world"]["rotation"]["matrix_3x3"]
        up = orient.rotate(matrix, up)
    except (OSError, KeyError, json.JSONDecodeError):
        pass  # no transform recorded: the training frame is the dataset frame
    return up


def stage_export(job: Job, results: dict) -> dict:
    out = job.dir / "output"
    out.mkdir(exist_ok=True)
    job.progress(None, "导出结果")
    splat_dst = out / "splat.ply"
    splat_dst.unlink(missing_ok=True)
    _link_or_copy(Path(results["train"]["splat"]), splat_dst)
    transform = Path(results["train"]["run"]) / "scene_transform.json"
    if transform.exists():
        shutil.copy2(transform, out / "scene_transform.json")

    import orient
    import splatconv

    outputs = {"splat": "output/splat.ply"}
    for key, name in (("mesh", "mesh.glb"), ("collision", "collision.glb")):
        if (out / name).exists():
            outputs[key] = f"output/{name}"

    # Compact copies of the splat for the in-browser viewer: the training PLY is far too large
    # to send to a phone. Losing them only costs the viewer a rebuild on demand.
    job.progress(None, "生成网页预览")
    try:
        splatconv.write_preview(splat_dst, out / splatconv.PREVIEW_NAME)
        splatconv.write_preview(splat_dst, out / splatconv.PREVIEW_LITE_NAME,
                                max_gaussians=splatconv.LITE_MAX_GAUSSIANS)
        outputs["preview"] = f"output/{splatconv.PREVIEW_NAME}"
        outputs["preview_lite"] = f"output/{splatconv.PREVIEW_LITE_NAME}"
    except (splatconv.PreviewError, OSError) as e:
        job.warn(f"网页预览文件生成失败，查看时会重新生成：{e}")

    orientation = None
    up = _scene_up(job, results)
    if up is not None:
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
        for key in ("mesh", "collision"):
            # A mesh stage restored from an earlier run already carries the rotation; setting
            # it again is harmless.
            if key in outputs:
                orient.set_glb_root_rotation(
                    job.dir / outputs[key], quaternion,
                    {"scan3d_root_rotation": "turns the scan's up direction to +Y"})
    report = {
        "job": job.name, "quality": job.quality, "finished_at": _now(),
        "warnings": job.warnings, "orientation": orientation, "stages": results,
        "files": {k: {"path": v, "size_mb": round((job.dir / v).stat().st_size / 1048576, 1)}
                  for k, v in outputs.items()},
    }
    _write_json_atomic(out / "report.json", report)
    outputs["report"] = "output/report.json"
    return {"outputs": outputs}


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------

def run_job(input_path: Path, name: str, quality: str, *, mesh: bool = True,
            holdout: bool = False, fresh: bool = False, overrides: Optional[dict] = None,
            runs_dir: Path = config.RUNS, echo: Optional[Callable[[str], None]] = print) -> int:
    """Run (or resume) a scan. Returns a process exit code: 0 done, 1 failed, 2 cancelled."""
    settings = config.load_settings()
    if quality not in settings["quality"]:
        raise ValueError(f"unknown quality {quality!r}")
    job_dir = runs_dir / name
    if fresh and job_dir.exists():
        for sub in ("stages", "dataset", "train", "mesh", "output", "logs", "input", "sfm"):
            shutil.rmtree(job_dir / sub, ignore_errors=True)
    job = Job(name, quality, runs_dir=runs_dir, echo=echo)
    job.cancel_file.unlink(missing_ok=True)

    preset = json.loads(json.dumps(settings["quality"][quality]))  # private copy
    for key, value in (overrides or {}).items():
        target, parts = preset, key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value

    results: dict = {}
    stage_calls = {
        "ingest": lambda: stage_ingest(job, Path(input_path), preset, settings),
        "sfm": lambda: stage_sfm(job, results["ingest"], settings),
        "dataset": lambda: stage_dataset(job, results["ingest"], preset, settings),
        "train": lambda: stage_train(job, results["ingest"], preset, settings, holdout,
                                     results["dataset"]),
        "mesh": lambda: stage_mesh(job, results["ingest"], results["train"], preset, settings),
        "export": lambda: stage_export(job, results),
    }
    job.log(f"任务 {name} 开始，质量档位 {quality}")
    job.update(state="running", message="开始")
    try:
        missing = config.missing_tools()
        if missing:
            raise PipelineError("缺少组件：" + "、".join(missing) + "。请先运行 setup.cmd")
        for index, stage in enumerate(STAGES, start=1):
            job.check_cancel()
            job.update(stage=stage, stage_index=index, stage_progress=None)
            if stage == "mesh" and not mesh:
                continue
            saved = job.stage_result(stage)
            # Export is cheap and depends on everything before it, so it always runs.
            if saved is not None and stage != "export":
                results[stage] = saved
                job.log(f"阶段 {stage} 已完成，跳过")
                continue
            started = time.monotonic()
            try:
                result = stage_calls[stage]()
            except (PipelineError, proc.ProcError) as e:
                if stage != "mesh":
                    raise
                # The splat is the primary result; a mesh failure should not discard it.
                job.warn(f"网格生成失败，已跳过：{e}")
                continue
            result["stage_seconds"] = round(time.monotonic() - started, 1)
            results[stage] = result
            job.save_stage(stage, result)
            job.log(f"阶段 {stage} 完成，用时 {result['stage_seconds']:.0f} 秒")
        job.update(state="done", stage_progress=1.0, message="完成", finished_at=_now(),
                   outputs=results["export"]["outputs"])
        job.log("任务完成，结果在 " + str(job.dir / "output"))
        return 0
    except proc.Cancelled:
        job.update(state="cancelled", message="已取消", finished_at=_now())
        job.log("任务已取消")
        return 2
    except (PipelineError, proc.ProcError) as e:
        job.update(state="failed", message="失败", error=str(e), finished_at=_now())
        job.log(f"任务失败：{e}")
        return 1
    except Exception as e:  # a bug in this program: keep the traceback for diagnosis
        job.log(traceback.format_exc())
        job.update(state="failed", message="失败", error=f"内部错误：{e}", finished_at=_now())
        return 1
