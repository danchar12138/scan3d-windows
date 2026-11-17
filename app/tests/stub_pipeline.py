"""Stand-in for `app/scan3d.py run ...` used by test_server.py.

It honours the same contract as the real pipeline: it removes a leftover `cancel` file when
it starts, writes `runs/<job>/status.json` atomically while it "works", appends to
`logs/pipeline.log`, stops when the `cancel` file appears, and leaves output files when done.

    stub_pipeline.py run <input> --name <job> --quality <q> --runs <dir>
                     [--set KEY=VALUE ...]
                     [--mode ok|fail|crash|silent|hang|linger] [--step SECONDS]
                     [--startup-delay SECONDS] [--child]

    --set    overrides of the quality preset, as the real command line takes them; recorded
             in <job>/stub_argv.json as "overrides" ({key: value}, values parsed as JSON)

    ok       run all six stages, write outputs, state done, exit 0
    fail     report state failed with an error, exit 1
    crash    exit with code 3 in the middle, leaving state running
    silent   exit with code 5 before writing any status
    hang     report running, then ignore the cancel file
    linger   finish like ok, but do not exit
    --child  also start a grandchild process and record its pid in <job>/child.pid

Nothing here runs longer than LIFETIME seconds, so a stub can never be left behind for good.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

STAGES = ("ingest", "sfm", "dataset", "train", "mesh", "export")
LIFETIME = 120.0
SPLAT_BYTES = bytes(range(256)) * 16


def now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    for attempt in range(40):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # a reader has the file open for an instant
            time.sleep(0.02)
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run"])
    parser.add_argument("input")
    parser.add_argument("--name", required=True)
    parser.add_argument("--quality", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--set", action="append", metavar="KEY=VALUE")
    parser.add_argument("--mode", default="ok",
                        choices=["ok", "fail", "crash", "silent", "hang", "linger"])
    parser.add_argument("--step", type=float, default=0.03)
    parser.add_argument("--startup-delay", type=float, default=0.0)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()

    born = time.monotonic()
    runs = Path(args.runs)
    job_dir = runs / args.name
    cancel = job_dir / "cancel"

    def event(kind: str) -> None:
        with open(runs / "_stub_events.log", "a", encoding="utf-8") as f:
            f.write(json.dumps({"job": args.name, "event": kind, "t": time.time()}) + "\n")

    print("stub 启动: %s" % args.name, flush=True)
    time.sleep(args.startup_delay)
    if args.mode == "silent":
        print("stub: silent exit", flush=True)
        return 5

    try:
        cancel.unlink()  # the real pipeline discards a leftover cancel file too
    except FileNotFoundError:
        pass
    (job_dir / "logs").mkdir(parents=True, exist_ok=True)
    overrides = {}
    for item in args.set or []:
        key, _, value = item.partition("=")
        try:
            overrides[key] = json.loads(value)
        except ValueError:
            overrides[key] = value
    write_json_atomic(job_dir / "stub_argv.json",
                      {"argv": sys.argv[1:], "input": args.input, "quality": args.quality,
                       "overrides": overrides})
    event("start")

    status = {"job": args.name, "state": "running", "stage": STAGES[0], "stage_index": 1,
              "stage_count": len(STAGES), "stage_progress": None, "message": "开始",
              "quality": args.quality, "started_at": now(), "updated_at": now(),
              "finished_at": None, "error": None, "pid": os.getpid(), "warnings": []}

    def update(**fields) -> None:
        status.update(fields)
        status["updated_at"] = now()
        write_json_atomic(job_dir / "status.json", status)

    def log(text: str) -> None:
        with open(job_dir / "logs" / "pipeline.log", "a", encoding="utf-8") as f:
            f.write(text + "\n")

    def pause(seconds: float, *, cancellable: bool = True) -> bool:
        """Sleep; True when the cancel file appeared. Exits if the stub outlives LIFETIME."""
        end = time.monotonic() + seconds
        while True:
            if time.monotonic() - born > LIFETIME:
                os._exit(9)
            if cancellable and cancel.exists():
                return True
            left = end - time.monotonic()
            if left <= 0:
                return False
            time.sleep(min(0.02, left))

    update()
    log("任务 %s 开始，质量档位 %s" % (args.name, args.quality))

    if args.child:
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(%d)" % LIFETIME],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
        (job_dir / "child.pid").write_text(str(child.pid), encoding="ascii")

    if args.mode == "hang":
        update(message="不理会取消")
        pause(LIFETIME, cancellable=False)
        return 9

    for index, stage in enumerate(STAGES, start=1):
        for tick in range(3):
            progress = None if stage == "export" else round(tick / 3, 4)
            update(stage=stage, stage_index=index, stage_progress=progress,
                   message="阶段 %s 进行中" % stage)
            log("stage %s tick %d\rprogress bar %d%%" % (stage, tick, tick * 33))
            if pause(args.step):
                update(state="cancelled", message="已取消", finished_at=now())
                log("任务已取消")
                event("end")
                return 2
        if index == 2 and args.mode == "fail":
            update(state="failed", message="失败", error="stub: sfm failed", finished_at=now())
            event("end")
            return 1
        if index == 2 and args.mode == "crash":
            print("stub: crashing now", flush=True)
            event("end")
            os._exit(3)

    out = job_dir / "output"
    out.mkdir(exist_ok=True)
    (out / "splat.ply").write_bytes(SPLAT_BYTES)
    (out / "mesh.glb").write_bytes(b"glTF" + b"\0" * 60)
    (out / "collision.glb").write_bytes(b"glTF" + b"\0" * 28)
    write_json_atomic(out / "report.json", {"job": args.name, "quality": args.quality})
    update(state="done", stage_progress=1.0, message="完成", finished_at=now(),
           outputs={"splat": "output/splat.ply", "mesh": "output/mesh.glb",
                    "collision": "output/collision.glb", "report": "output/report.json"})
    log("任务完成")
    event("end")
    if args.mode == "linger":
        pause(LIFETIME, cancellable=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
