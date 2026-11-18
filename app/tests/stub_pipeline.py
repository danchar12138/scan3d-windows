"""Stand-in for `app/scan3d.py run ...` used by test_server.py, and a stand-in for the
pipeline's file helpers (job_view, write_workflow, reset_stage) that the server calls.

As a command it honours the same contract as the real pipeline: it reads `<job>/workflow.json`,
plans the run with workflow.plan() from the stage records on disk, runs the planned stages one
after another (a few short ticks each), writes a record `stages/<id>.json` for every stage that
finishes and keeps `status.json` (state, stage, `stages`, `run`) up to date atomically. It
appends to `logs/pipeline.log` and `logs/<stage>.log`, removes a leftover `cancel` file when it
starts, stops when the `cancel` file appears, and leaves output files when done.

    stub_pipeline.py run <input> --name <job> --runs <dir>
                     [--to STAGE] [--only a,b] [--force a,b]
                     [--mode ok|fail|crash|silent|hang|linger] [--fail-stage STAGE]
                     [--step SECONDS] [--startup-delay SECONDS] [--child]

    ok       run the plan, write outputs, state done, exit 0
    fail     report state failed with an error after --fail-stage (default sfm), exit 1
    crash    exit with code 3 in --fail-stage, leaving state running
    silent   exit with code 5 before writing any status
    hang     report running, then ignore the cancel file
    linger   finish like ok, but do not exit
    --child  also start a grandchild process and record its pid in <job>/child.pid

Nothing here runs longer than LIFETIME seconds, so a stub can never be left behind for good.

The file helpers use the real workflow module and the same layout the contract describes, but
the input signature is a constant: the stub does not look at the photos.
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

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:     # the embeddable Python keeps the script folder off sys.path
    sys.path.insert(0, str(APP_DIR))

import workflow as wf  # noqa: E402

LIFETIME = 120.0
SIGNATURE = "stub"
SPLAT_BYTES = bytes(range(256)) * 16
# what each stage leaves behind (and what resetting it deletes), relative to the job folder
STAGE_FILES = {
    "masks": ["dataset/masks/mask-0001.png"],
    "train": ["output/splat.ply"],
    "dense": ["dense/fused.ply"],
    "mesh": ["output/mesh.glb", "output/collision.glb"],
    "coverage": ["output/coverage.json"],
    "export": ["output/report.json"],
}
STAGE_SUMMARY = {
    "ingest": {"images": 1}, "masks": {"masks": 1}, "sfm": {"registered": 1},
    "normals": {"normals": 1}, "dense_seed": {"points": 1000}, "train": {"gaussians": 1999962},
    "dense": {"points": 250000}, "mesh": {"mesh_faces": 275409}, "coverage": {"holes": 3},
    "export": {},
}
OUTPUT_KEYS = {"splat": "output/splat.ply", "mesh": "output/mesh.glb",
               "collision": "output/collision.glb", "report": "output/report.json",
               "coverage": "output/coverage.json"}


def now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    for attempt in range(40):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # a reader has the file open for an instant
            time.sleep(0.02)
    os.replace(tmp, path)


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------- file helpers

def write_workflow(job_dir, workflow: dict) -> None:
    write_json_atomic(Path(job_dir) / "workflow.json", workflow)


def load_workflow(job_dir) -> dict:
    """The job's workflow.json; the default template (no user templates) when there is none."""
    path = Path(job_dir) / "workflow.json"
    if not path.exists():
        return wf.resolve(None, user={})
    stored = read_json(path)
    if stored is None:
        raise wf.WorkflowError(["workflow.json 无法读取"])
    return wf.validate(stored)


def read_records(job_dir) -> dict:
    records = {}
    for sid in wf.STAGE_ORDER:
        record = read_json(Path(job_dir) / "stages" / (sid + ".json"))
        if isinstance(record, dict):
            records[sid] = record
    return records


def _entry(state, record=None, **fields) -> dict:
    entry = {"state": state, "progress": None, "message": None,
             "seconds": (record or {}).get("seconds"), "error": None,
             "summary": ((record or {}).get("result") or {}).get("summary") or {}}
    entry.update(fields)
    return entry


def job_view(job_dir) -> dict:
    """{"workflow", "template", "stages", "run"}: what pipeline.job_view gives."""
    job_dir = Path(job_dir)
    status = read_json(job_dir / "status.json") or {}
    workflow = load_workflow(job_dir)
    records = read_records(job_dir)
    states = wf.stage_states(workflow, records, SIGNATURE)
    live = status.get("stages") if status.get("state") == "running" else None
    kept = status.get("stages") if isinstance(status.get("stages"), dict) else {}
    stages = {}
    for sid in wf.STAGE_ORDER:
        entry = _entry(states[sid], records.get(sid))
        if live is not None and isinstance(live.get(sid), dict):
            entry.update(live[sid])
        elif states[sid] in ("pending", "stale") and (kept.get(sid) or {}).get("state") == "failed":
            entry.update(state="failed", error=kept[sid].get("error"))
        stages[sid] = entry
    run = status.get("run") if isinstance(status.get("run"), dict) else {}
    return {"workflow": workflow, "template": workflow.get("template"), "stages": stages,
            "run": {"to": run.get("to"), "only": list(run.get("only") or []),
                    "force": list(run.get("force") or []), "plan": list(run.get("plan") or [])}}


def delete_stage_results(job_dir, stage: str) -> None:
    job_dir = Path(job_dir)
    try:
        (job_dir / "stages" / (stage + ".json")).unlink()
    except FileNotFoundError:
        pass
    for rel in STAGE_FILES.get(stage, []):
        try:
            (job_dir / rel).unlink()
        except FileNotFoundError:
            pass


def reset_stage(job_dir, stage: str) -> None:
    if stage not in wf.STAGES:
        raise ValueError("没有这个阶段：%s" % stage)
    delete_stage_results(job_dir, stage)


# ---------------------------------------------------------------------------- the command

def ids(text):
    return [part for part in (text or "").split(",") if part]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run"])
    parser.add_argument("input")
    parser.add_argument("--name", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--to")
    parser.add_argument("--only")
    parser.add_argument("--force")
    parser.add_argument("--mode", default="ok",
                        choices=["ok", "fail", "crash", "silent", "hang", "linger"])
    parser.add_argument("--fail-stage", default="sfm")
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
    write_json_atomic(job_dir / "stub_argv.json",
                      {"argv": sys.argv[1:], "input": args.input, "to": args.to,
                       "only": ids(args.only), "force": ids(args.force)})
    event("start")

    workflow = load_workflow(job_dir)
    records = read_records(job_dir)
    states = wf.stage_states(workflow, records, SIGNATURE)
    try:
        steps = wf.plan(workflow, states, to=args.to or None, only=ids(args.only) or None,
                        force=ids(args.force))
    except wf.PlanError as exc:
        print("stub: %s: %s" % (exc.code, exc.message), file=sys.stderr, flush=True)
        event("end")
        return 64
    plan = [step["stage"] for step in steps]
    fingerprints = wf.fingerprints(workflow, SIGNATURE)

    stages = {sid: _entry(states[sid], records.get(sid)) for sid in wf.STAGE_ORDER}
    for sid in plan:
        stages[sid] = _entry("queued")
    status = {"job": args.name, "state": "running", "stage": None, "stage_progress": None,
              "message": "开始", "started_at": now(), "updated_at": now(), "finished_at": None,
              "error": None, "pid": os.getpid(), "warnings": [], "stages": stages,
              "run": {"to": args.to or None, "only": ids(args.only), "force": ids(args.force),
                      "plan": plan}}
    previous = read_json(job_dir / "status.json") or {}
    status["outputs"] = previous.get("outputs") or {}

    def update(**fields) -> None:
        status.update(fields)
        status["updated_at"] = now()
        write_json_atomic(job_dir / "status.json", status)

    def log(text: str, name: str = "pipeline.log") -> None:
        with open(job_dir / "logs" / name, "a", encoding="utf-8") as f:
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

    def outputs_now() -> dict:
        return {key: rel for key, rel in OUTPUT_KEYS.items() if (job_dir / rel).is_file()}

    def settle(final: str) -> None:
        """The run ended before its plan did: what was waiting is pending again."""
        for entry in stages.values():
            if entry["state"] in ("queued", "running"):
                entry.update(state="pending", progress=None)

    update()
    log("任务 %s 开始，计划：%s" % (args.name, " ".join(plan)))

    if args.child:
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(%d)" % LIFETIME],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
        (job_dir / "child.pid").write_text(str(child.pid), encoding="ascii")

    if args.mode == "hang":
        update(message="不理会取消")
        pause(LIFETIME, cancellable=False)
        return 9

    for sid in plan:
        delete_stage_results(job_dir, sid)       # a stage that runs again starts from nothing
        started = time.monotonic()
        stages[sid] = _entry("running", progress=0.0)
        for tick in range(3):
            stages[sid]["progress"] = round(tick / 3, 4)
            update(stage=sid, stage_progress=stages[sid]["progress"],
                   message="阶段 %s 进行中" % sid)
            log("stage %s tick %d\rprogress bar %d%%" % (sid, tick, tick * 33), sid + ".log")
            if pause(args.step):
                settle("cancelled")
                update(state="cancelled", message="已取消", finished_at=now())
                log("任务已取消")
                event("end")
                return 2
        if args.mode == "crash" and sid == args.fail_stage:
            print("stub: crashing now", flush=True)
            event("end")
            os._exit(3)
        if args.mode == "fail" and sid == args.fail_stage:
            message = "stub: %s failed" % sid
            stages[sid] = _entry("failed", error=message)
            settle("failed")
            update(state="failed", message="失败", error=message, finished_at=now())
            event("end")
            return 1
        for rel in STAGE_FILES.get(sid, []):
            path = job_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            if rel.endswith(".ply"):
                path.write_bytes(SPLAT_BYTES)
            elif rel.endswith(".glb"):
                path.write_bytes(b"glTF" + b"\0" * 60)
            elif rel.endswith(".json"):
                path.write_text(json.dumps({"stub": sid, "version": 1}), encoding="utf-8")
            else:
                path.write_bytes(b"stub")
        seconds = round(time.monotonic() - started, 3)
        record = {"fingerprint": fingerprints[sid], "finished_at": now(), "seconds": seconds,
                  "result": {"summary": STAGE_SUMMARY.get(sid, {})}}
        if sid == "export":
            record["inputs"] = sorted(d for d in wf.consumed(workflow, "export")
                                      if stages[d]["state"] == "done")
        write_json_atomic(job_dir / "stages" / (sid + ".json"), record)
        stages[sid] = _entry("done", record)
        update(stage=sid, stage_progress=1.0, outputs=outputs_now())

    update(state="done", stage=None, stage_progress=None, message="完成", finished_at=now(),
           outputs=outputs_now())
    log("任务完成")
    event("end")
    if args.mode == "linger":
        pause(LIFETIME, cancellable=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
