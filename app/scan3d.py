"""Command line entry point.

    scan3d.py serve                 start the local web UI
    scan3d.py run <folder|video> [--name N] [--template T] [--enable a,b] [--disable a,b]
                 [--set stage.param=value ...] [--to STAGE] [--only a,b] [--force a,b] [--fresh]
                                    run (or continue) one scan: the stages of its workflow
    scan3d.py stages [--job N]      the stage table of a job (state of every stage)
    scan3d.py list | cancel <job> | doctor | models

`run`: with an existing runs/N/workflow.json it is read first and the flags change it (and it is
saved back); `--template T` starts again from that template; with no workflow file the template
(default: the user's default, else `standard`) is used. `--to STAGE` runs up to a stage (and what
it needs that is not done), `--only a,b` exactly those stages, `--force a,b` runs them again, and
whatever consumes them. `--fresh` deletes the job's results first. Stage ids: ingest masks sfm
normals dense_seed train dense mesh coverage export.

Exit codes of `run`: 0 done, 1 failed, 2 cancelled, 64 bad request (message on stderr).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import sys
from pathlib import Path

# The embeddable Python ignores the script directory when building sys.path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402


def _fix_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream.isatty():
                stream.reconfigure(errors="replace")
            else:
                # When output is piped (the web server reads it) the default would be the
                # system code page, which cannot hold Chinese on every machine.
                stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _ids(values) -> list:
    """Stage ids from repeated comma-separated options."""
    return [i.strip() for v in values or [] for i in v.split(",") if i.strip()]


def cmd_run(args) -> int:
    import pipeline
    import workflow

    name = args.name or "scan-" + _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    try:
        flow = pipeline.build_workflow(config.RUNS / name, args.template, _ids(args.enable),
                                       _ids(args.disable), args.set or [])
        return pipeline.run_job(Path(args.input).resolve() if args.input else None, name, to=args.to,
                                only=_ids(args.only), force=_ids(args.force), fresh=args.fresh,
                                workflow=flow)
    except workflow.WorkflowError as e:
        print("流程设置不正确：" + "；".join(e.problems), file=sys.stderr)
        return 64
    except workflow.PlanError as e:
        print(e.message, file=sys.stderr)
        return 64


def cmd_stages(args) -> int:
    import pipeline

    job = args.job
    if not job:
        jobs = sorted((p.parent for p in config.RUNS.glob("*/status.json")),
                      key=lambda p: (p / "status.json").stat().st_mtime)
        if not jobs:
            print("没有任务；用 --job 指定名字", file=sys.stderr)
            return 64
        job = jobs[-1].name
    job_dir = config.RUNS / job
    if not job_dir.is_dir():
        print(f"没有这个任务：{job}", file=sys.stderr)
        return 64
    view = pipeline.job_view(job_dir)
    print(f"任务 {job}，模板 {view['template']}")
    for sid, label, state, facts in pipeline.stage_table(job_dir):
        print(f"{sid:11} {label:10} {state:9} {facts}")
    return 0


def cmd_serve(args) -> int:
    import server

    server.serve(args.port, open_browser=not args.no_browser)
    return 0


def cmd_list(_args) -> int:
    rows = []
    for status in sorted(config.RUNS.glob("*/status.json")):
        try:
            s = json.loads(status.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows.append((s.get("started_at") or "", s.get("job", status.parent.name),
                     s.get("state", "?"), s.get("stage") or "", s.get("message", "")))
    for started, job, state, stage, message in sorted(rows):
        print(f"{started[:19]:19}  {job:28}  {state:9}  {stage:8}  {message}")
    return 0


def cmd_cancel(args) -> int:
    job_dir = config.RUNS / args.job
    if not job_dir.is_dir():
        print(f"no such job: {args.job}", file=sys.stderr)
        return 1
    (job_dir / "cancel").touch()
    print("cancel requested")
    return 0


def cmd_models(args) -> int:
    """Show or install the network weights. Which features a scan uses is its workflow's choice;
    the licences are recorded when a stage that needs the weights runs."""
    import models

    if args.accept:
        models.accept_licences([i.strip() for i in args.accept.split(",") if i.strip()],
                               via="cli")
    if args.download is not None:
        features = [f.strip() for f in args.download.split(",") if f.strip()] or None
        models.download_features(features, log_path=config.TOOLS / "model-download.log")
    table = models.features_table()
    for f in table["features"]:
        lic = f["licence"]
        note = ""
        if lic:
            note = f"  {lic['label']} ({'recorded' if lic['accepted'] else 'recorded when used'})"
        size = sum(x["mb"] for x in f["files"] if not x["installed"])
        if f["ready"]:
            print(f"ok       {f['id']:13}{note}")
        else:
            print(f"MISSING  {f['id']:13} about {size} MB, downloaded when a stage that uses it "
                  f"starts{note}")
    print(f"{table['disk_mb']} MB of weights installed")
    return 0


def cmd_doctor(_args) -> int:
    import models

    ok = True
    for name, path in config.TOOL_PATHS.items():
        present = path.exists()
        ok &= present
        print(f"{'ok     ' if present else 'MISSING'}  {name:8} {path}")
    # Missing weights are not a fault: they are downloaded when a stage that uses them starts.
    free_gb = shutil.disk_usage(config.ROOT).free / 2**30
    print(f"{'ok     ' if free_gb > 20 else 'LOW    '}  disk     {free_gb:.0f} GB free")
    try:
        str(config.ROOT).encode("ascii")
        print(f"ok       path     {config.ROOT}")
    except UnicodeEncodeError:
        # The tools are not reliable when their own install path has non-ASCII characters.
        ok = False
        print(f"PROBLEM  path     {config.ROOT} contains non-ASCII characters; "
              f"move this folder to a path with only English letters")
    if not ok:
        print("\nRun setup.cmd to install missing tools.")
    return 0 if ok else 1


def main(argv=None) -> int:
    _fix_console()
    parser = argparse.ArgumentParser(prog="scan3d", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run one scan (the stages of its workflow)")
    run.add_argument("input", nargs="?", help="folder of photos and/or videos, or one video file "
                                              "(left out: the job's own, from its last run)")
    run.add_argument("--name", help="job name (default: scan-<date>-<time>); an existing name resumes")
    run.add_argument("--template", help="start from this template (default: the workflow the job "
                                        "has, else the default template)")
    run.add_argument("--enable", action="append", metavar="STAGES",
                     help="switch these optional stages on (comma separated)")
    run.add_argument("--disable", action="append", metavar="STAGES",
                     help="switch these optional stages off (comma separated)")
    run.add_argument("--set", action="append", metavar="STAGE.PARAM=VALUE",
                     help="set one parameter, e.g. --set train.iterations=7000 (repeatable)")
    run.add_argument("--to", metavar="STAGE", help="run up to this stage, and what it needs")
    run.add_argument("--only", action="append", metavar="STAGES",
                     help="run exactly these stages (comma separated)")
    run.add_argument("--force", action="append", metavar="STAGES",
                     help="run these stages again even when done (and what consumes them)")
    run.add_argument("--fresh", action="store_true", help="discard earlier results of this job")
    run.set_defaults(func=cmd_run)

    stages = sub.add_parser("stages", help="show the state of every stage of a job")
    stages.add_argument("--job", help="job name (default: the most recent one)")
    stages.set_defaults(func=cmd_stages)

    serve = sub.add_parser("serve", help="start the local web UI")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--no-browser", action="store_true")
    serve.set_defaults(func=cmd_serve)

    sub.add_parser("list", help="list jobs").set_defaults(func=cmd_list)
    cancel = sub.add_parser("cancel", help="stop a running job")
    cancel.add_argument("job")
    cancel.set_defaults(func=cmd_cancel)
    sub.add_parser("doctor", help="check the installation").set_defaults(func=cmd_doctor)
    mod = sub.add_parser("models", help="show or install the network weights")
    mod.add_argument("--download", nargs="?", const="", metavar="FEATURES",
                     help="download the weights of the default template's stages; or name "
                          "features (comma separated), e.g. sam3,dense_seed")
    mod.add_argument("--accept", metavar="IDS",
                     help="record third-party licences in advance (comma separated ids); "
                          "normally this happens when a stage that needs them runs")
    mod.set_defaults(func=cmd_models)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
