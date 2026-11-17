"""Command line entry point.

    scan3d.py serve                 start the local web UI
    scan3d.py run <folder|video>    run one scan from the command line
    scan3d.py list | cancel <job> | doctor
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


def cmd_run(args) -> int:
    import pipeline

    name = args.name or "scan-" + _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    overrides = {}
    for item in args.set or []:
        key, sep, value = item.partition("=")
        if not sep:
            print(f"--set expects key=value, got {item!r}", file=sys.stderr)
            return 64
        try:
            overrides[key] = json.loads(value)
        except json.JSONDecodeError:
            overrides[key] = value
    return pipeline.run_job(Path(args.input).resolve(), name, args.quality, mesh=not args.no_mesh,
                            holdout=args.eval, fresh=args.fresh, overrides=overrides)


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
        rows.append((s.get("started_at", ""), s.get("job", status.parent.name),
                     s.get("state", "?"), s.get("stage", ""), s.get("message", "")))
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
    """Show or install the network weights the current settings need."""
    import models

    settings = config.load_settings()
    if args.accept:
        models.accept_licences([i.strip() for i in args.accept.split(",") if i.strip()])
    if args.download is not None:
        features = [f.strip() for f in args.download.split(",") if f.strip()] or None
        models.download_missing(settings, features=features,
                                log_path=config.TOOLS / "model-download.log")
    state = models.status(settings)
    for key, lic in state["licences"].items():
        mark = "accepted" if lic["accepted"] else "not accepted"
        print(f"licence  {key:9} {mark:13} {lic['name']}  {lic['url']}")
    for feature, info in state["features"].items():
        use = "on " if info["enabled"] else "off"
        if info["ready"]:
            print(f"ok       {feature:13} {use}")
        elif info["licences_needed"]:
            print(f"LICENCE  {feature:13} {use}  accept first: {', '.join(info['licences_needed'])}")
        else:
            print(f"MISSING  {feature:13} {use}  {', '.join(info['files_missing'])} "
                  f"(about {info['download_mb']} MB)")
    if state["files_missing"]:
        print()
        print(f"{len(state['files_missing'])} file(s) to download for the features that are "
              f"on, about {state['download_mb']} MB: scan3d.py models --download")
    return 1 if state["files_missing"] else 0


def cmd_doctor(_args) -> int:
    import models

    ok = True
    for name, path in config.TOOL_PATHS.items():
        present = path.exists()
        ok &= present
        print(f"{'ok     ' if present else 'MISSING'}  {name:8} {path}")
    for name in models.status(config.load_settings())["files_missing"]:
        ok = False
        print(f"MISSING  model    {name}")
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

    run = sub.add_parser("run", help="run one scan")
    run.add_argument("input", help="folder of photos and/or videos, or one video file")
    run.add_argument("--name", help="job name (default: scan-<date>-<time>); an existing name resumes")
    run.add_argument("--quality", choices=config.QUALITIES, default="standard")
    run.add_argument("--no-mesh", action="store_true", help="only produce the splat")
    run.add_argument("--fresh", action="store_true", help="discard earlier results of this job")
    run.add_argument("--eval", action="store_true",
                     help="hold out every 8th image and report quality metrics")
    run.add_argument("--set", action="append", metavar="KEY=VALUE",
                     help="override one value of the quality preset, e.g. --set iterations=7000")
    run.set_defaults(func=cmd_run)

    serve = sub.add_parser("serve", help="start the local web UI")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--no-browser", action="store_true")
    serve.set_defaults(func=cmd_serve)

    sub.add_parser("list", help="list jobs").set_defaults(func=cmd_list)
    cancel = sub.add_parser("cancel", help="stop a running job")
    cancel.add_argument("job")
    cancel.set_defaults(func=cmd_cancel)
    sub.add_parser("doctor", help="check the installation").set_defaults(func=cmd_doctor)
    mod = sub.add_parser("models", help="show or install the network weights the settings need")
    mod.add_argument("--download", nargs="?", const="", metavar="FEATURES",
                     help="download what the features that are on still need; or name "
                          "features (comma separated) to fetch theirs, e.g. dense_seed")
    mod.add_argument("--accept", metavar="IDS",
                     help="record that you accept these third-party licences (comma separated ids)")
    mod.set_defaults(func=cmd_models)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
