"""scan3d local web server: upload photos or a video, queue scan jobs, watch them, fetch results.

Standard library only (runs on the embeddable Python). One process, one `ThreadingHTTPServer`,
one worker thread that owns the FIFO queue and the single running pipeline process (the GPU can
serve only one job at a time). The UI is the static files in `app/web/`.

A job is a workflow (app/workflow.py: stages, their parameters and switches) plus its input.
The workflow is chosen when the job is created (a template and an override), kept in
`runs/<job>/workflow.json`, edited with PUT /workflow, and run as often as the user likes:
each run is a request ("everything switched on that is not done", "up to this stage", "only
these stages", "these again") that the server turns into a plan with `workflow.plan()` and
queues. Nothing here knows what a stage does.

Entry points
    serve(port=8765, *, open_browser=True)        blocking, Ctrl+C stops
    create_server(port, *, root=None, runs_dir=None, web_dir=None, token=None, host="0.0.0.0")
    build_run_command(job, input_path, *, to=None, only=None, force=None)
                                                  the one place the pipeline command is built:
                                                  `scan3d.py run <input> --name <job>` plus
                                                  `--to`, `--only`, `--force` when given. The
                                                  workflow is not on the command line: the
                                                  server has written workflow.json already.
    build_preview_command(source, out, max_gaussians)   the same for a browser preview
    build_models_command(features)                the same for a download of network weights
    pipeline_job_view(job_dir), pipeline_write_workflow(job_dir, workflow),
    pipeline_reset_stage(job_dir, stage), models_features(), models_default_features(),
    workflow_needs(workflow), models_dir(), load_user_settings(), save_user_settings(data)
                                                  the only calls into pipeline.py, models.py,
                                                  config.py and the user's settings file; tests
                                                  replace them, so the server can be imported
                                                  and tested with a stub pipeline

Files the server keeps in `runs/<job>/`
    status.json      the contract file (the pipeline owns it while it runs; see "Status" below).
                     The server writes it only while no pipeline process of its own is running
                     for the job: state `created` (job exists, not started; a server-only
                     state), `queued` (with `run`, the request and its plan), and the final
                     state when the pipeline did not leave one (failed with the exit code,
                     cancelled, interrupted).
    workflow.json    the job's resolved workflow. Written by the server when the job is created
                     and by PUT /workflow (through pipeline.write_workflow); read by the
                     pipeline. A job made before workflows existed has none: it is shown with
                     the default template and gets the file at its first edit.
    stages/<id>.json the record of a finished stage (the pipeline's; the server only reads them,
                     through pipeline.job_view, and deletes them through pipeline.reset_stage).
    server-job.json  what the server needs to start the job: source, input path, created_at and
                     `run`, the {to, only, force} of the run last requested (that is what the
                     queue runs). Older files with `quality` and `options` are read and the two
                     fields ignored.
    upload/          uploaded files; this folder is the pipeline's input path.
    upload.part/     partially received uploads (resumable); removed when the job starts.
    launcher.log     stdout/stderr of the pipeline process itself (tracebacks at startup).
    cancel           empty file; its presence asks the pipeline to stop.
    web-preview/     browser previews of the splat that the server built itself, because the
                     job has none in output/ or has outdated ones there (see app/splatconv.py
                     for the names and format).
                     Each has a small .json beside it that records which splat.ply it was made
                     from and what it holds; a preview whose splat has changed is rebuilt.
A folder in `runs/` is a job only if it has a status.json and a valid job name.
`runs/.server-token` holds the pairing token (see "Network exposure").
`<root>/user-settings.json` holds the user's templates and default template (workflow.py).

Queue
    Jobs run strictly one after another, first in first out; a job is queued with the run it
    asked for. After spawning the pipeline the server only reads status.json; if the process
    exits without a final state the job is marked failed with the exit code. When the server
    starts, every job still marked queued or running is marked failed with the message
    "interrupted" (nothing can be running them: the pipeline process is tied to the server
    process and dies with it). Stopping the server kills the running pipeline's process tree.
    A job that is queued or running cannot be edited (409 `job_busy`).

Status (status.json as the clients see it)
    state: created | queued | running | done | failed | cancelled (| interrupted is a failed
    job with that message). `done` means the last run reached its target, not that every
    optional stage ran. Besides the pipeline's own fields (message, error, warnings, outputs,
    stage = the running stage or null, stage_progress, started_at, finished_at ...) the job's
    view has:
      stages  {id: {state, progress, message, seconds, error, summary}} for every stage id;
              state is disabled | pending | queued | running | done | stale | failed
      run     {to, only: [ids], force: [ids], plan: [ids]} of the current or last run
    Fields of old jobs (`quality`, `stage_index`, `stage_count`) are dropped from the view.

HTTP API (JSON, UTF-8). Errors are `{"error": "<code>", "message": "<Chinese text>"}` (plus
the extra fields named below).
    GET    /                                   the UI (app/web/index.html)
    GET    /static/<path>                      UI assets from app/web/ (ETag, 304)
    GET    /api/info                           {app, loopback, stages [ids], default_template,
                                               extensions, max_upload_bytes, server_time};
                                               for loopback clients only: `lan_urls`, the
                                               addresses a phone opens, token included
    GET    /api/workflow/schema                workflow.schema() ({stages: [{id, label, desc,
                                               optional, gpu, deps, soft, features, params:
                                               [{key, label, type bool|int|float|choice|text,
                                               default, help, advanced, min?, max?,
                                               choices?}]}], default_template}) plus
                                               `templates` (as GET /api/templates) and
                                               `features` (as in GET /api/models)
    GET    /api/templates                      {"templates": [{id, label, desc, builtin, base?,
                                               enabled: [stage ids that really run] | null (and
                                               `problems`) }], "default_template": id}
    PUT    /api/templates/<id>                 loopback only. Body {"label", "desc"?, "base"?
                                               (a template id, default "standard"), "stages":
                                               {stage: {"enabled"?, "params"?}}}: save a user
                                               template (it must resolve). A built-in id is
                                               replaced. 400 `bad_workflow` (plus `problems`),
                                               `bad_name`, `bad_request`; 409
                                               `too_many_templates`. -> as GET /api/templates
    DELETE /api/templates/<id>                 loopback only. A user template (a built-in one:
                                               409 `builtin_template`; one that is the base of
                                               another: 409 `template_in_use`; unknown: 404).
                                               Was the default: the default is reset. -> as GET
    PUT    /api/templates/default              loopback only. Body {"id"}: the template new jobs
                                               start from (400 `bad_template`). -> as GET
    GET    /api/jobs                           all jobs, newest first, plus queue and current;
                                               each as the job view below without the large
                                               parts (`stages` is {id: {state, progress}}, no
                                               `workflow`)
    POST   /api/jobs                           create. Body {"name"?, "template"? (default:
                                               the user's default template), "workflow"? (an
                                               override {stage: {"enabled"?, "params"?}} on top
                                               of the template), "path"?, "start"?, "to"?,
                                               "only"?, "force"?}. Without `path`: an upload
                                               job in state `created`. With `path` (loopback
                                               clients only): a job that reads an existing
                                               folder or video on this PC; "start": true queues
                                               it at once, with the run `to`/`only`/`force`
                                               describe (they need `start`). Name missing ->
                                               scan-YYYYMMDD-HHMMSS. -> 201 with the job view.
                                               400 `bad_workflow` (plus `problems` [str]) for
                                               an unknown template or an invalid override; a
                                               plan error (below) with `start`; 400 for the old
                                               fields `quality` and `options`.
    GET    /api/jobs/<job>                     the job view: every field of status.json (also
                                               ones this file does not know, e.g. warnings)
                                               plus: source, template, workflow (the resolved
                                               one, every stage and parameter), changes (what
                                               differs from the template), stages, run,
                                               licences [{id, label, url, stage}] (the licences
                                               that apply to the stages switched on; for
                                               information, nothing is ticked), models_needed
                                               [{file, mb, stage}] (weights that will be
                                               downloaded when their stage starts),
                                               created_at, queue_position, elapsed_seconds,
                                               actions (what this client may do now: run [=
                                               start], cancel, edit, delete, open_folder,
                                               open_viewer), cancelling, uploads {count,
                                               bytes}, output_files [{key, path, name, size}],
                                               logs [names]. A browser preview in output/ that
                                               the viewer does not use (see /preview) also has
                                               `outdated: true`. A workflow file that cannot be
                                               read gives `workflow_error` instead of a
                                               workflow.
    DELETE /api/jobs/<job>                     delete the job directory (refused while running)
    PUT    /api/jobs/<job>/workflow            body: an override {stage: {"enabled"?,
                                               "params"?}} and optionally "template": merged
                                               into the stored workflow, or, with a template,
                                               applied on top of that template's own settings
                                               (the switch of template keeps nothing of the old
                                               settings). Validated, saved. -> the job view.
                                               400 `bad_workflow` (plus `problems`); 409
                                               `job_busy` while queued or running.
    POST   /api/jobs/<job>/run                 queue a run. Body {"to"?: stage, "only"?:
                                               [stages], "force"?: [stages]} (none: everything
                                               switched on that is not done; see
                                               workflow.plan). -> 202 with the job view (its
                                               `run.plan` is what will run). 400 with the code
                                               of the plan error (unknown_stage, stage_disabled,
                                               needs_upstream, nothing_to_do); 400 `no_input`;
                                               409 `already_active`, `uploading`. Weights that
                                               are missing are not an error: they download at
                                               the start of the stage that needs them.
    POST   /api/jobs/<job>/start               the old name of /run
    POST   /api/jobs/<job>/plan                the same body as /run: what it would do, nothing
                                               is queued: {"steps": [{stage, reason (requested |
                                               needed | forced | downstream), label}],
                                               "licences": [...], "models_needed": [...]} for
                                               the stages of the plan; the plan error as in /run,
                                               except that an empty plan (everything done) is a
                                               200 with {"steps": [], "nothing_to_do": true}
    POST   /api/jobs/<job>/stages/<stage>/reset
                                               delete that stage's record and what it alone
                                               produced; stages after it show `stale`.
                                               -> the job view. 400 `unknown_stage`; 409
                                               `job_busy`
    PUT    /api/jobs/<job>/files/<name>        upload one file; the raw request body is the file
                                               (Content-Length required, streamed to disk).
                                               `?offset=N` appends to a partial upload of N bytes.
                                               POST is accepted as an alias. -> 201
    GET    /api/jobs/<job>/files/<name>        {"name","size","complete"} of an upload, so a
                                               client can resume after a dropped connection
    POST   /api/jobs/<job>/cancel              queued: dropped from the queue; running: the
                                               `cancel` file is created and the pipeline stops.
                                               A pipeline that ignores it is killed after
                                               CANCEL_GRACE_SECONDS.
    GET    /api/jobs/<job>/log                 log text. `?file=` one of the names in `logs`
                                               (pipeline.log, then one log per stage that wrote
                                               one in stage order: ingest.log, masks.log,
                                               sfm.log, train.log ..., then launcher.log;
                                               default pipeline.log), `?offset=N` to continue
                                               from a previous `next_offset`, else the tail;
                                               `?max=` bytes (default 65536)
    GET    /api/jobs/<job>/download/<path>     an output file (under output/ or listed in
                                               status.outputs). Supports Range / If-Range, HEAD.
    POST   /api/jobs/<job>/open-folder         open the output folder in Explorer (loopback only)
    POST   /api/jobs/<job>/open-viewer         open the splat in the Spirula viewer (loopback only)
    GET    /api/jobs/<job>/preview             the browser previews of the splat: {"variants":
                                               {"full": {...}, "lite": {...}}, "source": {...}}.
                                               Each variant has `state`: ready (plus size,
                                               gaussians, sh_degree, sh_bits, origin
                                               output|cache), missing, queued, building, failed
                                               (plus error), no_splat; and `wanted_sh_degree`,
                                               the SH degree (view-dependent colour) this
                                               variant should have for this splat: 3 for full
                                               (less if the splat has less), 0 for lite. A file
                                               in output/ that has less (a "full" preview written
                                               before previews kept view-dependent colour, or
                                               with too few bits of it), or that is not an SPZ
                                               file, does not count as ready: `output_outdated`
                                               is true and the preview is built into
                                               web-preview/ like a missing one.
                                               `source` describes splat.ply itself, which the
                                               viewer can also load directly through
                                               /download/<path>: {name, path, size, gaussians,
                                               sh_degree}; null when there is no splat.
    POST   /api/jobs/<job>/preview/<variant>   build a missing preview (variant full | lite) in
                                               the background; answers at once with its state
                                               (202 while queued or building, else 200: ready,
                                               or failed again already).
                                               One conversion runs at a time, in a child process
                                               with a time limit. The job must not be queued
                                               or running (a job that failed or was cancelled
                                               after its splat was made still has one to see).
    GET    /api/jobs/<job>/preview/<variant>   the preview file (SPZ). Range / If-Range, HEAD.
                                               404 `preview_missing` until it is ready.
    GET    /api/jobs/<job>/cameras             up to 200 of the training camera poses, in the
                                               coordinates of splat.ply: {"count", "cameras":
                                               [{"position", "forward", "up"}]}. Empty when the
                                               job folder has no dataset/sparse/images.bin. The
                                               viewer starts from one of them.
    GET    /api/jobs/<job>/coverage            how well the photos saw each part of the scene;
                                               the map is the result of the `coverage` stage,
                                               output/coverage.json (app/coverage.py has the
                                               method): {"state": ready (plus size, and
                                               `stale: true` when the photos or settings
                                               changed since it was made) | missing (plus
                                               `enabled`, whether the stage is switched on) |
                                               queued | building | failed (plus error) |
                                               unavailable (plus reason no_splat and message)}.
                                               Nothing is computed by asking.
    POST   /api/jobs/<job>/coverage            queue `--only coverage` through the normal queue
                                               (the same rules as /run: 400 `stage_disabled`
                                               when the stage is switched off, `needs_upstream`
                                               when there is no splat yet); answers at once with
                                               the state (202 while queued or building, 200
                                               when ready and current: nothing is queued; 409
                                               with the reason as error code when unavailable).
    GET    /api/jobs/<job>/coverage/data       the coverage map (JSON, a few hundred KB; the
                                               format is in app/coverage.py): output/coverage.json
                                               as it is. ETag, HEAD. 404 coverage_missing until
                                               it exists.
    GET    /api/models                         the network weights the stages need, for any
                                               authorised client: {"features": [{id, label,
                                               stage, files: [{name, mb, installed}], licence:
                                               {id, label, url, accepted} | null, ready}],
                                               "disk_mb" (installed weights), "download_mb"
                                               (still to download), "free_mb" (free disk space),
                                               "download", "can_manage"}. `licence` is the one
                                               that applies to the weights, shown to the user;
                                               there is no consent step (switching the stage on
                                               is the choice, the pipeline records it).
                                               `download` is the one download this server runs
                                               or ran last: {state: idle | running | done |
                                               failed, features (the ones being fetched), error
                                               (null unless failed), downloaded_mb, total_mb,
                                               files: [{name, size_mb, downloaded_mb}]}; the
                                               numbers are read from the files on disk, partial
                                               ones included, each time. `can_manage` is true
                                               for loopback clients: only they may download.
    POST   /api/models/download                loopback only. Body {"features": [<id>, ...]}:
                                               fetch the weights these features lack (the same
                                               download the stage would start by itself). Empty
                                               or missing: the features the default template
                                               uses. 400
                                               `bad_feature` for an unknown id; 409
                                               `download_busy` while a download runs; 507
                                               `disk_full`. -> 202 with the overview when a
                                               download started, 200 with it when nothing was
                                               missing. One download at a time, in a child
                                               process with a time limit, killed when the server
                                               stops (what was fetched stays in
                                               tools/models/_part and the next download
                                               continues from there). Whether it worked is
                                               judged by reading the table again, not by the
                                               exit code.

Network exposure
    There is one mode. The server always listens on every interface (0.0.0.0), so the same
    program serves this PC and the phones on its network; if that address cannot be bound the
    server does not start (no fallback to another address or port).
    Requests from this PC itself (loopback) never need a token. Every request from any other
    address must carry the pairing token: `?token=` once (the server answers with a cookie that
    lasts a year), then the `scan3d_token` cookie, or the `X-Scan3D-Token` header. The token is
    created at the first start and kept in `runs/.server-token`, so the address a phone has
    bookmarked keeps working after a restart; delete that file to pair all phones anew.
    The token and the phone addresses are shown only to loopback clients. "Loopback only"
    actions (writing templates, downloading weights, opening a folder or the viewer, jobs
    that read a path on this PC) stay loopback-only even with a valid token. Requests whose
    Host header is a DNS name other than localhost, and state-changing requests whose Origin
    differs from Host, are refused (DNS rebinding, CSRF).
"""
from __future__ import annotations

import base64
import collections
import copy
import ctypes
import datetime as _dt
import hashlib
import hmac
import ipaddress
import json
import logging
import os
import re
import secrets
import shutil
import socket
import socketserver
import struct
import subprocess
import sys
import threading
import time
import unicodedata
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
WEB_DIR = APP_DIR / "web"

try:  # imported as part of a package
    from . import proc as _proc  # type: ignore[import-not-found]
    from . import splatconv as _splatconv  # type: ignore[import-not-found]
    from . import workflow as _wf  # type: ignore[import-not-found]
except ImportError:  # run as a plain script next to proc.py (the embeddable Python's
    # ._pth file can keep the script folder off sys.path, so add it explicitly)
    if str(APP_DIR) not in sys.path:
        sys.path.insert(0, str(APP_DIR))
    import proc as _proc  # type: ignore[no-redef]
    import splatconv as _splatconv  # type: ignore[no-redef]
    import workflow as _wf  # type: ignore[no-redef]

log = logging.getLogger("scan3d.server")

# ---------------------------------------------------------------------------- configuration

IMAGE_EXTENSIONS = ("jpg", "jpeg", "png", "webp", "heic", "heif", "tif", "tiff", "bmp")
VIDEO_EXTENSIONS = ("mp4", "mov", "m4v", "mkv", "avi", "webm")
ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS + VIDEO_EXTENSIONS

STAGES = tuple(_wf.STAGE_ORDER)     # the stage ids, in execution order (workflow.py owns them)

STATE_CREATED = "created"  # server-only: the job exists but was never started
ACTIVE_STATES = ("queued", "running")
TERMINAL_STATES = ("done", "failed", "cancelled")
ALL_STATES = (STATE_CREATED,) + ACTIVE_STATES + TERMINAL_STATES

INTERRUPTED_MESSAGE = "interrupted"

SOCKET_TIMEOUT = 30.0            # seconds a client may stay silent mid-request
UPLOAD_CHUNK = 1 << 20           # bytes read from the socket and written to disk at a time
DOWNLOAD_CHUNK = 64 * 1024
UPLOAD_MIN_RATE = 1024           # bytes/second averaged over UPLOAD_RATE_WINDOW, else abort
UPLOAD_RATE_WINDOW = 60.0
MAX_UPLOAD_BYTES = 64 * 1024 ** 3
MAX_FILES_PER_JOB = 5000
DISK_RESERVE_BYTES = 512 * 1024 ** 2   # refuse uploads that would leave less free space
MAX_JSON_BODY = 64 * 1024
SMALL_BODY = 1 << 20             # unread request bodies up to this size are drained, not cut
MAX_LOG_BYTES = 512 * 1024
DEFAULT_LOG_BYTES = 64 * 1024

CANCEL_GRACE_SECONDS = 30.0      # after `cancel` appears, the pipeline gets this long to stop
EXIT_GRACE_SECONDS = 30.0        # after a terminal status, the process gets this long to exit
JOB_TIME_LIMIT_SECONDS = 24 * 3600.0   # hard cap on one pipeline run
PATH_CHECK_TIMEOUT = 8.0         # checking a user-supplied path (dead network shares block)
STALE_CHECK_SECONDS = 3.0        # how often the worker looks for jobs whose process died
STATUS_CACHE_SECONDS = 0.4       # see App.read_status

PREVIEW_VARIANTS = {            # variant -> (file name, most Gaussians kept or None for all)
    "full": (_splatconv.PREVIEW_NAME, None),
    "lite": (_splatconv.PREVIEW_LITE_NAME, _splatconv.LITE_MAX_GAUSSIANS),
}
PREVIEW_CACHE_DIR = "web-preview"       # in the job folder: previews the server built itself
PREVIEW_TIME_LIMIT_SECONDS = 600.0      # one conversion; 2,000,000 Gaussians take 15-30 s
MAX_VIEWER_CAMERAS = 200
COVERAGE_FILE = ("output", "coverage.json")     # written by the coverage stage

STATUS_DROPPED = ("quality", "stage_index", "stage_count", "stage_total")   # fields of old jobs
MAX_USER_TEMPLATES = 50
MAX_TEMPLATE_LABEL = 60
MODELS_TIME_LIMIT_SECONDS = 12 * 3600.0  # one download of weights (all of them: about 5 GB)
MODELS_PART_DIR = "_part"                # in the models folder: files still being fetched

BIND_HOST = "0.0.0.0"            # every interface: the one and only mode
TOKEN_FILE = ".server-token"     # in runs/
TOKEN_COOKIE = "scan3d_token"
TOKEN_COOKIE_MAX_AGE = 365 * 24 * 3600
TOKEN_HEADER = "X-Scan3D-Token"
TOKEN_ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"   # no look-alikes: typed on a phone
TOKEN_LENGTH = 12
AUTH_MAX_FAILURES = 10           # wrong tokens per address ...
AUTH_WINDOW_SECONDS = 60.0       # ... within this window, then 429 until it passes

STATIC_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".json": "application/json; charset=utf-8",
}
DOWNLOAD_TYPES = {
    ".json": "application/json; charset=utf-8",
    ".glb": "model/gltf-binary",
    ".txt": "text/plain; charset=utf-8",
    ".log": "text/plain; charset=utf-8",
}
# The page loads nothing from other hosts. The 3D viewer's libraries need three allowances:
# compiling the WebAssembly they carry inside their own file, a worker started from a blob,
# and reading blobs (textures inside a GLB). The import map in index.html is the one inline
# script; it is allowed by its hash (see App.page_csp).
PAGE_CSP = ("default-src 'self'; script-src 'self' 'wasm-unsafe-eval'%s; worker-src 'self' blob:; "
            "connect-src 'self' blob:; img-src 'self' data: blob:; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'; form-action 'self'")
_IMPORTMAP_RE = re.compile(rb'<script type="importmap">(.*?)</script>', re.S)

CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200
DETACHED_PROCESS = 0x00000008


class ServerStartError(RuntimeError):
    """The server could not start (the port is taken, or 0.0.0.0 cannot be bound)."""


class ApiError(Exception):
    """An error that is reported to the client as JSON."""

    def __init__(self, status: int, code: str, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.extra = extra


# ---------------------------------------------------------------------------- the pipeline contract

def build_run_command(job: str, input_path: str, *, to: Optional[str] = None,
                      only: Optional[Sequence[str]] = None,
                      force: Optional[Sequence[str]] = None) -> List[str]:
    """The command that runs the pipeline for one job. Tests replace this with a stub.

    The workflow itself is not on the command line: the server has already written
    `runs/<job>/workflow.json`, which the pipeline reads. `to`, `only` and `force` are the
    request of the run (see POST /api/jobs/<job>/run); each is passed only when given."""
    cmd = [sys.executable, str(APP_DIR / "scan3d.py"), "run", str(input_path), "--name", job]
    if to:
        cmd += ["--to", to]
    if only:
        cmd += ["--only", ",".join(only)]
    if force:
        cmd += ["--force", ",".join(force)]
    return cmd


def build_preview_command(source: os.PathLike, out: os.PathLike,
                          max_gaussians: Optional[int]) -> List[str]:
    """The command that writes one browser preview of a splat. Tests may replace this too."""
    cmd = [sys.executable, str(APP_DIR / "splatconv.py"), str(source), str(out)]
    if max_gaussians is not None:
        cmd += ["--max", str(max_gaussians)]
    return cmd


def build_models_command(features: Optional[Sequence[str]]) -> List[str]:
    """The command that downloads network weights: those of the features named or, with no
    names, everything that is missing. Tests replace this as well."""
    cmd = [sys.executable, str(APP_DIR / "scan3d.py"), "models", "--download"]
    if features:
        cmd.append(",".join(features))
    return cmd


# The server's only calls into the pipeline, models.py, config.py and the user's settings file.
# The modules are imported when first needed and reached through these functions, which tests
# replace (so the server can be tested with a stub pipeline and without the real install).

def _app_module(name: str) -> Any:
    if str(APP_DIR) not in sys.path:    # all are plain modules next to this file
        sys.path.insert(0, str(APP_DIR))
    return __import__(name)


def pipeline_job_view(job_dir: Path) -> dict:
    """{"workflow", "template", "stages", "run"} of a job, from its files (pipeline.job_view).
    A job without workflow.json gets the default template. _wf.WorkflowError: damaged file."""
    return _app_module("pipeline").job_view(job_dir)


def pipeline_write_workflow(job_dir: Path, workflow: dict) -> None:
    """Store a (resolved, validated) workflow as the job's workflow.json."""
    _app_module("pipeline").write_workflow(job_dir, workflow)


def pipeline_reset_stage(job_dir: Path, stage: str) -> None:
    """Delete a stage's record and what it alone produced (ValueError: cannot be reset)."""
    _app_module("pipeline").reset_stage(job_dir, stage)


def models_features() -> dict:
    """{"features": [...], "disk_mb": int}: the table of the features that need network
    weights (models.features_table)."""
    return _app_module("models").features_table()


def models_default_features() -> List[str]:
    """The ids of the features the default template's stages use (models.default_features):
    what a download without names fetches."""
    return list(_app_module("models").default_features())


def workflow_needs(workflow: dict) -> dict:
    """{"licences_needed": [{id, label, url, stage}], "models_needed": [{file, mb, stage}]}
    of the stages a workflow switches on (models.needs)."""
    return _app_module("models").needs(workflow)


def models_dir() -> Path:
    """Where the weights are installed; files still being fetched are in its _part folder."""
    return Path(_app_module("config").MODELS)


def load_user_settings() -> dict:
    """The user's templates and default template (user-settings.json in the project root)."""
    return _wf.load_user()


def save_user_settings(data: dict) -> None:
    _wf.save_user(data)


# ---------------------------------------------------------------------------- names and paths

_JOB_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_BAD_COMPONENT_CHARS = re.compile(r'[<>:"/\\|?*]')
_RESERVED_NAMES = frozenset(
    ["CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"]
    + ["COM%s" % c for c in "0123456789"] + ["LPT%s" % c for c in "0123456789"])
MAX_COMPONENT_LENGTH = 120


def is_reserved_windows_name(name: str) -> bool:
    """CON, NUL, COM1 ... are devices on Windows, with or without an extension."""
    return name.split(".")[0].strip(" ").upper() in _RESERVED_NAMES


def valid_job_name(name: Any) -> bool:
    """Job names are ASCII letters, digits, '-' and '_' (they become folder names and command
    line arguments for tools that mishandle anything else), and never a Windows device name."""
    return (isinstance(name, str) and _JOB_NAME_RE.match(name) is not None
            and not is_reserved_windows_name(name))


def valid_component(name: Any) -> bool:
    """One file or folder name that is safe to join onto a directory on Windows."""
    if not isinstance(name, str) or not name or len(name) > MAX_COMPONENT_LENGTH:
        return False
    if name in (".", "..") or name[0] in ". " or name[-1] in ". ":
        return False
    if _BAD_COMPONENT_CHARS.search(name):
        return False
    # control characters, invisible format characters (right-to-left override ...), surrogates
    if any(unicodedata.category(ch) in ("Cc", "Cf", "Cs") for ch in name):
        return False
    return not is_reserved_windows_name(name)


def file_extension(name: str) -> str:
    """Lower-case extension without the dot ('' when there is none)."""
    dot = name.rfind(".")
    return name[dot + 1:].lower() if dot > 0 else ""


def safe_join(base: Path, parts: Sequence[str]) -> Path:
    """Join `parts` under `base`, refusing anything that could leave it.

    Each part is validated on its own, and the result is resolved so a link or junction inside
    the tree cannot point the server somewhere else.
    """
    if not parts or not all(valid_component(p) for p in parts):
        raise ApiError(400, "bad_path", "路径不合法")
    target = base.joinpath(*parts)
    try:
        real_base = os.path.normcase(os.path.realpath(base))
        real_target = os.path.normcase(os.path.realpath(target))
    except (OSError, ValueError):
        raise ApiError(400, "bad_path", "路径不合法")
    if not real_target.startswith(real_base.rstrip("\\/") + os.sep):
        raise ApiError(400, "bad_path", "路径不合法")
    return target


def is_loopback_address(address: str) -> bool:
    """True when the peer address of a connection is this machine itself."""
    try:
        ip = ipaddress.ip_address(str(address).split("%", 1)[0])
    except ValueError:
        return False
    mapped = getattr(ip, "ipv4_mapped", None)
    return bool((mapped or ip).is_loopback)


def generate_token() -> str:
    return "".join(secrets.choice(TOKEN_ALPHABET) for _ in range(TOKEN_LENGTH))


_TOKEN_RE = re.compile(r"[%s]{%d}\Z" % (TOKEN_ALPHABET, TOKEN_LENGTH))


def load_or_create_token(runs_dir: os.PathLike) -> str:
    """The pairing token kept in `runs/.server-token`; created when missing or damaged.

    It survives restarts so that the address a phone has saved keeps working. If the file
    cannot be written the token still works until this process ends.
    """
    path = Path(runs_dir) / TOKEN_FILE
    try:
        with open(path, "r", encoding="ascii") as f:
            saved = f.read(256).strip()
    except (OSError, UnicodeError):
        saved = ""
    if _TOKEN_RE.match(saved):
        return saved
    token = generate_token()
    tmp = path.with_name("%s.%d.tmp" % (path.name, os.getpid()))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w", encoding="ascii") as f:
            f.write(token + "\n")
        os.replace(tmp, path)
    except OSError:
        log.warning("could not save the pairing token to %s; phones will have to pair again "
                    "after a restart", path)
        try:
            os.unlink(tmp)
        except OSError:
            pass
    return token


# ---------------------------------------------------------------------------- small helpers

def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def parse_iso(value: Any) -> Optional[_dt.datetime]:
    if not isinstance(value, str):
        return None
    try:
        parsed = _dt.datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.astimezone()


def read_json(path: Path, attempts: int = 4) -> Optional[dict]:
    """Read a JSON object. None when the file is missing; ValueError when it stays unreadable.

    The writer replaces the file atomically, and on Windows a replace and a read can collide
    for an instant, so a failed read is retried.
    """
    for attempt in range(attempts):
        try:
            with open(path, "rb") as f:
                raw = f.read()
            data = json.loads(raw.decode("utf-8-sig"))
            if isinstance(data, dict):
                return data
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            pass
        if attempt + 1 < attempts:
            time.sleep(0.03)
    raise ValueError("unreadable JSON: %s" % path)


def write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_name("%s.%d.%d.tmp" % (path.name, os.getpid(), threading.get_ident()))
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    for attempt in range(40):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # someone has the target open right now
            time.sleep(0.025)
    try:
        os.replace(tmp, path)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def pid_alive(pid: Any) -> bool:
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except OSError:
            return True
        return True
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel32.OpenProcess(0x1000, 0, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return ctypes.get_last_error() == 5  # access denied: it exists, it is just not ours
    try:
        code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == 259  # STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def call_with_timeout(fn: Callable[[], Any], timeout: float) -> Any:
    """Run `fn` in a helper thread; TimeoutError if it has not returned in time."""
    box: Dict[str, Any] = {}

    def runner() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - handed to the caller
            box["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError("no answer within %.0f s" % timeout)
    if "error" in box:
        raise box["error"]
    return box.get("value")


def spawn_detached(cmd: Sequence[str]) -> None:
    """Start a GUI program that outlives the server and is never waited for."""
    flags = (DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0
    subprocess.Popen(list(cmd), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, close_fds=True, creationflags=flags)


def remove_tree(path: Path) -> bool:
    """Delete a directory tree, including read-only files. True when it is gone."""
    shutil.rmtree(path, ignore_errors=True)
    if not os.path.lexists(path):
        return True
    for folder, _dirs, files in os.walk(path):
        for name in files:
            try:
                os.chmod(os.path.join(folder, name), 0o666)
            except OSError:
                pass
    shutil.rmtree(path, ignore_errors=True)
    return not os.path.lexists(path)


def lan_addresses() -> List[str]:
    """IPv4 addresses of this PC that a phone on the same network could reach."""
    found: List[str] = []

    def add(address: str) -> None:
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            return
        if ip.version == 4 and not (ip.is_loopback or ip.is_link_local or ip.is_unspecified):
            if address not in found:
                found.append(address)

    # A UDP "connect" sends nothing; it only makes the OS pick the outgoing interface.
    for probe in ("192.0.2.1", "10.255.255.255"):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.settimeout(0.5)
            sock.connect((probe, 9))
            add(sock.getsockname()[0])
        except OSError:
            pass
        finally:
            sock.close()
    try:
        for address in call_with_timeout(
                lambda: socket.gethostbyname_ex(socket.gethostname())[2], 2.0):
            add(address)
    except (OSError, TimeoutError):
        pass
    # Home and office networks first; VPN and virtual adapters use other ranges.
    home = [ipaddress.ip_network(n) for n in ("192.168.0.0/16", "10.0.0.0/8", "172.16.0.0/12")]
    found.sort(key=lambda a: not any(ipaddress.ip_address(a) in n for n in home))
    return found


def read_colmap_cameras(images_bin: os.PathLike, limit: int) -> Tuple[int, List[dict]]:
    """Poses from a COLMAP images.bin: (number of images, at most `limit` of them, evenly
    spread). Each is {"position", "forward", "up"} in the model's own frame.

    Raises OSError, ValueError or struct.error for a file that is not what it should be.
    """
    cameras: List[dict] = []
    with open(images_bin, "rb") as f:
        (count,) = struct.unpack("<Q", f.read(8))
        if count > 1_000_000:
            raise ValueError("implausible image count")
        step = max(1, -(-count // max(1, limit)))
        for index in range(count):
            _image_id, qw, qx, qy, qz, tx, ty, tz, _camera = struct.unpack("<I7dI", f.read(64))
            for _ in range(4096):   # the image name, zero-terminated
                byte = f.read(1)
                if byte in (b"\x00", b""):
                    break
            else:
                raise ValueError("image name too long")
            (points,) = struct.unpack("<Q", f.read(8))
            f.seek(points * 24, 1)
            if index % step:
                continue
            norm = (qw * qw + qx * qx + qy * qy + qz * qz) ** 0.5
            if not norm > 1e-12:
                continue
            qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
            # Rows of the world-to-camera rotation: the camera's right, down and viewing
            # directions in the world. The camera centre is -R^T t.
            right = (1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw))
            down = (2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw))
            ahead = (2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy))
            position = [-(right[i] * tx + down[i] * ty + ahead[i] * tz) for i in range(3)]
            values = position + list(ahead) + list(down)
            if not all(v == v and abs(v) < 1e12 for v in values):
                continue
            cameras.append({"position": position, "forward": list(ahead),
                            "up": [-c for c in down]})
    return count, cameras


def read_train_transform(path: os.PathLike) -> Optional[Tuple[float, List[List[float]], List[float]]]:
    """(scale, 3x3 rotation, translation) of `train_from_world` in a scene_transform.json:
    p_train = scale * rotation @ p_world + translation. None when the file is missing;
    ValueError when it is not usable."""
    data = read_json(Path(path))
    if data is None:
        return None
    try:
        part = data["train_from_world"]
        scale = float(part.get("scale", 1))
        matrix = [[float(v) for v in row] for row in part["rotation"]["matrix_3x3"]]
        translation = [float(v) for v in part.get("translation", [0, 0, 0])]
    except (KeyError, TypeError, ValueError, AttributeError):
        raise ValueError("unusable scene transform") from None
    numbers = [scale] + translation + [v for row in matrix for v in row]
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix) or len(translation) != 3 \
            or not all(v == v and abs(v) < 1e12 for v in numbers) or scale <= 0:
        raise ValueError("unusable scene transform")
    return scale, matrix, translation


class RangeNotSatisfiable(Exception):
    pass


def parse_range(header: Optional[str], size: int) -> Optional[Tuple[int, int]]:
    """Parse a Range header for a file of `size` bytes.

    Returns (first, last) inclusive, or None when the whole file should be sent (no header,
    a unit other than bytes, several ranges, or a malformed value: all may be ignored).
    Raises RangeNotSatisfiable when the range lies outside the file.
    """
    if not header:
        return None
    unit, sep, spec = header.strip().partition("=")
    if not sep or unit.strip().lower() != "bytes" or "," in spec:
        return None
    match = re.fullmatch(r"\s*([0-9]*)\s*-\s*([0-9]*)\s*", spec)
    if match is None:
        return None
    first_s, last_s = match.group(1), match.group(2)
    if first_s == "":
        if last_s == "":
            return None
        suffix = int(last_s)
        if suffix == 0 or size == 0:
            raise RangeNotSatisfiable()
        return max(0, size - suffix), size - 1
    first = int(first_s)
    if first >= size:
        raise RangeNotSatisfiable()
    last = min(int(last_s), size - 1) if last_s else size - 1
    if last < first:
        return None
    return first, last


def _trim_partial_utf8(data: bytes) -> bytes:
    """Drop a multi-byte character that is cut off at the end of `data`."""
    for back in range(1, min(4, len(data)) + 1):
        byte = data[-back]
        if byte < 0x80:
            return data
        if byte >= 0xC0:
            need = 2 if byte < 0xE0 else 3 if byte < 0xF0 else 4
            return data if back >= need else data[:-back]
    return data


def _say(text: str) -> None:
    """Print to the console without ever failing on a code page that lacks the characters."""
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.encode(encoding, "replace").decode(encoding), flush=True)
    except (OSError, ValueError):
        pass


# ---------------------------------------------------------------------------- workflow requests

_TEMPLATE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
RESERVED_TEMPLATE_IDS = ("default",)        # /api/templates/default sets the default template


def valid_template_id(value: Any) -> bool:
    return (isinstance(value, str) and _TEMPLATE_ID_RE.match(value) is not None
            and value not in RESERVED_TEMPLATE_IDS)


def bad_workflow(error: "_wf.WorkflowError") -> ApiError:
    """The 400 for a workflow, template or override that does not validate."""
    return ApiError(400, "bad_workflow", "流程设置不合法：%s" % "；".join(error.problems)[:600],
                    problems=list(error.problems))


def _stage_ids(raw: Any, name: str) -> List[str]:
    if raw is None:
        return []
    if not isinstance(raw, list) or not all(isinstance(i, str) for i in raw):
        raise ApiError(400, "bad_request", "%s 必须是阶段名称的列表" % name)
    return list(dict.fromkeys(raw))


def parse_run_request(body: dict) -> Tuple[Optional[str], List[str], List[str]]:
    """(to, only, force) of the body of /run, /plan, /start and of a create with `start`.
    Names are checked against the plan, not here (an unknown one is a PlanError)."""
    to = body.get("to")
    if to is not None and not isinstance(to, str):
        raise ApiError(400, "bad_request", "to 必须是一个阶段名称")
    return to or None, _stage_ids(body.get("only"), "only"), _stage_ids(body.get("force"), "force")


def reject_removed_fields(body: dict) -> None:
    """`quality` and `options` are gone: a client that still sends them is told so."""
    for key in ("quality", "options"):
        if key in body:
            raise ApiError(400, "bad_request",
                           "%s 已经取消：请用 template（模板）和 workflow（流程设置）" % key)


def check_override(raw: Any) -> Optional[dict]:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise bad_workflow(_wf.WorkflowError(["workflow 必须是对象：{阶段: {enabled, params}}"]))
    return raw


def changes_from_template(workflow: dict, user: dict) -> dict:
    """The partial override that turns the job's template into this workflow (the template as
    it is now); {} when the template no longer exists."""
    try:
        base = _wf.resolve(workflow.get("template"), user=user)
    except _wf.WorkflowError:
        return {}
    diff: Dict[str, Any] = {}
    for sid in _wf.STAGE_ORDER:
        a, b = workflow["stages"][sid], base["stages"][sid]
        entry: Dict[str, Any] = {}
        if a["enabled"] != b["enabled"]:
            entry["enabled"] = a["enabled"]
        params = {k: v for k, v in a["params"].items() if v != b["params"].get(k)}
        if params:
            entry["params"] = params
        if entry:
            diff[sid] = entry
    return diff


def stage_overview(stages: Any) -> Optional[dict]:
    """{id: {"state", "progress"}} of the full per-stage map: what a list of jobs needs."""
    if not isinstance(stages, dict):
        return None
    return {sid: {"state": (entry or {}).get("state"), "progress": (entry or {}).get("progress")}
            for sid, entry in stages.items() if isinstance(entry, dict)}


def _with_enabled(entry: dict, user: dict) -> dict:
    entry = dict(entry)
    try:
        entry["enabled"] = _wf.effective_enabled(_wf.resolve(entry["id"], user=user))
    except _wf.WorkflowError as exc:
        entry["enabled"] = None
        entry["problems"] = list(exc.problems)
    return entry


def _default_template_id() -> str:
    return load_user_settings().get("default_template") or _wf.DEFAULT_TEMPLATE


def workflow_schema() -> dict:
    """GET /api/workflow/schema: workflow.schema() plus the templates and the table of the
    features that need weights. A models table that cannot be read does not take the
    schema down with it: `features` is empty and `features_error` says why."""
    user = load_user_settings()
    result = _wf.schema()
    result["default_template"] = user.get("default_template") or _wf.DEFAULT_TEMPLATE
    result["templates"] = templates_view(user)["templates"]
    try:
        result["features"] = ModelDownloads.table()
    except ApiError as err:
        result["features"] = []
        result["features_error"] = err.message
    return result


def templates_view(user: Optional[dict] = None) -> dict:
    user = load_user_settings() if user is None else user
    default = user.get("default_template") or _wf.DEFAULT_TEMPLATE
    return {"templates": [_with_enabled(t, user) for t in _wf.templates(user)],
            "default_template": default}


# ---------------------------------------------------------------------------- derived files

class BackgroundBuilder:
    """Work derived from a finished job that the server does itself (browser previews): one
    piece at a time, each in a child process (so a big file neither slows
    the server's own threads nor can it hang them), with a time limit. Requests only queue a
    key (job, what) and report its state; nobody waits for the work. The child dies with the
    server, is stopped when its job is deleted, and is killed after `time_limit` seconds.
    Subclasses implement _build(job, what): None on success, else the message for the user.
    """

    label = "work"                          # in log lines and the thread name
    failure = "无法完成"                     # the start of the user's message when _build raised

    def __init__(self, app: "App", time_limit: float) -> None:
        self.app = app
        self.lock = threading.Lock()
        self.pending: "collections.deque[Tuple[str, str]]" = collections.deque()
        self.current: Optional[Tuple[str, str]] = None
        self.child: Optional[subprocess.Popen] = None
        self.abort_current = False
        self.errors: Dict[Tuple[str, str], str] = {}
        self.time_limit = time_limit
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _queue_state(self, key: Tuple[str, str]) -> dict:
        """{"state": building | queued | failed (plus error) | missing} of work not done."""
        with self.lock:
            if self.current == key:
                return {"state": "building"}
            if key in self.pending:
                return {"state": "queued"}
            if key in self.errors:
                return {"state": "failed", "error": self.errors[key]}
        return {"state": "missing"}

    def _enqueue(self, key: Tuple[str, str]) -> None:
        with self.lock:
            if self.app.stop_event.is_set():
                raise ApiError(503, "stopping", "扫描程序正在停止")
            self.errors.pop(key, None)
            if key != self.current and key not in self.pending:
                self.pending.append(key)
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._loop, daemon=True,
                                                name="scan3d-" + self.label)
                self._thread.start()
        self._wake.set()

    # ---- stopping

    def cancel(self, job: str, timeout: float = 10.0) -> None:
        """Drop everything queued for a job and stop its running child (the job is being
        deleted, and a child process with files open would keep the folder in use)."""
        with self.lock:
            self.pending = collections.deque(k for k in self.pending if k[0] != job)
            for key in [k for k in self.errors if k[0] == job]:
                del self.errors[key]
            busy = self.current is not None and self.current[0] == job
            if busy:
                self.abort_current = True
        deadline = time.monotonic() + timeout
        while busy and time.monotonic() < deadline:
            time.sleep(0.05)
            with self.lock:
                busy = self.current is not None and self.current[0] == job

    def shutdown(self, timeout: float = 10.0) -> None:
        """Called with app.stop_event set: the loop kills its child and ends."""
        with self.lock:
            self.pending.clear()
        self._wake.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)
        child = self.child
        if child is not None and child.poll() is None:  # the loop did not get to it
            _proc.kill_tree(child.pid)

    # ---- the worker

    def _loop(self) -> None:
        app = self.app
        while not app.stop_event.is_set():
            with self.lock:
                key = self.pending.popleft() if self.pending else None
                self.current = key
                self.abort_current = False
            if key is None:
                self._wake.wait(0.5)
                self._wake.clear()
                continue
            error: Optional[str] = None
            try:
                error = self._build(*key)
            except Exception as exc:  # noqa: BLE001 - one bad job must not stop the others
                log.exception("%s %s of %s could not be built", self.label, key[1], key[0])
                error = "%s：%s" % (self.failure, exc)
            with self.lock:
                if error is not None and not self.abort_current:
                    self.errors[key] = error
                self.current = None
                self.child = None

    def _build(self, job: str, what: str) -> Optional[str]:
        raise NotImplementedError

    def _run_child(self, cmd: Sequence[str]) -> Tuple[Optional[int], bytes, Optional[str], float]:
        """Run one child to its end: (exit code, what it printed, None | "stopped" |
        "time_limit", seconds). It is stopped when the server stops or its job is deleted."""
        app = self.app
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        child = subprocess.Popen(
            list(cmd), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=str(app.root), env=env,
            creationflags=(CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0)
        with self.lock:
            self.child = child
        assign = getattr(_proc, "_assign_to_job", None)
        if assign is not None:
            try:
                assign(child)   # dies with this process, like the pipeline
            except Exception:  # noqa: BLE001 - best effort
                pass
        started = time.monotonic()
        reason = None
        output = b""
        while True:
            try:
                output, _ = child.communicate(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                pass
            if app.stop_event.is_set() or self.abort_current:
                reason = "stopped"
            elif time.monotonic() - started > self.time_limit:
                reason = "time_limit"
            if reason is not None:
                _proc.kill_tree(child.pid)
                try:
                    output, _ = child.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                break
        return child.returncode, output or b"", reason, time.monotonic() - started


def _last_error_line(output: bytes) -> str:
    """What a child said last: the "error" of a JSON line, else the last non-empty line."""
    for line in reversed(output.decode("utf-8", "replace").strip().splitlines()):
        try:
            detail = str(json.loads(line).get("error") or "")
        except (ValueError, AttributeError):
            detail = line.strip()
        if detail:
            return detail
    return ""


# ---------------------------------------------------------------------------- browser previews

class PreviewBuilder(BackgroundBuilder):
    """Builds the browser previews a job lacks (see BackgroundBuilder for how).

    A preview is looked for in output/ first (the pipeline writes it there), then in the
    job's web-preview/ folder, where the ones built here are kept.
    """

    label = "preview"
    failure = "无法生成预览"

    def __init__(self, app: "App") -> None:
        super().__init__(app, PREVIEW_TIME_LIMIT_SECONDS)
        self._headers: Dict[str, Tuple[tuple, Any]] = {}    # file -> (stat key, what its header says)

    # ---- where things are

    def source(self, job: str, status: dict) -> Optional[Path]:
        """The splat a preview is made from, or None."""
        outputs = status.get("outputs")
        rel = outputs.get("splat") if isinstance(outputs, dict) else None
        if not isinstance(rel, str):
            rel = "output/splat.ply"
        try:
            path = safe_join(self.app.job_dir(job), rel.replace("\\", "/").split("/"))
        except ApiError:
            return None
        return path if path.is_file() else None

    @staticmethod
    def _source_key(source: Path) -> dict:
        info = source.stat()
        return {"source_size": info.st_size, "source_mtime_ns": info.st_mtime_ns}

    def _header(self, path: Path, reader: Callable[[Path], Any]) -> Any:
        """reader(path), remembered until the file changes (the page asks again and again
        while it waits, and reading a header means opening the file). None when the file
        cannot be read or is not what the reader expects."""
        info = path.stat()
        key = (info.st_mtime_ns, info.st_size)
        cached = self._headers.get(str(path))
        if cached is not None and cached[0] == key:
            return cached[1]
        try:
            value = reader(path)
        except (ValueError, OSError):       # PreviewError is a ValueError
            value = None
        if len(self._headers) > 500:
            self._headers.clear()
        self._headers[str(path)] = (key, value)
        return value

    def source_info(self, source: Path) -> dict:
        """{"gaussians", "sh_degree"} of the splat; {} when its header cannot be read (the
        conversion will say why)."""
        try:
            return self._header(source, _splatconv.splat_info) or {}
        except OSError:
            return {}

    def wanted_sh_degree(self, variant: str, source: Path) -> int:
        """The SH degree a preview of this variant should have: what splatconv writes by
        default for it, limited by what the splat has."""
        return min(_splatconv.default_sh_degree(PREVIEW_VARIANTS[variant][1]),
                   self.source_info(source).get("sh_degree", 0))

    @staticmethod
    def _has_enough_sh(holds: dict, wanted_degree: int) -> bool:
        """Does a preview that holds {"sh_degree", "sh_bits"} count as this variant?"""
        if wanted_degree <= 0:
            return True
        degree, bits = holds.get("sh_degree"), holds.get("sh_bits")
        if not isinstance(degree, int) or degree < wanted_degree:
            return False
        # bits None: every coefficient looked at was zero, so there was nothing to round
        return bits is None or (isinstance(bits, int) and bits >= _splatconv.DEFAULT_SH_BITS)

    def find(self, job: str, variant: str, source: Path
             ) -> Tuple[Optional[Tuple[Path, str, dict]], bool]:
        """((path, "output" | "cache", {"count", "sh_degree", "sh_bits"}) of a usable preview
        or None, whether output/ holds a file by that name that is not usable).

        A preview in output/ was written by the pipeline, possibly by an older version of it:
        its own header says what it is. One in web-preview/ was written by this server, which
        noted beside it what it was made from and what it holds."""
        name = PREVIEW_VARIANTS[variant][0]
        job_dir = self.app.job_dir(job)
        wanted = self.wanted_sh_degree(variant, source)
        outdated = False
        for origin, parts in (("output", ["output", name]), ("cache", [PREVIEW_CACHE_DIR, name])):
            try:
                path = safe_join(job_dir, parts)
                if not path.is_file():
                    continue
                if origin == "cache":
                    # Built from this very splat? (A job can be run again.)
                    made_from = read_json(path.with_name(name + ".json"))
                    key = self._source_key(source)
                    if not isinstance(made_from, dict) \
                            or any(made_from.get(k) != v for k, v in key.items()):
                        continue
                    holds = {"count": made_from.get("gaussians"),
                             "sh_degree": made_from.get("sh_degree"),
                             "sh_bits": made_from.get("sh_bits")}
                    if not isinstance(holds["sh_degree"], int):
                        holds["sh_degree"] = 0      # noted before previews had any SH
                else:
                    holds = self._header(path, _splatconv.read_preview_header)
                if holds is None or not self._has_enough_sh(holds, wanted):
                    # Not an SPZ file at all, or written when previews had no (or only
                    # coarsely rounded) view-dependent colour: not this variant.
                    outdated = outdated or origin == "output"
                    continue
                return (path, origin, holds), outdated
            except (ApiError, OSError, ValueError):
                continue
        return None, outdated

    def ready_file(self, job: str, variant: str, source: Path) -> Optional[Tuple[Path, str, dict]]:
        return self.find(job, variant, source)[0]

    def outdated_outputs(self, job: str, status: dict) -> List[str]:
        """Paths ("output/preview.spz") of the previews in output/ that the viewer does not
        use because they are not what the variant should be (see `find`)."""
        source = self.source(job, status) if status.get("state") in TERMINAL_STATES else None
        if source is None:
            return []
        return ["output/" + PREVIEW_VARIANTS[variant][0] for variant in PREVIEW_VARIANTS
                if self.find(job, variant, source)[1]]

    # ---- what the API reports

    def state(self, job: str, variant: str, status: dict) -> dict:
        result: Dict[str, Any] = {"variant": variant, "name": PREVIEW_VARIANTS[variant][0],
                                  "max_gaussians": PREVIEW_VARIANTS[variant][1]}
        source = self.source(job, status) if status.get("state") in TERMINAL_STATES else None
        if source is None:
            result["state"] = "no_splat"
            return result
        result["wanted_sh_degree"] = self.wanted_sh_degree(variant, source)
        ready, outdated = self.find(job, variant, source)
        if outdated:
            result["output_outdated"] = True
        if ready is not None:
            try:
                result.update(state="ready", size=ready[0].stat().st_size, origin=ready[1],
                              sh_degree=ready[2]["sh_degree"], sh_bits=ready[2]["sh_bits"],
                              gaussians=ready[2]["count"])
                return result
            except OSError:
                pass
        result.update(self._queue_state((job, variant)))
        return result

    def overview(self, job: str) -> dict:
        status = self.app.require_job(job)
        result: Dict[str, Any] = {"variants": {variant: self.state(job, variant, status)
                                               for variant in PREVIEW_VARIANTS}, "source": None}
        source = self.source(job, status) if status.get("state") in TERMINAL_STATES else None
        if source is not None:
            try:
                info = self.source_info(source)
                result["source"] = {
                    "name": source.name,
                    "path": source.relative_to(self.app.job_dir(job)).as_posix(),
                    "size": source.stat().st_size,
                    "gaussians": info.get("gaussians"), "sh_degree": info.get("sh_degree")}
            except (OSError, ValueError):
                pass
        return result

    def request(self, job: str, variant: str) -> dict:
        """Queue the conversion unless the preview exists or is already on its way."""
        app = self.app
        status = app.require_job(job)
        if status["state"] not in TERMINAL_STATES:
            raise ApiError(409, "not_done", "任务运行结束之后才能生成预览")
        state = self.state(job, variant, status)
        if state["state"] == "no_splat":
            raise ApiError(409, "no_splat", "找不到高斯点云文件")
        if state["state"] in ("missing", "failed"):
            self._enqueue((job, variant))
            state = self.state(job, variant, status)
        return state

    def file(self, job: str, variant: str) -> Path:
        status = self.app.require_job(job)
        source = self.source(job, status) if status["state"] in TERMINAL_STATES else None
        ready = self.ready_file(job, variant, source) if source is not None else None
        if ready is None:
            raise ApiError(404, "preview_missing", "预览文件还没有生成")
        return ready[0]

    def _build(self, job: str, variant: str) -> Optional[str]:
        """Run one conversion. Returns None on success, else the message for the user."""
        app = self.app
        status = app.read_status(job, fresh=True)
        source = self.source(job, status) if status and status["state"] in TERMINAL_STATES else None
        if source is None:
            return "找不到高斯点云文件"
        name, limit = PREVIEW_VARIANTS[variant]
        made_from = self._source_key(source)
        folder = safe_join(app.job_dir(job), [PREVIEW_CACHE_DIR])
        folder.mkdir(exist_ok=True)
        out = safe_join(app.job_dir(job), [PREVIEW_CACHE_DIR, name])
        if shutil.disk_usage(folder).free < made_from["source_size"] // 4 + DISK_RESERVE_BYTES:
            return "电脑磁盘空间不足，无法生成预览"
        returncode, output, reason, seconds = self._run_child(
            build_preview_command(source, out, limit))
        log.info("preview %s of %s: exit code %s after %.1f s%s", variant, job, returncode,
                 seconds, ", %s" % reason if reason else "")
        if reason == "stopped":
            return "已中断"
        if reason == "time_limit":
            return "生成预览超过 %d 分钟，已被停止" % (self.time_limit // 60)
        if returncode != 0 or not out.is_file():
            detail = _last_error_line(output)
            return "无法生成预览（退出码 %s）%s" % (returncode, "：" + detail[:300] if detail else "")
        try:
            if self._source_key(source) != made_from:
                return "高斯点云文件在转换过程中发生了变化，请重试"
            # What the file holds: the converter says so on its last line; failing that,
            # the file itself.
            holds: Dict[str, Any] = {}
            for line in reversed(output.decode("utf-8", "replace").strip().splitlines()):
                try:
                    said = json.loads(line)
                except ValueError:
                    continue
                if isinstance(said, dict) and isinstance(said.get("sh_degree"), int) \
                        and isinstance(said.get("gaussians_out"), int):
                    holds = {"sh_degree": said["sh_degree"], "sh_bits": said.get("sh_bits"),
                             "gaussians": said["gaussians_out"]}
                break
            if not holds:
                written = _splatconv.read_preview_header(out)
                holds = {"sh_degree": written["sh_degree"], "sh_bits": written["sh_bits"],
                         "gaussians": written["count"]}
            write_json_atomic(out.with_name(name + ".json"), dict(made_from, **holds))
        except (OSError, ValueError) as exc:
            return "无法保存预览：%s" % exc
        return None


# ---------------------------------------------------------------------------- network weights

class ModelDownloads:
    """What /api/models reports and does: the state of the weights, and one download at a time.

    There is no licence step here: every feature works the same way (switching the stage on is
    the choice, the weights come when the stage starts, the pipeline records the licence).
    This download is only the optional way to fetch weights ahead of time.

    A download is a child process (build_models_command) with a time limit that is killed when
    the server stops; requests only start it and report where it is.
    """

    def __init__(self, app: "App") -> None:
        self.app = app
        self.lock = threading.Lock()
        self.state = "idle"                 # idle | running | done | failed
        self.features: List[str] = []       # the features the download is for
        self.files: List[Tuple[str, float]] = []    # (file name, size in MB) it fetches
        self.error: Optional[str] = None
        self.child: Optional[subprocess.Popen] = None
        self.time_limit = MODELS_TIME_LIMIT_SECONDS
        self._thread: Optional[threading.Thread] = None

    # ---- what the API reports

    @staticmethod
    def whole_table() -> dict:
        """models.features_table(), checked for its shape: {"features": [...], "disk_mb"}."""
        try:
            table = models_features()
            if not isinstance(table, dict) or not isinstance(table.get("features"), list) \
                    or not all(isinstance(f, dict) and isinstance(f.get("id"), str)
                               and isinstance(f.get("files", []), list)
                               for f in table["features"]):
                raise ValueError("unexpected answer")
        except Exception as exc:  # noqa: BLE001 - a damaged settings.json, a missing module ...
            # The page asks again every few seconds and each refused request is logged anyway.
            log.debug("the state of the models could not be read: %r", exc)
            raise ApiError(500, "models_unavailable", "无法读取模型状态：%s" % exc)
        return table

    @classmethod
    def table(cls) -> List[dict]:
        return cls.whole_table()["features"]

    @staticmethod
    def missing_files(feature: dict) -> List[Tuple[str, float]]:
        return [(str(f.get("name")), float(f.get("mb") or 0)) for f in feature.get("files") or []
                if isinstance(f, dict) and not f.get("installed")]

    def download_view(self) -> dict:
        with self.lock:
            state, features, files, error = self.state, list(self.features), list(self.files), \
                self.error
        rows = []
        if files:
            try:
                folder: Optional[Path] = models_dir()
            except Exception:  # noqa: BLE001 - the numbers are a convenience
                folder = None
            for name, size in files:
                got = 0.0
                try:
                    if state == "done":
                        got = size
                    elif folder is None:
                        got = 0.0
                    elif (folder / name).is_file():      # arrived: moved out of _part
                        got = size
                    else:
                        part = (folder / MODELS_PART_DIR / name).stat().st_size / (1024.0 * 1024.0)
                        got = min(part, size) if size else part
                except OSError:     # not begun yet
                    got = 0.0
                rows.append({"name": name, "size_mb": size, "downloaded_mb": round(got, 1)})
        return {"state": state, "features": features, "error": error,
                "downloaded_mb": round(sum(row["downloaded_mb"] for row in rows), 1),
                "total_mb": sum(row["size_mb"] for row in rows), "files": rows}

    def overview(self, *, loopback: bool) -> dict:
        whole = self.whole_table()
        table = whole["features"]
        lack: Dict[str, float] = {}
        for feature in table:
            for f in feature.get("files") or []:
                if isinstance(f, dict) and not f.get("installed"):
                    lack[str(f.get("name"))] = float(f.get("mb") or 0)
        free_mb = None
        try:
            probe = models_dir()
            while not probe.exists() and probe.parent != probe:
                probe = probe.parent
            free_mb = int(shutil.disk_usage(probe).free // (1024 * 1024))
        except Exception:  # noqa: BLE001 - a number for the panel, not worth failing for
            pass
        return {"features": table, "disk_mb": whole.get("disk_mb"),
                "download_mb": round(sum(lack.values())), "free_mb": free_mb,
                "download": self.download_view(), "can_manage": bool(loopback)}

    # ---- actions (loopback clients only: the handler checks)

    def start(self, names: Sequence[str]) -> bool:
        """Start a download for the features named (none: the ones the default template
        uses). False when nothing is missing, so nothing was started."""
        app = self.app
        with self.lock:
            by_id = {f["id"]: f for f in self.table()}
            unknown = [n for n in names if n not in by_id]
            if unknown:
                raise ApiError(400, "bad_feature", "没有这个功能：%s" % "、".join(unknown)[:200])
            if app.stop_event.is_set():
                raise ApiError(503, "stopping", "扫描程序正在停止")
            if self.state == "running":
                raise ApiError(409, "download_busy", "已经有模型在下载，请等它结束")
            if names:
                wanted = list(dict.fromkeys(names))
            else:
                try:
                    wanted = [n for n in models_default_features() if n in by_id]
                except Exception as exc:  # noqa: BLE001 - said as the error of this call
                    raise ApiError(500, "models_unavailable", "无法读取模型状态：%s" % exc)
            wanted = [n for n in wanted if self.missing_files(by_id[n])]
            files: List[Tuple[str, float]] = []
            for n in wanted:
                for name, size in self.missing_files(by_id[n]):
                    if name not in [f[0] for f in files]:
                        files.append((name, size))
            if not files:
                return False
            self._check_disk(files)
            self.state, self.features, self.files, self.error = "running", wanted, files, None
            cmd = build_models_command(wanted)
            self._thread = threading.Thread(target=self._run, args=(cmd, wanted),
                                            name="scan3d-models", daemon=True)
            self._thread.start()
        return True

    @staticmethod
    def _check_disk(files: Sequence[Tuple[str, float]]) -> None:
        try:
            folder = models_dir()
            need = 0.0
            for name, size in files:
                try:
                    have = (folder / MODELS_PART_DIR / name).stat().st_size
                except OSError:
                    have = 0
                need += max(0.0, size * 1024 * 1024 - have)
            probe = folder
            while not probe.exists() and probe.parent != probe:
                probe = probe.parent
            free = shutil.disk_usage(probe).free
        except Exception:  # noqa: BLE001 - not knowing is no reason to refuse
            return
        if free < need + DISK_RESERVE_BYTES:
            raise ApiError(507, "disk_full", "电脑磁盘空间不足：还需要约 %d MB" % (need // (1024 * 1024)))

    def shutdown(self, timeout: float = 10.0) -> None:
        """Called with app.stop_event set: the download's thread kills its child and ends."""
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)
        child = self.child
        if child is not None and child.poll() is None:  # the thread did not get to it
            _proc.kill_tree(child.pid)

    # ---- the one download

    def _run(self, cmd: List[str], wanted: List[str]) -> None:
        error: Optional[str]
        try:
            error = self._download(cmd, wanted)
        except Exception as exc:  # noqa: BLE001 - whatever happens, the state must not stay "running"
            log.exception("the model download could not be run")
            error = "无法下载：%s" % exc
        with self.lock:
            self.state = "failed" if error else "done"
            self.error = error
            self.child = None

    def _download(self, cmd: List[str], wanted: List[str]) -> Optional[str]:
        """Run the download. Returns None when the features have their files, else the
        message for the user."""
        app = self.app
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        child = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=str(app.root), env=env,
            creationflags=(CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0)
        self.child = child
        assign = getattr(_proc, "_assign_to_job", None)
        if assign is not None:
            try:
                assign(child)   # dies with this process, like the pipeline
            except Exception:  # noqa: BLE001 - best effort
                pass
        log.info("model download started for %s (pid %s)", ", ".join(wanted), child.pid)
        started = time.monotonic()
        reason = None
        output = b""
        while True:
            try:
                output, _ = child.communicate(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                pass
            if app.stop_event.is_set():
                reason = "stopped"
            elif time.monotonic() - started > self.time_limit:
                reason = "time_limit"
            if reason is not None:
                _proc.kill_tree(child.pid)
                try:
                    output, _ = child.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                break
        log.info("model download: exit code %s after %.0f s%s", child.returncode,
                 time.monotonic() - started, ", %s" % reason if reason else "")
        kept = "已经下载的部分会保留，重试时接着下载。"
        if reason == "stopped":
            return "下载被中断：扫描程序被关闭了。" + kept
        if reason == "time_limit":
            return "下载超过 %d 小时，已被停止。%s" % (self.time_limit // 3600, kept)
        # The command line may exit with 1 for a reason nobody asked about here: what counts is
        # whether the files asked for have arrived.
        by_id = {f["id"]: f for f in self.table()}
        if not any(self.missing_files(by_id.get(n) or {}) for n in wanted):
            return None
        detail = ""
        for line in reversed(output.decode("utf-8", "replace").strip().splitlines()):
            detail = line.strip()
            if detail:
                break
        return "没有下载完（退出码 %s）%s。%s" % (child.returncode,
                                           "：" + detail[:300] if detail else "", kept)


# ---------------------------------------------------------------------------- jobs, queue, worker

class App:
    """Everything the request handlers share: job files, the queue and the worker thread."""

    def __init__(self, root: Optional[os.PathLike] = None, *,
                 runs_dir: Optional[os.PathLike] = None,
                 web_dir: Optional[os.PathLike] = None,
                 token: Optional[str] = None) -> None:
        self.root = Path(root).resolve() if root else ROOT_DIR
        self.runs_dir = Path(runs_dir).resolve() if runs_dir else self.root / "runs"
        self.web_dir = Path(web_dir).resolve() if web_dir else WEB_DIR
        self.viewer_exe = self.root / "tools" / "spirula" / "spirula.exe"
        # Clients on other machines must present this; `token` is for tests, the product
        # keeps one in runs/.server-token so it stays the same from start to start.
        self.token: str = token or load_or_create_token(self.runs_dir)
        self.port: Optional[int] = None

        # Replaceable so tests can simulate a remote client and avoid opening real windows.
        self.loopback_check: Callable[[str], bool] = is_loopback_address
        self.spawn_detached: Callable[[Sequence[str]], None] = spawn_detached

        self.socket_timeout = SOCKET_TIMEOUT
        self.cancel_grace = CANCEL_GRACE_SECONDS
        self.exit_grace = EXIT_GRACE_SECONDS
        self.job_time_limit = JOB_TIME_LIMIT_SECONDS

        self.lock = threading.RLock()
        self.queue: "collections.deque[str]" = collections.deque()
        self.current: Optional[str] = None
        self.current_proc: Optional[subprocess.Popen] = None
        self.cancel_requested: Optional[float] = None   # when cancel was asked for `current`
        self.uploads: Dict[str, set] = {}          # job -> names being received right now
        self.auth_failures: Dict[str, List[float]] = {}
        self._status_cache: Dict[str, Tuple[float, dict]] = {}
        self._csp_cache: Optional[Tuple[tuple, str]] = None
        self._camera_cache: Dict[str, Tuple[tuple, dict]] = {}
        self._wake = threading.Event()
        self.stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self.previews = PreviewBuilder(self)
        self.models = ModelDownloads(self)

    # ---- lifecycle

    def start(self) -> None:
        """Recover jobs a previous run left behind, then start the worker thread."""
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._remove_trash()
        self.recover_stale(startup=True)
        self._worker = threading.Thread(target=self._worker_loop, name="scan3d-worker",
                                        daemon=True)
        self._worker.start()

    def shutdown(self, timeout: float = 30.0) -> None:
        """Stop the worker and kill the running pipeline's process tree."""
        self.stop_event.set()
        self._wake.set()
        self.previews.shutdown()
        self.models.shutdown()
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout)
        with self.lock:
            child = self.current_proc
            queued = list(self.queue)
            self.queue.clear()
        if child is not None and child.poll() is None:  # the worker did not get to it
            _proc.kill_tree(child.pid)
        for job in queued:
            self._finish(job, "failed", INTERRUPTED_MESSAGE, error=INTERRUPTED_MESSAGE)

    # ---- job files

    def job_dir(self, job: str) -> Path:
        if not valid_job_name(job):
            raise ApiError(400, "bad_name", "任务名称不合法")
        return self.runs_dir / job

    def read_status(self, job: str, *, fresh: bool = False) -> Optional[dict]:
        """status.json of a job, or None when the folder is not a job.

        Results are cached for a moment: on Windows even a brief read can make the pipeline's
        atomic replace of the file fail, so the file is touched at most a few times a second
        no matter how many clients are polling.
        """
        now = time.monotonic()
        if not fresh:
            cached = self._status_cache.get(job)
            if cached and now - cached[0] < STATUS_CACHE_SECONDS:
                return dict(cached[1])
        path = self.runs_dir / job / "status.json"
        try:
            data = read_json(path)
        except ValueError:
            cached = self._status_cache.get(job)
            if cached:  # keep showing the last good status instead of flickering
                return dict(cached[1])
            data = {"state": "failed", "message": "状态文件无法读取",
                    "error": "status.json 无法读取"}
        if data is None:
            self._status_cache.pop(job, None)
            return None
        data["job"] = job
        if data.get("state") not in ALL_STATES:
            data["error"] = data.get("error") or ("未知状态: %r" % (data.get("state"),))
            data["state"] = "failed"
        self._status_cache[job] = (now, dict(data))
        return data

    def write_status(self, job: str, data: dict) -> None:
        data = dict(data)
        data["job"] = job
        data["updated_at"] = now_iso()
        write_json_atomic(self.runs_dir / job / "status.json", data)
        self._status_cache[job] = (time.monotonic(), dict(data))

    def read_meta(self, job: str) -> Optional[dict]:
        try:
            return read_json(self.runs_dir / job / "server-job.json")
        except ValueError:
            return None

    def require_job(self, job: str) -> dict:
        self.job_dir(job)
        status = self.read_status(job, fresh=True)
        if status is None:
            raise ApiError(404, "not_found", "找不到这个任务")
        return status

    def job_names(self) -> List[str]:
        names = []
        try:
            with os.scandir(self.runs_dir) as entries:
                for entry in entries:
                    if valid_job_name(entry.name) and entry.is_dir(follow_symlinks=False) \
                            and os.path.isfile(os.path.join(entry.path, "status.json")):
                        names.append(entry.name)
        except OSError:
            pass
        return names

    @staticmethod
    def _new_status(job: str, state: str, message: str) -> dict:
        return {"job": job, "state": state, "stage": None, "stage_progress": None,
                "message": message, "started_at": None, "updated_at": now_iso(),
                "finished_at": None, "error": None, "pid": None, "outputs": {}}

    def _finish(self, job: str, state: str, message: str, *, error: Optional[str] = None,
                extra: Optional[dict] = None) -> None:
        """Write a terminal state for a job that has no live pipeline process of ours."""
        try:
            status = self.read_status(job, fresh=True)
            if status is None:
                return
            status.update({"state": state, "message": message, "error": error,
                           "finished_at": now_iso(), "pid": None})
            self._settle_stages(status, state, error)
            if extra:
                status.update(extra)
            self.write_status(job, status)
        except OSError:
            log.exception("could not write the final status of %s", job)

    # ---- views

    @staticmethod
    def _settle_stages(status: dict, state: str, error: Optional[str]) -> None:
        """A run that ended without the pipeline's word (it died, was stopped, was cancelled
        while queued) leaves stages marked queued or running: a stage that was running
        failed, one that was only waiting is pending again."""
        stages = status.get("stages")
        if not isinstance(stages, dict):
            return
        for entry in stages.values():
            if not isinstance(entry, dict):
                continue
            if entry.get("state") == "running":
                entry["state"] = "failed" if state == "failed" else "pending"
                if state == "failed":
                    entry["error"] = entry.get("error") or error
            elif entry.get("state") == "queued":
                entry["state"] = "pending"
            if entry.get("state") != "running":
                entry["progress"] = None

    def list_uploads(self, job: str) -> Tuple[int, int]:
        """(count, total bytes) of the complete uploads of a job."""
        count = total = 0
        try:
            with os.scandir(self.runs_dir / job / "upload") as entries:
                for entry in entries:
                    if entry.is_file(follow_symlinks=False) \
                            and file_extension(entry.name) in ALLOWED_EXTENSIONS:
                        count += 1
                        total += entry.stat(follow_symlinks=False).st_size
        except OSError:
            pass
        return count, total

    def output_files(self, job: str, status: dict) -> List[dict]:
        """Existing output files: the ones status.json names, then anything else in output/."""
        job_dir = self.runs_dir / job
        files: List[dict] = []
        seen = set()
        outputs = status.get("outputs")
        for key, rel in (outputs.items() if isinstance(outputs, dict) else ()):
            if not isinstance(rel, str):
                continue
            parts = rel.replace("\\", "/").split("/")
            try:
                path = safe_join(job_dir, parts)
                if not path.is_file():
                    continue
                size = path.stat().st_size
            except (ApiError, OSError):
                continue
            seen.add("/".join(parts).lower())
            files.append({"key": str(key), "path": "/".join(parts), "name": parts[-1],
                          "size": size})
        try:
            with os.scandir(job_dir / "output") as entries:
                extra = sorted((e for e in entries if e.is_file(follow_symlinks=False)
                                and valid_component(e.name)), key=lambda e: e.name.lower())
                for entry in extra[:200]:
                    rel = "output/" + entry.name
                    if rel.lower() not in seen:
                        files.append({"key": None, "path": rel, "name": entry.name,
                                      "size": entry.stat(follow_symlinks=False).st_size})
        except OSError:
            pass
        return files

    def log_names(self, job: str) -> List[str]:
        """pipeline.log, then one log per stage that wrote one (in stage order), any other
        log of the job, and the launcher's own."""
        job_dir = self.runs_dir / job
        names: List[str] = []
        try:
            with os.scandir(job_dir / "logs") as entries:
                names = sorted(e.name for e in entries
                               if e.is_file(follow_symlinks=False) and valid_component(e.name)
                               and e.name.lower().endswith(".log"))
        except OSError:
            pass

        def order(name: str) -> Tuple[int, int, str]:
            stem = name[:-4]
            if name == "pipeline.log":
                return 0, 0, name
            if stem in STAGES:
                return 1, STAGES.index(stem), name
            return 2, 0, name

        names.sort(key=order)
        if (job_dir / "launcher.log").is_file():
            names.append("launcher.log")
        return names

    def workflow_state(self, job: str) -> Tuple[Optional[dict], Optional[str]]:
        """(pipeline.job_view of the job, None), or (None, why not): a damaged workflow.json
        must not make the job vanish from the list."""
        try:
            return pipeline_job_view(self.runs_dir / job), None
        except _wf.WorkflowError as exc:
            return None, "；".join(exc.problems)
        except Exception as exc:  # noqa: BLE001 - shown in the job's own view
            log.debug("workflow of %s could not be read", job, exc_info=True)
            return None, "无法读取流程：%s" % exc

    def job_view(self, job: str, status: Optional[dict] = None, *, detail: bool = False,
                 loopback: bool = False,
                 queue_now: Optional[Tuple[List[str], Optional[str]]] = None) -> Optional[dict]:
        status = status if status is not None else self.read_status(job)
        if status is None:
            return None
        meta = self.read_meta(job)
        view = self._new_status(job, status["state"], "")
        view["updated_at"] = None
        view.update(status)
        for key in STATUS_DROPPED:      # fields of jobs made before the workflow redesign
            view.pop(key, None)
        if queue_now is None:
            with self.lock:
                queue_now = (list(self.queue), self.current)
        queue, current = queue_now
        state = view["state"]
        view["queue_position"] = queue.index(job) + 1 if job in queue else None
        view["source"] = (meta or {}).get("source", "external")
        view["created_at"] = (meta or {}).get("created_at")

        info, problem = self.workflow_state(job)
        run = dict(info.get("run") or {}) if info else {}
        stages = None
        if info:
            stages = {sid: dict(entry) if isinstance(entry, dict) else entry
                      for sid, entry in (info.get("stages") or {}).items()}
        elif isinstance(status.get("stages"), dict):
            stages = status["stages"]
        if state == "queued" and isinstance(status.get("run"), dict):
            # Queued by the server, not started yet: the plan is the run it asked for.
            run = dict(status["run"])
            for sid in run.get("plan") or []:
                if stages is not None and isinstance(stages.get(sid), dict):
                    stages[sid].update(state="queued", progress=None)
        view["run"] = {"to": run.get("to"), "only": list(run.get("only") or []),
                       "force": list(run.get("force") or []), "plan": list(run.get("plan") or [])}
        view["template"] = info.get("template") if info else None
        if problem:
            view["workflow_error"] = problem
        if detail:
            view["stages"] = stages
            view["workflow"] = info.get("workflow") if info else None
        else:
            view["stages"] = stage_overview(stages)
            view.pop("workflow", None)

        started = parse_iso(view.get("started_at"))
        elapsed = None
        if started is not None and state != "queued":
            end = parse_iso(view.get("finished_at")) if state in TERMINAL_STATES else None
            if state == "running":
                end = _dt.datetime.now().astimezone()
            if end is not None:
                elapsed = max(0.0, (end - started).total_seconds())
        view["elapsed_seconds"] = elapsed

        actions: List[str] = []
        idle = state not in ACTIVE_STATES and job != current
        if state in ACTIVE_STATES:
            actions.append("cancel")
        if idle and meta:
            actions += ["run", "start"]     # `start` is the old name of `run`
        if idle:
            actions.append("edit")          # change the workflow, clear a stage
        if state != "running" and job != current:
            actions.append("delete")
        if loopback:
            actions.append("open_folder")
            view["input_path"] = (meta or {}).get("input_path")
        view["actions"] = actions
        view["cancelling"] = state in ACTIVE_STATES and os.path.exists(
            self.runs_dir / job / "cancel")

        if detail:
            count, total = self.list_uploads(job)
            view["uploads"] = {"count": count, "bytes": total}
            view["output_files"] = self.output_files(job, status)
            if state in TERMINAL_STATES:
                try:
                    outdated = self.previews.outdated_outputs(job, status)
                except (ApiError, OSError, ValueError):
                    outdated = []
                for entry in view["output_files"]:
                    if entry["path"] in outdated:
                        entry["outdated"] = True
            view["logs"] = self.log_names(job)
            if loopback and state in TERMINAL_STATES and self.viewer_exe.is_file() and any(
                    f["key"] == "splat" for f in view["output_files"]):
                actions.append("open_viewer")
            view["licences"], view["models_needed"] = [], []
            if info:
                try:
                    view["changes"] = changes_from_template(info["workflow"],
                                                            load_user_settings())
                    needs = workflow_needs(info["workflow"])
                    view["licences"] = list(needs.get("licences") or [])
                    view["models_needed"] = list(needs.get("models_needed") or [])
                except Exception as exc:  # noqa: BLE001 - the rest of the view is still right
                    log.debug("needs of %s could not be worked out", job, exc_info=True)
                    view["needs_error"] = "无法确定需要的模型：%s" % exc
        return view

    def list_jobs(self, *, loopback: bool = False) -> dict:
        # The queue is read before the jobs, so that a job is never shown as running while
        # the list says another one is current.
        with self.lock:
            queue_now = (list(self.queue), self.current)
        views = []
        for name in self.job_names():
            view = self.job_view(name, loopback=loopback, queue_now=queue_now)
            if view is not None:
                views.append(view)

        def sort_key(view: dict) -> float:
            for field in ("created_at", "started_at", "updated_at"):
                parsed = parse_iso(view.get(field))
                if parsed is not None:
                    return parsed.timestamp()
            return 0.0

        views.sort(key=sort_key, reverse=True)
        return {"jobs": views, "queue": queue_now[0], "current": queue_now[1]}

    # ---- actions

    def resolve_workflow(self, template: Optional[str], override: Optional[dict]) -> dict:
        """The complete workflow of a template with an override on top (ApiError 400
        bad_workflow when it does not validate)."""
        try:
            return _wf.resolve(template, override, user=load_user_settings())
        except _wf.WorkflowError as exc:
            raise bad_workflow(exc)

    def create_job(self, name: Optional[str], workflow: dict, *, source: str = "upload",
                   input_path: Optional[str] = None) -> str:
        if name is not None and not valid_job_name(name):
            raise ApiError(400, "bad_name",
                           "名称只能包含英文字母、数字、- 和 _，以字母或数字开头，最长 64 个字符")
        with self.lock:
            self.runs_dir.mkdir(parents=True, exist_ok=True)
            if name is not None:
                candidates = [name]
            else:
                base = time.strftime("scan-%Y%m%d-%H%M%S")
                candidates = [base] + ["%s-%d" % (base, i) for i in range(2, 100)]
            job = None
            for candidate in candidates:
                try:
                    os.mkdir(self.runs_dir / candidate)
                    job = candidate
                    break
                except FileExistsError:
                    continue
            if job is None:
                raise ApiError(409, "exists", "这个名称已经被使用，请换一个")
            job_dir = self.runs_dir / job
            try:
                if source == "upload":
                    (job_dir / "upload").mkdir()
                    input_path = str(job_dir / "upload")
                pipeline_write_workflow(job_dir, workflow)
                write_json_atomic(job_dir / "server-job.json", {
                    "job": job, "source": source, "input_path": input_path, "run": None,
                    "created_at": now_iso()})
                self.write_status(job, self._new_status(job, STATE_CREATED, "等待开始"))
            except Exception:
                remove_tree(job_dir)
                raise
        return job

    def check_local_path(self, raw: Any) -> str:
        """Validate a path typed by the user of this PC: a folder with media, or a video."""
        if not isinstance(raw, str):
            raise ApiError(400, "bad_path", "请填写文件夹或视频文件的完整路径")
        text = raw.strip().strip('"').strip()
        if not text or "\x00" in text or not os.path.isabs(text):
            raise ApiError(400, "bad_path", "请填写完整路径，例如 D:\\photos\\room")

        def probe() -> str:
            if os.path.isdir(text):
                with os.scandir(text) as entries:
                    for index, entry in enumerate(entries):
                        if file_extension(entry.name) in ALLOWED_EXTENSIONS and entry.is_file():
                            return "ok"
                        if index > 20000:
                            break
                return "empty"
            if os.path.isfile(text):
                return "ok" if file_extension(text) in VIDEO_EXTENSIONS else "not_video"
            return "missing"

        try:
            result = call_with_timeout(probe, PATH_CHECK_TIMEOUT)
        except TimeoutError:
            raise ApiError(400, "bad_path", "路径没有响应（网络位置不可用？）")
        except OSError as exc:
            raise ApiError(400, "bad_path", "无法读取这个路径：%s" % exc)
        if result == "missing":
            raise ApiError(400, "bad_path", "这个路径不存在")
        if result == "empty":
            raise ApiError(400, "bad_path", "这个文件夹里没有图片或视频文件")
        if result == "not_video":
            raise ApiError(400, "bad_path", "请选择一个文件夹，或一个视频文件")
        return os.path.abspath(text)

    # ---- running: plan, queue

    def plan_run(self, job: str, to: Optional[str], only: Sequence[str],
                 force: Sequence[str], *, with_needs: bool = True) -> dict:
        """What a run with this request would do, without doing it: {"steps": [{"stage",
        "reason", "label"}], "licences": [...], "models_needed": [...]} for the stages of the
        plan (`with_needs` False leaves the last two out: queueing does not need them).
        ApiError 400 with the PlanError's code when it cannot be carried out."""
        self.require_job(job)
        info, problem = self.workflow_state(job)
        if info is None:
            raise ApiError(409, "bad_workflow", "这个任务的流程设置无法读取：%s" % problem)
        states = {sid: (entry or {}).get("state") for sid, entry in info["stages"].items()}
        try:
            steps = _wf.plan(info["workflow"], states, to=to, only=only or None, force=force)
        except _wf.PlanError as exc:
            raise ApiError(400, exc.code, exc.message)
        ids = [step["stage"] for step in steps]
        result = {"steps": [dict(step, label=_wf.stage_label(step["stage"])) for step in steps]}
        if not with_needs:
            return result
        try:
            needs = workflow_needs(info["workflow"])
        except Exception as exc:  # noqa: BLE001 - said in the plan's own words
            log.debug("needs of %s could not be worked out", job, exc_info=True)
            raise ApiError(500, "models_unavailable", "无法确定需要的模型：%s" % exc)

        def of_plan(items: Any) -> list:
            return [i for i in (items or []) if not isinstance(i, dict) or i.get("stage") is None
                    or i.get("stage") in ids]

        result["licences"] = of_plan(needs.get("licences"))
        result["models_needed"] = of_plan(needs.get("models_needed"))
        return result

    def start_job(self, job: str, to: Optional[str] = None, only: Sequence[str] = (),
                  force: Sequence[str] = ()) -> dict:
        """Queue a run of the job. Returns its plan (see plan_run). The job's stored
        workflow decides what runs; `to`, `only` and `force` narrow or widen this run."""
        self.require_job(job)
        meta = self.read_meta(job)
        if meta is None:
            raise ApiError(409, "not_startable", "这个任务不是在这里创建的，无法从这里启动")
        if meta.get("source") == "path":  # may block on a dead share: check outside the lock
            self.check_local_path(meta.get("input_path"))
        with self.lock:
            status = self.require_job(job)
            if job == self.current or job in self.queue or status["state"] in ACTIVE_STATES:
                raise ApiError(409, "already_active", "任务已经在排队或运行")
            if self.uploads.get(job):
                raise ApiError(409, "uploading", "还有文件正在上传，请稍候")
            job_dir = self.runs_dir / job
            if meta.get("source") != "path":
                meta["input_path"] = str(job_dir / "upload")
                if self.list_uploads(job)[0] == 0:
                    raise ApiError(400, "no_input", "还没有上传任何照片或视频")
            plan = self.plan_run(job, to, only, force, with_needs=False)
            run = {"to": to, "only": list(only), "force": list(force)}
            meta["run"] = run
            write_json_atomic(job_dir / "server-job.json", meta)
            try:
                os.unlink(job_dir / "cancel")
            except FileNotFoundError:
                pass
            remove_tree(job_dir / "upload.part")
            # What the last run left (outputs, so the results stay viewable) is kept; what
            # described that run is replaced.
            queued = dict(status)
            for key in ("stages", "warnings", "exit_code"):
                queued.pop(key, None)
            queued.update(self._new_status(job, "queued", "排队中"))
            queued["outputs"] = status.get("outputs") or {}
            queued["run"] = dict(run, plan=[step["stage"] for step in plan["steps"]])
            self.write_status(job, queued)
            self.queue.append(job)
        self._wake.set()
        return plan

    def set_workflow(self, job: str, template: Optional[str], override: Optional[dict]) -> None:
        """Change the stored workflow: with a template, start again from that template's
        settings plus the override; without, merge the override into what the job has."""
        self.require_job(job)
        with self.lock:
            self._require_idle(job)
            info, problem = self.workflow_state(job)
            if template is None:
                if info is None:
                    raise ApiError(409, "bad_workflow", "这个任务的流程设置无法读取：%s" % problem)
                try:
                    new = _wf.apply_overrides(info["workflow"], override or {})
                except _wf.WorkflowError as exc:
                    raise bad_workflow(exc)
            else:
                new = self.resolve_workflow(template, override)
            pipeline_write_workflow(self.runs_dir / job, new)

    def _require_idle(self, job: str) -> None:
        status = self.require_job(job)
        if job == self.current or job in self.queue or status["state"] in ACTIVE_STATES:
            raise ApiError(409, "job_busy", "任务正在排队或运行，请先等它结束或取消")

    def reset_stage(self, job: str, stage: str) -> None:
        """Delete the result of one stage (its record and what only it produced)."""
        self.require_job(job)
        if stage not in STAGES:
            raise ApiError(400, "unknown_stage", "没有这个阶段：%s" % stage)
        self._require_idle(job)
        self.previews.cancel(job)   # a conversion may be reading the splat
        with self.lock:
            self._require_idle(job)
            try:
                pipeline_reset_stage(self.runs_dir / job, stage)
            except ValueError as exc:
                raise ApiError(400, "cannot_reset", str(exc))
            except OSError as exc:
                raise ApiError(409, "in_use", "无法清除：文件正被其他程序使用（%s）" % exc)
            self._status_cache.pop(job, None)
            self._camera_cache.pop(job, None)

    # ---- the coverage map: a stage like the others, its result is output/coverage.json

    def coverage_file(self, job: str) -> Optional[Path]:
        try:
            path = safe_join(self.job_dir(job), list(COVERAGE_FILE))
        except ApiError:
            return None
        return path if path.is_file() else None

    def coverage_state(self, job: str) -> dict:
        """{"state": ready (plus size, and `stale` when the photos or settings changed since)
        | missing | queued | building | failed (plus error) | unavailable (plus reason and
        message)}. Nothing is computed by asking."""
        status = self.require_job(job)
        info, _problem = self.workflow_state(job)
        entry = ((info or {}).get("stages") or {}).get("coverage")
        entry = entry if isinstance(entry, dict) else {}
        stage_state = entry.get("state")
        if status["state"] == "queued" and "coverage" in (status.get("run") or {}).get("plan", []):
            stage_state = "queued"
        if stage_state == "queued":
            return {"state": "queued"}
        if stage_state == "running":
            return {"state": "building"}
        if stage_state == "failed":
            return {"state": "failed", "error": entry.get("error") or "拍摄覆盖没有算出来"}
        path = self.coverage_file(job)
        if path is not None:
            try:
                result: Dict[str, Any] = {"state": "ready", "size": path.stat().st_size}
            except OSError:
                result = {}
            if result:
                if stage_state == "stale":
                    result["stale"] = True
                return result
        if self.previews.source(job, status) is None:
            return {"state": "unavailable", "reason": "no_splat",
                    "message": "还没有高斯点云，无法计算拍摄覆盖"}
        return {"state": "missing", "enabled": stage_state != "disabled"}

    def request_coverage(self, job: str) -> dict:
        """Queue `--only coverage` unless the map exists (and is current) or is on its way."""
        state = self.coverage_state(job)
        if state["state"] == "unavailable":
            raise ApiError(409, state["reason"], state["message"])
        if state["state"] in ("queued", "building") or (
                state["state"] == "ready" and not state.get("stale")):
            return state
        self.start_job(job, only=["coverage"])
        return self.coverage_state(job)

    def cancel_job(self, job: str) -> None:
        with self.lock:
            status = self.require_job(job)
            if job in self.queue:
                self.queue.remove(job)
                self._finish(job, "cancelled", "已取消")
                return
            if job == self.current or status["state"] in ACTIVE_STATES:
                # The pipeline owns status.json now; ask it to stop through the cancel file.
                if job == self.current and self.cancel_requested is None:
                    self.cancel_requested = time.monotonic()
                self._touch_cancel(job)
                return
            raise ApiError(409, "not_active", "任务没有在排队或运行")

    def _touch_cancel(self, job: str) -> None:
        with open(self.runs_dir / job / "cancel", "a"):
            pass

    def delete_job(self, job: str) -> None:
        self.require_job(job)
        self.previews.cancel(job)
        with self.lock:
            status = self.require_job(job)
            if job == self.current or status["state"] == "running":
                raise ApiError(409, "running", "任务正在运行，请先取消")
            if self.uploads.get(job):
                raise ApiError(409, "uploading", "还有文件正在上传，请稍候")
            if job in self.queue:
                self.queue.remove(job)
            job_dir = self.runs_dir / job
            trash = self.runs_dir / (".trash-%s-%s" % (job, secrets.token_hex(4)))
            try:
                # One atomic step: a folder with a file open elsewhere cannot be renamed, so
                # the job is either gone from the list or completely untouched.
                os.rename(job_dir, trash)
            except OSError:
                raise ApiError(409, "in_use", "任务文件夹正被其他程序使用，请关闭后重试")
            self._status_cache.pop(job, None)
            self._camera_cache.pop(job, None)
        if not remove_tree(trash):
            log.warning("could not fully remove %s; it will be removed at the next start", trash)

    def _remove_trash(self) -> None:
        try:
            with os.scandir(self.runs_dir) as entries:
                leftovers = [e.path for e in entries if e.is_dir(follow_symlinks=False)
                             and re.match(r"\.trash-[A-Za-z0-9_-]+-[0-9a-f]{8}\Z", e.name)]
        except OSError:
            return
        for path in leftovers:
            remove_tree(Path(path))

    def recover_stale(self, *, startup: bool) -> List[str]:
        """Mark jobs failed whose status claims queued/running although nothing runs them.

        At startup that is every such job: this server instance has started none of them.
        Later it is only jobs this instance does not own, whose process is gone and whose
        status has not changed for a while (a pipeline that was started by hand and died).
        """
        marked = []
        for job in self.job_names():
            status = self.read_status(job, fresh=startup)
            if status is None or status["state"] not in ACTIVE_STATES:
                continue
            with self.lock:
                if job == self.current or job in self.queue:
                    continue
                status = self.read_status(job, fresh=True)
                if status is None or status["state"] not in ACTIVE_STATES:
                    continue
                if not startup:
                    try:
                        age = time.time() - os.path.getmtime(self.runs_dir / job / "status.json")
                    except OSError:
                        continue
                    if age < 15.0 or pid_alive(status.get("pid")):
                        continue
                self._finish(job, "failed", INTERRUPTED_MESSAGE, error=INTERRUPTED_MESSAGE)
                marked.append(job)
        return marked

    # ---- the worker: one job at a time

    def _worker_loop(self) -> None:
        last_stale_check = time.monotonic()
        while not self.stop_event.is_set():
            job = None
            with self.lock:
                if self.queue:
                    job = self.queue.popleft()
                    self.current = job
            if job is None:
                self._wake.wait(0.5)
                self._wake.clear()
                if time.monotonic() - last_stale_check > STALE_CHECK_SECONDS:
                    last_stale_check = time.monotonic()
                    try:
                        self.recover_stale(startup=False)
                    except Exception:  # noqa: BLE001 - housekeeping must not stop the worker
                        log.exception("stale job check failed")
                continue
            try:
                self._run_job(job)
            except Exception as exc:  # noqa: BLE001 - one bad job must not stop the queue
                log.exception("job %s could not be run", job)
                self._finish(job, "failed", "无法启动流水线", error="无法启动流水线：%s" % exc)
            finally:
                with self.lock:
                    self.current = None
                    self.current_proc = None
                    self.cancel_requested = None

    def _launcher_tail(self, job: str, lines: int = 8) -> str:
        """The last lines the pipeline process printed in its latest run (not our own
        "$ command" header): usually the traceback that explains an unexpected exit."""
        try:
            with open(self.runs_dir / job / "launcher.log", "rb") as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - 8192))
                text = f.read().decode("utf-8", "replace")
        except OSError:
            return ""
        latest: List[str] = []
        for line in text.splitlines():
            if line.startswith("$ "):
                latest = []
            elif line.strip():
                latest.append(line)
        return "\n".join(latest[-lines:])

    def _run_job(self, job: str) -> None:
        job_dir = self.runs_dir / job
        cancel_file = job_dir / "cancel"
        meta = self.read_meta(job) or {}
        with self.lock:
            if self.stop_event.is_set():
                self._finish(job, "failed", INTERRUPTED_MESSAGE, error=INTERRUPTED_MESSAGE)
                return
            if cancel_file.exists():
                self._finish(job, "cancelled", "已取消")
                return
            run = meta.get("run") if isinstance(meta.get("run"), dict) else {}
            cmd = build_run_command(job, str(meta.get("input_path") or job_dir / "upload"),
                                    to=run.get("to") if isinstance(run.get("to"), str) else None,
                                    only=_stage_ids(run.get("only"), "only"),
                                    force=_stage_ids(run.get("force"), "force"))
            env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
            with open(job_dir / "launcher.log", "ab") as launcher_log:
                launcher_log.write(("\n$ %s\n" % subprocess.list2cmdline(cmd)).encode("utf-8"))
                launcher_log.flush()
                child = subprocess.Popen(
                    cmd, stdin=subprocess.DEVNULL, stdout=launcher_log,
                    stderr=subprocess.STDOUT, cwd=str(self.root), env=env,
                    creationflags=(CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP)
                    if os.name == "nt" else 0)
            self.current_proc = child
        # If this process dies without cleaning up (console window closed), the pipeline must
        # not keep the GPU busy unsupervised: put it in proc's kill-on-close job when possible.
        assign = getattr(_proc, "_assign_to_job", None)
        if assign is not None:
            try:
                assign(child)
            except Exception:  # noqa: BLE001 - best effort
                pass
        log.info("job %s started (pid %s)", job, child.pid)

        started = time.monotonic()
        cancel_seen: Optional[float] = None
        terminal_seen: Optional[float] = None
        last_status_check = started
        reason = None
        while True:
            try:
                child.wait(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                pass
            now = time.monotonic()
            if self.stop_event.is_set():
                reason = "shutdown"
            elif now - started > self.job_time_limit:
                reason = "time_limit"
            elif self.cancel_requested is not None or cancel_file.exists():
                cancel_seen = cancel_seen if cancel_seen is not None else now
                if now - cancel_seen > self.cancel_grace:
                    reason = "cancel_forced"
                elif self.cancel_requested is not None and not cancel_file.exists():
                    # The pipeline deletes a leftover cancel file when it starts. A request
                    # made in that first moment would be lost, so put the file back.
                    try:
                        self._touch_cancel(job)
                    except OSError:
                        pass
            if reason is None and now - last_status_check > 2.0:
                # A pipeline that reported its result but never exits would block the queue.
                last_status_check = now
                status = self.read_status(job)
                if status is not None and status["state"] in TERMINAL_STATES:
                    terminal_seen = terminal_seen if terminal_seen is not None else now
                    if now - terminal_seen > self.exit_grace:
                        reason = "lingering"
            if reason is not None:
                _proc.kill_tree(child.pid)
                break
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        code = child.returncode
        log.info("job %s ended (exit code %s%s)", job, code,
                 ", %s" % reason if reason else "")

        try:  # nobody writes the file any more; wait out a scanner that has it open
            final = read_json(job_dir / "status.json", attempts=40)
        except ValueError:
            final = None
        self._status_cache.pop(job, None)
        if final is not None and final.get("state") in TERMINAL_STATES:
            return  # the pipeline reported its own result
        extra = {"exit_code": code}
        if reason == "shutdown":
            self._finish(job, "failed", INTERRUPTED_MESSAGE, error=INTERRUPTED_MESSAGE,
                         extra=extra)
        elif (reason == "cancel_forced" or self.cancel_requested is not None
              or cancel_file.exists()):
            self._finish(job, "cancelled", "已取消", extra=extra)
        elif reason == "time_limit":
            self._finish(job, "failed", "运行超时",
                         error="运行超过 %d 小时，已被停止" % (self.job_time_limit // 3600),
                         extra=extra)
        else:
            error = "流水线进程意外退出（退出码 %s），没有给出结果" % code
            tail = self._launcher_tail(job)
            if tail:
                error += "\n" + tail
            self._finish(job, "failed", "流水线意外退出", error=error, extra=extra)

    # ---- what the 3D viewer needs

    def page_csp(self, page: Path) -> str:
        """The Content-Security-Policy for the UI page, with the hash of its import map."""
        try:
            info = page.stat()
            key = (str(page), info.st_mtime_ns, info.st_size)
            cached = self._csp_cache
            if cached is not None and cached[0] == key:
                return cached[1]
            with open(page, "rb") as f:
                match = _IMPORTMAP_RE.search(f.read(1 << 20))
        except OSError:
            return PAGE_CSP % ""
        allowed = ""
        if match is not None:
            # Browsers hash the script text after turning CRLF into LF.
            text = match.group(1).replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            allowed = " 'sha256-%s'" % base64.b64encode(hashlib.sha256(text).digest()).decode("ascii")
        policy = PAGE_CSP % allowed
        self._csp_cache = (key, policy)
        return policy

    def viewer_cameras(self, job: str) -> dict:
        """Training camera poses in the coordinates of splat.ply, for the viewer's start view.

        They come from the dataset the pipeline trained on (COLMAP's images.bin), moved into
        the training frame with scene_transform.json. A job without these files simply has
        none: the viewer then frames the scene by its bounding box.
        """
        job_dir = self.job_dir(job)
        empty: Dict[str, Any] = {"count": 0, "cameras": []}

        def first_file(candidates: Sequence[Sequence[str]]) -> Optional[Path]:
            for parts in candidates:
                try:
                    path = safe_join(job_dir, parts)
                except ApiError:
                    continue
                if path.is_file():
                    return path
            return None

        images = first_file((["dataset", "sparse", "images.bin"],
                             ["dataset", "sparse", "0", "images.bin"]))
        if images is None:
            return empty
        transform_file = first_file((["output", "scene_transform.json"],
                                     ["train", "run", "scene_transform.json"]))
        try:
            key = tuple((p.stat().st_mtime_ns, p.stat().st_size) if p else None
                        for p in (images, transform_file))
            cached = self._camera_cache.get(job)
            if cached is not None and cached[0] == key:
                return cached[1]
            count, cameras = read_colmap_cameras(images, MAX_VIEWER_CAMERAS)
            transform = read_train_transform(transform_file) if transform_file else None
        except (OSError, ValueError, struct.error):
            return empty
        if transform is not None:
            scale, matrix, translation = transform

            def turn(v: Sequence[float]) -> List[float]:
                return [sum(matrix[r][c] * v[c] for c in range(3)) for r in range(3)]

            for camera in cameras:
                camera["position"] = [scale * a + b for a, b in
                                      zip(turn(camera["position"]), translation)]
                camera["forward"] = turn(camera["forward"])
                camera["up"] = turn(camera["up"])
        for camera in cameras:
            for field in ("position", "forward", "up"):
                camera[field] = [round(v, 6) for v in camera[field]]
        result = {"count": count, "cameras": cameras}
        if len(self._camera_cache) > 200:
            self._camera_cache.clear()
        self._camera_cache[job] = (key, result)
        return result

    # ---- logs

    def read_log(self, job: str, name: str, offset: Optional[int], max_bytes: int) -> dict:
        job_dir = self.job_dir(job)
        if name == "launcher.log":
            path = job_dir / "launcher.log"
        else:
            if not valid_component(name) or not name.lower().endswith(".log"):
                raise ApiError(400, "bad_path", "日志名称不合法")
            path = safe_join(job_dir, ["logs", name])
        max_bytes = max(1, min(max_bytes, MAX_LOG_BYTES))
        result = {"file": name, "exists": False, "size": 0, "offset": 0, "next_offset": 0,
                  "truncated": False, "text": ""}
        try:
            with open(path, "rb") as f:
                size = os.fstat(f.fileno()).st_size
                tail = offset is None or offset > size
                start = max(0, size - max_bytes) if tail else int(offset or 0)
                f.seek(start)
                data = f.read(max_bytes)
        except FileNotFoundError:
            return result
        except OSError as exc:
            raise ApiError(500, "io_error", "无法读取日志：%s" % exc)
        if tail and start > 0:  # begin at a line start, not in the middle of one
            cut = data.find(b"\n")
            if cut >= 0:
                start += cut + 1
                data = data[cut + 1:]
        data = _trim_partial_utf8(data)
        result.update({"exists": True, "size": size, "offset": start,
                       "next_offset": start + len(data), "truncated": tail and start > 0,
                       "text": data.decode("utf-8", "replace")})
        return result


# ---------------------------------------------------------------------------- HTTP

class Handler(BaseHTTPRequestHandler):
    server_version = "scan3d"
    sys_version = ""
    protocol_version = "HTTP/1.1"

    app: App
    _body_left = 0
    _set_cookie = False
    _loopback = False

    def setup(self) -> None:
        self.app = self.server.app  # type: ignore[attr-defined]
        self.timeout = self.app.socket_timeout  # a silent client cannot pin this thread
        super().setup()

    # ---- logging: quiet for polling, visible for actions and failures

    def log_request(self, code: Any = "-", size: Any = "-") -> None:
        try:
            status = int(code)
        except (TypeError, ValueError):
            status = 0
        if status >= 400 or self.command not in ("GET", "HEAD"):
            log.info("%s %s %s -> %s", self.client_address[0], self.command,
                     self.path.split("?", 1)[0], code)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - base signature
        log.debug("%s %s", self.client_address[0], format % args)

    # ---- responses

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        if self._set_cookie:
            # A year: with the token kept across restarts, a paired phone stays paired.
            self.send_header("Set-Cookie", "%s=%s; Path=/; Max-Age=%d; HttpOnly; SameSite=Lax"
                             % (TOKEN_COOKIE, self.app.token, TOKEN_COOKIE_MAX_AGE))
        if self._body_left > SMALL_BODY:  # the rest of the request will not be read
            self.send_header("Connection", "close")
        super().end_headers()

    def _send_bytes(self, status: int, body: bytes, content_type: str,
                    headers: Optional[Dict[str, str]] = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, payload: Any, status: int = 200,
                   headers: Optional[Dict[str, str]] = None) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(status, body, "application/json; charset=utf-8", headers)

    def _send_api_error(self, err: ApiError) -> None:
        payload = {"error": err.code, "message": err.message}
        headers: Dict[str, str] = {}
        extra = dict(err.extra)
        for key in ("Allow", "Retry-After"):
            if key in extra:
                headers[key] = str(extra.pop(key))
        payload.update(extra)
        self._send_json(payload, err.status, headers)

    def _send_file(self, path: Path, content_type: str, *, download_name: Optional[str] = None,
                   headers: Optional[Dict[str, str]] = None, ranges: bool = False) -> None:
        try:
            f = open(path, "rb")
        except OSError:
            raise ApiError(404, "not_found", "文件不存在")
        with f:
            info = os.fstat(f.fileno())
            size = info.st_size
            etag = '"%x-%x"' % (info.st_mtime_ns, size)
            first, last, status = 0, size - 1, 200
            requested = self.headers.get("Range") if ranges else None
            known = self.headers.get("If-None-Match")
            if known and not requested and etag in [t.strip() for t in known.split(",")]:
                # The browser already has this exact file (a 30 MB mesh, the viewer's
                # libraries): say so instead of sending it again.
                self.send_response(304)
                self.send_header("ETag", etag)
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                return
            if ranges:
                if_range = self.headers.get("If-Range")
                if requested and if_range and if_range.strip() != etag:
                    requested = None  # the file changed since the client's first part
                try:
                    span = parse_range(requested, size)
                except RangeNotSatisfiable:
                    self.send_response(416)
                    self.send_header("Content-Range", "bytes */%d" % size)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if span is not None:
                    first, last = span
                    status = 206
            length = last - first + 1 if size else 0
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("ETag", etag)
            if ranges:
                self.send_header("Accept-Ranges", "bytes")
                if status == 206:
                    self.send_header("Content-Range", "bytes %d-%d/%d" % (first, last, size))
            if download_name:
                ascii_name = download_name.encode("ascii", "replace").decode("ascii")
                ascii_name = re.sub(r'["\\]', "_", ascii_name)
                self.send_header("Content-Disposition", "attachment; filename=\"%s\"; "
                                 "filename*=UTF-8''%s"
                                 % (ascii_name, urllib.parse.quote(download_name, safe="")))
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            if self.command == "HEAD" or length == 0:
                return
            f.seek(first)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(DOWNLOAD_CHUNK, remaining))
                if not chunk:  # the file shrank underneath us: the response cannot be completed
                    self.close_connection = True
                    return
                self.wfile.write(chunk)  # each write is bounded by the socket timeout
                remaining -= len(chunk)

    # ---- request bodies

    def _read_exact(self, count: int) -> bytes:
        parts = []
        while count > 0:
            chunk = self.rfile.read1(min(count, 65536))
            if not chunk:
                raise ConnectionError("client closed the connection mid-request")
            parts.append(chunk)
            count -= len(chunk)
            self._body_left -= len(chunk)
        return b"".join(parts)

    def _read_json_body(self) -> dict:
        if self._body_left == 0:
            return {}
        if self._body_left > MAX_JSON_BODY:
            raise ApiError(413, "too_large", "请求内容过大")
        try:
            data = json.loads(self._read_exact(self._body_left).decode("utf-8"))
        except ValueError:
            raise ApiError(400, "bad_json", "请求内容不是合法的 JSON")
        if not isinstance(data, dict):
            raise ApiError(400, "bad_json", "请求内容必须是 JSON 对象")
        return data

    def _finish_body(self) -> None:
        """Deal with request bytes the handler did not read, so the connection stays in step."""
        if self._body_left <= 0:
            return
        if self._body_left <= SMALL_BODY and not self.close_connection:
            try:
                self._read_exact(self._body_left)
                return
            except (OSError, ConnectionError):
                pass
        # Too much to read: close, but let the client finish reading our answer first. Closing
        # with unread input makes Windows reset the connection and the answer can be lost.
        self.close_connection = True
        try:
            self.connection.shutdown(socket.SHUT_WR)
            self.connection.settimeout(0.5)
            deadline = time.monotonic() + 2.0
            budget = 16 * 1024 * 1024
            while budget > 0 and time.monotonic() < deadline:
                data = self.connection.recv(65536)
                if not data:
                    break
                budget -= len(data)
        except OSError:
            pass

    # ---- entry points

    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        self._dispatch()

    do_HEAD = do_POST = do_PUT = do_DELETE = do_GET

    def _dispatch(self) -> None:
        self._body_left = 0
        self._set_cookie = False
        try:
            try:
                if self.headers.get("Transfer-Encoding"):
                    self.close_connection = True
                    raise ApiError(411, "length_required", "需要 Content-Length")
                raw_length = self.headers.get("Content-Length")
                if raw_length is not None:
                    if not re.fullmatch(r"[0-9]{1,18}", raw_length.strip()):
                        self.close_connection = True
                        raise ApiError(400, "bad_request", "Content-Length 不合法")
                    self._body_left = int(raw_length)
                self._route()
            except ApiError as err:
                self._send_api_error(err)
            self._finish_body()
        except (ConnectionError, TimeoutError, socket.timeout):
            self.close_connection = True
        except Exception:  # noqa: BLE001 - a bug in one request must not kill the thread loudly
            log.exception("unhandled error for %s %s", self.command, self.path.split("?", 1)[0])
            self.close_connection = True
            try:
                self._send_json({"error": "internal", "message": "服务器内部错误"}, 500)
            except Exception:  # noqa: BLE001 - the response may already be half sent
                pass

    def _route(self) -> None:
        app = self.app
        raw_path, _, raw_query = self.path.partition("?")
        raw_path = raw_path.partition("#")[0]
        if not raw_path.startswith("/"):
            raise ApiError(400, "bad_request", "请求不合法")
        try:
            segments = [urllib.parse.unquote(s, errors="strict") for s in raw_path.split("/")[1:]]
            query = urllib.parse.parse_qs(raw_query, keep_blank_values=True, max_num_fields=20)
        except (UnicodeDecodeError, ValueError):
            raise ApiError(400, "bad_request", "请求不合法")
        if segments and segments[-1] == "":
            segments.pop()
        is_api = bool(segments) and segments[0] == "api"

        self._loopback = bool(app.loopback_check(self.client_address[0]))
        self._check_host()
        if not self._loopback and not self._authorize(segments, query, is_api):
            return
        if self.command not in ("GET", "HEAD"):
            self._check_origin()

        if is_api:
            self._route_api(segments[1:], query)
        elif not segments:
            self._static(["index.html"])
        elif segments[0] == "static" and len(segments) > 1:
            self._static(segments[1:])
        else:
            raise ApiError(404, "not_found", "页面不存在")

    def _require_method(self, *allowed: str) -> None:
        if self.command not in allowed:
            raise ApiError(405, "method_not_allowed", "不支持的请求方法", Allow=", ".join(allowed))

    def _require_loopback(self) -> None:
        if not self._loopback:
            raise ApiError(403, "loopback_only", "这个操作只能在运行扫描程序的电脑上进行")

    # ---- access control

    def _check_host(self) -> None:
        """Refuse DNS names other than localhost: a hostile site must not reach us by rebinding."""
        host = self.headers.get("Host")
        if host is None:
            return
        name = host.strip().lower()
        if name.startswith("["):
            name = name[1:].partition("]")[0]
        else:
            name = name.rsplit(":", 1)[0] if ":" in name else name
        if name == "localhost":
            return
        try:
            ipaddress.ip_address(name)
        except ValueError:
            raise ApiError(403, "bad_host", "请用 IP 地址或 localhost 访问")

    def _check_origin(self) -> None:
        """A state-changing request sent by a page must come from our own page."""
        origin = self.headers.get("Origin")
        if origin is None:
            return
        host = (self.headers.get("Host") or "").strip().lower()
        if urllib.parse.urlsplit(origin).netloc.lower() != host or not host:
            raise ApiError(403, "bad_origin", "不允许来自其他网站的请求")

    def _token_ok(self, candidate: Optional[str]) -> bool:
        token = self.app.token
        return bool(token and candidate) and hmac.compare_digest(
            candidate.encode("utf-8", "replace"), token.encode("utf-8"))

    def _authorize(self, segments: List[str], query: Dict[str, List[str]], is_api: bool) -> bool:
        """Token check for a non-loopback client. False when the response is already sent."""
        app = self.app
        from_query = (query.get("token") or [None])[0]
        from_cookie = None
        for part in (self.headers.get("Cookie") or "").split(";"):
            key, _, value = part.strip().partition("=")
            if key == TOKEN_COOKIE:
                from_cookie = value
        candidates = [from_query, self.headers.get(TOKEN_HEADER), from_cookie]
        if any(self._token_ok(c) for c in candidates):
            if self._token_ok(from_query):
                self._set_cookie = True
                if self.command == "GET" and not is_api:
                    # Take the token out of the address bar (and out of history and sharing).
                    rest = [(k, v) for k, vs in query.items() if k != "token" for v in vs]
                    target = "/" + "/".join(urllib.parse.quote(s, safe="") for s in segments)
                    if rest:
                        target += "?" + urllib.parse.urlencode(rest)
                    self.send_response(302)
                    self.send_header("Location", target)
                    self.send_header("Content-Length", "0")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    return False
            return True

        if any(candidates):  # a wrong token, not merely a missing one
            # The token is far too long to guess over the network; counting wrong ones only
            # keeps a misbehaving client cheap to answer. A correct token is never locked out
            # (a phone holding last session's cookie would otherwise block its own re-pairing).
            address = self.client_address[0]
            now = time.monotonic()
            with app.lock:
                if len(app.auth_failures) > 1000:
                    app.auth_failures.clear()
                record = app.auth_failures.get(address)
                if record is None or now - record[1] > AUTH_WINDOW_SECONDS:
                    record = app.auth_failures[address] = [0, now]
                record[0] += 1
                if record[0] > AUTH_MAX_FAILURES:
                    wait = int(AUTH_WINDOW_SECONDS - (now - record[1])) + 1
                    raise ApiError(429, "too_many_attempts", "配对码错误次数过多，请稍后再试",
                                   **{"Retry-After": wait})
        if is_api or self.command != "GET":
            raise ApiError(401, "token_required", "需要配对码：请打开电脑上显示的手机链接")
        body = _PAIRING_PAGE.encode("utf-8")
        self._send_bytes(401, body, "text/html; charset=utf-8", {
            "Content-Security-Policy":
                "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'"})
        return False

    # ---- static files

    def _static(self, parts: List[str]) -> None:
        self._require_method("GET", "HEAD")
        try:
            path = safe_join(self.app.web_dir, parts)
        except ApiError:
            raise ApiError(404, "not_found", "页面不存在")
        content_type = STATIC_TYPES.get(path.suffix.lower())
        if content_type is None or not path.is_file():
            raise ApiError(404, "not_found", "页面不存在")
        headers = {"X-Frame-Options": "DENY"}
        if path.suffix.lower() == ".html":
            headers["Content-Security-Policy"] = self.app.page_csp(path)
        self._send_file(path, content_type, headers=headers)

    # ---- API

    def _route_api(self, parts: List[str], query: Dict[str, List[str]]) -> None:
        app = self.app
        if parts == ["info"]:
            self._require_method("GET", "HEAD")
            info: Dict[str, Any] = {
                "app": "scan3d", "loopback": self._loopback,
                "stages": list(STAGES), "default_template": _default_template_id(),
                "extensions": list(ALLOWED_EXTENSIONS),
                "max_upload_bytes": MAX_UPLOAD_BYTES, "server_time": now_iso()}
            if self._loopback:   # the token is in these: never for any other client
                info["lan_urls"] = lan_urls(app)
            self._send_json(info)
            return
        if parts == ["jobs"]:
            self._require_method("GET", "HEAD", "POST")
            if self.command == "POST":
                self._api_create()
            else:
                self._send_json(app.list_jobs(loopback=self._loopback))
            return
        if parts and parts[0] == "models":
            self._api_models(parts[1:])
            return
        if parts == ["workflow", "schema"]:
            self._require_method("GET", "HEAD")
            self._send_json(workflow_schema())
            return
        if parts and parts[0] == "templates":
            self._api_templates(parts[1:])
            return
        if len(parts) < 2 or parts[0] != "jobs":
            raise ApiError(404, "not_found", "接口不存在")

        job, rest = parts[1], parts[2:]
        if not valid_job_name(job):
            raise ApiError(400, "bad_name", "任务名称不合法")
        if not rest:
            self._require_method("GET", "HEAD", "DELETE")
            if self.command == "DELETE":
                app.delete_job(job)
                self._send_json({"ok": True})
            else:
                self._send_json(self._job_or_404(job, fresh=False))
        elif rest in (["run"], ["start"]):     # `start` is the old name of `run`
            self._require_method("POST")
            body = self._read_json_body()
            reject_removed_fields(body)
            to, only, force = parse_run_request(body)
            app.start_job(job, to, only, force)
            self._send_json(self._job_or_404(job), 202)
        elif rest == ["plan"]:
            self._require_method("POST")
            app.require_job(job)
            body = self._read_json_body()
            to, only, force = parse_run_request(body)
            try:
                self._send_json(app.plan_run(job, to, only, force))
            except ApiError as exc:
                if exc.code != "nothing_to_do":
                    raise
                # a plan that is empty is an answer, not a failure (/run still says 400)
                self._send_json({"steps": [], "nothing_to_do": True, "message": exc.message,
                                 "licences": [], "models_needed": []})
        elif rest == ["workflow"]:
            self._require_method("PUT")
            app.require_job(job)
            body = dict(self._read_json_body())
            template = body.pop("template", None)
            if template is not None and not isinstance(template, str):
                raise bad_workflow(_wf.WorkflowError(["template 必须是模板名称"]))
            app.set_workflow(job, template, check_override(body))
            self._send_json(self._job_or_404(job))
        elif len(rest) == 3 and rest[0] == "stages" and rest[2] == "reset":
            self._require_method("POST")
            app.reset_stage(job, rest[1])
            self._send_json(self._job_or_404(job))
        elif rest == ["cancel"]:
            self._require_method("POST")
            app.cancel_job(job)
            self._send_json(self._job_or_404(job))
        elif rest == ["log"]:
            self._require_method("GET", "HEAD")
            app.require_job(job)
            self._send_json(app.read_log(
                job, (query.get("file") or ["pipeline.log"])[0],
                self._int_param(query, "offset"),
                self._int_param(query, "max") or DEFAULT_LOG_BYTES))
        elif len(rest) == 2 and rest[0] == "files":
            self._require_method("GET", "HEAD", "PUT", "POST")
            if self.command in ("PUT", "POST"):
                self._api_upload(job, rest[1], self._int_param(query, "offset") or 0)
            else:
                self._api_upload_state(job, rest[1])
        elif len(rest) >= 2 and rest[0] == "download":
            self._require_method("GET", "HEAD")
            self._api_download(job, rest[1:])
        elif rest == ["open-folder"]:
            self._require_method("POST")
            self._require_loopback()
            self._api_open_folder(job)
        elif rest == ["open-viewer"]:
            self._require_method("POST")
            self._require_loopback()
            self._api_open_viewer(job)
        elif rest == ["preview"]:
            self._require_method("GET", "HEAD")
            self._send_json(app.previews.overview(job))
        elif len(rest) == 2 and rest[0] == "preview":
            if rest[1] not in PREVIEW_VARIANTS:
                raise ApiError(404, "not_found", "接口不存在")
            self._require_method("GET", "HEAD", "POST")
            if self.command == "POST":
                state = app.previews.request(job, rest[1])
                self._send_json(state, 202 if state["state"] in ("queued", "building") else 200)
            else:
                self._send_file(app.previews.file(job, rest[1]), "application/octet-stream",
                                ranges=True)
        elif rest == ["cameras"]:
            self._require_method("GET", "HEAD")
            app.require_job(job)
            self._send_json(app.viewer_cameras(job))
        elif rest == ["coverage"]:
            self._require_method("GET", "HEAD", "POST")
            if self.command == "POST":
                state = app.request_coverage(job)
                self._send_json(state, 202 if state["state"] in ("queued", "building") else 200)
            else:
                self._send_json(app.coverage_state(job))
        elif rest == ["coverage", "data"]:
            self._require_method("GET", "HEAD")
            app.require_job(job)
            path = app.coverage_file(job)
            if path is None:
                raise ApiError(404, "coverage_missing", "拍摄覆盖还没有计算")
            self._send_file(path, "application/json; charset=utf-8")
        else:
            raise ApiError(404, "not_found", "接口不存在")

    @staticmethod
    def _int_param(query: Dict[str, List[str]], name: str) -> Optional[int]:
        values = query.get(name)
        if not values or values[0] == "":
            return None
        if not re.fullmatch(r"[0-9]{1,18}", values[0]):
            raise ApiError(400, "bad_request", "参数 %s 不合法" % name)
        return int(values[0])

    def _job_or_404(self, job: str, *, fresh: bool = True) -> dict:
        view = self.app.job_view(job, self.app.read_status(job, fresh=fresh), detail=True,
                                 loopback=self._loopback)
        if view is None:
            raise ApiError(404, "not_found", "找不到这个任务")
        return view

    def _api_create(self) -> None:
        app = self.app
        body = self._read_json_body()
        reject_removed_fields(body)
        name = body.get("name")
        if name == "":
            name = None   # an empty field means "generate one"; anything else must be valid
        if name is not None and not valid_job_name(name):
            raise ApiError(400, "bad_name",
                           "名称只能包含英文字母、数字、- 和 _，以字母或数字开头，最长 64 个字符")
        template = body.get("template")
        if template is not None and not isinstance(template, str):
            raise bad_workflow(_wf.WorkflowError(["template 必须是模板名称"]))
        to, only, force = parse_run_request(body)
        start = bool(body.get("start"))
        if (to or only or force) and not start:
            raise ApiError(400, "bad_request", "to、only、force 只能和 start 一起用")
        workflow = app.resolve_workflow(template or None, check_override(body.get("workflow")))
        if body.get("path") is not None:
            self._require_loopback()
            path = app.check_local_path(body.get("path"))
            if start:    # a request that cannot be run must not leave a job behind
                try:
                    _wf.plan(workflow, _wf.stage_states(workflow, {}), to=to, only=only or None,
                             force=force)
                except _wf.PlanError as exc:
                    raise ApiError(400, exc.code, exc.message)
            job = app.create_job(name, workflow, source="path", input_path=path)
            if start:
                app.start_job(job, to, only, force)
        else:
            job = app.create_job(name, workflow)
        self._send_json(self._job_or_404(job), 201)

    def _api_models(self, rest: List[str]) -> None:
        downloads = self.app.models
        if not rest:
            self._require_method("GET", "HEAD")
            self._send_json(downloads.overview(loopback=self._loopback))
        elif rest == ["download"]:
            self._require_method("POST")
            self._require_loopback()
            names = self._read_json_body().get("features")
            if names is None:
                names = []
            if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
                raise ApiError(400, "bad_feature", "features 必须是功能名称的列表")
            started = downloads.start(names)
            self._send_json(downloads.overview(loopback=True), 202 if started else 200)
        else:
            raise ApiError(404, "not_found", "接口不存在")

    def _api_templates(self, rest: List[str]) -> None:
        app = self.app
        if not rest:
            self._require_method("GET", "HEAD")
            self._send_json(templates_view())
            return
        if len(rest) != 1:
            raise ApiError(404, "not_found", "接口不存在")
        tid = rest[0]
        self._require_method("PUT", "DELETE")
        self._require_loopback()
        body = self._read_json_body() if self.command == "PUT" else {}
        with app.lock:      # read, change, write: one at a time
            user = copy.deepcopy(load_user_settings())
            if self.command == "DELETE":
                self._delete_template(tid, user)
            elif tid == "default":
                self._set_default_template(body.get("id"), user)
            else:
                self._save_template(tid, body, user)
            self._send_json(templates_view(user))

    @staticmethod
    def _set_default_template(tid: Any, user: dict) -> None:
        if not isinstance(tid, str) or tid not in [t["id"] for t in _wf.templates(user)]:
            raise ApiError(400, "bad_template", "没有这个模板：%s" % (tid,))
        try:
            _wf.resolve(tid, user=user)
        except _wf.WorkflowError as exc:
            raise bad_workflow(exc)
        user["default_template"] = tid
        save_user_settings(user)

    @staticmethod
    def _delete_template(tid: str, user: dict) -> None:
        saved = user.get("templates") if isinstance(user.get("templates"), dict) else {}
        if tid not in saved:
            if tid in _wf.TEMPLATES:
                raise ApiError(409, "builtin_template", "内置模板不能删除")
            raise ApiError(404, "not_found", "没有这个模板")
        users = [other for other, entry in saved.items()
                 if other != tid and isinstance(entry, dict) and entry.get("base") == tid]
        if users:
            raise ApiError(409, "template_in_use", "「%s」是别的模板的基础，先删除它们：%s"
                           % (tid, "、".join(users)))
        del saved[tid]
        user["templates"] = saved
        if user.get("default_template") == tid:
            del user["default_template"]
        save_user_settings(user)

    @staticmethod
    def _save_template(tid: str, body: dict, user: dict) -> None:
        if not valid_template_id(tid):
            raise ApiError(400, "bad_name", "模板名称只能包含英文字母、数字、- 和 _，最长 64 个字符")
        label, desc = body.get("label"), body.get("desc", "")
        if not isinstance(label, str) or not label.strip() or len(label) > MAX_TEMPLATE_LABEL:
            raise ApiError(400, "bad_request", "请给模板起一个名字（最多 %d 个字）" % MAX_TEMPLATE_LABEL)
        if not isinstance(desc, str) or len(desc) > 300:
            raise ApiError(400, "bad_request", "模板说明必须是文字，最多 300 个字")
        stages = body.get("stages", {})
        if not isinstance(stages, dict):
            raise bad_workflow(_wf.WorkflowError(["stages 必须是对象：{阶段: {enabled, params}}"]))
        known = [t["id"] for t in _wf.templates(user)]
        base = body.get("base") or _wf.DEFAULT_TEMPLATE
        if not isinstance(base, str) or base not in known or base == tid and tid not in _wf.TEMPLATES:
            raise bad_workflow(_wf.WorkflowError(["没有这个基础模板：%s" % (base,)]))
        saved = user.get("templates") if isinstance(user.get("templates"), dict) else {}
        if tid not in saved and len(saved) >= MAX_USER_TEMPLATES:
            raise ApiError(409, "too_many_templates", "自己保存的模板太多了（最多 %d 个）"
                           % MAX_USER_TEMPLATES)
        saved = dict(saved)
        saved[tid] = {"label": label.strip(), "desc": desc.strip(), "base": base,
                      "stages": stages}
        trial = dict(user, templates=saved)
        try:
            # the template must resolve, and the override must name real stages and parameters
            resolved = _wf.resolve(tid, None, user=trial)
            _wf.validate(copy.deepcopy(resolved), overrides=stages)
        except _wf.WorkflowError as exc:
            raise bad_workflow(exc)
        user["templates"] = saved
        save_user_settings(user)

    def _upload_paths(self, job: str, name: str) -> Tuple[Path, Path]:
        if not valid_component(name):
            raise ApiError(400, "bad_filename", "文件名不合法")
        if file_extension(name) not in ALLOWED_EXTENSIONS:
            raise ApiError(415, "bad_type", "不支持的文件类型，只接受图片和视频")
        job_dir = self.app.job_dir(job)
        return safe_join(job_dir, ["upload", name]), safe_join(job_dir, ["upload.part", name])

    def _api_upload_state(self, job: str, name: str) -> None:
        self.app.require_job(job)
        final, part = self._upload_paths(job, name)
        for path, complete in ((final, True), (part, False)):
            try:
                self._send_json({"name": name, "size": path.stat().st_size,
                                 "complete": complete})
                return
            except OSError:
                continue
        self._send_json({"name": name, "size": 0, "complete": False})

    def _api_upload(self, job: str, name: str, offset: int) -> None:
        app = self.app
        final, part = self._upload_paths(job, name)
        if self.headers.get("Content-Length") is None:
            raise ApiError(411, "length_required", "需要 Content-Length")
        length = self._body_left
        if length <= 0:
            raise ApiError(400, "empty_file", "文件是空的")
        if offset + length > MAX_UPLOAD_BYTES:
            raise ApiError(413, "too_large", "文件太大")

        with app.lock:
            status = app.require_job(job)
            if status["state"] not in (STATE_CREATED, "failed", "cancelled") \
                    or job == app.current or job in app.queue:
                raise ApiError(409, "not_accepting", "任务已经开始，不能再上传文件")
            if (app.read_meta(job) or {}).get("source") != "upload":
                raise ApiError(409, "not_accepting", "这个任务不接受上传")
            active = app.uploads.setdefault(job, set())
            if name.lower() in active:
                raise ApiError(409, "upload_busy", "这个文件正在上传中",
                               **{"Retry-After": 5})
            if not final.exists() and app.list_uploads(job)[0] + len(active) >= MAX_FILES_PER_JOB:
                raise ApiError(413, "too_many_files", "文件数量过多")
            active.add(name.lower())
        try:
            try:
                if shutil.disk_usage(app.runs_dir).free < length + DISK_RESERVE_BYTES:
                    raise ApiError(507, "disk_full", "电脑磁盘空间不足")
                part.parent.mkdir(exist_ok=True)
                have = part.stat().st_size if part.exists() else 0
                if offset and offset != have:
                    raise ApiError(409, "offset_mismatch", "续传位置不一致", offset=have)
                stream = open(part, "ab" if offset else "wb")
            except OSError as exc:
                raise ApiError(500, "io_error", "无法写入文件：%s" % exc)
            with stream:
                self._receive(stream, length)
            if self._body_left > 0:
                # The client went away mid-file. Keep what arrived so it can resume.
                self.close_connection = True
                raise ConnectionError("upload interrupted")
            try:
                os.replace(part, final)
                size = final.stat().st_size
            except OSError as exc:
                raise ApiError(500, "io_error", "无法保存文件：%s" % exc)
        finally:
            with app.lock:
                names = app.uploads.get(job)
                if names is not None:
                    names.discard(name.lower())
                    if not names:
                        app.uploads.pop(job, None)
        self._send_json({"name": name, "size": size, "complete": True}, 201)

    def _receive(self, stream: Any, length: int) -> None:
        """Copy `length` body bytes to `stream` in chunks; stops early if the client stalls."""
        app = self.app
        window_start = time.monotonic()
        window_bytes = 0
        try:
            while self._body_left > 0:
                if app.stop_event.is_set():
                    break
                chunk = self.rfile.read1(min(self._body_left, UPLOAD_CHUNK))
                if not chunk:
                    break
                stream.write(chunk)
                self._body_left -= len(chunk)
                window_bytes += len(chunk)
                now = time.monotonic()
                if now - window_start >= UPLOAD_RATE_WINDOW:
                    if window_bytes < UPLOAD_MIN_RATE * UPLOAD_RATE_WINDOW:
                        break  # trickling: as good as stalled
                    window_start, window_bytes = now, 0
        except (OSError, ConnectionError) as exc:
            if isinstance(exc, (TimeoutError, socket.timeout, ConnectionError)):
                return  # caller sees _body_left > 0
            raise ApiError(500, "io_error", "写入文件失败：%s" % exc)

    def _api_download(self, job: str, parts: List[str]) -> None:
        app = self.app
        status = app.require_job(job)
        job_dir = app.job_dir(job)
        path = safe_join(job_dir, parts)
        rel = "/".join(parts).lower()
        outputs = status.get("outputs")
        listed = {str(v).replace("\\", "/").lower()
                  for v in (outputs.values() if isinstance(outputs, dict) else ())}
        if not ((parts[0].lower() == "output" and len(parts) > 1) or rel in listed):
            raise ApiError(404, "not_found", "文件不存在")
        if not path.is_file():
            raise ApiError(404, "not_found", "文件不存在")
        content_type = DOWNLOAD_TYPES.get(path.suffix.lower(), "application/octet-stream")
        self._send_file(path, content_type, download_name="%s-%s" % (job, parts[-1]),
                        ranges=True)

    def _api_open_folder(self, job: str) -> None:
        app = self.app
        app.require_job(job)
        job_dir = app.job_dir(job)
        folder = job_dir / "output" if (job_dir / "output").is_dir() else job_dir
        explorer = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "explorer.exe"
        try:
            app.spawn_detached([str(explorer), str(folder)])
        except OSError as exc:
            raise ApiError(500, "launch_failed", "无法打开文件夹：%s" % exc)
        self._send_json({"ok": True, "path": str(folder)})

    def _api_open_viewer(self, job: str) -> None:
        app = self.app
        status = app.require_job(job)
        outputs = status.get("outputs")
        rel = outputs.get("splat") if isinstance(outputs, dict) else None
        if not isinstance(rel, str):
            raise ApiError(409, "no_splat", "这个任务还没有高斯点云结果")
        splat = safe_join(app.job_dir(job), rel.replace("\\", "/").split("/"))
        if not splat.is_file():
            raise ApiError(409, "no_splat", "找不到高斯点云文件")
        if not app.viewer_exe.is_file():
            raise ApiError(500, "no_viewer", "找不到查看器程序：%s" % app.viewer_exe)
        try:
            app.spawn_detached([str(app.viewer_exe), str(splat)])
        except OSError as exc:
            raise ApiError(500, "launch_failed", "无法启动查看器：%s" % exc)
        self._send_json({"ok": True, "path": str(splat)})


_PAIRING_PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>三维扫描 · 需要配对</title>
<style>
body{font:16px/1.6 system-ui,"Microsoft YaHei UI","PingFang SC",sans-serif;margin:0;
padding:32px 20px;background:#f5f6f8;color:#1c2126}
main{max-width:420px;margin:0 auto}
input,button{font:inherit;width:100%;box-sizing:border-box;padding:12px;margin-top:12px;
border-radius:10px;border:1px solid #c5ccd3}
button{background:#0f766e;color:#fff;border-color:#0f766e}
@media (prefers-color-scheme:dark){body{background:#111417;color:#e6e9ec}
input{background:#1a1e23;color:inherit;border-color:#3a424a}}
</style></head><body><main>
<h1>需要配对</h1>
<p>请在手机浏览器中打开电脑上显示的完整链接（包含 token），或在下面输入配对码。</p>
<form method="get" action="/">
<input name="token" autocomplete="off" autocapitalize="none" spellcheck="false"
 placeholder="配对码" aria-label="配对码" required>
<button type="submit">连接</button>
</form></main></body></html>
"""


class ScanHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    # On Windows SO_REUSEADDR would let a second server bind a port that is already in use.
    allow_reuse_address = os.name != "nt"
    request_queue_size = 64

    def __init__(self, address: Tuple[str, int], app: App) -> None:
        self.app = app
        super().__init__(address, Handler)

    def server_bind(self) -> None:
        if os.name == "nt":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        # Not HTTPServer.server_bind: it calls getfqdn(), a DNS lookup that can stall.
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = port

    def handle_error(self, request: Any, client_address: Any) -> None:
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionError, TimeoutError, socket.timeout)):
            log.debug("connection from %s dropped: %r", client_address, exc)
        else:
            log.exception("error while handling a request from %s", client_address)


def lan_urls(app: App) -> List[str]:
    return ["http://%s:%s/?token=%s" % (address, app.port, app.token)
            for address in lan_addresses()]


def create_server(port: int = 8765, *, root: Optional[os.PathLike] = None,
                  runs_dir: Optional[os.PathLike] = None,
                  web_dir: Optional[os.PathLike] = None, token: Optional[str] = None,
                  host: str = BIND_HOST) -> ScanHTTPServer:
    """Bind the server without starting anything. Call `.app.start()` and `.serve_forever()`.

    `host` is for tests, which must not open a port to the network. The product has one mode
    and always uses the default: every interface.
    """
    app = App(root, runs_dir=runs_dir, web_dir=web_dir, token=token)
    try:
        httpd = ScanHTTPServer((host, int(port)), app)
    except (OSError, OverflowError) as exc:
        raise ServerStartError(
            "无法在 %s:%s 上启动服务：这个端口可能已被占用（是否已经有一个 scan3d 在运行？），"
            "或者系统不允许程序监听网络。请关闭占用它的程序，或用 --port 换一个端口后重试。"
            " / Cannot listen on %s:%s: %s" % (host, port, host, port, exc)) from exc
    app.port = httpd.server_address[1]
    return httpd


def _open_browser(url: str) -> None:
    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:  # noqa: BLE001 - the URL is printed; opening it is a convenience
        log.debug("could not open a browser", exc_info=True)


def serve(port: int = 8765, *, open_browser: bool = True) -> None:
    """Run the web server until Ctrl+C: one mode, listening on every interface.

    Exits with a clear message if 0.0.0.0:<port> cannot be bound; it never falls back to
    another address or port.
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                            datefmt="%H:%M:%S")
    try:
        httpd = create_server(port)
    except ServerStartError as exc:
        _say("[scan3d] %s" % exc)
        raise SystemExit(1)
    app = httpd.app

    def interrupt(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    try:  # closing via Ctrl+Break or a polite terminate should clean up like Ctrl+C
        import signal
        for name in ("SIGTERM", "SIGBREAK"):
            if hasattr(signal, name):
                signal.signal(getattr(signal, name), interrupt)
    except (ValueError, OSError):  # not the main thread
        pass

    try:
        app.start()
        local_url = "http://127.0.0.1:%s/" % app.port
        _say("scan3d 已启动 / scan3d is running")
        _say("  本机 : %s" % local_url)
        urls = lan_urls(app)
        for url in urls:
            _say("  手机 : %s" % url)
        if urls:
            _say("         （手机与电脑需在同一 Wi-Fi；打不开时请在 Windows 防火墙提示中允许访问。"
                 "地址每次启动都相同，可以存为书签）")
        else:
            _say("  手机 : 没有找到这台电脑的局域网地址。连上网络后刷新本机页面，"
                 "“用手机访问”一栏会显示手机要打开的地址。")
        _say("按 Ctrl+C 停止 / press Ctrl+C to stop")
        if open_browser:
            timer = threading.Timer(0.5, _open_browser, [local_url])
            timer.daemon = True
            timer.start()
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        _say("正在停止… / stopping")
    finally:
        try:
            app.shutdown()
        except KeyboardInterrupt:  # a second Ctrl+C: still never leave the pipeline running
            for child in (app.current_proc, app.models.child):   # ... or a download
                if child is not None and child.poll() is None:
                    _proc.kill_tree(child.pid)
        httpd.server_close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="scan3d web server")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    serve(args.port, open_browser=not args.no_browser)
