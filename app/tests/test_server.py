"""Tests for app/server.py. Standard library only.

    python -m unittest discover -s app/tests          (from the project root)

Every test runs a real server on an ephemeral loopback port with a temporary runs folder, and
a stub (stub_pipeline.py) in place of the real pipeline. Nothing touches the project's runs/.
The network weights and licences are a stand-in too (FakeModels, in a temp folder): nothing
reads or writes the project's tools/, and nothing is ever downloaded.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import http.client
import inspect
import json
import logging
import os
import shutil
import re
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import server  # noqa: E402

STUB = Path(__file__).with_name("stub_pipeline.py")
TOKEN = "pairme234567"
MIB = 1024 * 1024

logging.getLogger("scan3d.server").setLevel(logging.CRITICAL)


def wait_until(predicate, timeout=20.0, interval=0.03, message="condition"):
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting for " + message)
        time.sleep(interval)


MODEL_HOOKS = ("models_status", "models_accept", "models_dir", "models_file_mb",
               "build_models_command")

# Stands in for `scan3d.py models --download`: fetches the files named into <folder>/_part and
# moves each into <folder> when it is complete, as models.download_missing does.
#   argv: folder, mode (ok | fail | hang), exit code, file=megabytes ...
#   ok    half of each file, then (once <folder>/gate is gone, if there is one) the rest
#   fail  half of the first file, an error on stderr, exit
#   hang  half of the first file, then nothing for ten minutes
FAKE_DOWNLOAD = r"""
import os, sys, time
folder, mode, code = sys.argv[1], sys.argv[2], int(sys.argv[3])
part_dir = os.path.join(folder, "_part")
os.makedirs(part_dir, exist_ok=True)
gate = os.path.join(folder, "gate")
for item in sys.argv[4:]:
    name, megabytes = item.split("=")
    total = int(megabytes) * 1024 * 1024
    part = os.path.join(part_dir, name)
    print("[download] " + name, flush=True)
    with open(part, "ab") as f:
        f.write(b"\0" * max(0, total // 2 - f.tell()))
    if mode == "fail":
        print("curl: (22) The requested URL returned error: 404", file=sys.stderr, flush=True)
        sys.exit(code)
    if mode == "hang":
        time.sleep(600)
    born = time.monotonic()
    while os.path.exists(gate) and time.monotonic() - born < 120:
        time.sleep(0.02)
    with open(part, "ab") as f:
        f.write(b"\0" * max(0, total - f.tell()))
    os.replace(part, os.path.join(folder, name))
    print("[ok] " + name, flush=True)
sys.exit(code)
"""


class FakeModels:
    """What the server asks models.py and config.py for, without either: a small catalogue,
    a temp folder for the weights, the accepted licences in memory. `status` answers in the
    shape of models.status(). Sizes are a few MB so that tests can write real partial files."""

    LICENCES = {
        "sam3": {"name": "SAM License (Meta)", "url": "https://example.invalid/sam3",
                 "used_for": "抠除移动物体"},
        "birefnet": {"name": "MIT License (BiRefNet)", "url": "https://example.invalid/birefnet",
                     "used_for": "只保留主体"},
        "dinov3": {"name": "DINOv3 License (Meta)", "url": "https://example.invalid/dinov3",
                   "used_for": "稠密点云初始化"},
    }
    FILES = {   # name -> (size in MB, licence)
        "sam3.bin": (6, "sam3"), "birefnet.bin": (2, "birefnet"), "moge.bin": (4, None),
        "roma.bin": (3, "dinov3"), "aliked.bin": (1, None), "lightglue.bin": (1, None)}
    FEATURES = {   # key -> (files, enabled)
        "masks": (["sam3.bin"], True), "subject_only": (["birefnet.bin"], False),
        "normals": (["moge.bin"], True), "dense_seed": (["roma.bin"], False),
        "sfm": (["aliked.bin", "lightglue.bin"], True)}

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.accepted = set()
        self.mode = "ok"          # how the stand-in download behaves
        self.exit_code = 1        # like the real command while any enabled feature lacks a file
        self.commands = []        # what the server asked build_models_command for

    def install(self, *names) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        for name in names or self.FILES:
            (self.folder / name).write_bytes(b"x")

    def installed(self, name) -> bool:
        return (self.folder / name).exists()

    def status(self) -> dict:
        features, pending, missing = {}, set(), []
        for key, (files, enabled) in self.FEATURES.items():
            needs = sorted({self.FILES[f][1] for f in files if self.FILES[f][1]} - self.accepted)
            absent = [f for f in files if not self.installed(f)
                      and (self.FILES[f][1] is None or self.FILES[f][1] in self.accepted)]
            features[key] = {
                "files": list(files), "enabled": enabled, "ready": not needs and not absent,
                "licences_needed": needs, "files_missing": absent,
                "download_mb": sum(self.FILES[f][0] for f in files if not self.installed(f))}
            if enabled:
                pending.update(needs)
                missing.extend(f for f in absent if f not in missing)
        return {"features": features,
                "licences": {k: dict(v, accepted=k in self.accepted, needed=k in pending)
                             for k, v in self.LICENCES.items()},
                "files_missing": missing,
                "download_mb": sum(self.FILES[f][0] for f in missing)}

    def accept(self, ids) -> None:
        unknown = [i for i in ids if i not in self.LICENCES]
        if unknown:
            raise ValueError("unknown licence id(s): %s" % unknown)
        self.accepted.update(ids)

    def command(self, features):
        self.commands.append(None if features is None else list(features))
        state = self.status()["features"]
        keys = features or [k for k, v in state.items() if v["enabled"]]
        files = []
        for key in keys:
            files.extend(f for f in state[key]["files_missing"] if f not in files)
        return [sys.executable, "-c", FAKE_DOWNLOAD, str(self.folder), self.mode,
                str(self.exit_code)] + ["%s=%d" % (f, self.FILES[f][0]) for f in files]

    def hooks(self) -> dict:
        return {"models_status": self.status, "models_accept": self.accept,
                "models_dir": lambda: self.folder,
                "models_file_mb": lambda name: self.FILES.get(name, (0, None))[0],
                "build_models_command": self.command}


class ServerCase(unittest.TestCase):
    """A running server with its own temp folder; subclasses may override prepare()."""

    def prepare(self) -> None:
        """Hook: fill self.runs before the server starts."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="scan3d-test-")).resolve()
        self.runs = self.tmp / "runs"
        self.runs.mkdir()
        self.stub_args = {}      # job -> extra stub arguments
        self.launched = []       # commands the server wanted to start detached
        self._real_build = server.build_run_command
        server.build_run_command = self._stub_command
        # No test may reach the real tools/ folder: every call into models.py and config.py
        # goes to a stand-in with its own temp folder.
        self.models = FakeModels(self.tmp / "models")
        self._real_hooks = {name: getattr(server, name) for name in MODEL_HOOKS}
        for name, hook in self.models.hooks().items():
            setattr(server, name, hook)
        self.addCleanup(self._cleanup)
        self.prepare()

        self.app = server.App(self.tmp, runs_dir=self.runs, token=TOKEN)
        self.app.spawn_detached = lambda cmd: self.launched.append(list(cmd))
        # Bound to loopback although the product listens on every interface: remote clients
        # are simulated through app.loopback_check, so no test opens a port to the network.
        self.httpd = server.ScanHTTPServer(("127.0.0.1", 0), self.app)
        self.port = self.app.port = self.httpd.server_address[1]
        self.app.start()
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       kwargs={"poll_interval": 0.05}, daemon=True)
        self.thread.start()

    def _cleanup(self) -> None:
        try:
            if hasattr(self, "httpd"):
                self.httpd.shutdown()
                self.app.shutdown(timeout=20)
                self.httpd.server_close()
                self.thread.join(5)
        finally:
            server.build_run_command = self._real_build
            for name, hook in self._real_hooks.items():
                setattr(server, name, hook)
            shutil.rmtree(self.tmp, ignore_errors=True)

    def _stub_command(self, job, input_path, quality, options=None):
        # The per-scan switches reach the stub exactly as the real command would carry them.
        real = self._real_build(job, input_path, quality, options)
        switches = real[real.index("--quality") + 2:]
        return [sys.executable, str(STUB), "run", str(input_path), "--name", job,
                "--quality", quality, "--runs", str(self.runs)] + switches \
            + self.stub_args.get(job, [])

    def stub_argv(self, job):
        return json.loads((self.runs / job / "stub_argv.json").read_text(encoding="utf-8"))

    def models_view(self, headers=None):
        status, view = self.api("GET", "/api/models", None, headers)
        self.assertEqual(status, 200, view)
        return view

    def wait_download(self, states=("done", "failed"), timeout=30.0):
        box = {}

        def reached():
            box["view"] = self.models_view()
            return box["view"]["download"]["state"] in states

        try:
            wait_until(reached, timeout, message="the download to be %s" % (states,))
        except AssertionError:
            raise AssertionError("the download never became %s: %r"
                                 % (states, box.get("view", {}).get("download")))
        return box["view"]

    # ---- HTTP helpers

    def request(self, method, path, body=None, headers=None):
        headers = dict(headers or {})
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode("utf-8")
            headers.setdefault("Content-Type", "application/json")
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        try:
            conn.request(method, path, body=body, headers=headers)
            response = conn.getresponse()
            data = response.read()
            return response.status, {k.lower(): v for k, v in response.getheaders()}, data
        finally:
            conn.close()

    def api(self, method, path, body=None, headers=None):
        status, _headers, data = self.request(method, path, body, headers)
        try:
            return status, json.loads(data.decode("utf-8")) if data else None
        except ValueError:
            return status, None

    def create(self, **body):
        status, job = self.api("POST", "/api/jobs", body or {})
        self.assertEqual(status, 201, job)
        return job["job"]

    def upload(self, job, name, data, query=""):
        return self.api("PUT", "/api/jobs/%s/files/%s%s" % (job, name, query), data,
                        {"Content-Type": "application/octet-stream"})

    def create_with_photo(self, mode="ok", step=0.02, extra=(), **body):
        job = self.create(**body)
        status, _ = self.upload(job, "a.jpg", b"\xff\xd8 not really a jpeg")
        self.assertEqual(status, 201)
        self.stub_args[job] = ["--mode", mode, "--step", str(step)] + list(extra)
        return job

    def start(self, job, body=None):
        status, view = self.api("POST", "/api/jobs/%s/start" % job, body or {})
        self.assertEqual(status, 200, view)
        return view

    def job(self, name):
        status, view = self.api("GET", "/api/jobs/" + name)
        self.assertEqual(status, 200, view)
        return view

    def wait_state(self, name, states, timeout=30.0):
        states = (states,) if isinstance(states, str) else states
        box = {}

        def reached():
            box["view"] = self.job(name)
            return box["view"]["state"] in states

        try:
            wait_until(reached, timeout, message="%s to be %s" % (name, states))
        except AssertionError:
            raise AssertionError("job %s never reached %s; last view: %r"
                                 % (name, states, box.get("view")))
        return box["view"]

    def wait_idle(self, timeout=30.0):
        wait_until(lambda: self.app.current is None and not self.app.queue, timeout,
                   message="the worker to become idle")

    def disk_status(self, name):
        return json.loads((self.runs / name / "status.json").read_text(encoding="utf-8"))

    def remote(self):
        """From now on every client counts as a phone on the Wi-Fi."""
        self.app.loopback_check = lambda address: False

    def make_done_job(self, name, gaussians=40, **outputs):
        """A finished job with a real (small) splat.ply, as the pipeline leaves it."""
        out = self.runs / name / "output"
        out.mkdir(parents=True)
        write_splat_ply(out / "splat.ply", gaussians)
        files = {"splat": "output/splat.ply"}
        files.update(outputs)
        status = {"job": name, "state": "done", "stage": "export", "stage_index": 6,
                  "stage_count": 6, "stage_progress": 1.0, "message": "完成",
                  "quality": "standard", "started_at": "2026-01-01T10:00:00+08:00",
                  "updated_at": "2026-01-01T10:30:00+08:00",
                  "finished_at": "2026-01-01T10:30:00+08:00", "error": None, "pid": None,
                  "outputs": files}
        (self.runs / name / "status.json").write_text(json.dumps(status), encoding="utf-8")
        return out

    def preview(self, job, variant):
        status, body = self.api("GET", "/api/jobs/%s/preview" % job)
        self.assertEqual(status, 200, body)
        return body["variants"][variant]

    def wait_preview(self, job, variant, states=("ready",), timeout=60.0):
        box = {}

        def reached():
            box["state"] = self.preview(job, variant)
            return box["state"]["state"] in states

        try:
            wait_until(reached, timeout, message="preview %s of %s" % (variant, job))
        except AssertionError:
            raise AssertionError("preview %s of %s never became %s: %r"
                                 % (variant, job, states, box.get("state")))
        return box["state"]


SPLAT_PROPERTIES = (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
                    + ["f_rest_%d" % i for i in range(45)]
                    + ["opacity", "scale_0", "scale_1", "scale_2",
                       "rot_0", "rot_1", "rot_2", "rot_3"])


def write_splat_ply(path: Path, count: int) -> None:
    """A 3DGS PLY in the trainer's layout (62 float32 per Gaussian) with plausible values."""
    header = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % count
    header += "".join("property float %s\n" % name for name in SPLAT_PROPERTIES) + "end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for i in range(count):
            values = [0.0] * 62
            values[0:3] = [i * 0.01, 1.0 - i * 0.02, 0.5]
            values[6:9] = [0.5, -0.5, 0.1]
            # view-dependent colour: small values, as in a real splat (-20/128 .. 20/128)
            values[9:54] = [((i * 7 + k * 3) % 41 - 20) / 128.0 for k in range(45)]
            values[54] = 4.0 - i * 0.1                   # opacity: earlier ones matter more
            values[55:58] = [-3.0, -4.0, -5.0]
            values[58:62] = [1.0, 0.0, 0.0, 0.0]
            f.write(struct.pack("<62f", *values))


# =========================================================================== static files

class StaticTests(ServerCase):
    def test_ui_files_are_served(self):
        status, headers, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers["content-type"])
        self.assertIn("三维扫描", body.decode("utf-8"))
        self.assertIn("default-src 'self'", headers["content-security-policy"])
        self.assertEqual(headers["x-content-type-options"], "nosniff")

        for path, kind, marker in (("/static/app.js", "javascript", "/api/jobs"),
                                   ("/static/style.css", "text/css", "prefers-color-scheme"),
                                   ("/static/favicon.svg", "image/svg+xml", "<svg")):
            status, headers, body = self.request("GET", path)
            self.assertEqual(status, 200, path)
            self.assertIn(kind, headers["content-type"])
            self.assertIn(marker, body.decode("utf-8"))

        status, headers, body = self.request("HEAD", "/static/app.js")
        self.assertEqual((status, body), (200, b""))
        self.assertGreater(int(headers["content-length"]), 1000)

    def test_ui_has_no_external_resources_or_agent_sniffing(self):
        for name in ("index.html", "app.js", "jobview.js", "viewer3d.js", "style.css"):
            text = (server.WEB_DIR / name).read_text(encoding="utf-8")
            self.assertNotIn("http://", text.replace("http://www.w3.org/2000/svg", ""), name)
            self.assertNotIn("https://", text, name)
            self.assertNotIn("userAgent", text, name)

    def test_unknown_paths_and_methods(self):
        self.assertEqual(self.request("GET", "/nope")[0], 404)
        self.assertEqual(self.request("GET", "/static/missing.js")[0], 404)
        self.assertEqual(self.request("GET", "/api/nope")[0], 404)
        status, headers, _ = self.request("DELETE", "/api/jobs")
        self.assertEqual(status, 405)
        self.assertIn("GET", headers["allow"])
        self.assertEqual(self.request("POST", "/static/app.js", {})[0], 405)

    def test_info(self):
        status, info = self.api("GET", "/api/info")
        self.assertEqual(status, 200)
        self.assertTrue(info["loopback"])
        self.assertNotIn("lan", info)            # there is no LAN mode to report: one mode
        self.assertEqual(info["qualities"], ["preview", "standard", "high"])
        self.assertEqual(info["stages"], list(server.STAGES))
        self.assertIn("heic", info["extensions"])
        self.assertIsInstance(info["lan_urls"], list)

    def test_viewer_files_and_vendored_libraries_are_served(self):
        for path, marker in (("/static/jobview.js", "scan3dJobView"),
                             ("/static/viewer3d.js", "SparkRenderer"),
                             ("/static/vendor/three/three.module.js", "WebGLRenderer"),
                             ("/static/vendor/three/three.core.js", "REVISION"),
                             ("/static/vendor/three/addons/controls/OrbitControls.js", "OrbitControls"),
                             ("/static/vendor/three/addons/loaders/GLTFLoader.js", "GLTFLoader"),
                             ("/static/vendor/three/addons/utils/BufferGeometryUtils.js", "toTrianglesDrawMode"),
                             ("/static/vendor/three/addons/utils/SkeletonUtils.js", "clone"),
                             ("/static/vendor/three/addons/postprocessing/Pass.js", "FullScreenQuad"),
                             ("/static/vendor/spark/spark.module.js", "SplatMesh"),
                             ("/static/vendor/qrcode-generator/qrcode.js", "qrcode")):
            status, headers, body = self.request("GET", path)
            self.assertEqual(status, 200, path)
            self.assertIn("javascript", headers["content-type"], path)
            self.assertIn(marker.encode(), body, path)
            self.assertEqual(headers["x-content-type-options"], "nosniff")
        # things in the vendor folder that are not page assets are not served
        for path in ("/static/vendor/README.md", "/static/vendor/three/LICENSE",
                     "/static/vendor/spark/LICENSE", "/static/vendor", "/static/vendor/three",
                     "/static/vendor/spark/spark.module.js.map"):
            self.assertEqual(self.request("GET", path)[0], 404, path)

    def test_every_import_of_the_viewer_is_a_local_file(self):
        """The import map and every `import ... from` in our module and the vendored modules
        resolve to files under app/web: nothing can be fetched from the internet."""
        web = server.WEB_DIR
        page = (web / "index.html").read_text(encoding="utf-8")
        mapping = json.loads(re.search(r'<script type="importmap">(.*?)</script>', page, re.S).group(1))["imports"]
        for target in mapping.values():
            self.assertTrue(target.startswith("/static/vendor/"), target)

        def resolve(spec, importer):
            if spec.startswith("/static/"):
                return web / spec[len("/static/"):]
            if spec.startswith("."):
                return (importer.parent / spec).resolve()
            for prefix, target in sorted(mapping.items(), key=lambda kv: -len(kv[0])):
                if spec == prefix or (prefix.endswith("/") and spec.startswith(prefix)):
                    return web / (target + spec[len(prefix):])[len("/static/"):]
            self.fail("%s imports %r, which the import map does not cover" % (importer, spec))

        pattern = re.compile(r"""^\s*(?:import|export)\b[^'"]*?\bfrom\s*['"]([^'"]+)['"]|^\s*import\s*['"]([^'"]+)['"]""", re.M)
        todo, seen = [web / "viewer3d.js"], set()
        while todo:
            module = todo.pop()
            if module in seen:
                continue
            seen.add(module)
            self.assertTrue(module.is_file(), module)
            self.assertIn(web.resolve(), module.resolve().parents)
            text = module.read_text(encoding="utf-8")
            if len(text) < 500_000:     # the big bundles keep their imports in the first lines
                found = pattern.findall(text)
            else:
                found = pattern.findall(text[:20000])
            for a, b in found:
                todo.append(resolve(a or b, module))
        names = {m.name for m in seen}
        self.assertLessEqual({"viewer3d.js", "three.module.js", "three.core.js", "OrbitControls.js",
                              "GLTFLoader.js", "BufferGeometryUtils.js", "SkeletonUtils.js",
                              "Pass.js", "spark.module.js"}, names)
        # our own code asks for nothing by absolute URL
        for name in ("viewer3d.js", "jobview.js", "app.js"):
            self.assertNotRegex((web / name).read_text(encoding="utf-8"), r"""['"](https?:)?//""")

    def test_vendored_files_match_their_recorded_checksums(self):
        vendor = server.WEB_DIR / "vendor"
        readme = (vendor / "README.md").read_text(encoding="utf-8")
        recorded = dict(re.findall(r"^\| `([^`]+)` \| `[^`]+` \| `([0-9a-f]{64})` \|$", readme, re.M))
        on_disk = sorted(p.relative_to(vendor).as_posix() for p in vendor.rglob("*")
                         if p.is_file() and p.name != "README.md")
        self.assertEqual(sorted(recorded), on_disk)
        for name, digest in recorded.items():
            self.assertEqual(hashlib.sha256((vendor / name).read_bytes()).hexdigest(), digest, name)
        for licence in ("MIT",):
            self.assertIn(licence, readme)
        for forbidden in ("GPL-", "AGPL-"):
            self.assertNotIn(forbidden, readme)

    def test_page_policy_allows_exactly_the_import_map(self):
        status, headers, body = self.request("GET", "/")
        policy = headers["content-security-policy"]
        match = re.search(rb'<script type="importmap">(.*?)</script>', body, re.S)
        self.assertIsNotNone(match)
        self.assertNotIn(b"\n", match.group(1))
        digest = base64.b64encode(hashlib.sha256(match.group(1)).digest()).decode("ascii")
        self.assertIn("script-src 'self' 'wasm-unsafe-eval' 'sha256-%s';" % digest, policy)
        self.assertIn("worker-src 'self' blob:", policy)
        self.assertNotIn("unsafe-inline", policy)
        self.assertNotIn("'unsafe-eval'", policy)
        self.assertNotIn("http", policy)
        # the only inline script of the page is that import map
        self.assertEqual(len(re.findall(rb"<script(?![^>]*\bsrc=)", body)), 1)
        self.assertNotIn(b"<style", body)
        self.assertNotIn(b" style=", body)

    def test_unchanged_static_files_are_not_sent_twice(self):
        status, headers, body = self.request("GET", "/static/vendor/three/three.core.js")
        self.assertEqual(status, 200)
        etag = headers["etag"]
        status, headers, again = self.request("GET", "/static/vendor/three/three.core.js",
                                              headers={"If-None-Match": etag})
        self.assertEqual((status, again), (304, b""))
        self.assertEqual(headers["etag"], etag)
        status, _headers, again = self.request("GET", "/static/vendor/three/three.core.js",
                                               headers={"If-None-Match": '"something-else"'})
        self.assertEqual((status, len(again)), (200, len(body)))


# =========================================================================== the main flow

class FlowTests(ServerCase):
    def test_create_upload_start_progress_done_download(self):
        status, created = self.api("POST", "/api/jobs", {"quality": "high"})
        self.assertEqual(status, 201)
        job = created["job"]
        self.assertRegex(job, r"^scan-\d{8}-\d{6}$")
        self.assertEqual(created["state"], "created")
        self.assertEqual(created["quality"], "high")
        self.assertIn("start", created["actions"])
        self.assertTrue((self.runs / job / "upload").is_dir())

        # starting with nothing uploaded is refused
        status, error = self.api("POST", "/api/jobs/%s/start" % job, {})
        self.assertEqual((status, error["error"]), (400, "no_input"))

        photo = os.urandom(5000)
        status, result = self.upload(job, "IMG_0001.JPG", photo)
        self.assertEqual(status, 201, result)
        self.assertEqual(result, {"name": "IMG_0001.JPG", "size": 5000, "complete": True})
        self.assertEqual((self.runs / job / "upload" / "IMG_0001.JPG").read_bytes(), photo)
        status, result = self.upload(job, "%E5%AE%A2%E5%8E%85%201.heic", b"heic bytes")
        self.assertEqual(status, 201, result)
        self.assertTrue((self.runs / job / "upload" / "客厅 1.heic").is_file())

        # a file type that is neither image nor video is refused and nothing is written
        for name in ("tool.exe", "notes.txt", "photo.jpg.exe", "noextension", "page.html"):
            status, error = self.upload(job, name, b"MZ")
            self.assertEqual((status, error["error"]), (415, "bad_type"), name)
        self.assertEqual(sorted(p.name for p in (self.runs / job / "upload").iterdir()),
                         ["IMG_0001.JPG", "客厅 1.heic"])
        status, error = self.upload(job, "empty.jpg", b"")
        self.assertEqual((status, error["error"]), (400, "empty_file"))

        view = self.job(job)
        self.assertEqual(view["uploads"], {"count": 2, "bytes": 5010})

        self.stub_args[job] = ["--step", "0.08"]
        view = self.start(job, {"quality": "preview"})
        self.assertIn(view["state"], ("queued", "running"))
        self.assertEqual(view["quality"], "preview")

        seen_stages, seen_progress = [], []

        def finished():
            view = self.job(job)
            if view["state"] == "running":
                if view["stage"] not in seen_stages:
                    seen_stages.append(view["stage"])
                if isinstance(view["stage_progress"], float):
                    seen_progress.append(view["stage_progress"])
                self.assertIn("cancel", view["actions"])
                self.assertNotIn("delete", view["actions"])
                self.assertIsInstance(view["elapsed_seconds"], float)
            return view["state"] == "done"

        wait_until(finished, 30, message="the job to finish")
        self.assertGreaterEqual(len(seen_stages), 3, seen_stages)
        self.assertEqual(seen_stages, [s for s in server.STAGES if s in seen_stages])
        self.assertTrue(any(0 < p < 1 for p in seen_progress), seen_progress)

        view = self.job(job)
        self.assertEqual(view["message"], "完成")
        self.assertEqual(view["quality"], "preview")
        self.assertIn("delete", view["actions"])
        self.assertNotIn("cancel", view["actions"])
        files = {f["key"]: f for f in view["output_files"]}
        self.assertEqual(set(files), {"splat", "mesh", "collision", "report"})
        self.assertEqual(files["splat"], {"key": "splat", "path": "output/splat.ply",
                                          "name": "splat.ply", "size": 4096})

        # the stub was started with the contract's arguments
        argv = json.loads((self.runs / job / "stub_argv.json").read_text(encoding="utf-8"))
        self.assertEqual(Path(argv["input"]), self.runs / job / "upload")
        self.assertEqual(argv["quality"], "preview")
        self.assertFalse((self.runs / job / "upload.part").exists())

        # ---- downloads, whole and by Range
        splat = bytes(range(256)) * 16
        url = "/api/jobs/%s/download/output/splat.ply" % job
        status, headers, body = self.request("GET", url)
        self.assertEqual((status, body), (200, splat))
        self.assertEqual(headers["accept-ranges"], "bytes")
        self.assertEqual(headers["content-length"], "4096")
        self.assertIn("attachment", headers["content-disposition"])
        self.assertIn("%s-splat.ply" % job, headers["content-disposition"])
        etag = headers["etag"]

        status, headers, body = self.request("GET", url, headers={"Range": "bytes=10-19"})
        self.assertEqual((status, body), (206, splat[10:20]))
        self.assertEqual(headers["content-range"], "bytes 10-19/4096")
        self.assertEqual(headers["content-length"], "10")

        status, headers, body = self.request("GET", url, headers={"Range": "bytes=4000-"})
        self.assertEqual((status, body), (206, splat[4000:]))
        self.assertEqual(headers["content-range"], "bytes 4000-4095/4096")

        status, headers, body = self.request("GET", url, headers={"Range": "bytes=-100"})
        self.assertEqual((status, body), (206, splat[-100:]))

        status, headers, body = self.request("GET", url, headers={"Range": "bytes=4090-99999"})
        self.assertEqual((status, body), (206, splat[4090:]))

        status, headers, body = self.request("GET", url, headers={"Range": "bytes=5000-"})
        self.assertEqual(status, 416)
        self.assertEqual(headers["content-range"], "bytes */4096")

        # resuming with a validator: same file -> the part, changed file -> everything
        status, _, body = self.request("GET", url, headers={"Range": "bytes=1-2", "If-Range": etag})
        self.assertEqual((status, body), (206, splat[1:3]))
        status, _, body = self.request("GET", url,
                                       headers={"Range": "bytes=1-2", "If-Range": '"other"'})
        self.assertEqual((status, body), (200, splat))

        status, headers, body = self.request("HEAD", url)
        self.assertEqual((status, body, headers["content-length"]), (200, b"", "4096"))

        status, headers, body = self.request("GET", "/api/jobs/%s/download/output/report.json" % job)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["job"], job)

        # ---- log tail
        status, chunk = self.api("GET", "/api/jobs/%s/log" % job)
        self.assertEqual(status, 200)
        self.assertTrue(chunk["exists"])
        self.assertIn("任务完成", chunk["text"])
        self.assertEqual(chunk["next_offset"], chunk["size"])
        status, more = self.api("GET", "/api/jobs/%s/log?offset=%d" % (job, chunk["size"]))
        self.assertEqual((more["text"], more["offset"]), ("", chunk["size"]))
        status, tail = self.api("GET", "/api/jobs/%s/log?max=40" % job)
        self.assertTrue(tail["truncated"])
        self.assertLessEqual(len(tail["text"].encode("utf-8")), 40)
        self.assertTrue(chunk["text"].endswith(tail["text"]))
        status, part = self.api("GET", "/api/jobs/%s/log?offset=0&max=64" % job)
        self.assertEqual(part["offset"], 0)
        self.assertTrue(chunk["text"].startswith(part["text"]))
        self.assertIn("launcher.log", view["logs"])
        self.assertEqual(view["logs"][0], "pipeline.log")
        status, launcher = self.api("GET", "/api/jobs/%s/log?file=launcher.log" % job)
        self.assertIn("stub 启动", launcher["text"])

        # ---- the job is in the list, a finished job cannot be started or uploaded to
        self.wait_idle()
        status, listing = self.api("GET", "/api/jobs")
        self.assertEqual([j["job"] for j in listing["jobs"]], [job])
        self.assertEqual((listing["queue"], listing["current"]), ([], None))
        self.assertEqual(self.api("POST", "/api/jobs/%s/start" % job, {})[0], 409)
        self.assertEqual(self.upload(job, "late.jpg", b"x")[0], 409)
        self.assertEqual(self.api("POST", "/api/jobs/%s/cancel" % job, {})[0], 409)

    def test_large_upload_is_streamed_to_disk(self):
        job = self.create()
        size = 48 * MIB
        block = os.urandom(MIB)
        digest = hashlib.sha256()
        part = self.runs / job / "upload.part" / "walk.mp4"
        final = self.runs / job / "upload" / "walk.mp4"

        sock = socket.create_connection(("127.0.0.1", self.port), timeout=30)
        try:
            sock.sendall(("PUT /api/jobs/%s/files/walk.mp4 HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                          "Content-Length: %d\r\nConnection: close\r\n\r\n" % (job, size)).encode())
            for _ in range(24):
                sock.sendall(block)
                digest.update(block)
            # Half of the body is sent and the request is still open: the bytes received so
            # far must already be in the partial file, i.e. the server is not buffering the
            # body in memory until the request completes.
            wait_until(lambda: part.exists() and os.path.getsize(part) >= 20 * MIB, 20,
                       message="the first half to reach the disk")
            self.assertFalse(final.exists())
            status, state = self.api("GET", "/api/jobs/%s/files/walk.mp4" % job)
            self.assertFalse(state["complete"])
            # the job cannot be started or deleted underneath a transfer
            status, error = self.api("POST", "/api/jobs/%s/start" % job, {})
            self.assertEqual((status, error["error"]), (409, "uploading"))
            status, error = self.api("DELETE", "/api/jobs/" + job)
            self.assertEqual((status, error["error"]), (409, "uploading"))
            status, error = self.upload(job, "walk.mp4", b"second writer")
            self.assertEqual((status, error["error"]), (409, "upload_busy"))
            for _ in range(24):
                sock.sendall(block)
                digest.update(block)
            reply = b""
            while True:
                data = sock.recv(65536)
                if not data:
                    break
                reply += data
        finally:
            sock.close()
        self.assertTrue(reply.startswith(b"HTTP/1.1 201"), reply[:200])
        self.assertLessEqual(server.UPLOAD_CHUNK, 4 * MIB)
        self.assertEqual(os.path.getsize(final), size)
        self.assertFalse(part.exists())
        on_disk = hashlib.sha256()
        with open(final, "rb") as f:
            for chunk in iter(lambda: f.read(MIB), b""):
                on_disk.update(chunk)
        self.assertEqual(on_disk.hexdigest(), digest.hexdigest())
        status, state = self.api("GET", "/api/jobs/%s/files/walk.mp4" % job)
        self.assertEqual(state, {"name": "walk.mp4", "size": size, "complete": True})

    def test_rejected_upload_is_answered_before_the_body_arrives(self):
        job = self.create()
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=20)
        try:
            sock.sendall(("PUT /api/jobs/%s/files/setup.exe HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                          "Content-Length: %d\r\n\r\n" % (job, 200 * MIB)).encode())
            sock.sendall(b"M" * MIB)   # the client has 199 MiB more to send
            reply = sock.recv(65536)
        finally:
            sock.close()
        self.assertTrue(reply.startswith(b"HTTP/1.1 415"), reply[:200])
        self.assertIn(b"Connection: close", reply)
        self.assertEqual(list((self.runs / job / "upload").iterdir()), [])
        self.assertFalse((self.runs / job / "upload.part").exists())

    def test_resume_after_stalled_upload(self):
        self.app.socket_timeout = 1.0   # applies to connections opened from now on
        job = self.create()
        content = os.urandom(300_000)
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=20)
        try:
            sock.sendall(("PUT /api/jobs/%s/files/clip.mov HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                          "Content-Length: %d\r\n\r\n" % (job, len(content))).encode())
            sock.sendall(content[:100_000])
            began = time.monotonic()
            # ... and then the phone goes silent. The server must give up by itself.
            self.assertEqual(sock.recv(65536), b"")
            self.assertLess(time.monotonic() - began, 10)
        finally:
            sock.close()
        wait_until(lambda: not self.app.uploads, 10, message="the upload slot to be released")
        status, state = self.api("GET", "/api/jobs/%s/files/clip.mov" % job)
        self.assertEqual(state, {"name": "clip.mov", "size": 100_000, "complete": False})
        self.assertEqual(self.job(job)["uploads"]["count"], 0)

        status, error = self.upload(job, "clip.mov", content[5:], "?offset=5")
        self.assertEqual((status, error["error"], error["offset"]), (409, "offset_mismatch", 100_000))
        status, result = self.upload(job, "clip.mov", content[100_000:], "?offset=100000")
        self.assertEqual(status, 201, result)
        self.assertEqual(result["size"], len(content))
        self.assertEqual((self.runs / job / "upload" / "clip.mov").read_bytes(), content)

        # uploading the same name again replaces the file
        status, result = self.upload(job, "clip.mov", b"shorter")
        self.assertEqual((status, result["size"]), (201, 7))
        self.assertEqual((self.runs / job / "upload" / "clip.mov").read_bytes(), b"shorter")

    def test_keep_alive_survives_errors_with_unread_bodies(self):
        job = self.create()
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        try:
            for _ in range(2):
                # refused before the body is read: the server must skip the body, not parse
                # it as the next request
                conn.request("PUT", "/api/jobs/%s/files/bad.txt" % job, body=b"z" * 5000)
                response = conn.getresponse()
                self.assertEqual(response.status, 415)
                response.read()
                conn.request("GET", "/api/jobs/" + job)
                response = conn.getresponse()
                self.assertEqual(response.status, 200)
                self.assertEqual(json.loads(response.read())["job"], job)
        finally:
            conn.close()

    def test_bad_requests(self):
        for body, code in (({"quality": "ultra"}, "bad_quality"),
                           ({"name": "has space"}, "bad_name"),
                           ({"name": 5}, "bad_name")):
            status, error = self.api("POST", "/api/jobs", body)
            self.assertEqual((status, error["error"]), (400, code), body)
        status, error = self.api("POST", "/api/jobs", b"{not json",
                                 {"Content-Type": "application/json"})
        self.assertEqual((status, error["error"]), (400, "bad_json"))
        status, error = self.api("POST", "/api/jobs", b"[1]", {"Content-Type": "application/json"})
        self.assertEqual((status, error["error"]), (400, "bad_json"))
        status, error = self.api("POST", "/api/jobs", b"x" * 70000)
        self.assertEqual(status, 413)

        job = self.create(name="living-room_2")
        self.assertEqual(job, "living-room_2")
        status, error = self.api("POST", "/api/jobs", {"name": "living-room_2"})
        self.assertEqual((status, error["error"]), (409, "exists"))
        status, error = self.api("POST", "/api/jobs", {"name": "LIVING-ROOM_2"})
        self.assertEqual(status, 409 if os.name == "nt" else 201)

        for method, path in (("GET", "/api/jobs/nope"), ("DELETE", "/api/jobs/nope"),
                             ("POST", "/api/jobs/nope/start"), ("POST", "/api/jobs/nope/cancel"),
                             ("GET", "/api/jobs/nope/log"), ("PUT", "/api/jobs/nope/files/a.jpg"),
                             ("GET", "/api/jobs/nope/download/output/splat.ply")):
            status, error = self.api(method, path, b"x" if method == "PUT" else None)
            self.assertEqual((status, error["error"]), (404, "not_found"), path)
        status, error = self.api("POST", "/api/jobs/%s/start" % job, {"quality": "best"})
        self.assertEqual((status, error["error"]), (400, "bad_quality"))
        status, error = self.api("GET", "/api/jobs/%s/log?offset=abc" % job)
        self.assertEqual(status, 400)
        status, chunk = self.api("GET", "/api/jobs/%s/log" % job)
        self.assertEqual((status, chunk["exists"], chunk["text"]), (200, False, ""))

    def test_generated_names_do_not_collide(self):
        names = {self.create() for _ in range(4)}
        self.assertEqual(len(names), 4)
        for name in names:
            self.assertTrue(server.valid_job_name(name), name)


# =========================================================================== queue and worker

class QueueTests(ServerCase):
    def test_two_jobs_run_one_after_another(self):
        first = self.create_with_photo(step=0.05, name="first")
        second = self.create_with_photo(step=0.05, name="second")
        self.start(first)
        view = self.start(second)
        self.assertEqual(view["state"], "queued")

        overlap = []
        waited_in_queue = []

        def both_done():
            status, listing = self.api("GET", "/api/jobs")
            states = {j["job"]: j for j in listing["jobs"]}
            if states[first]["state"] == "running" and states[second]["state"] == "running":
                overlap.append(listing)
            if states[first]["state"] == "running" and states[second]["state"] == "queued":
                waited_in_queue.append(states[second]["queue_position"])
                self.assertEqual(listing["current"], first)
                self.assertEqual(listing["queue"], [second])
            return states[first]["state"] == "done" and states[second]["state"] == "done"

        wait_until(both_done, 40, message="both jobs to finish")
        self.assertEqual(overlap, [])
        self.assertTrue(waited_in_queue)
        self.assertEqual(set(waited_in_queue), {1})

        events = [json.loads(line) for line in
                  (self.runs / "_stub_events.log").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([(e["job"], e["event"]) for e in events],
                         [(first, "start"), (first, "end"), (second, "start"), (second, "end")])
        self.assertLessEqual(events[1]["t"], events[2]["t"])
        self.wait_idle()

    def test_cancel_running_job(self):
        job = self.create_with_photo(step=5)
        self.start(job)
        view = self.wait_state(job, "running")
        pid = view["pid"]
        self.assertTrue(server.pid_alive(pid))
        status, error = self.api("DELETE", "/api/jobs/" + job)
        self.assertEqual((status, error["error"]), (409, "running"))
        self.assertEqual(self.api("POST", "/api/jobs/%s/start" % job, {})[0], 409)

        began = time.monotonic()
        status, view = self.api("POST", "/api/jobs/%s/cancel" % job, {})
        self.assertEqual(status, 200)
        self.assertTrue(view["cancelling"])
        self.assertTrue((self.runs / job / "cancel").exists())
        view = self.wait_state(job, "cancelled", 15)
        self.assertLess(time.monotonic() - began, 10)
        self.assertIsNotNone(view["finished_at"])
        self.wait_idle()
        self.assertFalse(server.pid_alive(pid))
        self.assertIn("start", self.job(job)["actions"])

        # a cancelled job can be started again and then runs to the end
        self.stub_args[job] = ["--step", "0.01"]
        self.start(job)
        self.assertEqual(self.wait_state(job, ("done", "failed", "cancelled"))["state"], "done")

    def test_cancel_queued_job(self):
        running = self.create_with_photo(step=5, name="running")
        waiting = self.create_with_photo(name="waiting")
        self.start(running)
        self.start(waiting)
        self.wait_state(running, "running")
        self.assertEqual(self.job(waiting)["queue_position"], 1)
        status, view = self.api("POST", "/api/jobs/%s/cancel" % waiting, {})
        self.assertEqual((status, view["state"]), (200, "cancelled"))
        self.assertEqual(list(self.app.queue), [])
        self.assertFalse((self.runs / waiting / "stub_argv.json").exists())
        self.api("POST", "/api/jobs/%s/cancel" % running, {})
        self.wait_state(running, "cancelled", 15)
        self.wait_idle()
        self.assertFalse((self.runs / waiting / "stub_argv.json").exists())

    def test_cancel_in_the_first_moment_is_not_lost(self):
        # The pipeline deletes a leftover cancel file when it starts. A request that arrives
        # just before that must still cancel the job.
        job = self.create_with_photo(step=5, extra=["--startup-delay", "0.8"])
        self.start(job)
        wait_until(lambda: self.app.current == job and self.app.current_proc is not None, 10,
                   message="the process to be spawned")
        self.assertEqual(self.api("POST", "/api/jobs/%s/cancel" % job, {})[0], 200)
        self.wait_state(job, "cancelled", 15)
        self.wait_idle()

    def test_cancel_kills_a_pipeline_that_ignores_it(self):
        self.app.cancel_grace = 0.6
        job = self.create_with_photo(mode="hang", extra=["--child"])
        self.start(job)
        view = self.wait_state(job, "running")
        pid = view["pid"]
        child_pid = int(wait_until(lambda: (self.runs / job / "child.pid").exists()
                                   and (self.runs / job / "child.pid").read_text().strip(),
                                   10, message="the grandchild pid"))
        self.assertTrue(server.pid_alive(child_pid))
        self.api("POST", "/api/jobs/%s/cancel" % job, {})
        view = self.wait_state(job, "cancelled", 20)
        self.wait_idle()
        self.assertFalse(server.pid_alive(pid))
        wait_until(lambda: not server.pid_alive(child_pid), 10, message="the grandchild to die")
        self.assertIn("exit_code", view)

    def test_pipeline_crash_is_reported_with_exit_code(self):
        job = self.create_with_photo(mode="crash")
        self.start(job)
        view = self.wait_state(job, ("failed", "done", "cancelled"))
        self.assertEqual(view["state"], "failed")
        self.assertEqual(view["exit_code"], 3)
        self.assertIn("3", view["error"])
        self.assertIn("stub: crashing now", view["error"])   # the launcher log tail
        self.assertNotIn("stub_pipeline.py", view["error"])  # ... without our command line
        self.assertEqual(view["stage"], "sfm")               # where it died stays visible
        self.assertIsNone(view["pid"])
        self.assertIsNotNone(view["finished_at"])
        self.wait_idle()

        # the queue keeps working after a crash, and the failed job can be retried
        self.stub_args[job] = ["--mode", "ok", "--step", "0.01"]
        self.start(job)
        self.assertEqual(self.wait_state(job, ("failed", "done"))["state"], "done")

    def test_pipeline_exit_before_any_status(self):
        job = self.create_with_photo(mode="silent")
        self.start(job)
        view = self.wait_state(job, ("failed", "done", "cancelled"))
        self.assertEqual((view["state"], view["exit_code"]), ("failed", 5))
        self.assertIn("5", view["error"])
        self.assertIn("stub: silent exit", view["error"])

    def test_pipeline_reported_failure_is_kept(self):
        job = self.create_with_photo(mode="fail")
        self.start(job)
        view = self.wait_state(job, ("failed", "done", "cancelled"))
        self.assertEqual((view["state"], view["error"]), ("failed", "stub: sfm failed"))
        self.assertNotIn("exit_code", view)
        self.wait_idle()

    def test_missing_pipeline_script_fails_the_job(self):
        job = self.create_with_photo()
        missing = str(self.tmp / "no-such-script.py")
        server.build_run_command = lambda j, i, q, o=None: [sys.executable, missing, "run", i]
        self.start(job)
        view = self.wait_state(job, ("failed", "done", "cancelled"))
        self.assertEqual(view["state"], "failed")
        self.assertEqual(view["exit_code"], 2)
        self.assertIn("no-such-script.py", view["error"])

        server.build_run_command = lambda j, i, q, o=None: [str(self.tmp / "no-such-program.exe")]
        other = self.create_with_photo()
        self.start(other)
        view = self.wait_state(other, ("failed", "done", "cancelled"))
        self.assertEqual(view["state"], "failed")
        self.wait_idle()

    def test_process_that_never_exits_does_not_block_the_queue(self):
        self.app.exit_grace = 0.5
        stuck = self.create_with_photo(mode="linger", name="stuck")
        after = self.create_with_photo(name="after")
        self.start(stuck)
        self.start(after)
        self.assertEqual(self.wait_state(stuck, ("done", "failed"))["state"], "done")
        self.assertEqual(self.wait_state(after, ("done", "failed"), 40)["state"], "done")
        self.assertEqual(self.job(stuck)["state"], "done")
        self.wait_idle()

    def test_delete(self):
        job = self.create_with_photo(step=5)
        self.start(job)
        self.wait_state(job, "running")
        self.assertEqual(self.api("DELETE", "/api/jobs/" + job)[0], 409)
        self.assertTrue((self.runs / job).is_dir())
        self.api("POST", "/api/jobs/%s/cancel" % job, {})
        self.wait_state(job, "cancelled", 15)
        self.wait_idle()
        status, result = self.api("DELETE", "/api/jobs/" + job)
        self.assertEqual((status, result), (200, {"ok": True}))
        self.assertFalse((self.runs / job).exists())
        self.assertEqual([p.name for p in self.runs.iterdir() if p.is_dir()], [])
        self.assertEqual(self.api("GET", "/api/jobs/" + job)[0], 404)
        self.assertEqual(self.api("GET", "/api/jobs")[1]["jobs"], [])

        # a queued job is taken out of the queue and deleted
        blocker = self.create_with_photo(step=5, name="blocker")
        queued = self.create_with_photo(name="queued")
        self.start(blocker)
        self.start(queued)
        self.wait_state(blocker, "running")
        self.assertEqual(self.api("DELETE", "/api/jobs/" + queued)[0], 200)
        self.assertFalse((self.runs / queued).exists())
        self.assertEqual(list(self.app.queue), [])
        self.api("POST", "/api/jobs/%s/cancel" % blocker, {})
        self.wait_state(blocker, "cancelled", 15)

        # folders that are not jobs cannot be deleted through the API
        (self.runs / "south-building").mkdir()
        (self.runs / "south-building" / "data.bin").write_bytes(b"keep")
        self.assertEqual(self.api("DELETE", "/api/jobs/south-building")[0], 404)
        self.assertTrue((self.runs / "south-building" / "data.bin").is_file())

    def test_job_from_local_path(self):
        photos = self.tmp / "my photos"
        photos.mkdir()
        (photos / "a.jpg").write_bytes(b"x")
        self.stub_args["from-disk"] = ["--step", "0.01"]
        status, view = self.api("POST", "/api/jobs", {"name": "from-disk", "path": str(photos),
                                                      "quality": "preview", "start": True})
        self.assertEqual(status, 201, view)
        self.assertEqual((view["source"], view["input_path"]), ("path", str(photos)))
        self.assertEqual(self.wait_state("from-disk", ("done", "failed"))["state"], "done")
        argv = json.loads((self.runs / "from-disk" / "stub_argv.json").read_text(encoding="utf-8"))
        self.assertEqual((argv["input"], argv["quality"]), (str(photos), "preview"))
        self.assertFalse((self.runs / "from-disk" / "upload").exists())
        self.assertEqual(self.upload("from-disk", "a.jpg", b"x")[0], 409)
        self.assertEqual(sorted(p.name for p in photos.iterdir()), ["a.jpg"])   # untouched

        # quoted, as "Copy as path" in Explorer produces it; and a single video file
        video = self.tmp / "walk.MP4"
        video.write_bytes(b"x")
        status, view = self.api("POST", "/api/jobs", {"path": '"%s"' % video})
        self.assertEqual((status, view["state"], view["input_path"]), (201, "created", str(video)))

        empty = self.tmp / "empty"
        empty.mkdir()
        for bad in ("relative\\folder", str(self.tmp / "missing"), str(empty),
                    str(photos / "a.jpg"), "", "   ", 7, ["x"]):
            status, error = self.api("POST", "/api/jobs", {"path": bad})
            self.assertEqual((status, error["error"]), (400, "bad_path"), bad)

        # the folder disappears before a retry: a clear error instead of a doomed run
        shutil.rmtree(photos)
        self.stub_args["gone"] = []
        photos.mkdir()
        (photos / "a.jpg").write_bytes(b"x")
        status, view = self.api("POST", "/api/jobs", {"name": "gone", "path": str(photos)})
        shutil.rmtree(photos)
        status, error = self.api("POST", "/api/jobs/gone/start", {})
        self.assertEqual((status, error["error"]), (400, "bad_path"))

    def test_open_folder_and_viewer(self):
        job = self.create_with_photo(step=0.01)
        status, result = self.api("POST", "/api/jobs/%s/open-folder" % job, {})
        self.assertEqual(status, 200)
        self.assertEqual(Path(self.launched[-1][0]).name.lower(), "explorer.exe")
        self.assertEqual(self.launched[-1][1], str(self.runs / job))
        status, error = self.api("POST", "/api/jobs/%s/open-viewer" % job, {})
        self.assertEqual((status, error["error"]), (409, "no_splat"))

        self.start(job)
        view = self.wait_state(job, ("done", "failed"))
        self.assertIn("open_folder", view["actions"])
        self.assertNotIn("open_viewer", view["actions"])   # the viewer is not installed here
        status, error = self.api("POST", "/api/jobs/%s/open-viewer" % job, {})
        self.assertEqual((status, error["error"]), (500, "no_viewer"))

        viewer = self.tmp / "tools" / "spirula" / "spirula.exe"
        viewer.parent.mkdir(parents=True)
        viewer.write_bytes(b"")
        self.assertIn("open_viewer", self.job(job)["actions"])
        status, result = self.api("POST", "/api/jobs/%s/open-viewer" % job, {})
        self.assertEqual(status, 200, result)
        self.assertEqual(self.launched[-1],
                         [str(viewer), str(self.runs / job / "output" / "splat.ply")])
        self.api("POST", "/api/jobs/%s/open-folder" % job, {})
        self.assertEqual(self.launched[-1][1], str(self.runs / job / "output"))
        self.assertEqual(self.request("GET", "/api/jobs/%s/open-viewer" % job)[0], 405)


# =========================================================================== hostile input

class TraversalTests(ServerCase):
    def setUp(self):
        super().setUp()
        self.secret = self.tmp / "secret.txt"
        self.secret.write_text("top secret", encoding="utf-8")
        self.job_name = self.create(name="victim")
        self.upload(self.job_name, "a.jpg", b"jpeg")

    def tearDown(self):
        # whatever was tried, the only things in the temp folder are the ones put there
        self.assertEqual(sorted(p.name for p in self.tmp.iterdir()),
                         sorted(self.expected_top_level))
        self.assertEqual(self.secret.read_text(encoding="utf-8"), "top secret")

    expected_top_level = ("runs", "secret.txt")

    def assert_refused(self, method, path, body=None):
        status, _headers, data = self.request(method, path, body)
        self.assertIn(status, (400, 403, 404, 405), "%s %s -> %s %r" % (method, path, status, data[:200]))
        self.assertNotIn(b"top secret", data)
        return status

    def test_job_names(self):
        for name in ("..", "%2e%2e", "..%2f..%2fx", "%2e%2e%5cx", "..%5c..%5cx", "CON", "nul",
                     "COM1", "lpt9", "C:%5Cx", "c:", "a%00b", ".hidden", "-x", "_meshlab",
                     "a.b", "a%20b", "%E5%90%8D", "x" * 65, "a%2fb", "%ff"):
            self.assert_refused("GET", "/api/jobs/" + name)
            self.assert_refused("DELETE", "/api/jobs/" + name)
            self.assert_refused("POST", "/api/jobs/%s/start" % name, b"{}")
            self.assert_refused("PUT", "/api/jobs/%s/files/a.jpg" % name, b"x")
        for name in ("../evil", "..", ".", "a/b", "a\\b", "CON", "nul", "com1", "LPT1", "aux",
                     "x" * 65, "-abc", "_abc", "名字", " a", "a ", "a.b", "a\x00b", "C:\\x",
                     "/abs", "\\\\server\\share"):
            status, error = self.api("POST", "/api/jobs", {"name": name})
            self.assertEqual((status, error["error"]), (400, "bad_name"), repr(name))
        self.assertEqual([p.name for p in self.runs.iterdir()], ["victim"])

    def test_upload_names(self):
        for name in ("..%2f..%2fevil.jpg", "%2e%2e%2fevil.jpg", "..%5cevil.jpg",
                     "..%5c..%5c..%5cevil.jpg", "C:%5Cevil.jpg", "C:evil.jpg", "%2fevil.jpg",
                     "%5cevil.jpg", "evil.jpg:stream.jpg", "NUL.jpg", "con.jpg", "COM1.png",
                     ".hidden.jpg", "..jpg", "evil.jpg.", "evil.jpg%20", "%20evil.jpg",
                     "a%00.jpg", "a%0a.jpg", "%e2%80%aegpj.exe.jpg", "a%3f.jpg", "a%2a.jpg",
                     "x/../../evil.jpg", "../evil.jpg", "..", "%2e%2e", "a%7c.jpg", "%ff.jpg",
                     "y" * 150 + ".jpg"):
            self.assert_refused("PUT", "/api/jobs/victim/files/" + name, b"evil")
            self.assert_refused("GET", "/api/jobs/victim/files/" + name)
        self.assertEqual([p.name for p in (self.runs / "victim" / "upload").iterdir()], ["a.jpg"])
        self.assertFalse(any(self.runs.rglob("evil*")))

    def test_downloads_logs_and_static(self):
        (self.runs / "victim" / "output").mkdir()
        (self.runs / "victim" / "output" / "splat.ply").write_bytes(b"ply")
        status = self.disk_status("victim")
        status.update(state="done", outputs={"splat": "output/splat.ply",
                                             "evil": "../../secret.txt",
                                             "abs": str(self.secret),
                                             "meta": "server-job.json"})
        (self.runs / "victim" / "status.json").write_text(json.dumps(status), encoding="utf-8")
        view = wait_until(lambda: (lambda v: v["state"] == "done" and v)(self.job("victim")))
        self.assertEqual(sorted(f["path"] for f in view["output_files"]),
                         ["output/splat.ply", "server-job.json"])
        self.assertEqual(self.request("GET", "/api/jobs/victim/download/output/splat.ply")[2], b"ply")

        for path in ("..%2fstatus.json", "../status.json", "..%2f..%2fsecret.txt",
                     "../../secret.txt", "..%5c..%5csecret.txt", "output/..%2f..%2f..%2fsecret.txt",
                     "output/%2e%2e/%2e%2e/%2e%2e/secret.txt", "output/../status.json",
                     "status.json", "upload/a.jpg", "launcher.log", "output", "output/",
                     "C:%5CWindows%5Cwin.ini", "%2f%2f..%2f..%2fsecret.txt", "output/nul",
                     "output/splat.ply%00.txt", "output/splat.ply:stream",
                     str(self.secret).replace("\\", "%5C").replace(":", "%3A")):
            self.assert_refused("GET", "/api/jobs/victim/download/" + path)

        for name in ("../status.json", "..%2f..%2fsecret.txt", "..%5c..%5csecret.txt",
                     "C:%5Cx.log", "launcher.log%2f..%2f..%2fsecret.txt", "status.json",
                     "secret.txt", "nul.log", "%2e%2e%2fserver-job.json"):
            self.assert_refused("GET", "/api/jobs/victim/log?file=" + name)

        for path in ("/static/..%2fserver.py", "/static/../server.py", "/static/%2e%2e/server.py",
                     "/static/..%5cserver.py", "/static/..%5c..%5cproc.py", "/../server.py",
                     "/..%2f..%2f..%2fWindows/win.ini", "/static/C:%5CWindows%5Cwin.ini",
                     "/static/index.html%00.js", "/static//app.js", "/static/.", "/static/..",
                     "//127.0.0.1/static/app.js", "/static/app.js/..%2f..%2fserver.py",
                     "/static/nul", "/server.py", "/static/%ff", "/%2e%2e/%2e%2e/secret.txt"):
            status, _headers, data = self.request("GET", path)
            self.assertIn(status, (400, 404), path)
            self.assertNotIn(b"ThreadingHTTPServer", data, path)

    def test_viewer_routes(self):
        """The preview, camera and vendor routes cannot be talked into reading or writing
        anything outside the job folder and app/web."""
        output = self.runs / "victim" / "output"
        output.mkdir()
        write_splat_ply(output / "splat.ply", 3)
        status = self.disk_status("victim")
        status.update(state="done", outputs={"splat": "../../secret.txt"})
        (self.runs / "victim" / "status.json").write_text(json.dumps(status), encoding="utf-8")
        wait_until(lambda: self.job("victim")["state"] == "done")
        # a status.json that points the splat outside the job: there is no splat, nothing runs
        self.assertEqual(self.preview("victim", "full")["state"], "no_splat")
        self.assertEqual(self.api("POST", "/api/jobs/victim/preview/full")[0], 409)
        for outside in (str(self.secret), "C:\\Windows\\win.ini", "output/../../../secret.txt",
                        "..\\..\\secret.txt", "output/splat.ply:stream", "nul"):
            status["outputs"] = {"splat": outside}
            (self.runs / "victim" / "status.json").write_text(json.dumps(status), encoding="utf-8")
            wait_until(lambda: self.app.read_status("victim", fresh=True)["outputs"]["splat"] == outside)
            self.assertEqual(self.preview("victim", "full")["state"], "no_splat", outside)
            self.assertEqual(self.api("POST", "/api/jobs/victim/preview/full")[0], 409, outside)
        self.assertIsNone(self.app.previews._thread)

        for path in ("preview/..%2f..%2f..%2fsecret.txt", "preview/..", "preview/%2e%2e",
                     "preview/full/..%2f..%2fstatus.json", "preview/full/extra",
                     "preview/..%5c..%5csecret.txt", "preview/full%00", "preview/full%2f..",
                     "preview/C:%5CWindows%5Cwin.ini", "preview/nul", "preview/preview.spz",
                     "preview/full/../../../secret.txt", "cameras/..%2f..%2fsecret.txt",
                     "cameras/x", "cameras%2f..%2f..%2fsecret.txt", "preview%2ffull",
                     "web-preview/preview.spz", "download/web-preview/preview.spz",
                     "download/dataset/sparse/images.bin"):
            self.assert_refused("GET", "/api/jobs/victim/" + path)
            self.assert_refused("POST", "/api/jobs/victim/" + path, b"{}")
        for name in ("..", "%2e%2e", "..%2fvictim", "CON", ".server-token", "victim%00", "a%2fb"):
            for rest in ("preview", "preview/full", "cameras"):
                self.assert_refused("GET", "/api/jobs/%s/%s" % (name, rest))
                self.assert_refused("POST", "/api/jobs/%s/%s" % (name, rest), b"{}")

        for path in ("/static/vendor/..%2f..%2fserver.py", "/static/vendor/../../server.py",
                     "/static/vendor/three/..%2f..%2f..%2fserver.py",
                     "/static/vendor/three/addons/..%5c..%5c..%5c..%5cserver.py",
                     "/static/vendor/%2e%2e/%2e%2e/server.py", "/static/vendor/spark/nul",
                     "/static/vendor/spark/spark.module.js%00.txt",
                     "/static/vendor/spark/spark.module.js/..%2f..%2f..%2f..%2fserver.py",
                     "/static/vendor/C:%5CWindows%5Cwin.ini", "/static/vendor//three/three.core.js",
                     "/static/vendor/three/three.core.js:stream", "/static/viewer3d.js%2f..%2f..%2fserver.py"):
            status_code, _headers, data = self.request("GET", path)
            self.assertIn(status_code, (400, 404), path)
            self.assertNotIn(b"ThreadingHTTPServer", data, path)

    def test_junctions_under_the_viewer_folders_are_not_followed(self):
        if os.name != "nt":
            self.skipTest("junctions are a Windows feature")
        outside = self.tmp / "outside"
        outside.mkdir()
        (outside / "preview.spz").write_bytes(b"private preview")
        (outside / "preview.spz.json").write_text("{}", encoding="utf-8")
        write_images_bin(outside / "images.bin", [(1, 0, 0, 0, 0, 0, -5)])
        self.expected_top_level = ("runs", "secret.txt", "outside")
        output = self.runs / "victim" / "output"
        output.mkdir()
        write_splat_ply(output / "splat.ply", 3)
        (self.runs / "victim" / "dataset").mkdir()
        status = self.disk_status("victim")
        status.update(state="done", outputs={"splat": "output/splat.ply"})
        (self.runs / "victim" / "status.json").write_text(json.dumps(status), encoding="utf-8")
        wait_until(lambda: self.job("victim")["state"] == "done")
        for link in (self.runs / "victim" / "web-preview", self.runs / "victim" / "dataset" / "sparse"):
            made = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(outside)],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
            if made.returncode != 0:
                self.skipTest("could not create a junction")
        # the file behind the link is neither reported nor served, and nothing is built into it
        self.assertEqual(self.preview("victim", "full")["state"], "missing")
        status_code, _headers, data = self.request("GET", "/api/jobs/victim/preview/full")
        self.assertEqual(status_code, 404)
        self.assertNotIn(b"private", data)
        self.assertIn(self.api("POST", "/api/jobs/victim/preview/full")[0], (200, 202))
        state = self.wait_preview("victim", "full", ("failed", "ready"))
        self.assertEqual(state["state"], "failed")
        self.assertEqual((outside / "preview.spz").read_bytes(), b"private preview")
        self.assertEqual(sorted(p.name for p in outside.iterdir()),
                         ["images.bin", "preview.spz", "preview.spz.json"])
        self.assertEqual(self.api("GET", "/api/jobs/victim/cameras")[1], {"count": 0, "cameras": []})

    def test_junction_inside_a_job_is_not_followed(self):
        if os.name != "nt":
            self.skipTest("junctions are a Windows feature")
        outside = self.tmp / "outside"
        outside.mkdir()
        (outside / "private.ply").write_bytes(b"private")
        self.expected_top_level = ("runs", "secret.txt", "outside")
        output = self.runs / "victim" / "output"
        output.mkdir()
        made = subprocess.run(["cmd", "/c", "mklink", "/J", str(output / "link"), str(outside)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
        if made.returncode != 0:
            self.skipTest("could not create a junction")
        self.assertEqual((output / "link" / "private.ply").read_bytes(), b"private")
        status, _headers, data = self.request("GET", "/api/jobs/victim/download/output/link/private.ply")
        self.assertEqual(status, 400)
        self.assertNotIn(b"private", data)
        # deleting the job removes the link, not what it points to
        self.assertEqual(self.api("DELETE", "/api/jobs/victim")[0], 200)
        self.assertFalse((self.runs / "victim").exists())
        self.assertEqual((outside / "private.ply").read_bytes(), b"private")

    def test_cross_site_and_rebinding_requests_are_refused(self):
        origin = "http://127.0.0.1:%d" % self.port
        status, error = self.api("POST", "/api/jobs", {}, {"Origin": "http://evil.example"})
        self.assertEqual((status, error["error"]), (403, "bad_origin"))
        status, error = self.api("POST", "/api/jobs", {}, {"Origin": "null"})
        self.assertEqual(status, 403)
        status, error = self.api("DELETE", "/api/jobs/victim", None, {"Origin": "http://127.0.0.1:1"})
        self.assertEqual(status, 403)
        self.assertTrue((self.runs / "victim").is_dir())
        status, view = self.api("POST", "/api/jobs", {"name": "same-origin"}, {"Origin": origin})
        self.assertEqual(status, 201)
        self.assertEqual(self.api("DELETE", "/api/jobs/same-origin", None, {"Origin": origin})[0], 200)

        status, error = self.api("GET", "/api/jobs", None, {"Host": "evil.example"})
        self.assertEqual((status, error["error"]), (403, "bad_host"))
        self.assertEqual(self.request("GET", "/", None, {"Host": "evil.example:%d" % self.port})[0], 403)
        for host in ("localhost:%d" % self.port, "127.0.0.1", "[::1]:%d" % self.port,
                     "192.168.1.20:%d" % self.port):
            self.assertEqual(self.api("GET", "/api/jobs", None, {"Host": host})[0], 200, host)

    def test_malformed_requests(self):
        for raw in (b"GET /api/jobs HTTP/1.1\r\nHost: x\r\nContent-Length: abc\r\n\r\n",
                    b"POST /api/jobs HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: -5\r\n\r\n",
                    b"POST /api/jobs HTTP/1.1\r\nHost: 127.0.0.1\r\nTransfer-Encoding: chunked\r\n\r\n"
                    b"5\r\nhello\r\n0\r\n\r\n",
                    b"GET api/jobs HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n",
                    b"GET http://evil.example/api/jobs HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n",
                    b"BREW /api/jobs HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n"):
            sock = socket.create_connection(("127.0.0.1", self.port), timeout=10)
            try:
                sock.sendall(raw)
                reply = sock.recv(65536)
            finally:
                sock.close()
            self.assertRegex(reply[:12], rb"HTTP/1\.[01] (4\d\d|501)", raw)
        # bytes that are not HTTP at all, and a client that connects and says nothing
        for raw in (b"\x00\x01\x02 garbage\r\n\r\n", b"GET / JUNK/9.9\r\n\r\n", b"\r\n\r\n",
                    b"GET /" + b"a" * 70000 + b" HTTP/1.1\r\n\r\n"):
            sock = socket.create_connection(("127.0.0.1", self.port), timeout=10)
            try:
                sock.sendall(raw)
                sock.settimeout(5)
                try:
                    reply = sock.recv(65536)
                except OSError:
                    reply = b""
            finally:
                sock.close()
            self.assertNotIn(b" 200 ", reply[:20])
        self.app.socket_timeout = 0.5
        silent = socket.create_connection(("127.0.0.1", self.port), timeout=10)
        try:
            began = time.monotonic()
            self.assertEqual(silent.recv(100), b"")   # the server hangs up on its own
            self.assertLess(time.monotonic() - began, 5)
        finally:
            silent.close()
        self.assertEqual(self.api("GET", "/api/jobs")[0], 200)   # still serving


# =========================================================================== LAN mode and tokens

class TokenTests(ServerCase):
    def test_loopback_needs_no_token(self):
        real = server.lan_addresses
        server.lan_addresses = lambda: ["192.168.1.20", "10.0.0.7"]
        self.addCleanup(setattr, server, "lan_addresses", real)
        status, info = self.api("GET", "/api/info")
        self.assertEqual(status, 200)
        self.assertTrue(info["loopback"])
        # the full addresses for a phone, token included, for the "用手机访问" panel
        self.assertEqual(info["lan_urls"], ["http://192.168.1.20:%d/?token=%s" % (self.port, TOKEN),
                                            "http://10.0.0.7:%d/?token=%s" % (self.port, TOKEN)])
        self.assertEqual(self.request("GET", "/")[0], 200)
        self.assertEqual(self.api("POST", "/api/jobs", {})[0], 201)

    def test_remote_client_must_present_the_token(self):
        job = self.create()
        self.remote()
        # nothing at all is served without it: not the API, not the page, not the assets
        for method, path in (("GET", "/api/jobs"), ("GET", "/api/info"), ("POST", "/api/jobs"),
                             ("GET", "/api/jobs/" + job), ("DELETE", "/api/jobs/" + job),
                             ("PUT", "/api/jobs/%s/files/a.jpg" % job),
                             ("GET", "/static/app.js"), ("GET", "/static/style.css"),
                             ("POST", "/"), ("GET", "/api/jobs?token=wrong"),
                             ("GET", "/api/jobs?token="), ("GET", "/static/app.js?token=nope")):
            status, _headers, data = self.request(method, path, b"x" if method == "PUT" else None)
            self.assertEqual(status, 401, "%s %s" % (method, path))
            self.assertNotIn(TOKEN.encode(), data)
        status, headers, data = self.request("GET", "/")
        self.assertEqual(status, 401)
        self.assertIn("text/html", headers["content-type"])
        self.assertIn("配对", data.decode("utf-8"))
        self.assertNotIn("set-cookie", headers)
        self.assertTrue((self.runs / job).is_dir())
        self.assertEqual(list((self.runs / job / "upload").iterdir()), [])

        self.app.auth_failures.clear()   # this test is about refusal, not about the limit
        for wrong in (TOKEN[:-1], TOKEN + "x", TOKEN.upper(), "", "None"):
            self.assertEqual(self.api("GET", "/api/jobs", None, {"X-Scan3D-Token": wrong})[0], 401)
            self.assertEqual(self.api("GET", "/api/jobs", None,
                                      {"Cookie": "scan3d_token=" + wrong})[0], 401)

    def test_token_by_query_cookie_and_header(self):
        self.remote()
        # the pairing link: one redirect that sets the cookie and drops the token from the URL
        status, headers, _ = self.request("GET", "/?token=" + TOKEN)
        self.assertEqual(status, 302)
        self.assertEqual(headers["location"], "/")
        cookie = headers["set-cookie"]
        self.assertIn("scan3d_token=" + TOKEN, cookie)
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=Lax", cookie)
        self.assertIn("Max-Age=31536000", cookie)     # the phone stays paired across restarts
        # a link to one job's viewer keeps its path; the fragment never reaches the server
        status, headers, _ = self.request("GET", "/?token=%s&x=1" % TOKEN)
        self.assertEqual((status, headers["location"]), (302, "/?x=1"))
        cookie_header = {"Cookie": cookie.split(";")[0]}

        status, headers, body = self.request("GET", "/", None, cookie_header)
        self.assertEqual(status, 200)
        self.assertIn("三维扫描", body.decode("utf-8"))
        self.assertEqual(self.request("GET", "/static/app.js", None, cookie_header)[0], 200)
        self.assertEqual(self.request("GET", "/static/app.js", None,
                                      {"Cookie": "other=1; scan3d_token=%s; x=y" % TOKEN})[0], 200)

        status, info = self.api("GET", "/api/info", None, cookie_header)
        self.assertEqual(status, 200)
        self.assertFalse(info["loopback"])
        self.assertNotIn("lan_urls", info)
        # ... and nothing a remote client is sent contains the token or other devices' URLs
        real = server.lan_addresses
        server.lan_addresses = lambda: ["192.168.1.20"]
        self.addCleanup(setattr, server, "lan_addresses", real)
        for path in ("/api/info", "/api/jobs", "/", "/static/app.js", "/static/jobview.js"):
            status, _headers, body = self.request("GET", path, None, cookie_header)
            self.assertEqual(status, 200, path)
            self.assertNotIn(TOKEN.encode(), body, path)
            self.assertNotIn(b"192.168.1.20", body, path)

        # an app would rather send a header; a query token also works on API calls
        header = {"X-Scan3D-Token": TOKEN}
        status, created = self.api("POST", "/api/jobs", {"name": "from-phone"}, header)
        self.assertEqual(status, 201)
        self.assertNotIn("input_path", created)
        self.assertNotIn("open_folder", created["actions"])
        self.assertEqual(self.api("PUT", "/api/jobs/from-phone/files/a.jpg", b"jpeg", header)[0], 201)
        status, headers, _ = self.request("GET", "/api/jobs?token=" + TOKEN)
        self.assertEqual(status, 200)
        self.assertIn("scan3d_token=" + TOKEN, headers["set-cookie"])

        # a stale cookie from an earlier session does not get in the way of a good token
        stale = {"Cookie": "scan3d_token=oldoldoldold", "X-Scan3D-Token": TOKEN}
        self.assertEqual(self.api("GET", "/api/jobs", None, stale)[0], 200)

    def test_loopback_only_actions_stay_local_even_with_token(self):
        folder = self.tmp / "photos"
        folder.mkdir()
        (folder / "a.jpg").write_bytes(b"x")
        job = self.create_with_photo(step=0.01)
        self.start(job)
        self.wait_state(job, "done")
        viewer = self.tmp / "tools" / "spirula" / "spirula.exe"
        viewer.parent.mkdir(parents=True)
        viewer.write_bytes(b"")

        self.remote()
        header = {"X-Scan3D-Token": TOKEN}
        status, error = self.api("POST", "/api/jobs", {"path": str(folder)}, header)
        self.assertEqual((status, error["error"]), (403, "loopback_only"))
        for action in ("open-folder", "open-viewer"):
            status, error = self.api("POST", "/api/jobs/%s/%s" % (job, action), {}, header)
            self.assertEqual((status, error["error"]), (403, "loopback_only"), action)
        self.assertEqual(self.launched, [])
        self.assertEqual([p.name for p in self.runs.iterdir() if p.is_dir()], [job])

        # ... while the ordinary things work for the phone
        status, view = self.api("GET", "/api/jobs/" + job, None, header)
        self.assertEqual(status, 200)
        self.assertNotIn("open_viewer", view["actions"])
        self.assertNotIn("open_folder", view["actions"])
        status, _headers, body = self.request(
            "GET", "/api/jobs/%s/download/output/splat.ply" % job, None, header)
        self.assertEqual((status, len(body)), (200, 4096))

    def test_wrong_tokens_are_limited_but_the_right_one_always_works(self):
        self.remote()
        statuses = [self.api("GET", "/api/jobs", None, {"X-Scan3D-Token": "guess%d" % i})[0]
                    for i in range(server.AUTH_MAX_FAILURES + 3)]
        self.assertEqual(statuses[:server.AUTH_MAX_FAILURES], [401] * server.AUTH_MAX_FAILURES)
        self.assertEqual(statuses[server.AUTH_MAX_FAILURES:], [429] * 3)
        self.assertEqual(self.api("GET", "/api/jobs", None, {"X-Scan3D-Token": TOKEN})[0], 200)
        self.assertEqual(self.api("GET", "/api/jobs")[0], 401)   # a missing token is not a guess


class SingleModeTests(ServerCase):
    def test_there_is_one_mode_and_it_listens_everywhere(self):
        self.assertEqual(server.BIND_HOST, "0.0.0.0")
        serve = inspect.signature(server.serve)
        self.assertEqual(list(serve.parameters), ["port", "open_browser"])
        self.assertEqual(serve.parameters["port"].default, 8765)
        self.assertEqual(serve.parameters["open_browser"].kind, inspect.Parameter.KEYWORD_ONLY)
        create = inspect.signature(server.create_server)
        self.assertEqual(create.parameters["host"].default, "0.0.0.0")
        for function in (server.serve, server.create_server, server.App.__init__):
            self.assertNotIn("lan", inspect.signature(function).parameters)
        self.assertFalse(hasattr(self.app, "lan"))
        self.assertFalse(hasattr(server, "resolve_bind_host"))

    def test_a_remote_client_is_asked_for_the_token_not_refused_outright(self):
        self.remote()
        for path in ("/", "/api/jobs", "/static/app.js", "/api/jobs?token=", "/?token=None"):
            self.assertEqual(self.request("GET", path)[0], 401, path)
        self.assertEqual(self.api("POST", "/api/jobs", {}, {"X-Scan3D-Token": ""})[0], 401)
        self.assertEqual(self.api("POST", "/api/jobs", {}, {"X-Scan3D-Token": TOKEN})[0], 201)

    def test_command_line_has_no_lan_switch(self):
        script = str(APP_DIR / "scan3d.py")
        refused = subprocess.run([sys.executable, script, "serve", "--lan"],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        self.assertEqual(refused.returncode, 2)             # argparse: unrecognized argument
        self.assertIn(b"--lan", refused.stderr)
        usage = subprocess.run([sys.executable, script, "serve", "--help"],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        self.assertEqual(usage.returncode, 0)
        self.assertIn(b"--port", usage.stdout)
        self.assertNotIn(b"--lan", usage.stdout)


# =========================================================================== browser previews

SLOW_CONVERTER = ("import sys, time; open(sys.argv[2] + '.started', 'w').close(); "
                  "time.sleep(float(sys.argv[1]))")


class PreviewTests(ServerCase):
    def use_slow_converter(self, seconds):
        """Replace the conversion by a process that only sleeps (and says that it started)."""
        real = server.build_preview_command
        server.build_preview_command = lambda source, out, limit: [
            sys.executable, "-c", SLOW_CONVERTER, str(seconds), str(out)]
        self.addCleanup(setattr, server, "build_preview_command", real)

    def test_command(self):
        cmd = server.build_preview_command(Path("a") / "splat.ply", Path("b") / "preview.spz", None)
        self.assertEqual(cmd, [sys.executable, str(APP_DIR / "splatconv.py"),
                               str(Path("a") / "splat.ply"), str(Path("b") / "preview.spz")])
        cmd = server.build_preview_command("in.ply", "out.spz", 300000)
        self.assertEqual(cmd[-2:], ["--max", "300000"])
        self.assertEqual(server.PREVIEW_VARIANTS, {"full": ("preview.spz", None),
                                                   "lite": ("preview-lite.spz", 300000)})

    def test_missing_preview_is_built_on_demand_and_cached_in_the_job_folder(self):
        out = self.make_done_job("old-job", gaussians=40)
        status, body = self.api("GET", "/api/jobs/old-job/preview")
        self.assertEqual(status, 200)
        self.assertEqual({k: v["state"] for k, v in body["variants"].items()},
                         {"full": "missing", "lite": "missing"})
        self.assertEqual(body["variants"]["lite"]["max_gaussians"], 300000)
        self.assertIsNone(body["variants"]["full"]["max_gaussians"])
        # looking does not build anything, and there is nothing to download yet
        status, error = self.api("GET", "/api/jobs/old-job/preview/full")
        self.assertEqual((status, error["error"]), (404, "preview_missing"))
        self.assertFalse((self.runs / "old-job" / "web-preview").exists())

        real = server.PREVIEW_VARIANTS
        server.PREVIEW_VARIANTS = dict(real, lite=("preview-lite.spz", 15))
        self.addCleanup(setattr, server, "PREVIEW_VARIANTS", real)
        began = time.monotonic()
        status, state = self.api("POST", "/api/jobs/old-job/preview/full", {})
        self.assertLess(time.monotonic() - began, 5)         # answered at once
        self.assertIn((status, state["state"]), ((202, "queued"), (202, "building"), (200, "ready")))
        status, state = self.api("POST", "/api/jobs/old-job/preview/lite")
        self.assertIn(status, (200, 202))
        full = self.wait_preview("old-job", "full")
        lite = self.wait_preview("old-job", "lite")
        self.assertEqual((full["origin"], lite["origin"]), ("cache", "cache"))
        cache = self.runs / "old-job" / "web-preview"
        self.assertEqual(sorted(p.name for p in cache.iterdir()),
                         ["preview-lite.spz", "preview-lite.spz.json",
                          "preview.spz", "preview.spz.json"])
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["splat.ply"])   # output/ untouched
        self.assertEqual(full["size"], (cache / "preview.spz").stat().st_size)

        import splatconv
        self.assertEqual(splatconv.read_preview(cache / "preview.spz")["count"], 40)
        decoded = splatconv.read_preview(cache / "preview-lite.spz")
        self.assertEqual(decoded["count"], 15)
        self.assertAlmostEqual(decoded["positions"][14][0], 0.14, delta=1e-4)   # the first 15

        # the file itself: whole, by Range, HEAD; asking again builds nothing new
        url = "/api/jobs/old-job/preview/full"
        status, headers, body = self.request("GET", url)
        self.assertEqual((status, body), (200, (cache / "preview.spz").read_bytes()))
        self.assertEqual(body[:2], b"\x1f\x8b")
        self.assertEqual(headers["content-type"], "application/octet-stream")
        self.assertEqual(headers["accept-ranges"], "bytes")
        self.assertNotIn("content-disposition", headers)
        status, headers, part = self.request("GET", url, headers={"Range": "bytes=2-9"})
        self.assertEqual((status, part), (206, body[2:10]))
        status, headers, nothing = self.request("HEAD", url)
        self.assertEqual((status, nothing, int(headers["content-length"])), (200, b"", len(body)))
        status, _headers, nothing = self.request("GET", url, headers={"If-None-Match": headers["etag"]})
        self.assertEqual((status, nothing), (304, b""))
        mtime = (cache / "preview.spz").stat().st_mtime_ns
        status, state = self.api("POST", url)
        self.assertEqual((status, state["state"]), (200, "ready"))
        time.sleep(0.3)
        self.assertEqual((cache / "preview.spz").stat().st_mtime_ns, mtime)

        # the job is run again and its splat changes: the cached preview no longer counts
        write_splat_ply(out / "splat.ply", 25)
        self.assertEqual(self.preview("old-job", "full")["state"], "missing")
        self.assertEqual(self.api("GET", url)[0], 404)
        self.api("POST", url)
        self.wait_preview("old-job", "full")
        self.assertEqual(splatconv.read_preview(cache / "preview.spz")["count"], 25)

    def test_previews_written_by_the_pipeline_are_used_as_they_are(self):
        import splatconv
        # the report now lists the two previews among the outputs
        out = self.make_done_job("new-job", preview="output/preview.spz",
                                 preview_lite="output/preview-lite.spz", report="output/report.json")
        (out / "report.json").write_text("{}", encoding="utf-8")
        # exactly the pipeline's two calls
        splatconv.write_preview(out / "splat.ply", out / splatconv.PREVIEW_NAME)
        splatconv.write_preview(out / "splat.ply", out / splatconv.PREVIEW_LITE_NAME,
                                max_gaussians=splatconv.LITE_MAX_GAUSSIANS)
        for variant, name, degree, bits in (("full", "preview.spz", 3, 8),
                                            ("lite", "preview-lite.spz", 0, None)):
            content = (out / name).read_bytes()
            state = self.preview("new-job", variant)
            self.assertEqual((state["state"], state["origin"], state["size"]),
                             ("ready", "output", len(content)))
            self.assertEqual((state["sh_degree"], state["sh_bits"], state["wanted_sh_degree"],
                              state["gaussians"]), (degree, bits, degree, 40))
            self.assertNotIn("output_outdated", state)
            status, _headers, body = self.request("GET", "/api/jobs/new-job/preview/" + variant)
            self.assertEqual((status, body), (200, content))
            status, state = self.api("POST", "/api/jobs/new-job/preview/" + variant)
            self.assertEqual((status, state["state"]), (200, "ready"))
        self.assertFalse((self.runs / "new-job" / "web-preview").exists())
        self.assertIsNone(self.app.previews._thread)             # nothing was ever converted
        # they also show up as ordinary output files, each once, under the report's keys
        files = self.job("new-job")["output_files"]
        self.assertEqual(sorted((f["key"], f["name"]) for f in files),
                         [("preview", "preview.spz"), ("preview_lite", "preview-lite.spz"),
                          ("report", "report.json"), ("splat", "splat.ply")])
        self.assertFalse(any("outdated" in f for f in files))

    def test_the_splat_itself_is_described_for_the_original_quality_view(self):
        out = self.make_done_job("orig", gaussians=40)
        status, body = self.api("GET", "/api/jobs/orig/preview")
        size = (out / "splat.ply").stat().st_size
        self.assertEqual(body["source"], {"name": "splat.ply", "path": "output/splat.ply",
                                          "size": size, "gaussians": 40, "sh_degree": 3})
        # ... and can be fetched whole or in parts from the download route the page uses
        url = "/api/jobs/orig/download/" + body["source"]["path"]
        status, headers, data = self.request("GET", url)
        self.assertEqual((status, data, int(headers["content-length"])),
                         (200, (out / "splat.ply").read_bytes(), size))
        status, headers, part = self.request("GET", url, headers={"Range": "bytes=0-2"})
        self.assertEqual((status, part), (206, b"ply"))
        # a splat whose header cannot be read: still listed, its numbers unknown
        (out / "splat.ply").write_bytes(b"not a ply")
        status, body = self.api("GET", "/api/jobs/orig/preview")
        self.assertEqual(body["source"], {"name": "splat.ply", "path": "output/splat.ply",
                                          "size": 9, "gaussians": None, "sh_degree": None})
        self.assertEqual(body["variants"]["full"]["wanted_sh_degree"], 0)
        # no splat, or not finished: nothing to describe
        (out / "splat.ply").unlink()
        self.assertIsNone(self.api("GET", "/api/jobs/orig/preview")[1]["source"])
        self.create(name="fresh2")
        self.assertIsNone(self.api("GET", "/api/jobs/fresh2/preview")[1]["source"])

    def test_an_old_preview_without_view_dependent_colour_is_replaced_not_shown(self):
        """A job from before previews kept SH has a degree-0 preview.spz in output/: it is
        not the "full" variant any more."""
        import splatconv
        out = self.make_done_job("old-sh0")
        splatconv.write_preview(out / "splat.ply", out / "preview.spz", sh_degree=0)
        splatconv.write_preview(out / "splat.ply", out / "preview-lite.spz", max_gaussians=300000)
        old = (out / "preview.spz").read_bytes()
        state = self.preview("old-sh0", "full")
        self.assertEqual((state["state"], state["wanted_sh_degree"], state.get("output_outdated")),
                         ("missing", 3, True))
        self.assertNotIn("origin", state)
        self.assertEqual(self.api("GET", "/api/jobs/old-sh0/preview/full")[0], 404)
        lite = self.preview("old-sh0", "lite")          # the lite variant never had SH
        self.assertEqual((lite["state"], lite["origin"], lite["sh_degree"], lite["wanted_sh_degree"]),
                         ("ready", "output", 0, 0))
        self.assertNotIn("output_outdated", lite)
        flags = {f["name"]: f.get("outdated", False) for f in self.job("old-sh0")["output_files"]}
        self.assertEqual(flags, {"splat.ply": False, "preview.spz": True, "preview-lite.spz": False})

        self.assertEqual(self.api("POST", "/api/jobs/old-sh0/preview/full")[0], 202)
        state = self.wait_preview("old-sh0", "full")
        self.assertEqual((state["origin"], state["sh_degree"], state["sh_bits"], state["gaussians"],
                          state.get("output_outdated")), ("cache", 3, 8, 40, True))
        cache = self.runs / "old-sh0" / "web-preview"
        self.assertEqual((out / "preview.spz").read_bytes(), old)        # output/ is not rewritten
        self.assertEqual(splatconv.read_preview_header(cache / "preview.spz")["sh_degree"], 3)
        noted = json.loads((cache / "preview.spz.json").read_text(encoding="utf-8"))
        self.assertEqual((noted["sh_degree"], noted["sh_bits"], noted["gaussians"]), (3, 8, 40))
        status, _headers, body = self.request("GET", "/api/jobs/old-sh0/preview/full")
        self.assertEqual((status, body), (200, (cache / "preview.spz").read_bytes()))

        # view-dependent colour rounded as coarsely as SPZ tools usually do: also not "full"
        splatconv.write_preview(out / "splat.ply", out / "preview.spz", sh_bits=5)
        shutil.rmtree(cache)
        state = self.preview("old-sh0", "full")
        self.assertEqual((state["state"], state.get("output_outdated")), ("missing", True))
        # something that is not an SPZ file at all
        (out / "preview.spz").write_bytes(b"\x1f\x8b pipeline full")
        state = self.preview("old-sh0", "full")
        self.assertEqual((state["state"], state.get("output_outdated")), ("missing", True))
        # the current format
        splatconv.write_preview(out / "splat.ply", out / "preview.spz")
        state = self.preview("old-sh0", "full")
        self.assertEqual((state["state"], state["origin"], state["sh_degree"], state["sh_bits"]),
                         ("ready", "output", 3, 8))
        self.assertNotIn("output_outdated", state)
        self.assertFalse(any("outdated" in f for f in self.job("old-sh0")["output_files"]))

    def test_a_preview_built_by_an_older_server_is_rebuilt(self):
        """web-preview/ of a job viewed before previews kept SH: the note beside the file does
        not say what it holds."""
        import splatconv
        out = self.make_done_job("old-cache")
        cache = self.runs / "old-cache" / "web-preview"
        cache.mkdir()
        info = (out / "splat.ply").stat()
        made_from = {"source_size": info.st_size, "source_mtime_ns": info.st_mtime_ns}
        for name, limit in (("preview.spz", None), ("preview-lite.spz", 300000)):
            splatconv.write_preview(out / "splat.ply", cache / name, max_gaussians=limit, sh_degree=0)
            (cache / (name + ".json")).write_text(json.dumps(made_from), encoding="utf-8")
        self.assertEqual(self.preview("old-cache", "full")["state"], "missing")
        self.assertNotIn("output_outdated", self.preview("old-cache", "full"))
        lite = self.preview("old-cache", "lite")
        self.assertEqual((lite["state"], lite["origin"], lite["sh_degree"]), ("ready", "cache", 0))
        self.api("POST", "/api/jobs/old-cache/preview/full")
        state = self.wait_preview("old-cache", "full")
        self.assertEqual((state["origin"], state["sh_degree"], state["sh_bits"]), ("cache", 3, 8))

    def test_a_splat_without_sh_bands_wants_none(self):
        import splatconv
        out = self.make_done_job("flat")
        properties = [name for name in SPLAT_PROPERTIES if not name.startswith("f_rest_")]
        header = "ply\nformat binary_little_endian 1.0\nelement vertex 3\n"
        header += "".join("property float %s\n" % name for name in properties) + "end_header\n"
        one = struct.pack("<17f", 0.1, 0.2, 0.3, 0, 0, 0, 0.5, 0.5, 0.5, 2.0, -3, -3, -3, 1, 0, 0, 0)
        (out / "splat.ply").write_bytes(header.encode("ascii") + one * 3)
        splatconv.write_preview(out / "splat.ply", out / "preview.spz")
        state = self.preview("flat", "full")
        self.assertEqual((state["state"], state["origin"], state["sh_degree"], state["sh_bits"],
                          state["wanted_sh_degree"]), ("ready", "output", 0, None, 0))
        self.assertEqual(self.api("GET", "/api/jobs/flat/preview")[1]["source"]["sh_degree"], 0)

    def test_a_failed_conversion_is_reported_and_can_be_retried(self):
        out = self.make_done_job("broken")
        (out / "splat.ply").write_bytes(b"this is not a ply file at all")
        status, state = self.api("POST", "/api/jobs/broken/preview/full")
        self.assertEqual(status, 202)
        state = self.wait_preview("broken", "full", ("failed",))
        self.assertIn("无法生成预览", state["error"])
        self.assertIn("PLY", state["error"])
        self.assertEqual(self.api("GET", "/api/jobs/broken/preview/full")[0], 404)
        self.assertEqual(self.preview("broken", "lite")["state"], "missing")   # its own state
        # nothing half-written is left, and a repaired file converts on the next request
        cache = self.runs / "broken" / "web-preview"
        self.assertEqual(list(cache.iterdir()), [])
        write_splat_ply(out / "splat.ply", 5)
        status, state = self.api("POST", "/api/jobs/broken/preview/full")
        self.assertEqual(status, 202)
        self.assertEqual(self.wait_preview("broken", "full")["origin"], "cache")

    def test_one_conversion_at_a_time_and_requests_never_wait_for_it(self):
        self.use_slow_converter(60)
        self.make_done_job("job-a")
        self.make_done_job("job-b")
        began = time.monotonic()
        status, state = self.api("POST", "/api/jobs/job-a/preview/full")
        self.assertEqual(status, 202)
        started = self.runs / "job-a" / "web-preview" / "preview.spz.started"
        wait_until(started.exists, 20, message="the first conversion to start")
        self.assertEqual(self.preview("job-a", "full")["state"], "building")
        for job, variant in (("job-a", "lite"), ("job-b", "full"), ("job-b", "lite")):
            status, state = self.api("POST", "/api/jobs/%s/preview/%s" % (job, variant))
            self.assertEqual((status, state["state"]), (202, "queued"), (job, variant))
        # asking again changes nothing: still one running, three waiting, each listed once
        for _ in range(3):
            self.assertEqual(self.api("POST", "/api/jobs/job-a/preview/full")[1]["state"], "building")
            self.assertEqual(self.api("POST", "/api/jobs/job-b/preview/lite")[1]["state"], "queued")
        self.assertEqual(self.app.previews.current, ("job-a", "full"))
        self.assertEqual(list(self.app.previews.pending),
                         [("job-a", "lite"), ("job-b", "full"), ("job-b", "lite")])
        time.sleep(0.5)
        self.assertFalse((self.runs / "job-b" / "web-preview").exists())     # nobody jumped the queue
        # every request above was answered while the conversion was still running
        self.assertLess(time.monotonic() - began, 30)
        self.assertEqual(self.api("GET", "/api/jobs")[0], 200)
        child = self.app.previews.child
        self.assertTrue(server.pid_alive(child.pid))

        # deleting a job stops its conversion and drops what it had queued
        began = time.monotonic()
        self.assertEqual(self.api("DELETE", "/api/jobs/job-a")[0], 200)
        self.assertLess(time.monotonic() - began, 15)
        self.assertFalse((self.runs / "job-a").exists())
        wait_until(lambda: not server.pid_alive(child.pid), 10, message="the conversion to die")
        self.assertNotIn(("job-a", "lite"), self.app.previews.pending)
        # ... and the next one in the queue gets its turn
        wait_until((self.runs / "job-b" / "web-preview" / "preview.spz.started").exists, 20,
                   message="the next conversion to start")
        self.assertEqual(self.preview("job-b", "full")["state"], "building")

    def test_a_conversion_that_hangs_is_stopped_by_the_time_limit(self):
        self.use_slow_converter(600)
        self.app.previews.time_limit = 1.0
        self.make_done_job("stuck")
        self.assertEqual(self.api("POST", "/api/jobs/stuck/preview/lite")[0], 202)
        began = time.monotonic()
        state = self.wait_preview("stuck", "lite", ("failed",), timeout=30)
        self.assertLess(time.monotonic() - began, 20)
        self.assertIn("已被停止", state["error"])
        wait_until(lambda: self.app.previews.child is None, 10, message="the child to be reaped")
        self.assertIsNone(self.app.previews.current)

    def test_shutdown_stops_a_running_conversion(self):
        self.use_slow_converter(600)
        self.make_done_job("closing")
        self.api("POST", "/api/jobs/closing/preview/full")
        self.api("POST", "/api/jobs/closing/preview/lite")
        wait_until(lambda: self.app.previews.child is not None, 20, message="the conversion to start")
        pid = self.app.previews.child.pid
        self.assertTrue(server.pid_alive(pid))
        began = time.monotonic()
        self.httpd.shutdown()
        self.app.shutdown()
        self.assertLess(time.monotonic() - began, 20)
        wait_until(lambda: not server.pid_alive(pid), 10, message="the conversion to die")
        self.assertFalse(self.app.previews._thread.is_alive())
        self.assertEqual(list(self.app.previews.pending), [])

    def test_only_finished_jobs_with_a_splat(self):
        job = self.create(name="fresh")                       # state created
        status, body = self.api("GET", "/api/jobs/fresh/preview")
        self.assertEqual(body["variants"]["full"]["state"], "no_splat")
        status, error = self.api("POST", "/api/jobs/fresh/preview/full")
        self.assertEqual((status, error["error"]), (409, "not_done"))
        self.assertEqual(self.api("GET", "/api/jobs/fresh/preview/full")[0], 404)

        out = self.make_done_job("meshless")
        (out / "splat.ply").unlink()
        self.assertEqual(self.preview("meshless", "lite")["state"], "no_splat")
        status, error = self.api("POST", "/api/jobs/meshless/preview/lite")
        self.assertEqual((status, error["error"]), (409, "no_splat"))

        self.assertEqual(self.api("GET", "/api/jobs/nobody/preview")[0], 404)
        self.assertEqual(self.api("POST", "/api/jobs/nobody/preview/full")[0], 404)
        self.assertEqual(self.api("GET", "/api/jobs/nobody/preview/full")[0], 404)
        self.assertEqual(self.api("GET", "/api/jobs/nobody/cameras")[0], 404)
        for variant in ("huge", "FULL", "full.spz", "preview.spz", ""):
            self.assertEqual(self.api("POST", "/api/jobs/meshless/preview/" + variant)[0],
                             404 if variant else 405, variant)
        for method, path in (("DELETE", "/api/jobs/meshless/preview/full"),
                             ("PUT", "/api/jobs/meshless/preview/full"),
                             ("POST", "/api/jobs/meshless/preview"),
                             ("POST", "/api/jobs/meshless/cameras"),
                             ("DELETE", "/api/jobs/meshless/cameras")):
            status, headers, _ = self.request(method, path, b"x" if method == "PUT" else None)
            self.assertEqual(status, 405, "%s %s" % (method, path))
        self.assertIsNone(self.app.previews._thread)

    def test_phone_needs_the_token_and_then_may_view(self):
        self.make_done_job("shared")
        self.remote()
        for method, path in (("GET", "/api/jobs/shared/preview"), ("POST", "/api/jobs/shared/preview/lite"),
                             ("GET", "/api/jobs/shared/preview/lite"), ("GET", "/api/jobs/shared/cameras"),
                             ("GET", "/static/viewer3d.js"), ("GET", "/static/jobview.js"),
                             ("GET", "/static/vendor/spark/spark.module.js"),
                             ("GET", "/static/vendor/three/three.module.js"),
                             ("GET", "/static/vendor/qrcode-generator/qrcode.js")):
            self.assertEqual(self.request(method, path)[0], 401, "%s %s" % (method, path))
        self.assertIsNone(self.app.previews._thread)
        header = {"X-Scan3D-Token": TOKEN}
        self.assertEqual(self.api("POST", "/api/jobs/shared/preview/lite", None, header)[0], 202)
        wait_until(lambda: self.api("GET", "/api/jobs/shared/preview", None, header)[1]
                   ["variants"]["lite"]["state"] == "ready", 60, message="the preview")
        status, _headers, body = self.request("GET", "/api/jobs/shared/preview/lite", None, header)
        self.assertEqual((status, body[:2]), (200, b"\x1f\x8b"))
        self.assertEqual(self.api("GET", "/api/jobs/shared/cameras", None, header)[0], 200)
        self.assertEqual(self.request("GET", "/static/vendor/spark/spark.module.js", None, header)[0], 200)
        # a page on another site cannot start conversions with the phone's cookie
        cookie = {"Cookie": "scan3d_token=" + TOKEN, "Origin": "http://evil.example"}
        status, error = self.api("POST", "/api/jobs/shared/preview/full", None, cookie)
        self.assertEqual((status, error["error"]), (403, "bad_origin"))


def write_images_bin(path: Path, poses) -> None:
    """A COLMAP images.bin with one image per (qw, qx, qy, qz, tx, ty, tz)."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(poses)))
        for index, pose in enumerate(poses):
            f.write(struct.pack("<I7dI", index + 1, *pose, 1))
            f.write(("image_%04d.jpg" % index).encode("ascii") + b"\x00")
            f.write(struct.pack("<Q", 2))
            f.write(struct.pack("<ddq", 10.0, 20.0, -1) * 2)        # two 2D points


class CameraTests(ServerCase):
    def test_camera_poses_in_splat_coordinates(self):
        out = self.make_done_job("scene")
        self.assertEqual(self.api("GET", "/api/jobs/scene/cameras"),
                         (200, {"count": 0, "cameras": []}))         # no dataset: none, no error
        sparse = self.runs / "scene" / "dataset" / "sparse"
        sparse.mkdir(parents=True)
        half = 0.5 ** 0.5
        write_images_bin(sparse / "images.bin", [
            (1, 0, 0, 0, 0, 0, -5),              # looking along +Z from (0, 0, 5)
            (half, 0, half, 0, 1, 2, 3),         # turned 90 degrees about Y
        ])
        status, body = self.api("GET", "/api/jobs/scene/cameras")
        self.assertEqual((status, body["count"]), (200, 2))
        first, second = body["cameras"]
        self.assertEqual(first, {"position": [0.0, 0.0, 5.0], "forward": [0.0, 0.0, 1.0],
                                 "up": [0.0, -1.0, 0.0]})
        # R = Ry(90): rows (0,0,1), (0,1,0), (-1,0,0); centre = -R^T t = (3, -2, -1)
        for got, want in ((second["position"], [3, -2, -1]), (second["forward"], [-1, 0, 0]),
                          (second["up"], [0, -1, 0])):
            for a, b in zip(got, want):
                self.assertAlmostEqual(a, b, places=5)

        # the training frame differs from the dataset's: p' = s R p + t (scene_transform.json)
        transform = {"train_from_world": {"scale": 2, "translation": [1, 0, 0], "rotation": {
            "matrix_3x3": [[0, -1, 0], [1, 0, 0], [0, 0, 1]]}}}           # 90 degrees about Z
        (out / "scene_transform.json").write_text(json.dumps(transform), encoding="utf-8")
        first = self.api("GET", "/api/jobs/scene/cameras")[1]["cameras"][0]
        self.assertEqual(first, {"position": [1.0, 0.0, 10.0], "forward": [0.0, 0.0, 1.0],
                                 "up": [1.0, 0.0, 0.0]})

        # an unusable transform or dataset gives no cameras rather than wrong ones or an error
        (out / "scene_transform.json").write_text('{"train_from_world": {"rotation": 5}}', encoding="utf-8")
        self.assertEqual(self.api("GET", "/api/jobs/scene/cameras")[1], {"count": 0, "cameras": []})
        (out / "scene_transform.json").unlink()
        for junk in (b"", b"\x01", struct.pack("<Q", 5), struct.pack("<Q", 2 ** 60) + b"x" * 100,
                     struct.pack("<Q", 1) + b"\xff" * 64 + b"no terminator " * 400):
            (sparse / "images.bin").write_bytes(junk)
            self.assertEqual(self.api("GET", "/api/jobs/scene/cameras"),
                             (200, {"count": 0, "cameras": []}), junk[:20])

    def test_many_cameras_are_thinned_out_and_the_other_layout_is_found(self):
        self.make_done_job("walk")
        sparse = self.runs / "walk" / "dataset" / "sparse" / "0"       # the newer folder layout
        sparse.mkdir(parents=True)
        write_images_bin(sparse / "images.bin", [(1, 0, 0, 0, -i, 0, 0) for i in range(1001)])
        status, body = self.api("GET", "/api/jobs/walk/cameras")
        self.assertEqual((status, body["count"]), (200, 1001))
        xs = [camera["position"][0] for camera in body["cameras"]]
        self.assertLessEqual(len(xs), server.MAX_VIEWER_CAMERAS)
        self.assertGreater(len(xs), 100)
        self.assertEqual(xs[:3], [0.0, 6.0, 12.0])                     # evenly spread, in order
        self.assertEqual(xs, sorted(xs))


# =========================================================================== models and licences

class ModelsTests(ServerCase):
    PHONE = {"X-Scan3D-Token": TOKEN}

    def download(self, body=None, headers=None):
        return self.api("POST", "/api/models/download", {} if body is None else body, headers)

    def accept(self, ids, headers=None):
        return self.api("POST", "/api/models/accept", {"licences": ids}, headers)

    def test_overview(self):
        self.models.install("moge.bin")
        view = self.models_view()
        self.assertEqual(set(view), {"features", "licences", "files_missing", "download_mb",
                                     "download", "can_manage"})
        self.assertTrue(view["can_manage"])
        self.assertEqual(list(view["features"]),
                         ["masks", "subject_only", "normals", "dense_seed", "sfm"])
        self.assertEqual({k: v["label"] for k, v in view["features"].items()}, {
            "masks": "抠除移动物体", "subject_only": "物体扫描：只保留主体",
            "normals": "表面法线先验", "dense_seed": "稠密点云初始化",
            "sfm": "位姿计算的备用特征"})
        self.assertEqual(view["features"]["masks"], {
            "label": "抠除移动物体", "files": ["sam3.bin"], "enabled": True, "ready": False,
            "licences_needed": ["sam3"], "files_missing": [], "download_mb": 6})
        self.assertEqual(view["features"]["normals"]["ready"], True)
        self.assertEqual(view["features"]["sfm"], {
            "label": "位姿计算的备用特征", "files": ["aliked.bin", "lightglue.bin"],
            "enabled": True, "ready": False, "licences_needed": [],
            "files_missing": ["aliked.bin", "lightglue.bin"], "download_mb": 2})
        self.assertEqual(view["features"]["dense_seed"]["enabled"], False)
        self.assertEqual(view["licences"]["sam3"], {
            "name": "SAM License (Meta)", "url": "https://example.invalid/sam3",
            "used_for": "抠除移动物体", "accepted": False, "needed": True})
        self.assertEqual(view["licences"]["dinov3"]["needed"], False)   # its feature is off
        self.assertEqual((view["files_missing"], view["download_mb"]),
                         (["aliked.bin", "lightglue.bin"], 2))
        self.assertEqual(view["download"], {"state": "idle", "features": [], "error": None,
                                            "downloaded_mb": 0, "total_mb": 0, "files": []})
        self.assertEqual(self.request("HEAD", "/api/models")[0], 200)
        self.assertEqual(self.request("DELETE", "/api/models")[0], 405)
        self.assertEqual(self.request("GET", "/api/models/accept")[0], 405)
        self.assertEqual(self.request("GET", "/api/models/download")[0], 405)
        self.assertEqual(self.request("GET", "/api/models/nope")[0], 404)

        # a phone sees the same states, and that it cannot manage them
        self.remote()
        self.assertEqual(self.api("GET", "/api/models")[0], 401)
        phone = self.models_view(self.PHONE)
        self.assertFalse(phone["can_manage"])
        self.assertEqual({k: v for k, v in phone.items() if k != "can_manage"},
                         {k: v for k, v in view.items() if k != "can_manage"})
        # looking never accepts anything
        self.assertEqual(self.models.accepted, set())

    def test_a_feature_the_server_has_no_name_for_keeps_its_key(self):
        real = server.models_status

        def with_new_feature():
            status = real()
            status["features"]["depth"] = dict(status["features"]["normals"])
            return status

        server.models_status = with_new_feature
        self.assertEqual(self.models_view()["features"]["depth"]["label"], "depth")

    def test_accepting_a_licence_is_an_explicit_request_from_this_pc(self):
        # a phone may not, token or not
        self.remote()
        status, error = self.accept(["sam3"], self.PHONE)
        self.assertEqual((status, error["error"]), (403, "loopback_only"))
        self.assertEqual(self.models.accepted, set())
        self.app.loopback_check = server.is_loopback_address

        # a page of another site may not either
        status, error = self.accept(["sam3"], {"Origin": "http://evil.example"})
        self.assertEqual((status, error["error"]), (403, "bad_origin"))

        # an unknown id: nothing is recorded, the known one beside it included
        for bad in (["nope"], ["birefnet", "nope"], ["../x"]):
            status, error = self.accept(bad)
            self.assertEqual((status, error["error"]), (400, "bad_licence"), bad)
        for bad in ({}, {"licences": []}, {"licences": "sam3"}, {"licences": [1]},
                    {"licences": [["sam3"]]}, {"licences": None}, {"licence": ["sam3"]}):
            status, error = self.api("POST", "/api/models/accept", bad)
            self.assertEqual((status, error["error"]), (400, "bad_licence"), bad)
        self.assertEqual(self.models.accepted, set())

        status, view = self.accept(["sam3"])
        self.assertEqual(status, 200, view)
        self.assertEqual(self.models.accepted, {"sam3"})
        self.assertTrue(view["can_manage"])
        self.assertEqual((view["licences"]["sam3"]["accepted"], view["licences"]["sam3"]["needed"]),
                         (True, False))
        self.assertFalse(view["licences"]["birefnet"]["accepted"])
        masks = view["features"]["masks"]
        self.assertEqual((masks["licences_needed"], masks["files_missing"], masks["ready"]),
                         ([], ["sam3.bin"], False))
        # accepting starts no download by itself
        self.assertEqual(view["download"]["state"], "idle")
        self.assertEqual(self.models.commands, [])

        status, view = self.accept(["birefnet", "dinov3"])
        self.assertEqual(status, 200)
        self.assertEqual(self.models.accepted, {"sam3", "birefnet", "dinov3"})

    def test_download_is_refused_for_phones_unknown_features_and_open_licences(self):
        self.remote()
        status, error = self.download({"features": ["normals"]}, self.PHONE)
        self.assertEqual((status, error["error"]), (403, "loopback_only"))
        self.app.loopback_check = server.is_loopback_address

        for bad in (["nope"], ["normals", "nope"], ["../normals"]):
            status, error = self.download({"features": bad})
            self.assertEqual((status, error["error"]), (400, "bad_feature"), bad)
        for bad in ("normals", [1], [["normals"]], {"normals": True}, 7):
            status, error = self.download({"features": bad})
            self.assertEqual((status, error["error"]), (400, "bad_feature"), bad)

        # a feature whose licence nobody accepted: said clearly, nothing fetched, nothing accepted
        for names in (["masks"], ["normals", "masks"], ["dense_seed", "subject_only"]):
            status, error = self.download({"features": names})
            self.assertEqual((status, error["error"]), (409, "licence_needed"), names)
        self.assertEqual(error["licences"], ["birefnet", "dinov3"])
        self.assertIn("同意许可", error["message"])
        self.assertIn("稠密点云初始化", error["message"])
        self.assertIn("物体扫描：只保留主体", error["message"])
        status, error = self.download({"features": ["normals", "masks"]})
        self.assertEqual(error["licences"], ["sam3"])
        self.assertIn("抠除移动物体", error["message"])
        self.assertNotIn("表面法线先验", error["message"])

        self.assertEqual(self.models.commands, [])
        self.assertEqual(self.models.accepted, set())
        self.assertEqual(self.models_view()["download"]["state"], "idle")
        self.assertFalse(self.models.folder.exists())

    def test_download_with_progress_and_one_at_a_time(self):
        self.models.accepted.add("sam3")
        self.models.folder.mkdir()
        gate = self.models.folder / "gate"
        gate.write_bytes(b"")
        status, view = self.download({"features": ["masks", "masks"]})
        self.assertEqual(status, 202, view)
        self.assertEqual(view["download"]["state"], "running")
        self.assertEqual(view["download"]["features"], ["masks"])
        self.assertEqual((view["download"]["total_mb"], view["download"]["error"]), (6, None))
        self.assertEqual(self.models.commands, [["masks"]])

        # the numbers come from the partial file
        def half_way():
            return self.models_view()["download"]["downloaded_mb"] == 3
        wait_until(half_way, 20, message="half of the file to arrive")
        view = self.models_view()
        self.assertEqual(view["download"]["state"], "running")
        self.assertEqual(view["download"]["files"],
                         [{"name": "sam3.bin", "size_mb": 6, "downloaded_mb": 3}])
        self.assertEqual((self.models.folder / "_part" / "sam3.bin").stat().st_size, 3 * MIB)
        self.assertFalse(view["features"]["masks"]["ready"])

        # one download at a time, whatever is asked for
        for body in ({"features": ["normals"]}, {"features": ["masks"]}, {}):
            status, error = self.download(body)
            self.assertEqual((status, error["error"]), (409, "download_busy"), body)
        self.assertEqual(self.models.commands, [["masks"]])

        # a phone watches the same progress
        self.remote()
        phone = self.models_view(self.PHONE)
        self.assertEqual((phone["can_manage"], phone["download"]["state"],
                          phone["download"]["downloaded_mb"]), (False, "running", 3))
        self.app.loopback_check = server.is_loopback_address

        gate.unlink()
        view = self.wait_download()
        # The stand-in exits with 1, as the real command does while another enabled feature
        # (here: normals, sfm) still lacks files. The download counts as done all the same.
        self.assertEqual(view["download"], {
            "state": "done", "features": ["masks"], "error": None, "downloaded_mb": 6,
            "total_mb": 6, "files": [{"name": "sam3.bin", "size_mb": 6, "downloaded_mb": 6}]})
        self.assertTrue(view["features"]["masks"]["ready"])
        self.assertEqual((self.models.folder / "sam3.bin").stat().st_size, 6 * MIB)
        self.assertFalse((self.models.folder / "_part" / "sam3.bin").exists())
        wait_until(lambda: self.app.models.child is None, 10, message="the child to be reaped")

        # nothing left for this feature: no second download
        status, view = self.download({"features": ["masks"]})
        self.assertEqual((status, view["download"]["state"]), (200, "done"))
        self.assertEqual(self.models.commands, [["masks"]])

    def test_download_without_names_fetches_what_the_enabled_features_may_have(self):
        self.models.exit_code = 0
        for body in ({}, {"features": []}, {"features": None}):
            for name in ("moge.bin", "aliked.bin", "lightglue.bin"):
                (self.models.folder / name).unlink(missing_ok=True)
            status, view = self.download(body)
            self.assertEqual(status, 202, (body, view))
            # masks is enabled too, but its licence is open: left alone, and not accepted
            self.assertEqual(view["download"]["features"], ["normals", "sfm"])
            self.assertEqual(view["download"]["total_mb"], 6)
            view = self.wait_download()
            self.assertEqual((view["download"]["state"], view["download"]["downloaded_mb"]),
                             ("done", 6), body)
        self.assertEqual(self.models.commands, [None, None, None])
        self.assertEqual(sorted(p.name for p in self.models.folder.iterdir() if p.is_file()),
                         ["aliked.bin", "lightglue.bin", "moge.bin"])
        self.assertEqual(self.models.accepted, set())
        self.assertEqual(view["features"]["masks"]["licences_needed"], ["sam3"])
        self.assertEqual(view["files_missing"], [])

        # nothing more to fetch: said with 200, and no process is started
        status, view = self.download({})
        self.assertEqual((status, view["download"]["state"]), (200, "done"))
        self.assertEqual(len(self.models.commands), 3)

        # features that are off by default can be prepared by name
        self.models.accepted.add("dinov3")
        status, view = self.download({"features": ["dense_seed", "normals"]})
        self.assertEqual(status, 202)
        self.assertEqual(view["download"]["features"], ["dense_seed"])   # normals has its file
        self.assertTrue(self.wait_download()["features"]["dense_seed"]["ready"])
        self.assertEqual(self.models.commands[-1], ["dense_seed"])

    def test_a_download_that_fails_says_why_and_can_be_retried(self):
        self.models.mode = "fail"
        status, view = self.download({"features": ["normals", "sfm"]})
        self.assertEqual(status, 202, view)
        view = self.wait_download()
        download = view["download"]
        self.assertEqual((download["state"], download["features"]), ("failed", ["normals", "sfm"]))
        self.assertIn("退出码 1", download["error"])
        self.assertIn("curl: (22) The requested URL returned error: 404", download["error"])
        # what arrived is kept and counted
        self.assertEqual((download["downloaded_mb"], download["total_mb"]), (2, 6))
        self.assertEqual(download["files"][0], {"name": "moge.bin", "size_mb": 4, "downloaded_mb": 2})
        self.assertFalse(view["features"]["normals"]["ready"])
        wait_until(lambda: self.app.models.child is None, 10, message="the child to be reaped")

        # exit code 0 with the files still missing is a failure as well
        self.models.exit_code = 0
        self.assertEqual(self.download({"features": ["normals"]})[0], 202)
        download = self.wait_download()["download"]
        self.assertEqual(download["state"], "failed")
        self.assertIn("退出码 0", download["error"])

        # the retry goes on from the partial file and clears the error
        self.models.mode = "ok"
        status, view = self.download({"features": ["normals", "sfm"]})
        self.assertEqual(status, 202, view)
        self.assertEqual((view["download"]["state"], view["download"]["error"]), ("running", None))
        view = self.wait_download()
        self.assertEqual((view["download"]["state"], view["download"]["error"],
                          view["download"]["downloaded_mb"]), ("done", None, 6))
        self.assertTrue(view["features"]["normals"]["ready"] and view["features"]["sfm"]["ready"])

    def test_a_download_command_that_cannot_start_is_a_failure_not_a_hang(self):
        server.build_models_command = lambda features: [str(self.tmp / "no-such-program.exe")]
        self.assertEqual(self.download({"features": ["normals"]})[0], 202)
        download = self.wait_download()["download"]
        self.assertEqual(download["state"], "failed")
        self.assertIn("无法下载", download["error"])
        self.assertIsNone(self.app.models.child)

    def test_a_download_that_hangs_is_stopped_by_the_time_limit(self):
        self.models.mode = "hang"
        self.app.models.time_limit = 1.0
        self.assertEqual(self.download({"features": ["normals"]})[0], 202)
        wait_until(lambda: self.app.models.child is not None, 20, message="the download to start")
        pid = self.app.models.child.pid
        began = time.monotonic()
        download = self.wait_download()["download"]
        self.assertLess(time.monotonic() - began, 20)
        self.assertEqual(download["state"], "failed")
        self.assertIn("已被停止", download["error"])
        wait_until(lambda: not server.pid_alive(pid), 10, message="the download to die")
        wait_until(lambda: self.app.models.child is None, 10, message="the child to be reaped")

    def test_shutdown_stops_a_running_download(self):
        self.models.mode = "hang"
        self.assertEqual(self.download({"features": ["normals"]})[0], 202)
        wait_until(lambda: self.app.models.child is not None, 20, message="the download to start")
        pid = self.app.models.child.pid
        self.assertTrue(server.pid_alive(pid))
        wait_until(lambda: (self.models.folder / "_part" / "moge.bin").exists(), 20,
                   message="the partial file")
        began = time.monotonic()
        self.httpd.shutdown()
        self.app.shutdown()
        self.assertLess(time.monotonic() - began, 20)
        wait_until(lambda: not server.pid_alive(pid), 10, message="the download to die")
        self.assertFalse(self.app.models._thread.is_alive())
        download = self.app.models.download_view()
        self.assertEqual(download["state"], "failed")
        self.assertIn("中断", download["error"])
        # what was fetched stays for the next start
        self.assertEqual((self.models.folder / "_part" / "moge.bin").stat().st_size, 2 * MIB)
        # and a stopping server starts nothing new
        with self.assertRaises(server.ApiError) as caught:
            self.app.models.start(["normals"])
        self.assertEqual(caught.exception.status, 503)

    def test_unreadable_model_state_is_an_error_of_that_call_only(self):
        def broken():
            raise ValueError("settings.json names a model file this version does not know: x.bin")

        server.models_status = broken
        status, error = self.api("GET", "/api/models")
        self.assertEqual((status, error["error"]), (500, "models_unavailable"))
        self.assertIn("x.bin", error["message"])
        for path, body in (("/api/models/accept", {"licences": ["sam3"]}),
                           ("/api/models/download", {})):
            status, error = self.api("POST", path, body)
            self.assertEqual((status, error["error"]), (500, "models_unavailable"), path)
        self.assertEqual(self.models.accepted, set())
        self.assertEqual(self.api("GET", "/api/jobs")[0], 200)
        self.assertEqual(self.api("GET", "/api/info")[0], 200)

    def test_download_is_refused_when_the_disk_is_too_full(self):
        real = server.shutil.disk_usage
        server.shutil.disk_usage = lambda path: real(path)._replace(free=7 * MIB)
        self.addCleanup(setattr, server.shutil, "disk_usage", real)
        status, error = self.download({"features": ["normals"]})
        self.assertEqual((status, error["error"]), (507, "disk_full"))
        self.assertEqual(self.models_view()["download"]["state"], "idle")
        self.assertEqual(self.models.commands, [])


# =========================================================================== per-scan switches

class OptionsTests(ServerCase):
    def test_only_the_known_switches_with_true_or_false(self):
        job = self.create_with_photo()
        for bad in ({"masks": 1}, {"masks": 0}, {"masks": "true"}, {"masks": None},
                    {"subject_only": "yes"}, {"other": True}, {"masks": True, "iterations": 5},
                    {"train_args": ["--x"]}, [], ["masks"], "masks", 5, True, False):
            status, error = self.api("POST", "/api/jobs", {"options": bad})
            self.assertEqual((status, error["error"]), (400, "bad_options"), bad)
            status, error = self.api("POST", "/api/jobs/%s/start" % job, {"options": bad})
            self.assertEqual((status, error["error"]), (400, "bad_options"), bad)
        # nothing was created and nothing was started
        self.assertEqual([p.name for p in self.runs.iterdir() if p.is_dir()], [job])
        self.assertEqual(self.job(job)["state"], "created")
        self.assertEqual(self.job(job)["options"], {})

    def test_switches_are_stored_shown_and_passed_to_the_pipeline(self):
        job = self.create_with_photo(step=0.01, options={"subject_only": True})
        self.assertEqual(self.job(job)["options"], {"subject_only": True})
        meta = json.loads((self.runs / job / "server-job.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["options"], {"subject_only": True})
        status, listing = self.api("GET", "/api/jobs")
        self.assertEqual(listing["jobs"][0]["options"], {"subject_only": True})
        self.start(job)
        view = self.wait_state(job, ("done", "failed"))
        self.assertEqual((view["state"], view["options"]), ("done", {"subject_only": True}))
        argv = self.stub_argv(job)
        self.assertEqual(argv["overrides"], {"subject_only": True})
        self.assertEqual(argv["argv"][argv["argv"].index("--set"):][:2],
                         ["--set", "subject_only=true"])
        self.assertEqual(argv["argv"].count("--set"), 1)

        both = self.create_with_photo(step=0.01, options={"subject_only": False, "masks": False})
        self.start(both)
        self.wait_state(both, ("done", "failed"))
        self.assertEqual(self.stub_argv(both)["overrides"], {"masks": False, "subject_only": False})
        self.assertEqual(self.job(both)["options"], {"masks": False, "subject_only": False})

    def test_a_switch_that_was_not_given_is_not_passed(self):
        for body in ({}, {"options": {}}, {"options": None}):
            job = self.create_with_photo(step=0.01, quality="preview", **body)
            self.assertEqual(self.job(job)["options"], {})
            self.start(job)
            self.assertEqual(self.wait_state(job, ("done", "failed"))["state"], "done")
            argv = self.stub_argv(job)
            self.assertEqual(argv["overrides"], {}, body)
            self.assertNotIn("--set", argv["argv"])
            self.assertFalse(any("masks" in a or "subject_only" in a for a in argv["argv"]))

    def test_switches_given_on_start(self):
        # given: they replace the stored ones as a whole
        job = self.create_with_photo(step=0.01, options={"masks": False})
        view = self.start(job, {"options": {"subject_only": True}, "quality": "high"})
        self.assertEqual(view["options"], {"subject_only": True})
        self.wait_state(job, ("done", "failed"))
        argv = self.stub_argv(job)
        self.assertEqual((argv["overrides"], argv["quality"]), ({"subject_only": True}, "high"))

        # not given: the stored ones are kept (a retry from the job page)
        kept = self.create_with_photo(mode="fail", step=0.01, options={"masks": False})
        self.start(kept)
        self.wait_state(kept, "failed")
        self.wait_idle()
        self.assertEqual(self.stub_argv(kept)["overrides"], {"masks": False})
        (self.runs / kept / "stub_argv.json").unlink()
        self.stub_args[kept] = ["--step", "0.01"]
        self.start(kept, {})
        self.assertEqual(self.wait_state(kept, ("done", "failed"))["state"], "done")
        self.assertEqual(self.stub_argv(kept)["overrides"], {"masks": False})
        self.assertEqual(self.job(kept)["options"], {"masks": False})

        # an empty set clears them
        cleared = self.create_with_photo(step=0.01, options={"masks": False})
        self.assertEqual(self.start(cleared, {"options": {}})["options"], {})
        self.wait_state(cleared, ("done", "failed"))
        self.assertEqual(self.stub_argv(cleared)["overrides"], {})

    def test_switches_for_a_job_from_a_local_path(self):
        photos = self.tmp / "photos"
        photos.mkdir()
        (photos / "a.jpg").write_bytes(b"x")
        self.stub_args["object"] = ["--step", "0.01"]
        status, view = self.api("POST", "/api/jobs", {
            "name": "object", "path": str(photos), "start": True,
            "options": {"subject_only": True}})
        self.assertEqual((status, view["options"]), (201, {"subject_only": True}))
        self.assertEqual(self.wait_state("object", ("done", "failed"))["state"], "done")
        self.assertEqual(self.stub_argv("object")["overrides"], {"subject_only": True})
        # refused switches leave no job behind, also on this path
        status, error = self.api("POST", "/api/jobs", {
            "name": "bad", "path": str(photos), "start": True, "options": {"masks": "no"}})
        self.assertEqual((status, error["error"]), (400, "bad_options"))
        self.assertFalse((self.runs / "bad").exists())

    def test_a_job_file_edited_by_hand_cannot_smuggle_other_overrides(self):
        job = self.create_with_photo(step=0.01)
        path = self.runs / job / "server-job.json"
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta["options"] = {"masks": "yes", "iterations": 1, "train_args": ["--x"],
                           "subject_only": True}
        path.write_text(json.dumps(meta), encoding="utf-8")
        self.assertEqual(self.job(job)["options"], {"subject_only": True})
        self.start(job)
        self.wait_state(job, ("done", "failed"))
        self.assertEqual(self.stub_argv(job)["overrides"], {"subject_only": True})

    def test_jobs_the_server_did_not_create_have_no_switches(self):
        self.make_done_job("by-hand")
        self.assertEqual(self.job("by-hand")["options"], {})


# =========================================================================== start and stop

def write_job(runs: Path, name: str, **fields) -> None:
    (runs / name).mkdir()
    status = {"job": name, "state": "running", "stage": "train", "stage_index": 4,
              "stage_count": 6, "stage_progress": 0.5, "message": "训练中", "quality": "standard",
              "started_at": "2026-01-01T10:00:00+08:00", "updated_at": "2026-01-01T10:05:00+08:00",
              "finished_at": None, "error": None, "pid": 999999}
    status.update(fields)
    (runs / name / "status.json").write_text(json.dumps(status), encoding="utf-8")


class RecoveryTests(ServerCase):
    def prepare(self):
        write_job(self.runs, "was-running")
        write_job(self.runs, "was-queued", state="queued", pid=None, stage=None)
        # a live pid proves nothing at startup: this server instance did not start the job
        write_job(self.runs, "live-pid", pid=os.getpid())
        write_job(self.runs, "finished", state="done", finished_at="2026-01-01T10:30:00+08:00",
                  outputs={"splat": "output/splat.ply"})
        write_job(self.runs, "broken", state="failed", error="earlier failure")
        (self.runs / "not-a-job").mkdir()
        (self.runs / "not-a-job" / "data.bin").write_bytes(b"keep me")
        (self.runs / "_meshlab").mkdir()
        (self.runs / ".trash-old-0123abcd").mkdir()
        (self.runs / ".trash-old-0123abcd" / "junk").write_bytes(b"x")
        (self.runs / "garbled").mkdir()
        (self.runs / "garbled" / "status.json").write_text("{ not json", encoding="utf-8")

    def test_stale_jobs_are_marked_interrupted_at_start(self):
        for name in ("was-running", "was-queued", "live-pid"):
            status = self.disk_status(name)
            self.assertEqual(status["state"], "failed", name)
            self.assertEqual(status["message"], "interrupted", name)
            self.assertEqual(status["error"], "interrupted", name)
            self.assertIsNotNone(status["finished_at"], name)
            self.assertIsNone(status["pid"], name)
            view = self.job(name)
            self.assertEqual(view["state"], "failed")
            self.assertNotIn("start", view["actions"])   # the server has no input recorded
            self.assertIn("delete", view["actions"])
        self.assertEqual(self.disk_status("was-running")["stage"], "train")

        finished = self.disk_status("finished")
        self.assertEqual((finished["state"], finished["message"]), ("done", "训练中"))
        self.assertEqual(finished["updated_at"], "2026-01-01T10:05:00+08:00")
        self.assertEqual(self.disk_status("broken")["error"], "earlier failure")

        status, listing = self.api("GET", "/api/jobs")
        self.assertEqual(sorted(j["job"] for j in listing["jobs"]),
                         ["broken", "finished", "garbled", "live-pid", "was-queued", "was-running"])
        self.assertEqual({j["job"]: j["state"] for j in listing["jobs"]}["garbled"], "failed")
        self.assertEqual((self.runs / "garbled" / "status.json").read_text(encoding="utf-8"),
                         "{ not json")
        self.assertEqual((self.runs / "not-a-job" / "data.bin").read_bytes(), b"keep me")
        self.assertTrue((self.runs / "_meshlab").is_dir())
        self.assertFalse((self.runs / ".trash-old-0123abcd").exists())
        self.assertEqual(self.job("finished")["elapsed_seconds"], 1800.0)

    def test_job_whose_process_dies_later_is_noticed(self):
        real_interval = server.STALE_CHECK_SECONDS
        server.STALE_CHECK_SECONDS = 0.2
        self.addCleanup(setattr, server, "STALE_CHECK_SECONDS", real_interval)
        # started by hand in another console while the server is up: one died, one is alive
        write_job(self.runs, "cli-dead")
        write_job(self.runs, "cli-alive", pid=os.getpid())
        write_job(self.runs, "cli-fresh")
        old = time.time() - 120
        for name in ("cli-dead", "cli-alive"):
            os.utime(self.runs / name / "status.json", (old, old))
        wait_until(lambda: self.disk_status("cli-dead")["state"] == "failed", 15,
                   message="the dead job to be noticed")
        self.assertEqual(self.disk_status("cli-dead")["message"], "interrupted")
        self.assertEqual(self.disk_status("cli-fresh")["state"], "running")   # just updated
        time.sleep(0.6)
        self.assertEqual(self.disk_status("cli-alive")["state"], "running")
        # an externally running job can be asked to stop, and cannot be deleted
        status, view = self.api("POST", "/api/jobs/cli-alive/cancel", {})
        self.assertEqual(status, 200)
        self.assertTrue((self.runs / "cli-alive" / "cancel").exists())
        self.assertEqual(self.api("DELETE", "/api/jobs/cli-alive")[0], 409)


class ShutdownTests(ServerCase):
    def test_shutdown_kills_the_pipeline_tree_and_marks_jobs(self):
        running = self.create_with_photo(mode="hang", extra=["--child"], name="running")
        queued = self.create_with_photo(name="queued")
        self.start(running)
        self.start(queued)
        view = self.wait_state(running, "running")
        pid = view["pid"]
        child_file = self.runs / running / "child.pid"
        child_pid = int(wait_until(lambda: child_file.exists() and child_file.read_text().strip(),
                                   10, message="the grandchild pid"))
        self.assertTrue(server.pid_alive(pid) and server.pid_alive(child_pid))

        began = time.monotonic()
        self.httpd.shutdown()
        self.app.shutdown()
        self.assertLess(time.monotonic() - began, 20)
        self.assertFalse(server.pid_alive(pid))
        wait_until(lambda: not server.pid_alive(child_pid), 10, message="the grandchild to die")
        for name in (running, queued):
            status = self.disk_status(name)
            self.assertEqual((status["state"], status["message"]), ("failed", "interrupted"), name)
        self.assertFalse((self.runs / queued / "stub_argv.json").exists())
        self.assertFalse(self.app._worker.is_alive())


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="scan3d-test-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        # serve() takes no folder: point the module at the temp folder, so that nothing in
        # this class can ever start a server on the project's real runs/.
        real_root = server.ROOT_DIR
        server.ROOT_DIR = Path(self.tmp)
        self.addCleanup(setattr, server, "ROOT_DIR", real_root)
        self.said = []
        real_say, real_config = server._say, server.logging.basicConfig
        server._say = self.said.append
        server.logging.basicConfig = lambda **kwargs: None
        self.addCleanup(setattr, server, "_say", real_say)
        self.addCleanup(setattr, server.logging, "basicConfig", real_config)

    def test_port_in_use_is_a_clear_error(self):
        first = server.create_server(0, host="127.0.0.1")
        self.addCleanup(first.server_close)
        port = first.server_address[1]
        with self.assertRaises(server.ServerStartError) as caught:
            server.create_server(port, host="127.0.0.1")
        self.assertIn(str(port), str(caught.exception))
        self.assertIn("占用", str(caught.exception))
        # ... and serve() turns that into an exit: no traceback, no other port, no other
        # address (it wants 0.0.0.0:<port>, which the loopback listener above blocks)
        real_start = server.App.start
        server.App.start = lambda app: self.fail("the server must not start")
        self.addCleanup(setattr, server.App, "start", real_start)
        with self.assertRaises(SystemExit) as exited:
            server.serve(port, open_browser=False)
        self.assertEqual(exited.exception.code, 1)
        self.assertEqual(len(self.said), 1)
        self.assertIn("0.0.0.0:%d" % port, self.said[0])
        self.assertIn("无法", self.said[0])

    def test_an_address_that_cannot_be_bound_is_a_clear_error(self):
        with self.assertRaises(server.ServerStartError) as caught:
            server.create_server(0, host="203.0.113.7")      # not an address of this PC
        self.assertIn("无法在 203.0.113.7:0 上启动服务", str(caught.exception))

    def test_serve_prints_the_local_and_the_phone_addresses(self):
        httpd = server.create_server(0, host="127.0.0.1", token=TOKEN)
        port = httpd.server_address[1]
        asked = []

        def stop_at_once(poll_interval=0.5):
            raise KeyboardInterrupt

        httpd.serve_forever = stop_at_once
        real_create, real_addresses = server.create_server, server.lan_addresses
        server.create_server = lambda wanted: asked.append(wanted) or httpd
        server.lan_addresses = lambda: ["192.168.1.20", "10.0.0.7"]
        self.addCleanup(setattr, server, "create_server", real_create)
        self.addCleanup(setattr, server, "lan_addresses", real_addresses)
        server.serve(port, open_browser=False)
        self.assertEqual(asked, [port])
        text = "\n".join(self.said)
        self.assertIn("http://127.0.0.1:%d/" % port, text)
        self.assertIn("http://192.168.1.20:%d/?token=%s" % (port, TOKEN), text)
        self.assertIn("http://10.0.0.7:%d/?token=%s" % (port, TOKEN), text)
        self.assertFalse(httpd.app._worker.is_alive())       # stopped cleanly

        # no network at all: say so instead of printing a made-up address
        self.said.clear()
        httpd = real_create(0, host="127.0.0.1", token=TOKEN)
        httpd.serve_forever = stop_at_once
        server.create_server = lambda wanted: httpd
        server.lan_addresses = lambda: []
        server.serve(0, open_browser=False)
        text = "\n".join(self.said)
        self.assertIn("http://127.0.0.1:%d/" % httpd.server_address[1], text)
        self.assertIn("没有找到", text)
        self.assertNotIn("token=", text)

    def test_token_is_created_once_and_kept_across_restarts(self):
        runs = Path(self.tmp) / "runs"
        first = server.App(self.tmp)
        self.assertRegex(first.token, r"^[a-z2-9]{12}$")
        saved = runs / ".server-token"
        self.assertEqual(saved.read_text(encoding="ascii").strip(), first.token)
        self.assertEqual(sorted(p.name for p in runs.iterdir()), [".server-token"])
        for _ in range(3):                                  # "restarts"
            self.assertEqual(server.App(self.tmp).token, first.token)
        # through create_server as well, which is what serve() uses
        httpd = server.create_server(0, host="127.0.0.1")
        self.addCleanup(httpd.server_close)
        self.assertEqual(httpd.app.token, first.token)

        # a token given by the caller (tests) is used as it is and not written anywhere
        self.assertEqual(server.App(self.tmp, token=TOKEN).token, TOKEN)
        self.assertEqual(saved.read_text(encoding="ascii").strip(), first.token)

        # a damaged or emptied file is replaced by a fresh token, which is then kept
        for damaged in ("", "short", "UPPERCASE123", "x" * 500, "abc def ghi jk", "\x00" * 12):
            saved.write_text(damaged, encoding="ascii")
            again = server.App(self.tmp)
            self.assertRegex(again.token, r"^[a-z2-9]{12}$")
            self.assertEqual(saved.read_text(encoding="ascii").strip(), again.token)
            self.assertEqual(server.App(self.tmp).token, again.token)
        saved.unlink()
        self.assertNotEqual(server.App(self.tmp).token, first.token)    # deleting it re-pairs

    def test_tokens_are_unpredictable(self):
        tokens = {server.generate_token() for _ in range(50)}
        self.assertEqual(len(tokens), 50)
        for token in tokens:
            self.assertRegex(token, r"^[a-z2-9]{12}$")

    def test_token_file_is_not_a_job_and_is_never_served(self):
        httpd = server.create_server(0, host="127.0.0.1")
        app = httpd.app
        app.start()
        thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05},
                                  daemon=True)
        thread.start()

        def stop():
            httpd.shutdown()
            app.shutdown(timeout=20)
            httpd.server_close()

        self.addCleanup(stop)
        port = httpd.server_address[1]

        def get(path):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
            try:
                conn.request("GET", path)
                response = conn.getresponse()
                return response.status, response.read()
            finally:
                conn.close()

        status, body = get("/api/jobs")
        self.assertEqual((status, json.loads(body)["jobs"]), (200, []))
        for path in ("/api/jobs/.server-token", "/api/jobs/.server-token/download/output/x",
                     "/static/..%2f..%2fruns%2f.server-token", "/.server-token",
                     "/api/jobs/x/download/..%2f.server-token"):
            status, body = get(path)
            self.assertIn(status, (400, 404), path)
            self.assertNotIn(app.token.encode(), body, path)
        # the first start on this folder wrote it; the server is still fine with it there
        self.assertTrue((Path(self.tmp) / "runs" / ".server-token").is_file())


# =========================================================================== pure functions

class UnitTests(unittest.TestCase):
    def test_build_run_command(self):
        plain = [sys.executable, str(APP_DIR / "scan3d.py"), "run",
                 r"C:\photos\room", "--name", "my-job", "--quality", "high"]
        # no switch given: nothing is passed, the quality preset decides
        self.assertEqual(server.build_run_command("my-job", r"C:\photos\room", "high"), plain)
        for nothing in (None, {}):
            self.assertEqual(
                server.build_run_command("my-job", r"C:\photos\room", "high", nothing), plain)
        # each given switch becomes one --set, false as well as true, in a fixed order
        cmd = server.build_run_command("my-job", r"C:\photos\room", "high",
                                       {"subject_only": True, "masks": False})
        self.assertEqual(cmd, plain + ["--set", "masks=false", "--set", "subject_only=true"])
        self.assertEqual(server.build_run_command("j", "p", "preview", {"masks": True}),
                         [sys.executable, str(APP_DIR / "scan3d.py"), "run", "p", "--name", "j",
                          "--quality", "preview", "--set", "masks=true"])
        # anything that is not a known switch with true/false never reaches the command line
        cmd = server.build_run_command("my-job", r"C:\photos\room", "high", {
            "masks": "false", "subject_only": 1, "iterations": 5, "train_args": ["--x"]})
        self.assertEqual(cmd, plain)

    def test_build_models_command(self):
        base = [sys.executable, str(APP_DIR / "scan3d.py"), "models", "--download"]
        for nothing in (None, [], ()):
            self.assertEqual(server.build_models_command(nothing), base)
        self.assertEqual(server.build_models_command(["masks"]), base + ["masks"])
        self.assertEqual(server.build_models_command(["masks", "dense_seed"]),
                         base + ["masks,dense_seed"])

    def test_job_options(self):
        self.assertIsNone(server.check_job_options(None))
        self.assertEqual(server.check_job_options({}), {})
        self.assertEqual(list(server.check_job_options({"subject_only": False, "masks": True})),
                         ["masks", "subject_only"])
        for bad in ({"masks": 1}, {"masks": "true"}, {"masks": None}, {"other": True},
                    {"masks": True, "iterations": 5}, [], "masks", 5, True, False):
            with self.assertRaises(server.ApiError, msg=repr(bad)) as caught:
                server.check_job_options(bad)
            self.assertEqual((caught.exception.status, caught.exception.code),
                             (400, "bad_options"))
        self.assertEqual(server.stored_job_options(None), {})
        self.assertEqual(server.stored_job_options({"options": "masks"}), {})
        self.assertEqual(server.stored_job_options(
            {"options": {"masks": "yes", "iterations": 1, "subject_only": True}}),
            {"subject_only": True})

    def test_the_real_models_module_gives_what_the_server_expects(self):
        """The four functions tests replace everywhere else, against the real models.py and
        settings.json, with the weights folder and the licence record in a temp folder."""
        import config
        import models
        saved = (config.MODELS, models._ACCEPTED_FILE)
        with tempfile.TemporaryDirectory() as tmp:
            config.MODELS = Path(tmp) / "models"
            models._ACCEPTED_FILE = Path(tmp) / "accepted-licences.json"
            try:
                self.assertEqual(server.models_dir(), Path(tmp) / "models")
                status = server.models_status()
                self.assertLessEqual({"features", "licences", "files_missing", "download_mb"},
                                     set(status))
                for key, info in status["features"].items():
                    self.assertIn(key, server.FEATURE_LABELS)
                    self.assertLessEqual({"files", "enabled", "ready", "licences_needed",
                                          "files_missing", "download_mb"}, set(info))
                    self.assertFalse(info["ready"], key)        # nothing is installed here
                    for name in info["files"]:
                        self.assertGreater(server.models_file_mb(name), 0, name)
                    for licence in info["licences_needed"]:
                        self.assertIn(licence, status["licences"])
                for licence in status["licences"].values():
                    self.assertFalse(licence["accepted"])
                    self.assertTrue(licence["name"] and licence["url"].startswith("https://"))
                self.assertEqual(server.models_file_mb("no-such-file.bin"), 0)

                first = sorted(status["licences"])[0]
                with self.assertRaises(ValueError):
                    server.models_accept([first, "no-such-licence"])
                self.assertFalse(models._ACCEPTED_FILE.exists())
                server.models_accept([first])
                self.assertEqual(list(json.loads(models._ACCEPTED_FILE.read_text(
                    encoding="utf-8"))["accepted"]), [first])
                self.assertTrue(server.models_status()["licences"][first]["accepted"])
            finally:
                config.MODELS, models._ACCEPTED_FILE = saved

    def test_job_names(self):
        for good in ("scan-20260101-120000", "a", "A1", "living_room-2", "x" * 64, "0"):
            self.assertTrue(server.valid_job_name(good), good)
        for bad in ("", "..", ".", "a/b", "a\\b", "a b", "a.b", "-a", "_a", "x" * 65, "名字",
                    "con", "CON", "nul", "Com1", "LPT9", "aux", "prn", "a\n", "a\x00", None, 5,
                    "C:", "a:b", "a*", "a?", "..\\x"):
            self.assertFalse(server.valid_job_name(bad), repr(bad))

    def test_file_names(self):
        for good in ("IMG_0001.JPG", "客厅 1.heic", "a.b.c.png", "x" * 116 + ".jpg", "é.png",
                     "photo (1).jpeg", "~temp.mp4"):
            self.assertTrue(server.valid_component(good), good)
        for bad in ("", ".", "..", "a/b.jpg", "a\\b.jpg", "C:a.jpg", "a.jpg:x", "a<b.jpg", "a>b",
                    'a"b', "a|b", "a?b", "a*b", "a\x00b", "a\nb", "a\tb", "a\x7fb", ".hidden",
                    " a.jpg", "a.jpg ", "a.jpg.", "con", "CON.jpg", "nul.tar.gz", "com1.png",
                    "LPT1", "aux.mp4", "\u202egpj.exe", "a\u200bb.jpg", "x" * 121, None, 3, b"a"):
            self.assertFalse(server.valid_component(bad), repr(bad))
        self.assertEqual(server.file_extension("A.JPG"), "jpg")
        self.assertEqual(server.file_extension("a.tar.GZ"), "gz")
        self.assertEqual(server.file_extension("noext"), "")
        self.assertEqual(server.file_extension(".jpg"), "")
        self.assertEqual(set(server.ALLOWED_EXTENSIONS),
                         set("jpg jpeg png webp heic heif tif tiff bmp mp4 mov m4v mkv avi webm".split()))

    def test_safe_join(self):
        base = Path(tempfile.mkdtemp(prefix="scan3d-test-"))
        self.addCleanup(shutil.rmtree, base, True)
        self.assertEqual(server.safe_join(base, ["a", "b.txt"]), base / "a" / "b.txt")
        for parts in ([], [".."], ["a", ".."], ["..", "x"], ["a/b"], ["a\\..\\..\\b"], ["C:"],
                      [""], ["a", ""], ["nul"], [str(base)]):
            with self.assertRaises(server.ApiError, msg=repr(parts)):
                server.safe_join(base, parts)

    def test_is_loopback_address(self):
        for address in ("127.0.0.1", "127.8.9.10", "::1", "::ffff:127.0.0.1"):
            self.assertTrue(server.is_loopback_address(address), address)
        for address in ("192.168.1.5", "10.0.0.1", "0.0.0.0", "8.8.8.8", "fe80::1", "::",
                        "::ffff:192.168.1.5", "localhost", "", "127.0.0.1.evil.example", "fe80::1%eth0"):
            self.assertFalse(server.is_loopback_address(address), address)

    def test_parse_range(self):
        self.assertIsNone(server.parse_range(None, 100))
        self.assertIsNone(server.parse_range("", 100))
        self.assertEqual(server.parse_range("bytes=0-0", 100), (0, 0))
        self.assertEqual(server.parse_range("bytes=0-", 100), (0, 99))
        self.assertEqual(server.parse_range("bytes=10-19", 100), (10, 19))
        self.assertEqual(server.parse_range("bytes=90-500", 100), (90, 99))
        self.assertEqual(server.parse_range("bytes=-10", 100), (90, 99))
        self.assertEqual(server.parse_range("bytes=-500", 100), (0, 99))
        self.assertEqual(server.parse_range(" Bytes = 5 - 6 ", 100), (5, 6))
        for ignored in ("items=0-5", "bytes=0-5,10-15", "bytes=abc", "bytes=5-2", "bytes=-",
                        "bytes", "bytes=1-2-3", "bytes=\u0661-\u0662"):
            self.assertIsNone(server.parse_range(ignored, 100), ignored)
        for impossible in ("bytes=100-", "bytes=100-200", "bytes=-0"):
            with self.assertRaises(server.RangeNotSatisfiable, msg=impossible):
                server.parse_range(impossible, 100)
        with self.assertRaises(server.RangeNotSatisfiable):
            server.parse_range("bytes=0-", 0)

    def test_log_text_is_cut_on_character_boundaries(self):
        text = "训练高斯 step 100\n".encode("utf-8")
        for cut in range(1, len(text) + 1):
            trimmed = server._trim_partial_utf8(text[:cut])
            trimmed.decode("utf-8")   # must not raise
            self.assertGreaterEqual(len(trimmed), cut - 3)

    def test_pid_alive(self):
        self.assertTrue(server.pid_alive(os.getpid()))
        for dead in (0, -1, None, "12", True, 999999):
            self.assertFalse(server.pid_alive(dead), repr(dead))


if __name__ == "__main__":
    unittest.main()
