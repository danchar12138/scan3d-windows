"""Tests for the scan pipeline's orchestration, with Spirula and COLMAP replaced by stand-ins.

Run by script path: tools\\python\\python.exe app\\tests\\test_pipeline.py

The stand-in writes the files and prints the lines the real tool was observed to produce
(tool version 2026.11.12), so these tests cover what this program does around the tools: which
stages run and in which order, the arguments they get, the files they write and delete, the
records, the status, resuming, and what the user is told. They do not show that the real tools
behave this way; the real runs do.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
import coverage  # noqa: E402
import meshpost  # noqa: E402
import models  # noqa: E402
import pipeline  # noqa: E402
import proc  # noqa: E402
import scan3d  # noqa: E402
import workflow as wf  # noqa: E402


def write_png(path: Path, width: int, height: int) -> None:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    rows = (b"\x00" + b"\x80" * width) * height
    path.write_bytes(b"\x89PNG\r\n\x1a\n"
                     + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def write_splat(path: Path, count: int = 6) -> None:
    """A minimal 3D Gaussian splat PLY with the property layout the trainer writes."""
    names = (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
             + [f"f_rest_{i}" for i in range(45)] + ["opacity", "scale_0", "scale_1", "scale_2",
                                                       "rot_0", "rot_1", "rot_2", "rot_3"])
    head = ("ply\nformat binary_little_endian 1.0\n" + f"element vertex {count}\n"
            + "".join(f"property float {n}\n" for n in names) + "end_header\n")
    body = b""
    for i in range(count):
        values = [0.0] * len(names)
        values[0], values[1], values[2] = i * 0.1, 0.2, 0.3
        values[names.index("opacity")] = 2.0
        for k in ("scale_0", "scale_1", "scale_2"):
            values[names.index(k)] = -3.0
        values[names.index("rot_0")] = 1.0
        body += struct.pack(f"<{len(names)}f", *values)
    path.write_bytes(head.encode("ascii") + body)


def write_glb(path: Path) -> None:
    body = json.dumps({"asset": {"version": "2.0"}, "scenes": [{"nodes": [0]}],
                       "nodes": [{"name": "mesh"}]}).encode("utf-8")
    body += b" " * ((-len(body)) % 4)
    path.write_bytes(struct.pack("<III", 0x46546C67, 2, 20 + len(body))
                     + struct.pack("<II", len(body), 0x4E4F534A) + body)


class FakeSpirula:
    """Stands in for Job.tool. Records every call and leaves behind what the tool would."""

    def __init__(self):
        self.calls: list = []
        self.video_frames = 400
        self.video_size = (1920, 1080)
        self.frames_written = 30
        # Images the pose estimation registers, per attempt; None = all of them.
        self.registered: list = [None]
        self.fail: set = set()          # commands that exit with an error
        self.masks_kept = 97.5
        self.lock_seen: list = []

    def commands(self, *prefix: str) -> list:
        return [c for c in self.calls if tuple(c[:len(prefix)]) == prefix]

    @staticmethod
    def value(call: list, flag: str) -> str:
        return call[call.index(flag) + 1]

    def __call__(self, job, name, args, *, log, timeout, on_line=None, **kw):
        assert name == "spirula"
        args = [str(a) for a in args]
        self.calls.append(args)
        self.lock_seen.append((job.dir.parent / ".gpu.lock").exists())
        say = on_line or (lambda line: None)
        key = " ".join(args[:2]) if args[0] in ("sam", "sfm", "train") else args[0]
        if key in self.fail:
            raise proc.ProcError(f"spirula.exe failed with exit code 1; see {log}")

        if args[:3] == ["sam", "video", "--info"]:
            for k, v in (("frames", self.video_frames), ("width", self.video_size[0]),
                         ("height", self.video_size[1]), ("codec", "h264")):
                say(f"{k}\t{v}")
        elif args[:2] == ["sam", "extract"]:
            out = Path(self.value(args, "-o"))
            out.mkdir(parents=True, exist_ok=True)
            for i in range(self.frames_written):
                write_png(out / f"{i * 10:05d}.jpg", *self.video_size)
        elif args[:2] == ["sam", "track"]:
            frames, out = Path(self.value(args, "--frames")), Path(self.value(args, "--out"))
            for i, image in enumerate(sorted(frames.iterdir())):
                write_png(out / f"frame_{i:05d}.png", 4, 4)
                say(f"frame {i}: {image.name} -- kept {self.masks_kept}%")
        elif args[:2] == ["sfm", "auto"]:
            images, workspace = Path(args[2]), Path(self.value(args, "-o"))
            total = sum(1 for p in images.rglob("*") if p.is_file())
            attempt = len(self.commands("sfm", "auto")) - 1
            wanted = self.registered[min(attempt, len(self.registered) - 1)]
            count = total if wanted is None else wanted
            if count:
                (workspace / "sparse" / "0").mkdir(parents=True)
                for f in ("cameras.bin", "images.bin", "points3D.bin"):
                    (workspace / "sparse" / "0" / f).write_bytes(b"\0" * 8)
                say(f"[map]     Wrote model 0 ({count} images, 4321 points) to {workspace}")
                say(f"[run]     RESULT: OK -- {100 * count // total}% of the images registered, "
                    f"0.52 px mean reprojection")
        elif args[0] == "geometry":
            (Path(args[1]) / "normals").mkdir()
            say("3 / 3 images, 340 ms each, 00:00:00.000 left")
        elif args[0] == "dense":
            (Path(args[1]) / "dense").mkdir()
            (Path(args[1]) / "dense" / "roma.ply").write_bytes(b"ply\n")
            say("Dense cloud completed: roma.ply; points: 2387575; seconds: 249.704")
        elif args[0] == "train":
            run = Path(self.value(args, "--output-dir-prefix")) / self.value(args, "--output-dir-name")
            ckpt = run / f"step-{int(self.value(args, '--num-iterations')):09d}.ckpt"
            ckpt.mkdir(parents=True)
            write_splat(ckpt / "splat.ply")
            say("Training complete. Steps: 600   Time: 00:00:05.000")
        else:
            raise AssertionError(f"the pipeline ran a command the stand-in does not know: {args[:3]}")
        return proc.RunResult(0, 1.0)


class PipelineCase(unittest.TestCase):
    """A job folder in a temporary runs folder, stand-ins for the tools, and helpers."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.runs = root / "runs"
        self.runs.mkdir()
        self.spirula = FakeSpirula()
        self.settings = config.load_settings()
        self.settings["limits"].pop("min_images", None)
        self.assertNotIn("min_images", self.settings["limits"])
        self._saved = (config.MODELS, config.RUNS, models._ACCEPTED_FILE, pipeline.Job.tool,
                       config.load_settings, config.missing_tools, models._download_one,
                       coverage.compute, meshpost.build_dense, meshpost.build_meshes)
        config.MODELS = root / "models"
        config.MODELS.mkdir()
        config.RUNS = self.runs
        models._ACCEPTED_FILE = root / "accepted-licences.json"
        fake = self.spirula
        pipeline.Job.tool = lambda job, name, args, **kw: fake(job, name, args, **kw)
        config.load_settings = lambda: json.loads(json.dumps(self.settings))
        config.missing_tools = lambda: []
        self.downloads: list = []

        def fake_download(name, *, log_path, cancel_file, on_progress):
            self.downloads.append(name)
            (config.MODELS / name).write_bytes(b"weights")

        models._download_one = fake_download
        for name in models.CATALOGUE:
            (config.MODELS / name).write_bytes(b"weights")
        self.coverage_calls: list = []

        def fake_coverage(splat, images, cameras, **kw):
            self.coverage_calls.append((Path(splat), Path(images), Path(cameras), kw))
            return {"voxels": {"count": 5}, "levels": [1], "regions": [{}, {}], "seconds": 0.1}

        coverage.compute = fake_coverage
        self.dense_calls: list = []
        self.mesh_calls: list = []
        self.mesh_fail: str = ""
        meshpost.build_dense = self.fake_build_dense
        meshpost.build_meshes = self.fake_build_meshes

        self.photos = root / "input"
        self.photos.mkdir()
        for i in range(1, 9):
            write_png(self.photos / f"IMG_{i}.png", 4000, 3000)

    def tearDown(self):
        (config.MODELS, config.RUNS, models._ACCEPTED_FILE, pipeline.Job.tool,
         config.load_settings, config.missing_tools, models._download_one,
         coverage.compute, meshpost.build_dense, meshpost.build_meshes) = self._saved
        self.tmp.cleanup()

    # ---- stand-ins for the two mesh entry points

    def fake_build_dense(self, *, dataset_dir, out_dir, work_dir, tools, log_path, cancel_file=None,
                         on_progress=None, **kw):
        self.dense_calls.append({"locked": (self.runs / ".gpu.lock").exists(),
                                 "out_dir": Path(out_dir), **kw})
        Path(work_dir).mkdir(parents=True)
        (Path(work_dir) / "depth.bin").write_bytes(b"x" * 100)       # the multi-GB scratch
        out_dir = Path(out_dir)
        (out_dir / "sparse").mkdir(parents=True)
        (out_dir / "dense_mesh.ply").write_bytes(b"ply\n")
        on_progress(0.5, "计算深度图")
        return {"path": out_dir / "dense_mesh.ply", "sparse": out_dir / "sparse", "images": 5,
                "points": 4321, "masked_photos": 0, "faces": 99, "photos_dropped": 1}

    def fake_build_meshes(self, *, run_dir, dataset_dir, out_dir, work_dir, tools, dense,
                          on_progress=None, **kw):
        self.mesh_calls.append({"locked": (self.runs / ".gpu.lock").exists(),
                                "run_dir": run_dir, "dense": Path(dense), **kw})
        if self.mesh_fail:
            raise meshpost.MeshError(self.mesh_fail)
        Path(work_dir).mkdir(parents=True)
        out_dir = Path(out_dir)
        write_glb(out_dir / "mesh.glb")
        write_glb(out_dir / "collision.glb")
        return {"mesh": out_dir / "mesh.glb", "collision": out_dir / "collision.glb",
                "mesh_faces": 10, "collision_faces": 4}

    # ---- helpers

    def job_file(self, *parts) -> Path:
        return self.runs.joinpath("job", *parts)

    def run_job(self, source=None, template=None, enable=(), disable=(), sets=(), **kw):
        """Build the job's workflow the way the command line does, then run it."""
        if template is None and not self.job_file("workflow.json").exists():
            template = "standard"
        self.lines: list = []
        sets = list(sets)
        if not any(s.startswith("ingest.min_images") for s in sets):
            sets.append("ingest.min_images=4")          # the test photo sets are small
        flow = pipeline.build_workflow(self.job_file(), template, enable, disable, sets)
        return pipeline.run_job(source or self.photos, "job", workflow=flow, runs_dir=self.runs,
                                echo=self.lines.append, **kw)

    def status(self) -> dict:
        return json.loads(self.job_file("status.json").read_text(encoding="utf-8"))

    def record(self, name: str) -> dict:
        return json.loads(self.job_file("stages", name + ".json").read_text(encoding="utf-8"))

    def has_record(self, name: str) -> bool:
        return self.job_file("stages", name + ".json").exists()

    def states(self) -> dict:
        return {sid: e["state"] for sid, e in self.status()["stages"].items()}

    def train_call(self) -> list:
        calls = self.spirula.commands("train")
        self.assertEqual(len(calls), 1)
        return calls[0]

    def order(self) -> list:
        return [" ".join(c[:2]) if c[0] in ("sam", "sfm") else c[0] for c in self.spirula.calls]

    def accepted(self) -> dict:
        return json.loads(models._ACCEPTED_FILE.read_text(encoding="utf-8"))["accepted"]


class StagesTest(PipelineCase):
    # ---- the whole run

    def test_photos_run_through_every_stage_and_leave_the_outputs(self):
        self.assertEqual(self.run_job(enable=["masks"]), 0)           # masks are opt-in
        status = self.status()
        self.assertEqual((status["state"], status["stage"], status["error"]), ("done", None, None))
        self.assertEqual(self.order(), ["sam track", "sfm auto", "geometry", "train"])
        for name in ("splat.ply", "report.json", "preview.spz", "preview-lite.spz", "coverage.json"):
            self.assertTrue(self.job_file("output", name).is_file(), name)
        report = json.loads(self.job_file("output", "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["stages"]["ingest"]["images"], 8)
        self.assertEqual(report["stages"]["sfm"]["registered"], 8)
        self.assertEqual(report["stages"]["sfm"]["mean_reprojection_error_px"], 0.52)
        self.assertEqual(report["stages"]["train"]["gaussians"], 6)
        self.assertIn("stage_seconds", report["stages"]["train"])
        self.assertEqual(report["template"], "standard")
        self.assertEqual(report["workflow"]["train"]["params"]["iterations"], 30000)
        self.assertEqual(report["orientation"]["up_in_file_coordinates"], [0.0, 0.0, 1.0])
        self.assertEqual(set(status["outputs"]), {"splat", "preview", "preview_lite", "report"})
        self.assertEqual(report["warnings"], [])
        self.assertEqual(status["run"], {"to": None, "only": [], "force": [], "plan": [
            "ingest", "masks", "sfm", "normals", "train", "coverage", "export"]})
        self.assertEqual(set(status["stages"]), set(wf.STAGE_ORDER))
        self.assertEqual(self.states(), {"ingest": "done", "masks": "done", "sfm": "done",
                                         "normals": "done", "dense_seed": "disabled",
                                         "train": "done", "dense": "disabled", "mesh": "disabled",
                                         "coverage": "done", "export": "done"})
        self.assertEqual(status["stages"]["train"]["summary"]["gaussians"], 6)
        self.assertEqual(status["stages"]["sfm"]["summary"], {"registered": 8, "images": 8})
        self.assertIsNotNone(status["stages"]["train"]["seconds"])
        self.assertNotIn("stage_index", status)
        # every stage left a record with its fingerprint; logs per stage that wrote one
        for sid in ("ingest", "masks", "sfm", "normals", "train", "coverage", "export"):
            rec = self.record(sid)
            self.assertEqual(rec["fingerprint"], wf.fingerprints(
                json.loads(self.job_file("workflow.json").read_text(encoding="utf-8")),
                pipeline.input_signature(self.photos))[sid])
        self.assertTrue(self.job_file("logs", "pipeline.log").is_file())

    def test_the_standard_template_runs_without_masks_unless_asked(self):
        self.assertEqual(self.run_job(), 0)
        self.assertEqual(self.order(), ["sfm auto", "geometry", "train"])
        self.assertEqual(self.status()["run"]["plan"],
                         ["ingest", "sfm", "normals", "train", "coverage", "export"])
        self.assertEqual(self.states()["masks"], "disabled")
        self.assertFalse(self.has_record("masks"))
        self.assertIn("--no-masks", self.spirula.commands("sfm", "auto")[0])
        self.assertFalse(models._ACCEPTED_FILE.exists())              # no SAM licence was taken

    def test_the_coverage_stage_reads_the_trained_splat_and_the_cameras(self):
        self.assertEqual(self.run_job(), 0)
        splat, images, cameras, kw = self.coverage_calls[0]
        self.assertTrue(str(splat).endswith("splat.ply") and "train" in str(splat))
        self.assertEqual(images, self.job_file("dataset", "sparse", "0", "images.bin"))
        self.assertEqual(cameras, self.job_file("dataset", "sparse", "0", "cameras.bin"))
        self.assertEqual(kw["up"], [0.0, 0.0, 1.0])
        self.assertEqual(self.record("coverage")["result"]["regions"], 2)

    def test_gpu_stages_hold_the_lock_and_the_others_do_not(self):
        self.assertEqual(self.run_job(enable=["masks"]), 0)
        locked = {" ".join(c[:2]) if c[0] in ("sam", "sfm") else c[0]: held
                  for c, held in zip(self.spirula.calls, self.spirula.lock_seen)}
        self.assertEqual(locked, {"sam track": True, "sfm auto": True, "geometry": True,
                                  "train": True})
        self.assertFalse((self.runs / ".gpu.lock").exists())
        self.assertEqual(list((self.runs / ".gpu.queue").iterdir()), [])

    def test_video_extraction_does_not_take_the_gpu(self):
        video = Path(self.tmp.name) / "walk.mp4"
        video.write_bytes(b"\0" * 64)
        self.assertEqual(self.run_job(source=video, template="preview"), 0)
        for call, held in zip(self.spirula.calls, self.spirula.lock_seen):
            if call[0] == "sam" and call[1] in ("video", "extract"):
                self.assertFalse(held)

    def test_photos_are_renamed_to_plain_names_and_the_originals_recorded(self):
        (self.photos / "照片 (10).png").write_bytes((self.photos / "IMG_1.png").read_bytes())
        self.assertEqual(self.run_job(), 0)
        names = sorted(p.name for p in self.job_file("dataset", "images", "photos").iterdir())
        self.assertEqual(names, [f"p_{i:05d}.png" for i in range(1, 10)])
        recorded = json.loads(self.job_file("dataset", "names.json").read_text(encoding="utf-8"))
        # Natural order: IMG_2 before IMG_10, and every original is traceable.
        self.assertTrue(recorded["photos/p_00002.png"].endswith("IMG_2.png"))
        self.assertEqual(len(recorded), 9)

    # ---- masks

    def test_masks_are_named_after_their_images(self):
        self.assertEqual(self.run_job(enable=["masks"]), 0)
        masks = sorted(p.name for p in self.job_file("dataset", "masks", "photos").iterdir())
        self.assertEqual(masks, [f"p_{i:05d}.png" for i in range(1, 9)])
        self.assertFalse(self.job_file("dataset", "masks_raw").exists())
        call = self.spirula.commands("sam", "track")[0]
        self.assertEqual(self.spirula.value(call, "--mask-mode"), "image")
        self.assertEqual(self.spirula.value(call, "--text"), "person")
        self.assertEqual(self.spirula.value(call, "--accept-license"), "sam3=yes")
        self.assertTrue(self.spirula.value(call, "--model").endswith("sam3-f16.ggml"))
        masks = self.record("masks")["result"]
        self.assertEqual((masks["masks"], masks["mask_kind"], masks["masked_images"]),
                         (True, "moving", 8))
        self.assertAlmostEqual(masks["masked_percent"], 2.5)
        self.assertEqual(self.spirula.value(self.train_call(), "--load-masks"), "1")

    def test_the_mask_prompt_is_a_workflow_parameter(self):
        self.assertEqual(self.run_job(enable=["masks"], sets=["masks.prompt=person; dog"]), 0)
        call = self.spirula.commands("sam", "track")[0]
        self.assertEqual(self.spirula.value(call, "--text"), "person; dog")

    def test_licences_of_the_weights_are_recorded_when_the_stage_runs(self):
        self.assertFalse(models._ACCEPTED_FILE.exists())
        self.assertEqual(self.run_job(enable=["masks"]), 0)
        accepted = self.accepted()
        self.assertEqual(set(accepted), {"sam3"})
        self.assertEqual(accepted["sam3"]["via"], "feature-enabled")
        self.assertEqual(accepted["sam3"]["url"], models.LICENCES["sam3"]["url"])
        call = self.spirula.commands("sam", "track")[0]
        self.assertEqual(self.spirula.value(call, "--accept-license"), "sam3=yes")

    def test_a_stage_that_does_not_run_records_nothing(self):
        self.assertEqual(self.run_job(template="preview"), 0)
        self.assertFalse(models._ACCEPTED_FILE.exists())

    def test_missing_weights_are_downloaded_when_the_stage_starts(self):
        (config.MODELS / "sam3-f16.ggml").unlink()
        (config.MODELS / "moge2-vitb-normal.onnx").unlink()
        self.assertEqual(self.run_job(enable=["masks"]), 0)
        self.assertEqual(sorted(self.downloads), ["moge2-vitb-normal.onnx", "sam3-f16.ggml"])
        self.assertIn("sam3=yes", self.spirula.commands("sam", "track")[0])

    def test_a_download_that_fails_fails_that_stage_with_a_plain_message(self):
        (config.MODELS / "sam3-f16.ggml").unlink()

        def broken(name, **kw):
            raise proc.ProcError("curl.exe failed with exit code 6")

        models._download_one = broken
        self.assertEqual(self.run_job(enable=["masks"]), 1)
        status = self.status()
        self.assertIn("下载模型失败", status["error"])
        self.assertEqual(status["stages"]["masks"]["state"], "failed")
        self.assertEqual(self.spirula.calls, [])

    def test_the_sam2_engine_passes_its_detector(self):
        self.assertEqual(self.run_job(enable=["masks"], sets=["masks.engine=sam2"]), 0)
        call = self.spirula.commands("sam", "track")[0]
        self.assertTrue(self.spirula.value(call, "--model").endswith("sam2.1_hiera_large_f16.ggml"))
        self.assertTrue(self.spirula.value(call, "--detector").endswith("grounding-dino-base.safetensors"))
        accepted = [call[i + 1] for i, a in enumerate(call) if a == "--accept-license"]
        self.assertEqual(accepted, ["sam2=yes", "gdino=yes"])
        self.assertEqual(set(self.accepted()), {"sam2", "gdino"})

    def test_large_masks_are_pointed_out(self):
        self.spirula.masks_kept = 40.0
        self.assertEqual(self.run_job(enable=["masks"]), 0)
        self.assertTrue(any("抠除的区域很大" in w for w in self.status()["warnings"]))

    def test_a_missing_mask_stops_the_scan(self):
        real = self.spirula.__call__

        def drop_one(job, name, args, **kw):
            result = real(job, name, args, **kw)
            if [str(a) for a in args[:2]] == ["sam", "track"]:
                out = Path(self.spirula.value([str(a) for a in args], "--out"))
                sorted(out.iterdir())[0].unlink()
            return result

        pipeline.Job.tool = lambda job, name, args, **kw: drop_one(job, name, args, **kw)
        self.assertEqual(self.run_job(enable=["masks"]), 1)
        self.assertIn("7/8", self.status()["error"])
        self.assertEqual(self.states()["masks"], "failed")

    def test_preview_template_has_no_masks_or_priors(self):
        self.assertEqual(self.run_job(template="preview"), 0)
        self.assertEqual([c[0] for c in self.spirula.calls], ["sfm", "train"])
        call = self.train_call()
        self.assertEqual([self.spirula.value(call, f) for f in
                          ("--load-normals", "--load-depths", "--load-masks", "--seed-pointcloud")],
                         ["0", "0", "0", "sparse"])
        self.assertIn("--no-masks", self.spirula.commands("sfm", "auto")[0])
        self.assertEqual(self.spirula.value(call, "--num-iterations"), "7000")

    def test_subject_scan_keeps_the_subject_and_trains_for_it(self):
        self.assertEqual(self.run_job(enable=["masks"], sets=["masks.kind=subject"]), 0)
        call = self.spirula.commands("sam", "track")[0]
        self.assertTrue(self.spirula.value(call, "--model").endswith("birefnet-general.safetensors"))
        self.assertNotIn("--text", call)
        self.assertEqual(self.spirula.value(call, "--accept-license"), "birefnet=yes")
        self.assertEqual(self.record("masks")["result"]["mask_kind"], "subject")
        self.assertEqual(self.train_call()[1], "centered-object")

    def test_masks_that_are_switched_off_are_not_seen_by_later_stages(self):
        self.assertEqual(self.run_job(enable=["masks"]), 0)
        self.assertTrue(self.job_file("dataset", "masks").is_dir())
        self.spirula.calls.clear()
        self.assertEqual(self.run_job(disable=["masks"]), 0)
        # the folder from the earlier run is still there, and nobody is told about it
        self.assertTrue(self.job_file("dataset", "masks").is_dir())
        self.assertEqual(self.order(), ["sfm auto", "geometry", "train"])
        self.assertIn("--no-masks", self.spirula.commands("sfm", "auto")[0])
        self.assertEqual(self.spirula.value(self.train_call(), "--load-masks"), "0")
        self.assertEqual(self.states()["masks"], "disabled")
        # ...and switching them on again finds them current: nothing is run for them
        self.spirula.calls.clear()
        self.assertEqual(self.run_job(enable=["masks"]), 0)
        self.assertEqual(self.spirula.commands("sam", "track"), [])

    # ---- video

    def test_video_frames_get_their_own_folder_and_are_treated_as_a_sequence(self):
        video = Path(self.tmp.name) / "walk.mp4"
        video.write_bytes(b"\0" * 64)
        self.assertEqual(self.run_job(source=video, enable=["masks"],
                                      sets=["ingest.video_frames=240"]), 0)
        extract = self.spirula.commands("sam", "extract")[0]
        self.assertEqual(extract[2], str(video))
        self.assertTrue(self.spirula.value(extract, "-o").endswith(os.path.join("images", "v1")))
        # 400 source frames, 240 wanted: every 2nd frame is a candidate.
        self.assertEqual(self.spirula.value(extract, "-s"), "2")
        self.assertNotIn("--scale", extract)
        self.assertEqual(self.spirula.value(extract, "--blur-reject"), "2.0")
        track = self.spirula.commands("sam", "track")[0]
        self.assertEqual(self.spirula.value(track, "--mask-mode"), "image")
        sfm = self.spirula.commands("sfm", "auto")[0]
        self.assertEqual(self.spirula.value(sfm, "--sequence"), "v1")
        masks = sorted(p.name for p in self.job_file("dataset", "masks", "v1").iterdir())
        self.assertEqual(masks[:2], ["00000.png", "00010.png"])
        ingest = self.record("ingest")["result"]
        self.assertEqual((ingest["videos"], ingest["video_frames"], ingest["sequences"]),
                         (1, 30, ["v1"]))

    def test_video_parameters_come_from_the_workflow(self):
        video = Path(self.tmp.name) / "walk.mp4"
        video.write_bytes(b"\0" * 64)
        self.assertEqual(self.run_job(source=video, template="preview",
                                      sets=["ingest.video_frames=100", "ingest.blur_reject_px=0"]), 0)
        extract = self.spirula.commands("sam", "extract")[0]
        self.assertEqual(self.spirula.value(extract, "-s"), "4")          # 400 / 100
        self.assertNotIn("--blur-reject", extract)

    def test_oversized_video_is_scaled_down_at_extraction(self):
        self.spirula.video_size = (7680, 4320)
        video = Path(self.tmp.name) / "big.mov"
        video.write_bytes(b"\0" * 64)
        self.assertEqual(self.run_job(source=video, template="preview"), 0)
        extract = self.spirula.commands("sam", "extract")[0]
        self.assertEqual(self.spirula.value(extract, "--scale"), "0.5000")

    def test_the_long_side_limit_is_a_parameter(self):
        video = Path(self.tmp.name) / "big.mov"
        video.write_bytes(b"\0" * 64)
        self.assertEqual(self.run_job(source=video, template="preview",
                                      sets=["ingest.max_long_side=0"]), 0)
        self.assertNotIn("--scale", self.spirula.commands("sam", "extract")[0])

    def test_photos_and_video_together(self):
        (self.photos / "clip.mp4").write_bytes(b"\0" * 64)
        self.assertEqual(self.run_job(enable=["masks"]), 0)
        tracked = [(Path(self.spirula.value(c, "--frames")).name, self.spirula.value(c, "--mask-mode"))
                   for c in self.spirula.commands("sam", "track")]
        self.assertEqual(tracked, [("photos", "image"), ("v1", "image")])
        self.assertEqual(self.record("ingest")["result"]["images"], 38)

    # ---- camera poses

    def test_pose_estimation_is_given_the_image_folder_not_the_dataset(self):
        self.assertEqual(self.run_job(enable=["masks"]), 0)
        call = self.spirula.commands("sfm", "auto")[0]
        self.assertEqual(Path(call[2]), self.job_file("dataset", "images"))
        self.assertEqual(Path(self.spirula.value(call, "-o")), self.job_file("dataset"))
        self.assertEqual((self.spirula.value(call, "--features"), self.spirula.value(call, "--matcher")),
                         ("sift", "bruteforce"))
        self.assertEqual(self.spirula.value(call, "--quality"), "high")
        self.assertNotIn("--sequence", call)
        self.assertNotIn("--no-masks", call)
        self.assertEqual(self.spirula.value(call, "--metric-gps"), "none")
        self.spirula.calls.clear()
        self.assertEqual(self.run_job(sets=["sfm.use_gps=true"], force=["sfm"], to="sfm"), 0)
        self.assertEqual(self.spirula.value(self.spirula.commands("sfm", "auto")[0], "--metric-gps"), "auto")

    def test_pose_quality_is_a_parameter(self):
        self.assertEqual(self.run_job(sets=["sfm.quality=medium"]), 0)
        self.assertEqual(self.spirula.value(self.spirula.commands("sfm", "auto")[0], "--quality"),
                         "medium")

    def test_second_feature_type_is_tried_when_the_first_registers_too_few(self):
        self.spirula.registered = [2, None]
        self.assertEqual(self.run_job(), 0)
        calls = self.spirula.commands("sfm", "auto")
        self.assertEqual([self.spirula.value(c, "--features") for c in calls], ["sift", "aliked-n16rot"])
        self.assertTrue(self.spirula.value(calls[1], "--aliked-model").endswith("aliked-n16rot.onnx"))
        self.assertTrue(self.spirula.value(calls[1], "--lightglue-model").endswith("aliked-lightglue.onnx"))
        sfm = self.record("sfm")["result"]
        self.assertEqual((sfm["registered"], sfm["method"]), (8, "aliked-n16rot + lightglue"))

    def test_the_fallback_can_be_switched_off(self):
        self.spirula.registered = [2, None]
        self.assertEqual(self.run_job(sets=["sfm.learned_fallback=false"]), 1)
        self.assertEqual(len(self.spirula.commands("sfm", "auto")), 1)

    def test_scan_fails_with_a_plain_message_when_no_attempt_registers_enough(self):
        self.spirula.registered = [2, 3]
        self.assertEqual(self.run_job(), 1)
        status = self.status()
        self.assertEqual((status["state"], status["stage"]), ("failed", None))
        self.assertEqual(status["stages"]["sfm"]["state"], "failed")
        self.assertIn("8 张图片里只有 3 张", status["error"])
        self.assertIn("8 张图片里只有 3 张", status["stages"]["sfm"]["error"])
        self.assertEqual(self.spirula.commands("train"), [])
        self.assertEqual(status["stages"]["train"]["state"], "pending")

    def test_the_registered_fraction_is_a_parameter(self):
        self.spirula.registered = [6]
        self.assertEqual(self.run_job(sets=["sfm.min_registered_fraction=0.9"]), 1)

    def test_partly_registered_capture_is_reported_as_a_warning(self):
        self.spirula.registered = [6]
        self.assertEqual(self.run_job(), 0)
        self.assertTrue(any("2 张图片未能参与重建" in w for w in self.status()["warnings"]))

    # ---- priors and training

    def test_standard_template_tells_the_trainer_exactly_what_to_read(self):
        self.assertEqual(self.run_job(), 0)
        call = self.train_call()
        self.assertEqual([self.spirula.value(call, f) for f in
                          ("--load-normals", "--load-depths", "--load-masks", "--seed-pointcloud")],
                         ["1", "0", "0", "sparse"])               # no masks unless asked for
        self.assertEqual(self.spirula.value(call, "--disable-viewer"), "1")
        self.assertEqual(self.spirula.value(call, "--keep-viewer-alive"), "0")
        self.assertEqual(call[1], "3dgs")
        self.assertEqual(self.spirula.value(call, "--primitive"), "mip")
        self.assertEqual(self.spirula.value(call, "--cap-max"), "2000000")
        self.assertEqual(self.spirula.commands("dense"), [])
        self.assertEqual(self.record("normals")["result"]["normals"], True)

    def test_dense_seed_is_used_when_switched_on(self):
        self.assertEqual(self.run_job(enable=["dense_seed", "masks"]), 0)
        dense = self.spirula.commands("dense")[0]
        self.assertTrue(self.spirula.value(dense, "--checkpoint").endswith("romav2.0.1.pt"))
        self.assertEqual(self.spirula.value(dense, "--preset"), "balanced")
        self.assertEqual(self.spirula.value(dense, "--use-masks"), "1")
        self.assertEqual(self.spirula.value(self.train_call(), "--seed-pointcloud"), "dense/roma.ply")
        result = self.record("dense_seed")["result"]
        self.assertEqual((result["dense_seed"], result["dense_points"]), (True, 2387575))
        self.assertTrue(self.job_file("logs").is_dir())
        self.assertEqual(self.accepted()["dinov3"]["via"], "feature-enabled")

    def test_dense_seed_ignores_masks_that_are_off(self):
        self.assertEqual(self.run_job(template="preview", enable=["dense_seed"],
                                      sets=["dense_seed.preset=fast"]), 0)
        dense = self.spirula.commands("dense")[0]
        self.assertEqual(self.spirula.value(dense, "--use-masks"), "0")
        self.assertEqual(self.spirula.value(dense, "--preset"), "fast")

    def test_the_normals_model_is_a_parameter(self):
        self.assertEqual(self.run_job(sets=["normals.model=vitl"]), 0)
        call = self.spirula.commands("geometry")[0]
        self.assertTrue(self.spirula.value(call, "--model").endswith("moge2-vitl-normal.onnx"))

    def test_working_resolution_is_a_divisor_of_the_source_size(self):
        self.assertEqual(self.run_job(), 0)           # 4000 px photos, standard works at 1920
        self.assertEqual(self.spirula.value(self.train_call(), "--train-resolution-divisor"), "2.0833")

    def test_small_photos_are_not_upscaled(self):
        for p in self.photos.iterdir():
            write_png(p, 1600, 1200)
        self.assertEqual(self.run_job(), 0)
        self.assertNotIn("--train-resolution-divisor", self.train_call())

    def test_training_parameters_reach_the_trainer(self):
        self.assertEqual(self.run_job(sets=["train.iterations=1500", "train.cap_max=300000",
                                            "train.primitive=3dgs",
                                            "train.extra_args=--floater-suppression mild",
                                            "train.max_image_size=1280"]), 0)
        call = self.train_call()
        self.assertEqual(self.spirula.value(call, "--num-iterations"), "1500")
        self.assertEqual(self.spirula.value(call, "--cap-max"), "300000")
        self.assertEqual(self.spirula.value(call, "--primitive"), "3dgs")
        self.assertEqual(call[-2:], ["--floater-suppression", "mild"])
        self.assertEqual(self.spirula.value(call, "--train-resolution-divisor"), "3.1250")

    def test_held_out_evaluation_is_requested_only_when_asked(self):
        self.assertEqual(self.run_job(), 0)
        self.assertNotIn("--eval-mode", self.train_call())
        self.spirula.calls.clear()
        self.assertEqual(self.run_job(sets=["train.eval_holdout=true"]), 0)
        self.assertEqual(self.spirula.value(self.train_call(), "--eval-mode"), "interval")

    # ---- failures, resuming, cancelling

    def test_too_few_photos_is_explained(self):
        for p in sorted(self.photos.iterdir())[3:]:
            p.unlink()
        self.assertEqual(self.run_job(sets=["ingest.min_images=20"]), 1)
        self.assertIn("只有 3 张", self.status()["error"])
        self.assertIn("至少需要 20 张", self.status()["error"])
        self.assertIn("整理素材", self.status()["error"])
        self.assertEqual(self.states()["ingest"], "failed")
        self.assertEqual(self.spirula.calls, [])

    def test_the_minimum_number_of_photos_is_a_workflow_parameter(self):
        for p in sorted(self.photos.iterdir())[3:]:
            p.unlink()
        self.assertEqual(self.run_job(sets=["ingest.min_images=4"]), 1)
        self.assertIn("至少需要 4 张", self.status()["error"])
        self.assertEqual(self.run_job(sets=["ingest.min_images=3"]), 0)
        self.assertEqual(self.record("ingest")["result"]["images"], 3)
        self.assertEqual(self.record("sfm")["result"]["registered"], 3)

    def test_a_workflow_saved_without_the_minimum_gets_the_default(self):
        flow = wf.resolve("preview")
        del flow["stages"]["ingest"]["params"]["min_images"]
        (self.runs / "old").mkdir()
        (self.runs / "old" / "workflow.json").write_text(json.dumps(flow), encoding="utf-8")
        view = pipeline.job_view(self.runs / "old")
        self.assertEqual(view["workflow"]["stages"]["ingest"]["params"]["min_images"], 20)
        self.assertEqual(view["template"], "preview")

    def test_unsupported_photo_formats_are_skipped_with_a_warning(self):
        (self.photos / "a.heic").write_bytes(b"x")
        self.assertEqual(self.run_job(), 0)
        self.assertEqual(self.record("ingest")["result"]["skipped_unsupported"], 1)
        self.assertTrue(any("a.heic" in w for w in self.status()["warnings"]))
        # a warning belongs to its stage: a later run that does not repeat ingest keeps it
        self.assertEqual(self.run_job(sets=["train.iterations=1000"]), 0)
        self.assertTrue(any("a.heic" in w for w in self.status()["warnings"]))

    def test_a_failed_stage_fails_the_run_and_a_second_run_resumes_after_the_finished_ones(self):
        self.spirula.fail.add("train 3dgs")
        self.assertEqual(self.run_job(), 1)
        status = self.status()
        self.assertEqual(status["state"], "failed")
        self.assertEqual(status["stages"]["train"]["state"], "failed")
        self.assertIn("spirula.exe failed", status["stages"]["train"]["error"])
        self.assertEqual(status["stages"]["coverage"]["state"], "pending")     # after it: not run
        self.assertEqual(status["stages"]["sfm"]["state"], "done")             # before it: kept
        self.assertFalse(self.has_record("train"))
        self.assertEqual(pipeline.job_view(self.job_file())["stages"]["train"]["state"], "failed")
        before = len(self.spirula.calls)
        self.spirula.fail.clear()
        self.assertEqual(self.run_job(), 0)
        self.assertEqual([c[0] for c in self.spirula.calls[before:]], ["train"])
        self.assertEqual(self.status()["state"], "done")
        self.assertEqual(self.states()["train"], "done")

    def test_fresh_discards_earlier_results_but_not_the_workflow_or_the_input(self):
        self.assertEqual(self.run_job(), 0)
        before = len(self.spirula.calls)
        (self.job_file("upload")).mkdir()
        (self.job_file("upload", "x.txt")).write_text("keep")
        self.assertEqual(self.run_job(fresh=True), 0)
        self.assertEqual(len(self.spirula.calls), 2 * before)
        self.assertTrue(self.job_file("workflow.json").is_file())
        self.assertTrue(self.job_file("upload", "x.txt").is_file())

    def test_cancel_file_stops_the_run_between_stages(self):
        real = self.spirula.__call__

        def cancel_after_poses(job, name, args, **kw):
            result = real(job, name, args, **kw)
            if str(args[0]) == "sfm":
                job.cancel_file.touch()
            return result

        pipeline.Job.tool = lambda job, name, args, **kw: cancel_after_poses(job, name, args, **kw)
        self.assertEqual(self.run_job(), 2)
        status = self.status()
        self.assertEqual((status["state"], status["stage"]), ("cancelled", None))
        self.assertEqual(self.spirula.commands("train"), [])
        self.assertFalse((self.runs / ".gpu.lock").exists())
        self.assertEqual(status["stages"]["sfm"]["state"], "done")       # finished before the stop
        self.assertEqual(status["stages"]["normals"]["state"], "pending")
        # and the next run goes on from there
        pipeline.Job.tool = lambda job, name, args, **kw: real(job, name, args, **kw)
        self.assertEqual(self.run_job(), 0)
        self.assertEqual(self.states()["train"], "done")
        self.assertEqual(len(self.spirula.commands("sfm", "auto")), 1)

    def test_missing_program_is_reported_before_anything_runs(self):
        config.missing_tools = lambda: ["spirula"]
        self.assertEqual(self.run_job(), 1)
        self.assertIn("setup.cmd", self.status()["error"])

    def test_a_missing_input_is_reported_by_the_ingest_stage(self):
        self.assertEqual(self.run_job(source=Path(self.tmp.name) / "nothing"), 1)
        self.assertIn("找不到输入", self.status()["error"])
        self.assertEqual(self.states()["ingest"], "failed")


class SeparationTest(PipelineCase):
    """The stages are separate: each run does only what it was asked for."""

    def test_to_sfm_then_to_train_reuses_the_finished_stages(self):
        self.assertEqual(self.run_job(template="preview", to="sfm"), 0)
        self.assertEqual([c[0] for c in self.spirula.calls], ["sfm"])
        status = self.status()
        self.assertEqual(status["state"], "done")
        self.assertEqual(status["run"]["plan"], ["ingest", "sfm", "export"])
        self.assertEqual(self.states(), {"ingest": "done", "masks": "disabled", "sfm": "done",
                                         "normals": "disabled", "dense_seed": "disabled",
                                         "train": "pending", "dense": "disabled",
                                         "mesh": "disabled", "coverage": "disabled",
                                         "export": "done"})
        self.assertFalse(self.job_file("output", "splat.ply").exists())
        self.assertEqual(set(status["outputs"]), {"report"})
        self.assertEqual(self.record("export")["inputs"], [])
        # the second invocation: ingest and sfm are done and are not run again
        self.spirula.calls.clear()
        self.assertEqual(self.run_job(to="train"), 0)
        self.assertEqual([c[0] for c in self.spirula.calls], ["train"])
        status = self.status()
        self.assertEqual(status["run"]["plan"], ["train", "export"])
        self.assertEqual(status["run"]["to"], "train")
        self.assertEqual(self.states()["train"], "done")
        self.assertEqual(self.states()["export"], "done")
        self.assertEqual(self.record("export")["inputs"], ["train"])
        self.assertTrue(self.job_file("output", "splat.ply").is_file())
        self.assertEqual(set(status["outputs"]), {"splat", "preview", "preview_lite", "report"})

    def test_a_run_with_nothing_to_do_says_so_and_runs_nothing(self):
        self.assertEqual(self.run_job(template="preview"), 0)
        self.spirula.calls.clear()
        self.assertEqual(self.run_job(), 0)
        self.assertEqual(self.spirula.calls, [])
        self.assertEqual((self.status()["state"], self.status()["run"]["plan"]), ("done", []))

    def test_a_changed_parameter_makes_that_stage_and_what_follows_stale_and_nothing_else(self):
        self.assertEqual(self.run_job(enable=["masks"]), 0)
        flow = pipeline.build_workflow(self.job_file(), None, [], [], ["train.iterations=1000"])
        pipeline.write_workflow(self.job_file(), flow)
        view = pipeline.job_view(self.job_file())
        self.assertEqual({k: v["state"] for k, v in view["stages"].items() if v["state"] != "disabled"},
                         {"ingest": "done", "masks": "done", "sfm": "done", "normals": "done",
                          "train": "stale", "coverage": "stale", "export": "stale"})
        self.spirula.calls.clear()
        self.assertEqual(pipeline.run_job(None, "job", runs_dir=self.runs, echo=None), 0)
        self.assertEqual([c[0] for c in self.spirula.calls], ["train"])

    def test_changing_the_masks_parameters_redoes_everything_after_them(self):
        self.assertEqual(self.run_job(enable=["masks"]), 0)
        self.spirula.calls.clear()
        self.assertEqual(self.run_job(sets=["masks.prompt=person; car"]), 0)
        self.assertEqual(self.order(), ["sam track", "sfm auto", "geometry", "train"])

    def test_only_runs_exactly_the_named_stage(self):
        self.assertEqual(self.run_job(template="preview", to="sfm"), 0)
        self.spirula.calls.clear()
        self.assertEqual(self.run_job(only=["sfm"]), 0)
        self.assertEqual([c[0] for c in self.spirula.calls], ["sfm"])
        self.assertEqual(self.status()["run"]["plan"], ["sfm"])

    def test_a_bad_request_is_refused_before_the_job_is_touched(self):
        self.assertEqual(self.run_job(template="preview", to="sfm"), 0)
        before = self.job_file("status.json").read_text(encoding="utf-8")
        with self.assertRaises(wf.PlanError) as ctx:
            self.run_job(only=["masks"])                       # not switched on
        self.assertEqual(ctx.exception.code, "stage_disabled")
        with self.assertRaises(wf.PlanError) as ctx:
            self.run_job(only=["train"], enable=["dense_seed"])   # its seed cloud is not made yet
        self.assertEqual(ctx.exception.code, "needs_upstream")
        self.assertEqual(self.job_file("status.json").read_text(encoding="utf-8"), before)

    def test_a_stage_whose_soft_input_is_not_done_refuses_to_run_alone(self):
        self.assertEqual(self.run_job(template="preview", to="sfm"), 0)
        with self.assertRaises(wf.PlanError) as ctx:
            self.run_job(only=["train"], enable=["normals"])
        self.assertEqual(ctx.exception.code, "needs_upstream")

    def test_export_with_no_splat_and_no_mesh_still_writes_the_report(self):
        self.assertEqual(self.run_job(template="preview", to="sfm"), 0)
        report = json.loads(self.job_file("output", "report.json").read_text(encoding="utf-8"))
        self.assertEqual(set(report["stages"]), {"ingest", "sfm"})
        self.assertEqual(report["files"], {})

    def test_reset_deletes_a_stage_and_what_only_it_produced(self):
        self.assertEqual(self.run_job(enable=["masks"]), 0)
        pipeline.reset_stage(self.job_file(), "train")
        self.assertFalse(self.has_record("train"))
        self.assertFalse(self.job_file("train").exists())
        self.assertTrue(self.job_file("dataset", "sparse").is_dir())
        self.assertTrue(self.has_record("sfm"))
        view = pipeline.job_view(self.job_file())
        states = {k: v["state"] for k, v in view["stages"].items()}
        self.assertEqual((states["train"], states["coverage"], states["export"], states["sfm"]),
                         ("pending", "stale", "stale", "done"))
        self.assertTrue(self.job_file("output", "splat.ply").is_file())   # export's, now stale
        with self.assertRaises(ValueError):
            pipeline.reset_stage(self.job_file(), "nonsense")
        pipeline.reset_stage(self.job_file(), "masks")
        self.assertFalse(self.job_file("dataset", "masks").exists())
        self.assertTrue(self.job_file("dataset", "images").is_dir())
        self.assertEqual(pipeline.job_view(self.job_file())["stages"]["sfm"]["state"], "stale")

    def test_job_view_of_a_job_that_never_ran_resolves_the_default_without_writing(self):
        (self.runs / "fresh").mkdir()
        view = pipeline.job_view(self.runs / "fresh")
        self.assertEqual(view["template"], "standard")
        self.assertEqual(set(view["stages"]), set(wf.STAGE_ORDER))
        self.assertEqual(view["stages"]["ingest"]["state"], "pending")
        self.assertEqual(view["stages"]["mesh"]["state"], "disabled")
        self.assertEqual(view["run"], {"to": None, "only": [], "force": [], "plan": []})
        self.assertEqual(list((self.runs / "fresh").iterdir()), [])

    def test_job_view_falls_back_when_the_workflow_file_is_damaged(self):
        (self.runs / "bad").mkdir()
        (self.runs / "bad" / "workflow.json").write_text("{nope", encoding="utf-8")
        self.assertEqual(pipeline.job_view(self.runs / "bad")["template"], "standard")

    def test_job_view_shows_the_live_map_while_a_run_is_active(self):
        seen = {}
        real = self.spirula.__call__

        def look(job, name, args, **kw):
            if str(args[0]) == "sfm":
                seen["view"] = pipeline.job_view(job.dir)
                seen["status"] = json.loads(job.status_path.read_text(encoding="utf-8"))
            return real(job, name, args, **kw)

        pipeline.Job.tool = lambda job, name, args, **kw: look(job, name, args, **kw)
        self.assertEqual(self.run_job(template="preview"), 0)
        stages = seen["view"]["stages"]
        self.assertEqual((stages["ingest"]["state"], stages["sfm"]["state"], stages["train"]["state"],
                          stages["export"]["state"]), ("done", "running", "queued", "queued"))
        self.assertEqual(stages["mesh"]["state"], "disabled")
        self.assertEqual(seen["view"]["run"]["plan"], ["ingest", "sfm", "train", "export"])
        self.assertEqual(seen["status"]["stage"], "sfm")
        self.assertEqual(seen["status"]["state"], "running")
        self.assertEqual(pipeline.job_view(self.job_file())["stages"]["sfm"]["state"], "done")

    def test_progress_goes_to_the_running_stage(self):
        seen = []
        real = self.spirula.__call__

        def talk(job, name, args, **kw):
            result = real(job, name, args, **kw)
            if str(args[0]) == "sfm":
                kw["on_line"]("[extract]   3/8")
                seen.append(json.loads(job.status_path.read_text(encoding="utf-8")))
            return result

        pipeline.Job.tool = lambda job, name, args, **kw: talk(job, name, args, **kw)
        self.assertEqual(self.run_job(template="preview"), 0)
        entry = seen[0]["stages"]["sfm"]
        self.assertEqual(entry["state"], "running")
        self.assertAlmostEqual(entry["progress"], 0.25 * 3 / 8, places=3)
        self.assertIn("3/8", entry["message"])
        self.assertAlmostEqual(seen[0]["stage_progress"], 0.25 * 3 / 8, places=3)

    # ---- dense and mesh

    def mesh_flow(self, **kw):
        return self.run_job(template="preview", enable=["dense", "mesh"], **kw)

    def test_dense_and_mesh_are_separate_stages_with_their_own_parameters(self):
        self.assertEqual(self.mesh_flow(sets=["dense.image_size=800", "dense.max_images=40",
                                              "dense.source_images=6", "dense.geom_consistency=false",
                                              "mesh.visual_faces=50000", "mesh.collision_faces=2000",
                                              "mesh.texture_size=2048", "mesh.drop_shards=false"]), 0)
        dense, mesh = self.dense_calls[0], self.mesh_calls[0]
        self.assertEqual((dense["image_size"], dense["max_images"], dense["source_images"],
                          dense["geom_consistency"]), (800, 40, 6, False))
        self.assertEqual((mesh["visual_faces"], mesh["collision_faces"], mesh["texture_size"],
                          mesh["drop_shards_filter"]), (50000, 2000, 2048, False))
        self.assertEqual(mesh["dense"], self.job_file("dense"))
        self.assertEqual(mesh["run_dir"], self.job_file("train", "run"))     # the splat's frame
        self.assertTrue(dense["locked"])                                      # GPU stage
        self.assertFalse(mesh["locked"])                                      # CPU only
        self.assertFalse(self.job_file("dense_work").exists())               # scratch is gone
        self.assertFalse(self.job_file("mesh").exists())
        self.assertEqual(self.status()["outputs"]["mesh"], "output/mesh.glb")
        self.assertEqual(set(self.status()["outputs"]), {"splat", "mesh", "collision", "preview",
                                                         "preview_lite", "report"})
        self.assertEqual(self.status()["stages"]["mesh"]["summary"], {"mesh_faces": 10,
                                                                      "collision_faces": 4})
        self.assertEqual(self.status()["stages"]["dense"]["summary"], {"dense_points": 4321,
                                                                       "photos": 5,
                                                                       "dense_photos_dropped": 1})
        self.assertEqual(self.record("export")["inputs"], ["mesh", "train"])

    def test_the_meshes_stand_upright_through_the_root_node(self):
        self.assertEqual(self.mesh_flow(), 0)
        data = self.job_file("output", "mesh.glb").read_bytes()
        doc = json.loads(data[20:20 + struct.unpack("<I", data[12:16])[0]])
        # +Z up in the file's numbers becomes +Y up through the root node.
        self.assertEqual(doc["nodes"][0]["rotation"], [-0.707106781, 0.0, 0.0, 0.707106781])
        report = json.loads(self.job_file("output", "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["orientation"]["rotation_to_y_up_xyzw"], doc["nodes"][0]["rotation"])
        self.assertEqual(report["files"]["mesh"]["path"], "output/mesh.glb")

    def test_changing_a_mesh_parameter_reruns_only_the_mesh(self):
        self.assertEqual(self.mesh_flow(), 0)
        self.assertEqual((len(self.dense_calls), len(self.mesh_calls)), (1, 1))
        self.spirula.calls.clear()
        self.assertEqual(self.run_job(sets=["mesh.visual_faces=100000"], only=["mesh"]), 0)
        self.assertEqual((len(self.dense_calls), len(self.mesh_calls)), (1, 2))
        self.assertEqual(self.spirula.calls, [])
        self.assertEqual(self.mesh_calls[1]["visual_faces"], 100000)
        self.assertEqual(self.status()["run"]["plan"], ["mesh"])
        self.assertEqual(self.states()["mesh"], "done")
        self.assertEqual(self.states()["export"], "stale")      # it listed the old mesh
        # a plain run then refreshes the export and nothing else
        self.assertEqual(self.run_job(), 0)
        self.assertEqual((len(self.dense_calls), len(self.mesh_calls)), (1, 2))
        self.assertEqual(self.status()["run"]["plan"], ["export"])

    def test_changing_a_dense_parameter_stales_the_mesh_too(self):
        self.assertEqual(self.mesh_flow(), 0)
        self.assertEqual(self.run_job(sets=["dense.image_size=1280"]), 0)
        self.assertEqual((len(self.dense_calls), len(self.mesh_calls)), (2, 2))
        self.assertEqual(self.status()["run"]["plan"], ["dense", "mesh", "export"])

    def test_a_mesh_failure_fails_the_run_and_keeps_what_finished_before_it(self):
        self.mesh_fail = "表面提取失败"
        self.assertEqual(self.mesh_flow(), 1)
        status = self.status()
        self.assertEqual(status["state"], "failed")
        self.assertEqual(status["stages"]["mesh"]["state"], "failed")
        self.assertIn("表面提取失败", status["stages"]["mesh"]["error"])
        self.assertEqual(status["stages"]["dense"]["state"], "done")
        self.assertEqual(status["stages"]["train"]["state"], "done")
        self.assertEqual(status["stages"]["export"]["state"], "pending")      # the run stopped
        self.assertEqual(status["warnings"], [])                              # nothing was skipped
        self.mesh_fail = ""
        self.assertEqual(self.run_job(), 0)
        self.assertEqual(len(self.dense_calls), 1)                            # not repeated
        self.assertEqual(self.states()["export"], "done")

    def test_a_mesh_without_a_splat_is_made_in_the_photos_frame(self):
        self.assertEqual(self.run_job(template="mesh_only", enable=["masks"], sets=["masks.kind=moving"]), 0)
        self.assertEqual(self.mesh_calls[0]["run_dir"], None)
        self.assertEqual(self.spirula.commands("train"), [])
        self.assertEqual(self.states()["train"], "disabled")
        self.assertEqual(set(self.status()["outputs"]), {"mesh", "collision", "report"})
        self.assertTrue(self.mesh_calls[0]["use_masks"])
        self.assertTrue(self.dense_calls[0]["use_masks"])

    def test_masks_are_not_given_to_the_mesh_steps_when_they_are_off(self):
        self.assertEqual(self.mesh_flow(), 0)
        self.assertFalse(self.mesh_calls[0]["use_masks"])
        self.assertFalse(self.dense_calls[0]["use_masks"])

    def test_dense_and_mesh_are_planned_in_order(self):
        self.assertEqual(self.mesh_flow(to="mesh"), 0)
        self.assertEqual(self.status()["run"]["plan"],
                         ["ingest", "sfm", "train", "dense", "mesh", "export"])

    # ---- the command line

    def cli(self, *argv):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = scan3d.main(list(argv))
        return code, err.getvalue()

    def test_cli_run_with_flags_and_exit_codes(self):
        code, _ = self.cli("run", str(self.photos), "--name", "job", "--template", "preview",
                           "--to", "sfm", "--set", "sfm.quality=low", "--set", "ingest.min_images=4")
        self.assertEqual(code, 0)
        saved = json.loads(self.job_file("workflow.json").read_text(encoding="utf-8"))
        self.assertEqual((saved["template"], saved["stages"]["sfm"]["params"]["quality"]), ("preview", "low"))
        # a second invocation reads the saved workflow; flags change it and it is saved back
        code, _ = self.cli("run", "--name", "job", "--to", "train", "--set", "train.iterations=2000",
                           "--enable", "normals", "--disable", "coverage")
        self.assertEqual(code, 0)
        saved = json.loads(self.job_file("workflow.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["template"], "preview")
        self.assertEqual(saved["stages"]["train"]["params"]["iterations"], 2000)
        self.assertTrue(saved["stages"]["normals"]["enabled"])
        self.assertEqual([c[0] for c in self.spirula.calls], ["sfm", "geometry", "train"])
        self.assertEqual(self.states()["train"], "done")

    def test_cli_bad_requests_exit_with_64_and_say_why(self):
        code, err = self.cli("run", str(self.photos), "--name", "job", "--set", "train.iterations=3")
        self.assertEqual(code, 64)
        self.assertIn("训练步数", err)
        code, err = self.cli("run", str(self.photos), "--name", "job", "--set", "nonsense")
        self.assertEqual(code, 64)
        code, err = self.cli("run", str(self.photos), "--name", "job", "--template", "nope")
        self.assertEqual(code, 64)
        self.assertIn("没有这个模板", err)
        code, err = self.cli("run", str(self.photos), "--name", "job", "--disable", "sfm")
        self.assertEqual(code, 64)
        code, err = self.cli("run", str(self.photos), "--name", "job", "--to", "mesh")
        self.assertEqual(code, 64)
        self.assertIn("没有打开", err)
        code, err = self.cli("run", str(self.photos), "--name", "job", "--only", "train")
        self.assertEqual(code, 64)
        self.assertIn("需要先完成", err)
        self.assertEqual(self.spirula.calls, [])

    def test_cli_failure_and_stage_table(self):
        self.spirula.registered = [2, 3]
        code, _ = self.cli("run", str(self.photos), "--name", "job", "--template", "preview",
                           "--set", "ingest.min_images=4")
        self.assertEqual(code, 1)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(scan3d.main(["stages", "--job", "job"]), 0)
        text = out.getvalue()
        self.assertIn("ingest", text)
        sfm = [l for l in text.splitlines() if l.startswith("sfm")][0]
        self.assertIn("failed", sfm)
        self.assertEqual(self.cli("stages", "--job", "nope")[0], 64)

    # ---- building a workflow

    def test_build_workflow_uses_the_default_the_saved_one_and_the_flags(self):
        job = self.job_file()
        flow = pipeline.build_workflow(job, None, [], [], [])
        self.assertEqual(flow["template"], "standard")
        flow = pipeline.build_workflow(job, "preview", ["normals"], [], ["train.iterations=2000"])
        self.assertEqual(flow["template"], "preview")
        self.assertTrue(flow["stages"]["normals"]["enabled"])
        again = pipeline.build_workflow(job, None, [], [], [])
        self.assertEqual(again, flow)                       # read back, not re-resolved
        switched = pipeline.build_workflow(job, "high", [], [], [])
        self.assertEqual(switched["template"], "high")
        self.assertFalse(switched["stages"]["dense"]["enabled"])
        self.assertEqual(json.loads((job / "workflow.json").read_text(encoding="utf-8")), switched)
        with self.assertRaises(wf.WorkflowError) as ctx:
            pipeline.build_workflow(job, None, ["nonsense"], [], ["sfm.quality=ultra"])
        self.assertEqual(len(ctx.exception.problems), 2)
        with self.assertRaises(wf.WorkflowError):
            pipeline.build_workflow(job, None, [], [], ["bad"])
        self.assertEqual(json.loads((job / "workflow.json").read_text(encoding="utf-8")), switched)

    def test_write_workflow_checks_what_it_stores(self):
        flow = wf.resolve("preview")
        flow["stages"]["train"]["params"]["iterations"] = 5
        with self.assertRaises(wf.WorkflowError):
            pipeline.write_workflow(self.runs / "x", flow)
        self.assertFalse((self.runs / "x" / "workflow.json").exists())
        flow["stages"]["train"]["params"]["iterations"] = 5000
        pipeline.write_workflow(self.runs / "x", flow)
        self.assertEqual(json.loads((self.runs / "x" / "workflow.json").read_text(encoding="utf-8")), flow)
        self.assertEqual([p.name for p in (self.runs / "x").iterdir()], ["workflow.json"])


if __name__ == "__main__":
    unittest.main()
