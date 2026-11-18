"""Tests for the scan pipeline's orchestration, with Spirula replaced by a stand-in.

Run by script path: tools\\python\\python.exe app\\tests\\test_pipeline.py

The stand-in writes the files and prints the lines the real tool was observed to produce
(tool version 2026.11.12), so these tests cover what this program does around the tool: the
order of the steps, the arguments it passes, the files it moves, resuming, and what it tells the
user. They do not show that the real tool behaves this way; the real runs do.
"""
from __future__ import annotations

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
import models  # noqa: E402
import pipeline  # noqa: E402
import proc  # noqa: E402


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
        self.calls: list[list[str]] = []
        self.video_frames = 400
        self.video_size = (1920, 1080)
        self.frames_written = 30
        # Images the pose estimation registers, per attempt; None = all of them.
        self.registered: list = [None]
        self.fail: set[str] = set()          # commands that exit with an error
        self.masks_kept = 97.5
        self.lock_seen: list[bool] = []

    def commands(self, *prefix: str) -> list[list[str]]:
        return [c for c in self.calls if tuple(c[:len(prefix)]) == prefix]

    @staticmethod
    def value(call: list[str], flag: str) -> str:
        return call[call.index(flag) + 1]

    def __call__(self, job, name, args, *, log, timeout, on_line=None, **kw):
        assert name == "spirula"
        args = [str(a) for a in args]
        self.calls.append(args)
        self.lock_seen.append((config.RUNS / ".gpu.lock").exists())
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


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.runs = root / "runs"
        self.runs.mkdir()
        self.spirula = FakeSpirula()
        self.settings = config.load_settings()
        self.settings["limits"]["min_images"] = 4
        self._saved = (config.MODELS, config.RUNS, models._ACCEPTED_FILE, pipeline.Job.tool,
                       config.load_settings, config.missing_tools)
        config.MODELS = root / "models"
        config.MODELS.mkdir()
        config.RUNS = self.runs
        models._ACCEPTED_FILE = root / "accepted-licences.json"
        fake = self.spirula
        pipeline.Job.tool = lambda job, name, args, **kw: fake(job, name, args, **kw)
        config.load_settings = lambda: json.loads(json.dumps(self.settings))
        config.missing_tools = lambda: []
        for name in models.CATALOGUE:
            (config.MODELS / name).write_bytes(b"weights")
        models.accept_licences(list(models.LICENCES))

        self.photos = root / "input"
        self.photos.mkdir()
        for i in range(1, 9):
            write_png(self.photos / f"IMG_{i}.png", 4000, 3000)

    def tearDown(self):
        (config.MODELS, config.RUNS, models._ACCEPTED_FILE, pipeline.Job.tool,
         config.load_settings, config.missing_tools) = self._saved
        self.tmp.cleanup()

    def run_job(self, source=None, quality="standard", **kw):
        kw.setdefault("mesh", False)
        self.lines: list[str] = []
        return pipeline.run_job(source or self.photos, "job", quality, runs_dir=self.runs,
                                echo=self.lines.append, **kw)

    def job_file(self, *parts) -> Path:
        return self.runs.joinpath("job", *parts)

    def status(self) -> dict:
        return json.loads(self.job_file("status.json").read_text(encoding="utf-8"))

    def stage(self, name: str) -> dict:
        return json.loads(self.job_file("stages", name + ".json").read_text(encoding="utf-8"))

    def train_call(self) -> list[str]:
        calls = self.spirula.commands("train")
        self.assertEqual(len(calls), 1)
        return calls[0]

    # ---- the whole run

    def test_photos_run_through_every_step_and_leave_the_outputs(self):
        self.assertEqual(self.run_job(), 0)
        status = self.status()
        self.assertEqual((status["state"], status["stage"], status["error"]), ("done", "export", None))
        order = [" ".join(c[:2]) if c[0] in ("sam", "sfm") else c[0] for c in self.spirula.calls]
        self.assertEqual(order, ["sam track", "sfm auto", "geometry", "train"])
        for name in ("splat.ply", "report.json", "preview.spz", "preview-lite.spz"):
            self.assertTrue(self.job_file("output", name).is_file(), name)
        report = json.loads(self.job_file("output", "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["stages"]["ingest"]["images"], 8)
        self.assertEqual(report["stages"]["sfm"]["registered"], 8)
        self.assertEqual(report["stages"]["sfm"]["mean_reprojection_error_px"], 0.52)
        self.assertEqual(report["stages"]["train"]["gaussians"], 6)
        self.assertEqual(report["orientation"]["up_in_file_coordinates"], [0.0, 0.0, 1.0])
        self.assertEqual(set(status["outputs"]), {"splat", "preview", "preview_lite", "report"})
        self.assertEqual(report["warnings"], [])

    def test_every_gpu_step_runs_while_holding_the_lock_and_releases_it(self):
        self.assertEqual(self.run_job(), 0)
        self.assertEqual(self.spirula.lock_seen, [True] * len(self.spirula.calls))
        self.assertFalse((self.runs / ".gpu.lock").exists())
        self.assertEqual(list((self.runs / ".gpu.queue").iterdir()), [])

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
        self.assertEqual(self.run_job(), 0)
        masks = sorted(p.name for p in self.job_file("dataset", "masks", "photos").iterdir())
        self.assertEqual(masks, [f"p_{i:05d}.png" for i in range(1, 9)])
        self.assertFalse(self.job_file("dataset", "masks_raw").exists())
        call = self.spirula.commands("sam", "track")[0]
        self.assertEqual(self.spirula.value(call, "--mask-mode"), "image")
        self.assertEqual(self.spirula.value(call, "--text"), self.settings["masks"]["prompt"])
        self.assertEqual(self.spirula.value(call, "--accept-license"), "sam3=yes")
        self.assertTrue(self.spirula.value(call, "--model").endswith("sam3-f16.ggml"))
        ingest = self.stage("ingest")
        self.assertEqual((ingest["masks"], ingest["mask_kind"], ingest["masked_images"]),
                         (True, "moving", 8))
        self.assertAlmostEqual(ingest["masked_percent"], 2.5)
        self.assertEqual(self.spirula.value(self.train_call(), "--load-masks"), "1")

    def test_masks_are_left_out_with_a_warning_until_the_licence_is_accepted(self):
        models._ACCEPTED_FILE.unlink()
        self.assertEqual(self.run_job(), 0)
        self.assertEqual(self.spirula.commands("sam", "track"), [])
        self.assertFalse(self.job_file("dataset", "masks").exists())
        self.assertEqual(self.spirula.value(self.train_call(), "--load-masks"), "0")
        warnings = self.status()["warnings"]
        self.assertEqual(len(warnings), 1)
        self.assertIn("SAM License", warnings[0])

    def test_accepted_licence_with_the_weights_missing_stops_the_scan(self):
        (config.MODELS / "sam3-f16.ggml").unlink()
        self.assertEqual(self.run_job(), 1)
        self.assertIn("sam3-f16.ggml", self.status()["error"])
        self.assertEqual(self.spirula.calls, [])

    def test_a_detector_from_settings_is_passed_along(self):
        self.settings["masks"].update(model_file="sam2.1_hiera_large_f16.ggml",
                                      detector_file="grounding-dino-base.safetensors")
        self.assertEqual(self.run_job(), 0)
        call = self.spirula.commands("sam", "track")[0]
        self.assertTrue(self.spirula.value(call, "--detector").endswith("grounding-dino-base.safetensors"))
        accepted = [call[i + 1] for i, a in enumerate(call) if a == "--accept-license"]
        self.assertEqual(accepted, ["sam2=yes", "gdino=yes"])

    def test_large_masks_are_pointed_out(self):
        self.spirula.masks_kept = 40.0
        self.assertEqual(self.run_job(), 0)
        self.assertTrue(any("masks.prompt" in w for w in self.status()["warnings"]))

    def test_a_missing_mask_stops_the_scan(self):
        real = self.spirula.__call__

        def drop_one(job, name, args, **kw):
            result = real(job, name, args, **kw)
            if [str(a) for a in args[:2]] == ["sam", "track"]:
                out = Path(self.spirula.value([str(a) for a in args], "--out"))
                sorted(out.iterdir())[0].unlink()
            return result

        pipeline.Job.tool = lambda job, name, args, **kw: drop_one(job, name, args, **kw)
        self.assertEqual(self.run_job(), 1)
        self.assertIn("7/8", self.status()["error"])

    def test_preview_quality_skips_masks_and_priors(self):
        self.assertEqual(self.run_job(quality="preview"), 0)
        order = [c[0] for c in self.spirula.calls]
        self.assertEqual(order, ["sfm", "train"])
        call = self.train_call()
        self.assertEqual([self.spirula.value(call, f) for f in
                          ("--load-normals", "--load-depths", "--load-masks", "--seed-pointcloud")],
                         ["0", "0", "0", "sparse"])

    def test_subject_scan_keeps_the_subject_and_trains_for_it(self):
        self.assertEqual(self.run_job(overrides={"subject_only": True}), 0)
        call = self.spirula.commands("sam", "track")[0]
        self.assertTrue(self.spirula.value(call, "--model").endswith("birefnet-general.safetensors"))
        self.assertNotIn("--text", call)
        self.assertEqual(self.spirula.value(call, "--accept-license"), "birefnet=yes")
        self.assertEqual(self.stage("ingest")["mask_kind"], "subject")
        self.assertEqual(self.train_call()[1], "centered-object")

    # ---- video

    def test_video_frames_get_their_own_folder_and_are_treated_as_a_sequence(self):
        video = Path(self.tmp.name) / "walk.mp4"
        video.write_bytes(b"\0" * 64)
        self.assertEqual(self.run_job(source=video), 0)
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
        ingest = self.stage("ingest")
        self.assertEqual((ingest["videos"], ingest["video_frames"], ingest["sequences"]),
                         (1, 30, ["v1"]))

    def test_oversized_video_is_scaled_down_at_extraction(self):
        self.spirula.video_size = (7680, 4320)
        video = Path(self.tmp.name) / "big.mov"
        video.write_bytes(b"\0" * 64)
        self.assertEqual(self.run_job(source=video, quality="preview"), 0)
        extract = self.spirula.commands("sam", "extract")[0]
        self.assertEqual(self.spirula.value(extract, "--scale"), "0.5000")

    def test_photos_and_video_together(self):
        (self.photos / "clip.mp4").write_bytes(b"\0" * 64)
        self.assertEqual(self.run_job(), 0)
        tracked = [(Path(self.spirula.value(c, "--frames")).name, self.spirula.value(c, "--mask-mode"))
                   for c in self.spirula.commands("sam", "track")]
        self.assertEqual(tracked, [("photos", "image"), ("v1", "image")])
        self.assertEqual(self.stage("ingest")["images"], 38)

    # ---- camera poses

    def test_pose_estimation_is_given_the_image_folder_not_the_dataset(self):
        self.assertEqual(self.run_job(), 0)
        call = self.spirula.commands("sfm", "auto")[0]
        self.assertEqual(Path(call[2]), self.job_file("dataset", "images"))
        self.assertEqual(Path(self.spirula.value(call, "-o")), self.job_file("dataset"))
        self.assertEqual((self.spirula.value(call, "--features"), self.spirula.value(call, "--matcher")),
                         ("sift", "bruteforce"))
        self.assertNotIn("--sequence", call)

    def test_second_feature_type_is_tried_when_the_first_registers_too_few(self):
        self.spirula.registered = [2, None]
        self.assertEqual(self.run_job(), 0)
        calls = self.spirula.commands("sfm", "auto")
        self.assertEqual([self.spirula.value(c, "--features") for c in calls], ["sift", "aliked-n16rot"])
        self.assertTrue(self.spirula.value(calls[1], "--aliked-model").endswith("aliked-n16rot.onnx"))
        self.assertTrue(self.spirula.value(calls[1], "--lightglue-model").endswith("aliked-lightglue.onnx"))
        sfm = self.stage("sfm")
        self.assertEqual((sfm["registered"], sfm["method"]), (8, "aliked-n16rot + lightglue"))

    def test_scan_fails_with_a_plain_message_when_no_attempt_registers_enough(self):
        self.spirula.registered = [2, 3]
        self.assertEqual(self.run_job(), 1)
        status = self.status()
        self.assertEqual((status["state"], status["stage"]), ("failed", "sfm"))
        self.assertIn("8 张图片里只有 3 张", status["error"])
        self.assertEqual(self.spirula.commands("train"), [])

    def test_partly_registered_capture_is_reported_as_a_warning(self):
        self.spirula.registered = [6]
        self.assertEqual(self.run_job(), 0)
        self.assertTrue(any("2 张图片未能参与重建" in w for w in self.status()["warnings"]))

    # ---- priors and training

    def test_standard_quality_tells_the_trainer_exactly_what_to_read(self):
        self.assertEqual(self.run_job(), 0)
        call = self.train_call()
        self.assertEqual([self.spirula.value(call, f) for f in
                          ("--load-normals", "--load-depths", "--load-masks", "--seed-pointcloud")],
                         ["1", "0", "1", "sparse"])
        self.assertEqual(self.spirula.value(call, "--disable-viewer"), "1")
        self.assertEqual(self.spirula.value(call, "--keep-viewer-alive"), "0")
        self.assertEqual(call[1], "3dgs")
        self.assertEqual(self.spirula.commands("dense"), [])
        self.assertEqual(self.stage("dataset")["normals"], True)

    def test_dense_seed_is_used_when_switched_on(self):
        self.assertEqual(self.run_job(overrides={"dense_seed": True}), 0)
        dense = self.spirula.commands("dense")[0]
        self.assertTrue(self.spirula.value(dense, "--checkpoint").endswith("romav2.0.1.pt"))
        self.assertEqual(self.spirula.value(self.train_call(), "--seed-pointcloud"), "dense/roma.ply")
        result = self.stage("dataset")
        self.assertEqual((result["dense_seed"], result["dense_points"]), (True, 2387575))

    def test_dense_seed_is_left_out_with_a_warning_until_its_licence_is_accepted(self):
        models._ACCEPTED_FILE.unlink()
        models.accept_licences(["sam3"])
        self.assertEqual(self.run_job(overrides={"dense_seed": True}), 0)
        self.assertEqual(self.spirula.commands("dense"), [])
        self.assertEqual(self.spirula.value(self.train_call(), "--seed-pointcloud"), "sparse")
        self.assertTrue(any("DINOv3" in w for w in self.status()["warnings"]))

    def test_working_resolution_is_a_divisor_of_the_source_size(self):
        self.assertEqual(self.run_job(), 0)           # 4000 px photos, standard works at 1920
        self.assertEqual(self.spirula.value(self.train_call(), "--train-resolution-divisor"), "2.0833")

    def test_small_photos_are_not_upscaled(self):
        for p in self.photos.iterdir():
            write_png(p, 1600, 1200)
        self.assertEqual(self.run_job(), 0)
        self.assertNotIn("--train-resolution-divisor", self.train_call())

    def test_overrides_reach_the_trainer(self):
        self.assertEqual(self.run_job(overrides={"iterations": 600, "train_args": ["--primitive", "mip"]}), 0)
        call = self.train_call()
        self.assertEqual(self.spirula.value(call, "--num-iterations"), "600")
        self.assertEqual(call[-2:], ["--primitive", "mip"])

    def test_held_out_evaluation_is_requested_only_when_asked(self):
        self.assertEqual(self.run_job(holdout=True), 0)
        self.assertEqual(self.spirula.value(self.train_call(), "--eval-mode"), "interval")

    # ---- failures, resuming, cancelling

    def test_too_few_photos_is_explained(self):
        for p in sorted(self.photos.iterdir())[3:]:
            p.unlink()
        self.assertEqual(self.run_job(), 1)
        self.assertIn("只有 3 张", self.status()["error"])
        self.assertEqual(self.spirula.calls, [])

    def test_unsupported_photo_formats_are_skipped_with_a_warning(self):
        (self.photos / "a.heic").write_bytes(b"x")
        self.assertEqual(self.run_job(), 0)
        self.assertEqual(self.stage("ingest")["skipped_unsupported"], 1)
        self.assertTrue(any("a.heic" in w for w in self.status()["warnings"]))

    def test_a_failed_step_fails_the_scan_and_a_second_run_resumes_after_the_finished_steps(self):
        self.spirula.fail.add("train 3dgs")
        self.assertEqual(self.run_job(), 1)
        self.assertEqual((self.status()["state"], self.status()["stage"]), ("failed", "train"))
        before = len(self.spirula.calls)
        self.spirula.fail.clear()
        self.assertEqual(self.run_job(), 0)
        self.assertEqual([c[0] for c in self.spirula.calls[before:]], ["train"])
        self.assertEqual(self.status()["state"], "done")

    def test_fresh_discards_earlier_results(self):
        self.assertEqual(self.run_job(), 0)
        before = len(self.spirula.calls)
        self.assertEqual(self.run_job(fresh=True), 0)
        self.assertEqual(len(self.spirula.calls), 2 * before)

    def test_cancel_file_stops_the_scan_between_steps(self):
        real = self.spirula.__call__

        def cancel_after_poses(job, name, args, **kw):
            result = real(job, name, args, **kw)
            if str(args[0]) == "sfm":
                job.cancel_file.touch()
            return result

        pipeline.Job.tool = lambda job, name, args, **kw: cancel_after_poses(job, name, args, **kw)
        self.assertEqual(self.run_job(), 2)
        self.assertEqual(self.status()["state"], "cancelled")
        self.assertEqual(self.spirula.commands("train"), [])
        self.assertFalse((self.runs / ".gpu.lock").exists())

    def test_missing_program_is_reported_before_anything_runs(self):
        config.missing_tools = lambda: ["spirula"]
        self.assertEqual(self.run_job(), 1)
        self.assertIn("setup.cmd", self.status()["error"])

    # ---- meshes

    def test_meshes_are_exported_upright_and_listed(self):
        import meshpost

        def build(*, out_dir, on_progress=None, **kw):
            self.assertEqual(set(kw["tools"]), {"spirula", "colmap"})
            self.assertTrue((config.RUNS / ".gpu.lock").exists())
            write_glb(out_dir / "mesh.glb")
            write_glb(out_dir / "collision.glb")
            return {"mesh": out_dir / "mesh.glb", "collision": out_dir / "collision.glb",
                    "mesh_faces": 10, "collision_faces": 4}

        saved, meshpost.build_meshes = meshpost.build_meshes, build
        try:
            self.assertEqual(self.run_job(mesh=True), 0)
        finally:
            meshpost.build_meshes = saved
        self.assertEqual(set(self.status()["outputs"]),
                         {"splat", "mesh", "collision", "preview", "preview_lite", "report"})
        data = self.job_file("output", "mesh.glb").read_bytes()
        doc = json.loads(data[20:20 + struct.unpack("<I", data[12:16])[0]])
        # +Z up in the file's numbers becomes +Y up through the root node.
        self.assertEqual(doc["nodes"][0]["rotation"], [-0.707106781, 0.0, 0.0, 0.707106781])
        report = json.loads(self.job_file("output", "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["orientation"]["rotation_to_y_up_xyzw"], doc["nodes"][0]["rotation"])

    def test_a_mesh_failure_keeps_the_splat(self):
        import meshpost

        def build(**kw):
            raise meshpost.MeshError("表面提取失败")

        saved, meshpost.build_meshes = meshpost.build_meshes, build
        try:
            self.assertEqual(self.run_job(mesh=True), 0)
        finally:
            meshpost.build_meshes = saved
        self.assertEqual(self.status()["state"], "done")
        self.assertTrue(any("网格生成失败" in w for w in self.status()["warnings"]))
        self.assertTrue(self.job_file("output", "splat.ply").is_file())


if __name__ == "__main__":
    unittest.main()
