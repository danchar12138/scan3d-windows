"""Tests for the weights catalogue: features, licences, what a workflow needs, and downloads.

Run by script path: tools\\python\\python.exe app\\tests\\test_models.py
Nothing here touches the network or the real tools\\models folder.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
import models  # noqa: E402
import proc  # noqa: E402
import workflow as wf  # noqa: E402


class ModelsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self._saved = (config.MODELS, models._ACCEPTED_FILE, proc.run, wf.load_user)
        config.MODELS = root / "models"
        config.MODELS.mkdir()
        models._ACCEPTED_FILE = root / "accepted-licences.json"
        wf.load_user = lambda: {}
        self.settings = config.load_settings()
        self.downloads: list = []
        self.log = root / "dl.log"

        def fake_run(cmd, **kw):
            # Stands in for curl: the output path follows "-o", the address is the last argument.
            Path(cmd[cmd.index("-o") + 1]).write_bytes(b"weights")
            self.downloads.append(cmd[-1])
            return proc.RunResult(0, 0.0)

        proc.run = fake_run

    def tearDown(self):
        config.MODELS, models._ACCEPTED_FILE, proc.run, wf.load_user = self._saved
        self.tmp.cleanup()

    def install(self, *names):
        for name in names:
            (config.MODELS / name).write_bytes(b"x")

    def table(self) -> dict:
        return {f["id"]: f for f in models.features_table(self.settings)["features"]}

    # ---- the shipped settings

    def test_every_file_the_shipped_settings_name_is_in_the_catalogue(self):
        for feature in models.FEATURES:
            files = models.feature_files(feature, self.settings)
            self.assertTrue(files, feature)
            for name in files:
                self.assertIn(name, models.CATALOGUE, feature)

    def test_every_catalogue_licence_is_known_and_every_address_is_https(self):
        for name, entry in models.CATALOGUE.items():
            self.assertTrue(entry["url"].startswith("https://"), name)
            self.assertTrue(entry["licence"] is None or entry["licence"] in models.LICENCES, name)
        for lic in models.LICENCES.values():
            self.assertTrue(lic["url"].startswith("https://"))

    def test_unknown_file_in_settings_is_reported(self):
        self.settings["models"]["normals_vitb"]["model"] = "nope.onnx"
        with self.assertRaises(ValueError):
            models.feature_files("normals_vitb", self.settings)
        with self.assertRaises(ValueError):
            models.feature_files("telepathy", self.settings)

    def test_every_stage_feature_exists_in_the_workflow_schema(self):
        for stage, d in wf.STAGES.items():
            for feature in d.get("features", []):
                self.assertIn(feature, models.FEATURES, stage)
                self.assertEqual(models.FEATURES[feature]["stage"], stage)
        offered = {f for d in wf.STAGES.values() for f in d.get("features", [])}
        self.assertEqual(offered, set(models.FEATURES))

    # ---- licences

    def test_nothing_is_recorded_on_a_fresh_installation(self):
        self.assertEqual(models.accepted_licences(), set())
        self.assertEqual(models.licence_args("sam3-f16.ggml"), [])

    def test_recording_keeps_the_address_the_date_and_how_it_came_about(self):
        models.accept_licences(["sam3"], via="feature-enabled")
        self.assertEqual(models.accepted_licences(), {"sam3"})
        saved = json.loads(models._ACCEPTED_FILE.read_text(encoding="utf-8"))
        entry = saved["accepted"]["sam3"]
        self.assertEqual((entry["url"], entry["via"]), (models.LICENCES["sam3"]["url"], "feature-enabled"))
        self.assertRegex(entry["accepted_at"], r"^\d{4}-\d\d-\d\d$")
        models.accept_licences(["sam3", "dinov3"])
        saved = json.loads(models._ACCEPTED_FILE.read_text(encoding="utf-8"))
        self.assertEqual(saved["accepted"]["sam3"], entry)         # the first record stays
        self.assertEqual(saved["accepted"]["dinov3"]["via"], "user")
        self.assertEqual(models.accepted_licences(), {"sam3", "dinov3"})

    def test_unknown_licence_id_is_refused_and_nothing_is_written(self):
        with self.assertRaises(ValueError):
            models.accept_licences(["sam3", "gpl"])
        self.assertEqual(models.accepted_licences(), set())
        self.assertFalse(models._ACCEPTED_FILE.exists())

    def test_damaged_record_counts_as_nothing_recorded(self):
        models._ACCEPTED_FILE.write_text("{not json", encoding="utf-8")
        self.assertEqual(models.accepted_licences(), set())
        models.accept_licences(["sam2"])
        self.assertEqual(models.accepted_licences(), {"sam2"})

    def test_the_tool_is_told_about_a_licence_once_it_is_recorded(self):
        self.assertEqual(models.licence_args("sam3-f16.ggml"), [])
        models.accept_licences(["sam3"])
        self.assertEqual(models.licence_args("sam3-f16.ggml"), ["--accept-license", "sam3=yes"])
        # Weights without a licence step, and licences the tool does not ask about.
        self.assertEqual(models.licence_args("moge2-vitb-normal.onnx"), [])
        models.accept_licences(["dinov3"])
        self.assertEqual(models.licence_args("romav2.0.1.pt"), [])

    # ---- which stage needs what

    def test_stage_features_follow_the_parameters(self):
        f = models.stage_features
        self.assertEqual(f("masks", {"kind": "moving", "engine": "sam3"}), ["sam3"])
        self.assertEqual(f("masks", {"kind": "moving", "engine": "sam2"}), ["sam2"])
        self.assertEqual(f("masks", {"kind": "subject", "engine": "sam3"}), ["subject"])
        self.assertEqual(f("normals", {"model": "vitb"}), ["normals_vitb"])
        self.assertEqual(f("normals", {"model": "vitl"}), ["normals_vitl"])
        self.assertEqual(f("dense_seed", {"preset": "balanced"}), ["dense_seed"])
        self.assertEqual(f("sfm", {"learned_fallback": True}), ["aliked"])
        self.assertEqual(f("sfm", {"learned_fallback": False}), [])
        for stage in ("ingest", "train", "dense", "mesh", "coverage", "export"):
            self.assertEqual(f(stage, {}), [])

    def test_stage_files_are_the_files_of_those_features(self):
        self.assertEqual(models.stage_files("masks", {"kind": "moving", "engine": "sam2"}, self.settings),
                         ["sam2.1_hiera_large_f16.ggml", "grounding-dino-base.safetensors"])
        self.assertEqual(models.stage_files("sfm", {"learned_fallback": True}, self.settings),
                         ["aliked-n16rot.onnx", "aliked-lightglue.onnx"])
        self.assertEqual(models.stage_files("train", {}, self.settings), [])

    # ---- the table the page shows

    def test_features_table_lists_files_licences_and_readiness(self):
        self.install("sam3-f16.ggml")
        table = self.table()
        sam3 = table["sam3"]
        self.assertEqual((sam3["stage"], sam3["ready"]), ("masks", True))
        self.assertEqual(sam3["files"], [{"name": "sam3-f16.ggml", "mb": 1753, "installed": True}])
        self.assertEqual(sam3["licence"], {"id": "sam3", "label": models.LICENCES["sam3"]["name"],
                                           "url": models.LICENCES["sam3"]["url"], "consent": "auto",
                                           "accepted": False})
        sam2 = table["sam2"]
        self.assertEqual([l["id"] for l in sam2["licences"]], ["sam2", "gdino"])
        self.assertEqual(sam2["licence"]["id"], "sam2")
        self.assertFalse(sam2["ready"])
        self.assertIsNone(table["normals_vitb"]["licence"])          # no licence step at all
        self.assertIsNone(table["aliked"]["licence"])
        models.accept_licences(["sam3"])
        self.assertTrue(self.table()["sam3"]["licence"]["accepted"])
        self.assertEqual(models.features_table(self.settings)["disk_mb"], 0)    # a 1-byte stand-in
        (config.MODELS / "sam3-f16.ggml").write_bytes(b"x" * 3 * 1048576)
        self.assertEqual(models.features_table(self.settings)["disk_mb"], 3)

    def test_needs_lists_licences_and_downloads_of_the_stages_that_will_run(self):
        flow = wf.resolve("standard")                 # masks (SAM 3) and normals (ViT-B), coverage
        need = models.needs(flow, self.settings)
        self.assertEqual([l["id"] for l in need["licences"]], ["sam3"])
        self.assertEqual(need["licences"][0]["stage"], "masks")
        self.assertEqual(need["licences"][0]["url"], models.LICENCES["sam3"]["url"])
        self.assertNotIn("licences_needed", need)
        wanted = {m["file"]: (m["stage"], m["mb"]) for m in need["models_needed"]}
        self.assertEqual(wanted, {"sam3-f16.ggml": ("masks", 1753),
                                  "moge2-vitb-normal.onnx": ("normals", 400),
                                  "aliked-n16rot.onnx": ("sfm", 3), "aliked-lightglue.onnx": ("sfm", 44)})
        self.install("sam3-f16.ggml", "aliked-n16rot.onnx", "aliked-lightglue.onnx")
        need = models.needs(flow, self.settings)
        self.assertEqual([m["file"] for m in need["models_needed"]], ["moge2-vitb-normal.onnx"])
        self.assertEqual([l["id"] for l in need["licences"]], ["sam3"])      # informational, always

    def test_needs_follows_the_switches_and_parameters(self):
        flow = wf.resolve("preview", {"sfm": {"params": {"learned_fallback": False}}})
        self.assertEqual(models.needs(flow, self.settings), {"licences": [], "models_needed": []})
        flow = wf.resolve("preview", {"dense_seed": {"enabled": True},
                                      "masks": {"enabled": True, "params": {"kind": "subject"}}})
        need = models.needs(flow, self.settings)
        self.assertEqual(sorted(l["id"] for l in need["licences"]), ["birefnet", "dinov3"])
        self.assertEqual(sorted(m["stage"] for m in need["models_needed"]),
                         ["dense_seed", "masks", "sfm", "sfm"])

    # ---- the start-of-stage step

    def test_a_stage_records_its_licences_and_downloads_its_weights(self):
        params = {"kind": "moving", "engine": "sam3", "prompt": "person"}
        seen = []
        done = models.ensure_stage_models("masks", params, log_path=self.log, settings=self.settings,
                                          on_progress=lambda f, m: seen.append((f, m)), echo=None)
        self.assertEqual(done["licences"], ["sam3"])
        self.assertEqual(done["downloaded"], ["sam3-f16.ggml"])
        self.assertEqual(models.accepted_licences(), {"sam3"})
        self.assertEqual(json.loads(models._ACCEPTED_FILE.read_text(encoding="utf-8"))
                         ["accepted"]["sam3"]["via"], "feature-enabled")
        self.assertEqual(self.downloads, [models.CATALOGUE["sam3-f16.ggml"]["url"]])
        self.assertTrue((config.MODELS / "sam3-f16.ggml").is_file())
        self.assertEqual(list((config.MODELS / "_part").iterdir()), [])
        self.assertTrue(seen and "SAM 3" in seen[0][1])
        # the next time there is nothing to record or fetch
        again = models.ensure_stage_models("masks", params, log_path=self.log,
                                           settings=self.settings, echo=None)
        self.assertEqual((again["licences"], again["downloaded"]), ([], []))
        self.assertEqual(len(self.downloads), 1)

    def test_a_stage_without_weights_does_nothing(self):
        done = models.ensure_stage_models("train", {}, log_path=self.log, echo=None,
                                          settings=self.settings)
        self.assertEqual(done, {"files": [], "licences": [], "downloaded": []})
        self.assertFalse(models._ACCEPTED_FILE.exists())

    def test_weights_without_a_licence_are_only_downloaded(self):
        done = models.ensure_stage_models("normals", {"model": "vitb"}, log_path=self.log,
                                          settings=self.settings, echo=None)
        self.assertEqual((done["licences"], done["downloaded"]), ([], ["moge2-vitb-normal.onnx"]))
        self.assertFalse(models._ACCEPTED_FILE.exists())

    def test_dinov3_is_recorded_like_the_others(self):
        models.ensure_stage_models("dense_seed", {"preset": "balanced"}, log_path=self.log,
                                   settings=self.settings, echo=None)
        self.assertEqual(models.accepted_licences(), {"dinov3"})
        self.assertEqual(models.licence_args("romav2.0.1.pt"), [])

    def test_two_licences_of_one_feature_are_both_recorded(self):
        models.ensure_stage_models("masks", {"kind": "moving", "engine": "sam2"}, log_path=self.log,
                                   settings=self.settings, echo=None)
        self.assertEqual(models.accepted_licences(), {"sam2", "gdino"})

    # ---- downloads

    def test_download_of_named_features(self):
        done = models.download_features(["normals_vitl", "aliked"], log_path=self.log,
                                        settings=self.settings, echo=None)
        self.assertEqual(done, ["moge2-vitl-normal.onnx", "aliked-n16rot.onnx", "aliked-lightglue.onnx"])
        self.assertEqual(sorted(self.downloads), sorted(models.CATALOGUE[n]["url"] for n in done))
        self.assertEqual(models.accepted_licences(), set())           # recorded only when used
        self.assertEqual(models.download_features(["aliked"], log_path=self.log,
                                                  settings=self.settings, echo=None), [])
        with self.assertRaises(ValueError):
            models.download_features(["telepathy"], log_path=self.log, settings=self.settings, echo=None)

    def test_download_with_no_names_fetches_what_the_default_template_uses(self):
        wanted = set(models.default_features(self.settings))
        self.assertTrue({"sam3", "normals_vitb", "aliked"} <= wanted)
        self.assertNotIn("dense_seed", wanted)
        done = models.download_features(None, log_path=self.log, settings=self.settings, echo=None)
        self.assertIn("sam3-f16.ggml", done)
        self.assertNotIn("romav2.0.1.pt", done)

    def test_progress_runs_over_the_whole_download(self):
        seen = []
        models.download_files(["moge2-vitb-normal.onnx", "aliked-n16rot.onnx"], log_path=self.log,
                             echo=None, on_progress=lambda f, m: seen.append((round(f, 3), m)))
        self.assertEqual([m for _, m in seen][0].split("（")[0], "下载模型 MoGe-2 表面法线估计")
        self.assertEqual(seen[0][0], 0.0)
        self.assertGreater(seen[1][0], 0.9)                          # 400 of 403 MB are done


if __name__ == "__main__":
    unittest.main()
