"""Tests for the weights catalogue: what the settings need, licences, and downloads.

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


class ModelsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self._saved = (config.MODELS, models._ACCEPTED_FILE, proc.run)
        config.MODELS = root / "models"
        config.MODELS.mkdir()
        models._ACCEPTED_FILE = root / "accepted-licences.json"
        self.settings = config.load_settings()
        self.downloads: list[str] = []

        def fake_run(cmd, **kw):
            # Stands in for curl: the output path follows "-o", the address is the last argument.
            Path(cmd[cmd.index("-o") + 1]).write_bytes(b"weights")
            self.downloads.append(cmd[-1])
            return proc.RunResult(0, 0.0)

        proc.run = fake_run

    def tearDown(self):
        config.MODELS, models._ACCEPTED_FILE, proc.run = self._saved
        self.tmp.cleanup()

    def install(self, *names):
        for name in names:
            (config.MODELS / name).write_bytes(b"x")

    def enable(self, **switches):
        for level in self.settings["quality"].values():
            level.update(switches)

    # ---- the shipped settings

    def test_every_file_the_shipped_settings_name_is_in_the_catalogue(self):
        for feature, info in models.feature_models(self.settings).items():
            for name in info["files"]:
                self.assertIn(name, models.CATALOGUE, feature)

    def test_every_catalogue_licence_is_known_and_every_address_is_https(self):
        for name, entry in models.CATALOGUE.items():
            self.assertTrue(entry["url"].startswith("https://"), name)
            self.assertTrue(entry["licence"] is None or entry["licence"] in models.LICENCES, name)

    def test_unknown_file_in_settings_is_reported(self):
        self.settings["normals"]["model_file"] = "nope.onnx"
        with self.assertRaises(ValueError):
            models.feature_models(self.settings)

    # ---- licences

    def test_nothing_is_accepted_on_a_fresh_installation(self):
        self.assertEqual(models.accepted_licences(), set())

    def test_accepting_is_recorded_with_the_licence_address_and_survives_rereading(self):
        models.accept_licences(["sam3"])
        self.assertEqual(models.accepted_licences(), {"sam3"})
        saved = json.loads(models._ACCEPTED_FILE.read_text(encoding="utf-8"))
        self.assertEqual(saved["accepted"]["sam3"]["url"], models.LICENCES["sam3"]["url"])
        first_date = saved["accepted"]["sam3"]["accepted_at"]
        models.accept_licences(["sam3", "dinov3"])
        saved = json.loads(models._ACCEPTED_FILE.read_text(encoding="utf-8"))
        self.assertEqual(saved["accepted"]["sam3"]["accepted_at"], first_date)
        self.assertEqual(models.accepted_licences(), {"sam3", "dinov3"})

    def test_unknown_licence_id_is_refused(self):
        with self.assertRaises(ValueError):
            models.accept_licences(["sam3", "gpl"])
        self.assertEqual(models.accepted_licences(), set())

    def test_damaged_record_counts_as_nothing_accepted(self):
        models._ACCEPTED_FILE.write_text("{not json", encoding="utf-8")
        self.assertEqual(models.accepted_licences(), set())
        models.accept_licences(["sam2"])
        self.assertEqual(models.accepted_licences(), {"sam2"})

    def test_the_tool_is_told_about_a_licence_only_after_it_was_accepted(self):
        self.assertEqual(models.licence_args("sam3-f16.ggml"), [])
        models.accept_licences(["sam3"])
        self.assertEqual(models.licence_args("sam3-f16.ggml"), ["--accept-license", "sam3=yes"])
        # Weights without a licence step, and licences the tool does not ask about.
        self.assertEqual(models.licence_args("moge2-vitb-normal.onnx"), [])
        models.accept_licences(["dinov3"])
        self.assertEqual(models.licence_args("romav2.0.1.pt"), [])

    def test_weights_are_usable_only_when_installed_and_accepted(self):
        self.assertFalse(models.usable("sam3-f16.ggml"))
        self.install("sam3-f16.ggml")
        self.assertFalse(models.usable("sam3-f16.ggml"))
        models.accept_licences(["sam3"])
        self.assertTrue(models.usable("sam3-f16.ggml"))
        self.assertFalse(models.usable("moge2-vitb-normal.onnx"))
        self.install("moge2-vitb-normal.onnx")
        self.assertTrue(models.usable("moge2-vitb-normal.onnx"))

    # ---- status

    def test_status_separates_licences_to_accept_from_files_to_fetch(self):
        self.enable(masks=True, normals=True, dense_seed=False)
        state = models.status(self.settings)
        masks, normals = state["features"]["masks"], state["features"]["normals"]
        self.assertEqual((masks["ready"], masks["licences_needed"], masks["files_missing"]),
                         (False, ["sam3"], []))
        self.assertEqual((normals["ready"], normals["licences_needed"]), (False, []))
        self.assertEqual(normals["files_missing"], [self.settings["normals"]["model_file"]])
        self.assertTrue(state["licences"]["sam3"]["needed"])
        self.assertFalse(state["licences"]["sam3"]["accepted"])
        # Only what can be fetched right now is counted as missing.
        self.assertNotIn("sam3-f16.ggml", state["files_missing"])
        self.assertIn(self.settings["normals"]["model_file"], state["files_missing"])
        self.assertNotIn("flag", state["licences"]["sam3"])

    def test_a_feature_that_is_off_is_listed_but_asks_for_nothing(self):
        self.enable(dense_seed=False)
        state = models.status(self.settings)
        dense = state["features"]["dense_seed"]
        self.assertFalse(dense["enabled"])
        self.assertEqual(dense["licences_needed"], ["dinov3"])
        self.assertFalse(state["licences"]["dinov3"]["needed"])
        self.assertNotIn("romav2.0.1.pt", state["files_missing"])
        self.assertGreater(dense["download_mb"], 0)

    def test_everything_installed_and_accepted_is_ready(self):
        self.enable(masks=True, normals=True, dense_seed=True, subject_only=True)
        models.accept_licences(list(models.LICENCES))
        for info in models.feature_models(self.settings).values():
            self.install(*info["files"])
        state = models.status(self.settings)
        self.assertEqual(state["files_missing"], [])
        self.assertEqual(state["download_mb"], 0)
        self.assertTrue(all(f["ready"] for f in state["features"].values()))

    def test_a_detector_named_in_settings_belongs_to_the_masks_feature(self):
        self.settings["masks"].update(model_file="sam2.1_hiera_large_f16.ggml",
                                      detector_file="grounding-dino-base.safetensors")
        files = models.feature_models(self.settings)["masks"]["files"]
        self.assertEqual(files, ["sam2.1_hiera_large_f16.ggml", "grounding-dino-base.safetensors"])

    # ---- downloads

    def test_download_fetches_enabled_features_and_skips_unaccepted_licences(self):
        self.enable(masks=True, normals=True, dense_seed=False)
        done = models.download_missing(self.settings, log_path=Path(self.tmp.name) / "dl.log",
                                       echo=None)
        self.assertIn(self.settings["normals"]["model_file"], done)
        self.assertNotIn("sam3-f16.ggml", done)
        self.assertNotIn("romav2.0.1.pt", done)
        for name in done:
            self.assertTrue((config.MODELS / name).is_file())
        self.assertEqual(sorted(self.downloads), sorted(models.CATALOGUE[n]["url"] for n in done))
        self.assertEqual(list((config.MODELS / "_part").iterdir()), [])
        # A second call has nothing left to do.
        self.assertEqual(models.download_missing(self.settings, echo=None,
                                                 log_path=Path(self.tmp.name) / "dl.log"), [])

    def test_download_of_a_named_feature_that_is_off(self):
        self.enable(dense_seed=False)
        log = Path(self.tmp.name) / "dl.log"
        self.assertEqual(models.download_missing(self.settings, features=["dense_seed"],
                                                 log_path=log, echo=None), [])
        models.accept_licences(["dinov3"])
        self.assertEqual(models.download_missing(self.settings, features=["dense_seed"],
                                                 log_path=log, echo=None), ["romav2.0.1.pt"])
        with self.assertRaises(ValueError):
            models.download_missing(self.settings, features=["telepathy"], log_path=log, echo=None)


if __name__ == "__main__":
    unittest.main()
