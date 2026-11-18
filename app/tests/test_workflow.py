"""Tests for the workflow description: templates, resolving, validation, fingerprints, states and
run plans. Pure logic; no files except a temporary user-settings.json.

Run by script path: tools\\python\\python.exe app\\tests\\test_workflow.py
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
import workflow as wf  # noqa: E402

NO_USER: dict = {}


def resolve(template=None, overrides=None, user=None):
    return wf.resolve(template, overrides, user=NO_USER if user is None else user)


def records_for(flow: dict, stages, signature: str = "sig", export_inputs=None) -> dict:
    """Records as the pipeline writes them for these stages, with current fingerprints."""
    fps = wf.fingerprints(flow, signature)
    out = {}
    for sid in stages:
        out[sid] = {"fingerprint": fps[sid], "result": {}}
        if sid == "export":
            out[sid]["inputs"] = list(export_inputs or [])
    return out


def states(flow, done, signature="sig", export_inputs=None):
    return wf.stage_states(flow, records_for(flow, done, signature, export_inputs), signature)


def plan_of(flow, st, **kw):
    return [(s["stage"], s["reason"]) for s in wf.plan(flow, st, **kw)]


class SchemaTest(unittest.TestCase):
    def test_stages_are_in_dependency_order_and_refer_to_known_stages(self):
        self.assertEqual(tuple(wf.STAGES), wf.STAGE_ORDER)
        for index, (sid, d) in enumerate(wf.STAGES.items()):
            for dep in d["deps"] + d["soft"]:
                self.assertIn(dep, wf.STAGES)
                self.assertLess(wf.STAGE_ORDER.index(dep), index, f"{sid} after {dep}")
            self.assertTrue(set(d["deps"]).isdisjoint(d["soft"]))
            self.assertIsInstance(d["version"], int)

    def test_the_stages_that_cannot_be_switched_off(self):
        self.assertEqual([s for s, d in wf.STAGES.items() if not d["optional"]],
                         ["ingest", "sfm", "export"])

    def test_gpu_stages(self):
        self.assertEqual([s for s, d in wf.STAGES.items() if d["gpu"]],
                         ["masks", "sfm", "normals", "dense_seed", "train", "dense"])

    def test_every_default_is_valid_for_its_own_parameter(self):
        for sid, d in wf.STAGES.items():
            for p in d["params"]:
                self.assertEqual(set(p) >= {"key", "label", "type", "default", "help", "advanced"}, True)
                self.assertIn(p["type"], ("bool", "int", "float", "choice", "text"))
                if p["type"] == "choice":
                    self.assertIn(p["default"], [c["value"] for c in p["choices"]], (sid, p["key"]))
                if "min" in p:
                    self.assertGreaterEqual(p["default"], p["min"], (sid, p["key"]))
                    self.assertLessEqual(p["default"], p["max"], (sid, p["key"]))

    def test_choices_match_what_the_tools_accept(self):
        # `spirula sfm auto --quality {low|medium|high|extreme}`, `spirula dense --preset
        # fast|balanced|high|custom`, `spirula train --primitive {3dgs|mip|3dgut}`
        choices = lambda s, k: [c["value"] for p in wf.STAGES[s]["params"] if p["key"] == k
                                for c in p["choices"]]
        self.assertEqual(choices("sfm", "quality"), ["extreme", "high", "medium", "low"])
        self.assertEqual(choices("dense_seed", "preset"), ["fast", "balanced", "high"])
        self.assertEqual(choices("train", "primitive"), ["mip", "3dgs"])
        self.assertEqual(choices("masks", "engine"), ["sam3", "sam2"])
        self.assertEqual(choices("masks", "kind"), ["moving", "subject"])
        self.assertEqual(choices("normals", "model"), ["vitb", "vitl"])

    def test_schema_is_json_able_and_complete(self):
        schema = json.loads(json.dumps(wf.schema()))
        self.assertEqual([s["id"] for s in schema["stages"]], list(wf.STAGE_ORDER))
        self.assertEqual(schema["default_template"], "standard")
        train = [s for s in schema["stages"] if s["id"] == "train"][0]
        self.assertEqual([p["key"] for p in train["params"]],
                         ["max_image_size", "iterations", "cap_max", "primitive", "eval_holdout",
                          "extra_args"])
        self.assertEqual(train["soft"], ["masks", "normals", "dense_seed"])
        self.assertTrue(all("label" in s and "desc" in s and "gpu" in s for s in schema["stages"]))
        masks = [s for s in schema["stages"] if s["id"] == "masks"][0]
        self.assertEqual(masks["features"], ["sam3", "sam2", "subject"])
        schema["stages"][0]["params"][0]["default"] = -1          # a copy: the module is untouched
        self.assertEqual(wf.STAGES["ingest"]["params"][0]["default"], 20)


class TemplateTest(unittest.TestCase):
    def test_every_builtin_template_resolves_to_a_complete_valid_workflow(self):
        for tid in wf.TEMPLATES:
            flow = resolve(tid)
            self.assertEqual(flow["template"], tid)
            self.assertEqual(list(flow["stages"]), list(wf.STAGE_ORDER))
            for sid, st in flow["stages"].items():
                self.assertEqual(set(st["params"]), {p["key"] for p in wf.STAGES[sid]["params"]})
            self.assertEqual(wf.validate(json.loads(json.dumps(flow))), flow)

    def test_what_each_template_switches_on(self):
        on = lambda t: [s for s in wf.STAGE_ORDER if resolve(t)["stages"][s]["enabled"]]
        self.assertEqual(on("preview"), ["ingest", "sfm", "train", "export"])
        self.assertEqual(on("standard"), ["ingest", "masks", "sfm", "normals", "train", "coverage", "export"])
        self.assertEqual(on("high"), on("standard"))
        self.assertEqual(on("full"), ["ingest", "masks", "sfm", "normals", "train", "dense", "mesh",
                                      "coverage", "export"])
        self.assertEqual(on("mesh_only"), ["ingest", "masks", "sfm", "dense", "mesh", "export"])

    def test_template_numbers(self):
        p = lambda t, s, k: resolve(t)["stages"][s]["params"][k]
        self.assertEqual((p("preview", "train", "iterations"), p("preview", "train", "cap_max"),
                          p("preview", "train", "max_image_size")), (7000, 500000, 1280))
        self.assertEqual((p("standard", "train", "iterations"), p("standard", "train", "cap_max")),
                         (30000, 2000000))
        self.assertEqual((p("high", "train", "iterations"), p("high", "train", "max_image_size")),
                         (50000, 3072))
        self.assertEqual(p("full", "mesh", "visual_faces"), 300000)

    def test_the_list_has_the_builtin_templates_then_the_users(self):
        user = {"templates": {"mine": {"label": "我的", "base": "preview", "stages": {}}}}
        listed = wf.templates(user)
        self.assertEqual([t["id"] for t in listed], list(wf.TEMPLATES) + ["mine"])
        self.assertTrue(all(t["builtin"] for t in listed[:-1]))
        self.assertEqual(listed[-1], {"id": "mine", "label": "我的", "desc": "", "builtin": False,
                                      "base": "preview"})

    def test_a_user_template_builds_on_its_base(self):
        user = {"templates": {"mine": {"label": "我的", "base": "preview",
                                       "stages": {"train": {"params": {"iterations": 9000}},
                                                  "normals": {"enabled": True}}}}}
        flow = resolve("mine", user=user)
        self.assertEqual(flow["template"], "mine")
        self.assertEqual(flow["stages"]["train"]["params"]["iterations"], 9000)
        self.assertEqual(flow["stages"]["train"]["params"]["cap_max"], 500000)   # preview's
        self.assertTrue(flow["stages"]["normals"]["enabled"])
        self.assertFalse(flow["stages"]["masks"]["enabled"])

    def test_user_templates_can_build_on_each_other(self):
        user = {"templates": {
            "a": {"base": "high", "stages": {"train": {"params": {"iterations": 12345}}}},
            "b": {"base": "a", "stages": {"coverage": {"enabled": False}}}}}
        flow = resolve("b", user=user)
        self.assertEqual(flow["stages"]["train"]["params"]["iterations"], 12345)
        self.assertEqual(flow["stages"]["train"]["params"]["max_image_size"], 3072)    # high's
        self.assertFalse(flow["stages"]["coverage"]["enabled"])

    def test_a_user_template_with_a_builtin_id_replaces_it(self):
        user = {"templates": {"preview": {"label": "我的预览", "base": "standard",
                                          "stages": {"masks": {"enabled": False}}}}}
        self.assertEqual([t for t in wf.templates(user) if t["id"] == "preview"][0]["label"], "我的预览")
        flow = resolve("preview", user=user)
        self.assertEqual(flow["stages"]["train"]["params"]["iterations"], 30000)      # standard's
        self.assertFalse(flow["stages"]["masks"]["enabled"])

    def test_a_template_that_names_itself_or_an_unknown_base_does_not_loop(self):
        user = {"templates": {"x": {"base": "x", "stages": {}},
                              "y": {"base": "ghost", "stages": {"train": {"params": {"iterations": 2000}}}}}}
        self.assertEqual(resolve("x", user=user)["stages"]["train"]["params"]["iterations"], 30000)
        flow = resolve("y", user=user)
        self.assertEqual(flow["stages"]["train"]["params"]["iterations"], 2000)
        self.assertTrue(flow["stages"]["masks"]["enabled"])                           # standard

    def test_the_default_comes_from_the_user_else_standard(self):
        self.assertEqual(resolve()["template"], "standard")
        self.assertEqual(resolve(user={"default_template": "full"})["template"], "full")
        self.assertEqual(resolve(user={"default_template": "mine",
                                       "templates": {"mine": {"stages": {}}}})["template"], "mine")

    def test_unknown_template_is_refused(self):
        with self.assertRaises(wf.WorkflowError) as ctx:
            resolve("nope")
        self.assertIn("nope", ctx.exception.problems[0])

    def test_a_user_template_with_a_bad_value_is_refused(self):
        user = {"templates": {"bad": {"base": "preview",
                                      "stages": {"train": {"params": {"iterations": 5}}}}}}
        with self.assertRaises(wf.WorkflowError):
            resolve("bad", user=user)

    def test_user_settings_are_read_and_written_beside_the_program(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved, config.ROOT = config.ROOT, Path(tmp)
            try:
                self.assertEqual(wf.load_user(), {})
                wf.save_user({"default_template": "full", "templates": {}})
                self.assertEqual(wf.load_user()["default_template"], "full")
                self.assertEqual(wf.resolve()["template"], "full")
                (Path(tmp) / "user-settings.json").write_text("{damaged", encoding="utf-8")
                self.assertEqual(wf.load_user(), {})
                self.assertEqual(wf.resolve()["template"], "standard")
            finally:
                config.ROOT = saved

    def test_user_settings_file_is_ignored_by_git(self):
        ignore = (Path(__file__).resolve().parents[2] / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("user-settings.json", ignore.split())


class ValidateTest(unittest.TestCase):
    def test_overrides_change_enabled_and_params(self):
        flow = resolve("preview", {"normals": {"enabled": True}, "train": {"params": {"iterations": 8000}}})
        self.assertTrue(flow["stages"]["normals"]["enabled"])
        self.assertEqual(flow["stages"]["train"]["params"]["iterations"], 8000)

    def problems(self, overrides=None, mutate=None, template="preview"):
        with self.assertRaises(wf.WorkflowError) as ctx:
            if mutate is None:
                resolve(template, overrides)
            else:
                flow = resolve(template)
                mutate(flow)
                wf.validate(flow)
        return ctx.exception.problems

    def test_unknown_stage_parameter_and_key(self):
        self.assertIn("没有这个阶段：ghost", self.problems({"ghost": {"enabled": True}}))
        self.assertTrue(any("没有这个参数 nope" in p for p in self.problems({"train": {"params": {"nope": 1}}})))
        self.assertTrue(any("不认识 colour" in p for p in self.problems({"train": {"colour": 1}})))
        self.assertTrue(any("必须是对象" in p for p in self.problems({"train": 5})))

    def test_values_are_checked_against_their_parameter(self):
        cases = [({"train": {"params": {"iterations": 10}}}, "训练步数"),
                 ({"train": {"params": {"iterations": 1.5}}}, "必须是整数"),
                 ({"train": {"params": {"iterations": "many"}}}, "必须是数字"),
                 ({"train": {"params": {"iterations": True}}}, "必须是数字"),
                 ({"train": {"params": {"primitive": "voxel"}}}, "不在可选项里"),
                 ({"train": {"params": {"eval_holdout": 1}}}, "开或关"),
                 ({"train": {"params": {"extra_args": 3}}}, "必须是文字"),
                 ({"sfm": {"params": {"min_registered_fraction": 5.0}}}, "之间"),
                 ({"sfm": {"params": {"quality": "ultra"}}}, "不在可选项里"),
                 ({"normals": {"enabled": "yes"}}, "开关必须是")]
        for override, word in cases:
            with self.subTest(override):
                self.assertTrue(any(word in p for p in self.problems(override)), override)

    def test_every_problem_is_listed_at_once(self):
        found = self.problems({"train": {"params": {"iterations": 10, "primitive": "x"}},
                               "ghost": {}})
        self.assertGreaterEqual(len(found), 3)

    def test_a_number_given_as_a_float_is_coerced(self):
        flow = resolve("preview", {"train": {"params": {"iterations": 8000.0}}})
        self.assertIsInstance(flow["stages"]["train"]["params"]["iterations"], int)
        self.assertEqual(flow["stages"]["sfm"]["params"]["min_registered_fraction"], 0.6)
        flow = resolve("preview", {"sfm": {"params": {"min_registered_fraction": 1}}})
        self.assertEqual(flow["stages"]["sfm"]["params"]["min_registered_fraction"], 1)

    def test_a_choice_may_be_a_number(self):
        flow = resolve("preview", {"train": {"params": {"max_image_size": 0}}})
        self.assertEqual(flow["stages"]["train"]["params"]["max_image_size"], 0)
        self.assertTrue(self.problems({"train": {"params": {"max_image_size": 1000}}}))

    def test_the_minimum_number_of_photos_has_its_bounds(self):
        flow = resolve("preview", {"ingest": {"params": {"min_images": 3}}})
        self.assertEqual(flow["stages"]["ingest"]["params"]["min_images"], 3)
        self.assertEqual(resolve("preview")["stages"]["ingest"]["params"]["min_images"], 20)
        for bad in (2, 201):
            self.assertTrue(any("最少照片数" in p for p in self.problems(
                {"ingest": {"params": {"min_images": bad}}})))

    def test_stages_that_must_run_cannot_be_switched_off(self):
        for sid in ("ingest", "sfm", "export"):
            self.assertTrue(any("不能关闭" in p for p in self.problems({sid: {"enabled": False}})), sid)

    def test_an_incomplete_workflow_is_refused(self):
        self.assertTrue(self.problems(mutate=lambda f: f["stages"].pop("train")))
        self.assertTrue(self.problems(mutate=lambda f: f["stages"]["train"].pop("params")))
        self.assertTrue(self.problems(mutate=lambda f: f["stages"]["train"]["params"].pop("cap_max")))
        self.assertTrue(self.problems(mutate=lambda f: f.__setitem__("stages", [])))

    def test_a_workflow_saved_before_a_parameter_existed_is_filled_in(self):
        old = resolve("preview")
        del old["stages"]["sfm"]["params"]["use_gps"]
        del old["stages"]["coverage"]
        with self.assertRaises(wf.WorkflowError):
            wf.validate(old)
        filled = wf.fill_defaults(old)
        self.assertEqual(filled, resolve("preview"))
        self.assertNotIn("use_gps", old["stages"]["sfm"]["params"])      # the original is untouched
        self.assertEqual(wf.fill_defaults({"stages": 5}), {"stages": 5})

    def test_apply_overrides_leaves_the_original_alone(self):
        flow = resolve("preview")
        changed = wf.apply_overrides(flow, {"train": {"params": {"iterations": 9000}}})
        self.assertEqual(flow["stages"]["train"]["params"]["iterations"], 7000)
        self.assertEqual(changed["stages"]["train"]["params"]["iterations"], 9000)
        with self.assertRaises(wf.WorkflowError):
            wf.apply_overrides(flow, {"train": {"params": {"iterations": 1}}})

    def test_changed_from_template_is_the_partial_override_that_gets_back(self):
        flow = resolve("preview", {"normals": {"enabled": True},
                                   "train": {"params": {"iterations": 9000, "cap_max": 500000}}})
        # `changed_from_template` resolves the template with the machine's own user settings, which
        # a test must not depend on: compare with a stand-in
        saved, wf.load_user = wf.load_user, lambda: {}
        try:
            diff = wf.changed_from_template(flow)
        finally:
            wf.load_user = saved
        self.assertEqual(diff, {"normals": {"enabled": True}, "train": {"params": {"iterations": 9000}}})
        self.assertEqual(resolve("preview", diff), flow)


class DependencyTest(unittest.TestCase):
    def test_effective_enabled_pulls_in_hard_dependencies(self):
        flow = resolve("preview", {"mesh": {"enabled": True}})
        self.assertEqual(wf.effective_enabled(flow), ["ingest", "sfm", "train", "dense", "mesh", "export"])
        self.assertFalse(flow["stages"]["dense"]["enabled"])          # the switch itself is unchanged
        flow = resolve("preview", {"coverage": {"enabled": True}, "train": {"enabled": False}})
        self.assertEqual(wf.effective_enabled(flow), ["ingest", "sfm", "train", "coverage", "export"])

    def test_nothing_optional_still_runs_ingest_sfm_and_export(self):
        flow = resolve("preview", {"train": {"enabled": False}})
        self.assertEqual(wf.effective_enabled(flow), ["ingest", "sfm", "export"])

    def test_consumed_are_the_hard_and_the_switched_on_soft_dependencies(self):
        flow = resolve("full")
        self.assertEqual(wf.consumed(flow, "train"), ["sfm", "masks", "normals"])
        self.assertEqual(wf.consumed(flow, "mesh"), ["dense", "masks", "train"])
        self.assertEqual(wf.consumed(flow, "dense"), ["sfm", "masks"])
        self.assertEqual(wf.consumed(flow, "export"), ["train", "mesh", "coverage"])
        flow = resolve("preview")
        self.assertEqual(wf.consumed(flow, "train"), ["sfm"])
        self.assertEqual(wf.consumed(flow, "export"), ["train"])
        self.assertEqual(wf.consumed(resolve("mesh_only"), "mesh"), ["dense", "masks"])

    def test_dense_seed_consumes_masks(self):
        flow = resolve("standard", {"dense_seed": {"enabled": True}})
        self.assertEqual(wf.consumed(flow, "dense_seed"), ["sfm", "masks"])
        self.assertEqual(wf.consumed(flow, "train"), ["sfm", "masks", "normals", "dense_seed"])

    def test_downstream_follows_consumption(self):
        flow = resolve("full")
        self.assertEqual(wf.downstream(flow, "masks"),
                         ["masks", "sfm", "normals", "train", "dense", "mesh", "coverage", "export"])
        self.assertEqual(wf.downstream(flow, "mesh"), ["mesh", "export"])
        self.assertEqual(wf.downstream(flow, "normals"), ["normals", "train", "mesh", "coverage", "export"])
        self.assertEqual(wf.downstream(flow, "export"), ["export"])


class FingerprintTest(unittest.TestCase):
    def fps(self, template="full", overrides=None, signature="sig"):
        return wf.fingerprints(resolve(template, overrides), signature)

    def test_stable_and_only_for_the_stages_that_run(self):
        a = self.fps("preview")
        self.assertEqual(a, self.fps("preview"))
        self.assertEqual(list(a), ["ingest", "sfm", "train", "export"])
        self.assertTrue(all(len(v) == 16 for v in a.values()))

    def test_a_parameter_changes_that_stage_and_what_consumes_it_only(self):
        a, b = self.fps("full"), self.fps("full", {"mesh": {"params": {"visual_faces": 100000}}})
        self.assertEqual([k for k in a if a[k] != b[k]], ["mesh", "export"])
        b = self.fps("full", {"train": {"params": {"iterations": 12000}}})
        self.assertEqual([k for k in a if a[k] != b[k]], ["train", "mesh", "coverage", "export"])
        b = self.fps("full", {"dense": {"params": {"image_size": 800}}})
        self.assertEqual([k for k in a if a[k] != b[k]], ["dense", "mesh", "export"])
        b = self.fps("full", {"masks": {"params": {"prompt": "person; car"}}})
        self.assertEqual([k for k in a if a[k] == b[k]], ["ingest"])

    def test_the_input_changes_everything(self):
        a, b = self.fps("full"), self.fps("full", signature="other")
        self.assertTrue(all(a[k] != b[k] for k in a))

    def test_switching_a_soft_input_on_or_off_changes_what_consumes_it(self):
        a = self.fps("preview")
        b = self.fps("preview", {"normals": {"enabled": True}})
        self.assertEqual([k for k in b if k in a and a[k] != b[k]], ["train", "export"])
        self.assertEqual(a["sfm"], b["sfm"])

    def test_a_parameter_of_a_stage_that_is_off_does_not_matter(self):
        a = self.fps("preview")
        b = self.fps("preview", {"mesh": {"params": {"visual_faces": 100000}}})
        self.assertEqual(a, b)

    def test_a_stage_pulled_in_by_a_dependency_has_a_fingerprint(self):
        self.assertIn("dense", self.fps("preview", {"mesh": {"enabled": True}}))


class StateTest(unittest.TestCase):
    def test_no_records_means_pending_and_disabled(self):
        st = states(resolve("preview"), [])
        self.assertEqual(st, {"ingest": "pending", "masks": "disabled", "sfm": "pending",
                              "normals": "disabled", "dense_seed": "disabled", "train": "pending",
                              "dense": "disabled", "mesh": "disabled", "coverage": "disabled",
                              "export": "pending"})

    def test_everything_current_is_done(self):
        flow = resolve("full")
        st = states(flow, wf.effective_enabled(flow), export_inputs=["train", "mesh", "coverage"])
        self.assertEqual({s for s in st.values()}, {"done", "disabled"})
        self.assertEqual([s for s, v in st.items() if v == "disabled"], ["dense_seed"])

    def test_a_record_with_another_fingerprint_is_stale(self):
        flow = resolve("preview")
        records = records_for(flow, ["ingest", "sfm", "train"])
        records["train"]["fingerprint"] = "0" * 16
        st = wf.stage_states(flow, records, "sig")
        self.assertEqual((st["ingest"], st["sfm"], st["train"]), ("done", "done", "stale"))

    def test_changing_a_parameter_makes_exactly_that_stage_and_its_followers_stale(self):
        flow = resolve("full")
        records = records_for(flow, wf.effective_enabled(flow), export_inputs=["train", "mesh", "coverage"])
        changed = resolve("full", {"dense": {"params": {"image_size": 800}}})
        st = wf.stage_states(changed, records, "sig")
        self.assertEqual({s: v for s, v in st.items() if v == "stale"}.keys(), {"dense", "mesh", "export"})
        self.assertEqual(st["train"], "done")
        self.assertEqual(st["coverage"], "done")

    def test_a_stage_whose_input_is_not_done_is_stale_even_with_a_matching_record(self):
        flow = resolve("preview")
        records = records_for(flow, ["ingest", "train"])          # sfm never ran
        st = wf.stage_states(flow, records, "sig")
        self.assertEqual((st["ingest"], st["sfm"], st["train"]), ("done", "pending", "stale"))

    def test_a_changed_input_makes_everything_stale(self):
        flow = resolve("preview")
        records = records_for(flow, ["ingest", "sfm", "train", "export"], export_inputs=["train"])
        st = wf.stage_states(flow, records, "another folder")
        self.assertEqual([st[s] for s in ("ingest", "sfm", "train", "export")], ["stale"] * 4)

    def test_export_is_done_only_for_the_inputs_it_saw(self):
        flow = resolve("full")
        enabled = wf.effective_enabled(flow)
        exported = [s for s in enabled if s != "export"]
        # export ran when only the splat was done; the mesh and coverage were pending
        records = records_for(flow, ["ingest", "masks", "sfm", "normals", "train", "export"],
                              export_inputs=["train"])
        st = wf.stage_states(flow, records, "sig")
        self.assertEqual((st["export"], st["mesh"]), ("done", "pending"))
        # later the mesh is made: the export no longer lists everything that exists
        records = records_for(flow, exported + ["export"], export_inputs=["train"])
        self.assertEqual(wf.stage_states(flow, records, "sig")["export"], "stale")
        records = records_for(flow, exported + ["export"], export_inputs=["train", "mesh", "coverage"])
        self.assertEqual(wf.stage_states(flow, records, "sig")["export"], "done")
        # an export that listed a stage that is not done any more
        records = records_for(flow, [s for s in exported if s != "mesh"] + ["export"],
                              export_inputs=["train", "mesh", "coverage"])
        self.assertEqual(wf.stage_states(flow, records, "sig")["export"], "stale")

    def test_export_with_nothing_to_list(self):
        flow = resolve("preview")
        records = records_for(flow, ["ingest", "sfm", "export"], export_inputs=[])
        self.assertEqual(wf.stage_states(flow, records, "sig")["export"], "done")
        records = records_for(flow, ["ingest", "sfm", "train", "export"], export_inputs=[])
        self.assertEqual(wf.stage_states(flow, records, "sig")["export"], "stale")


class PlanTest(unittest.TestCase):
    def setUp(self):
        self.full = resolve("full")
        self.all = wf.effective_enabled(self.full)
        self.none = states(self.full, [])
        self.done = states(self.full, self.all, export_inputs=["train", "mesh", "coverage"])

    def test_everything_switched_on_when_nothing_is_done(self):
        self.assertEqual(plan_of(self.full, self.none),
                         [(s, "requested") for s in self.all])

    def test_only_what_is_not_done(self):
        st = states(self.full, ["ingest", "masks", "sfm", "normals", "train"])
        self.assertEqual(plan_of(self.full, st), [("dense", "requested"), ("mesh", "requested"),
                                                  ("coverage", "requested"), ("export", "requested")])

    def test_everything_done_has_nothing_to_do(self):
        with self.assertRaises(wf.PlanError) as ctx:
            wf.plan(self.full, self.done)
        self.assertEqual(ctx.exception.code, "nothing_to_do")

    def test_to_runs_the_target_and_what_it_needs_then_export(self):
        self.assertEqual(plan_of(self.full, self.none, to="sfm"),      # sfm uses the masks
                         [("ingest", "needed"), ("masks", "needed"), ("sfm", "requested"),
                          ("export", "needed")])
        flow = resolve("preview")
        self.assertEqual(plan_of(flow, states(flow, []), to="sfm"),
                         [("ingest", "needed"), ("sfm", "requested"), ("export", "needed")])
        self.assertEqual(plan_of(self.full, self.none, to="train"),
                         [("ingest", "needed"), ("masks", "needed"), ("sfm", "needed"),
                          ("normals", "needed"), ("train", "requested"), ("export", "needed")])
        self.assertEqual(plan_of(self.full, self.none, to="mesh"),
                         [("ingest", "needed"), ("masks", "needed"), ("sfm", "needed"),
                          ("normals", "needed"), ("train", "needed"), ("dense", "needed"),
                          ("mesh", "requested"), ("export", "needed")])

    def test_to_skips_what_is_done_and_reexports_when_the_export_would_be_out_of_date(self):
        st = states(self.full, ["ingest", "masks", "sfm", "normals", "train", "coverage", "export"],
                    export_inputs=["train", "coverage"])
        self.assertEqual(plan_of(self.full, st, to="mesh"),
                         [("dense", "needed"), ("mesh", "requested"), ("export", "needed")])
        # nothing it lists is going to change: the export is current and stays
        st = states(self.full, ["ingest", "masks", "sfm", "normals", "dense", "train", "export"],
                    export_inputs=["train"])
        with self.assertRaises(wf.PlanError) as ctx:
            wf.plan(self.full, st, to="train")
        self.assertEqual(ctx.exception.code, "nothing_to_do")

    def test_to_needs_a_stage_that_is_switched_on(self):
        with self.assertRaises(wf.PlanError) as ctx:
            wf.plan(resolve("preview"), states(resolve("preview"), []), to="mesh")
        self.assertEqual(ctx.exception.code, "stage_disabled")
        self.assertIn("网格", ctx.exception.message)

    def test_unknown_stages_are_refused(self):
        for kw in ({"to": "ghost"}, {"only": ["ghost"]}, {"force": ["ghost"]}):
            with self.assertRaises(wf.PlanError) as ctx:
                wf.plan(self.full, self.none, **kw)
            self.assertEqual(ctx.exception.code, "unknown_stage")

    def test_only_runs_exactly_those_stages_even_when_done(self):
        self.assertEqual(plan_of(self.full, self.done, only=["mesh"]), [("mesh", "requested")])
        self.assertEqual(plan_of(self.full, self.done, only=["mesh", "train"]),
                         [("train", "requested"), ("mesh", "requested")])

    def test_only_needs_its_inputs_done_or_in_the_same_run(self):
        st = states(self.full, ["ingest", "masks", "sfm", "normals", "train"])
        with self.assertRaises(wf.PlanError) as ctx:
            wf.plan(self.full, st, only=["mesh"])
        self.assertEqual(ctx.exception.code, "needs_upstream")
        self.assertIn("稠密重建", ctx.exception.message)
        self.assertEqual(plan_of(self.full, st, only=["dense", "mesh"]),
                         [("dense", "requested"), ("mesh", "requested")])
        with self.assertRaises(wf.PlanError):
            wf.plan(self.full, self.none, only=["sfm"])           # ingest is not done

    def test_a_switched_on_soft_input_must_be_done_too(self):
        st = states(self.full, ["ingest", "masks", "sfm"])        # the normals are pending
        with self.assertRaises(wf.PlanError) as ctx:
            wf.plan(self.full, st, only=["train"])
        self.assertEqual(ctx.exception.code, "needs_upstream")
        self.assertIn("法线先验", ctx.exception.message)
        # a soft input that is switched off is no obstacle
        flow = resolve("preview")
        self.assertEqual(plan_of(flow, states(flow, ["ingest", "sfm"]), only=["train"]),
                         [("train", "requested")])
        # and a stale one is as bad as a missing one
        stale = dict(self.done, normals="stale")
        with self.assertRaises(wf.PlanError):
            wf.plan(self.full, stale, only=["train"])

    def test_export_takes_whatever_exists(self):
        flow = resolve("preview")
        self.assertEqual(plan_of(flow, states(flow, ["ingest", "sfm"]), only=["export"]),
                         [("export", "requested")])
        self.assertEqual(plan_of(self.full, self.none, only=["export"]),
                         [("export", "requested")])

    def test_force_runs_a_done_stage_again_and_what_follows_it_in_the_plan(self):
        self.assertEqual(plan_of(self.full, self.done, force=["train"]),
                         [("train", "forced"), ("mesh", "downstream"), ("coverage", "downstream"),
                          ("export", "downstream")])
        # `to` limits the run: what is not needed for the target is not dragged in
        steps = plan_of(self.full, self.done, force=["train"], to="train")
        self.assertEqual(steps, [("train", "forced"), ("export", "downstream")])
        steps = plan_of(self.full, self.done, force=["masks"], only=["masks"])
        self.assertEqual(steps, [("masks", "forced")])        # only this one was asked for
        steps = plan_of(self.full, states(self.full, ["ingest", "masks", "sfm", "normals", "train"]),
                        force=["normals"])
        self.assertEqual(steps[:2], [("normals", "forced"), ("train", "downstream")])

    def test_force_of_a_stage_that_is_off_is_refused(self):
        flow = resolve("preview")
        with self.assertRaises(wf.PlanError) as ctx:
            wf.plan(flow, states(flow, []), force=["mesh"])
        self.assertEqual(ctx.exception.code, "stage_disabled")

    def test_stale_stages_are_run_like_pending_ones(self):
        flow = resolve("full")
        records = records_for(flow, self.all, export_inputs=["train", "mesh", "coverage"])
        changed = resolve("full", {"mesh": {"params": {"visual_faces": 100000}}})
        st = wf.stage_states(changed, records, "sig")
        self.assertEqual(plan_of(changed, st), [("mesh", "requested"), ("export", "requested")])

    def test_a_plan_is_in_stage_order_whatever_order_was_asked_for(self):
        steps = plan_of(self.full, self.done, only=["export", "mesh", "ingest"])
        self.assertEqual([s for s, _ in steps], ["ingest", "mesh", "export"])


if __name__ == "__main__":
    unittest.main()
