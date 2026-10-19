"""Accept must add a history row, or say plainly that it did not.

Regression: a parrot JPG freshly opened in darktable 5.6 (only the four
automatic rows, change_timestamp -1, empty masks_history) took an Accept of
"Lift the parrot a little and crisp its feathers", reported success and moved
the plan on, with no new history item.
"""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

from ai_dt.drawn_mask import splice_shapes
from ai_dt.features import _leave_off, _load_recipe, restore_waiting_step
from ai_dt.mask_steps import spec_from_forms

ROOT = Path(__file__).resolve().parents[1]
PARROT = ROOT / "tests" / "fixtures" / "dt56_parrot_fresh_jpg.xmp"
LUA = (ROOT / "ai_editor.lua").read_text(encoding="utf-8")


class FreshJpgSpliceTests(unittest.TestCase):
    def test_the_fixture_is_the_fresh_open_case(self) -> None:
        doc = PARROT.read_text(encoding="utf-8")
        self.assertIn('darktable:history_end="4"', doc)
        self.assertIn('change_timestamp="-1"', doc)
        self.assertNotIn('operation="exposure"', doc)

    def test_mask_animal_splices_a_fifth_masked_row(self) -> None:
        forms = [{"type": "path", "points": [[0.4, 0.3], [0.6, 0.3], [0.6, 0.7], [0.4, 0.7]], "border": 0.005, "mode": "union"}]
        spec = spec_from_forms("mask_animal", forms, action="lift")
        out = splice_shapes(
            PARROT.read_text(encoding="utf-8"), forms=spec["forms"], operation=spec["operation"],
            params=bytes.fromhex(spec["params"]), modversion=spec["modversion"], name=spec["name"], invert=spec["invert"],
        )
        self.assertEqual(re.search(r'darktable:history_end="(\d+)"', out).group(1), "5")
        self.assertEqual(out.count("darktable:operation="), 5)
        self.assertIn("darktable:mask_num", out)
        self.assertIn(f'darktable:multi_name="{spec["name"]}"', out)


class NoMaskIsAFailureTests(unittest.TestCase):
    def test_a_left_off_step_comes_back_on_restore(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            queue = [{"id": "mask_animal", "action": "lift", "title": "Lift the parrot"},
                     {"id": "mask_animal_background", "action": "darken", "title": "Darken around the parrot"}]
            recipe = {"plan": {"step_queue": queue, "features_path": [q["id"] for q in queue]},
                      "features_next": "mask_animal", "features_next_title": "Lift the parrot", "features_next_action": "lift"}
            (out / "output.json").write_text(json.dumps(recipe), encoding="utf-8")
            directory, path, recipe, plan = _load_recipe(out)
            _leave_off(directory, path, recipe, plan, "mask_animal", "No animal outline was found.")
            moved = json.loads((out / "output.json").read_text(encoding="utf-8"))
            self.assertNotEqual(moved["features_next"], "mask_animal")
            back = restore_waiting_step(out)
            self.assertEqual(back["features_next"], "mask_animal")
            self.assertIn("No animal outline was found.", back["feature_note"])
            self.assertNotIn("mask_animal", back["plan"].get("features_skipped") or [])


class LuaChecksTests(unittest.TestCase):
    def test_accept_checks_the_history_grew(self) -> None:
        self.assertIn("local before = request_ui.history_count(image)", LUA)
        self.assertIn("after <= before", LUA)
        self.assertIn("Accept wrote nothing", LUA)

    def test_a_step_without_mask_or_style_is_a_failure(self) -> None:
        self.assertIn('step_error = "This step was not added: "', LUA)

    def test_the_sidecar_is_synced_before_the_splice(self) -> None:
        body = LUA[LUA.index("apply_masked_step = function"):]
        body = body[: body.index("\nend\n")]
        self.assertLess(body.index("request_ui.sync_sidecar(image)"), body.index('"masked.xmp"'))
        self.assertIn("rows_after <= rows_before", body)


if __name__ == "__main__":
    unittest.main()


class FirstAskDoneStateTests(unittest.TestCase):
    def test_first_ask_drops_a_done_mark_the_sidecar_does_not_hold(self) -> None:
        import shutil

        from ai_dt.pipeline import analyze
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            image = out / "parrot.jpg"
            Image.new("RGB", (64, 48), (90, 140, 60)).save(image)
            stored = {"reading": {"sentence": ""}, "plan": {"features_done": ["subject"], "step_log": [
                {"id": "subject", "name": "Subject lift", "title": "Lift the parrot", "active": True}]}}
            (out / "output.json").write_text(json.dumps(stored), encoding="utf-8")
            xmp = out / "parrot.jpg.xmp"
            shutil.copy(PARROT, xmp)
            recipe = analyze(image, detector="heuristic", advisor="histogram", output_dir=out,
                             steer="warmer please", history_xmp=str(xmp))
            self.assertNotIn("subject", recipe["plan"].get("features_done") or [])
            # Control: without a sidecar the saved mark is kept.
            (out / "output.json").write_text(json.dumps(stored), encoding="utf-8")
            kept = analyze(image, detector="heuristic", advisor="histogram", output_dir=out,
                           steer="warmer please")
            self.assertIn("subject", kept["plan"].get("features_done") or [])
