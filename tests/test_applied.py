"""A new plan does not repeat steps already on the photo. No network, no keys.

Duck report: after "Lift the subject" was accepted, Ask proposed subject,
mask_background and foreground again. The planner is now told what is
applied and skipped, and a repeated step is filtered from the queue.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_dt import reasoner
from ai_dt.applied import (
    applied_prompt_text,
    applied_steps,
    drop_repeated,
    history_names_from_xmp,
    wants_redo,
)
from ai_dt.features import plan_chat

SIDECAR = """<x:xmpmeta><rdf:RDF><rdf:Description darktable:history_end="3">
<darktable:history><rdf:Seq>
<rdf:li darktable:num="0" darktable:operation="exposure" darktable:multi_name=""/>
<rdf:li darktable:num="1" darktable:operation="exposure" darktable:multi_name="Subject lift"/>
<rdf:li darktable:num="2" darktable:operation="colorbalancergb" darktable:multi_name="Quiet the background"/>
<rdf:li darktable:num="3" darktable:operation="colorbalancergb" darktable:multi_name="Grade the sky"/>
</rdf:Seq></darktable:history></rdf:Description></rdf:RDF></x:xmpmeta>
"""


def _duck_recipe(log: list[dict] | None = None, skipped: list[str] | None = None) -> dict:
    return {
        "features_next": "",
        "reading": {"sentence": "A duck on a pond."},
        "scene": {"summary": "A mallard duck swimming on a pond.", "source": "mistral"},
        "analysis": {"atmosphere": {}},
        "separation": {"stages_ran": ["grade"]},
        "plan": {
            "guide_stages": ["grade", "subject"],
            "skip": [],
            "features_done": [item["id"] for item in (log or [])],
            "features_skipped": list(skipped or []),
            "features_path": [],
            "step_log": list(log or []),
            "reasoning": {},
            "asks": [],
        },
    }


def _reply(*ids: str) -> dict:
    return {
        "ask_ollama": False,
        "proposal": {
            "status": "gemini",
            "backend": "gemini-3.5-flash-lite",
            "paragraph": "Lift the duck, quiet the background, darken the foreground.",
            "stages": [],
            "operations": [{"id": ident, "action": "include", "reason": ident} for ident in ids],
            "features": [],
        },
    }


SUBJECT_LOG = [{"id": "subject", "name": "Subject lift", "title": "Lift the subject", "action": "lift", "active": True}]


class AppliedStepTests(unittest.TestCase):
    def test_history_names_stop_at_history_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            xmp = Path(tmp) / "duck.NEF.xmp"
            xmp.write_text(SIDECAR, encoding="utf-8")
            self.assertEqual(history_names_from_xmp(xmp), ["", "Subject lift", "Quiet the background"])
        self.assertIsNone(history_names_from_xmp(Path("/nonexistent/x.xmp")))

    def test_applied_steps_come_from_history_and_log(self) -> None:
        plan = {"step_log": [{"id": "eyes", "name": "Brighten the eyes", "title": "Brighten the eyes", "active": True}]}
        found = applied_steps(plan, ["", "Subject lift", "Quiet the background", "My own exposure"])
        self.assertEqual([item["id"] for item in found], ["subject", "mask_background", "eyes"])
        # Without a sidecar, the plan's own record is used.
        self.assertEqual([item["id"] for item in applied_steps({"features_done": ["foreground"]})], ["foreground"])

    def test_repeated_step_is_dropped_unless_the_request_asks_to_redo_it(self) -> None:
        applied = [{"id": "subject", "name": "Subject lift", "title": "Lift the duck"}]
        queue = [{"id": "subject"}, {"id": "mask_animal"}, {"id": "mask_background"}, {"id": "foreground"}]
        kept, dropped = drop_repeated(queue, applied, [], "make it nicer")
        self.assertEqual([item["id"] for item in kept], ["mask_background", "foreground"])
        self.assertEqual(dropped, ["subject", "mask_animal"])
        kept, _ = drop_repeated(queue, applied, [], "lift the duck more")
        self.assertEqual(kept[0]["id"], "subject")
        kept, _ = drop_repeated(queue, applied, [], "do the subject again")
        self.assertEqual(kept[0]["id"], "subject")
        self.assertFalse(wants_redo("make the sky more dramatic", applied[0]))

    def test_skipped_step_stays_out_unless_named(self) -> None:
        queue = [{"id": "mask_background"}, {"id": "foreground"}]
        kept, _ = drop_repeated(queue, [], ["mask_background"], "warmer please")
        self.assertEqual([item["id"] for item in kept], ["foreground"])
        kept, _ = drop_repeated(queue, [], ["mask_background"], "darken the background after all")
        self.assertEqual([item["id"] for item in kept], ["mask_background", "foreground"])

    def test_prompt_lists_applied_and_skipped(self) -> None:
        text = applied_prompt_text([{"id": "subject", "name": "Subject lift"}], ["foreground"])
        self.assertIn("Already applied to this photo", text)
        self.assertIn("subject (Subject lift)", text)
        self.assertIn("Skipped by the photographer", text)
        self.assertIn("foreground", text)
        self.assertEqual(applied_prompt_text([], []), "")

    def test_model_prompt_carries_the_edit_state(self) -> None:
        prompt = reasoner._fill("A duck.", "", "", "make it pop", applied="Edit state:\nAlready applied: subject")
        self.assertIn("Already applied: subject", prompt)


class PlanChatRepeatTests(unittest.TestCase):
    def test_mock_planner_repeating_the_subject_gets_it_filtered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "output.json").write_text(json.dumps(_duck_recipe(SUBJECT_LOG)), encoding="utf-8")
            with patch("ai_dt.reasoner.propose_gemini", return_value=_reply("subject", "mask_background", "foreground")) as ask:
                plan_chat(directory, "finish the edit")
            prompt_state = ask.call_args.kwargs["applied"]
            self.assertIn("subject (Subject lift)", prompt_state)
            saved = json.loads((directory / "output.json").read_text(encoding="utf-8"))
            self.assertNotEqual(saved["features_next"], "subject")
            self.assertNotIn("subject", saved["plan"]["features_path"])
            self.assertIn("mask_background", saved["plan"]["features_path"])
            self.assertIn("subject", saved["plan"]["features_done"])
            self.assertIn("Already on this photo", saved["feature_note"])

    def test_the_sidecar_history_counts_as_applied(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            xmp = directory / "duck.NEF.xmp"
            xmp.write_text(SIDECAR, encoding="utf-8")
            (directory / "output.json").write_text(json.dumps(_duck_recipe()), encoding="utf-8")
            with patch("ai_dt.reasoner.propose_gemini", return_value=_reply("subject", "mask_background", "foreground")) as ask:
                plan_chat(directory, "finish the edit", history_xmp=str(xmp))
            self.assertIn("mask_background (Quiet the background)", ask.call_args.kwargs["applied"])
            saved = json.loads((directory / "output.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["plan"]["features_path"], ["foreground"])

    def test_skipped_steps_reach_the_prompt_and_stay_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            recipe = _duck_recipe(SUBJECT_LOG, skipped=["mask_background"])
            (directory / "output.json").write_text(json.dumps(recipe), encoding="utf-8")
            with patch("ai_dt.reasoner.propose_gemini", return_value=_reply("mask_background", "foreground")) as ask:
                plan_chat(directory, "finish the edit")
            self.assertIn("Skipped by the photographer in this session: mask_background", ask.call_args.kwargs["applied"])
            saved = json.loads((directory / "output.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["features_next"], "foreground")

    def test_a_plan_of_only_repeats_says_so(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "output.json").write_text(json.dumps(_duck_recipe(SUBJECT_LOG)), encoding="utf-8")
            with patch("ai_dt.reasoner.propose_gemini", return_value=_reply("subject")):
                plan_chat(directory, "finish the edit")
            saved = json.loads((directory / "output.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["features_next"], "")
            self.assertIn("already on this photo", saved["feature_note"])

    def test_asking_to_strengthen_keeps_the_repeat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "output.json").write_text(json.dumps(_duck_recipe(SUBJECT_LOG)), encoding="utf-8")
            with patch("ai_dt.reasoner.propose_gemini", return_value=_reply("subject")):
                plan_chat(directory, "lift the subject again, stronger")
            saved = json.loads((directory / "output.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["features_next"], "subject")


if __name__ == "__main__":
    unittest.main()
