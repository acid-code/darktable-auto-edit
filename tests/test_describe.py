"""Opening a photo can write a description without planning or painting."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ai_dt.pipeline import analyze
from tests.test_guides import _save_array


def _flat_field() -> np.ndarray:
    return np.full((80, 120, 3), (0.32, 0.46, 0.20), dtype=np.float32)


class DescribeOnlyTests(unittest.TestCase):
    def test_a_description_is_stored_without_asking_for_a_plan(self) -> None:
        def boom(*_args, **_kwargs):
            raise AssertionError("describing a photo must not plan")

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            path = folder / "frame.png"
            _save_array(_flat_field(), path)
            work = folder / "work"
            with patch("ai_dt.pipeline.propose_stages", boom):
                recipe = analyze(
                    path,
                    detector="heuristic",
                    advisor="histogram",
                    describe_only=True,
                    output_dir=work,
                )
            self.assertTrue(str(recipe["scene"]["summary"]).strip())
            self.assertFalse(recipe.get("offer_looks"))
            self.assertFalse((work / "edit.dtstyle").exists())
            saved = json.loads((work / "output.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["scene"]["summary"], recipe["scene"]["summary"])

    def test_an_existing_plan_keeps_its_steps_when_the_description_is_added(self) -> None:
        def boom(*_args, **_kwargs):
            raise AssertionError("describing a photo must not plan")

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            path = folder / "frame.png"
            _save_array(_flat_field(), path)
            work = folder / "work"
            work.mkdir()
            (work / "output.json").write_text(
                json.dumps(
                    {
                        "plan": {"text": "Keep this plan."},
                        "features_next": "subject",
                        "status_text": "The plan is already here.",
                    }
                ),
                encoding="utf-8",
            )
            with patch("ai_dt.pipeline.propose_stages", boom):
                analyze(
                    path,
                    detector="heuristic",
                    advisor="histogram",
                    describe_only=True,
                    output_dir=work,
                )
            saved = json.loads((work / "output.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["plan"]["text"], "Keep this plan.")
            self.assertEqual(saved["features_next"], "subject")
            self.assertEqual(saved["status_text"], "The plan is already here.")
            self.assertTrue(str(saved["scene"]["summary"]).strip())
