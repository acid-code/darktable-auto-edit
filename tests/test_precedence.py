"""User words, then locks from the picture, then the model, then the guide."""

from __future__ import annotations

import unittest

from ai_dt.guides.decide import Picture
from ai_dt.precedence import resolve_plan
from ai_dt.schema import Detection
from ai_dt.steer import empty_plan


def _portrait() -> Picture:
    return Picture(
        type="portrait",
        reason="A face is the clear subject, so this is a portrait.",
        brightness=0.45,
        warmth=0.05,
        saturation=0.1,
        edges=0.04,
        line_fraction=0.01,
        sky=0.0,
        skin=0.4,
        highlight_specks=0.0,
        compact_area=0.15,
        compact_fill=0.7,
        subject_box=(0.3, 0.2, 0.3, 0.5),
    )


class PrecedenceTests(unittest.TestCase):
    def test_a_found_face_keeps_the_eyes_a_reasoner_wanted_to_skip(self) -> None:
        plan = empty_plan(["grade", "subject", "eyes", "skin", "background"])
        reading = type("Reading", (), {"sentence": "A face in daylight.", "regions": []})()
        proposal = {
            "status": "gemini",
            "backend": "gemini-test",
            "paragraph": "Skip the eyes.",
            "stages": [{"stage": "eyes", "action": "skip", "reason": "The eyes are already clear."}],
        }
        resolved = resolve_plan(
            plan,
            proposal=proposal,
            steer="",
            look="",
            feature_look="",
            picture=_portrait(),
            reading=reading,
            detections=[Detection("face", "face", 0.9, (0.3, 0.2, 0.3, 0.5), "test")],
        )
        self.assertNotIn("eyes", resolved["skip"])

    def test_a_user_skip_of_the_eyes_stays(self) -> None:
        plan = empty_plan(["grade", "subject", "eyes", "skin", "background"])
        reading = type("Reading", (), {"sentence": "A face in daylight.", "regions": []})()
        proposal = {
            "status": "gemini",
            "backend": "gemini-test",
            "paragraph": "Lift the eyes.",
            "stages": [{"stage": "eyes", "action": "lift", "reason": "The eyes need clarity."}],
        }
        resolved = resolve_plan(
            plan,
            proposal=proposal,
            steer="skip eyes",
            look="",
            feature_look="",
            picture=_portrait(),
            reading=reading,
            detections=[Detection("face", "face", 0.9, (0.3, 0.2, 0.3, 0.5), "test")],
        )
        self.assertIn("eyes", resolved["skip"])
        self.assertEqual(resolved["skip_by"].get("eyes"), "user")

    def test_red_aprons_stay_a_step_beside_the_subject_lift(self) -> None:
        plan = empty_plan(["grade", "subject", "eyes"])
        plan["features_done"] = ["subject"]
        reading = type(
            "Reading",
            (),
            {"sentence": "Two stone fox statues with red aprons.", "regions": []},
        )()
        proposal = {
            "status": "gemini",
            "backend": "gemini-test",
            "paragraph": (
                "the request asks to highlight the red aprons and make the foxes more "
                "prominent, which requires a color-zones lift for the red aprons and a "
                "subject lift for the foxes keeping the night dark."
            ),
            "stages": [
                {"stage": "subject", "action": "lift", "reason": "make the foxes more prominent"}
            ],
            "operations": [
                {"id": "exposure", "action": "lift", "reason": "base exposure for the scene"},
                {"id": "colorzones", "action": "lift", "reason": "highlight the red aprons"},
                {"id": "subject", "action": "lift", "reason": "make the foxes more prominent"},
                {"id": "sigmoid", "action": "include", "reason": "faithful display transform"},
            ],
        }
        resolved = resolve_plan(
            plan,
            proposal=proposal,
            steer=(
                "the main theme are the two fozes and their red aprons. "
                "make the aprons highlight and the foxes more promenent"
            ),
            look="",
            feature_look="",
            picture=_portrait(),
            reading=reading,
            detections=[],
            scene="Two stone fox statues with red aprons.",
        )
        self.assertEqual(resolved["features_path"], ["colorzones", "subject"])


if __name__ == "__main__":
    unittest.main()
