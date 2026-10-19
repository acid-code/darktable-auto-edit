"""Steer grammar, the text reasoner, and the plain stage list.

The model is stubbed. These tests do not open a socket or call Ollama or Gemini.
"""

from __future__ import annotations

import json
import os
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from ai_dt.ollama_host import ollama_base_url
from ai_dt.reasoner import plain_gemini_failure, propose_stages, validate_proposal
from ai_dt.steer import (
    apply_proposal,
    empty_plan,
    format_plan_label,
    format_plan_text,
    parse_steer,
    plan_stages,
    stage_intents,
)


def _boom(*_args, **_kwargs):
    raise AssertionError("network access is not allowed in steer tests")


def _ollama_must_not_run(*_args, **_kwargs):
    raise AssertionError("ollama")


def _gemini_plan() -> dict:
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": json.dumps(
                                {
                                    "stages": [
                                        {
                                            "stage": "subject",
                                            "action": "hold",
                                            "reason": "Hold the facade.",
                                        }
                                    ]
                                }
                            )
                        }
                    ]
                }
            }
        ]
    }


def _models_called(urls: list[str]) -> list[str]:
    models = []
    for url in urls:
        start = url.find("/models/")
        tail = url[start + len("/models/") :]
        models.append(tail.split(":", 1)[0])
    return models


@contextmanager
def _no_network():
    with (
        patch("urllib.request.urlopen", _boom),
        patch("socket.create_connection", _boom),
    ):
        yield


def _env(**extra: str):
    values = {
        "GEMINI_API_KEY": "",
        "AI_EDITOR_REASONER": "",
        "OLLAMA_HOST": "127.0.0.1:11434",
    }
    values.update(extra)
    return patch.dict(os.environ, values, clear=False)


class GrammarTests(unittest.TestCase):
    def test_epid_is_unknown_and_epic_opens_caps(self) -> None:
        ignored = parse_steer("epid")
        self.assertFalse(ignored["caps_open"])
        self.assertFalse(ignored["strong"])
        self.assertEqual(ignored["ignored"], ["epid"])
        self.assertEqual(ignored["summary"], "Ignored: epid.")

        opened = parse_steer("epic, darken the subject")
        self.assertTrue(opened["caps_open"])
        self.assertTrue(opened["strong"])
        self.assertEqual(opened["strong_word"], "epic")
        self.assertEqual(opened["asks"], [{"verb": "darken", "target": "subject", "source": "user"}])

    def test_strong_and_epic_are_the_same_and_case_does_not_matter(self) -> None:
        for word in ("strong", "EPIC", "Strong", "ePiC"):
            parsed = parse_steer(word)
            self.assertTrue(parsed["caps_open"], word)
            self.assertTrue(parsed["strong"], word)
        self.assertFalse(parse_steer("   ")["caps_open"])
        self.assertFalse(parse_steer("moody")["caps_open"])

    def test_commas_and_the_word_and_split_clauses(self) -> None:
        parsed = parse_steer("DARKEN the sky and HOLD the reflection, lift the rock")
        self.assertEqual(
            [(item["verb"], item["target"]) for item in parsed["asks"]],
            [("darken", "sky"), ("hold", "reflection"), ("lift", "rock")],
        )
        self.assertFalse(parsed["caps_open"])

    def test_each_verb_accepts_each_target(self) -> None:
        for verb in ("darken", "lift", "hold"):
            for target in ("subject", "foreground", "rock", "sky", "behind", "reflection"):
                parsed = parse_steer(f"{verb} the {target}")
                self.assertEqual(
                    parsed["asks"],
                    [{"verb": verb, "target": target, "source": "user"}],
                    f"{verb} {target}",
                )
                self.assertFalse(parsed["caps_open"])

    def test_light_behind_the_subject_is_lift_behind(self) -> None:
        parsed = parse_steer("light behind the subject")
        self.assertEqual(parsed["asks"], [{"verb": "lift", "target": "behind", "source": "user"}])

    def test_gradient_is_requested_with_caps_closed(self) -> None:
        parsed = parse_steer("Gradient")
        self.assertTrue(parsed["gradient"])
        self.assertFalse(parsed["caps_open"])
        self.assertEqual(parsed["asks"], [])

    def test_revert_and_skip_name_the_stage_list(self) -> None:
        mapping = {
            "gradient": "gradient",
            "subject": "subject",
            "surround": "background",
            "vignette": "vignette",
            "theme": "theme",
            "reflection": "reflection",
            "eyes": "eyes",
            "skin": "skin",
            "sky": "sky",
        }
        sentence = ", ".join(f"skip the {word}" for word in mapping)
        parsed = parse_steer(sentence)
        self.assertEqual(parsed["skip"], list(mapping.values()))
        self.assertEqual(parse_steer("revert the eyes")["skip"], ["eyes"])
        self.assertEqual(parse_steer("REVERT the surround")["skip"], ["background"])

    def test_unknown_words_are_listed_and_a_real_clause_still_applies(self) -> None:
        parsed = parse_steer("epid, please, epic")
        self.assertTrue(parsed["caps_open"])
        self.assertEqual(parsed["ignored"], ["epid", "please"])
        self.assertEqual(parsed["summary"], "Ignored: epid, please.")

    def test_a_direct_sentence_names_the_expert_modules(self) -> None:
        parsed = parse_steer("lift the shadows, warmer, more color, and hold the highlights")
        self.assertEqual(parsed["tools"]["shadow_lift"], 0.5)
        self.assertEqual(parsed["tools"]["warmth"], 0.4)
        self.assertEqual(parsed["tools"]["color"], 0.35)
        self.assertTrue(parsed["tools"]["highlight_hold"])
        self.assertNotIn("shadows", parsed["ignored"])
        self.assertNotIn("warmer", parsed["ignored"])
        self.assertNotIn("color", parsed["ignored"])
        loose = parse_steer("please make it warmer and open the shadows")
        self.assertEqual(loose["tools"]["warmth"], 0.4)
        self.assertEqual(loose["tools"]["shadow_lift"], 0.5)
        self.assertNotIn("warmer", loose["ignored"])
        self.assertNotIn("shadows", loose["ignored"])

    def test_the_heron_sentence_is_not_an_ignored_word_list(self) -> None:
        sentence = "epic, dark end result with the bird as the main subject. Make the refelction clearer"
        parsed = parse_steer(sentence)
        self.assertTrue(parsed["caps_open"])
        asks = {(item["verb"], item["target"]) for item in parsed["asks"]}
        self.assertIn(("darken", "foreground"), asks)
        self.assertIn(("lift", "subject"), asks)
        self.assertIn(("lift", "reflection"), asks)
        for word in ("dark", "bird", "subject", "subject.", "refelction", "reflection", "make", "clearer"):
            self.assertNotIn(word, parsed["ignored"])
        spelled = parse_steer(
            "epic, dark end result with the bird as the main subject. Make the reflection clearer"
        )
        self.assertNotIn("dark", spelled["ignored"])
        self.assertNotIn("reflection", spelled["ignored"])
        for typo in ("refleciton", "refection"):
            again = parse_steer(f"make the {typo} clearer")
            self.assertEqual(again["asks"][0]["target"], "reflection")
            self.assertNotIn(typo, again["ignored"])

    def test_a_named_cat_and_its_eyes_are_the_request(self) -> None:
        parsed = parse_steer("the cat is the main theme, fix the colors, and make his eyes pop")
        asks = {(item["verb"], item["target"]) for item in parsed["asks"]}
        self.assertIn(("lift", "subject"), asks)
        self.assertIn(("lift", "eyes"), asks)
        for word in ("cat", "theme", "eyes", "pop", "make", "his"):
            self.assertNotIn(word, parsed["ignored"])
        mountain = parse_steer("treat mount fuji as the main theme")
        self.assertIn(("lift", "subject"), {(item["verb"], item["target"]) for item in mountain["asks"]})
        self.assertNotIn("fuji", mountain["ignored"])
        train = parse_steer("make the train more seen")
        self.assertIn(("lift", "subject"), {(item["verb"], item["target"]) for item in train["asks"]})
        self.assertNotIn("train", train["ignored"])


class ReasonerTests(unittest.TestCase):
    def test_off_switch_skips_without_a_request(self) -> None:
        with _no_network(), _env(AI_EDITOR_REASONER="off"):
            result = propose_stages("a bird", "moody", "chrome")
        self.assertEqual(result["status"], "skipped")
        self.assertIn("Reasoning was skipped", result["note"])
        self.assertEqual(result["stages"], [])
        self.assertNotIn("caps_open", result)

    def test_nothing_reachable_skips_and_says_so(self) -> None:
        def post(*_args, **_kwargs):
            raise AssertionError("request")

        with (
            _no_network(),
            _env(),
            patch("ai_dt.reasoner.ollama_reachable", return_value=False),
            patch("ai_dt.reasoner._post_json", post),
        ):
            result = propose_stages("a dry bird", "quiet", "none")
        self.assertEqual(result["status"], "skipped")
        self.assertIn("Reasoning was skipped", result["note"])
        self.assertIn("No local model", result["note"])

    def test_ollama_is_text_only_and_cannot_open_caps(self) -> None:
        seen: dict = {}

        def post(url, body, timeout):
            seen["url"] = url
            seen["body"] = body
            return {
                "message": {
                    "content": json.dumps(
                        {
                            "caps_open": True,
                            "strong": True,
                            "stages": [
                                {"stage": "subject", "action": "darken", "reason": "Darken the body."},
                                {"stage": "epic", "action": "strong", "reason": "Open the caps."},
                                {"stage": "surround", "action": "darken", "reason": "Quiet the surround."},
                            ],
                        }
                    )
                }
            }

        with (
            _no_network(),
            _env(OLLAMA_HOST="http://127.0.0.1:11434"),
            patch("ai_dt.reasoner.ollama_reachable", return_value=True),
            patch("ai_dt.reasoner._post_json", post),
        ):
            result = propose_stages("a bird on a rock", "moody", "chrome")
            self.assertEqual(ollama_base_url(), "http://127.0.0.1:11434")
        self.assertEqual(result["status"], "ollama")
        self.assertNotIn("caps_open", result)
        self.assertEqual([item["stage"] for item in result["stages"]], ["subject", "surround"])
        self.assertTrue(str(seen["url"]).startswith("http://127.0.0.1:11434/api/chat"))
        self.assertNotIn("images", seen["body"])
        self.assertIn("a bird on a rock", seen["body"]["messages"][0]["content"])

    def test_gemini_is_not_called_without_a_key(self) -> None:
        def gemini(*_args, **_kwargs):
            raise AssertionError("gemini")

        def post(url, body, timeout):
            return {"message": {"content": '{"stages": []}'}}

        with (
            _no_network(),
            _env(),
            patch("ai_dt.reasoner.ollama_reachable", return_value=True),
            patch("ai_dt.reasoner._post_json", post),
            patch("ai_dt.reasoner._ask_gemini", gemini),
        ):
            result = propose_stages("facade", "", "")
        self.assertEqual(result["status"], "ollama")

    def test_gemini_is_asked_before_ollama(self) -> None:
        def ollama(*_args, **_kwargs):
            raise AssertionError("ollama")

        def gemini(*_args, **_kwargs):
            return {
                "paragraph": "Hold the facade.",
                "stages": [{"stage": "subject", "action": "hold", "reason": "Hold the facade."}],
            }

        with (
            _no_network(),
            _env(GEMINI_API_KEY="test-key"),
            patch("ai_dt.reasoner._ask_ollama", ollama),
            patch("ai_dt.reasoner._ask_gemini", gemini),
        ):
            os.environ.pop("GEMINI_REASON_MODEL", None)
            result = propose_stages("facade", "", "")
        self.assertEqual(result["status"], "gemini")
        self.assertEqual(result["backend"], "gemini-3.5-flash-lite")
        self.assertEqual(result["stages"][0]["action"], "hold")
        self.assertNotIn("caps_open", result)

        with (
            _no_network(),
            _env(GEMINI_API_KEY="test-key", GEMINI_REASON_MODEL="gemini-2.5-flash"),
            patch("ai_dt.reasoner._ask_ollama", ollama),
            patch("ai_dt.reasoner._ask_gemini", gemini),
        ):
            named = propose_stages("facade", "", "")
        self.assertEqual(named["status"], "gemini")
        self.assertEqual(named["backend"], "gemini-2.5-flash")

    def test_ollama_is_the_fallback_when_gemini_has_no_plan(self) -> None:
        def no_plan(*_args, **_kwargs):
            return None

        def empty_plan(*_args, **_kwargs):
            return {"paragraph": "", "stages": []}

        def post(url, body, timeout):
            self.assertIn("/api/chat", url)
            return {
                "message": {
                    "content": json.dumps(
                        {
                            "caps_open": True,
                            "stages": [
                                {"stage": "subject", "action": "hold", "reason": "Hold the facade."}
                            ],
                        }
                    )
                }
            }

        with (
            _no_network(),
            _env(GEMINI_API_KEY="test-key"),
            patch("ai_dt.reasoner.ollama_reachable", return_value=True),
            patch("ai_dt.reasoner._post_json", post),
            patch("ai_dt.reasoner._ask_gemini", no_plan),
        ):
            missing = propose_stages("facade", "", "")
        self.assertEqual(missing["status"], "ollama")
        self.assertNotIn("caps_open", missing)
        self.assertEqual(missing["stages"][0]["action"], "hold")

        with (
            _no_network(),
            _env(GEMINI_API_KEY="test-key"),
            patch("ai_dt.reasoner.ollama_reachable", return_value=True),
            patch("ai_dt.reasoner._post_json", post),
            patch("ai_dt.reasoner._ask_gemini", empty_plan),
        ):
            empty = propose_stages("facade", "", "")
        self.assertEqual(empty["status"], "ollama")

    def test_gemini_runs_only_when_a_key_is_set(self) -> None:
        def post(url, body, timeout):
            self.assertIn("generativelanguage.googleapis.com", url)
            self.assertNotIn("images", json.dumps(body))
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": json.dumps(
                                        {
                                            "caps_open": True,
                                            "stages": [
                                                {
                                                    "stage": "sky",
                                                    "action": "hold",
                                                    "reason": "Hold the sky.",
                                                }
                                            ],
                                        }
                                    )
                                }
                            ]
                        }
                    }
                ]
            }

        with (
            _no_network(),
            _env(GEMINI_API_KEY="test-key"),
            patch("ai_dt.reasoner.ollama_reachable", return_value=False),
            patch("ai_dt.reasoner._post_json", post),
        ):
            result = propose_stages("blue hour", "quiet", "none")
        self.assertEqual(result["status"], "gemini")
        self.assertEqual(result["stages"][0]["action"], "hold")
        plan = apply_proposal(empty_plan(["sky"]), result)
        self.assertFalse(plan["caps_open"])
        self.assertFalse(plan["strong"])

    def test_a_gemini_http_failure_is_named_on_the_ollama_plan(self) -> None:
        from ai_dt.status_text import plan_backend_line

        def gemini(*_args, **_kwargs):
            return {
                "paragraph": "",
                "stages": [],
                "error": (
                    'HTTP 503 {"error": {"code": 503, "message": '
                    '"This model is currently experiencing high demand.", "status": "UNAVAILABLE"}}'
                ),
            }

        def ollama(*_args, **_kwargs):
            return {
                "paragraph": "Keep the train as the subject.",
                "stages": [{"stage": "highlights", "action": "hold", "reason": "Hold the lamps."}],
            }

        with (
            _no_network(),
            _env(GEMINI_API_KEY="test-key"),
            patch("ai_dt.reasoner._ask_gemini", gemini),
            patch("ai_dt.reasoner._ask_ollama", ollama),
        ):
            result = propose_stages("night train", "", "")
        self.assertEqual(result["status"], "ollama")
        self.assertEqual(result["failure"], "HTTP 503, high demand")
        self.assertEqual(
            plan_backend_line(result),
            "Plan: Gemini failed (HTTP 503, high demand), Ollama llama3.2",
        )
        self.assertEqual(plain_gemini_failure("timed out"), "timed out after 30s")
        self.assertEqual(
            plain_gemini_failure("HTTP 404 This model is no longer available."),
            "HTTP 404, model not available",
        )
        self.assertEqual(
            plain_gemini_failure(
                'HTTP 429 {"error": {"status": "RESOURCE_EXHAUSTED", "message": "Free tier finished."}}'
            ),
            "HTTP 429, rate limit",
        )

    def test_a_503_tries_the_next_gemini_model(self) -> None:
        from ai_dt.reasoner import REASON_MODELS

        calls: list[str] = []

        def post(url, body, timeout):
            calls.append(url)
            if len(calls) == 1:
                return {
                    "_http_error": (
                        'HTTP 503 {"error": {"code": 503, "message": '
                        '"This model is currently experiencing high demand.", "status": "UNAVAILABLE"}}'
                    )
                }
            return _gemini_plan()

        with (
            _no_network(),
            _env(GEMINI_API_KEY="test-key", GEMINI_REASON_MODEL=""),
            patch("ai_dt.reasoner.ollama_reachable", return_value=True),
            patch("ai_dt.reasoner._post_json", post),
            patch("ai_dt.reasoner._ask_ollama", _ollama_must_not_run),
        ):
            result = propose_stages("facade", "", "")
        self.assertEqual(result["status"], "gemini")
        self.assertEqual(result["backend"], REASON_MODELS[1])
        self.assertEqual(_models_called(calls), [REASON_MODELS[0], REASON_MODELS[1]])
        self.assertNotIn("failure", result)

    def test_a_quota_error_tries_the_next_gemini_model(self) -> None:
        from ai_dt.reasoner import REASON_MODELS

        calls: list[str] = []

        def post(url, body, timeout):
            calls.append(url)
            if len(calls) == 1:
                return {
                    "_http_error": (
                        'HTTP 429 {"error": {"code": 429, "message": '
                        '"You exceeded your current quota. Free tier finished.", '
                        '"status": "RESOURCE_EXHAUSTED"}}'
                    )
                }
            return _gemini_plan()

        with (
            _no_network(),
            _env(GEMINI_API_KEY="test-key", GEMINI_REASON_MODEL=""),
            patch("ai_dt.reasoner._post_json", post),
            patch("ai_dt.reasoner._ask_ollama", _ollama_must_not_run),
        ):
            result = propose_stages("facade", "", "")
        self.assertEqual(result["status"], "gemini")
        self.assertEqual(result["backend"], REASON_MODELS[1])
        self.assertEqual(_models_called(calls), [REASON_MODELS[0], REASON_MODELS[1]])

    def test_a_missing_model_tries_the_next_and_a_named_model_is_first(self) -> None:
        calls: list[str] = []

        def post(url, body, timeout):
            calls.append(url)
            if len(calls) == 1:
                return {
                    "_http_error": (
                        "HTTP 404 This model models/gemini-2.5-flash is no longer available."
                    )
                }
            return _gemini_plan()

        with (
            _no_network(),
            _env(GEMINI_API_KEY="test-key", GEMINI_REASON_MODEL="gemini-2.5-flash"),
            patch("ai_dt.reasoner._post_json", post),
            patch("ai_dt.reasoner._ask_ollama", _ollama_must_not_run),
        ):
            result = propose_stages("facade", "", "")
        self.assertEqual(result["status"], "gemini")
        self.assertEqual(result["backend"], "gemini-3.5-flash-lite")
        self.assertEqual(_models_called(calls), ["gemini-2.5-flash", "gemini-3.5-flash-lite"])

    def test_a_rejected_key_does_not_try_the_next_model(self) -> None:
        calls: list[str] = []

        def post(url, body, timeout):
            calls.append(url)
            return {"_http_error": "HTTP 401 API key not valid"}

        def ollama(*_args, **_kwargs):
            return {
                "paragraph": "Hold the facade.",
                "stages": [{"stage": "subject", "action": "hold", "reason": "Hold the facade."}],
            }

        with (
            _no_network(),
            _env(GEMINI_API_KEY="test-key", GEMINI_REASON_MODEL=""),
            patch("ai_dt.reasoner._post_json", post),
            patch("ai_dt.reasoner._ask_ollama", ollama),
        ):
            result = propose_stages("facade", "", "")
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["status"], "ollama")
        self.assertIn("HTTP 401, API key rejected", result["failure"])

    def test_every_gemini_model_fails_before_ollama(self) -> None:
        from ai_dt.reasoner import REASON_MODELS
        from ai_dt.status_text import plan_backend_line

        calls: list[str] = []

        def post(url, body, timeout):
            calls.append(url)
            return {
                "_http_error": (
                    'HTTP 503 {"error": {"code": 503, "status": "UNAVAILABLE", '
                    '"message": "high demand"}}'
                )
            }

        def ollama(*_args, **_kwargs):
            return {
                "paragraph": "Hold the facade.",
                "stages": [{"stage": "subject", "action": "hold", "reason": "Hold the facade."}],
            }

        with (
            _no_network(),
            _env(GEMINI_API_KEY="test-key", GEMINI_REASON_MODEL=""),
            patch("ai_dt.reasoner._post_json", post),
            patch("ai_dt.reasoner._ask_ollama", ollama),
        ):
            result = propose_stages("facade", "", "")
        self.assertEqual(_models_called(calls), list(REASON_MODELS))
        self.assertEqual(result["status"], "ollama")
        self.assertEqual(result["backend"], "llama3.2")
        self.assertIn("gemini-3.5-flash-lite: HTTP 503, high demand", result["failure"])
        self.assertIn("gemini-3-flash-preview: HTTP 503, high demand", result["failure"])
        self.assertEqual(
            plan_backend_line(result),
            "Plan: Gemini failed (" + result["failure"] + "), Ollama llama3.2",
        )

    def test_validate_drops_caps_words_and_unknown_stages(self) -> None:
        kept = validate_proposal(
            {
                "stages": [
                    {"stage": "subject", "action": "lift", "reason": "Lift the body."},
                    {"stage": "epic", "action": "strong", "reason": "no"},
                    {"stage": "grade", "action": "lift", "reason": "no"},
                    {"stage": "foreground", "action": "darken", "reason": "Darken the perch."},
                    {"stage": "eyes", "action": "revert", "reason": "Leave them."},
                ]
            }
        )
        self.assertEqual(
            [(item["stage"], item["action"]) for item in kept],
            [("subject", "lift"), ("foreground", "darken"), ("eyes", "skip")],
        )


class PlanListTests(unittest.TestCase):
    def test_user_clauses_override_the_model_and_keep_caps_shut(self) -> None:
        proposal = {
            "status": "ollama",
            "backend": "llama3.2",
            "note": "Stages proposed by the local model llama3.2.",
            "caps_open": True,
            "strong": True,
            "stages": [
                {"stage": "subject", "action": "lift", "reason": "Lift the body."},
                {"stage": "surround", "action": "darken", "reason": "Quiet the surround."},
                {"stage": "eyes", "action": "skip", "reason": "Leave the eyes."},
            ],
        }
        plan = plan_stages(
            "darken the subject",
            proposal,
            guide_stages=["subject", "background", "eyes"],
            separation={
                "stages_ran": ["subject", "background"],
                "skipped": ["eyes"],
                "vignette": False,
                "gradient": {},
            },
        )
        subject = next(line for line in plan["lines"] if line["stage"] == "subject")
        eyes = next(line for line in plan["lines"] if line["stage"] == "eyes")
        self.assertEqual(subject["source"], "user")
        self.assertTrue(subject["override"])
        self.assertEqual(eyes["source"], "reasoner")
        self.assertTrue(eyes["skipped"])
        self.assertFalse(plan["caps_open"])
        self.assertIn("Caps closed.", plan["text"])
        self.assertIn("llama3.2", plan["text"])
        self.assertIn("user", plan["text"])
        self.assertIn("reasoner", plan["text"])
        self.assertIn("skipped", plan["text"].lower())
        intents = stage_intents(plan)
        subject_ask = next(item for item in intents["asks"] if item["target"] == "subject")
        self.assertFalse(intents["caps_open"])
        self.assertEqual(subject_ask["verb"], "darken")
        self.assertEqual(subject_ask["source"], "user")

    def test_plain_list_names_caps_ignored_words_and_sources(self) -> None:
        proposal = {
            "status": "skipped",
            "backend": "",
            "note": "Reasoning was skipped.",
            "stages": [{"stage": "surround", "action": "darken", "reason": "Quiet the surround."}],
        }
        plan = plan_stages(
            "epic, darken the subject, light behind the subject, epid",
            proposal,
            guide_stages=["subject", "background"],
            separation={
                "stages_ran": ["subject", "background"],
                "skipped": [],
                "vignette": False,
                "gradient": {},
            },
        )
        text = plan["text"]
        self.assertEqual(text, plan["label"])
        self.assertEqual(text, format_plan_label(plan))
        self.assertEqual(text, format_plan_text(plan))
        self.assertTrue(plan["caps_open"])
        self.assertIn("Caps open because you said epic.", text)
        self.assertIn("Ignored: epid.", text)
        self.assertIn("Reasoning was skipped.", text)
        self.assertIn("subject —", text)
        self.assertIn("surround —", text)
        self.assertIn("behind —", text)
        self.assertIn("user", text)
        self.assertIn("reasoner", text)
        intents = stage_intents(plan)
        self.assertTrue(intents["caps_open"])
        self.assertIn(("lift", "behind"), [(item["verb"], item["target"]) for item in intents["asks"]])


class ReplyShapeTests(unittest.TestCase):
    def test_fenced_json_and_prose_become_a_paragraph_and_stages(self) -> None:
        fenced = """The bird stays up.
```json
{"reasoning": "The bird is the subject. The reflection should get clearer.", "stages": [{"stage": "reflection", "action": "lift", "reason": "The reflection should get clearer."}, {"stage": "subject", "action": "lift", "reason": "one short sentence"}]}
```
"""
        prose = (
            "The bird is the subject and should not be crushed. "
            "The water should go darker. Lift the reflection so it is clearer."
        )
        replies = {"fenced": fenced, "prose": prose}

        def post(url, body, timeout):
            kind = body["messages"][0]["content"]
            key = "prose" if "PROSE-CASE" in kind else "fenced"
            return {"message": {"content": replies[key]}}

        with (
            _no_network(),
            _env(),
            patch("ai_dt.reasoner.ollama_reachable", return_value=True),
            patch("ai_dt.reasoner._post_json", post),
        ):
            fenced_result = propose_stages("a heron", "", "", "epic")
            prose_result = propose_stages("PROSE-CASE", "", "", "epic")
        self.assertEqual(fenced_result["status"], "ollama")
        self.assertIn("bird", fenced_result["paragraph"].lower())
        self.assertIn("reflection", fenced_result["paragraph"].lower())
        self.assertNotIn("one short sentence", fenced_result["paragraph"].lower())
        self.assertEqual([item["stage"] for item in fenced_result["stages"]], ["reflection"])
        self.assertEqual(prose_result["status"], "ollama")
        self.assertIn("bird", prose_result["paragraph"].lower())
        self.assertIn("reflection", prose_result["paragraph"].lower())
        stages = {(item["stage"], item["action"]) for item in prose_result["stages"]}
        self.assertIn(("reflection", "lift"), stages)
        self.assertIn(("subject", "lift"), stages)
        self.assertNotIn("did not return a plan", prose_result["note"])


if __name__ == "__main__":
    unittest.main()
