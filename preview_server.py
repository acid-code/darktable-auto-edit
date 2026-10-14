#!/usr/bin/env python3
"""Local page for the analyzer. Darktable is not involved.

Run ``python3 preview_server.py`` and open http://127.0.0.1:8741
The after image is an approximation of the recipe, not a Darktable render.
"""

from __future__ import annotations

import base64
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image, UnidentifiedImageError

from ai_dt.pipeline import analyze, write_outputs
from ai_dt.simulate import render_preview, sample_image

ROOT = Path(__file__).resolve().parent
PAGE = ROOT / "preview" / "index.html"
HOST = "0.0.0.0"
PORT = 8741
MAX_BODY = 8_000_000
SAMPLES = ("portrait", "architecture", "night", "blue")


def _png_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/png;base64," + encoded


def _load_upload(payload: str) -> Image.Image:
    raw = payload.split(",", 1)[-1]
    try:
        data = base64.b64decode(raw, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("That upload is not valid base64") from exc
    if len(data) > MAX_BODY:
        raise ValueError("That image is too large")
    try:
        with Image.open(io.BytesIO(data)) as handle:
            image = handle.convert("RGB")
            image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("That file is not an image the preview can read") from exc
    image.thumbnail((1280, 1280))
    return image


def analyze_request(body: dict) -> dict:
    vibe = body.get("vibe") or ""
    steer = body.get("steer") or ""
    look = body.get("look") or ""
    if not isinstance(vibe, str):
        raise ValueError("vibe must be text")
    if not isinstance(steer, str):
        raise ValueError("steer must be text")
    if not isinstance(look, str):
        raise ValueError("look must be text")
    if len(vibe) > 400 or len(steer) > 400 or len(look) > 80:
        raise ValueError("Keep the vibe under 400 characters")
    sample = body.get("sample")
    upload = body.get("image_base64")
    if isinstance(sample, str) and sample:
        if sample not in SAMPLES:
            raise ValueError(f"Unknown sample {sample!r}")
        image = sample_image(sample)
        label = sample
    elif isinstance(upload, str) and upload:
        image = _load_upload(upload)
        label = "upload"
    else:
        raise ValueError("Choose a sample or upload an image")

    work = ROOT / "ai-editor-work" / "preview"
    work.mkdir(parents=True, exist_ok=True)
    source = work / "source.png"
    image.save(source)
    recipe = analyze(
        source,
        detector="heuristic",
        advisor="histogram",
        vibe=vibe,
        steer=steer,
        look=look,
        style_name="AI Edit preview",
        output_dir=work,
    )
    write_outputs(recipe, work)
    after = render_preview(image, recipe)
    match = recipe.get("recipe_match") or {}
    options = []
    for choice in (recipe.get("choices") or [])[:3]:
        if not isinstance(choice, dict):
            continue
        if choice.get("applied"):
            preview = after
        else:
            alternate = analyze(
                source,
                detector="heuristic",
                advisor="histogram",
                vibe=vibe,
                steer=steer,
                look=str(choice.get("id") or ""),
                style_name="AI Edit preview",
            )
            preview = render_preview(image, alternate)
        options.append(
            {
                "id": choice.get("id"),
                "title": choice.get("title"),
                "why": choice.get("why"),
                "applied": bool(choice.get("applied")),
                "png": _png_data_url(preview),
            }
        )
    version = recipe.get("version_record") or {}
    return {
        "sample": label,
        "before_png": _png_data_url(image),
        "after_png": _png_data_url(after),
        "mood": recipe["analysis"]["mood"],
        "rationale": recipe["analysis"]["rationale"],
        "atmosphere": recipe["analysis"]["atmosphere"],
        "subjects": recipe["analysis"]["subjects"],
        "match": match,
        "reading": recipe.get("reading") or {},
        "recommendations": recipe.get("recommendations") or [],
        "choices": recipe.get("choices") or [],
        "options": options,
        "sentence": version.get("sentence") or "",
        "plan": recipe.get("plan") or {},
        "modules": [
            {"operation": item["operation"], "multi_name": item["multi_name"], "params": item["params"]}
            for item in recipe["modules"]
        ],
        "note": (
            "The after image is a Pillow approximation of the recipe. "
            "Darktable applies the same numbers with its own modules."
        ),
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "AIEditorPreview/1.1"

    def log_message(self, fmt: str, *args) -> None:
        print(f"[preview] {self.address_string()} {fmt % args}")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            page = PAGE.read_bytes()
            self._send(200, page, "text/html; charset=utf-8")
            return
        if path == "/api/health":
            self._send(200, b'{"ok":true}\n', "application/json")
            return
        self._send(404, b"Not found\n", "text/plain; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/api/analyze":
            self._send(404, b'{"error":"Not found"}\n', "application/json")
            return
        length = int(self.headers.get("Content-Length") or "0")
        if length <= 0 or length > MAX_BODY:
            self._send(400, b'{"error":"Request body is missing or too large"}\n', "application/json")
            return
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8"))
            if not isinstance(body, dict):
                raise ValueError("Expected a JSON object")
            payload = analyze_request(body)
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            message = json.dumps({"error": str(exc)}).encode("utf-8")
            self._send(400, message, "application/json")
            return
        self._send(200, json.dumps(payload).encode("utf-8"), "application/json")


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"AI editor preview at http://127.0.0.1:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\npreview stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
