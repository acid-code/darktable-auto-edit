#!/usr/bin/env python3
"""Local page for the analyzer. Darktable is not involved.

Run ``python3 preview_server.py`` and open http://127.0.0.1:8741
The after image is an approximation of the recipe, not a Darktable render.
"""

from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image, UnidentifiedImageError

from ai_dt.gallery import load_cards, read_pick, render_option, work_is_gallery, write_pick
from ai_dt.pipeline import analyze, write_outputs
from ai_dt.simulate import render_preview, sample_image

ROOT = Path(__file__).resolve().parent
PAGE = ROOT / "preview" / "index.html"
OPTIONS_PAGE = ROOT / "preview" / "options.html"
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


def _rendered_image(recipe: dict, fallback: Image.Image) -> Image.Image:
    """The separated pixels when the run produced them, otherwise the grade sketch."""

    rendered = recipe.get("_rendered")
    if rendered is None:
        return render_preview(fallback, recipe)
    array = np.asarray(rendered)
    if array.ndim != 3 or array.shape[2] != 3:
        return render_preview(fallback, recipe)
    scaled = np.clip(array.astype(np.float32, copy=False), 0.0, 1.0)
    data = np.rint(scaled * 255.0).astype(np.uint8)
    return Image.fromarray(data, "RGB")


def _ran_line(recipe: dict) -> str:
    separation = recipe.get("separation") if isinstance(recipe.get("separation"), dict) else {}
    ran = [str(stage) for stage in (separation.get("stages_ran") or []) if str(stage).strip()]
    if not ran:
        return ""
    return "Ran " + ", ".join(ran) + "."


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
    after = _rendered_image(recipe, image)
    next_image = None
    if recipe.get("_next_rendered") is not None:
        next_image = _rendered_image({"_rendered": recipe.get("_next_rendered")}, image)
    write_outputs(recipe, work)
    match = recipe.get("recipe_match") or {}
    ran = _ran_line(recipe)
    options = []
    for choice in (recipe.get("choices") or [])[:3]:
        if not isinstance(choice, dict):
            continue
        if choice.get("applied"):
            preview = after
            preview_ran = ran
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
            preview = _rendered_image(alternate, image)
            preview_ran = _ran_line(alternate)
        why = str(choice.get("why") or "")
        if preview_ran:
            why = f"{preview_ran} {why}".strip()
        options.append(
            {
                "id": choice.get("id"),
                "title": choice.get("title"),
                "why": why,
                "applied": bool(choice.get("applied")),
                "png": _png_data_url(preview),
            }
        )
    version = recipe.get("version_record") or {}
    return {
        "sample": label,
        "before_png": _png_data_url(image),
        "after_png": _png_data_url(after),
        "next_png": _png_data_url(next_image) if next_image is not None else "",
        "next_title": str(recipe.get("features_next_title") or ""),
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


def _query_work(query: dict[str, list[str]]) -> Path:
    raw = (query.get("work") or [""])[0]
    return Path(raw)


def _option_source(work: Path) -> Path:
    """The developed preview when it is there, otherwise the file named in the recipe."""

    preview = work / "preview.jpg"
    if preview.is_file():
        return preview
    recipe = json.loads((work / "output.json").read_text(encoding="utf-8"))
    named = ""
    if isinstance(recipe, dict):
        named = str(recipe.get("image") or "")
        if not named and isinstance(recipe.get("input"), dict):
            named = str(recipe["input"].get("path") or "")
    return Path(named) if named else preview


class Handler(BaseHTTPRequestHandler):
    server_version = "AIEditorPreview/1.2"

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
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        if path in ("/", "/index.html"):
            page = PAGE.read_bytes()
            self._send(200, page, "text/html; charset=utf-8")
            return
        if path in ("/options", "/options.html"):
            self._send(200, OPTIONS_PAGE.read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/api/health":
            self._send(200, b'{"ok":true}\n', "application/json")
            return
        if path == "/api/options":
            self._send_options(query)
            return
        if path == "/api/option.jpg":
            self._send_option_jpeg(query)
            return
        if path == "/api/pick":
            self._send_pick(query)
            return
        self._send(404, b"Not found\n", "text/plain; charset=utf-8")

    def _send_options(self, query: dict[str, list[str]]) -> None:
        work = _query_work(query)
        if not work_is_gallery(work):
            self._send(404, b'{"error":"That folder is not an editor work folder"}\n', "application/json")
            return
        body = json.dumps({"cards": load_cards(work)}).encode("utf-8")
        self._send(200, body, "application/json")

    def _send_option_jpeg(self, query: dict[str, list[str]]) -> None:
        work = _query_work(query)
        look = (query.get("id") or [""])[0]
        if not work_is_gallery(work):
            self._send(404, b"Not found\n", "text/plain; charset=utf-8")
            return
        allowed = {str(item["id"]) for item in load_cards(work)}
        if look not in allowed:
            self._send(404, b"Not found\n", "text/plain; charset=utf-8")
            return
        try:
            jpeg = render_option(work, _option_source(work), look)
            payload = jpeg.read_bytes()
        except (OSError, ValueError) as exc:
            message = str(exc).encode("utf-8")
            self._send(400, message, "text/plain; charset=utf-8")
            return
        self._send(200, payload, "image/jpeg")

    def _send_pick(self, query: dict[str, list[str]]) -> None:
        work = _query_work(query)
        if not work_is_gallery(work):
            self._send(404, b'{"error":"That folder is not an editor work folder"}\n', "application/json")
            return
        pick = read_pick(work) or {}
        self._send(200, json.dumps(pick).encode("utf-8"), "application/json")

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/pick":
            self._choose()
            return
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

    def _choose(self) -> None:
        length = int(self.headers.get("Content-Length") or "0")
        if length <= 0 or length > 16_000:
            self._send(400, b'{"error":"Request body is missing or too large"}\n', "application/json")
            return
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(body, dict):
                raise ValueError("Expected a JSON object")
            work = Path(str(body.get("work") or ""))
            look = str(body.get("id") or "")
            if not work_is_gallery(work):
                self._send(404, b'{"error":"That folder is not an editor work folder"}\n', "application/json")
                return
            cards = load_cards(work)
            match = next((item for item in cards if item["id"] == look), None)
            if match is None:
                raise ValueError("That look is not on this sheet")
            record = write_pick(work, look, source="preview", title=str(match["title"]), applied=False)
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            message = json.dumps({"error": str(exc)}).encode("utf-8")
            self._send(400, message, "application/json")
            return
        self._send(200, json.dumps(record).encode("utf-8"), "application/json")


def server_responds(timeout: float = 0.5) -> bool:
    """True when this preview is answering on the local port."""

    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{PORT}/api/health",
            timeout=timeout,
        ) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError, TimeoutError):
        return False


def _spawn_server() -> None:
    """Start a preview process that keeps running after this one exits."""

    kwargs: dict = {
        "cwd": str(ROOT),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = 0x08000000 | 0x00000200
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen([sys.executable, str(Path(__file__).resolve())], **kwargs)


def _listening_pids(port: int) -> list[int]:
    pids: list[int] = []
    if sys.platform == "win32":
        completed = subprocess.run(
            ["netstat", "-ano", "-p", "tcp"],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=10,
            check=False,
        )
        for line in completed.stdout.splitlines():
            if "LISTENING" not in line.upper():
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            _, _, port_text = parts[1].rpartition(":")
            if port_text == str(port):
                try:
                    pids.append(int(parts[-1]))
                except ValueError:
                    continue
        return pids
    completed = subprocess.run(
        ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=10,
        check=False,
    )
    for line in completed.stdout.splitlines():
        try:
            pids.append(int(line.strip()))
        except ValueError:
            continue
    return pids


def _command_line(pid: int) -> str:
    if sys.platform == "win32":
        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"(Get-CimInstance Win32_Process -Filter \"ProcessId={pid}\").CommandLine",
            ],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=15,
            check=False,
        )
        return completed.stdout or ""
    completed = subprocess.run(
        ["ps", "-p", str(pid), "-o", "args="],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=10,
        check=False,
    )
    return completed.stdout or ""


def _stop_stuck_preview() -> None:
    """Stop a preview process that holds the port and is not answering."""

    for pid in _listening_pids(PORT):
        if pid == os.getpid() or pid <= 0:
            continue
        if "preview_server.py" not in _command_line(pid):
            continue
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/F"],
                capture_output=True,
                timeout=10,
                check=False,
            )
        else:
            try:
                os.kill(pid, 15)
            except OSError:
                continue


def _wait_until_up(seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if server_responds():
            return True
        time.sleep(0.25)
    return server_responds()


def ensure_running(wait_seconds: float = 20.0) -> bool:
    """Start the preview when it does not answer, then wait until it does.

    A process that still holds the port but never answers is replaced once.
    """

    if server_responds():
        return True
    _spawn_server()
    if _wait_until_up(wait_seconds):
        return True
    _stop_stuck_preview()
    time.sleep(0.3)
    _spawn_server()
    return _wait_until_up(wait_seconds)


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
    if "--ensure" in sys.argv:
        raise SystemExit(0 if ensure_running() else 1)
    main()
