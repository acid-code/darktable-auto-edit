#!/usr/bin/env python3
"""Check the editor's darktable writes against a real darktable-cli.

Optional. Needs darktable-cli (``--darktable-cli``, ``$DTCLI`` or on PATH),
numpy and Pillow. For every step kind the editor writes it splices the step
into a copy of a darktable-written sidecar, renders base and test with
darktable-cli (CPU, ``:memory:`` library, private configdir) and reports

* darktable log lines that look like warnings and are not in the base render
  (e.g. "cannot get iop-order for colorbalancergb instance 1"),
* the mean absolute difference to the base render (0-255), so a step that
  loads but is silently dropped shows up as "no effect".

Kinds: drawn-mask steps (paths, inverted background, gradients, groups with
intersect/difference, ellipse glow), a colour-zones ellipse region, every
history step style (exposure/colour balance/channel mixer/colour zones looks,
split tone, ...) applied as a history row the way ``dt.styles.apply`` adds a
module, and a stack of the animal plan on one sidecar (second instances of
the same module).

    python tools/verify_dt.py IMAGE --base DARKTABLE_WRITTEN.xmp \
        [--box x,y,w,h] [--label duck] [--out DIR] [--only NAME,...] [--width 768]

Exit code 0 when every case loads without a new warning and changes the
render (cases whose region is not in the photo are reported, not failed).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WARN_RE = re.compile(
    r"(error|warn|unknown|can't|cannot|could not|failed|invalid|mismatch|not found|skipp)", re.I
)
NOISE_RE = re.compile(r"(dbind-WARNING|AT-SPI|ALSA lib|opencl|gtk|gdk|fontconfig)", re.I)
_TIME = re.compile(r"^\s*\d+\.\d+\s+")

MASK_CASES = (
    ("mask_animal", "lift"),
    ("mask_animals", "lift"),
    ("mask_animal_background", "darken"),
    ("mask_background", "darken"),
    ("mask_bottom_gradient", "darken"),
    ("mask_edge_burn", "darken"),
    ("mask_glow", "lift"),
    ("mask_sky_gradient", "darken"),
    ("mask_sky", "darken"),
    ("mask_water", "darken"),
    ("mask_vegetation", "lift"),
    ("mask_ground", "darken"),
    ("mask_architecture", "lift"),
    ("mask_people", "lift"),
)
STYLE_CASES = (
    ("subject", "", "lift"),
    ("foreground", "", "darken"),
    ("opening", "", "darken"),
    ("shafts", "", "darken"),
    ("highlights", "", "lift"),
    ("eyes", "", "lift"),
    ("recolor", "make the water warmer", "lift"),
    ("colorzones", "make the reds richer", "lift"),
    ("split_tone", "", "lift"),
    ("palander", "", "lift"),
    ("keep_color", "keep only the red", "lift"),
)
STACK = (
    ("mask_animal", "lift"),
    ("mask_animal_background", "darken"),
    ("mask_bottom_gradient", "darken"),
    ("mask_edge_burn", "darken"),
    ("mask_glow", "lift"),
)


def find_cli(explicit: str = "") -> str:
    """darktable-cli to use, or "" when there is none."""

    for candidate in (explicit, os.environ.get("DTCLI", ""), shutil.which("darktable-cli") or ""):
        if candidate and Path(candidate).exists():
            return candidate
    return ""


def warnings_in(lines: list[str]) -> list[str]:
    return [_TIME.sub("", line.strip()) for line in lines if WARN_RE.search(line) and not NOISE_RE.search(line)]


def render(cli: str, image: Path, xmp: Path, dest: Path, width: int) -> list[str]:
    """Render with darktable-cli; return the log lines. Raises on failure."""

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        # A copy, so darktable does not pick up a sidecar next to the image.
        picture = work / image.name
        shutil.copy2(image, picture)
        config = work / "config"
        config.mkdir()
        command = [
            cli, str(picture), str(xmp), str(dest), "--width", str(width), "--height", str(width),
            "--hq", "true", "--apply-custom-presets", "false",
            "--core", "--configdir", str(config), "--library", ":memory:", "--conf", "opencl=FALSE",
        ]
        done = subprocess.run(command, capture_output=True, text=True, timeout=900)
    lines = (done.stdout + done.stderr).splitlines()
    if done.returncode != 0 or not dest.is_file():
        raise RuntimeError(f"darktable-cli failed ({done.returncode}): " + " | ".join(lines[-8:]))
    return lines


def mean_diff(a: Path, b: Path) -> tuple[float, float]:
    """Mean and 99th percentile of the per-pixel max-channel difference (0-255)."""

    import numpy as np
    from PIL import Image

    one = np.asarray(Image.open(a).convert("RGB"), dtype=np.float32)
    two = np.asarray(Image.open(b).convert("RGB"), dtype=np.float32)
    if one.shape != two.shape:
        return float("nan"), float("nan")
    diff = np.abs(one - two).max(axis=2)
    return float(diff.mean()), float(np.percentile(diff, 99))


def _recipe(box: tuple[float, float, float, float], label: str) -> dict:
    category = "person" if label == "person" else "animal"
    return {"analysis": {"subjects": [{"category": category, "label": label, "confidence": 0.95, "bbox": list(box)}]}}


def splice_mask_step(document: str, recipe: dict, step_id: str, action: str, image) -> tuple[str | None, str]:
    from ai_dt import mask_steps
    from ai_dt.drawn_mask import splice_shapes

    forms, source = mask_steps.step_forms(recipe, step_id, image)
    if not forms:
        return None, "no region in this photo"
    spec = mask_steps.spec_from_forms(step_id, forms, action=action, source=source)
    kinds = sorted({str(form.get("type")) + ("/" + str(form.get("mode")) if form.get("mode") not in (None, "union") else "") for form in forms})
    spliced = splice_shapes(
        document,
        forms=spec["forms"],
        operation=spec["operation"],
        params=bytes.fromhex(spec["params"]),
        modversion=spec["modversion"],
        name=spec["name"],
        invert=spec["invert"],
    )
    return spliced, f"{spec['operation']}; forms {', '.join(kinds)}" + ("; inverted" if spec["invert"] else "")


def splice_region(document: str) -> tuple[str, str]:
    from ai_dt.drawn_mask import splice_region_module
    from ai_dt.modules_extra import colorzones_hue_params

    shapes = [{"cx": 0.3, "cy": 0.4, "rx": 0.12, "ry": 0.1}, {"cx": 0.7, "cy": 0.6, "rx": 0.1, "ry": 0.12}]
    spliced = splice_region_module(
        document, shapes=shapes, params=colorzones_hue_params("green"), operation="colorzones", modversion=5,
        name="AI colour zones region",
    )
    return spliced, "colorzones; two ellipses in a group"


def style_plugins(xml: str) -> list[dict]:
    root = ET.fromstring(xml.encode("utf-8"))
    plugins = []
    for node in root.iter("plugin"):
        plugins.append({child.tag: (child.text or "") for child in node})
    return plugins


def apply_style_rows(document: str, xml: str) -> str:
    """Add each style module as a new history row, as ``dt.styles.apply`` does."""

    from ai_dt.drawn_mask import _attr, _pipeline_slot, _place_history_item, _prepare_insert
    from html import escape

    for plugin in style_plugins(xml):
        operation = plugin["operation"]
        document, history, history_end, next_num = _prepare_insert(document)
        priorities = [int(_attr(tag, "darktable:multi_priority") or "0") for tag in history if _attr(tag, "darktable:operation") == operation]
        priority = (max(priorities) + 1) if priorities else 0
        item = (
            f'<rdf:li darktable:num="{next_num}" darktable:operation="{operation}" darktable:enabled="1" '
            f'darktable:modversion="{int(plugin["module"])}" darktable:params="{plugin["op_params"]}" '
            f'darktable:blendop_params="{plugin["blendop_params"]}" darktable:blendop_version="{int(plugin["blendop_version"])}" '
            f'darktable:multi_name="{escape(plugin.get("multi_name", ""), quote=True)}" '
            f'darktable:multi_priority="{priority}" darktable:multi_name_hand_edited="1"/>'
        )
        document = _place_history_item(document, item, next_num)
        document = re.sub(r'darktable:history_end="\d+"', f'darktable:history_end="{history_end + 1}"', document, count=1)
        document = _pipeline_slot(document, operation, priority)
    return document


def build_cases(base: str, image, box, label: str, only: set[str]) -> list[dict]:
    """Every case as ``{name, kind, xmp text or None, detail}`` (no darktable needed)."""

    from ai_dt.mapping import step_style

    recipe = _recipe(box, label)
    cases: list[dict] = []

    def wanted(name: str) -> bool:
        return not only or name in only

    for step_id, action in MASK_CASES:
        if wanted(step_id):
            try:
                text, detail = splice_mask_step(base, recipe, step_id, action, image)
            except Exception as exc:  # report, keep going
                text, detail = None, f"splice failed: {exc}"
            cases.append({"name": step_id, "kind": "mask", "xmp": text, "detail": detail})
    if wanted("group_intersect"):
        # A gradient intersected with the subject outline (the sky-gradient
        # shape, example 9), on a photo that may have no sky.
        from ai_dt import mask_steps
        from ai_dt.dt_shapes import edge_gradient
        from ai_dt.drawn_mask import splice_shapes

        outline, _source = mask_steps.step_forms(recipe, "mask_animal", image)
        if outline:
            forms = [dict(edge_gradient("top", 0.6, compression=0.3), mode="union"), dict(outline[0], mode="intersect")]
            spec = mask_steps.spec_from_forms("mask_sky_gradient", forms, action="darken")
            text = splice_shapes(
                base, forms=forms, operation=spec["operation"], params=bytes.fromhex(spec["params"]),
                modversion=spec["modversion"], name=spec["name"], invert=False,
            )
            cases.append({"name": "group_intersect", "kind": "mask", "xmp": text, "detail": "exposure; gradient intersect path"})
        else:
            cases.append({"name": "group_intersect", "kind": "mask", "xmp": None, "detail": "no region in this photo"})
    if wanted("colorzones_region"):
        text, detail = splice_region(base)
        cases.append({"name": "colorzones_region", "kind": "region", "xmp": text, "detail": detail})
    for feature, title, action in STYLE_CASES:
        name = f"style_{feature}"
        if wanted(name) or wanted(feature):
            xml = step_style(feature, title=title, action=action)
            if xml is None:
                cases.append({"name": name, "kind": "style", "xmp": None, "detail": "no style for this id"})
                continue
            ops = ", ".join(plugin["operation"] for plugin in style_plugins(xml))
            cases.append({"name": name, "kind": "style", "xmp": apply_style_rows(base, xml), "detail": ops})
    if wanted("animal_stack"):
        document = base
        parts = []
        for step_id, action in STACK:
            spliced, detail = splice_mask_step(document, recipe, step_id, action, image)
            if spliced is not None:
                document = spliced
                parts.append(step_id)
        split = step_style("split_tone")
        if split:
            document = apply_style_rows(document, split)
            parts.append("split_tone")
        cases.append({"name": "animal_stack", "kind": "stack", "xmp": document, "detail": " + ".join(parts)})
    return cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("image", type=Path)
    parser.add_argument("--base", type=Path, required=True, help="a darktable-written sidecar for IMAGE")
    parser.add_argument("--box", default="0.525,0.33,0.245,0.63", help="subject box x,y,w,h (0-1)")
    parser.add_argument("--label", default="duck")
    parser.add_argument("--out", type=Path, default=Path("verify_dt_out"))
    parser.add_argument("--only", default="", help="comma-separated case names")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--darktable-cli", default="")
    args = parser.parse_args(argv)

    cli = find_cli(args.darktable_cli)
    if not cli:
        print("darktable-cli not found (use --darktable-cli or $DTCLI)", file=sys.stderr)
        return 2
    from PIL import Image

    image = Image.open(args.image).convert("RGB")
    image.thumbnail((1024, 1024))
    box = tuple(float(value) for value in args.box.split(","))
    base_text = args.base.read_text(encoding="utf-8")
    args.out.mkdir(parents=True, exist_ok=True)
    base_png = args.out / "base.png"
    base_warn = set(warnings_in(render(cli, args.image, args.base, base_png, args.width)))
    only = {item.strip() for item in args.only.split(",") if item.strip()}
    results = []
    failed = 0
    for case in build_cases(base_text, image, box, args.label, only):
        row = {key: case[key] for key in ("name", "kind", "detail")}
        if case["xmp"] is None:
            row["status"] = "not in photo" if "no region" in case["detail"] else "error"
            failed += row["status"] == "error"
            results.append(row)
            print(f"{row['name']:<24} {row['status']:<12} {row['detail']}")
            continue
        xmp = args.out / f"{case['name']}.xmp"
        xmp.write_text(case["xmp"], encoding="utf-8")
        png = args.out / f"{case['name']}.png"
        try:
            lines = render(cli, args.image, xmp, png, args.width)
        except RuntimeError as exc:
            row.update(status="render failed", warnings=[str(exc)])
            failed += 1
            results.append(row)
            print(f"{row['name']:<24} render failed  {exc}")
            continue
        new = [line for line in warnings_in(lines) if line not in base_warn]
        diff, p99 = mean_diff(base_png, png)
        status = "ok"
        if new:
            status = "warnings"
        elif not (diff > 0.2 or p99 >= 3.0):
            # A small local region moves few pixels: the p99 still shows it.
            status = "no effect"
        failed += status != "ok"
        row.update(status=status, warnings=new, mean_abs_diff=round(diff, 3), p99_abs_diff=round(p99, 1))
        results.append(row)
        print(f"{row['name']:<24} {status:<12} diff {diff:6.2f} p99 {p99:5.1f}  {row['detail']}" + (f"\n    {new}" if new else ""))
    (args.out / "report.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"{len(results)} cases, {failed} failed; report {args.out / 'report.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
