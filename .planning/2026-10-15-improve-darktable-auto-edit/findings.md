# Findings & Decisions

## Requirements

- Deep-dive the current Darktable auto-editor and recommend how to improve it.
- Do not change product code in this pass.
- Respect the project's constraints: local by default, Darktable 5 style blobs, masks cannot ride in a `.dtstyle`.

## Research Findings

### What the system is

A Darktable Lua panel (`ai_editor.lua`, ~1800 lines) exports a preview, runs `ai_processor.py`, and applies `edit.dtstyle` to the selected image. Python (`analyze` in `ai_dt/pipeline.py`) decodes RAW when it can, detects subjects, names a picture type, matches a guide and a recipe, optionally asks Gemini then Ollama for a stage plan, paints masks into pixels, and writes `edited.tif`, `output.json`, and `edit.dtstyle`.

Version is `1.4.0`. Recent commits added Mistral vision, Gemini, presets, revert, and a speed fix that collapsed picture/Gemini/finish into one Python process. The working tree (uncommitted) adds mask-feature accept/skip (`ai_dt/features.py`), a finish gate (`ai_dt/runbook.py`), claim guards (`ai_dt/review.py`), and more separation/Lua/preview behavior. About +843 / -140 across 12 tracked files, plus those new modules.

### Data flow on a click

1. Lua exports a JPEG (longest edge from the preview preference, 256–2048).
2. `analyze` may replace that with a half-resolution RAW decode.
3. Detectors (YOLO / MediaPipe / heuristic) produce boxes. `decide_picture` names the type from boxes plus pixel stats.
4. A guide JSON supplies stages and mask grades. A recipe JSON supplies the global look. Both match by term frequency, not by folder name.
5. `read_frame` writes a sentence (subject, reflection, horizon, windows).
6. Gemini (if `GEMINI_API_KEY`) else Ollama proposes stages. The steer grammar overrides the model. `review_plan` drops ungrounded claims. `lock_found_stages` restores subject/eyes/skin when a box exists and the user did not skip them.
7. `select_mask_features` commits `separate` when a person or animal box exists. Other tools (foreground, opening, shafts, highlights) stay a preview until Accept.
8. `render_finish` (`ai_dt/separate.py`, ~1766 lines) paints the masks. `build_modules` / `append_extra_modules` build the style (exposure, color balance, vignette, tone equalizer, filmic or sigmoid, and more).
9. Lua `apply_recipe` imports the style, applies it, and deletes the style entry. It never imports `edited.tif`.

### The split that matters

`ai_dt/masks_xmp.py` states the constraint and hard-codes it: `masks_can_attach()` returns `False`. A sidecar would replace the whole history. A style cannot store the mask table. So local stages stay in `edited.tif`.

The panel text tells the user the TIFF was written. The darkroom image only receives the style. Subject lift, eye clarity, perch sharpness, background blur, and the new feature previews are in a file Darktable does not open.

### How decisions stack

Precedence is implicit and enforced by later functions undoing earlier ones:

user steer > measurement locks (`lock_found_stages`, `guard_faithful_grade`, `guard_warm_grade`) > model proposal > guide defaults.

Planners that can disagree: detectors, `decide_picture`, steer scene override, guide match, recipe match, Gemini/Ollama, steer grammar, `review_plan`, `recipes/reason.py`, feature selection. Caps stay closed unless the user wrote strong/epic. The model cannot open them.

### Tests

`unittest` coverage is broad and behavioral: guides, picture types, steer grammar, Gemini fallback, revert, gradients, versions, feature accept/skip. Pictures are synthetic. The only real fixture noted is `tests/fixtures/heron.jpg`. Regressions of "this branch stayed sharp" are locked by constructed arrays, not by a photo set.

### Other constraints already paid for

- Panel used to block on several Python startups. `run_staged` now forces `--stage all` even though the CLI still has `picture`, `gemini`, `ollama`, and `finish`.
- Apply of a named look can reuse `stage.json` and `proposal.json` and skip a second model call. Accept/skip of a feature stays on the same photo so a new duplicate does not cover a vignette the user fixed.
- `assess_run` withholds look buttons until required stages landed (portrait needs subject, eyes, skin, background).
- Windows install, hidden console (`windows-start-hidden.vbs`, a small Go host), and Flatpak spawn are already special-cased. Do not grow that surface for edit quality.

## Technical Decisions

| Decision | Rationale |
|----------|-----------|
| Rank "show the masked picture" first | The largest body of code (`separate.py`) does not reach the darkroom image |
| Rank a single plan precedence second | Safety is already correct; the cost is that `analyze` is the only place that knows the order |
| Defer another model backend | Gemini, Ollama, Mistral, and the histogram already overlap, and each is clamped |
| Defer more Darktable modules until masks attach | `modules_extra.py` adds parametric modules the style can carry; it does not fix local grades |

### Contact sheet

Propose opens `http://127.0.0.1:8741/options?work=<work dir>`. The page lists every choice from the recipe, plus Accept and Skip when `features_next` is set. Each card is a JPEG painted from the saved plan, so a card does not call Gemini again. A click writes `picked.json` with `source=preview` and `applied=false`. A panel button writes the same file with `source=editor` and `applied=true`. The page polls that file and highlights the chosen card. Darktable applies a pending preview pick when the darkroom image loads or the pointer moves over a picture, using the same open-the-TIFF path as the panel. The original file stays unchanged.

The sheet only accepts a folder whose path contains `ai-editor-work` and an `output.json`.

## Issues Encountered

| Issue | Resolution |
|-------|------------|
| No prior plan in the repo | Created `.planning/2026-10-15-improve-darktable-auto-edit/` |

## Resources

- `ai_editor.lua` `apply_recipe` — style only
- `ai_dt/masks_xmp.py` — why masks stay in the TIFF
- `ai_dt/pipeline.py` `analyze` — orchestration
- `ai_dt/separate.py` `render_finish` — pixel masks
- `README.md` Limits — style has no drawn mask; TIFF is the masked picture
- Commit `60a5d9e` — Mistral, Gemini, speed
- Commit message on `0873764` — "next thing is Darktable real changes"
