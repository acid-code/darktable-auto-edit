# HANDOFF — `feature/ai-masks`

Branch `feature/ai-masks`, cut from `feature/gui-improvement` at `dd888ef`.
Local only; nothing was pushed. Work was done on the Grok Bot box (Linux,
Python 3.13, no darktable, no API keys).

## Plan (all phases)

| Phase | What | State |
| --- | --- | --- |
| 1 | Path / gradient / group shape writer for `masks_history` (+ reader, reference renderer, tests) | **done** |
| 2 | Segmentation backends (Subject, Sky, Background, People, Landscape classes), wired into Accept steps, guide `build` values, Lua names | **done** |
| 2b | Animal masks: per-animal outlines + union (user priority: landscapes, animals, then people) | **done** |
| 3 | Smarter editing from the photoserge video lessons (ordering, caps, landscape class grades, sky-intersected gradients/glow, split tone, selective color, edge burn, crop suggestion, few-shot prompt data), plus animal Accept steps | **done** |
| 4 | Tests for everything new, full suite green | rolling, per phase |

## Commits (on `feature/ai-masks`, pushed to origin)

| Commit | Phase |
| --- | --- |
| `44700c9` | 1 — shape writer |
| `061adf5` | 2 — segmentation + drawn-mask Accept steps |
| `9e76cfb` | 2b — animal instance masks |
| `0ee8388` | 3 — lessons, animal steps, looks, ordering, prompts, crop |
| (this commit) | HANDOFF update |

Patches: `/workspace/dt_patches/*.patch`; bundle:
`/workspace/dt_patches/ai-masks.bundle`
(`git fetch ai-masks.bundle feature/ai-masks:feature/ai-masks` or
`git am /path/*.patch` on top of `feature/gui-improvement` @ `dd888ef`).

## What is left

- Verify everything in darktable 5.6 (checklist below). Nothing here has
  been loaded by a real darktable.
- No panel buttons for the new steps; they come from **Ask** (grammar or
  model plan). A Lua button row would need the feature-timing rule
  (enabled only when the state is ready) and a darkroom test.
- Rectangle and brush shapes are still not written.
- Real model backends (rembg, ADE20K, darktable SAM, Gemini) are only
  tested with fakes. darktable SAM is skipped on animal crops < 256 px.
- Coordinates are preview space (a crop/rotation/lens correction already
  in history shifts the shapes) — same as the pre-existing mask steps.
- Pre-existing failing test (bird eye heuristic, from `dd888ef`).

## Done

### Phase 1 — shape writer

- `ai_dt/dt_shapes.py` (new): darktable 5.6 record layouts (checked against
  `src/develop/masks.h` at tag `release-5.6.0`): ellipse 28 B, path point
  36 B, gradient 28 B, group member 16 B; type ids PATH=2, GROUP=4,
  GRADIENT=16, ELLIPSE=32; group states (first member `USE|SHOW`=3, union
  11, intersect 19, difference 35, `+4` inverse). Pure-NumPy outline
  pipeline: block downsample → 4-connected labelling → Moore boundary
  trace → closed Douglas-Peucker → Catmull-Rom handles (darktable's own
  formula) → `PathPoint`s with state `USER` (what 5.6's potrace
  vectoriser writes). Holes become `difference` members, the way 5.6's AI
  object tool stores them. `edge_gradient()` builds linear gradients
  (rotation 0 = top on, 90 = left, 180 = bottom, 270 = right — derived
  from `gradient.c`'s formula). `render_forms()` is a reference renderer
  mirroring `group.c`'s left-fold combine and `gradient.c`'s erf formula,
  for tests and previews.
- `ai_dt/drawn_mask.py`: `splice_shapes()` inserts one module
  (`exposure` 24 B, `colorbalancergb` 132 B or `colorzones` 520 B) whose
  blend points at a **group** of paths/gradients/ellipses combined in
  order (`union`/`intersect`/`difference`/`exclusion`, per-member
  `invert`, per-member opacity). `invert=True` sets
  `DEVELOP_COMBINE_MASKS_POS` (0x04) at blend offset 20 — darktable's
  drawn-mask polarity toggle — so a background is the inverse of a
  subject group. Same insert rules as the existing ellipse/path splices
  (insert at `history_end`, priority 0 for a first instance, `iop_order_list`
  slot, forms carry the history `num` in `mask_num` and the step name in
  `mask_name`, so Revert and Back drop them). `read_mask_records()` and
  `decode_blend()` decode a sidecar's shapes and blend blob.
  CLI: `drawn_mask.py --mask-json` now also accepts `{"kind": "shapes", ...}`.
- `ai_editor.lua`: `apply_masked_step` treats `kind == "shapes"` like the
  other mask-json kinds (optional `mask.what` text for status lines).
  `luac -p` passes.
- `tests/test_dt_shapes.py` (new, 25 tests).
- `tests/test_preview_server.py`: the taskkill test pins
  `sys.platform = "win32"` (it was a Windows-only assertion that failed on
  Linux; behaviour unchanged).

### Phase 2 — segmentation and mask steps

- `ai_dt/segment/` (new package). `segment(image, classes, boxes=…,
  backend="auto")` returns one soft mask per class plus which backend made
  it. Backends, tried in order and each optional/lazy:
  `darktable-sam` (darktable 5.6's own SAM 2.1 ONNX files via
  `object_segment._segment`; subject/people from a detector box),
  `rembg` (subject; `AI_DT_REMBG_MODEL`, default `isnet-general-use`,
  `u2net_human_seg` for people), `ade20k` (`transformers` SegFormer
  `nvidia/segformer-b2-finetuned-ade-512-512` by default, OneFormer ids
  also work; `AI_DT_ADE20K_MODEL`), `gemini` (cloud; **opt-in** with
  `AI_DT_SEGMENT_GEMINI=1` + `GEMINI_API_KEY`, key sent as the
  `x-goog-api-key` header, `AI_DT_GEMINI_SEGMENT_MODEL`), and `heuristic`
  (NumPy only, always last). `AI_DT_SEGMENTER=<name>` forces one backend
  plus the heuristic fallback. Background = 1 − subject. Classes: subject,
  people, background, sky, architecture, vegetation, water, ground.
- `ai_dt/mask_steps.py` (new): seven Accept steps (`mask_sky`,
  `mask_architecture`, `mask_vegetation`, `mask_water`, `mask_ground`,
  `mask_people`, `mask_background`) plus the group id `mask_landscape`.
  Each segments `preview.jpg`, vectorises with `dt_shapes.mask_to_forms`,
  writes a `kind: "shapes"` `mask.json`, and Lua's existing
  `apply_masked_step` splices it. Grades follow photoserge example 7 and
  are capped (`LOCAL_EV_CAP = 0.5`, brilliance ±0.25). A class that is
  not in the photo leaves the step off with a sentence.
- `ai_dt/mask_grammar.py` (new): "darken the sky", "make the trees
  greener", "brighten the buildings", "lift the people", "darken the
  background", "do the landscape masks" → mask operations in `plan_chat`.
  Words the model plan already claimed are skipped (keeps "quiet the
  path" a model step).
- Wiring: `features.history_steps`/`commit_waiting_step`, `mapping.step_history_name`,
  `catalog` MASKS section, `revert._ACCEPTED_STEP_NAMES` ↔ Lua
  `editor_history_names` (7 new names, in sync, tested), guide library
  stages/`build` values `people`, `architecture`, `vegetation`, `water`,
  `ground` (painted into `edited.tif` by `separate._region_mask`), and
  `guides/landscape.json` / `guides/architecture.json` gained matching
  optional stages.
- `requirements-masks.txt` (new, optional): `rembg[cpu]`, `transformers`,
  `torch`. Kept out of `requirements-ai.txt` so `install-windows.ps1
  -WithAi` does not suddenly pull torch.
- README: "Drawn-mask steps" section.
- Tests: `tests/test_segment.py` (21), `tests/test_mask_steps.py` (15),
  `tests/synthetic_scenes.py` (helpers). Optional backends are tested
  with fakes; no network.

### Phase 2b — Animal masks (user priority: landscapes, then animals, then people)

- `ai_dt/segment/animals.py` (new): `segment_animals(image, boxes)` gives
  one outline per animal (`Animal`: label, confidence, bbox, full-frame
  mask, crop mask, source) and `.union()` for "all animals";
  `segment(image, ["animal"], boxes=…)` returns the union as
  `masks["animal"]` and the list as `instances["animal"]`.
  `animal_forms(result, which="all"|"main")` vectorises each animal on its
  own crop (so a distant bird keeps its shape) and joins them by union.
- Boxes: the existing YOLO COCO mapping (bird, cat, dog, horse, sheep, cow,
  elephant, bear, zebra, giraffe → `animal`). Size-aware confidence bar
  (≥0.15 under 1% of the frame, ≥0.2 under 5%, ≥0.25 otherwise), boxes
  under 0.8% of a side dropped, duplicates (IoU > 0.6) merged, at most 12.
- Refinement per crop: model backends first (darktable SAM, rembg, Gemini
  if opted in), then an occlusion-tolerant color heuristic (colors common
  outside the box are removed from the core model, then a close bridges
  grass blades / branches / fence wires, fragments far from the core are
  dropped), then a soft box ellipse if nothing is trustworthy. Masks with
  < 24 preview pixels are skipped with a note. Pose-agnostic by design.
- The existing guide `build: "animal"` path (`separate.extend_animal`) is
  unchanged.
- Note: darktable SAM refuses previews under 256 px, so on small crops it
  is skipped and rembg/heuristic refine instead. Real-photo quality of the
  heuristic is unverified (synthetic scenes only).
- Tests: `tests/test_animals.py` (10): multiple animals (standing, lying,
  bird in flight), occluded dog behind grass, tiny distant bird, none
  found, model-first with untrusted-model fallback, size-aware confidence.

### Phase 3 — lessons

- `ai_dt/lessons/` (new): `photoserge_examples.json` (trimmed copy of
  `/workspace/fbvideo/.../examples.json`: id, title, goal, edits,
  darktable modules/equivalent; no images, timestamps or diff stats,
  ~17 KB), `few_shot_text(type)` (general tips + 2 matching examples +
  suggested steps + the ±0.5 EV rule; animal photos get `ANIMAL_HINTS`),
  `guess_type`, `STEP_RANK`/`order_steps` (global first: look → regional
  → gradients → edge burn/glow; a stable sort so every pre-existing step
  keeps its order), `PRESETS` per picture type, `crop_suggestion`.
- Prompts: `reasoner._fill` appends the lesson block (Ollama + Gemini
  planner). `pipeline._vision_context` (Mistral/Ollama vision) appends the
  global-first tips. The word "images" is rewritten to "photos" so the
  "no image is sent" test and promise hold.
- `mask_steps`: `MaskStep.builder`; new steps `mask_sky_gradient` (sky
  outline ∩ top gradient, ex. 9), `mask_bottom_gradient` (ex. 8),
  `mask_edge_burn` (4 edge gradients − subject, ex. 11), `mask_glow`
  (ellipse − subject, ex. 10), `mask_animal`, `mask_animals`,
  `mask_animal_background` (animal outlines from Phase 2b; the inverse
  sets drawn-mask polarity). Lesson EVs (−0.85, +1.2…2.2) are capped to
  ±0.5 EV on purpose — stack a step for more.
- Animal hints: `mask_animal` is brilliance +0.1 / contrast +0.12 /
  vibrance +0.05 (fur and eyes crispness, no exposure jump);
  `mask_animal_background` darkens/mutes/softens; the `animal` preset is
  animal → background → edge burn. The existing `eyes` step still covers
  animal eyes.
- Global looks via `mapping.step_style` (one module each): `split_tone`
  (color balance rgb), `palander` (warm-only, color zones chroma curve),
  `keep_color` (selective color, hue from the request).
  `modules_extra.colorzones_keep_params` writes the curve.
- Grammar: lesson phrases fire without an edit verb and take the clause;
  animal phrases; "the wildlife/landscape/... look" expands a preset.
- Names (Python `revert` ↔ Lua `editor_history_names`, tested):
  Darken the sky gradient, Darken the bottom, Burn the edges, Glow behind
  the subject, Lift the animal, Lift the animals, Quiet around the
  animals, Split tone, Warm-only look, Keep only the
  red/orange/yellow/green/blue.
- `output.json` → `analysis.crop_suggestion` (advisory, never applied).
- Tests: `tests/test_lessons.py` (29).

## Fixes after the duck report (pushed to `origin/feature/ai-masks`)

### Bug 1 — accepted step not the active top / later step loses the earlier mask (`707e033`)

Repro: duck photo, plan subject → mask_background → foreground. Step 1
"Lift the subject" said done; its row was not the highlighted top and the
next step did not build on it.

Causes found (all in `ai_dt/drawn_mask.py`):

1. **Pre-existing (`dd888ef`)**: the subject outline (`splice_path_exposure`)
   and eye ellipse (`splice_ellipse_exposure`) filed their shape under
   `mask_num = highest mask_num + 1`, not under the new row's `num`.
   darktable attaches a shape to the history row whose `num == mask_num`,
   so the outline belonged to some other (usually an early) row.
2. **Pre-existing, and copied into my `splice_shapes`/`splice_region_module`**:
   each masked row carried only its own shapes. darktable keeps the full
   shape list on every masked row and loads the list of the *last* masked
   row below `history_end` (`dt_masks_read_masks_history`), so after step 2
   step 1's mask id was gone and darktable blended it with a filled
   (whole-frame or empty) mask.
3. **Pre-existing (`_prepare_insert`)**: with `history_end` below the row
   count (a row clicked in the history panel, or a stale sidecar), the new
   row was inserted at `history_end` and the old rows were kept above it,
   so the accepted step was mid-stack and not the highlighted top.

Fix: rows past `history_end` are discarded (as darktable does on any new
edit), the new row is appended as the top with `history_end = row count`,
its shapes are filed under its own `num`, and the previous snapshot is
carried onto the new row. `verify_top()` runs before `drawn_mask.py`
writes: if `history_end != rows`, the top row is not the step, or its mask
is not filed under it, the script exits 1 and the panel shows the reason
instead of "done". `strip_ai_history` also drops shapes filed under a
removed row. A commit with no module for the step (`step_style` is `None`)
now leaves the step off with "No darktable module could be written for …"
instead of keeping the old "This step is on this photo" note; Lua
`apply_step` returns a reason when `step_style` is malformed or its file
is missing/empty.

Tests: `tests/test_history_chain.py` (7 — chain of 3 masked steps through
the functions and through `main()`, `history_end == rows` after each,
every AI module's mask id present in darktable's current snapshot, stale
`history_end`, Back/Revert after the chain, `verify_top` refusals; 6 of 7
fail on the old code), plus one in `tests/test_features.py`.

### Bug 2 — a new plan repeated steps already applied (`1167eaa`)

Cause: `plan_chat` gave the planner no edit state, and
`_install_history_steps` even moved a re-proposed step back out of
`features_done`.

Fix (`ai_dt/applied.py`, `features.plan_chat`, `reasoner._fill`):

- Applied steps = active rows of the photo's sidecar (Lua now passes
  `--history-xmp image.sidecar` on Ask; the CLI falls back to
  `<original>.xmp`), names matched with `revert.accepted_step_names()` /
  `step_log` names, plus active `step_log` entries (the sidecar can lag the
  darkroom). Skipped steps = `features_skipped`.
- The prompt gets an "Edit state:" block: "Already applied to this photo,
  in order: subject (Subject lift)… Do not propose these again unless the
  request explicitly asks to redo or strengthen them", and the skipped list.
- `_drop_applied_steps` filters the new queue by id **or kind**
  (`subject` ≈ `mask_animal`/`mask_animals`, `mask_background` ≈
  `mask_animal_background`, `foreground` ≈ `mask_bottom_gradient`,
  `mask_sky` ≈ `mask_sky_gradient`). "again/redo/repeat" keeps any repeat;
  "more/stronger/…" keeps it only when the request names that step. A
  skipped step stays out unless the request names it. The note says what
  was left out, or that every proposed step is already on the photo.

Tests: `tests/test_applied.py` (11; mock planner returning a repeat, prompt
contains the applied and skipped steps, sidecar history, redo wording).

### Bug 3 — editing the request hid the plan; Accept/Skip were ambiguous (`d6f43b9`)

darktable 5.6's Lua `entry` has no change signal (`src/lua/widget/entry.c`
registers only `text`, `placeholder`, `is_password`, `editable`), so the
panel cannot react to each keystroke. What changed:

- The plan stays the **current plan** until Ask returns a new one. The
  sheet header reads `Current plan (from “<request>”)`; `plan_chat` and the
  pipeline record `plan_request`. An Ask that adds no step keeps the waiting
  plan ("This request did not add a new step, so the current plan stays: …").
- `status_text.accept_skip_state/accept_skip_lines` (Python, tested) give
  the line for every state: `Accept/Skip apply to: <step>`,
  `Request changed — press Ask for a new plan; Accept/Skip still apply to:
  <step>`, or "No step is waiting, so Accept and Skip are off…". They are
  stored on the recipe as `accept_skip`.
- Lua `plan_status()` swaps the line when the request box differs from
  `plan_request` (same normalisation as Python); it runs on every panel
  refresh (Ask, Accept/Skip, Back/Next, darkroom reload, selection change)
  and on `mouse-over-image-changed` / `pixelpipe-processing-complete`, so
  the line updates on the next interaction rather than on each keystroke.
- The buttons are labelled `Accept: <step>` / `Skip: <step>` (tooltips
  too); with no step they are plain, off, and the tooltip says why.
- New Lua helpers live in one `request_ui` table (the main chunk is at
  Lua's 200-local limit). `luac -p ai_editor.lua` passes.

Tests: `tests/test_request_status.py` (9, including a lua5.4 run of the
real `plan_status` block from `ai_editor.lua` against the Python lines).

## Duck test 2: animal routing, animal steps, encoding, plan label

Report: mallard on a bright concrete ledge by green water, "Make the duck
the main theme" → Stages subject, skin, eyes, background; "Painted grades
for separate, person"; "Another lift, for this request."; plan text said
"eider"; after editing the description and re-proposing: same, plus
"duckÔÇÖs" and a stale "Current plan (from …request…)" header.

Cause (confirmed with real YOLO on `uploads/duck.jpg` in `/workspace/yolo-venv`):
YOLO keeps people down to 0.15 for recall and put a 0.49 "person" box on the
duck's head, inside its 0.90 "bird" box. `decide_picture` lets people win,
so the photo became a portrait and the portrait guide planned skin/person.
Nothing re-read the type from the words, so an edited description could not
fix it. The heuristic eye search also put "eyes" on the orange feet.

Changes:
- `detectors/heuristic.drop_animal_heads` (after `gate_faces`): a non-MediaPipe
  person/face whose centre is inside a confident animal box, smaller than half
  of it and less confident, is dropped. `_eye_search_box` keeps the animal-eye
  search in the top 45% of a tall bird box.
- `animal_plan.retype_from_text` (pipeline, before `match_guide`): request,
  edited description, Mistral reading. A portrait/group/street/general type
  becomes `animal` when the words name an animal and no person (and there is
  no MediaPipe face); person boxes (or, without YOLO, the heuristic subject
  box) become a `reading` animal box so `segment_animals` has somewhere to look.
  Statues/toys/decoys are not animals.
- `animal_plan.animal_plan` (pipeline after `resolve_plan`; Ask after
  `_drop_applied_steps`): person steps are dropped (eyes unless asked);
  subject/separate → `mask_animal` (`mask_animals` for 2+), background →
  `mask_animal_background`, foreground → `mask_bottom_gradient`. A general
  emphasis request fills the default recipe in order: `mask_animal` →
  `mask_animal_background` → `mask_bottom_gradient` (only with a bright
  ground band, `bright_ground`) → `mask_edge_burn`; steps already applied or
  skipped (either naming) are not re-added. Titles are concrete actions
  ("Lift the duck a little and crisp its feathers", …), never the request.
- Planner prompt: `catalog_text` lists the animal steps, the order, and the
  generic→animal mapping; the brief gets a `subject_line` (noun, ledge/water
  regions). The reasoning paragraph is cleaned: species words become the
  user's word or "the duck"/"the bird" (`generic_species`), and "keeping the
  water/ledge untouched" is replaced by what the steps actually touch
  (`consistent_regions`).
- "Another lift, for this request." is gone: only when a subject lift is
  already on the photo the note says "A second, gentler subject lift is
  planned, because this request asks for more on the subject."
  (`plan.subject_again` flag).
- Encoding: Lua `run_processor` writes the vibe/request/description to
  UTF-8 files (`arg-vibe.txt`, `arg-request.txt`, `arg-description.txt` in the
  work folder) and passes `--vibe-file/--steer-file/--scene-file` (argv is
  the fallback); `cmd.exe` no longer re-encodes them. `ai_dt/text_io.clean_text`
  repairs cp850/cp437/cp1252 mojibake and turns smart quotes into ASCII, on
  the CLI args and on stored observations.
- Plan label: recipes record `plan_description` and
  `plan_description_source`; the header says
  `Current plan (from request “…” + the photo description)` /
  `(from your edited description)`. A Propose with an edited description
  does not carry over the old waiting steps. Lua: Ask with an empty request
  but a changed description builds a new plan; the Accept/Skip line says
  "Description changed — press Ask for a new plan…".

End to end on the real duck (offline, Mistral/Gemini mocked with the
portrait-style plan subject/skin/eyes/background): YOLO → `animal`; plan
`mask_animal, mask_animal_background, mask_bottom_gradient, mask_edge_burn`;
the `mask_animal` outline covers the duck (head to feet) with nothing outside
its box — `/workspace/diag/duck_mask.png`. Without YOLO (heuristic only) the
reading box gives the same plan and a slightly looser outline —
`/workspace/diag/duck_mask_heuristic.png`.

Tests: `tests/test_animal_routing.py` (19). Updated expectations in
`test_applied.py`, `test_features.py`, `test_request_status.py`: animal
photos now get animal step ids, and the header names request/description.

## Real darktable 5.6.2: second instances, tools/verify_dt.py, Linux Lua

Found with darktable 5.6.2 (AppImage, `darktable-cli`, `/workspace/dtverify`).

**Second instance dropped.** Splicing two steps on one module (duck:
`mask_animal` + `mask_animal_background`, both colour balance rgb) gave the
second `multi_priority=1`. A stock sidecar has `iop_order_version="5"` and
no `iop_order_list`, and `_pipeline_slot` only edited an existing list, so
darktable logged `cannot get iop-order for colorbalancergb instance 1` and
`[_dev_read_masks_history] can't find history entry 5 while adding mask …`
and silently dropped the row and its masks.
Fix: `drawn_mask._write_pipeline_list` writes the full list, as darktable
does itself (`dt_ioppr_write_iop_order` writes a list once a module has two
instances; the version stays). `ai_dt/iop_order.py` holds darktable's
built-in tables (legacy, v3.0, v3.0 JPEG, v5.0, v5.0 JPEG, 93 modules each)
copied from `src/common/iop_order.c` at release-5.6.2. Every extra instance in
the history goes right after its module. A first instance still writes no
list; a sidecar without a version is left alone.

Harness (`verify.py`, base = the real 5.6.2 sidecar, region = duck outline):

| render | warnings | inside duck (signed) | outside (signed) |
| --- | --- | --- | --- |
| background only (before) | 0 | -0.35 | -17.55 |
| 3 steps, before the fix | 7 | -0.81 | -6.76 (background dropped) |
| 2 steps (lift + background), fixed | 0 | +7.81 | -17.39 |
| 3 steps (+ bottom gradient), fixed | 0 | +7.78 | -22.62 |

The bottom gradient also darkened the duck's feet and belly (inside -1.1
with the fix alone), so for an animal subject `mask_bottom_gradient` is now
the gradient minus the animal outline.

**`tools/verify_dt.py`** (optional): splices every step kind into a copy of
a darktable-written sidecar, renders base and test with `darktable-cli`, and
fails on new warning lines or a render that does not change. On the duck,
all 28 cases pass with 0 new warnings (2 report "not in photo": sky,
people): paths, inverted background, gradients, gradient-minus-path groups,
ellipse glow, a gradient intersected with a path, a colour-zones ellipse
group, every history step style (exposure, colour balance, channel mixer,
colour zones, split tone, Palander, keep colour), and the full animal stack
with split tone. Styles are written as history rows the way
`dt.styles.apply` adds a module (darktable-cli cannot import a .dtstyle
without a library); the GUI style path itself is not covered.
`python tools/verify_dt.py IMAGE --base SIDECAR.xmp [--darktable-cli PATH]`.
`tests/test_verify_dt.py` builds every case without darktable and runs one
real render only when `$DTCLI` / `darktable-cli` exists (skipped otherwise).

**Lua off Windows.** `run_hidden` always went through the Windows worker:
on Linux `ensure_worker` slept ~2.4 s waiting for a script host that never
starts, then ran `"<command>" < NUL 2>&1` (quoted whole, and `NUL` is a file
name there), so the preview-server probe and start failed. Now
`run_hidden` and `run_captured` use a plain `io.popen(command .. " < /dev/null
2>&1")` off Windows and `ensure_worker` returns at once; the Windows path is
unchanged. `tests/test_lua_portability.py` runs that block with lua5.4
(Linux: no sleep, real output; the old code slept 2400 ms and exited 2).

Tests: `tests/test_iop_order.py` (5, real 5.6.2 sidecar fixture
`tests/fixtures/dt562_jpeg.xmp`), `tests/test_verify_dt.py` (3, one skipped
without darktable-cli), `tests/test_lua_portability.py` (2).

## Test counts

`python3 -m unittest discover -s tests -t . -v` (inside `.venv` with
`requirements.txt` installed; the box also has `lua5.4`, so the Lua JSON
test runs instead of skipping).

| When | Ran | Failures | Errors | Skipped |
| --- | --- | --- | --- | --- |
| Baseline `dd888ef`, system Python, no rawpy | 318 | 4 | 2 | 1 |
| Baseline `dd888ef`, venv with requirements | 318 | 2 | 0 | 1 |
| After phase 1 | 343 | 1 | 0 | 0 |
| After phase 2 | 379 | 1 | 0 | 0 |
| After phase 2b (animals) | 389 | 1 | 0 | 0 |
| After phase 3 | 418 | 1 | 0 | 0 |
| After bug 1 fix | 426 | 1 | 0 | 0 |
| After bug 2 fix | 437 | 1 | 0 | 0 |
| After bug 3 fix | 446 | 1 | 0 | 0 |
| After duck test 2 (animal routing) | 465 | 1 | 0 | 0 |
| After bird-eye fix (venv) | 468 | 0 | 0 | 0 |
| After bird-eye fix (system python3, no rawpy) | 468 | 0 | 0 | 4 |
| After iop-order fix (venv) | 478 | 0 | 0 | 1 (darktable-cli) |
| After iop-order fix (venv, `DTCLI` set) | 478 | 0 | 0 | 0 |
| After iop-order fix (system python3) | 478 | 0 | 0 | 5 (4 rawpy, 1 darktable-cli) |
| After halo fix (venv) | 494 | 0 | 0 | 1 (darktable-cli) |
| After halo fix (venv, `DTCLI` set) | 494 | 0 | 0 | 0 |
| After halo fix (system python3) | 494 | 0 | 0 | 5 (4 rawpy, 1 darktable-cli) |
| After offline plan + Ask (venv) | 504 | 0 | 0 | 1 (darktable-cli) |

The last pre-existing failure (`test_guides…red_yellow_bird…`, broken
since `dd888ef`) is fixed in code — see "Bird-eye fix" below. The test
expectation was right.

System `python3` without the venv (no `rawpy`): the 4 DNG tests in
`tests/test_raw.py` (`RawPipelineTests`: dng decode, full res, batch,
raw+jpeg) are skipped with "rawpy is not installed (optional RAW decoder;
pip install -r requirements.txt)"; everything else passes. In the venv they
run.

## Status and TODO (2026-10-19 ~18:00 Asia/Jerusalem)

Done (each item is a commit on feature/ai-masks; the full suite is 524 OK, 1 skipped):
- **Parrot no-op Accept** (`eb10f0c`): a step that writes no mask and no style
  fails with a plain reason, and `--restore-step` puts it back (`plan.left_off`).
  Accept syncs the sidecar first (`image.rating = image.rating`; the Lua API has
  no `write_sidecar_files`) and checks that `history_end` grew. The fixture is
  `tests/fixtures/dt56_parrot_fresh_jpg.xmp`.
- **Done state only from the sidecar** (`a1e40bb` for plan_chat/applied_steps,
  `902d1f8` for the first Ask): `analyze()` takes `history_xmp` (the CLI passes
  `--history-xmp`, or else `<original>.xmp`) and drops saved done marks that are
  not in the history.
- **New plan button** (`a1e40bb`, tests and README in `e2b9b29`).
- **Looks UI removed** (`b1df262`): the Looks button, the "Look:" line, the
  vibe box and the Preset dropdown are gone. The vibe and preset only chose the
  frame look, so the panel now always sends a blank (faithful) vibe. The CLI
  `--vibe` still works. The `open_looks` code is still there but no longer
  reachable.
- **Grey status text** (`c442f1b`): a wrapped label with no focus.
- **darktable 5.4 + 5.6** (see below): the params, blend/mask versions and Lua
  APIs we use are identical. The only difference is the module
  `colorharmonizer`, which 5.6 adds to every built-in iop order. Lua passes
  `--dt-version` (from `dt.configuration.version`) to every processor call, and
  the CLI sets `AI_DT_DARKTABLE_VERSION`. `iop_order.order_for()` drops modules
  newer than the running darktable (default 5.6). Tests are in
  `tests/test_dt_versions.py`.
  - **5.4 is not verified in real darktable yet** (Asaf is on 5.6). Only one
    quick darktable-cli 5.4.1 render was tried: no iop-order warnings. There
    has been no GUI test on 5.4.

Still TODO:
1. On Windows 5.6, GUI-test the full Accept on a freshly opened JPG. Confirm
   that the rating-sync writes the sidecar there and that the history-grew
   check passes.
2. GUI-test the New plan button and the panel without Looks on 5.6, and
   later verify 5.4 in real darktable.
3. Online launcher (`/workspace/dtverify/launch-gui-online.sh`) and the online
   GUI test. Both were skipped earlier because of the budget.
4. Optional: delete the now-unreachable `open_looks`/`watch_preview_pick` code
   and `ai_dt/choices.py` look paths.
5. The order-dependent pair `test_pipeline` + `test_guides` (portrait vignette
   and face gate) sometimes fails when the two run together, even on `f99ce55`.
   It passed in the last full run.

Resume: `cd /workspace/darktable-auto-edit && .venv/bin/python -m unittest discover -s tests`
(with GEMINI/MISTRAL keys unset for offline). GUI: `/workspace/dtverify/launch-gui.sh`.
The plugin copy is in `/workspace/dtverify/guiconfig/lua/ai_editor`.

## Default editor steps (started 2026-10-20)

Every plan automatically gets standard editor steps, measured from the image,
placed first in the plan as normal steps (accept / Back works), never duplicating
AI steps of the same module.

Plan:
1. Clipping: measure clipped (>=250) and crushed (<=5) fractions; iterate a cheap
   preview estimate of exposure + shadows/highlights to shrink them without
   flattening (contrast floor).
2. Dehaze (hazeremoval) only when a dark-channel haze metric says so; strength
   scaled to the metric.
3. Subject sharpening (sharpen + subject mask), moderate amount with threshold;
   reduced/skipped when estimated noise is high.

Checklist:
- [x] Find plan builder + step format (features.plan_chat -> step_queue; accept in commit_waiting_step)
- [x] Metrics + default-step module (ai_dt/default_steps.py)
- [x] Wire into plan builder (first, dedup/merge with AI steps)
- [x] Unit tests (tests/test_default_steps.py)
- [ ] Full suite run
- [x] darktable-cli 5.6.2 render check (and 5.4.1 cli): toneequal v2, hazeremoval v3, masked sharpen v1 all 'params ok'
- [x] README

Done notes: the steps are injected in `features.plan_chat` (`_with_default_steps`)
before `_keep_current_plan`. Applied or skipped defaults are not re-added.
Accept writes the global ones through `default_steps.style_for`, and the
sharpen through the mask-step path (`SHAPE_OPERATIONS` now allows sharpen,
12 bytes). The hazeremoval v3 blob is 16 bytes (from introspection on 5.4.1
and 5.6.2). Names are known to Revert (`revert.py`) and Lua
(`editor_history_names`).
Left: no GUI test of the default steps yet. `analyze` (Propose) does not add
them; only Ask plans do. Thresholds are tuned on synthetic images only.
