# HANDOFF — `feature/ai-masks`

Branch `feature/ai-masks`, cut from `feature/gui-improvement` at `dd888ef`.
Local only; nothing was pushed. Work was done on the Grok Bot box (Linux,
Python 3.13, no darktable, no API keys).

## Plan (all phases)

| Phase | What | State |
| --- | --- | --- |
| 1 | Path / gradient / group shape writer for `masks_history` (+ reader, reference renderer, tests) | **done** |
| 2 | Segmentation backends (Subject, Sky, Background, People, Landscape classes), wired into Accept steps, guide `build` values, Lua names | pending |
| 3 | Smarter editing from the photoserge video lessons (ordering, caps, landscape class grades, sky-intersected gradients/glow, split tone, selective color, edge burn, crop suggestion, few-shot prompt data) | pending |
| 4 | Tests for everything new, full suite green | rolling, per phase |

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

## Test counts

`python3 -m unittest discover -s tests -t . -v` (inside `.venv` with
`requirements.txt` installed; the box also has `lua5.4`, so the Lua JSON
test runs instead of skipping).

| When | Ran | Failures | Errors | Skipped |
| --- | --- | --- | --- | --- |
| Baseline `dd888ef`, system Python, no rawpy | 318 | 4 | 2 | 1 |
| Baseline `dd888ef`, venv with requirements | 318 | 2 | 0 | 1 |
| After phase 1 | 343 | 1 | 0 | 0 |

The one remaining failure is **pre-existing** and not touched here:
`tests.test_guides.PictureTypeTests.test_a_red_yellow_bird_uses_the_wildlife_guide_and_defocuses`
expects `separation.eyes >= 1`. It passes at `d5115dd` and fails at
`dd888ef`: the reworked heuristic animal-eye search (`animal_eye_boxes`,
pairs only) no longer finds the single profile eye of the synthetic bird.
Decide whether the test or the heuristic is right; I did not change the
detector.

## Known risks / unverified against real darktable 5.6

- No darktable on the box. Layouts were read from the 5.6.0 sources, not
  from a sidecar written by 5.6. The existing README notes a 5.6.2 import
  kept an ellipse (mask version 6) and a 420-byte blend v14; paths,
  gradients and groups written by `splice_shapes` have **not** been loaded
  in darktable yet.
- Path border (feather) is stored in `border[0..1]`, which darktable
  scales by `MIN(width, height)`; 0.01 is a guess at a pleasant feather.
- Gradient `compression` default 0.25 (fraction of the image diagonal);
  check the fall-off visually.
- Coordinates are taken from `preview.jpg`, which is the *rendered* image.
  Drawn masks live in pre-distortion input space, so a crop, rotation or
  lens correction already in history will shift the shapes. Same caveat
  as the existing eye/hue/object steps.

## Manual checklist in darkroom (5.6)

(Expanded as phases land.)

1. Pick a test photo with an XMP. Run
   `python ai_dt/drawn_mask.py --xmp PHOTO.xmp --xmp-out out.xmp --mask-json spec.json`
   with a `"kind": "shapes"` spec (see `tests/test_dt_shapes.py`), copy
   `out.xmp` over the sidecar with darktable closed, start darktable and
   open the photo.
2. History shows the new module with its name; the mask manager shows one
   group with the path(s) and the gradient; the group's combine icons say
   union / intersection as written.
3. Toggle "display mask" on the module: the sky path ∩ top gradient matches
   what `render_forms` predicts. Edit a path node — it must drag smoothly.
4. For an inverted (background) spec, the module's drawn-mask polarity
   toggle is on, and the effect lands outside the subject.

## How to run

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python3 -m unittest discover -s tests -t . -v
```

## Not mine, left as found

`Usersasaf2AppDataLocalTempdt562-mask-check/` and `dt-launch.log` at the
repo root look like accidental commits from Windows; left untouched.
