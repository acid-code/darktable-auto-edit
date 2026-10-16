# ADR-001: Show the masked edit in Darktable

**Status:** Accepted
**Date:** 2026-10-15
**Deciders:** Project owner

## Context

The editor's local grades (subject, eyes, skin, background, perch, reflections, and the new mask features) are painted by `render_finish` into `edited.tif`. The Lua click applies only `edit.dtstyle`. `masks_can_attach()` is permanently false because a sidecar replace would wipe the photographer's history, and a style file has no mask table.

The working tree is adding an accept/skip loop for those TIFF features. That work stays invisible in the darkroom until the TIFF, or an equivalent parametric mask, is what the user is looking at.

Constraints: stay local by default, do not replace existing history, Darktable 5.0+, Windows and Flatpak already have fragile install paths.

## Decision

Make the next product change the handoff into Darktable, not another planner or model. Prefer a duplicate whose pixels are `edited.tif` (or a Darktable-developed copy of it) so the masks are on screen and the original RAW history is untouched. Keep the style-on-RAW path only for the global grade when the user wants a parametric step they can undo.

Revisit `masks_can_attach()` when Darktable can add a drawn mask without replacing history. Until that API exists, do not spend the next cycle on more modules in `modules_extra.py`.

## Options Considered

### Option A: Import `edited.tif` as a duplicate

| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium. Lua already forks duplicates and has a work folder per image. |
| Cost | No new model or network. |
| Scalability | One TIFF per run. Half-res by default, full-res on demand. |
| Team familiarity | High. The file is already written. |

**Pros:** The picture the tests describe is the picture in the darkroom. Original RAW history stays. Accept/skip features become visible.
**Cons:** The result is baked pixels. A later exposure tweak in Darktable does not recompute the mask. Color management must match the camera white balance already used for the TIFF. The library gains a duplicate per run.

### Option B: Attach real Darktable masks to the RAW

| Dimension | Assessment |
|-----------|------------|
| Complexity | High. Blocked on a Lua or XMP path that inserts mask rows without `apply_sidecar`. |
| Cost | No bake. Fully parametric. |
| Scalability | Best long-term edit. |
| Team familiarity | The spike in `masks_xmp.py` already rejected the obvious sidecar. |

**Pros:** Subject lift and blur stay editable modules. Undo matches the rest of Darktable.
**Cons:** Not available today. Guessing at the mask table will corrupt history. The spike exists specifically to stop that.

### Option C: Keep painting the TIFF and only apply the style

| Dimension | Assessment |
|-----------|------------|
| Complexity | Low. This is the current behavior. |
| Cost | Ongoing. `separate.py` grows while the darkroom does not change. |
| Scalability | Style path stays fast. Mask quality does not reach the user. |
| Team familiarity | Highest. |

**Pros:** No new Darktable risk. Global grade still appends cleanly.
**Cons:** The largest subsystem is a side file. Feature accept/skip, perch logic, and eye rollback are easy to misjudge because the panel and the darkroom disagree.

## Trade-off Analysis

Option B is the right end state and the wrong next step. The code already records why it cannot be done safely.

Option C preserves a clean style append and hides the edit the project is actually building. The uncommitted feature loop increases that gap.

Option A is the smallest change that makes the masked picture true in the UI. It keeps the original RAW. It accepts a baked duplicate as the cost of masks Darktable cannot store yet. When Option B becomes possible, the duplicate path can stop and the same plan can drive module masks instead of `render_finish`.

## Consequences

- The darkroom shows subject, eyes, background, and accepted features.
- Users get a duplicate per applied edit. The existing version record should name it.
- White balance and resolution of that duplicate need a single rule (camera WB, half-res unless `--full-res`) stated in the panel, not only in the README.
- `analyze` should stay the producer of both artifacts. Lua should gain one import of the TIFF onto the duplicate, and should stop implying that the style contains the masks.
- More color heuristics in `separate.py` are lower value than this handoff. A local matte model is the quality step after the TIFF is visible.

## Action Items

1. [ ] On apply, import `edited.tif` as a duplicate and open it, leaving the source RAW unchanged.
2. [ ] Say in the panel which file holds the masks and which file holds the style.
3. [ ] Keep `masks_can_attach()` false until a history-safe mask write exists.
4. [ ] After the duplicate is on screen, replace the color subject mask with a local matte seeded by the existing box.
