# Task Plan: Improve darktable-auto-edit

## Goal

Make the masked picture the edit Darktable opens, with the existing features included, the original file left unchanged, and the plan order explicit.

## Next Step

Restart Darktable, run Propose, and choose a look in the preview window and on the panel.

## Current Phase

Phase 9

## Phases

### Phase 1: Map the system
- [x] Read README, entry points, and module boundaries
- [x] Record the current data flow
- [x] Note uncommitted work so recommendations do not fight it
- **Status:** complete

### Phase 2: Find the weak spots
- [x] Read pipeline, separation, reasoning, Lua, and preview
- [x] Check tests against the behaviors they claim to lock
- [x] Record concrete failure modes and coupling
- **Status:** complete

### Phase 3: Rank improvements
- [x] Apply system-design constraints (local, no upload, Darktable limits)
- [x] Compare a few options with trade-offs
- [x] Write a prioritized recommendation
- **Status:** complete

### Phase 4: Deliver
- [x] Re-read the plan and findings
- [x] Give the user the recommendation without implementing it
- **Status:** complete

### Phase 5: Handoff contract
- [x] Test that the recipe names the TIFF as the picture to open and leaves the source unchanged
- [x] A follow-up opened on a copy of that TIFF rereads the camera file
- [x] Panel text says which file holds the masks
- **Status:** complete

### Phase 6: One plan order
- [x] Test that a reasoner skip of the eyes loses to a found face, and a user skip stays
- [x] `analyze` calls that one function
- **Status:** complete

### Phase 7: Seeded subject matte
- [x] Test that a loose box around a colored subject does not paint the box corners
- [x] Use it where the subject mask is built
- **Status:** complete

### Phase 8: Open the picture from Darktable
- [x] Apply and Accept import a copy of the TIFF and open it
- [x] The source RAW is not forked and the style is not applied to it
- [x] A later steer on that opened file continues from the camera file
- **Status:** complete

### Phase 9: Contact sheet of every look
- [x] List every choice, including Accept and Skip, as one set of ids
- [x] A preview pick and a panel pick share that id
- [x] Darktable opens the sheet, and either place can choose
- **Status:** complete

## Key Questions

1. What is the real edit path from a Darktable click to a masked TIFF and a style? Answer: Lua exports a preview, `analyze` writes both artifacts, `apply_recipe` applies only the style.
2. Which parts of that path are heuristic, model-backed, or just text matching? Detectors and `decide_picture` are boxes plus pixels. Guides and recipes are term frequency. Gemini then Ollama propose stages. The steer grammar and review guards override them.
3. What is already changing in the working tree? Feature accept/skip, a finish gate, claim guards, and more separation and panel text. About 843 lines added on top of committed 1.4.0.
4. Which improvements raise edit quality without breaking the local-only constraint? Show `edited.tif` in Darktable first. Then one plan precedence. Then a local matte. Not another cloud model.

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Research only; do not change product code | The request is to think about improvements, not to implement them |
| Named plan under .planning/ | Parallel-task workflow; project had no existing plan |
| Prefer importing the TIFF as a duplicate over new mask APIs | `masks_can_attach()` is false because a sidecar would replace history |

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|
|       | 1       |            |

## Notes

- Uncommitted work already touches pipeline, separate, reasoner, Lua, preview, and new features/runbook modules.
- Constraints from README: nothing uploaded by default; masks cannot live in a .dtstyle; Darktable 5.0+.
