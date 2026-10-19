---
name: darktable-history
description: >-
  Read and diagnose Darktable 5.6.2 sidecar history, drawn masks, and a module
  that reached the XMP but not the darkroom. Use when Accept, revert, a sidecar,
  an XMP, a history stack, or a mask did not change the picture or the history panel.
---

# Darktable history

The history panel shows the library and the open darkroom's copy. A rewritten sidecar can be correct while both of those stay on the old stack.

## Read before guessing

Run `scripts/read_sidecar.py` on the photo's `.xmp`. It prints `history_end`, each module, and any vignette numbers. Do not hand-roll a decoder in the shell. PowerShell eats quotes and backslashes.

```
.\.venv\Scripts\python.exe .cursor/skills/darktable-history/scripts/read_sidecar.py PATH.xmp
```

Compare with the library, read-only:

```
SELECT num, operation, multi_name, multi_priority, enabled
FROM history WHERE imgid=? ORDER BY num
```

`images.history_end` can sit past the last row. That means a module was counted, then dropped. Masks live in `masks_history`.

## What makes a module stick

- Live sidecars use blend version **14** and a 420-byte blend blob. Styles still say version 13. A new module must copy the sidecar's blend version. Version 13 of the same 420 bytes converts by copy, but a second instance with no pipeline slot is still dropped.
- `iop_order_list` already contains `colorzones,0`. Priority 1 has no slot unless the list gains `colorzones,1`. Use priority 0 when that operation is not already in the history.
- `history_end` may be one past the items after a drop. Append from the real item count. Rows after `history_end` stay in the stack. A new mask module is inserted at that marker, and the later rows move up one number.
- After revert, an empty mask list is `<rdf:Seq/>` with no closing tag. A drawn mask still has to be inserted there.
- One ellipse is the blend mask. Two or more need a group (type 4) whose id is the blend `mask_id`. Ellipse type is 32, version 6. Ids start at `0x10000000`. Every shape for one history item shares that item's `num`.
- An empty drawn mask (mode 3, id 0) hides the module.

## Apply

`image:apply_sidecar` calls `dt_history_load_and_apply` with history-only off. Lua true means the C call returned false (success). On the open darkroom image the reload deadlocks the pixelpipe. Leave the darkroom first.

Back removes the last active history row from the sidecar and stores that row in the photo's `history-redo.json`. Next turns on when that file still holds a row, and splices the same row back. Lowering `history_end` alone only moves the highlight; the row stays in the stack. Accept clears the redo file. The Next button is read after the panel leaves its busy state, because a refresh while busy leaves the button off.

Leaving does not drop develop's copy. Develop can write that old stack back over the database after a successful apply. `drop_cache` only drops mipmaps. There is no Lua call that deletes one history row.

`write_sidecar_files` is `on import`, so the panel will not pick up a file edit while the photo stays open.

## Check without the Accept button

Use a fresh config and library so the user's open library is untouched. Pass forward-slash paths. Backslashes are stripped and the jpeg is written under the repo.

```
darktable-cli IMAGE.CR2 SIDECAR.xmp OUTDIR --width 1024 --hq 0 --core --disable-opencl --configdir TEMP/cfg --library TEMP/cfg/library.db

`--library` belongs after `--core`. In front of `--core` it is the CLI switch that ignores the sidecar and reads the library instead.
```

Then query that temp `library.db`. A kept module is a history row with `num` below that image's `history_end`, plus its mask rows. Compare a plain export and the spliced export on the subject pixels and on a patch that should not move.
