---
name: darktable-accept
description: >-
  Diagnose a Darktable Accept that did not write a mask, deleted history, or
  left the panel and the sidecar on different stacks. Use when Accept fails,
  history_end does not match, steps disappear, an eye or color mask is missing,
  or the work folder has masked.xmp, history-before-mask.xmp, or history.xmp.
---

# Darktable Accept

The panel message is the Python error. The live sidecar is what stuck. The work folder is what Accept tried. Read the work folder before editing the live `.xmp`.

Work folder: `C:/Users/asaf2/AppData/Local/darktable/ai-editor-work/<image id>/v<duplicate>`.

```
.\.venv\Scripts\python.exe .cursor/skills/darktable-accept/scripts/read_accept.py WORKDIR
```

| File | What it is |
| --- | --- |
| `history.xmp` | Sidecar copied before a new module, after the active-marker pass |
| `history-before-mask.xmp` | Live sidecar after `drawn_mask.py` succeeded, before `apply_sidecar` |
| `masked.xmp` | Spliced sidecar that Accept loads |
| `history-step.xmp` | Sidecar after Back or Next |
| `history-redo.json` | Row Back removed. Names only. The XML inside is not a second stack |

A failed splice does not call `apply_sidecar`. The live sidecar stays as it was. `history-before-mask.xmp` is written only after the splice succeeds.

The history panel can keep the old stack after a successful splice. Develop still holds the photo and writes that stack back. Before `apply_sidecar`, park develop on another photo and leave, then apply, then open the target. The sidecar and the library are the same stack only after that reload.

## Insert, do not delete

`history_end` is how many rows are active. Rows after it stay.

- Insert the new module at that index. Add 1 to every later `darktable:num` and to every `darktable:mask_num` or mask `darktable:num` that is at least that index. Set `history_end` to the insert index plus 1.
- When `history_end` is already at or past the item count, append with `max(num)+1`. Do not shift. A gap in the numbers must not turn an active row off.
- `collapse_history_to_cursor` returns the document unchanged and `removed 0`. Do not make it drop the tail again.

One eye step is one ellipse around every eye box. The panel writes it only when the step has `cx`, `cy`, `rx`, `ry`, and `exposure_ev`. Whole-frame color balance means the plan had no eye category.

## Library versus sidecar

The history panel shows the library, not the sidecar. They can disagree in count and in order. Read the library without writing it:

```
.\.venv\Scripts\python.exe .cursor/skills/darktable-accept/scripts/read_library.py 8182
```

`images.version` is the duplicate index. Lua calls that `duplicate_index`. `IMG_8182.CR2` and `IMG_8182_02.CR2` are different image ids. The original can still hold the stack a duplicate lost.

Do not write `library.db` or the open photo's sidecar while `darktable.exe` is running. Develop writes its in-memory stack back over both. Put probes in the temp directory as a `.py` file. PowerShell eats inline quotes. Delete the probe afterward. Do not commit `ai-editor-work/`.
