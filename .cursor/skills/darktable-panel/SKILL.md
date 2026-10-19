---
name: darktable-panel
description: >-
  Keep the Darktable AI editor buttons on the state that enables them. Use when
  Propose, Ask, Accept, Back, Next, or another panel control does the wrong
  action, stays off, or ignores a Lua change until restart.
---

# Darktable panel

One refresh decides every control. `set_panel_busy(true)` turns them off. A refresh inside that busy call cannot turn one back on. After the work, call `history_flags`, then `set_panel_busy(false)`.

A `local function` declared below the widget is not visible in its `clicked_callback`. Call helpers that are already declared above the widget, such as `read_recipe` and `edit_work_dir`.

## Propose and Ask

`asking` is `run_button.label == "Ask"`. Only that label uses `--stage chat`, and chat does not re-detect. A stored plan, a duplicate, or `chat: true` must not turn a Propose click into Ask.

Propose with text in the steer box exports and full-grades. Propose with an empty box and an existing plan returns that plan and does not grade again.

Copy keeps the highlighted history prefix (`history_end` rows) and drops every later row on the duplicate. The original keeps its full stack. Accept does not drop that tail.

## Back and Next

Back removes the last active row and appends it to `<work>/history-redo.json`. Next is on when that file contains the literal `"history"`. Read the redo file even when the sidecar read fails. An early return must not leave Next off after a successful Back.

Rows after `history_end` are not the Next stack. Next splices the redo file. Accept, and a revert that removed rows, write `{"items":[]}`.

## What a restart loads

Lua in `ai_editor.lua` loads when Darktable starts. Python under `ai_dt` loads on the next Propose, Ask, Accept, Back, Next, or Revert.

Compare the process start with the file write before telling the user to restart:

```
Get-Process darktable | Select-Object StartTime
Get-Item ai_editor.lua | Select-Object LastWriteTime
```

If the process is older than `ai_editor.lua`, the open panel is still the previous Lua. A Python fix is already live on the next click.
