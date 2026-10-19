---
name: darktable-editor
description: >-
  Work in this repo's Darktable AI editor using the local install, Lua reload,
  and Accept path. Use when editing ai_editor.lua, ai_dt, Accept, Ask, Propose,
  revert, or the open photo's history.
---

# Darktable editor

Darktable 5.6.2 is `C:/Program Files/darktable/bin/darktable.exe`. The CLI is `darktable-cli.exe` beside it. Lua is API 9.7.0. This build has no `lfs`. The mask manager's AI object mask is a manual click tool. Eyes and hue Accept still write an ellipse. A subject Accept on a preview at least 256px on the long side uses `models/mask-object-sam21-small` and writes a path. Without that model it stays a vignette.

Config dir: `C:/Users/asaf2/AppData/Local/darktable`. Library: `library.db` there. Darktable 5.6 runs the bundled script manager, and that manager runs `luarc`. `luarc` must require only `ai_editor/ai_editor`. A `tools/script_manager.lua` in the config `lua` folder is loaded instead of the bundled one and never reaches `luarc`. The pre-5.6 scripts are in `lua-pre-5.6`. The lua folder `ai_editor` is a junction to this repo. A Lua change loads on the next Darktable start. Python plan, mask, and revert changes load on the next Propose, Ask, Accept, or Revert.

Python is `.venv/Scripts/python.exe`. System Python has no deps. Run focused tests with `.\.venv\Scripts\python.exe -m unittest`. Do not restart the full suite unless asked.

On Windows, one hidden worker started with Darktable runs every queued command. Do not call `dt.control.execute`, `os.execute`, or `io.popen` for a panel action. Each of those opens a command window. A view wait uses `dt.control.sleep`.

Darktable launches `ai_dt/drawn_mask.py` and `ai_dt/revert.py` by path. `sys.path[0]` is `ai_dt`, so those scripts must insert the repo root before importing the package. `revert.py` must not import `ai_dt`. Names it recognizes are hardcoded there and duplicated in `editor_history_names` in `ai_editor.lua`.

Accept writes one history step. A mask goes through the sidecar splice (`apply_masked_step`). Anything else is a style. Commit runs before apply. If apply fails, `--restore-step` puts that step back. Propose reads the photo and writes a plan even when one is already stored. Ask is only the follow-up labeled Ask; that path is `--stage chat` and does not re-detect. An old plan must not turn Propose into Ask.

Button timing is the workspace rule. Do not invent a fallback action when the enabling state is missing.

Work files for the open photo are under `C:/Users/asaf2/AppData/Local/darktable/ai-editor-work/<image id>/v<duplicate>`. The fox photo in this session is image 21, version 0. Do not commit `ai-editor-work/` probes or reference images.
