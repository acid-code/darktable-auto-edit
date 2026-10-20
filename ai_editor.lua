--[[
AI editor for Darktable.

Opening a photo writes the description when that box is empty. Ask writes
a plan of steps; there is no looks window or vibe box any more (use
darktable presets for a look). Cancel on this panel leaves the current plan
in place. Copy makes a duplicate when
you want one. Accept, Skip, and a chosen look stay on the photo that is
open. Back and Next move one history step, for every module. Revert AI edit
still removes the AI steps.

Load it from luarc with:  require "ai_editor/ai_editor"
The ai_processor.py file and the ai_dt package must stay next to this script.
]]

local dt = require "darktable"
local script_data = {}

script_data.metadata = {
  name = "AI editor",
  purpose = "Edit the open photo, or a copy, and step through its history",
  author = "ai_editor",
}

local MAX_IMAGES = 8
local PREFERENCE = "ai_editor"

local function script_directory()
  local source = debug.getinfo(1, "S").source or ""
  if source:sub(1, 1) == "@" then
    source = source:sub(2)
  end
  return source:match("^(.*)[/\\]") or "."
end

local json = assert(loadfile(script_directory() .. "/lib/json.lua"))()

local function path_separator()
  return package.config:sub(1, 1)
end

local function join_path(left, right)
  if left:sub(-1) == "/" or left:sub(-1) == "\\" then
    return left .. right
  end
  return left .. path_separator() .. right
end

local function on_windows()
  local name = tostring(dt.configuration.running_os or ""):lower()
  return name:find("windows", 1, true) ~= nil
end

local function in_flatpak()
  local probe = io.open("/.flatpak-info", "r")
  if not probe then
    return false
  end
  probe:close()
  return true
end

local function shell_quote(value)
  value = tostring(value)
  if value:find("\0", 1, true) then
    error("a path contains a null byte", 2)
  end
  if on_windows() then
    return '"' .. value:gsub('"', '""') .. '"'
  end
  return "'" .. value:gsub("'", "'\\''") .. "'"
end

-- os.execute("mkdir") opens a cmd window. The lighttable calls this on every
-- filmstrip move, so an existing folder must not start a process.
local lfs_ok, lfs = pcall(require, "lfs")
local made_dirs = {}
local hidden_mkdir

local function mkdir_p(path)
  if type(path) ~= "string" or path == "" or made_dirs[path] then
    return nil
  end
  -- Renaming a folder onto itself succeeds when it is already there.
  -- This build of Lua has no filesystem module, and mkdir on an existing
  -- folder opens a command window and prints an error.
  if os.rename(path, path) then
    made_dirs[path] = true
    return nil
  end
  if lfs_ok and lfs.attributes and lfs.attributes(path, "mode") == "directory" then
    made_dirs[path] = true
    return nil
  end
  if lfs_ok and lfs.mkdir then
    local parent = path:match("^(.*[/\\])[^/\\]+$")
    if parent then
      parent = parent:gsub("[/\\]+$", "")
      if parent ~= "" and parent ~= path then
        local err = mkdir_p(parent)
        if err then
          return err
        end
      end
    end
    lfs.mkdir(path)
    if lfs.attributes(path, "mode") == "directory" then
      made_dirs[path] = true
      return nil
    end
  end
  if on_windows() and hidden_mkdir and hidden_mkdir(path) then
    made_dirs[path] = true
    return nil
  end
  if on_windows() then
    os.execute('mkdir "' .. path:gsub("/", "\\") .. '"')
  else
    os.execute("mkdir -p " .. shell_quote(path))
  end
  made_dirs[path] = true
  return nil
end

local function read_file(path)
  local handle, open_error = io.open(path, "rb")
  if not handle then
    return nil, open_error or ("cannot open " .. path)
  end
  local contents = handle:read("*a")
  handle:close()
  return contents
end

local function write_file(path, contents)
  local handle, open_error = io.open(path, "wb")
  if not handle then
    return open_error or ("cannot write " .. path)
  end
  handle:write(contents)
  handle:close()
  return nil
end

local function tail(text, limit)
  text = tostring(text or ""):gsub("%s+$", "")
  if #text <= limit then
    return text
  end
  return text:sub(#text - limit + 1)
end

local function yield_ui()
  if dt.control and type(dt.control.sleep) == "function" then
    pcall(dt.control.sleep, 40)
  end
end

local DEFAULTS = {
  python = "python3",
  detector = "auto",
  advisor = "auto",
  yolo_model = "yolov8n.pt",
  ollama_model = "llava",
  prompt = "",
  preview = 1280,
  vibe = "",
  preset = "",
  steer = "",
}

local function register_preferences()
  pcall(
    dt.preferences.register,
    PREFERENCE,
    "python",
    "string",
    "AI editor: Python command",
    "Executable that runs ai_processor.py. Windows often needs python or py. Inside Flatpak this is the host executable.",
    DEFAULTS.python
  )
  pcall(
    dt.preferences.register,
    PREFERENCE,
    "detector",
    "enum",
    "AI editor: subject detector",
    "auto uses YOLO and MediaPipe when they are installed, and always includes the built-in heuristic.",
    "auto",
    "auto",
    "yolo",
    "mediapipe",
    "heuristic"
  )
  pcall(
    dt.preferences.register,
    PREFERENCE,
    "advisor",
    "enum",
    "AI editor: lighting advisor",
    "auto measures the preview. When MISTRAL_API_KEY is set it blends in a Mistral vision suggestion. Otherwise, when Ollama is running, it blends in a LLaVA suggestion.",
    "auto",
    "auto",
    "ollama",
    "histogram"
  )
  pcall(
    dt.preferences.register,
    PREFERENCE,
    "yolo_model",
    "string",
    "AI editor: YOLO model",
    "Ultralytics model name or path. yolov8n.pt is downloaded on first use.",
    DEFAULTS.yolo_model
  )
  pcall(
    dt.preferences.register,
    PREFERENCE,
    "ollama_model",
    "string",
    "AI editor: Ollama model",
    "Local vision model, for example llava or llava:13b.",
    DEFAULTS.ollama_model
  )
  pcall(
    dt.preferences.register,
    PREFERENCE,
    "prompt",
    "string",
    "AI editor: prompt file",
    "Optional replacement for ai_dt/prompts/llava_lighting.txt. Leave empty for the built-in prompt.",
    DEFAULTS.prompt
  )
  pcall(
    dt.preferences.register,
    PREFERENCE,
    "preview",
    "integer",
    "AI editor: preview size",
    "Longest edge, in pixels, of the JPEG sent to the analyzer.",
    DEFAULTS.preview,
    256,
    2048
  )
  pcall(
    dt.preferences.register,
    PREFERENCE,
    "vibe",
    "string",
    "AI editor: vibe",
    "Words typed in the Vibe box on the AI Editor panel. Leave blank for a faithful correction. Examples: warm, moody, quiet, provia, velvia, astia, classic chrome, acros, pro neg, eterna, nostalgic neg, reala ace. fuji or fujicam picks one film simulation for the picture.",
    DEFAULTS.vibe
  )
  pcall(
    dt.preferences.register,
    PREFERENCE,
    "steer",
    "string",
    "AI editor: steer",
    "Explicit clauses: strong or epic, darken or lift a target, gradient, revert or skip a stage. Unknown words are ignored.",
    DEFAULTS.steer
  )
  pcall(
    dt.preferences.register,
    PREFERENCE,
    "preset",
    "string",
    "AI editor: preset",
    "Preset chosen on the AI Editor panel. Empty uses the typed vibe, or a faithful correction when that box is also empty.",
    DEFAULTS.preset
  )
end

local function preference(name, kind)
  local ok, value = pcall(dt.preferences.read, PREFERENCE, name, kind)
  if ok and value ~= nil and value ~= "" then
    return value
  end
  return DEFAULTS[name]
end

local function selected_images()
  local images = {}
  local action = dt.gui.action_images
  if type(action) == "table" then
    for _, image in ipairs(action) do
      images[#images + 1] = image
    end
  end
  if #images > 0 then
    return images
  end
  local selection = dt.gui.selection
  if type(selection) == "function" then
    local ok, result = pcall(selection)
    selection = ok and result or nil
  end
  if type(selection) == "table" then
    for _, image in ipairs(selection) do
      images[#images + 1] = image
    end
  end
  return images
end

local function find_style(name)
  for _, style in ipairs(dt.styles) do
    if style.name == name then
      return style
    end
  end
  return nil
end

local function file_size(path)
  local handle = io.open(path, "rb")
  if not handle then
    return 0
  end
  local size = handle:seek("end") or 0
  handle:close()
  return size
end

local function export_preview(image, destination, edge)
  local jpeg = dt.new_format("jpeg")
  jpeg.quality = 82
  jpeg.max_width = edge
  jpeg.max_height = edge
  pcall(function()
    jpeg:write_image(image, destination, false)
  end)
  if file_size(destination) < 32 then
    return "Darktable did not write a preview JPEG"
  end
  return nil
end

local function image_source_path(image)
  local folder = image.path or ""
  local name = image.filename or ""
  if name == "" then
    return ""
  end
  if folder == "" then
    return name
  end
  local last = folder:sub(-1)
  if last == "/" or last == "\\" then
    return folder .. name
  end
  local sep = "/"
  if folder:find("\\", 1, true) and not folder:find("/", 1, true) then
    sep = "\\"
  end
  return folder .. sep .. name
end

local run_captured

-- darktable 5.4 and 5.6 read the same params; 5.6 adds modules to the
-- pipeline order (colorharmonizer). Python writes the order of this version.
local function dt_version_arg()
  local version = tostring(dt.configuration.version or ""):match("(%d+%.%d+[%.%d]*)") or ""
  if version == "" then
    return ""
  end
  return " --dt-version " .. shell_quote(version)
end

local function run_processor(preview, work_dir, original, vibe, steer, options)
  local processor = join_path(script_directory(), "ai_processor.py")
  if file_size(processor) < 10 then
    return nil, "ai_processor.py is not next to ai_editor.lua"
  end
  local parts = {
    in_flatpak() and "flatpak-spawn --host " or "",
    shell_quote(preference("python", "string")),
    " ",
    shell_quote(processor),
    dt_version_arg(),
    " ",
    shell_quote(preview),
    " --output-dir ",
    shell_quote(work_dir),
    " --detector ",
    shell_quote(preference("detector", "enum")),
    " --advisor ",
    shell_quote(preference("advisor", "enum")),
    " --yolo-model ",
    shell_quote(preference("yolo_model", "string")),
    " --ollama-model ",
    shell_quote(preference("ollama_model", "string")),
  }
  local prompt = preference("prompt", "string")
  if prompt ~= "" then
    parts[#parts + 1] = " --prompt "
    parts[#parts + 1] = shell_quote(prompt)
  end
  if type(vibe) ~= "string" then
    vibe = preference("vibe", "string")
  end
  -- Typed text goes through a UTF-8 file. On Windows io.popen runs cmd.exe
  -- in the ANSI code page, which turned "duck’s" into "duckÔÇÖs" in argv.
  local function text_arg(flag, value, file_name)
    if type(value) ~= "string" or value == "" then
      return
    end
    local file = join_path(work_dir, file_name)
    if write_file(file, value) == nil then
      parts[#parts + 1] = " " .. flag .. "-file "
      parts[#parts + 1] = shell_quote(file)
    else
      parts[#parts + 1] = " " .. flag .. " "
      parts[#parts + 1] = shell_quote(value)
    end
  end
  text_arg("--vibe", vibe, "arg-vibe.txt")
  text_arg("--steer", steer, "arg-request.txt")
  text_arg("--scene", options.scene, "arg-description.txt")
  if options.rewrite_scene == true then
    parts[#parts + 1] = " --rewrite-scene"
  end
  options = type(options) == "table" and options or {}
  if type(options.look) == "string" and options.look ~= "" then
    parts[#parts + 1] = " --look "
    parts[#parts + 1] = shell_quote(options.look)
  end
  if options.image_id ~= nil then
    parts[#parts + 1] = " --image-id "
    parts[#parts + 1] = shell_quote(tostring(options.image_id))
  end
  if options.duplicate_index ~= nil then
    parts[#parts + 1] = " --duplicate-index "
    parts[#parts + 1] = shell_quote(tostring(options.duplicate_index))
  end
  if options.parent_id ~= nil then
    parts[#parts + 1] = " --parent-id "
    parts[#parts + 1] = shell_quote(tostring(options.parent_id))
  end
  if options.parent_index ~= nil then
    parts[#parts + 1] = " --parent-index "
    parts[#parts + 1] = shell_quote(tostring(options.parent_index))
  end
  if type(options.library) == "string" and options.library ~= "" then
    parts[#parts + 1] = " --library "
    parts[#parts + 1] = shell_quote(options.library)
  end
  if type(options.history_xmp) == "string" and options.history_xmp ~= "" then
    parts[#parts + 1] = " --history-xmp "
    parts[#parts + 1] = shell_quote(options.history_xmp)
  end
  if type(options.stage) == "string" and options.stage ~= "" and options.stage ~= "all" then
    parts[#parts + 1] = " --stage "
    parts[#parts + 1] = shell_quote(options.stage)
  end
  if type(original) == "string" and original ~= "" then
    parts[#parts + 1] = " --original "
    parts[#parts + 1] = shell_quote(original)
  end
  local command = table.concat(parts)
  local output, exit_code = run_captured(command, work_dir)
  if not output then
    return nil, exit_code or "could not start Python"
  end
  write_file(join_path(work_dir, "run.log"), output)
  if exit_code ~= 0 then
    local body = tostring(output or ""):gsub("%s+$", "")
    if body == "" then
      body = "Python exited with code " .. tostring(exit_code) .. " and wrote no error."
    else
      if #body > 2000 then
        body = body:sub(#body - 1999)
      end
      body = "Python exited with code " .. tostring(exit_code) .. "\n" .. body
    end
    return nil, body
  end
  return output
end

-- A style cannot carry drawn masks. It still carries the global grade,
-- and that grade is what lands on the duplicate.
local function apply_recipe(image, work_dir)
  if continue_from_viewed_history then
    continue_from_viewed_history(image)
  end
  local recipe_path = join_path(work_dir, "output.json")
  local contents, read_error = read_file(recipe_path)
  if not contents then
    return nil, read_error
  end
  local ok, recipe = pcall(json.decode, contents)
  if not ok then
    return nil, "Could not read output.json: " .. tostring(recipe)
  end
  local style_info = recipe.style
  if type(style_info) ~= "table" or type(style_info.name) ~= "string" or type(style_info.file) ~= "string" then
    return nil, "output.json has no style name"
  end
  if file_size(style_info.file) < 32 then
    return nil, "the style file is missing: " .. style_info.file
  end
  local imported, import_error = pcall(dt.styles.import, style_info.file)
  if not imported then
    return nil, "Darktable could not import the style: " .. tostring(import_error)
  end
  local style = find_style(style_info.name)
  if not style then
    return nil, "Imported style was not found: " .. style_info.name
  end
  local applied, apply_error = pcall(function()
    dt.styles.apply(style, image)
  end)
  if not applied then
    return nil, "Could not apply " .. style_info.name .. ": " .. tostring(apply_error)
      .. " The style is still in the styles list."
  end
  pcall(dt.styles.delete, style)
  return recipe
end

local function summary(recipe, image)
  local analysis = type(recipe.analysis) == "table" and recipe.analysis or {}
  local atmosphere = type(analysis.atmosphere) == "table" and analysis.atmosphere or {}
  local subjects = type(analysis.subjects) == "table" and analysis.subjects or {}
  local backends = type(recipe.backends) == "table" and recipe.backends or {}
  local match = type(recipe.recipe_match) == "table" and recipe.recipe_match or {}
  local look = ""
  if type(match.title) == "string" and match.title ~= "" then
    look = string.format("\nLook: %s", match.title)
  end
  local picture = type(recipe.picture) == "table" and recipe.picture or {}
  local guide = type(recipe.guide) == "table" and recipe.guide or {}
  local named = ""
  if type(picture.type) == "string" and picture.type ~= "" and type(guide.title) == "string" and guide.title ~= "" then
    local kept = ""
    local features = recipe.features
    if type(features) == "table" then
      local names = {}
      for _, feature in ipairs(features) do
        if type(feature) == "string" and feature ~= "" then
          names[#names + 1] = feature
        end
      end
      if #names > 0 then
        kept = ", kept " .. table.concat(names, ", ")
      end
    end
    named = string.format("%s — %s guide%s\n", picture.type, guide.title, kept)
  end
  local input = type(recipe.input) == "table" and recipe.input or {}
  local separation = type(recipe.separation) == "table" and recipe.separation or {}
  local reading_line = ""
  local reading = type(recipe.reading) == "table" and recipe.reading or nil
  if reading and type(reading.sentence) == "string" and reading.sentence ~= "" then
    reading_line = reading.sentence .. "\n"
  end
  local reasoning_line = ""
  local decision = ""
  local reasoning = type(recipe.reasoning) == "table" and recipe.reasoning or nil
  if reasoning and type(reasoning.decision) == "string" and reasoning.decision ~= "" then
    decision = "\n" .. reasoning.decision
  end
  if reasoning and type(reasoning.description) == "string" and reasoning.description ~= "" then
    reasoning_line = "\n" .. reasoning.description
    local shown = 0
    local function add_why(items, limit)
      if type(items) ~= "table" then
        return
      end
      for _, item in ipairs(items) do
        if shown >= limit then
          return
        end
        if type(item) == "table" and type(item.why) == "string" and item.why ~= "" then
          if item.place ~= "frame" or items == reasoning.set_aside then
            reasoning_line = reasoning_line .. "\n" .. item.why
            shown = shown + 1
          end
        end
      end
    end
    add_why(reasoning.used, 3)
    add_why(reasoning.set_aside, 5)
  end
  local render = ""
  if input.kind == "raw" then
    render = "\nRead the RAW and wrote edited.tif."
  else
    render = "\nWrote edited.tif."
  end
  if separation.applied == true then
    local together = {}
    together[#together + 1] = "subject"
    if (tonumber(separation.eyes) or 0) > 0 then
      together[#together + 1] = "eyes"
    end
    local ran = separation.stages_ran
    if type(ran) == "table" then
      for _, stage in ipairs(ran) do
        if stage == "reflection" then
          together[#together + 1] = "reflection"
        end
      end
    end
    render = render .. " Lifted together: " .. table.concat(together, ", ") .. "."
    if separation.background == "defocus" then
      render = render .. " Background defocused."
    end
    render = render .. " The history vignette only darkens around the subject. The brightness of these layers is in edited.tif."
  end
  return string.format(
    "%s%s%s\n%+.2f EV, contrast %+.2f, warmth %+.2f\n%d subject region(s) via %s / %s%s%s%s%s\nHistory updated for %s.",
    reading_line,
    named,
    tostring(analysis.mood or "Edit applied"),
    tonumber(atmosphere.exposure_ev) or 0,
    tonumber(atmosphere.contrast) or 0,
    tonumber(atmosphere.temperature_shift) or 0,
    #subjects,
    tostring(backends.detector or "detector"),
    tostring(backends.advisor or "advisor"),
    look,
    decision,
    reasoning_line,
    render,
    image.filename or "image"
  )
end

local status_label
local look_name_label
local run_button
local revert_button
local vibe_entry
local scene_entry
local scene_text
local apply_masked_step
local discard_history_redo
local rewrite_scene_button
local rewrite_scene_once = false
local describe_image
local describing = false
local steer_entry
local preset_box
local preset_ids = {}
local preset_queries = {}
local choice_buttons = {}
local choice_ids = {}
local accept_button
local skip_button
local looks_button
local cancel_button
local copy_button
local back_button
local next_button
local sidecar_file
local feature_waiting = false
local looks_pending = false
local history_back = false
local history_next = false
local can_revert = false
local can_remove = false
local has_photo = false
local remove_button
local shown_recipe = nil
local panel_is_busy = false
local sync_waiting_step
local continue_from_viewed_history

-- Global on purpose. The main chunk is already at Lua's 200-local limit.
function duplicate_index_of(image)
  local index = tonumber(image.duplicate_index)
  if index == nil then
    return 0
  end
  return index
end

function library_path()
  return join_path(dt.configuration.config_dir, "library.db")
end

-- Same layout as ai_dt.versions.version_work_dir: <image id>/v<duplicate index>
function work_dir_for(image)
  local root = join_path(dt.configuration.config_dir, "ai-editor-work")
  local folder = join_path(root, tostring(image.id))
  return join_path(folder, "v" .. tostring(duplicate_index_of(image)))
end

-- An opened TIFF lives in the work folder that already holds output.json.
-- Follow-up steers write there, so the camera file is read again.
function edit_work_dir(image)
  local path = image_source_path(image)
  local lower = path:lower()
  local parent = path:match("^(.*)[/\\][^/\\]+$")
  if parent and lower:find("[/\\]opened[/\\][^/\\]+%.tif$") then
    local work = parent:match("^(.*)[/\\]opened$")
    if work and work ~= "" then
      return work
    end
  end
  if parent and lower:find("edited%.tif$") then
    return parent
  end
  return work_dir_for(image)
end

-- current_view() pushes a module entry. view.id is the module name
-- ("darkroom" or "lighttable"). Object identity is only a fallback.
function view_name(view)
  if view == nil then
    return nil
  end
  local ok, id = pcall(function()
    return view.id
  end)
  if ok and type(id) == "string" and id ~= "" then
    return id
  end
  if not dt.gui.views then
    return nil
  end
  if view == dt.gui.views.darkroom then
    return "darkroom"
  end
  if view == dt.gui.views.lighttable then
    return "lighttable"
  end
  return nil
end

function current_view_name()
  if type(dt.gui.current_view) ~= "function" then
    return nil
  end
  local ok, view = pcall(dt.gui.current_view)
  if not ok then
    return nil
  end
  return view_name(view)
end

function in_darkroom()
  return current_view_name() == "darkroom"
end

function darkroom_image()
  if not in_darkroom() then
    return nil
  end
  local ok, image = pcall(function()
    return dt.gui.views.darkroom.display_image()
  end)
  if ok and (type(image) == "userdata" or type(image) == "table") then
    return image
  end
  return nil
end

local LOOK_TITLES = {
  faithful = "Faithful",
  ["shadow-lift"] = "Open shadows",
  ["highlight-hold"] = "Hold highlights",
  warmer = "Warmer",
  ["more-color"] = "More color",
  ["sky-gradient"] = "Sky gradient",
  ["subject-lift"] = "Subject lift",
  ["eye-detail"] = "Eye detail",
  ["hold-reflection"] = "Hold the reflection",
}

function look_title_of(record)
  if type(record) ~= "table" then
    return ""
  end
  if type(record.look_title) == "string" and record.look_title ~= "" then
    return record.look_title
  end
  local id = record.look_id
  if type(id) == "string" and LOOK_TITLES[id] then
    return LOOK_TITLES[id]
  end
  return ""
end

function read_version_record(image)
  if image == nil then
    return nil
  end
  local contents = read_file(join_path(edit_work_dir(image), "version.json"))
  if not contents then
    return nil
  end
  local ok, record = pcall(json.decode, contents)
  if not ok or type(record) ~= "table" then
    return nil
  end
  return record
end

local set_version_name

function show_loaded_look(image)
  if look_name_label == nil then
    return
  end
  if image == nil then
    look_name_label.label = "Look: —"
    return
  end
  local title = look_title_of(read_version_record(image))
  if title == "" then
    look_name_label.label = "Look: not named yet"
    return
  end
  look_name_label.label = "Look: " .. title
  local current = ""
  pcall(function()
    current = image.version_name or ""
  end)
  if current ~= title and (current == "" or current:find("Faithful", 1, true) or #current > 40) then
    set_version_name(image, title)
  end
end

set_version_name = function(image, sentence)
  if type(sentence) ~= "string" or sentence == "" then
    return
  end
  if #sentence > 240 then
    sentence = sentence:sub(1, 240)
  end
  pcall(function()
    image.version_name = sentence
  end)
end

function fork_version(image)
  local function different(copy)
    return copy and tonumber(copy.id) ~= tonumber(image.id)
  end
  if type(image.duplicate_with_history) == "function" then
    local ok, copy = pcall(function()
      return image:duplicate_with_history()
    end)
    if ok and different(copy) then
      return copy
    end
  end
  if dt.database and type(dt.database.duplicate_with_history) == "function" then
    local ok, copy = pcall(dt.database.duplicate_with_history, image)
    if ok and different(copy) then
      return copy
    end
  end
  return nil, "Darktable could not duplicate this image"
end

-- Select the new duplicate and open it. Reading a missing field on a
-- Darktable view throws, so center_on stays inside pcall. A throw here
-- used to leave the darkroom on the original and apply the look again.
function show_in_darkroom(image)
  pcall(function()
    if type(dt.gui.selection) == "function" then
      dt.gui.selection({ image })
    end
  end)
  pcall(function()
    dt.gui.views.lighttable.center_on(image)
  end)
  pcall(function()
    if type(dt.gui.current_view) == "function" and dt.gui.views and dt.gui.views.darkroom then
      dt.gui.current_view(dt.gui.views.darkroom)
    end
  end)
  yield_ui()
  pcall(function()
    dt.gui.views.darkroom.display_image(image)
  end)
  yield_ui()
  local shown_ok, shown = pcall(function()
    return dt.gui.views.darkroom.display_image()
  end)
  if (not shown_ok) or shown == nil or shown.id ~= image.id then
    pcall(function()
      dt.gui.views.darkroom.display_image(image)
    end)
  end
end

function read_recipe(work_dir)
  local recipe_path = join_path(work_dir, "output.json")
  local contents, read_error = read_file(recipe_path)
  if not contents then
    return nil, read_error
  end
  local ok, recipe = pcall(json.decode, contents)
  if not ok then
    return nil, "Could not read output.json: " .. tostring(recipe)
  end
  return recipe
end

function apply_step(image, work_dir)
  -- One history module for the step just accepted. Skip writes no file.
  if continue_from_viewed_history then
    continue_from_viewed_history(image)
  end
  local recipe = read_recipe(work_dir)
  if type(recipe) ~= "table" then
    return nil
  end
  local mask = recipe.step_mask
  if type(mask) == "table" then
    if type(apply_masked_step) ~= "function" then
      return nil, "The eye mask could not be applied. The darkroom history was left as it is."
    end
    return apply_masked_step(image, work_dir, mask)
  end
  local info = recipe.step_style
  if info == nil then
    -- The commit left this step off; feature_note says why.
    return nil
  end
  if type(info) ~= "table" or type(info.file) ~= "string" or type(info.name) ~= "string" then
    return nil, "The step has no style to apply, so nothing was added. The darkroom history was left as it is."
  end
  if file_size(info.file) < 32 then
    return nil, "The step file is missing or empty (" .. info.file .. "), so nothing was added."
  end
  local imported, import_error = pcall(dt.styles.import, info.file)
  if not imported then
    return nil, "Darktable could not import the step: " .. tostring(import_error)
  end
  local style = find_style(info.name)
  if not style then
    return nil, "Imported step was not found: " .. info.name
  end
  local applied, apply_error = pcall(function()
    dt.styles.apply(style, image)
  end)
  if not applied then
    return nil, "Could not apply the step: " .. tostring(apply_error)
  end
  pcall(dt.styles.delete, style)
  discard_history_redo(image)
  return true
end

local function version_sentence_of(recipe)
  local record = type(recipe.version_record) == "table" and recipe.version_record or nil
  if record and type(record.sentence) == "string" and record.sentence ~= "" then
    return record.sentence
  end
  local plan = type(recipe.plan) == "table" and recipe.plan or {}
  if type(plan.label) == "string" and plan.label ~= "" then
    return plan.label
  end
  return ""
end

-- The panel shows every line. The duplicate version name stays short.
local function show_scene(recipe)
  if scene_entry == nil then
    return
  end
  local scene = type(recipe) == "table" and recipe.scene or nil
  if type(scene) ~= "table" or type(scene.summary) ~= "string" then
    scene_entry.text = ""
    return
  end
  scene_entry.text = scene.summary
end

local function panel_body(recipe)
  if type(recipe) ~= "table" then
    return ""
  end
  if type(recipe.status_text) == "string" and recipe.status_text ~= "" then
    return recipe.status_text
  end
  local lines = {}
  if type(recipe.picture_backend) == "string" and recipe.picture_backend ~= "" then
    lines[#lines + 1] = recipe.picture_backend
  end
  if type(recipe.plan_backend) == "string" and recipe.plan_backend ~= "" then
    lines[#lines + 1] = recipe.plan_backend
  end
  local sentence = version_sentence_of(recipe)
  if sentence ~= "" then
    lines[#lines + 1] = sentence
  end
  return table.concat(lines, "\n")
end

-- The request box has no change signal in darktable 5.6 (lua/widget/entry.c
-- registers only text, placeholder, is_password and editable). The status
-- is recomputed from it whenever the panel refreshes instead.
-- One table keeps the main chunk under Lua's 200-local limit.
local request_ui = { was_changed = nil }

function request_ui.key(text)
  return tostring(text or ""):gsub("%s+", " "):gsub("^ ", ""):gsub(" $", ""):lower()
end

function request_ui.typed()
  if not steer_entry then
    return nil
  end
  return tostring(steer_entry.text or "")
end

-- Same rule as status_text.request_changed: a plan without plan_request
-- (written before plans recorded it) counts as unchanged.
function request_ui.changed(recipe)
  if type(recipe) ~= "table" or type(recipe.plan_request) ~= "string" then
    return false
  end
  local typed = request_ui.typed()
  if typed == nil then
    return false
  end
  return request_ui.key(typed) ~= request_ui.key(recipe.plan_request)
end

-- Same rule as status_text.description_changed.
function request_ui.described(recipe)
  if type(recipe) ~= "table" or type(recipe.plan_description) ~= "string" or type(scene_text) ~= "function" then
    return false
  end
  return request_ui.key(scene_text()) ~= request_ui.key(recipe.plan_description)
end

function request_ui.waiting(recipe)
  if type(recipe) ~= "table" or type(recipe.features_next) ~= "string" or recipe.features_next == "" then
    return ""
  end
  local title = recipe.features_next_title
  if type(title) == "string" and title ~= "" then
    return title
  end
  return recipe.features_next
end

-- The plan text, kept as the current plan, with the Accept/Skip line for
-- the request box as it is now (status_text.accept_skip_lines).
local function plan_status(recipe)
  local body = panel_body(recipe)
  if type(recipe) ~= "table" then
    return body
  end
  local state = recipe.accept_skip
  local asked_changed = request_ui.changed(recipe)
  local described_changed = (not asked_changed) and request_ui.described(recipe)
  local changed = asked_changed or described_changed
  if type(state) == "table" and type(state.line) == "string" and type(state.changed_line) == "string" then
    local described_line = type(state.description_changed_line) == "string" and state.description_changed_line or state.changed_line
    local wanted = state.line
    if asked_changed then
      wanted = state.changed_line
    elseif described_changed then
      wanted = described_line
    end
    for _, other in ipairs({ state.line, state.changed_line, described_line }) do
      if other ~= wanted then
        local at, finish = body:find(other, 1, true)
        if at then
          return body:sub(1, at - 1) .. wanted .. body:sub(finish + 1)
        end
      end
    end
    if body:find(wanted, 1, true) then
      return body
    end
    return body ~= "" and (body .. "\n" .. wanted) or wanted
  end
  if changed then
    local step = request_ui.waiting(recipe)
    local line = step ~= ""
      and ("Request changed — press Ask for a new plan; Accept/Skip still apply to: " .. step)
      or "Request changed — press Ask for a new plan. No step is waiting, so Accept and Skip are off."
    return body ~= "" and (body .. "\n" .. line) or line
  end
  return body
end

local function failure_text(work, failure)
  local detail = tostring(failure or "unknown error")
  detail = detail:gsub("^%s+", ""):gsub("%s+$", "")
  if detail == "" or detail == "Failure" then
    detail = "The analysis failed and did not report an error."
  end
  if #detail > 2000 then
    detail = detail:sub(1, 2000)
  end
  local recipe = nil
  if type(work) == "string" and work ~= "" then
    recipe = read_recipe(work)
  end
  if type(recipe) == "table" then
    local bits = {}
    if type(recipe.picture_backend) == "string" and recipe.picture_backend ~= "" then
      bits[#bits + 1] = recipe.picture_backend
    end
    if type(recipe.plan_backend) == "string" and recipe.plan_backend ~= "" then
      bits[#bits + 1] = recipe.plan_backend
    end
    if #bits > 0 then
      return table.concat(bits, "\n") .. "\n" .. detail
    end
  end
  return detail
end

local function processor_options(image, parent)
  local options = {
    image_id = image.id,
    duplicate_index = duplicate_index_of(image),
    library = library_path(),
  }
  if parent then
    options.parent_id = parent.id
    options.parent_index = duplicate_index_of(parent)
  end
  return options
end

local progress_job = nil

local function close_progress()
  if not progress_job then
    return
  end
  pcall(function()
    progress_job.valid = false
  end)
  progress_job = nil
end

-- The panel text is set before each blocking command. yield_ui() is one
-- short pause so the label can paint. A long sleep loop is not used.
-- create_job is Darktable's progress bar (there is no panel progress widget).
local function show_progress(message, fraction)
  status_label.text = message
  if progress_job then
    pcall(function()
      progress_job.valid = false
    end)
    progress_job = nil
  end
  if dt.gui and type(dt.gui.create_job) == "function" then
    local ok, job = pcall(dt.gui.create_job, message, true)
    if ok and job ~= nil and type(job) ~= "string" and type(job) ~= "number" then
      progress_job = job
      pcall(function()
        job.percent = fraction
      end)
    end
  end
  yield_ui()
end

local function read_json_file(path)
  local contents = read_file(path)
  if not contents then
    return nil
  end
  local ok, data = pcall(json.decode, contents)
  if not ok or type(data) ~= "table" then
    return nil
  end
  return data
end

local function copy_file(source, destination, overwrite)
  if file_size(source) < 2 then
    return false
  end
  if not overwrite and file_size(destination) > 2 then
    return false
  end
  local contents = read_file(source)
  if not contents then
    return false
  end
  return write_file(destination, contents) == nil
end

local function json_string(value)
  return (tostring(value or ""):gsub("\\", "\\\\"):gsub("\"", "\\\""):gsub("\r", "\\r"):gsub("\n", "\\n"))
end

local function picture_key(path)
  local text = tostring(path or ""):gsub("/", "\\"):lower()
  local hash = 5381
  for index = 1, #text do
    hash = math.floor((hash * 33 + text:byte(index)) % 4294967296)
  end
  return string.format("%08x", hash)
end

local function picture_cache_path(source)
  local root = join_path(dt.configuration.config_dir, "ai-editor-work")
  local folder = join_path(root, "cache")
  mkdir_p(folder)
  return join_path(folder, picture_key(source) .. ".json")
end

local function picture_has_look(work)
  local context = read_json_file(join_path(work, "context.json"))
  return type(context) == "table" and context.look_applied == true
end

local function is_original(image)
  return duplicate_index_of(image) == 0
end

local function original_of(image)
  local path = tostring(image.path or "")
  local filename = tostring(image.filename or "")
  local function is_root(candidate)
    if candidate == nil or duplicate_index_of(candidate) ~= 0 then
      return false
    end
    return tostring(candidate.path or "") == path and tostring(candidate.filename or "") == filename
  end
  local leader_ok, leader = pcall(function()
    return image.group_leader
  end)
  if leader_ok and is_root(leader) then
    return leader
  end
  if type(image.get_group_members) == "function" then
    local members_ok, members = pcall(function()
      return image:get_group_members()
    end)
    if members_ok and type(members) == "table" then
      for _, candidate in ipairs(members) do
        if is_root(candidate) then
          return candidate
        end
      end
    end
  end
  if type(dt.database) == "table" then
    for _, candidate in ipairs(dt.database) do
      if is_root(candidate) then
        return candidate
      end
    end
  end
  return nil
end

-- A duplicate starts with an empty folder, so a later Ask would read the
-- photo as new and paint the whole look again. The cache remembers which
-- copy already has the grade.
local function restore_picture_context(image, work)
  if picture_has_look(work) then
    return true
  end
  local source = image_source_path(image)
  if source == "" then
    return false
  end
  local cached = read_json_file(picture_cache_path(source))
  if type(cached) ~= "table" or cached.look_applied ~= true then
    return false
  end
  if tonumber(cached.applied_image_id) ~= tonumber(image.id) then
    return false
  end
  local from = cached.work
  if type(from) ~= "string" or from == "" then
    return false
  end
  mkdir_p(work)
  for _, name in ipairs({ "output.json", "proposal.json", "stage.json", "preview.jpg", "edit.dtstyle" }) do
    copy_file(join_path(from, name), join_path(work, name))
  end
  if file_size(join_path(work, "output.json")) < 2 then
    return false
  end
  write_file(
    join_path(work, "context.json"),
    string.format(
      '{"look_applied":true,"look_id":"%s","source":"%s"}\n',
      json_string(cached.look_id),
      json_string(source)
    )
  )
  return true
end

local function remember_applied_look(source_image, copy, work, look)
  local copy_work = work_dir_for(copy)
  local mkdir_error = mkdir_p(copy_work)
  if mkdir_error then
    return
  end
  for _, name in ipairs({ "output.json", "proposal.json", "stage.json", "preview.jpg", "edit.dtstyle" }) do
    -- The copy may already have an older plan with no waiting step.
    -- Replace it with the plan just written, or Accept stays off.
    copy_file(join_path(work, name), join_path(copy_work, name), true)
  end
  local source = image_source_path(source_image)
  write_file(
    join_path(copy_work, "context.json"),
    string.format(
      '{"look_applied":true,"look_id":"%s","source":"%s"}\n',
      json_string(look),
      json_string(source)
    )
  )
  if source == "" then
    return
  end
  write_file(
    picture_cache_path(source),
    string.format(
      '{"look_applied":true,"look_id":"%s","source":"%s","applied_image_id":%d,"work":"%s","plan_work":"%s"}\n',
      json_string(look),
      json_string(source),
      tonumber(copy.id) or 0,
      json_string(copy_work),
      json_string(work)
    )
  )
end

local function run_staged(preview, work, original, vibe, steer, options)
  -- One Python process. Picture, Gemini, and the looks used to be separate
  -- launches, and each one paid interpreter startup again.
  options = type(options) == "table" and options or {}
  options.stage = "all"
  show_progress("Starting the edit…", 0.12)
  return run_processor(preview, work, original, vibe, steer, options)
end

local function propose_image(image, index, total, vibe, steer)
  local edge = tonumber(preference("preview", "integer")) or DEFAULTS.preview
  if edge < 256 then
    edge = 256
  elseif edge > 2048 then
    edge = 2048
  end
  local work = edit_work_dir(image)
  local mkdir_error = mkdir_p(work)
  if mkdir_error then
    return nil, mkdir_error
  end
  local steered = tostring(steer or ""):gsub("%s", "") ~= ""
  -- The camera XMP lists every copy, so it cannot say that this original already has the grade.
  local already = (not is_original(image)) and restore_picture_context(image, work)
  local existing = read_recipe(work)
  -- Ask is a follow-up. It keeps the saved subjects and does not grade again.
  -- Ask reads the photo even when this folder already has a plan.
  local asking = run_button ~= nil and run_button.label == "Ask"
  -- A follow-up needs the saved reading. With none (a new photo, or after
  -- Clear plan) Ask reads the photo and plans from scratch.
  if asking and (type(existing) ~= "table" or type(existing.plan) ~= "table") then
    asking = false
  end
  -- An edited description is a new plan even with the request box empty.
  local described = false
  if type(existing) == "table" and type(scene_text) == "function" then
    local before = existing.plan_description
    if type(before) ~= "string" then
      before = type(existing.scene) == "table" and existing.scene.summary or ""
    end
    described = scene_text() ~= "" and request_ui.key(scene_text()) ~= request_ui.key(before)
  end
  if asking and not steered and not described then
    if existing then
      existing.chat = true
      existing.choices = {}
      return existing, nil, true
    end
  end
  local options = processor_options(image, nil)
  options.scene = scene_text()
  if rewrite_scene_once then
    options.rewrite_scene = true
    rewrite_scene_once = false
  end
  if asking then
    show_progress("Asking for the steps…", 0.45)
    options.stage = "chat"
    -- The planner leaves out steps whose rows are already in this sidecar.
    local has_sidecar, sidecar = pcall(function()
      return image.sidecar
    end)
    if has_sidecar and type(sidecar) == "string" and sidecar ~= "" then
      options.history_xmp = sidecar
    end
    local _, run_error = run_processor("", work, image_source_path(image), vibe, steer, options)
    close_progress()
    if run_error then
      return nil, failure_text(work, run_error)
    end
    local recipe = read_recipe(work)
    if recipe then
      recipe.chat = true
      recipe.choices = {}
    end
    return recipe, nil, true
  end
  if not steered and type(existing) == "table" and (already or existing.chat == true) then
    existing.choices = {}
    return existing, nil, false
  end
  local preview = join_path(work, "preview.jpg")
  status_label.text = string.format("Exporting %s (%d of %d)…", image.filename or "image", index, total)
  yield_ui()
  local export_error = export_preview(image, preview, edge)
  if export_error then
    return nil, export_error
  end
  local _, run_error = run_staged(preview, work, image_source_path(image), vibe, steer, options)
  if run_error then
    return nil, failure_text(work, run_error)
  end
  return read_recipe(work)
end

local function apply_image(image, vibe, steer, look)
  -- Accept, Skip, and a chosen look stay on the open photo. Copy is the
  -- button that makes a duplicate.
  local feature_turn = look == "accept-feature" or look == "skip-feature"
  local edge = tonumber(preference("preview", "integer")) or DEFAULTS.preview
  if edge < 256 then
    edge = 256
  elseif edge > 2048 then
    edge = 2048
  end
  local work = edit_work_dir(image)
  local mkdir_error = mkdir_p(work)
  if mkdir_error then
    return nil, mkdir_error
  end
  local already = (not is_original(image)) and restore_picture_context(image, work)
  if feature_turn then
    local recipe_before = read_recipe(work)
    local waiting_name = type(recipe_before) == "table" and recipe_before.features_next or ""
    if type(waiting_name) ~= "string" or waiting_name == "" then
      close_progress()
      return "Ask for a plan first. Nothing is added until there is a step."
    end
    local flag = look == "accept-feature" and "--commit-step" or "--skip-step"
    show_progress(look == "accept-feature" and "Applying this step…" or "Skipping this step…", 0.7)
    local processor = join_path(script_directory(), "ai_processor.py")
    local command = table.concat({
      shell_quote(preference("python", "string")),
      " ",
      shell_quote(processor),
      dt_version_arg(),
      " ",
      flag,
      " --output-dir ",
      shell_quote(work),
    })
    local output, exit_code = run_captured(command, work)
    if not output or exit_code ~= 0 then
      close_progress()
      return nil, output or "The step could not be updated."
    end
    if look == "accept-feature" then
      local before = request_ui.history_count(image)
      local stepped, step_error = apply_step(image, work)
      if stepped then
        -- Success means a new history row. Check it rather than trust the write.
        local after = request_ui.history_count(image)
        -- darktable saves the sidecar on its own timer (the auto-save
        -- interval), so a count that has not grown yet is not a failure.
        -- Python already fails a step that made no mask or module.
        if before and after and after <= before then
          request_ui.history_note = "If the new step is not in the history panel yet, wait a few seconds."
        else
          request_ui.history_note = nil
        end
      elseif not step_error then
        -- The commit made no mask and no style (no animal outline found, an
        -- empty region). That is a failed step, not a silent success.
        local done = read_recipe(work)
        local why = type(done) == "table" and type(done.feature_note) == "string" and done.feature_note ~= "" and done.feature_note
          or "no mask or module was made for this step"
        step_error = "This step was not added: " .. why .. " The plan did not move on."
      end
      if not stepped then
        local restore = table.concat({
          shell_quote(preference("python", "string")),
          " ",
          shell_quote(processor),
          " --restore-step --output-dir ",
          shell_quote(work),
        })
        run_captured(restore, work)
        close_progress()
        return nil, step_error
      end
    end
    close_progress()
    local recipe = read_recipe(work)
    local body = recipe and plan_status(recipe) or ""
    body = body ~= "" and body or "The plan is updated."
    if look == "accept-feature" and request_ui.history_note then
      body = body .. "\n" .. request_ui.history_note
    end
    return body
  end
  if not feature_turn and already then
    local recipe = read_recipe(work)
    local body = ""
    if recipe then
      body = panel_body(recipe)
    end
    if body == "" then
      body = "The grade already on this photo stays as it is."
    end
    return "This photo already has its look. Type what should change, then press Ask.\n" .. body
  end
  -- Ask already read the picture and wrote the plan. Applying a look
  -- only rebuilds the grade. Another Mistral and Gemini round, and another
  -- preview export, were the long wait after the button.
  local preview = join_path(work, "preview.jpg")
  local reused = file_size(join_path(work, "stage.json")) > 2 and file_size(join_path(work, "proposal.json")) > 2
  local options = processor_options(image, nil)
  options.look = look
  local run_error
  if reused and file_size(preview) > 32 then
    if feature_turn then
      show_progress(look == "accept-feature" and "Applying this step…" or "Skipping this step…", 0.55)
    else
      show_progress("Applying the look…", 0.55)
    end
    options.stage = "finish"
    _, run_error = run_processor(preview, work, image_source_path(image), vibe, steer, options)
  else
    status_label.text = string.format("Exporting %s…", image.filename or "image")
    yield_ui()
    local export_error = export_preview(image, preview, edge)
    if export_error then
      return nil, export_error
    end
    _, run_error = run_staged(preview, work, image_source_path(image), vibe, steer, options)
  end
  if run_error then
    return nil, run_error
  end
  local recipe = read_recipe(work)
  if not recipe then
    return nil, "The masked picture was written, but output.json could not be read."
  end
  local sentence = version_sentence_of(recipe)
  local record = type(recipe.version_record) == "table" and recipe.version_record or nil
  local label = look_title_of(record)
  if label == "" then
    label = sentence
  end
  if feature_turn then
    close_progress()
    return nil, "The step was not applied, so this photo was left as it is."
  end
  do
    show_progress("Applying the look on this photo…", 0.9)
    local applied, apply_error = apply_recipe(image, work)
    if not applied then
      return nil, apply_error
    end
    set_version_name(image, label)
    remember_applied_look(image, image, work, look)
    close_progress()
  end
  if look_name_label ~= nil and label ~= "" and #label < 40 then
    look_name_label.label = "Look: " .. label
  end
  local body = panel_body(recipe)
  if body ~= "" then
    return body
  end
  return sentence ~= "" and sentence or summary(recipe, image)
end

local function remember_vibe()
  -- The vibe box is gone (looks come from darktable presets now).
  if vibe_entry == nil then
    return ""
  end
  local text = tostring(vibe_entry.text or "")
  text = text:gsub("^%s+", ""):gsub("%s+$", "")
  if #text > 400 then
    text = text:sub(1, 400)
  end
  pcall(dt.preferences.write, PREFERENCE, "vibe", "string", text)
  return text
end

local shown_work = nil

scene_text = function()
  local text = ""
  if scene_entry then
    text = tostring(scene_entry.text or "")
  end
  text = text:gsub("^%s+", ""):gsub("%s+$", "")
  if #text > 1200 then
    text = text:sub(1, 1200)
  end
  return text
end

local function steer_text()
  local text = ""
  if steer_entry then
    text = tostring(steer_entry.text or "")
  end
  text = text:gsub("^%s+", ""):gsub("%s+$", "")
  if #text > 400 then
    text = text:sub(1, 400)
  end
  return text
end

local function save_photo_steer(work)
  if type(work) ~= "string" or work == "" then
    return
  end
  write_file(join_path(work, "steer.txt"), steer_text())
end

local function load_photo_steer(work)
  if not steer_entry then
    return
  end
  local text = ""
  if type(work) == "string" and work ~= "" then
    text = read_file(join_path(work, "steer.txt")) or ""
    text = text:gsub("%s+$", "")
  end
  steer_entry.text = text
end

local function remember_steer()
  local text = steer_text()
  save_photo_steer(shown_work)
  pcall(dt.preferences.write, PREFERENCE, "steer", "string", text)
  return text
end

local function vibe_to_send()
  -- No vibe box and no preset dropdown: always faithful (blank).
  if vibe_entry == nil and preset_box == nil then
    return ""
  end
  local typed = remember_vibe()
  local index = 1
  if preset_box and preset_box.selected ~= nil then
    index = tonumber(preset_box.selected) or 1
  end
  if index < 1 then
    index = 1
  end
  local preset_id = preset_ids[index] or ""
  pcall(dt.preferences.write, PREFERENCE, "preset", "string", preset_id)
  if preset_id ~= "" then
    return preset_queries[index] or preset_id:gsub("-", " ")
  end
  return typed
end

local function target_images()
  local current = darkroom_image()
  if current then
    return { current }, false
  end
  if in_darkroom() then
    return {}, false
  end
  local images = selected_images()
  local truncated = false
  if #images > MAX_IMAGES then
    truncated = true
    while #images > MAX_IMAGES do
      table.remove(images)
    end
  end
  return images, truncated
end

local function looks_ready(recipe)
  if type(recipe) ~= "table" or recipe.offer_looks ~= true then
    return false
  end
  if recipe.chat == true or recipe._continued == true then
    return false
  end
  local choices = recipe.choices
  local first = type(choices) == "table" and choices[1] or nil
  return type(first) == "table" and type(first.id) == "string" and first.id ~= ""
end

-- Names from step_history_name. Accept writes these instead of an "AI " prefix.
local editor_history_names = {
  "Recover clipping",
  "Remove haze",
  "Sharpen the subject",
  "Subject lift",
  "Shine the subject",
  "Darken the foreground",
  "Darken the bright gap",
  "Darken the shafts",
  "Warm the highlights",
  "Highlight the color",
  "Highlight the red",
  "Highlight the orange",
  "Highlight the yellow",
  "Highlight the green",
  "Highlight the blue",
  "Recolor",
  "Eyes",
  "Vignette the subject",
  -- Drawn-mask steps (ai_dt/mask_steps.py). Keep in sync with
  -- _ACCEPTED_STEP_NAMES in ai_dt/revert.py.
  "Grade the sky",
  "Grade the buildings",
  "Grade the vegetation",
  "Grade the water",
  "Grade the ground",
  "Lift the people",
  "Quiet the background",
  -- Lesson steps (photoserge examples 4-11) and animal masks.
  "Darken the sky gradient",
  "Darken the bottom",
  "Burn the edges",
  "Glow behind the subject",
  "Lift the animal",
  "Lift the animals",
  "Quiet around the animals",
  "Split tone",
  "Warm-only look",
  "Keep only the red",
  "Keep only the orange",
  "Keep only the yellow",
  "Keep only the green",
  "Keep only the blue",
}

local function sidecar_has_editor_history(body, recipe)
  if body:find('darktable:multi_name="AI ', 1, true) then
    return true
  end
  for _, name in ipairs(editor_history_names) do
    if body:find('darktable:multi_name="' .. name .. '"', 1, true) then
      return true
    end
  end
  local plan = type(recipe) == "table" and recipe.plan or nil
  local log = type(plan) == "table" and plan.step_log or nil
  if type(log) ~= "table" then
    return false
  end
  for _, item in ipairs(log) do
    local name = type(item) == "table" and item.name or nil
    if type(name) == "string" and name ~= "" and body:find('darktable:multi_name="' .. name .. '"', 1, true) then
      return true
    end
  end
  return false
end

local function history_flags(image, recipe)
  history_back = false
  history_next = false
  can_revert = false
  can_remove = false
  has_photo = image ~= nil
  if image == nil then
    return
  end
  can_remove = not is_original(image)
  local redo = read_file(join_path(edit_work_dir(image), "history-redo.json"))
  history_next = type(redo) == "string" and redo:find('"history"', 1, true) ~= nil
  if type(sidecar_file) ~= "function" then
    return
  end
  local xmp = sidecar_file(image)
  if not xmp then
    return
  end
  local body = read_file(xmp)
  if not body or body == "" then
    return
  end
  local ending = tonumber(body:match('darktable:history_end="(%d+)"')) or 0
  history_back = ending > 0
  can_revert = sidecar_has_editor_history(body, recipe)
end

local function refresh_controls(recipe)
  if run_button then
    run_button.sensitive = (not panel_is_busy) and has_photo
  end
  if looks_button then
    looks_button.sensitive = (not panel_is_busy) and looks_ready(recipe)
  end
  if cancel_button then
    cancel_button.sensitive = (not panel_is_busy) and looks_pending
  end
  if copy_button then
    copy_button.sensitive = (not panel_is_busy) and has_photo
  end
  if remove_button then
    remove_button.sensitive = (not panel_is_busy) and can_remove
  end
  if back_button then
    back_button.sensitive = (not panel_is_busy) and history_back
  end
  if next_button then
    next_button.sensitive = (not panel_is_busy) and history_next
  end
  if revert_button then
    revert_button.sensitive = (not panel_is_busy) and can_revert
  end
  if accept_button then
    accept_button.sensitive = (not panel_is_busy) and feature_waiting
  end
  if skip_button then
    skip_button.sensitive = (not panel_is_busy) and feature_waiting
  end
end

-- Ask darktable to write this photo's sidecar from its library now. A photo
-- just opened in the darkroom has only its automatic rows in memory and a
-- sidecar darktable has not written yet (change_timestamp -1); splicing that
-- file and reading the count from it went wrong without a word.
-- The 5.4 and 5.6 Lua APIs have no "write sidecar" call. Writing any image
-- field goes through dt_image_cache_write_release(…, DT_IMAGE_CACHE_SAFE)
-- (src/lua/image.c), which writes the XMP from the library, history included.
-- Setting the rating to itself is that write with no visible change.
function request_ui.sync_sidecar(image)
  pcall(function()
    local rating = image.rating
    image.rating = rating
  end)
end

-- darktable:history_end of the photo's sidecar after a sync, or nil.
function request_ui.history_count(image)
  request_ui.sync_sidecar(image)
  local path = sidecar_file(image)
  local body = path and read_file(path) or nil
  if type(body) ~= "string" then
    return nil
  end
  return tonumber(body:match('darktable:history_end="(%d+)"'))
end

function request_ui.short(title)
  local text = tostring(title or ""):gsub("%s+", " ")
  if #text > 28 then
    text = text:sub(1, 27) .. "…"
  end
  return text
end

-- Accept and Skip always say which step they act on. With no step waiting
-- they are off and say so.
local function show_feature_pair(title)
  feature_waiting = type(title) == "string" and title ~= ""
  local accept_tip = "No step is waiting, so Accept is off. Type a change and press Ask."
  local skip_tip = "No step is waiting, so Skip is off. Type a change and press Ask."
  local accept_label = "Accept"
  local skip_label = "Skip"
  if feature_waiting then
    accept_tip = "Accept applies to: " .. title .. ". Paint this onto the edit."
    skip_tip = "Skip applies to: " .. title .. ". Leave it off and show the one after it."
    accept_label = "Accept: " .. request_ui.short(title)
    skip_label = "Skip: " .. request_ui.short(title)
  end
  if accept_button then
    pcall(function()
      accept_button.label = accept_label
      accept_button.tooltip = accept_tip
    end)
  end
  if skip_button then
    pcall(function()
      skip_button.label = skip_label
      skip_button.tooltip = skip_tip
    end)
  end
end

local function show_choices(recipe)
  shown_recipe = type(recipe) == "table" and recipe or nil
  local follow_up = type(recipe) == "table" and recipe._continued == true
  local chatting = type(recipe) == "table" and (recipe.chat == true or follow_up)
  local offer = type(recipe) == "table" and recipe.offer_looks == true and not chatting
  local choices = offer and recipe.choices or nil
  local slots = {}
  local next_id = type(recipe) == "table" and recipe.features_next or ""
  local next_title = type(recipe) == "table" and recipe.features_next_title or next_id
  if run_button then
    run_button.label = "Ask"
  end
  if steer_entry then
    pcall(function()
      steer_entry.placeholder = "what should change on this photo?"
    end)
  end
  if type(choices) == "table" then
    for _, item in ipairs(choices) do
      if #slots >= 6 then
        break
      end
      table.insert(slots, item)
    end
  end
  local waiting = ""
  if type(next_id) == "string" and next_id ~= "" then
    waiting = type(next_title) == "string" and next_title ~= "" and next_title or next_id
  end
  show_feature_pair(waiting)
  for index = 1, 6 do
    local button = choice_buttons[index]
    local item = slots[index]
    if button and type(item) == "table" and type(item.id) == "string" and item.id ~= "" then
      button.label = tostring(item.title or item.id)
      button.sensitive = true
      pcall(function()
        button.tooltip = tostring(item.why or "")
      end)
      choice_ids[index] = item.id
    elseif button then
      button.label = " "
      button.sensitive = false
      pcall(function()
        button.tooltip = ""
      end)
      choice_ids[index] = nil
    end
  end
  request_ui.was_changed = request_ui.changed(shown_recipe) or request_ui.described(shown_recipe)
  refresh_controls(shown_recipe)
end

local preview_image = nil
local preview_applying = false
local preview_watch = 0
local pick_serial = 0
local open_options_window

local function url_encode(value)
  return (tostring(value):gsub("([^%w%-%.%_%~])", function(char)
    return string.format("%%%02X", string.byte(char))
  end))
end

local function remember_pick(work, look, source, applied)
  if type(work) ~= "string" or work == "" then
    return
  end
  local safe = tostring(look or ""):gsub("[^%w%-]", "")
  if safe == "" then
    return
  end
  local origin = source == "editor" and "editor" or "preview"
  local flag = applied and "true" or "false"
  pick_serial = pick_serial + 1
  local payload = string.format(
    '{"id":"%s","title":"%s","source":"%s","applied":%s,"at":%d}\n',
    safe,
    safe,
    origin,
    flag,
    os.time() * 1000 + pick_serial
  )
  write_file(join_path(work, "picked.json"), payload)
end

local function run_analysis()
  preview_image = nil
  local vibe = vibe_to_send()
  local steer = remember_steer()
  local images, truncated = target_images()
  if #images == 0 then
    if in_darkroom() then
      status_label.text = "Open a photo in the darkroom, then press Ask again."
    else
      status_label.text = "Select an image on the lighttable, then press Ask again."
    end
    dt.print("Select an image first")
    return
  end
  local last = ""
  local shown = nil
  for index, image in ipairs(images) do
    local recipe, failure, continued = propose_image(image, index, #images, vibe, steer)
    if not recipe then
      status_label.text = string.format("%s failed.\n%s", image.filename or "Image", failure or "unknown error")
      dt.print("AI analysis failed")
      return
    end
    if shown == nil then
      shown = recipe
      shown._continued = continued and true or false
    end
    show_scene(recipe)
    last = plan_status(recipe)
    if last == "" then
      last = summary(recipe, image)
    end
  end
  show_choices(shown)
  local continued = type(shown) == "table" and shown._continued == true
  local where = "Type what should change, then press Ask. Accept and Skip stay on this photo."
  if continued then
    where = "Type what should change, then press Ask. Accept and Skip stay on this photo."
  end
  if truncated then
    last = last .. string.format("\nOnly the first %d selected images were proposed.", MAX_IMAGES)
  end
  if continued then
    last = last .. "\n" .. where
  elseif type(shown) == "table" and type(shown.features_next) == "string" and shown.features_next ~= "" then
    last = last .. "\nAccept or skip the next step. " .. where
  elseif type(shown) ~= "table" or shown.offer_looks ~= false then
    last = last .. "\n" .. where
  end
  status_label.text = last
  dt.print(continued and "Continuing this photo" or "Plan is ready")
end

local function apply_choice(index, forced_look)
  local look = forced_look or choice_ids[index]
  if type(look) ~= "string" or look == "" then
    status_label.text = "Press Ask first."
    return
  end
  local vibe = vibe_to_send()
  local steer = remember_steer()
  local images, truncated = target_images()
  if #images == 0 then
    status_label.text = "Select an image, then pick the look again."
    return
  end
  local last = ""
  local feature_turn = look == "accept-feature" or look == "skip-feature"
  for _, image in ipairs(images) do
    remember_pick(edit_work_dir(image), look, "editor", true)
    local text, failure = apply_image(image, vibe, steer, look)
    if not text then
      local shown = string.format("%s failed.\n%s", image.filename or "Image", failure or "unknown error")
      -- Show the saved plan as it is now. A failed Accept put its step back,
      -- so the buttons must not still name the next one.
      local current = read_recipe(edit_work_dir(image))
      if current then
        pcall(show_choices, current)
      end
      status_label.text = shown
      dt.print("AI edit failed")
      return false, shown
    end
    last = text
    local recipe = read_recipe(edit_work_dir(image))
    if recipe and not feature_turn then
      recipe.chat = true
      recipe.choices = {}
    end
    if recipe then
      show_choices(recipe)
    end
  end
  if truncated then
    last = last .. string.format("\nOnly the first %d selected images were duplicated.", MAX_IMAGES)
  end
  status_label.text = last
  if look == "accept-feature" then
    dt.print("Applied this step")
  elseif look == "skip-feature" then
    dt.print("Skipped this step")
  elseif type(last) == "string" and last:find("already has its look", 1, true) then
    dt.print("Already on this photo")
  elseif type(last) == "string" and last:find("added as one step", 1, true) then
    dt.print("Applied this step")
  elseif type(last) == "string" and last:find("did not become a step", 1, true) then
    dt.print("No step for this sentence")
  elseif type(last) == "string" and last:find("added on this photo", 1, true) then
    dt.print("Look added on this photo")
  else
    dt.print("Look applied on a new duplicate")
  end
end

local function set_panel_busy(busy)
  panel_is_busy = busy and true or false
  refresh_controls(shown_recipe)
end

local function run_popen(command)
  local pipe, popen_error = io.popen(command, "r")
  if not pipe then
    return nil, popen_error or "could not start the command"
  end
  local output = pipe:read("*a") or ""
  local closed, _, code = pipe:close()
  local exit_code = 0
  if closed ~= true then
    exit_code = type(code) == "number" and code or 1
  end
  return output, exit_code
end

-- One wait for a hidden command. dt.control.execute releases the Lua lock
-- for the whole run, so the window keeps handling input. A sleep loop on
-- the callback does not: it holds this call until the exit file appears,
-- and a missing exit file used to sit there for about fifteen minutes.
local function run_blocking(command)
  if dt.control and type(dt.control.execute) == "function" then
    local ok, status = pcall(dt.control.execute, command)
    if ok then
      return status
    end
  end
  return os.execute(command)
end

-- The exit file appears when wscript finishes. Sleep releases the Lua
-- lock, so the panel can paint. A missing file stops after ten minutes
-- instead of waiting forever.
local function wait_for_exit(exit_path, folder)
  local seen = ""
  local tries = 0
  while file_size(exit_path) < 1 do
    if not (dt.control and type(dt.control.sleep) == "function") then
      return false
    end
    dt.control.sleep(200)
    tries = tries + 1
    local status = read_file(join_path(folder, "status.txt")) or ""
    status = status:gsub("%s+$", "")
    if status ~= "" and status ~= seen then
      seen = status
      show_progress(status, 0.45)
    end
    if tries >= 3000 then
      return false
    end
  end
  return true
end

-- One hidden worker, started with Darktable, runs every queued command.
-- dt.control.execute and io.popen each open a visible cmd window, so a click
-- does not start one of those.
local function queue_directory()
  return join_path(join_path(script_directory(), "ai-editor-work"), "queue")
end

local function worker_is_ready()
  return file_size(join_path(queue_directory(), "ready.txt")) > 0
end

local function start_hidden_commands(commands)
  if not on_windows() or #commands == 0 then
    return
  end
  local folder = join_path(script_directory(), "ai-editor-work")
  if not os.rename(folder, folder) then
    os.execute('mkdir "' .. folder:gsub("/", "\\") .. '"')
  end
  local lines = {}
  for _, command in ipairs(commands) do
    lines[#lines + 1] = 'CreateObject("Wscript.Shell").Run "' .. command:gsub('"', '""') .. '", 0, False'
  end
  local vbs_path = join_path(folder, "launch-once.vbs")
  write_file(vbs_path, table.concat(lines, "\r\n") .. "\r\n")
  os.execute("wscript //nologo //b " .. shell_quote(vbs_path))
end

local worker_ready = false

local function worker_is_alive()
  local beat = join_path(queue_directory(), "heartbeat.txt")
  os.remove(beat)
  if not (dt.control and type(dt.control.sleep) == "function") then
    return false
  end
  dt.control.sleep(150)
  return file_size(beat) > 0
end

local function ensure_worker()
  if worker_ready then
    return true
  end
  -- The worker is a Windows script host; elsewhere there is none to wait
  -- for (this used to sleep about 2.5 s on Linux before giving up).
  if not on_windows() then
    return false
  end
  if worker_is_ready() and worker_is_alive() then
    worker_ready = true
    return true
  end
  local folder = queue_directory()
  if not os.rename(folder, folder) then
    os.execute('mkdir "' .. folder:gsub("/", "\\") .. '"')
  end
  write_file(join_path(folder, "owner.txt"), "restart\n")
  if dt.control and type(dt.control.sleep) == "function" then
    dt.control.sleep(400)
  end
  local vbs = join_path(script_directory(), "windows-hidden-worker.vbs")
  start_hidden_commands({
    "wscript //nologo //b " .. shell_quote(vbs) .. " " .. shell_quote(folder),
  })
  local tries = 0
  while file_size(join_path(folder, "ready.txt")) < 1 and tries < 20 do
    if not (dt.control and type(dt.control.sleep) == "function") then
      break
    end
    dt.control.sleep(100)
    tries = tries + 1
  end
  worker_ready = worker_is_ready()
  return worker_ready
end

local function submit_hidden_job(lines, exit_path, status_folder)
  local folder = queue_directory()
  local request = join_path(folder, "request.txt")
  local temp = join_path(folder, "request.tmp")
  local write_error = write_file(temp, table.concat(lines, "\n") .. "\n")
  if write_error then
    return nil, write_error
  end
  os.remove(request)
  if not os.rename(temp, request) then
    return nil, "could not queue the command"
  end
  if not wait_for_exit(exit_path, status_folder) and file_size(exit_path) < 1 then
    return nil, "The edit took too long and was stopped."
  end
  return true
end

hidden_mkdir = function(path)
  if not ensure_worker() then
    return false
  end
  local exit_path = join_path(queue_directory(), "mkdir.exit")
  os.remove(exit_path)
  local queued = submit_hidden_job({ "mkdir", path, exit_path }, exit_path, queue_directory())
  return queued and os.rename(path, path) and true or false
end

-- Windows io.popen keeps a cmd window on screen for the whole run.
-- The worker is already running, so this only writes the command file.
local function run_hidden(command, work_dir)
  -- Only Windows opens a console window for io.popen. Elsewhere a plain
  -- popen is the hidden run; stdin is /dev/null (NUL is a file name there).
  if not on_windows() then
    return run_popen(command .. " < /dev/null 2>&1")
  end
  local folder = work_dir
  if type(folder) ~= "string" or folder == "" then
    folder = join_path(script_directory(), "ai-editor-work")
  end
  mkdir_p(folder)
  local cmd_path = join_path(folder, "run.cmd")
  local log_path = join_path(folder, "run.log")
  local exit_path = join_path(folder, "run.exit")
  os.remove(exit_path)
  -- A failed launch must not report the previous command's log.
  write_file(log_path, "")
  local write_error = write_file(cmd_path, "@echo off\r\n" .. command .. " < NUL\r\n")
  if write_error then
    return nil, write_error
  end
  os.remove(join_path(folder, "status.txt"))
  if not ensure_worker() then
    return run_popen('"' .. command .. '" < NUL 2>&1')
  end
  local queued, queue_error = submit_hidden_job(
    { "run", cmd_path, log_path, exit_path },
    exit_path,
    folder
  )
  if not queued then
    return nil, queue_error
  end
  if file_size(exit_path) < 1 then
    return nil, "The edit did not finish. No exit code was written."
  end
  local code_text = read_file(exit_path) or ""
  local exit_code = tonumber(code_text:match("%d+")) or 1
  return read_file(log_path) or "", exit_code
end

function run_captured(command, work_dir)
  if on_windows() then
    return run_hidden(command, work_dir)
  end
  return run_popen(command .. " < /dev/null 2>&1")
end

local function browser_app()
  local roots = {
    os.getenv("PROGRAMFILES"),
    os.getenv("PROGRAMFILES(X86)"),
    os.getenv("LOCALAPPDATA"),
  }
  local names = {
    "Microsoft\\Edge\\Application\\msedge.exe",
    "Google\\Chrome\\Application\\chrome.exe",
  }
  for _, root in ipairs(roots) do
    if type(root) == "string" and root ~= "" then
      for _, name in ipairs(names) do
        local path = join_path(root, name)
        if file_size(path) > 10 then
          return path
        end
      end
    end
  end
  return nil
end

local function start_detached(command)
  local folder = join_path(script_directory(), "ai-editor-work")
  mkdir_p(folder)
  local vbs_path = join_path(folder, "launch-once.vbs")
  local body = 'CreateObject("Wscript.Shell").Run "' .. command:gsub('"', '""') .. '", 0, False\r\n'
  if not on_windows() then
    return os.execute(command .. " >/dev/null 2>&1 &")
  end
  write_file(vbs_path, body)
  return os.execute("wscript //nologo //b " .. shell_quote(vbs_path))
end

local function preview_server_command()
  local server = join_path(script_directory(), "preview_server.py")
  if file_size(server) < 10 then
    return nil
  end
  return table.concat({
    shell_quote(preference("python", "string")),
    " ",
    shell_quote(server),
    " --ensure",
  })
end

local function preview_responds()
  local probe = table.concat({
    shell_quote(preference("python", "string")),
    " -c ",
    shell_quote("import urllib.request; urllib.request.urlopen('http://127.0.0.1:8741/api/health', timeout=0.5)"),
  })
  local _, code = run_hidden(probe, join_path(script_directory(), "ai-editor-work"))
  return code == 0
end

local function ensure_preview_server()
  local command = preview_server_command()
  if not command then
    return false
  end
  if preview_responds() then
    return true
  end
  -- The page is opened only after this returns. A start that has not
  -- answered yet used to open a browser onto a dead port.
  show_progress("Starting the preview…", 0.85)
  local _, code = run_hidden(command, join_path(script_directory(), "ai-editor-work"))
  close_progress()
  return code == 0
end

function open_options_window(work)
  if type(work) ~= "string" or work == "" then
    return false
  end
  if not ensure_preview_server() then
    return false
  end
  local url = "http://127.0.0.1:8741/options?work=" .. url_encode(work)
  if on_windows() then
    -- An app window can close itself after a choice. A normal browser tab cannot.
    local browser = browser_app()
    if browser then
      start_detached(shell_quote(browser) .. " --app=" .. shell_quote(url))
    else
      start_detached('cmd /c start "" "' .. url .. '"')
    end
  else
    start_detached("xdg-open " .. shell_quote(url))
  end
  return true
end

local function maybe_apply_preview_pick()
  if panel_is_busy or preview_applying or preview_image == nil then
    return
  end
  local work = edit_work_dir(preview_image)
  local pick = read_json_file(join_path(work, "picked.json"))
  if type(pick) ~= "table" or pick.source ~= "preview" or pick.applied == true then
    return
  end
  local look = pick.id
  if type(look) ~= "string" or look == "" then
    return
  end
  if look == "cancel" or pick.cancelled == true then
    os.remove(join_path(work, "picked.json"))
    looks_pending = false
    refresh_controls(shown_recipe)
    status_label.text = "Look cancelled. This photo stays on its current plan."
    dt.print("Look cancelled")
    return
  end
  preview_applying = true
  set_panel_busy(true)
  remember_pick(work, look, "preview", true)
  local ok_apply, text, failure = pcall(apply_image, preview_image, vibe_to_send(), remember_steer(), look)
  close_progress()
  if not ok_apply then
    failure = text
    text = nil
  end
  if text then
    local recipe = read_recipe(work)
    if recipe then
      show_choices(recipe)
    end
    status_label.text = text
    if look == "accept-feature" then
      dt.print("Applied this step")
    elseif look == "skip-feature" then
      dt.print("Skipped this step")
    elseif type(text) == "string" and text:find("already has its look", 1, true) then
      dt.print("Already on this photo")
    elseif type(text) == "string" and text:find("added as one step", 1, true) then
      dt.print("Applied this step")
    elseif type(text) == "string" and text:find("did not become a step", 1, true) then
      dt.print("No step for this sentence")
    elseif type(text) == "string" and text:find("added on this photo", 1, true) then
      dt.print("Look added on this photo")
    else
      dt.print("Look applied on a new duplicate")
    end
  else
    -- Leave the pick applied. Clearing it made the next mouse move
    -- duplicate the photo again.
    -- A failed Accept put its step back: show the saved plan, not the next step.
    local current = read_recipe(work)
    if current then
      pcall(show_choices, current)
    end
    status_label.text = string.format(
      "%s failed.\n%s",
      preview_image.filename or "Image",
      failure or "unknown error"
    )
    dt.print("AI edit failed")
  end
  looks_pending = false
  set_panel_busy(false)
  preview_applying = false
  if sync_waiting_step then
    sync_waiting_step(darkroom_image())
  end
end

-- The looks page writes the choice and closes. Nothing in Darktable
-- moves until the filmstrip is hovered, so the click used to sit there.
-- This wait runs after Ask and starts the edit when the file appears.
local function watch_preview_pick()
  if preview_image == nil then
    return
  end
  if not (dt.control and type(dt.control.sleep) == "function") then
    return
  end
  preview_watch = preview_watch + 1
  local token = preview_watch
  for _ = 1, 450 do
    if token ~= preview_watch or preview_image == nil then
      return
    end
    local work = edit_work_dir(preview_image)
    local pick = read_json_file(join_path(work, "picked.json"))
    if type(pick) == "table" and pick.source == "preview" and pick.applied ~= true
      and not panel_is_busy and not preview_applying then
      maybe_apply_preview_pick()
      return
    end
    dt.control.sleep(200)
  end
end

local function history_steps(count)
  if count == 1 then
    return "1 history step"
  end
  return string.format("%d history steps", count)
end

-- The lighttable button deleted AI rows in library.db and called
-- image:drop_cache(). That refreshed the thumbnail, which is drawn from
-- the database, and develop was not holding the stack. In the darkroom
-- the same delete leaves the history panel alone: the panel shows
-- develop's copy, drop_cache() only drops mipmaps, and develop writes
-- that copy back over the database. There is no Lua call that deletes
-- one history row. image:apply_sidecar() replaces the stack. On the
-- image open in the darkroom it also reloads develop while the pixelpipe
-- is running, and that reload deadlocks the window. Leave first so the
-- develop image is cleared, apply the sidecar on the lighttable path,
-- then open the photo again. The darkroom loads the replaced stack
-- through the normal enter path.
local function image_label(image)
  local name = image.filename or "image"
  return string.format("%s (version %d)", name, duplicate_index_of(image))
end

-- try_enter() opens whatever dt_act_on_get_main_image() is, which is the
-- selection when nothing is under the pointer. Select this photo first so
-- coming back loads the stack we just replaced.
local function select_image(image)
  if type(dt.gui.selection) ~= "function" then
    return
  end
  pcall(dt.gui.selection, { image })
end

-- dt.gui.current_view(view) runs dt_ctl_switch_mode_to_by_view on the GUI
-- thread, then returns whichever view the getter can see. The setter can
-- raise after the window has already moved, and the getter can still say
-- "darkroom" until the GUI thread finishes that switch. execute() drops the
-- Lua lock for one short process so the GUI can finish. This is not a poll.
local function yield_view()
  -- Sleep releases the Lua lock so the view can finish switching.
  -- A process here would open a command window on every wait.
  if dt.control and type(dt.control.sleep) == "function" then
    pcall(dt.control.sleep, 50)
    return
  end
  if dt.control and type(dt.control.execute) == "function" then
    pcall(dt.control.execute, "sleep 0.05")
  end
end

local function leave_darkroom()
  if type(dt.gui.current_view) ~= "function" or not dt.gui.views or not dt.gui.views.lighttable then
    return false
  end
  pcall(dt.gui.current_view, dt.gui.views.lighttable)
  -- The window moves before current_view() says lighttable. A short wait
  -- used to return while the getter still said darkroom, and Back then
  -- stopped on the lighttable without touching the history.
  for _ = 1, 40 do
    local name = current_view_name()
    if name == "lighttable" or (name ~= nil and name ~= "darkroom") then
      return true
    end
    yield_view()
  end
  return current_view_name() == "lighttable"
end

local function return_to_darkroom(image)
  select_image(image)
  if type(dt.gui.current_view) ~= "function" or not dt.gui.views or not dt.gui.views.darkroom then
    return
  end
  for _ = 1, 40 do
    if current_view_name() ~= "darkroom" then
      pcall(dt.gui.current_view, dt.gui.views.darkroom)
    end
    yield_view()
    if in_darkroom() then
      break
    end
  end
  pcall(function()
    dt.gui.views.darkroom.display_image(image)
  end)
end

-- Develop keeps the open photo after the darkroom closes and writes that
-- old stack back over the library. Parking on another photo makes the next
-- open load the sidecar that was just applied.
local function another_image(image)
  local original = original_of(image)
  if original and tonumber(original.id) ~= tonumber(image.id) then
    return original
  end
  if type(dt.database) == "table" then
    for _, candidate in ipairs(dt.database) do
      if candidate and tonumber(candidate.id) ~= tonumber(image.id) then
        return candidate
      end
    end
  end
  return nil
end

local function park_develop(image)
  local other = another_image(image)
  if other == nil then
    return false
  end
  show_in_darkroom(other)
  if in_darkroom() then
    leave_darkroom()
  end
  for _ = 1, 4 do
    yield_view()
  end
  return current_view_name() ~= "darkroom"
end

sidecar_file = function(image)
  local ok, path = pcall(function()
    return image.sidecar
  end)
  if ok and type(path) == "string" and file_size(path) > 32 then
    return path
  end
  local source = image_source_path(image)
  if source ~= "" and file_size(source .. ".xmp") > 32 then
    return source .. ".xmp"
  end
  return nil
end

-- A new history row replaces the step Back removed, so Next must not splice it back.
discard_history_redo = function(image)
  if image == nil then
    return
  end
  local folder = edit_work_dir(image)
  mkdir_p(folder)
  write_file(join_path(folder, "history-redo.json"), "{\"items\":[]}\n")
end

-- revert.py --collapse-xmp is a no-op now. A drawn-mask step discards rows
-- past history_end itself (drawn_mask._prepare_insert, as darktable does on
-- a new edit) and becomes the active top; dt.styles.apply does the same.
function continue_from_viewed_history(image)
  local xmp = sidecar_file(image)
  if not xmp then
    return
  end
  local script = join_path(script_directory(), join_path("ai_dt", "revert.py"))
  if file_size(script) < 10 or type(image.apply_sidecar) ~= "function" then
    return
  end
  local work = work_dir_for(image)
  mkdir_p(work)
  local cleaned = join_path(work, "history.xmp")
  local command = table.concat({
    shell_quote(preference("python", "string")),
    " ",
    shell_quote(script),
    " --collapse-xmp --xmp ",
    shell_quote(xmp),
    " --xmp-out ",
    shell_quote(cleaned),
  })
  local output, exit_code = run_captured(command, work)
  if not output or exit_code ~= 0 then
    return
  end
  local removed = tonumber(output:match("removed%s+(%d+)"))
  if not removed or removed == 0 then
    return
  end
  local was_darkroom = in_darkroom()
  if was_darkroom then
    leave_darkroom()
    if current_view_name() == "darkroom" then
      return
    end
  end
  pcall(function()
    image:apply_sidecar(cleaned)
  end)
  if was_darkroom then
    return_to_darkroom(image)
  end
end

-- Recount after a view switch. Leaving the darkroom writes the sidecar from
-- the library, so the AI steps can already be gone before apply_sidecar runs.
local function sidecar_still_has_ai(command, work)
  local output, exit_code = run_captured(command, work)
  if not output or exit_code ~= 0 then
    return nil
  end
  return tonumber(output:match("removed%s+(%d+)"))
end

local function revert_via_sidecar(image)
  local xmp = sidecar_file(image)
  if not xmp then
    return nil, "Darktable has the library open, and this photo has no XMP sidecar to rewrite."
  end
  local script = join_path(script_directory(), join_path("ai_dt", "revert.py"))
  if file_size(script) < 10 then
    return nil, "ai_dt/revert.py is not next to ai_editor.lua"
  end
  local work = work_dir_for(image)
  local cleaned = join_path(work, "revert.xmp")
  local command = table.concat({
    shell_quote(preference("python", "string")),
    " ",
    shell_quote(script),
    " --xmp ",
    shell_quote(xmp),
    " --xmp-out ",
    shell_quote(cleaned),
  })
  local recipe = read_recipe(work)
  local plan = type(recipe) == "table" and recipe.plan or nil
  local log = type(plan) == "table" and plan.step_log or nil
  if type(log) == "table" then
    for _, item in ipairs(log) do
      local name = type(item) == "table" and item.name or nil
      if type(name) == "string" and name ~= "" then
        command = command .. " --also " .. shell_quote(name)
      end
    end
  end
  status_label.text = string.format("Rewriting the sidecar for %s…", image_label(image))
  yield_ui()
  local output, exit_code = run_captured(command, work)
  if not output then
    return nil, exit_code
  end
  if exit_code ~= 0 then
    return nil, output
  end
  local removed = tonumber(output:match("removed%s+(%d+)"))
  if removed == nil then
    return nil, output
  end
  -- The strip count is not a result. An unchanged sidecar still contains
  -- the AI steps, so a second press must see zero before anything is applied.
  if removed == 0 then
    return 0
  end
  if type(image.apply_sidecar) ~= "function" then
    return nil, "This Darktable build cannot apply a sidecar."
  end
  -- apply_sidecar on the open darkroom image reloads the pipe and does
  -- not return. Leave first. The setter can error after the window has
  -- already moved, so a failed call is not proof that we are still there.
  local stepped_aside = false
  if in_darkroom() then
    status_label.text = string.format("Closing the darkroom to revert %s…", image_label(image))
    yield_ui()
    local left = leave_darkroom()
    local where = current_view_name()
    if where == "darkroom" then
      return nil, "Could not close the darkroom, so the AI edit was not applied."
    end
    -- Lighttable, or the getter could not say. The switch was requested
    -- and the window may already be on the lighttable, so come back.
    stepped_aside = true
    if not left then
      return_to_darkroom(image)
      if sidecar_still_has_ai(command, work) == 0 then
        return removed
      end
      return nil, "The view changed before the sidecar could be applied."
    end
  end
  local failure = nil
  if current_view_name() == "darkroom" then
    failure = "Could not close the darkroom, so the AI edit was not applied."
  else
    local applied, result = pcall(function()
      return image:apply_sidecar(cleaned)
    end)
    if not applied or result ~= true then
      failure = "Darktable did not replace the history stack.\n" .. tostring(result)
    else
      local body, read_error = read_file(cleaned)
      if not body or body == "" then
        failure = "Darktable replaced the history, but the rewritten sidecar could not be saved.\n"
          .. tostring(read_error)
      else
        local write_error = write_file(xmp, body)
        if write_error then
          failure = "Darktable replaced the history, but the rewritten sidecar could not be saved.\n"
            .. tostring(write_error)
        end
      end
    end
  end
  if stepped_aside then
    return_to_darkroom(image)
  end
  if failure then
    if sidecar_still_has_ai(command, work) == 0 then
      return removed
    end
    return nil, failure
  end
  return removed
end

apply_masked_step = function(image, work_dir, mask)
  -- The shapes live in the sidecar, not in a style. Read the history
  -- after the darkroom has written it, append one module, then load
  -- that file back. A failed splice does not call apply_sidecar.
  local regions = type(mask) == "table" and (mask.kind == "regions" or mask.kind == "path" or mask.kind == "shapes")
  local what = "eye mask"
  if type(mask) == "table" and mask.kind == "shapes" then
    what = (type(mask.what) == "string" and mask.what ~= "") and mask.what or "drawn mask"
  elseif type(mask) == "table" and mask.kind == "path" then
    what = "object mask"
  elseif type(mask) == "table" and mask.kind == "regions" then
    what = "color mask"
  end
  local xmp = sidecar_file(image)
  if not xmp then
    return nil, "This photo has no XMP sidecar, so the " .. what .. " was not written. The darkroom history was left as it is."
  end
  local script = join_path(script_directory(), join_path("ai_dt", "drawn_mask.py"))
  if file_size(script) < 10 then
    return nil, "ai_dt/drawn_mask.py is not next to ai_editor.lua. The darkroom history was left as it is."
  end
  if type(image.apply_sidecar) ~= "function" then
    return nil, "This Darktable build cannot apply a sidecar. The darkroom history was left as it is."
  end
  local cx, cy, rx, ry, ev
  if regions then
    if type(mask.file) ~= "string" or file_size(mask.file) < 32 then
      return nil, "The " .. what .. " is incomplete, so the history was left as it is."
    end
  else
    cx = tonumber(mask.cx)
    cy = tonumber(mask.cy)
    rx = tonumber(mask.rx)
    ry = tonumber(mask.ry)
    ev = tonumber(mask.exposure_ev)
    if not cx or not cy or not rx or not ry or not ev then
      return nil, "The eye ellipse is incomplete, so the history was left as it is."
    end
  end
  local stepped_aside = false
  if in_darkroom() then
    status_label.text = string.format("Closing the darkroom to add the %s on %s…", what, image_label(image))
    yield_ui()
    local left = leave_darkroom()
    if current_view_name() == "darkroom" then
      return nil, "Could not close the darkroom, so the " .. what .. " was not written."
    end
    stepped_aside = true
    if not left then
      return_to_darkroom(image)
      return nil, "The view changed before the " .. what .. " could be written. The history was left as it is."
    end
    -- The darkroom writes its open history just after the view changes.
    -- Wait for that write, or it replaces the stack this function applies.
    for _ = 1, 6 do
      yield_view()
    end
  end
  request_ui.sync_sidecar(image)
  mkdir_p(work_dir)
  local cleaned = join_path(work_dir, "masked.xmp")
  local backup = join_path(work_dir, "history-before-mask.xmp")
  local command
  if regions then
    command = table.concat({
      shell_quote(preference("python", "string")),
      " ",
      shell_quote(script),
      " --xmp ",
      shell_quote(xmp),
      " --xmp-out ",
      shell_quote(cleaned),
      " --mask-json ",
      shell_quote(mask.file),
    })
  else
    command = table.concat({
    shell_quote(preference("python", "string")),
    " ",
    shell_quote(script),
    " --xmp ",
    shell_quote(xmp),
    " --xmp-out ",
    shell_quote(cleaned),
    " --cx ",
    tostring(cx),
    " --cy ",
    tostring(cy),
    " --rx ",
    tostring(rx),
    " --ry ",
    tostring(ry),
    " --ev ",
    tostring(ev),
    " --name ",
    shell_quote((type(mask.name) == "string" and mask.name ~= "") and mask.name or "AI eyes"),
  })
  end
  local output, exit_code = run_captured(command, work_dir)
  if not output or exit_code ~= 0 or file_size(cleaned) < 32 then
    if stepped_aside then
      return_to_darkroom(image)
    end
    return nil, "The " .. what .. " was not written, so the darkroom history was left as it is.\n" .. tail(tostring(output or exit_code), 400)
  end
  local source_body = read_file(xmp)
  if source_body and source_body ~= "" then
    write_file(backup, source_body)
  end
  local rows_before = tonumber(tostring(source_body or ""):match('darktable:history_end="(%d+)"'))
  park_develop(image)
  local failure = nil
  if current_view_name() == "darkroom" then
    failure = "Could not close the darkroom, so the " .. what .. " was not written."
  else
    local applied, result = pcall(function()
      return image:apply_sidecar(cleaned)
    end)
    if not applied or result ~= true then
      failure = "Darktable did not replace the history stack.\n" .. tostring(result)
    else
      local body, read_error = read_file(cleaned)
      if not body or body == "" then
        failure = "Darktable replaced the history, but the rewritten sidecar could not be saved.\n" .. tostring(read_error)
      else
        local write_error = write_file(xmp, body)
        if write_error then
          failure = "Darktable replaced the history, but the rewritten sidecar could not be saved.\n" .. tostring(write_error)
        else
          -- Apply once more after the late darkroom write has had its turn.
          for _ = 1, 4 do
            yield_view()
          end
          local kept, kept_result = pcall(function()
            return image:apply_sidecar(xmp)
          end)
          if not kept or kept_result ~= true then
            failure = "Darktable did not keep the history stack.\n" .. tostring(kept_result)
          end
        end
      end
    end
  end
  if not failure and rows_before then
    local rows_after = request_ui.history_count(image)
    if rows_after and rows_after <= rows_before then
      failure = string.format(
        "The %s was written, but darktable still shows %d history item(s), so nothing was added.",
        what,
        rows_after
      )
    end
  end
  if stepped_aside then
    return_to_darkroom(image)
  end
  if failure then
    return nil, failure
  end
  discard_history_redo(image)
  return true
end

local function revert_one(image)
  status_label.text = string.format("Reverting the AI edit on %s…", image_label(image))
  yield_ui()
  local removed, failure = revert_via_sidecar(image)
  if removed == nil then
    return nil, "Could not revert the history.\n" .. tail(tostring(failure), 500)
  end
  if removed > 0 then
    discard_history_redo(image)
  end
  pcall(function()
    image:drop_cache()
  end)
  return removed
end

local function revert_edits()
  local images, truncated = target_images()
  if #images == 0 then
    if in_darkroom() then
      status_label.text = "Open a photo in the darkroom, then revert again."
    else
      status_label.text = "Select an image on the lighttable, then revert again."
    end
    dt.print("Select an image first")
    return
  end
  local lines = {}
  local any = false
  for _, image in ipairs(images) do
    local removed, failure = revert_one(image)
    if removed == nil then
      status_label.text = string.format("%s failed.\n%s", image_label(image), failure or "unknown error")
      dt.print("Revert failed")
      return
    end
    local name = image_label(image)
    if removed > 0 then
      any = true
      lines[#lines + 1] = { name = name, removed = removed }
    else
      lines[#lines + 1] = { name = name, removed = 0 }
    end
  end
  local text
  if #images == 1 then
    local item = lines[1]
    if item.removed > 0 then
      text = string.format("Reverted the AI edit on %s. Removed %s.", item.name, history_steps(item.removed))
    else
      text = string.format("No AI edit on %s.", item.name)
    end
  elseif any then
    local parts = {}
    for _, item in ipairs(lines) do
      if item.removed > 0 then
        parts[#parts + 1] = string.format("%s: removed %s.", item.name, history_steps(item.removed))
      else
        parts[#parts + 1] = string.format("%s: no AI edit.", item.name)
      end
    end
    text = "Reverted the AI edit.\n" .. table.concat(parts, "\n")
  else
    text = "No AI edit on the selected images."
  end
  if truncated then
    text = text .. string.format("\nOnly the first %d selected images were reverted.", MAX_IMAGES)
  end
  status_label.text = text
  if any then
    dt.print("AI edit reverted")
  else
    dt.print("No AI edit to revert")
  end
end

local function cancel_look_selection()
  if not looks_pending then
    return
  end
  local images = target_images()
  local image = images[1]
  if image == nil then
    looks_pending = false
    refresh_controls(shown_recipe)
    return
  end
  local work = edit_work_dir(image)
  mkdir_p(work)
  write_file(
    join_path(work, "picked.json"),
    '{"id":"cancel","title":"Cancel","source":"editor","applied":false,"cancelled":true}\n'
  )
  looks_pending = false
  preview_watch = preview_watch + 1
  refresh_controls(shown_recipe)
  status_label.text = "Look cancelled. This photo stays on its current plan."
  dt.print("Look cancelled")
end

local function open_looks()
  local images = target_images()
  if #images == 0 then
    status_label.text = "Open a photo first."
    dt.print("Select an image first")
    return false
  end
  local image = images[1]
  local work = edit_work_dir(image)
  local recipe = read_recipe(work)
  if not looks_ready(recipe) then
    status_label.text = "Press Ask first. Looks stays closed until that plan has choices."
    return false
  end
  preview_image = image
  os.remove(join_path(work, "picked.json"))
  if open_options_window(work) == false then
    looks_pending = false
    refresh_controls(recipe)
    status_label.text = "The looks window did not open."
    return false
  end
  looks_pending = true
  refresh_controls(recipe)
  status_label.text = "The looks window is open. Choose one there, or press Cancel to keep this plan."
  return true
end

local function highlighted_history_end(image)
  local xmp = sidecar_file(image)
  if not xmp then
    return nil
  end
  local body = read_file(xmp)
  if not body or body == "" then
    return nil
  end
  return tonumber(body:match('darktable:history_end="(%d+)"'))
end

-- Darktable's duplicate keeps every history row, including steps after the
-- highlight. The copy should contain only the highlighted prefix.
local function trim_copy_to_highlight(copy, ending, source)
  if copy == nil or ending == nil or type(copy.apply_sidecar) ~= "function" then
    return
  end
  local xmp = sidecar_file(copy)
  local input = xmp
  if not input and source then
    input = sidecar_file(source)
  end
  if not input then
    return
  end
  local body = read_file(input)
  local file_end = body and tonumber(body:match('darktable:history_end="(%d+)"'))
  local count = 0
  if body then
    for _ in body:gmatch('darktable:operation="') do
      count = count + 1
    end
  end
  if file_end and file_end < count then
    ending = file_end
  end
  local work = edit_work_dir(copy)
  mkdir_p(work)
  local cleaned = join_path(work, "history-copy.xmp")
  local script = join_path(script_directory(), join_path("ai_dt", "revert.py"))
  local command = table.concat({
    shell_quote(preference("python", "string")),
    " ",
    shell_quote(script),
    " --keep-prefix --end ",
    tostring(ending),
    " --xmp ",
    shell_quote(input),
    " --xmp-out ",
    shell_quote(cleaned),
  })
  local output, exit_code = run_captured(command, work)
  local removed = output and tonumber(output:match("removed%s+(%d+)"))
  if not output or exit_code ~= 0 or not removed or removed == 0 or file_size(cleaned) < 32 then
    return
  end
  local applied, result = pcall(function()
    return copy:apply_sidecar(cleaned)
  end)
  if not applied or result ~= true then
    return
  end
  if xmp then
    local body = read_file(cleaned)
    if body and body ~= "" then
      write_file(xmp, body)
    end
  end
end

local function duplicate_current()
  local images = target_images()
  local image = images[1]
  if image == nil then
    status_label.text = "Open a photo, then press Copy."
    return
  end
  local ending = highlighted_history_end(image)
  show_progress("Copying the highlighted history…", 0.4)
  local copy, fork_error = fork_version(image)
  if not copy then
    close_progress()
    status_label.text = fork_error or "Darktable could not duplicate this image."
    return
  end
  for _ = 1, 4 do
    yield_view()
  end
  trim_copy_to_highlight(copy, ending, image)
  remember_applied_look(image, copy, edit_work_dir(image), "copy")
  close_progress()
  show_in_darkroom(copy)
  status_label.text = "Working on a copy of the highlighted history. The original was left as it is."
  dt.print("Working on a copy")
end

-- image:delete() only drops the database row. The darkroom filmstrip keeps that
-- id, and opening the blank thumbnail crashes the darkroom. The library Remove
-- button reloads the collection; a grouping toggle does the same reload.
local function action_number(value)
  return type(value) == "number" and value == value
end

local function library_has_image(imgid)
  if not (dt.database and type(dt.database.get_image) == "function") then
    return nil
  end
  local ok, found = pcall(dt.database.get_image, imgid)
  if not ok then
    return nil
  end
  return found ~= nil
end

local function wait_until_image_gone(imgid)
  for _ = 1, 40 do
    local present = library_has_image(imgid)
    if present == false then
      yield_ui()
      return true
    end
    if present == nil then
      return nil
    end
    yield_ui()
  end
  return false
end

local function images_acted_on()
  if type(dt.gui.action_images) ~= "function" then
    return nil
  end
  local ok, list = pcall(dt.gui.action_images)
  if not ok or type(list) ~= "table" then
    return nil
  end
  local ids = {}
  for _, img in ipairs(list) do
    local id_ok, imgid = pcall(function()
      return tonumber(img.id)
    end)
    if id_ok and imgid then
      ids[#ids + 1] = imgid
    end
  end
  return ids
end

local function acts_only_on(ids, imgid)
  return type(ids) == "table" and #ids == 1 and ids[1] == imgid
end

local function grouping_enabled()
  if type(dt.gui.action) ~= "function" then
    return nil
  end
  local ok, value = pcall(dt.gui.action, "global/grouping")
  if not ok or not action_number(value) then
    return nil
  end
  return value ~= 0
end

local function set_grouping(enabled)
  if type(dt.gui.action) ~= "function" then
    return false
  end
  local ok, value = pcall(dt.gui.action, "global/grouping", "", enabled and "on" or "off", 1)
  yield_ui()
  return ok and action_number(value)
end

-- Two toggles return grouping to the state they started from and force the
-- filmstrip to rebuild from the library.
local function reload_filmstrip()
  local enabled = grouping_enabled()
  if enabled == nil then
    return false
  end
  if not set_grouping(not enabled) then
    return false
  end
  return set_grouping(enabled)
end

local function finish_on_original(original)
  yield_ui()
  yield_ui()
  if library_has_image(tonumber(original.id)) == false then
    status_label.text = "The original is no longer in the library, so the darkroom was left closed."
    return
  end
  show_in_darkroom(original)
  local recipe = read_recipe(edit_work_dir(original))
  history_flags(original, recipe)
  refresh_controls(recipe)
  status_label.text = "Duplicate removed. You are on the original."
  dt.print("Back on the original")
end

local function remove_duplicate()
  local images = target_images()
  local image = images[1]
  if image == nil or is_original(image) then
    can_remove = false
    refresh_controls(shown_recipe)
    status_label.text = "Remove is only for a duplicate. The original stays."
    return
  end
  local original = original_of(image)
  if original == nil or tonumber(original.id) == tonumber(image.id) then
    status_label.text = "The original could not be found, so this duplicate was left in place."
    return
  end
  if in_darkroom() then
    leave_darkroom()
    if current_view_name() == "darkroom" then
      status_label.text = "Could not close the darkroom, so the duplicate was left in place."
      return
    end
  end

  local duplicate_id = tonumber(image.id)
  local grouping = grouping_enabled()
  -- A collapsed group makes the library Remove button include the original.
  if grouping == true and not set_grouping(false) then
    show_in_darkroom(image)
    status_label.text = "Could not separate this duplicate from its group, so it was left in place."
    return
  end
  select_image(image)
  yield_ui()

  local function restore_grouping()
    if grouping == true then
      set_grouping(true)
    end
  end

  local removed = false
  if acts_only_on(images_acted_on(), duplicate_id) and type(dt.gui.action) == "function" then
    local ok, value = pcall(dt.gui.action, "lib/image/remove", 1)
    if ok and action_number(value) then
      local gone = wait_until_image_gone(duplicate_id)
      if gone == false then
        restore_grouping()
        show_in_darkroom(image)
        status_label.text = "The duplicate was left in place."
        return
      end
      removed = gone ~= false
    end
  end

  if not removed then
    if type(image.delete) ~= "function" then
      restore_grouping()
      show_in_darkroom(image)
      status_label.text = "This Darktable build cannot remove a duplicate. It was left in place."
      return
    end
    -- Reload has to be possible before the row is dropped, or the blank
    -- thumbnail stays in the filmstrip.
    if grouping_enabled() == nil then
      restore_grouping()
      show_in_darkroom(image)
      status_label.text = "The filmstrip cannot refresh, so this duplicate was left in place."
      return
    end
    local deleted = pcall(function()
      image:delete()
    end)
    local gone = deleted and wait_until_image_gone(duplicate_id)
    if gone == nil then
      gone = deleted
    end
    if not gone then
      restore_grouping()
      if library_has_image(duplicate_id) ~= false then
        show_in_darkroom(image)
      end
      status_label.text = "The duplicate was not removed."
      return
    end
    if not reload_filmstrip() then
      restore_grouping()
      status_label.text = "The duplicate was removed, but the filmstrip could not refresh. Stay on the lighttable and do not open a blank thumbnail."
      return
    end
    removed = true
  else
    reload_filmstrip()
  end

  restore_grouping()
  finish_on_original(original)
end

local function move_history(direction)
  local images = target_images()
  local image = images[1]
  if image == nil then
    status_label.text = "Open a photo, then step through its history."
    return
  end
  local xmp = sidecar_file(image)
  if not xmp then
    status_label.text = "This photo has no XMP sidecar, so the history was left as it is."
    return
  end
  if type(image.apply_sidecar) ~= "function" then
    status_label.text = "This Darktable build cannot apply a sidecar. The history was left as it is."
    return
  end
  local stepped_aside = false
  if in_darkroom() then
    status_label.text = direction == "back" and "Closing the darkroom to step back…" or "Closing the darkroom to step forward…"
    yield_ui()
    local left = leave_darkroom()
    if not left and current_view_name() == "darkroom" then
      status_label.text = "Could not close the darkroom, so the history was left as it is."
      return
    end
    stepped_aside = true
    -- The darkroom writes its open history just after the view changes.
    -- Wait for that write, or it puts the removed step back.
    for _ = 1, 6 do
      yield_view()
    end
  end
  local work = edit_work_dir(image)
  mkdir_p(work)
  local cleaned = join_path(work, "history-step.xmp")
  local script = join_path(script_directory(), join_path("ai_dt", "revert.py"))
  local command = table.concat({
    shell_quote(preference("python", "string")),
    " ",
    shell_quote(script),
    " --step-xmp --direction ",
    direction,
    " --xmp ",
    shell_quote(xmp),
    " --xmp-out ",
    shell_quote(cleaned),
    " --redo ",
    shell_quote(join_path(work, "history-redo.json")),
  })
  local output, exit_code = run_captured(command, work)
  local moved = output and output:match("moved%s+(%d+)")
  if not output or exit_code ~= 0 or moved ~= "1" or file_size(cleaned) < 32 then
    if stepped_aside then
      return_to_darkroom(image)
    end
    if moved == "0" then
      status_label.text = direction == "back" and "Already at the first history step." or "Already at the latest history step."
    else
      status_label.text = "The history was left as it is.\n" .. tail(tostring(output or exit_code), 300)
    end
    return
  end
  local name = output:match("name%s+([^\r\n]*)") or ""
  if stepped_aside then
    park_develop(image)
  end
  local failure = nil
  if current_view_name() == "darkroom" then
    failure = "Could not close the darkroom, so the history was left as it is."
  else
    local applied, result = pcall(function()
      return image:apply_sidecar(cleaned)
    end)
    if not applied or result ~= true then
      failure = "Darktable did not move the history.\n" .. tostring(result)
    else
      local body = read_file(cleaned)
      if body and body ~= "" then
        write_file(xmp, body)
      end
      for _ = 1, 4 do
        yield_view()
      end
      local kept, kept_result = pcall(function()
        return image:apply_sidecar(xmp)
      end)
      if not kept or kept_result ~= true then
        failure = "Darktable did not keep the history step.\n" .. tostring(kept_result)
      end
    end
  end
  if not failure and name ~= "" then
    local processor = join_path(script_directory(), "ai_processor.py")
    local align = table.concat({
      shell_quote(preference("python", "string")),
      " ",
      shell_quote(processor),
      dt_version_arg(),
      " --align-history --direction ",
      direction,
      " --history-name ",
      shell_quote(name),
      " --output-dir ",
      shell_quote(work),
    })
    run_captured(align, work)
  end
  if stepped_aside then
    return_to_darkroom(image)
  end
  if failure then
    status_label.text = failure
    return
  end
  local recipe = read_recipe(work)
  local body = recipe and plan_status(recipe) or ""
  local lead = direction == "back" and "Back one history step." or "Forward one history step."
  if name ~= "" then
    lead = lead .. " " .. name .. "."
  end
  history_flags(image, recipe)
  refresh_controls(recipe)
  status_label.text = body ~= "" and (lead .. "\n" .. body) or lead
  if sync_waiting_step then
    sync_waiting_step(image)
  end
end

local function on_click()
  set_panel_busy(true)
  local ok, failure = pcall(run_analysis)
  close_progress()
  set_panel_busy(false)
  if not ok then
    status_label.text = "Failed: " .. tostring(failure)
    dt.print("AI analysis failed")
  end
end

local function on_revert_click()
  set_panel_busy(true)
  local ok, failure = pcall(revert_edits)
  set_panel_busy(false)
  if not ok then
    status_label.text = "Failed: " .. tostring(failure)
    dt.print("Revert failed")
  end
end

register_preferences()

local function stored_string(name)
  local ok, value = pcall(dt.preferences.read, PREFERENCE, name, "string")
  if ok and type(value) == "string" then
    return value
  end
  return ""
end

local function list_json_names(directory)
  local names = {}
  local command
  if on_windows() then
    local folder = directory:gsub("/", "\\")
    command = 'dir /b "' .. folder .. '\\*.json"'
  else
    command = "find " .. shell_quote(directory) .. " -maxdepth 1 -name '*.json' -printf '%f\\n'"
  end
  local pipe = io.popen(command, "r")
  if not pipe then
    return names
  end
  for line in pipe:lines() do
    line = line:gsub("[\r\n]+$", "")
    local base = line:match("([^/\\]+)$") or line
    if base:lower():sub(-5) == ".json" then
      names[#names + 1] = base
    end
  end
  pipe:close()
  table.sort(names)
  return names
end

local function load_presets()
  local directory = join_path(script_directory(), join_path("recipes", "presets"))
  local loaded = {}
  for _, name in ipairs(list_json_names(directory)) do
    local contents = read_file(join_path(directory, name))
    if contents then
      local ok, payload = pcall(json.decode, contents)
      if ok and type(payload) == "table" and type(payload.id) == "string" and type(payload.title) == "string" then
        if payload.place == nil or payload.place == "frame" then
          loaded[#loaded + 1] = {
            id = payload.id,
            title = payload.title,
          }
        end
      end
    end
  end
  table.sort(loaded, function(left, right)
    return left.title:lower() < right.title:lower()
  end)
  local seen = {}
  for _, item in ipairs(loaded) do
    local title = item.title
    if seen[title] then
      title = item.title .. " (" .. item.id .. ")"
    end
    seen[item.title] = true
    item.label = title
  end
  return loaded
end


scene_entry = dt.new_widget("text_view") {
  text = "",
  editable = true,
  tooltip = "What is in this photo: the subject, the light, and the background. Not an edit instruction. Edit it if the subject is wrong. Ask uses this text.",
}

local scene_label = dt.new_widget("label") {
  label = "Photo",
  halign = "start",
}

rewrite_scene_button = dt.new_widget("button") {
  label = "Rewrite description",
  tooltip = "Ask the picture model to describe this photo again. This does not open looks or change the original.",
  clicked_callback = function()
    local images = target_images()
    local image = images[1]
    if image == nil then
      rewrite_scene_once = false
      status_label.text = "Open a photo, then rewrite the description."
      return
    end
    rewrite_scene_once = true
    set_panel_busy(true)
    local ok, recipe, failure = pcall(describe_image, image)
    close_progress()
    set_panel_busy(false)
    if not ok then
      status_label.text = "Failed: " .. tostring(recipe)
      dt.print("Description failed")
    elseif not recipe then
      status_label.text = tostring(failure or "The description did not come back.")
      dt.print("Description failed")
    end
  end,
}

steer_entry = dt.new_widget("entry") {
  text = stored_string("steer"),
  placeholder = "what should change on this photo?",
  editable = true,
  tooltip = "Ask reads this photo and writes a plan, even when a plan is already stored. Accept and Skip stay on this photo.",
}

-- text_view wraps at word and character breaks. A vertical box gives each
-- status line its own height. A horizontal row stayed one line tall, and
-- the status is not ellipsized.
-- The status box must never hold the keyboard focus. darktable (5.4 and
-- 5.6) gives the focused widget the first look at a key and runs a shortcut
-- for any key it does not consume (gui/accelerators.c). A read-only
-- text_view consumes nothing, so after a click on it capital letters became
-- shortcuts; making it insensitive fixed that but drew the text dim grey.
-- A label cannot take the focus and draws normal text. A Lua label does not
-- wrap and its setter turns on end-ellipsis, so the text is wrapped here at
-- word breaks and ellipsis is turned off again after every set: long plans
-- stay complete. status_label keeps the old ``.text =`` interface.
request_ui.status_columns = 46

function request_ui.wrap_status(text)
  local out = {}
  for line in (tostring(text or "") .. "\n"):gmatch("(.-)\n") do
    local row = ""
    for word in line:gmatch("%S+") do
      while #word > request_ui.status_columns do
        if row ~= "" then
          out[#out + 1] = row
          row = ""
        end
        out[#out + 1] = word:sub(1, request_ui.status_columns)
        word = word:sub(request_ui.status_columns + 1)
      end
      if row == "" then
        row = word
      elseif #row + 1 + #word <= request_ui.status_columns then
        row = row .. " " .. word
      else
        out[#out + 1] = row
        row = word
      end
    end
    out[#out + 1] = row
  end
  return table.concat(out, "\n")
end

request_ui.status_widget = dt.new_widget("label") {
  label = "",
  halign = "start",
  -- Selectable so the text can be copied (drag over it, then Ctrl+C).
  selectable = true,
}
request_ui.status_text = ""
status_label = setmetatable({}, {
  __index = function(_, key)
    if key == "text" then
      return request_ui.status_text
    end
    return request_ui.status_widget[key]
  end,
  __newindex = function(_, key, value)
    if key == "text" then
      request_ui.status_text = tostring(value or "")
      request_ui.status_widget.label = request_ui.wrap_status(request_ui.status_text)
      pcall(function()
        request_ui.status_widget.ellipsize = "none"
      end)
      return
    end
    request_ui.status_widget[key] = value
  end,
})
status_label.text = "Open a photo. The description is written when that box is empty. Ask writes a plan of steps to Accept or Skip."
local status_row = dt.new_widget("box") {
  orientation = "vertical",
  request_ui.status_widget,
}

run_button = dt.new_widget("button") {
  label = "Ask",
  sensitive = false,
  tooltip = "Read the open photo and write a plan of steps to Accept or Skip.",
  clicked_callback = on_click,
}


cancel_button = dt.new_widget("button") {
  label = "Cancel",
  sensitive = false,
  tooltip = "Drop the open look selection and keep the current plan. Nothing is applied.",
  clicked_callback = function()
    cancel_look_selection()
  end,
}

copy_button = dt.new_widget("button") {
  label = "Copy",
  sensitive = false,
  tooltip = "Duplicate this photo and keep working on the copy. The original stays as it is.",
  clicked_callback = function()
    set_panel_busy(true)
    local ok, failure = pcall(duplicate_current)
    close_progress()
    set_panel_busy(false)
    if not ok then
      status_label.text = "Failed: " .. tostring(failure)
      dt.print("Copy failed")
    end
  end,
}

revert_button = dt.new_widget("button") {
  label = "Revert AI edit",
  sensitive = false,
  tooltip = "Remove the AI history steps from this photo. Your other history stays. Back and Next are separate from this.",
  clicked_callback = on_revert_click,
}

remove_button = dt.new_widget("button") {
  label = "Remove",
  sensitive = false,
  tooltip = "Remove this duplicate from the library, drop it from the filmstrip, and open the original. The camera file stays. Off on the original.",
  clicked_callback = function()
    set_panel_busy(true)
    local ok, failure = pcall(remove_duplicate)
    close_progress()
    set_panel_busy(false)
    if not ok then
      status_label.text = "Failed: " .. tostring(failure)
      dt.print("Remove failed")
    end
  end,
}

-- Plan files in this photo's work folder (<config dir>/ai-editor-work/<id>/v<n>).
-- Clear plan removes only these. The darktable history, the sidecar, the
-- camera file and an opened edited.tif stay.
request_ui.plan_files = {
  "output.json", "proposal.json", "stage.json", "step.dtstyle", "edit.dtstyle", "mask.json",
  "masked.xmp", "version.json", "steer.txt", "status.txt", "arg-request.txt",
  "arg-description.txt", "arg-vibe.txt", "history-before-mask.xmp", "masks.xmp", "look.jpg",
}

function request_ui.new_plan()
  local image = target_images()[1]
  if image == nil then
    status_label.text = "Open a photo, then press Clear plan."
    return
  end
  local work = edit_work_dir(image)
  -- The photo description belongs to the photo, not the plan, so it stays.
  local description = type(scene_text) == "function" and scene_text() or ""
  local removed = 0
  for _, name in ipairs(request_ui.plan_files) do
    if os.remove(join_path(work, name)) then
      removed = removed + 1
    end
  end
  if description ~= "" then
    write_file(join_path(work, "output.json"), '{"scene": {"summary": "'
      .. json_string(description):gsub("\t", "\\t") .. '"}}')
  end
  shown_recipe = nil
  pcall(show_choices, nil)
  pcall(show_feature_pair, nil)
  -- The request box keeps its text so the next Ask can reuse it.
  refresh_controls(nil)
  status_label.text = "This photo's saved plan is cleared (" .. tostring(removed)
    .. " file(s)). Your darktable edits, request and description stay, so press Ask for a fresh plan."
end

request_ui.new_plan_button = dt.new_widget("button") {
  label = "Clear plan",
  tooltip = "Forget this photo's saved plan and start fresh. Your darktable edits stay as they are.",
  clicked_callback = function()
    local ok, failure = pcall(request_ui.new_plan)
    if not ok then
      status_label.text = "Clear plan failed: " .. tostring(failure)
    end
  end,
}

local action_row = dt.new_widget("box") {
  orientation = "horizontal",
  run_button,
  cancel_button,
  copy_button,
  remove_button,
  request_ui.new_plan_button,
}

back_button = dt.new_widget("button") {
  label = "Back",
  sensitive = false,
  tooltip = "Remove the latest history step from the stack. Next puts that same step back.",
  clicked_callback = function()
    set_panel_busy(true)
    local ok, failure = pcall(move_history, "back")
    close_progress()
    local images = target_images()
    local image = images[1]
    history_flags(image, image and read_recipe(edit_work_dir(image)) or nil)
    set_panel_busy(false)
    if not ok then
      status_label.text = "Failed: " .. tostring(failure)
      dt.print("History step failed")
    end
  end,
}

next_button = dt.new_widget("button") {
  label = "Next",
  sensitive = false,
  tooltip = "Put back the history step Back removed. On only while that step is still waiting.",
  clicked_callback = function()
    set_panel_busy(true)
    local ok, failure = pcall(move_history, "next")
    close_progress()
    local images = target_images()
    local image = images[1]
    history_flags(image, image and read_recipe(edit_work_dir(image)) or nil)
    set_panel_busy(false)
    if not ok then
      status_label.text = "Failed: " .. tostring(failure)
      dt.print("History step failed")
    end
  end,
}

local history_row = dt.new_widget("box") {
  orientation = "horizontal",
  back_button,
  next_button,
  revert_button,
}

local function on_feature_click(look)
  set_panel_busy(true)
  local ok, applied, detail = pcall(apply_choice, nil, look)
  close_progress()
  set_panel_busy(false)
  if sync_waiting_step then
    sync_waiting_step(darkroom_image())
  end
  if not ok then
    status_label.text = "Failed: " .. tostring(applied)
    dt.print("AI edit failed")
  elseif applied == false then
    status_label.text = detail
  end
end

accept_button = dt.new_widget("button") {
  label = "Accept",
  sensitive = false,
  tooltip = "",
  clicked_callback = function()
    on_feature_click("accept-feature")
  end,
}

skip_button = dt.new_widget("button") {
  label = "Skip",
  sensitive = false,
  tooltip = "",
  clicked_callback = function()
    on_feature_click("skip-feature")
  end,
}

local feature_row = dt.new_widget("box") {
  orientation = "horizontal",
  accept_button,
  skip_button,
}

local panel = dt.new_widget("box") {
  orientation = "vertical",
  scene_label,
  scene_entry,
  rewrite_scene_button,
  steer_entry,
  action_row,
  history_row,
  feature_row,
  dt.new_widget("separator") {},
  status_row,
}

local panel_registered = false

local function register_panel()
  if panel_registered then
    return
  end
  local registered, register_error = pcall(
    dt.register_lib,
    "ai_editor",
    "AI Editor",
    true,
    false,
    {
      [dt.gui.views.lighttable] = { "DT_UI_CONTAINER_PANEL_RIGHT_CENTER", 100 },
      [dt.gui.views.darkroom] = { "DT_UI_CONTAINER_PANEL_RIGHT_CENTER", 100 },
    },
    panel,
    nil,
    nil
  )
  if not registered then
    print("AI editor could not register its panel: " .. tostring(register_error))
    return
  end
  panel_registered = true
end

-- 5.6 sets darktable_gui_safe only after the first real view. Registering
-- the panel before that leaves it off the window. 5.4 has no flag.
if darktable_gui_safe == false then
  pcall(dt.register_event, script_data, "view-changed", function(event, old_view, new_view)
    local id = new_view and (new_view.id or new_view.name)
    if id == "lighttable" or id == "darkroom" then
      register_panel()
    end
  end)
else
  register_panel()
end

local function recipe_for_image(image)
  if image == nil then
    return nil
  end
  return read_recipe(edit_work_dir(image))
end

-- Accept and Skip start off. A darkroom reload, including after a module
-- is moved, keeps the plan on disk but does not turn the buttons back on.
function sync_waiting_step(image)
  if panel_is_busy or preview_applying or image == nil then
    return
  end
  local work = edit_work_dir(image)
  if shown_work ~= work then
    looks_pending = false
    preview_watch = preview_watch + 1
    save_photo_steer(shown_work)
    shown_work = work
    load_photo_steer(work)
    show_scene(read_recipe(work) or {})
  end
  local recipe = read_recipe(work)
  history_flags(image, recipe)
  if type(recipe) == "table" and picture_has_look(work) then
    recipe.chat = true
    recipe.choices = {}
    recipe.offer_looks = false
  end
  if type(recipe) ~= "table" then
    feature_waiting = false
    show_choices({})
    status_label.text = "No plan on this photo yet."
    return
  end
  show_choices(recipe)
  local body = plan_status(recipe)
  status_label.text = body ~= "" and body or "No plan on this photo yet."
end

describe_image = function(image)
  if image == nil then
    return nil, "Open a photo first."
  end
  local edge = tonumber(preference("preview", "integer")) or DEFAULTS.preview
  if edge < 256 then
    edge = 256
  elseif edge > 2048 then
    edge = 2048
  end
  local work = edit_work_dir(image)
  local mkdir_error = mkdir_p(work)
  if mkdir_error then
    return nil, mkdir_error
  end
  local preview = join_path(work, "preview.jpg")
  show_progress("Reading this photo…", 0.25)
  local export_error = export_preview(image, preview, edge)
  if export_error then
    close_progress()
    return nil, export_error
  end
  local options = processor_options(image, nil)
  options.stage = "describe"
  if rewrite_scene_once then
    options.rewrite_scene = true
    rewrite_scene_once = false
  else
    options.scene = scene_text()
  end
  show_progress("Writing the description…", 0.55)
  local _, run_error = run_processor(preview, work, image_source_path(image), "", "", options)
  close_progress()
  if run_error then
    return nil, failure_text(work, run_error)
  end
  local recipe = read_recipe(work)
  show_scene(recipe or {})
  if scene_text() == "" then
    return nil, "The description did not come back."
  end
  status_label.text = "Description is ready. Edit it if the subject is wrong."
  return recipe
end

local function image_from_event(...)
  for index = 1, select("#", ...) do
    local value = select(index, ...)
    if type(value) == "userdata" or (type(value) == "table" and value.id ~= nil) then
      return value
    end
  end
  return nil
end

local function selected_image()
  if type(dt.gui.action_images) ~= "function" then
    return nil
  end
  local ok, images = pcall(dt.gui.action_images)
  if ok and type(images) == "table" then
    return images[1]
  end
  return nil
end

pcall(dt.register_event, script_data, "darkroom-image-loaded", function(...)
  local image = image_from_event(...)
  show_loaded_look(image)
  sync_waiting_step(image)
  maybe_apply_preview_pick()
end)
-- Hovering a thumbnail is not a photo change. The lighttable selection is.
pcall(dt.register_event, script_data, "selection-changed", function()
  if in_darkroom() then
    return
  end
  local image = selected_image()
  show_loaded_look(image)
  if image ~= nil then
    sync_waiting_step(image)
  end
  maybe_apply_preview_pick()
end)
-- No change signal on the request box: refresh its status line on the
-- frequent events instead (and on every panel refresh).
function request_ui.refresh()
  if panel_is_busy or preview_applying or type(shown_recipe) ~= "table" or not status_label then
    return
  end
  local changed = request_ui.changed(shown_recipe) or request_ui.described(shown_recipe)
  if changed == request_ui.was_changed then
    return
  end
  request_ui.was_changed = changed
  local body = plan_status(shown_recipe)
  if body ~= "" then
    status_label.text = body
  end
end

pcall(dt.register_event, script_data, "mouse-over-image-changed", function()
  maybe_apply_preview_pick()
  request_ui.refresh()
end)
pcall(dt.register_event, script_data, "pixelpipe-processing-complete", function()
  request_ui.refresh()
end)
if type(dt.gui.action_images) == "function" then
  local ok, images = pcall(dt.gui.action_images)
  if ok and type(images) == "table" then
    show_loaded_look(images[1])
    sync_waiting_step(images[1])
  end
end

local startup_commands = {}
if on_windows() then
  local queue = queue_directory()
  startup_commands[#startup_commands + 1] = table.concat({
    "wscript //nologo //b ",
    shell_quote(join_path(script_directory(), "windows-hidden-worker.vbs")),
    " ",
    shell_quote(queue),
  })
end
local warm_preview = preview_server_command()
if warm_preview then
  startup_commands[#startup_commands + 1] = warm_preview
end
if on_windows() then
  start_hidden_commands(startup_commands)
elseif warm_preview then
  start_detached(warm_preview)
end

script_data.destroy = function()
  local lib = dt.gui.libs.ai_editor or dt.gui.libs["ai_editor"]
  if lib then
    lib.visible = false
  end
end

return script_data
