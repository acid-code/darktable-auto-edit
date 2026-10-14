--[[
AI editor for Darktable.

Propose looks for the lighttable selection or the darkroom image, then
apply the one you pick onto a new duplicate. The script reads the original
RAW when Python can open it, writes edited.tif, and appends the style to
that duplicate's history. Revert AI edit removes only those appended steps.

Load it from luarc with:  require "ai_editor/ai_editor"
The ai_processor.py file and the ai_dt package must stay next to this script.
]]

local dt = require "darktable"
local script_data = {}

script_data.metadata = {
  name = "AI editor",
  purpose = "Analyze a selected image locally and append exposure, color, and subject edits",
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

local function mkdir_p(path)
  if on_windows() then
    os.execute('mkdir "' .. path:gsub("/", "\\") .. '"')
  else
    os.execute("mkdir -p " .. shell_quote(path))
  end
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
    "auto measures the preview and, when Ollama is running, blends in a LLaVA suggestion.",
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
  if vibe ~= "" then
    parts[#parts + 1] = " --vibe "
    parts[#parts + 1] = shell_quote(vibe)
  end
  if type(steer) == "string" and steer ~= "" then
    parts[#parts + 1] = " --steer "
    parts[#parts + 1] = shell_quote(steer)
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
    return nil, "Python exited with code " .. tostring(exit_code) .. "\n" .. tail(output, 500)
  end
  return output
end

local function apply_recipe(image, work_dir)
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
local run_button
local revert_button
local vibe_entry
local steer_entry
local preset_box
local preset_ids = {}
local preset_queries = {}
local choice_buttons = {}
local choice_ids = {}

local function duplicate_index_of(image)
  local index = tonumber(image.duplicate_index)
  if index == nil then
    return 0
  end
  return index
end

local function library_path()
  return join_path(dt.configuration.config_dir, "library.db")
end

-- Same layout as ai_dt.versions.version_work_dir: <image id>/v<duplicate index>
local function work_dir_for(image)
  local root = join_path(dt.configuration.config_dir, "ai-editor-work")
  local folder = join_path(root, tostring(image.id))
  return join_path(folder, "v" .. tostring(duplicate_index_of(image)))
end

-- current_view() pushes a module entry. view.id is the module name
-- ("darkroom" or "lighttable"). Object identity is only a fallback.
local function view_name(view)
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

local function current_view_name()
  if type(dt.gui.current_view) ~= "function" then
    return nil
  end
  local ok, view = pcall(dt.gui.current_view)
  if not ok then
    return nil
  end
  return view_name(view)
end

local function in_darkroom()
  return current_view_name() == "darkroom"
end

local function darkroom_image()
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

local function fork_version(image)
  if type(image.duplicate_with_history) == "function" then
    local ok, copy = pcall(function()
      return image:duplicate_with_history()
    end)
    if ok and copy then
      return copy
    end
  end
  if dt.database and type(dt.database.duplicate_with_history) == "function" then
    local ok, copy = pcall(dt.database.duplicate_with_history, image)
    if ok and copy then
      return copy
    end
  end
  return nil, "Darktable could not duplicate this image"
end

local function set_version_name(image, sentence)
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

local function open_darkroom(image)
  pcall(function()
    dt.gui.views.darkroom.display_image(image)
  end)
end

local function read_recipe(work_dir)
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

local function propose_image(image, index, total, vibe, steer)
  local edge = tonumber(preference("preview", "integer")) or DEFAULTS.preview
  if edge < 256 then
    edge = 256
  elseif edge > 2048 then
    edge = 2048
  end
  local work = work_dir_for(image)
  local mkdir_error = mkdir_p(work)
  if mkdir_error then
    return nil, mkdir_error
  end
  local preview = join_path(work, "preview.jpg")
  status_label.text = string.format("Exporting %s (%d of %d)…", image.filename or "image", index, total)
  yield_ui()
  local export_error = export_preview(image, preview, edge)
  if export_error then
    return nil, export_error
  end
  status_label.text = string.format("Analyzing %s…", image.filename or "image")
  yield_ui()
  local options = processor_options(image, nil)
  local _, run_error = run_processor(preview, work, image_source_path(image), vibe, steer, options)
  if run_error then
    return nil, run_error
  end
  return read_recipe(work)
end

-- The new duplicate has an empty work folder. Copy the plan just proposed
-- on the source so this steer turn continues that duplicate's skips and asks.
local function carry_plan(source_dir, dest_dir)
  if source_dir == dest_dir then
    return
  end
  local contents = read_file(join_path(source_dir, "output.json"))
  if not contents then
    return
  end
  write_file(join_path(dest_dir, "output.json"), contents)
end

local function apply_image(image, vibe, steer, look)
  local copy, fork_error = fork_version(image)
  if not copy then
    return nil, fork_error
  end
  local edge = tonumber(preference("preview", "integer")) or DEFAULTS.preview
  if edge < 256 then
    edge = 256
  elseif edge > 2048 then
    edge = 2048
  end
  local work = work_dir_for(copy)
  local mkdir_error = mkdir_p(work)
  if mkdir_error then
    return nil, mkdir_error
  end
  carry_plan(work_dir_for(image), work)
  local preview = join_path(work, "preview.jpg")
  status_label.text = string.format("Exporting %s…", copy.filename or "duplicate")
  yield_ui()
  local export_error = export_preview(copy, preview, edge)
  if export_error then
    return nil, export_error
  end
  status_label.text = string.format("Applying %s…", look)
  yield_ui()
  local options = processor_options(copy, image)
  options.look = look
  local _, run_error = run_processor(preview, work, image_source_path(copy), vibe, steer, options)
  if run_error then
    return nil, run_error
  end
  local recipe, apply_error = apply_recipe(copy, work)
  if not recipe then
    return nil, apply_error
  end
  local sentence = version_sentence_of(recipe)
  set_version_name(copy, sentence)
  open_darkroom(copy)
  return sentence ~= "" and sentence or summary(recipe, copy)
end

local function remember_vibe()
  local text = tostring(vibe_entry.text or "")
  text = text:gsub("^%s+", ""):gsub("%s+$", "")
  if #text > 400 then
    text = text:sub(1, 400)
  end
  pcall(dt.preferences.write, PREFERENCE, "vibe", "string", text)
  return text
end

local function remember_steer()
  local text = ""
  if steer_entry then
    text = tostring(steer_entry.text or "")
  end
  text = text:gsub("^%s+", ""):gsub("%s+$", "")
  if #text > 400 then
    text = text:sub(1, 400)
  end
  pcall(dt.preferences.write, PREFERENCE, "steer", "string", text)
  return text
end

local function vibe_to_send()
  -- The typed words are always kept. A chosen preset is the only frame
  -- look sent to Python, so the two are not applied together.
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

local function show_choices(recipe)
  local choices = type(recipe) == "table" and recipe.choices or nil
  for index = 1, 6 do
    local button = choice_buttons[index]
    local item = type(choices) == "table" and choices[index] or nil
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
      choice_ids[index] = nil
    end
  end
end

local function run_analysis()
  local vibe = vibe_to_send()
  local steer = remember_steer()
  local images, truncated = target_images()
  if #images == 0 then
    if in_darkroom() then
      status_label.text = "Open a photo in the darkroom, then propose looks again."
    else
      status_label.text = "Select an image on the lighttable, then propose looks again."
    end
    dt.print("Select an image first")
    return
  end
  local last = ""
  local shown = nil
  for index, image in ipairs(images) do
    local recipe, failure = propose_image(image, index, #images, vibe, steer)
    if not recipe then
      status_label.text = string.format("%s failed.\n%s", image.filename or "Image", failure or "unknown error")
      dt.print("AI analysis failed")
      return
    end
    if shown == nil then
      shown = recipe
    end
    last = version_sentence_of(recipe)
    if last == "" then
      last = summary(recipe, image)
    end
  end
  show_choices(shown)
  if truncated then
    last = last .. string.format("\nOnly the first %d selected images were proposed.", MAX_IMAGES)
  end
  last = last .. "\nPick a look. Nothing is on the history until you do."
  status_label.text = last
  dt.print("Looks are ready")
end

local function apply_choice(index)
  local look = choice_ids[index]
  if type(look) ~= "string" or look == "" then
    status_label.text = "Propose looks first."
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
  for _, image in ipairs(images) do
    local text, failure = apply_image(image, vibe, steer, look)
    if not text then
      status_label.text = string.format("%s failed.\n%s", image.filename or "Image", failure or "unknown error")
      dt.print("AI edit failed")
      return
    end
    last = text
  end
  if truncated then
    last = last .. string.format("\nOnly the first %d selected images were duplicated.", MAX_IMAGES)
  end
  status_label.text = last
  dt.print("AI edit applied")
end

local function set_panel_busy(busy)
  run_button.sensitive = not busy
  if revert_button then
    revert_button.sensitive = not busy
  end
  for index, button in ipairs(choice_buttons) do
    if busy then
      button.sensitive = false
    elseif choice_ids[index] then
      button.sensitive = true
    end
  end
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

-- Windows io.popen keeps a cmd window on screen for the whole run.
-- wscript starts that same command with the window hidden and waits
-- until it exits, then writes run.exit.
local function run_hidden(command, work_dir)
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
  local vbs = join_path(script_directory(), "windows-hidden-run.vbs")
  if file_size(vbs) < 10 then
    return run_popen('"' .. command .. '" < NUL 2>&1')
  end
  run_blocking(table.concat({
    "wscript //nologo ",
    shell_quote(vbs),
    " ",
    shell_quote(cmd_path),
    " ",
    shell_quote(log_path),
    " ",
    shell_quote(exit_path),
  }))
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
  return run_popen(command .. " 2>&1")
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
  if not (dt.control and type(dt.control.execute) == "function") then
    return
  end
  if on_windows() then
    local vbs = join_path(script_directory(), "windows-hidden-run.vbs")
    -- No arguments: the script quits immediately, and wscript has no console.
    pcall(dt.control.execute, "wscript //nologo //b " .. shell_quote(vbs))
  else
    pcall(dt.control.execute, "sleep 0.05")
  end
end

local function leave_darkroom()
  if type(dt.gui.current_view) ~= "function" or not dt.gui.views or not dt.gui.views.lighttable then
    return false
  end
  pcall(dt.gui.current_view, dt.gui.views.lighttable)
  if current_view_name() == "lighttable" then
    return true
  end
  yield_view()
  if current_view_name() == "lighttable" then
    return true
  end
  yield_view()
  return current_view_name() == "lighttable"
end

local function return_to_darkroom(image)
  select_image(image)
  if type(dt.gui.current_view) ~= "function" or not dt.gui.views or not dt.gui.views.darkroom then
    return
  end
  pcall(dt.gui.current_view, dt.gui.views.darkroom)
  if current_view_name() ~= "darkroom" then
    yield_view()
    pcall(dt.gui.current_view, dt.gui.views.darkroom)
  end
  local shown = darkroom_image()
  if in_darkroom() and (not shown or shown.id ~= image.id) then
    pcall(function()
      dt.gui.views.darkroom.display_image(image)
    end)
  end
end

local function sidecar_file(image)
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

local function revert_one(image)
  status_label.text = string.format("Reverting the AI edit on %s…", image_label(image))
  yield_ui()
  local removed, failure = revert_via_sidecar(image)
  if removed == nil then
    return nil, "Could not revert the history.\n" .. tail(tostring(failure), 500)
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

local function on_click()
  set_panel_busy(true)
  local ok, failure = pcall(run_analysis)
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

local preset_ready = false

local function save_preset_choice(widget)
  if not preset_ready then
    return
  end
  local index = tonumber(widget.selected) or 1
  if index < 1 then
    index = 1
  end
  pcall(dt.preferences.write, PREFERENCE, "preset", "string", preset_ids[index] or "")
end

local vibe_label = dt.new_widget("label") {
  label = "Vibe",
  halign = "start",
}

vibe_entry = dt.new_widget("entry") {
  text = stored_string("vibe"),
  placeholder = "blank = faithful",
  editable = true,
  tooltip = "Faithful (leave blank), warm, moody, quiet. Fuji sims: provia, velvia, astia, classic chrome, acros, pro neg, eterna, nostalgic neg, reala ace. fuji or fujicam picks one simulation for the picture. A preset chosen beside this box wins over this text.",
}

preset_ids = { "" }
preset_queries = { "" }
local preset_labels = { "Faithful" }
for _, item in ipairs(load_presets()) do
  preset_ids[#preset_ids + 1] = item.id
  preset_queries[#preset_queries + 1] = (item.id:gsub("-", " "))
  preset_labels[#preset_labels + 1] = item.label
end
local saved_preset = stored_string("preset")
local preset_selected = 1
for index, preset_id in ipairs(preset_ids) do
  if preset_id ~= "" and preset_id == saved_preset then
    preset_selected = index
    break
  end
end
local preset_spec = {
  label = "Preset",
  tooltip = "Film and color presets. Faithful uses the vibe box, or the measured correction when that box is empty. Any other entry is the only frame look sent to the analyzer.",
  selected = preset_selected,
  changed_callback = save_preset_choice,
}
for index, label in ipairs(preset_labels) do
  preset_spec[index] = label
end
local preset_ok, preset_widget = pcall(function()
  return dt.new_widget("combobox")(preset_spec)
end)
if preset_ok then
  preset_box = preset_widget
  preset_ready = true
else
  print("AI editor: preset dropdown was not created: " .. tostring(preset_widget))
  preset_box = nil
end

local look_row_spec = {
  orientation = "horizontal",
  vibe_entry,
}
if preset_box then
  look_row_spec[2] = preset_box
end
local look_row = dt.new_widget("box")(look_row_spec)

steer_entry = dt.new_widget("entry") {
  text = stored_string("steer"),
  placeholder = "lift the shadows, warmer, more color",
  editable = true,
  tooltip = "Say it directly, separated by commas or the word and. lift the shadows, hold the highlights, warmer, cooler, more color, mute, more contrast, gradient, crop. epic or strong opens the caps. darken, lift, or hold a subject, foreground, rock, sky, behind, or reflection. revert or skip a stage.",
}

-- Labels stay on one line. text_view wraps at word and character breaks
-- (set by Darktable) and is stretched to the width the panel gives this row.
status_label = dt.new_widget("text_view") {
  text = "Select an image, then propose looks. A picked look is applied on a new duplicate.",
  editable = false,
}
local status_row = dt.new_widget("box") {
  orientation = "horizontal",
  status_label,
}
status_row.expand = true
status_row.fill = true

run_button = dt.new_widget("button") {
  label = "Propose looks",
  tooltip = "Read the photo and list looks. Nothing is added to the history until you pick one.",
  clicked_callback = on_click,
}

for index = 1, 6 do
  choice_buttons[index] = dt.new_widget("button") {
    label = " ",
    sensitive = false,
    tooltip = "",
    clicked_callback = function()
      set_panel_busy(true)
      local ok, failure = pcall(apply_choice, index)
      set_panel_busy(false)
      if not ok then
        status_label.text = "Failed: " .. tostring(failure)
        dt.print("AI edit failed")
      end
    end,
  }
end

revert_button = dt.new_widget("button") {
  label = "Revert AI edit",
  tooltip = "Remove the AI exposure, AI color, and AI vignette history steps from the selected images. Your other history stays. A module you renamed by hand is left in place.",
  clicked_callback = on_revert_click,
}

local panel = dt.new_widget("box") {
  orientation = "vertical",
  vibe_label,
  look_row,
  steer_entry,
  run_button,
  choice_buttons[1],
  choice_buttons[2],
  choice_buttons[3],
  choice_buttons[4],
  choice_buttons[5],
  choice_buttons[6],
  revert_button,
  dt.new_widget("separator") {},
  status_row,
}

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
end

script_data.destroy = function()
  local lib = dt.gui.libs.ai_editor or dt.gui.libs["ai_editor"]
  if lib then
    lib.visible = false
  end
end

return script_data
