--[[
AI editor for Darktable.

Select one or more images on the lighttable and press Run AI Analysis.
The script reads the original RAW when Python can open it, writes edited.tif
(subject, eye, and background masks), and appends the global .dtstyle to
that image's history stack.

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
    "Optional words for the grade, such as moody or crisp. Empty keeps a faithful correction. The picture type still chooses the guide, the stages, and the masks.",
    DEFAULTS.vibe
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

local function run_processor(preview, work_dir, original)
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
  local vibe = preference("vibe", "string")
  if type(vibe) == "string" and vibe ~= "" then
    parts[#parts + 1] = " --vibe "
    parts[#parts + 1] = shell_quote(vibe)
  end
  if type(original) == "string" and original ~= "" then
    parts[#parts + 1] = " --original "
    parts[#parts + 1] = shell_quote(original)
  end
  local command = table.concat(parts)
  local pipe, popen_error = io.popen(command .. " 2>&1", "r")
  if not pipe then
    return nil, popen_error or "could not start Python"
  end
  local output = pipe:read("*a") or ""
  local closed, _, code = pipe:close()
  write_file(join_path(work_dir, "run.log"), output)
  local exit_code = 0
  if closed ~= true then
    exit_code = type(code) == "number" and code or 1
  end
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
  local render = ""
  if input.kind == "raw" then
    render = "\nRead the RAW and wrote edited.tif."
  else
    render = "\nWrote edited.tif."
  end
  if separation.applied == true then
    render = render .. " Subject isolated, background defocused."
    if (tonumber(separation.eyes) or 0) > 0 then
      render = render .. " Eye mask applied."
    end
  end
  return string.format(
    "%s%s\n%+.2f EV, contrast %+.2f, warmth %+.2f\n%d subject region(s) via %s / %s%s%s\nHistory updated for %s.",
    named,
    tostring(analysis.mood or "Edit applied"),
    tonumber(atmosphere.exposure_ev) or 0,
    tonumber(atmosphere.contrast) or 0,
    tonumber(atmosphere.temperature_shift) or 0,
    #subjects,
    tostring(backends.detector or "detector"),
    tostring(backends.advisor or "advisor"),
    look,
    render,
    image.filename or "image"
  )
end

local status_label
local run_button

local function process_image(image, index, total)
  local edge = tonumber(preference("preview", "integer")) or DEFAULTS.preview
  if edge < 256 then
    edge = 256
  elseif edge > 2048 then
    edge = 2048
  end
  local root = join_path(dt.configuration.config_dir, "ai-editor-work")
  local work = join_path(root, tostring(image.id))
  local mkdir_error = mkdir_p(work)
  if mkdir_error then
    return nil, mkdir_error
  end
  local preview = join_path(work, "preview.jpg")
  status_label.label = string.format("Exporting %s (%d of %d)…", image.filename or "image", index, total)
  yield_ui()
  local export_error = export_preview(image, preview, edge)
  if export_error then
    return nil, export_error
  end
  status_label.label = string.format("Analyzing %s…", image.filename or "image")
  yield_ui()
  local _, run_error = run_processor(preview, work, image_source_path(image))
  if run_error then
    return nil, run_error
  end
  status_label.label = string.format("Applying edits to %s…", image.filename or "image")
  yield_ui()
  local recipe, apply_error = apply_recipe(image, work)
  if not recipe then
    return nil, apply_error
  end
  return summary(recipe, image) .. "\nRecipe: " .. join_path(work, "output.json")
end

local function run_analysis()
  local images = selected_images()
  if #images == 0 then
    status_label.label = "Select an image on the lighttable, then run the analysis again."
    dt.print("Select an image first")
    return
  end
  local truncated = false
  if #images > MAX_IMAGES then
    truncated = true
    while #images > MAX_IMAGES do
      table.remove(images)
    end
  end
  local last = ""
  for index, image in ipairs(images) do
    local text, failure = process_image(image, index, #images)
    if not text then
      status_label.label = string.format("%s failed.\n%s", image.filename or "Image", failure or "unknown error")
      dt.print("AI analysis failed")
      return
    end
    last = text
    dt.print("AI edit applied to " .. (image.filename or "image"))
  end
  if truncated then
    last = last .. string.format("\nOnly the first %d selected images were processed.", MAX_IMAGES)
  end
  status_label.label = last
end

local function on_click()
  run_button.sensitive = false
  local ok, failure = pcall(run_analysis)
  run_button.sensitive = true
  if not ok then
    status_label.label = "Failed: " .. tostring(failure)
    dt.print("AI analysis failed")
  end
end

register_preferences()

status_label = dt.new_widget("label") {
  label = "Select an image, then run a local analysis. Edits are appended to the history stack.",
  selectable = true,
}

run_button = dt.new_widget("button") {
  label = "Run AI Analysis",
  tooltip = "Export a preview, run the local analyzer, and append a style to the history stack.",
  clicked_callback = on_click,
}

local panel = dt.new_widget("box") {
  orientation = "vertical",
  run_button,
  dt.new_widget("separator") {},
  status_label,
}

local registered, register_error = pcall(
  dt.register_lib,
  "ai_editor",
  "AI Editor",
  true,
  false,
  { [dt.gui.views.lighttable] = { "DT_UI_CONTAINER_PANEL_RIGHT_CENTER", 100 } },
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
