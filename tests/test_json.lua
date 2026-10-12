-- Exercise lib/json.lua against a recipe written by ai_processor.py.
-- Usage: lua5.4 tests/test_json.lua lib/json.lua /path/output.json

local json_path = arg[1]
local recipe_path = arg[2]
local json = dofile(json_path)

local function expect(condition, message)
  if not condition then
    io.stderr:write((message or "assertion failed") .. "\n")
    os.exit(1)
  end
end

local cafe = json.decode('{"mood":"caf\\u00e9","n":-0.25,"ok":true,"missing":null,"items":[1,2]}')
expect(cafe.mood == "café", "unicode escape")
expect(cafe.n == -0.25, "negative number")
expect(cafe.ok == true, "boolean")
expect(cafe.missing == json.null, "null")
expect(cafe.items[2] == 2, "array")

local pretty = json.decode('{\n  "exposure_ev": 1.5e-1,\n  "subjects": []\n}\n')
expect(math.abs(pretty.exposure_ev - 0.15) < 1e-9, "exponent")
expect(#pretty.subjects == 0, "empty array")

local handle = assert(io.open(recipe_path, "rb"))
local recipe = json.decode(handle:read("*a"))
handle:close()
expect(recipe.version == 1, "version")
expect(recipe.modules[1].operation == "exposure", "first module")
expect(type(recipe.analysis.atmosphere.exposure_ev) == "number", "exposure type")
expect(type(recipe.style.name) == "string", "style name")
expect(recipe.style.file:find("edit.dtstyle", 1, true) ~= nil, "style file")

io.write("lua json ok\n")
