-- JSON decoder used by ai_editor.lua.
-- This file is a library. Darktable's script list should enable ai_editor only.

local json = {}

json.null = setmetatable({}, {
  __tostring = function()
    return "null"
  end,
})

local function skip(source, index)
  while index <= #source do
    local character = source:sub(index, index)
    if character ~= " " and character ~= "\t" and character ~= "\n" and character ~= "\r" then
      break
    end
    index = index + 1
  end
  return index
end

local function decode_error(index, message)
  error(string.format("JSON error at character %d: %s", index, message), 2)
end

local function utf8_char(codepoint)
  if utf8 and utf8.char then
    return utf8.char(codepoint)
  end
  if codepoint < 0x80 then
    return string.char(codepoint)
  end
  if codepoint < 0x800 then
    return string.char(
      0xC0 + math.floor(codepoint / 0x40),
      0x80 + (codepoint % 0x40)
    )
  end
  if codepoint < 0x10000 then
    return string.char(
      0xE0 + math.floor(codepoint / 0x1000),
      0x80 + (math.floor(codepoint / 0x40) % 0x40),
      0x80 + (codepoint % 0x40)
    )
  end
  return string.char(
    0xF0 + math.floor(codepoint / 0x40000),
    0x80 + (math.floor(codepoint / 0x1000) % 0x40),
    0x80 + (math.floor(codepoint / 0x40) % 0x40),
    0x80 + (codepoint % 0x40)
  )
end

local parse_value

local function parse_string(source, index)
  local parts = {}
  index = index + 1
  while index <= #source do
    local character = source:sub(index, index)
    if character == '"' then
      return table.concat(parts), index + 1
    end
    if character == "\\" then
      local escaped = source:sub(index + 1, index + 1)
      local simple = {
        ['"'] = '"',
        ["\\"] = "\\",
        ["/"] = "/",
        b = "\b",
        f = "\f",
        n = "\n",
        r = "\r",
        t = "\t",
      }
      if simple[escaped] then
        parts[#parts + 1] = simple[escaped]
        index = index + 2
      elseif escaped == "u" then
        local hex = source:sub(index + 2, index + 5)
        if not hex:match("^%x%x%x%x$") then
          decode_error(index, "bad unicode escape")
        end
        local codepoint = tonumber(hex, 16)
        index = index + 6
        if codepoint >= 0xD800 and codepoint <= 0xDBFF then
          local low_hex = source:sub(index, index + 5):match("^\\u(%x%x%x%x)$")
          if not low_hex then
            decode_error(index, "lone surrogate")
          end
          local low = tonumber(low_hex, 16)
          codepoint = 0x10000 + (codepoint - 0xD800) * 0x400 + (low - 0xDC00)
          index = index + 6
        end
        parts[#parts + 1] = utf8_char(codepoint)
      else
        decode_error(index, "bad escape")
      end
    else
      parts[#parts + 1] = character
      index = index + 1
    end
  end
  decode_error(index, "unterminated string")
end

local function parse_number(source, index)
  local start = index
  if source:sub(index, index) == "-" then
    index = index + 1
  end
  if source:sub(index, index):match("%d") == nil then
    decode_error(start, "expected a number")
  end
  while source:sub(index, index):match("%d") do
    index = index + 1
  end
  if source:sub(index, index) == "." then
    index = index + 1
    if source:sub(index, index):match("%d") == nil then
      decode_error(start, "expected digits after the decimal point")
    end
    while source:sub(index, index):match("%d") do
      index = index + 1
    end
  end
  local exponent = source:sub(index, index)
  if exponent == "e" or exponent == "E" then
    index = index + 1
    local sign = source:sub(index, index)
    if sign == "+" or sign == "-" then
      index = index + 1
    end
    if source:sub(index, index):match("%d") == nil then
      decode_error(start, "expected an exponent")
    end
    while source:sub(index, index):match("%d") do
      index = index + 1
    end
  end
  local token = source:sub(start, index - 1)
  local number = tonumber(token)
  if number == nil then
    decode_error(start, "invalid number " .. token)
  end
  return number, index
end

local function parse_literal(source, index, word, value)
  if source:sub(index, index + #word - 1) ~= word then
    decode_error(index, "expected " .. word)
  end
  return value, index + #word
end

local function parse_array(source, index)
  local array = {}
  index = skip(source, index + 1)
  if source:sub(index, index) == "]" then
    return array, index + 1
  end
  while true do
    local value, next_index = parse_value(source, index)
    array[#array + 1] = value
    index = skip(source, next_index)
    local mark = source:sub(index, index)
    if mark == "]" then
      return array, index + 1
    end
    if mark ~= "," then
      decode_error(index, "expected ',' or ']'")
    end
    index = skip(source, index + 1)
  end
end

local function parse_object(source, index)
  local object = {}
  index = skip(source, index + 1)
  if source:sub(index, index) == "}" then
    return object, index + 1
  end
  while true do
    if source:sub(index, index) ~= '"' then
      decode_error(index, "expected an object key")
    end
    local key, after_key = parse_string(source, index)
    index = skip(source, after_key)
    if source:sub(index, index) ~= ":" then
      decode_error(index, "expected ':'")
    end
    local value, after_value = parse_value(source, skip(source, index + 1))
    object[key] = value
    index = skip(source, after_value)
    local mark = source:sub(index, index)
    if mark == "}" then
      return object, index + 1
    end
    if mark ~= "," then
      decode_error(index, "expected ',' or '}'")
    end
    index = skip(source, index + 1)
  end
end

parse_value = function(source, index)
  index = skip(source, index)
  local character = source:sub(index, index)
  if character == '"' then
    return parse_string(source, index)
  end
  if character == "{" then
    return parse_object(source, index)
  end
  if character == "[" then
    return parse_array(source, index)
  end
  if character == "t" then
    return parse_literal(source, index, "true", true)
  end
  if character == "f" then
    return parse_literal(source, index, "false", false)
  end
  if character == "n" then
    return parse_literal(source, index, "null", json.null)
  end
  if character == "-" or character:match("%d") then
    return parse_number(source, index)
  end
  decode_error(index, "unexpected '" .. character .. "'")
end

function json.decode(source)
  if type(source) ~= "string" then
    error("json.decode expected a string", 2)
  end
  local value, index = parse_value(source, 1)
  index = skip(source, index)
  if index <= #source then
    decode_error(index, "trailing data")
  end
  return value
end

return json
