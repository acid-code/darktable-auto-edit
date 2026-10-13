# AI editor for Darktable

Point it at a camera RAW file, or at a folder of them. The processor names the picture type, finds the matching text guide, and runs that guide's stages and masks. A portrait is lifted at the face and the eyes. A building is left sharp. A night frame does not grow a subject out of a window. Nothing is uploaded.

Darktable can still append a global grade to the original RAW. A style cannot store a drawn mask, so the masked picture is `edited.tif`. The style is the exposure and color step on the RAW's history. A vibe, if you type one, changes that global grade only. It does not choose the stages or the masks.

## Run it on RAW files

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 ai_processor.py /path/to/photo.NEF \
  --output-dir /tmp/ai-edit
```

The first line of the summary names the decision and the features kept sharp, for example `portrait — Portrait guide, kept person` or `animal — Wildlife guide, kept animal, perch`. That writes `edited.tif`, `output.json`, and `edit.dtstyle`. A folder of files:

```bash
python3 -m ai_dt.batch /path/to/raws --output-dir /path/to/finished
```

`--vibe "moody dark"` is optional. It nudges the global grade. The picture type still picks the guide.

Supported RAW types include DNG, CR2, CR3, NEF, NRW, ARW, RAF, ORF, RW2, PEF, and RAW. JPEG, PNG, TIFF, and WebP work the same way. The default RAW decode is half resolution, which is the practical size for a folder. Add `--full-res` when you want every pixel. The TIFF uses the camera white balance.

`rawpy` is in `requirements.txt`. Without it, a RAW file stops with an install message. A developed preview is only used when the RAW itself cannot be decoded.

## What happens on a click

1. The Lua panel exports a JPEG of the current rendering, longest edge 1280 pixels, into the Darktable config directory. That JPEG is the fallback.
2. It runs `ai_processor.py` with the original file path. When that file is a RAW, Python decodes the RAW.
3. The pipeline measures the picture, names the picture type, finds the matching guide, and writes the masked TIFF from that guide's stages.
4. It also writes `output.json` and `edit.dtstyle`.
5. Lua imports that style, applies it to the RAW, and deletes the style entry so the styles list stays clean. The history step remains and can be undone.

Work files are kept in:

```text
<darktable config>/ai-editor-work/<image id>/
  preview.jpg
  edited.tif
  output.json
  edit.dtstyle
  run.log
```

That folder is used instead of `/tmp` so the same path is visible to Python when Darktable itself is a Flatpak. The click uses half-resolution RAW. Run the command above with `--full-res` when the TIFF should match the sensor.

## What gets applied

| Recipe field | Darktable module | Role |
| --- | --- | --- |
| exposure, black | exposure (module version 6) | Brightness delta in EV |
| contrast, saturation, vibrance, warmth, tint, shadows, highlights | color balance rgb (version 5) | Global tone and color |
| primary subject box | vignette (version 4) | Darkens the area around the subject |

Exposure and color balance are appended as extra instances named **AI exposure** and **AI color**. They do not overwrite the exposure module already in the pipe. When a face, person, or animal is found, the exposure follows that subject instead of the frame average, so a dark background does not blow out a lit face, and skin is not treated as a white-balance error.

Subject boxes are also stored under `local_edits` in `output.json`, including an ellipse in normalized coordinates. A `.dtstyle` file can carry module parameters, but it cannot carry a drawn mask (those live in a separate mask table). The vignette is the spatial edit that Darktable can apply from the style alone. A dark face also gets a small shadow lift in color balance rgb, because that actually changes the dark tones.

The masked picture is `edited.tif`, not the style. Which masks are built, and how hard they push, comes from the guide for the picture type. The numbers in a guide are capped so one file cannot blow a photo out. A `.dtstyle` still cannot store a drawn mask.

The parameter layouts were checked against the exposure and color balance blobs in Darktable 5.4.1's own camera styles. Darktable upgrades exposure version 6 when it imports the style. You want **Darktable 5.0 or newer**.

## Detectors and advisors

Both are swappable from the command line, from Darktable's Lua preferences, or by editing one file.

**Detectors** (`--detector`, preference "AI editor: subject detector"):

- `auto` — YOLO if `ultralytics` is installed, MediaPipe if it is installed, and always the built-in heuristic
- `yolo` — Ultralytics only. Default weights are `yolov8n.pt`
- `mediapipe` — face boxes, plus eyes when Face Mesh is available
- `heuristic` — skin-tone face, dark eye spots, and a high-contrast line region. Needs only NumPy and Pillow

YOLOv8's COCO weights know people and animals. They do not know eyes or buildings. Eyes come from MediaPipe or the heuristic. Buildings come from the heuristic. To point YOLO at other weights, change the model path. To follow a different set of class names, edit `COCO_CATEGORIES` in `ai_dt/detectors/yolo.py`. Any class whose name contains `face`, `eye`, or `building` is picked up automatically.

**Advisors** (`--advisor`, preference "AI editor: lighting advisor"):

- `auto` — measure the histogram, and if Ollama is running, blend in a vision-model suggestion at 35%
- `histogram` — pixels only
- `ollama` — require a local Ollama vision model (default `llava`)

The model is not allowed to ignore the measurement. Its numbers are clamped, then mixed with the histogram, so a hallucinated +4 EV cannot blow the photo out.

The LLaVA prompt is `ai_dt/prompts/llava_lighting.txt`. Replace it by editing that file, passing `--prompt /path/to/prompt.txt`, or setting the Lua preference "AI editor: prompt file".

## Local reasoning model

On each run the editor tries Ollama at `127.0.0.1:11434` by itself. It sends the picture description, the vibe words, and the preset name, and asks for a short stage plan. The photo is not uploaded.

If nothing is listening and `GEMINI_API_KEY` is unset, reasoning is skipped. The steer line is still applied, and the rest of the edit continues. The note is `Reasoning was skipped. No local model and no Gemini key.`

On Windows, install Ollama with winget. Approve the Windows prompt if it appears:

```powershell
winget install --id Ollama.Ollama --accept-package-agreements --accept-source-agreements
```

The same installer is `OllamaSetup.exe` from [ollama.com/download](https://ollama.com/download). Open a new terminal so `ollama` is on `PATH`, then start the server and pull the text model the reasoner asks for:

```powershell
ollama serve
ollama pull llama3.2
```

The Ollama app listens on that port after it starts, so `ollama serve` is only needed when nothing is already there. The reasoner model is `llama3.2`. Set `OLLAMA_REASON_MODEL` to another model you have pulled. Set `OLLAMA_HOST` when the server is somewhere other than `127.0.0.1:11434`. That is the same variable the Ollama client reads, and the lighting advisor uses it too.

Gemini stays off unless `GEMINI_API_KEY` is set. It is optional.

The lighting advisor is a separate step. Its vision model defaults to `llava`, and it runs only when the advisor is `ollama` or when `auto` finds Ollama already up. Pull it with `ollama pull llava` when you want those suggestions.

## Picture types and guides

Each file in `guides/` is one picture type. The folder is only where the file lives. Matching uses the words inside it: the title, the type names, the words, the description, and the reason. Those words become a term-frequency vector, the same way a vibe matches a recipe. The query is the picture type, not the vibe and not the folder name. Cosine similarity picks the nearest guide. A small bonus is added when the guide lists that type.

Shipped types:

| Type | What the guide runs |
| --- | --- |
| portrait | Lift the face, smooth skin slightly, clarify the eyes, soften the background |
| group | The same idea, lighter, when several faces share the frame |
| animal | Keep the creature and the perch it stands on sharp, lift the animal a little, clarify the eye, defocus the foliage |
| landscape | Keep the landform sharp. The sky can be graded on its own. Usually no defocus |
| architecture | Keep the building sharp. Sky beside it can be treated differently. No subject blur |
| night | Hold highlights back. Window lights are not turned into a subject |
| food | Lift the dish a little and soften the table, less than an animal |
| still-life | Clarify one quiet object and soften the surround |
| street | Lift a small person and only slightly soften the street |
| macro | Clarify a close subject and let the corners fall soft |
| general | The measured global grade only |

A blank or unknown type, or a sentence that does not land near any guide, uses the faithful general guide. The summary and `output.json` say so. Processing a photo never writes a new guide. You add one, and the next photo can find it.

Each guide says, in plain language, which features stay sharp and which fall quiet. The sharp mask is the union of the features that are actually there. For an animal that is the creature and the perch it is standing on: a branch, a rock, a hand, or a post. Masking only the bird would blur the branch into a distraction, and masking every branch would separate nothing. The perch is found from the subject, without a new model. Nearby pixels that share the branch's brown, woody color, and that are thick and connected to the body, stay in the sharp mask. The search stops where that region becomes the fine green background. Thin twigs fall away with the needles. The edge is feathered so the branch does not get a cutout halo. The animal is still the part that is lifted. The perch stays at the global grade, only sharp.

A portrait can keep a held object or the chair the person is clearly on, the same way. A building stays sharp and the sky can be graded apart. A landscape keeps the landform and usually does not defocus. Night keeps no subject; highlights are quieted so windows stay lights.

`output.json` records `picture.type`, `picture.reason`, `guide.file`, `guide.reason`, `guide.stages`, `guide.stages_ran`, and `features` (for example `["animal", "perch"]`). The summary line names those features so you can see why the branch was kept.

Add a guide by writing the note yourself, then filing it. This does not download anything. JSON and plain text are both accepted.

```bash
python3 -m ai_dt.guides.ingest /path/to/my-guide.json
```

```json
{
  "id": "tide-pool",
  "title": "Tide pool",
  "types": ["tide-pool", "tide"],
  "words": ["tide", "pool", "water", "rocks"],
  "description": "A small tide pool. Keep the rocks sharp and do not invent a creature from the reflections.",
  "reason": "Water and rock stay as they are, so this picture keeps the global grade.",
  "keep": [],
  "quiet": [],
  "features": {},
  "stages": ["grade"],
  "masks": {
    "subject": {"build": "never", "skip": "Skip the subject. A tide pool has no creature to cut out.", "exposure": 0, "contrast": 0, "saturation": 0, "clarity": 0, "defocus": 0, "mute": 0},
    "eyes": {"build": "never", "skip": "Skip the eyes. There is no eye to clarify.", "exposure": 0, "contrast": 0, "saturation": 0, "clarity": 0, "defocus": 0, "mute": 0},
    "skin": {"build": "never", "skip": "Skip skin. Skin smoothing does not belong here.", "exposure": 0, "contrast": 0, "saturation": 0, "clarity": 0, "defocus": 0, "mute": 0},
    "sky": {"build": "never", "skip": "Skip the sky. The sky is not a separate grade here.", "exposure": 0, "contrast": 0, "saturation": 0, "clarity": 0, "defocus": 0, "mute": 0},
    "background": {"build": "never", "skip": "Skip the background. Nothing here should be blurred.", "exposure": 0, "contrast": 0, "saturation": 0, "clarity": 0, "defocus": 0, "mute": 0},
    "foreground": {"build": "never", "skip": "Skip the foreground. The global grade is enough.", "exposure": 0, "contrast": 0, "saturation": 0, "clarity": 0, "defocus": 0, "mute": 0}
  }
}
```

The same note can be plain text (`id:`, `types:`, `keep:`, `quiet:`, `feature animal:`, `stages:`, then a `mask subject` block for each mask). Grades you leave out are zero. `keep` and `quiet` name the features, and each one needs a sentence. The file is stored under `guides/<type>/`. That folder is for you. Matching still uses the words inside the file.

A mask's `build` says when it is attempted: `face`, `animal`, `subject`, `eyes`, `skin`, `sky`, `background`, `foreground`, `structure`, `highlights`, or `never`. `skip` is the sentence for a person. Only the background mask may set `defocus`. Exposure on a mask stays inside -0.35 to 0.35 EV.

The type itself comes from the detectors you already have, plus the frame: brightness, color, edges, a sky-like upper region, skin, and whether a compact body stands apart from a scene that fills the frame. A face is a portrait, a group, or a small person in a street. A solid creature of any color is an animal, not only a red or yellow bird. A branch the animal is standing on is part of that subject, not a second picture and not a reason to call the frame architecture just because the needles are busy. A dark cool frame with no compact body is night. A line-filled facade, with nothing standing apart, is architecture. Open sky over land is a landscape. If none of those fit, the type is general. YOLO is not required, and no model weights are downloaded for this decision.

## Vibes and recipes

A vibe is a sentence you type, such as `moody dark` or `crisp building`. It changes the global grade. It does not search folder names, it does not choose the guide, and a vibe named "wildlife" does not put a bird treatment on a portrait.

Each file in `recipes/` is one look. The words that get matched are the title, the scene tags, the vibe words, and the description. Those words become a term-frequency vector. Your sentence, plus the scene the detector named (portrait, animal, architecture, or general), becomes another vector. Cosine similarity picks the nearest recipe. A small bonus is added when that recipe lists the detected scene.

A blank vibe becomes the query `faithful neutral correction` plus the scene from the detectors, so the grade stays a measurement and the creative offsets stay at zero. Exposure is only allowed 15% of a recipe's exposure offset. Contrast, warmth, and saturation take a larger share, still clamped. If the sentence does not land near any recipe, the faithful correction for that scene is used instead.

The picture type and the recipe scene are separate. Night and landscape guides run when the frame measures that way. A night or landscape recipe still has to be close to the vibe sentence, as in `quiet night` or `golden sunset`.

The vectorizer is `term_frequency` in `ai_dt/recipes/library.py`. A sentence embedding can replace that one function later. The library is a folder of JSON files, so there is no separate vector database to run.

Add a look by writing the note yourself, then filing it. This does not download presets.

```bash
python3 -m ai_dt.recipes.ingest /path/to/my-note.json
```

```json
{
  "id": "portrait-bleach",
  "title": "Bleach portrait",
  "scenes": ["portrait"],
  "vibes": ["bleach", "faded", "high-key"],
  "description": "A faded high-key portrait with lifted shadows and pale color.",
  "stages": {
    "exposure_ev": 0.1,
    "contrast": -0.2,
    "temperature_shift": 0.1,
    "tint": 0.0,
    "saturation": -0.3,
    "shadows": 0.4,
    "highlights": 0.05,
    "vibrance": -0.1,
    "vignette_strength": 0.2
  }
}
```

The file is stored under `recipes/<scene>/`. That folder is for you. Matching still uses the words inside the file, which is why a note filed in the wrong folder can still be found.

From Darktable, set **AI editor: vibe** in the Lua options. From the command line, pass `--vibe "moody dark"`.

## Preview without Darktable

The preview page is a sketch of the global grade on synthetic pictures. It is not the RAW render and it is not `edited.tif`. Use the command in "Run it on RAW files" for the picture.

```bash
python3 preview_server.py
```

Open http://127.0.0.1:8741. Leave the vibe blank, then try `moody dark` on the portrait and `crisp building` on the facade.

## Install the Python side

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

That install includes `rawpy`, which opens the camera RAW files. It is enough for the heuristic detector and the histogram advisor. Optional backends:

```bash
pip install -r requirements-ai.txt
ollama pull llava
```

`ollama serve` has to be running before a click that uses the model. The first YOLO run downloads `yolov8n.pt` from the Ultralytics servers.

Check the pipeline without Darktable:

```bash
python3 ai_processor.py /path/to/photo.NEF \
  --output-dir /tmp/ai-edit \
  --detector heuristic \
  --advisor histogram \
  --vibe "moody dark"
```

A good result starts with one line naming the picture type, the guide, and the features kept sharp, then writes `/tmp/ai-edit/edited.tif`, `/tmp/ai-edit/output.json`, and `/tmp/ai-edit/edit.dtstyle`. Exit code 2 means the path was wrong. Exit code 3 means a requested backend was missing or the file could not be decoded.

## Install the Lua script

The Lua file, `lib/json.lua`, `ai_processor.py`, and the `ai_dt` folder have to stay together. Darktable finds Lua scripts in a config directory that depends on how Darktable was installed. Confirm the path on your machine with:

```bash
darktable --print-paths
```

Look for the config directory. The usual locations:

| System | Config directory |
| --- | --- |
| Linux package or AppImage | `~/.config/darktable` |
| Linux Flatpak | `~/.var/app/org.darktable.Darktable/config/darktable` |
| Linux snap | `~/snap/darktable/current` |
| macOS | `~/.config/darktable` |
| Windows | `%LOCALAPPDATA%\darktable` (typically `C:\Users\<you>\AppData\Local\darktable`) |

### Windows

Quit Darktable first. In this folder, double-click `install-windows.cmd`, or from a normal (not Administrator) PowerShell window:

```powershell
.\install-windows.cmd
```

The script looks for a real 64-bit Python 3.9–3.13. The Microsoft Store alias in `WindowsApps` is ignored. It creates `.venv`, points Darktable at that interpreter, and links this folder into `%LOCALAPPDATA%\darktable\lua\ai_editor` with a directory junction so you do not need an elevated prompt. It adds `require "ai_editor/ai_editor"` to `luarc` when that line is missing, and it will not delete a real folder that is already at the destination.

Run the same command again after you update the project. These switches cover the cases that do not fit the default:

| Switch | When |
| --- | --- |
| `-InstallPython` | No usable Python was found. Installs Python 3.12 with winget for the current user. |
| `-WithAi` | Also install YOLO, MediaPipe, and the Ollama client. |
| `-Darktable` | Darktable is portable or not on `PATH`. Pass the full path to `darktable.exe`. |
| `-ConfigDir` | You start Darktable with `--configdir`. |
| `-Copy` | A junction cannot be created (network folder). Copies the files instead. |
| `-Force` | Retarget an existing link, turn a commented-out `luarc` line back on, or replace a custom Python preference. |

Do not choose **Run as administrator**. An elevated window can write into a different user's AppData than the one Darktable uses. If Darktable is still open, the script stops: Darktable rewrites `darktablerc` when it exits and would drop the Python path.

### Linux, macOS, or Windows

1. Start Darktable once so it creates the config directory, then quit.
2. Create the Lua folder if it is not there:
   - Linux and macOS: `mkdir -p ~/.config/darktable/lua`
   - Flatpak: `mkdir -p ~/.var/app/org.darktable.Darktable/config/darktable/lua`
   - Windows, in PowerShell: `New-Item -ItemType Directory -Force "$env:LOCALAPPDATA\darktable\lua"`
3. Put this project where Darktable will scan it. A symlink keeps your checkout as the copy Darktable runs:

   ```bash
   ln -s /path/to/this/project ~/.config/darktable/lua/ai_editor
   ```

   On Windows, use `install-windows.cmd` instead of doing this by hand. It creates a junction, which does not need an Administrator prompt. A symbolic link does, unless Developer Mode is on.

   The file Darktable loads must end up at `<config>/lua/ai_editor/ai_editor.lua`.
4. Tell Darktable to load it. Create `<config>/luarc` (no extension) if you do not already have one, and add this line:

   ```lua
   require "ai_editor/ai_editor"
   ```

   `luarc.example` in this project is that one line. If `luarc` already has other `require` lines, add this one underneath them. Do not give the path a `.lua` suffix.
5. On Windows, `install-windows.cmd` sets **AI editor: Python command** to the virtual environment it created. Set that preference by hand only if you did not use the installer. `python3` is often missing, and the `python` on `PATH` may be the Store alias that does not run the analyzer.
6. Quit Darktable completely and start it again. Lua is read at startup.
7. On the lighttable, select a photo. The right-hand panel should contain **AI Editor**. Press **Run AI Analysis**.

Darktable 5 can hide panel modules. If you do not see AI Editor, right-click an empty part of the right panel and enable **AI Editor**.

From Darktable 5.6 onward you can instead enable the script from the scripts module: the folder `ai_editor` shows up there. Enable **ai_editor** only. Leave `json` alone if it is listed; `lib/json.lua` is a library, not a second plugin. Use either `luarc` or the scripts module, not both, or the panel will be registered twice.

### Flatpak

The sandbox cannot see a normal `python3` on the host. This script prefixes the command with `flatpak-spawn --host` when it is running inside Flatpak, and it writes the preview into the config directory rather than the sandbox `/tmp`. You still need to allow the host spawn and home access once:

```bash
flatpak override --user --talk-name=org.freedesktop.Flatpak org.darktable.Darktable
flatpak override --user --filesystem=home org.darktable.Darktable
```

The Python environment that has the requirements installed must be the host `python3`. If your venv is elsewhere, set **AI editor: Python command** to that interpreter's full path, for example `/home/you/ai-editor/.venv/bin/python3`.

## Preferences

After the script has started once, Darktable → preferences → **Lua options** has:

- Python command
- subject detector (`auto`, `yolo`, `mediapipe`, `heuristic`)
- lighting advisor (`auto`, `ollama`, `histogram`)
- YOLO model
- Ollama model
- prompt file (empty means the built-in prompt)
- preview size, 256 to 2048 pixels
- vibe (empty means the faithful grade; the picture type still chooses the guide)

## Recipe

`output.json` is version 1. The fields Lua needs are `style.name` and `style.file`. `picture` is the type and why it was chosen. `guide` is the file, the reason, the stages in the file, and `stages_ran`. `features` is the list of features kept sharp, such as `animal` and `perch`. `recipe_match` records which grade was nearest. The rest is there so you can see why a value was chosen.

```json
{
  "version": 1,
  "backends": {"detector": "heuristic", "advisor": "histogram"},
  "analysis": {
    "mood": "dark, neutral, flat light",
    "subjects": [
      {"label": "face", "category": "face", "confidence": 0.8, "bbox": [0.3, 0.1, 0.4, 0.6], "source": "heuristic"}
    ],
    "atmosphere": {"exposure_ev": 0.6, "contrast": 0.1, "temperature_shift": 0.0}
  },
  "modules": [
    {"operation": "exposure", "modversion": 6, "multi_name": "AI exposure", "params": {"exposure": 0.6, "black": 0.0}}
  ],
  "local_edits": [
    {"target": "face", "ellipse": {"cx": 0.5, "cy": 0.4, "rx": 0.2, "ry": 0.3, "feather": 0.35}, "exposure_ev": 0.3}
  ],
  "style": {"name": "AI Edit …", "file": "…/edit.dtstyle"}
}
```

`bbox` is `x, y, width, height` in 0–1 coordinates with the origin at the top left. `temperature_shift` is positive to warm the photo. `input.kind` is `raw` or `raster`. `picture.type` is the decided type, and `guide.file` is the guide that ran. `features` lists what stayed sharp. `separation.applied` says whether the TIFF isolated a subject. `guide.stages_ran` is the stages that actually ran, in order. `local_edits[].exposure_ev` is the masked correction recorded for that ellipse. The style can apply the vignette and the shadow lift. The TIFF is where the guide's masks are actually painted.

## Adding a backend

- New detector: implement `detect(self, image) -> list[Detection]` and call it from `_detect` in `ai_dt/pipeline.py`. `Detection` is documented in `ai_dt/schema.py`.
- New advisor: implement an object that returns an `Atmosphere`, then blend it in `_advise`.
- New prompt: replace `ai_dt/prompts/llava_lighting.txt`. The Ollama client asks for JSON and will accept a model other than LLaVA through `--ollama-model`.

## Tests

```bash
pip install -r requirements.txt
python3 -m unittest discover -s tests -t . -v
```

The tests build synthetic pictures, write a style, and check that the exposure bytes decode back to the value in the recipe. If a Lua interpreter is on `PATH`, they also parse `output.json` with the same decoder the plugin uses.

## Limits

- Each click appends another AI step. Undo that history step before running again if you do not want both.
- The first eight selected images are processed. A large selection is cut short on purpose. Use `python3 -m ai_dt.batch` for a folder.
- The panel blocks until Python returns. A RAW decode plus the background blur takes longer than the old preview pass. `run.log` in the work folder has the output.
- The style still has no drawn mask. The masks are in `edited.tif`.
- The TIFF is developed with the camera white balance, at half resolution unless you pass `--full-res`. The style does not touch Darktable's white balance module. Warmth there is a color-balance move on the history.
