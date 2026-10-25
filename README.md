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
3. The pipeline measures the picture, names the picture type, finds the matching guide, and writes the masked TIFF from that guide's stages. Accepted features are painted into that TIFF.
4. It also writes `output.json` and `edit.dtstyle`. The style is the global grade only.
5. A preview window opens with a picture of every look, including Accept and Skip when a feature is waiting. Choosing one there, or the same choice on the panel, imports a copy of the masked TIFF and opens it. The original file is not changed, and the style is not applied to it.

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

Subject boxes are also stored under `local_edits` in `output.json`, including an ellipse in normalized coordinates. A `.dtstyle` file can carry module parameters, but it cannot carry a drawn mask (those live in the sidecar's `masks_history` table). The vignette is the spatial edit a style can apply by itself. A dark face also gets a small shadow lift in color balance rgb, because that actually changes the dark tones.

The guide's masked picture is `edited.tif`. Which of those masks are built, and how hard they push, comes from the guide for the picture type. The numbers in a guide are capped so one file cannot blow a photo out. An accepted step is different: it inserts one module into the photo's existing sidecar. Eyes and a highlighted hue are drawn ellipses. A whole-frame grade is still a style. See [Sidecar history and drawn masks](#sidecar-history-and-drawn-masks).

The parameter layouts were checked against the exposure and color balance blobs in Darktable 5.4.1's own camera styles. Darktable 5.6.2 still imports those blobs and upgrades exposure version 6 to version 7. You want **Darktable 5.0 or newer**.

## Detectors and advisors

Both are swappable from the command line, from Darktable's Lua preferences, or by editing one file.

**Detectors** (`--detector`, preference "AI editor: subject detector"):

- `auto` — YOLO if `ultralytics` is installed, MediaPipe if it is installed, and always the built-in heuristic
- `yolo` — Ultralytics only. Default weights are `yolov8n.pt`
- `mediapipe` — face boxes, plus eyes when Face Mesh is available
- `heuristic` — skin-tone face, dark eye spots, and a high-contrast line region. Needs only NumPy and Pillow

YOLOv8's COCO weights know people and animals. They do not know eyes or buildings. Eyes come from MediaPipe or the heuristic. Buildings come from the heuristic. To point YOLO at other weights, change the model path. To follow a different set of class names, edit `COCO_CATEGORIES` in `ai_dt/detectors/yolo.py`. Any class whose name contains `face`, `eye`, or `building` is picked up automatically.

**Advisors** (`--advisor`, preference "AI editor: lighting advisor"):

- `auto` — measure the histogram. If `MISTRAL_API_KEY` is set, blend in a Mistral vision suggestion at 35%. Otherwise, if Ollama is running, blend in LLaVA at 35%
- `histogram` — pixels only
- `ollama` — require a local Ollama vision model (default `llava`)

The model is not allowed to ignore the measurement. Its numbers are clamped, then mixed with the histogram, so a hallucinated +4 EV cannot blow the photo out.

The lighting prompt is `ai_dt/prompts/llava_lighting.txt`. Mistral vision and LLaVA both use it. Replace it by editing that file, passing `--prompt /path/to/prompt.txt`, or setting the Lua preference "AI editor: prompt file".

## Local reasoning model

On each run the editor tries Ollama at `127.0.0.1:11434` by itself. It sends the picture description, the vibe words, and the preset name, and asks for a short stage plan. The photo is not uploaded.

If nothing is listening and `GEMINI_API_KEY` is unset, reasoning is skipped. The request line is still applied, and the rest of the edit continues. The note is `Reasoning was skipped. No local model and no Gemini key.`

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

The lighting advisor is a separate step from the text reasoner, and it does not send the picture to Gemini. With `--advisor auto`, `MISTRAL_API_KEY` sends the preview to Mistral's vision chat model. The default id is `ministral-14b-2512` (Ministral 3 14B, the documented replacement for retired Pixtral). Set `MISTRAL_VISION_MODEL` to another vision id. If that key is unset or the call fails, and Ollama is running, auto blends in LLaVA instead. `ollama` requires LLaVA. `histogram` is pixels only. Pull the local model with `ollama pull llava` when you want that fallback. These variables are read from the process environment. On startup the analyzer also loads a `.env` file next to `ai_processor.py` when that file exists. A variable already set in the environment wins over the file. Copy `.env.example` to `.env` for local keys. `.env` is not committed.

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

The darktable panel no longer has a Looks button, a vibe box, a Preset
dropdown or a "Look:" line: use darktable presets for a look. The panel
always sends a blank (faithful) vibe; the `--vibe` flag below still works
from the command line.

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

The preview page is a sketch of the global grade on synthetic pictures. It is not the RAW render and it is not `edited.tif`. After Ask, Darktable also opens http://127.0.0.1:8741/options with a picture of every look for that photo. A choice on that page and a choice on the panel write the same pick. Use the command in "Run it on RAW files" for a picture outside Darktable.

```bash
python3 preview_server.py
```

Open http://127.0.0.1:8741. Leave the vibe blank, then try `moody dark` on the portrait and `crisp building` on the facade.

## Supported darktable versions

- **darktable 5.6**: tested, including in the darkroom.
- **darktable 5.4**: supported in the code. The plugin reads the running darktable version and passes it to the Python side, which leaves 5.6-only pipeline modules (such as `colorharmonizer`) out of the order it writes. One darktable-cli render on 5.4.1 worked, but it has not been tested in the 5.4 darkroom yet.

If the version can't be read, the plugin assumes 5.6.

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
- lighting advisor (`auto`, `ollama`, `histogram`). `auto` uses Mistral when `MISTRAL_API_KEY` is set, otherwise Ollama, otherwise the histogram
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

## Sidecar history and drawn masks

Darktable 5.6.2 keeps two copies of a photo's history. The history panel shows the library (`library.db`) and the stack develop is holding. The `.xmp` next to the camera file is a third copy. They can disagree. A correct sidecar does not change the panel until `image:apply_sidecar()` runs, and only after the darkroom has left that photo. Applying it while the photo is on screen deadlocks the pixelpipe. Develop can then write its old in-memory stack back over the library and the sidecar, so Accept parks develop on another photo, leaves, applies, writes the sidecar, applies once more, and only then returns to the darkroom.

There is no Lua call that deletes one history row or attaches one shape. `image:reset()` replaces the whole stack. A style import appends module parameters and cannot store `masks_history`. The editor therefore edits the sidecar text and loads that file.

This is the same idea as [Chemigram](https://github.com/chipi/chemigram): keep the existing sidecar, write the shape into `masks_history`, and point the module's blend blob at that shape. Chemigram checked ellipses, gradients, and rectangles against real 5.4.1 files. This repo implements the ellipse, the ellipse group, closed paths, linear gradients, and groups that combine them with union, intersection, difference and inversion (`ai_dt/dt_shapes.py`). A rectangle and a brush are not written yet.

Darktable 5.6 adds an AI object mask in the mask manager. That tool is a click on the photo (SAM2.1 or SegNext). It stores a different shape, `DT_MASKS_OBJECT`, and Lua still cannot create it. Accept still writes the detected ellipse for eyes and for a hue. A subject step on a real preview uses the downloaded SAM 2.1 model the same way the mask tool does: the detector box is the prompt, and the silhouette is written as a closed path (type 2) on a Subject lift exposure. If that model is missing, the subject stays the vignette. A 5.6.2 import of a spliced sidecar kept the ellipse (mask version 6, type 32) and upgraded the exposure module from version 6 to version 7. The blend blob is still version 14 and 420 bytes.

### Drawn-mask steps: sky, landscape classes, people, background

Typing a request such as `darken the sky`, `make the trees greener, brighten the buildings`, `make the water bluer`, `lift the people`, `darken the background` or `do the landscape masks` and pressing **Ask** queues one Accept step per region (`ai_dt/mask_grammar.py`; a model plan can name the same `mask_*` operations from the catalog). Accept segments `preview.jpg`, turns the region into darktable path shapes, and splices one module into the sidecar whose drawn mask is a **group** of those paths (`ai_dt/mask_steps.py`, `ai_dt/dt_shapes.py`, `splice_shapes` in `ai_dt/drawn_mask.py`). Holes are `difference` members, the way darktable 5.6's own AI object tool stores them. A background is the inverted subject group (drawn-mask polarity in the blend blob).

| Step | History name | Module | Default grade (photoserge example 7) |
| --- | --- | --- | --- |
| `mask_sky` | Grade the sky | color balance rgb | darker, a little warmer |
| `mask_architecture` | Grade the buildings | color balance rgb | brighter, more contrast |
| `mask_vegetation` | Grade the vegetation | color balance rgb | greener (saturation, vibrance) |
| `mask_water` | Grade the water | color balance rgb | darker, bluer, softer |
| `mask_ground` | Grade the ground | color balance rgb | darker |
| `mask_people` | Lift the people | exposure | +0.25 EV (local edits stay within ±0.5 EV) |
| `mask_background` | Quiet the background | color balance rgb | darker, muted |

`mask_landscape` queues the five landscape steps in that order. A region that is not in the photo leaves its step off with a sentence.

Segmenters (`ai_dt/segment/`) are tried in order and each one is optional: darktable's own SAM 2.1 files (subject/people from a detector box), `rembg` (subject), an ADE20K SegFormer/OneFormer through `transformers` (sky, buildings, vegetation, water, ground, people), then a NumPy heuristic that always runs. `pip install -r requirements-masks.txt` adds the model backends; all run on CPU. Gemini segmentation uploads the preview, so it is off unless `AI_DT_SEGMENT_GEMINI=1` and `GEMINI_API_KEY` are set. `AI_DT_SEGMENTER=heuristic|rembg|ade20k|gemini|darktable-sam` forces one backend (plus the heuristic fallback). A guide can also paint these classes into `edited.tif` with `build` values `people`, `architecture`, `vegetation`, `water` and `ground`.

### Lesson steps, animal masks and looks (photoserge examples)

`ai_dt/lessons/` holds a trimmed copy of the lesson notes (eleven before/after edits, no images). They feed a short "editing lessons" block into the planner prompt (`reasoner._fill`, Ollama and Gemini) and the global-first order into the vision prompt. Accept steps are ordered global first: a global look, then regional grades, then gradients, then edge burn and glow; existing steps keep their order. Every local exposure stays within ±0.5 EV.

| Request (Ask) | Step | What lands in history |
| --- | --- | --- |
| `darken the sky with a gradient` | `mask_sky_gradient` | Darken the sky gradient: exposure −0.5 EV through sky outline ∩ top gradient (ex. 9) |
| `a gradient from the bottom` | `mask_bottom_gradient` | Darken the bottom: exposure −0.5 EV through a bottom gradient (ex. 8) |
| `burn the edges` | `mask_edge_burn` | Burn the edges: four edge gradients minus the subject (ex. 11) |
| `add a glow behind him` | `mask_glow` | Glow behind the subject: soft ellipse minus the subject, +0.5 EV (ex. 10) |
| `brighten the dog` / `lift the birds` / `darken around the dog` | `mask_animal` / `mask_animals` / `mask_animal_background` | One outline per animal (detector box → model or heuristic outline), union, or its inverse |
| `split tone` | `split_tone` | Split tone (color balance rgb: warm highlights, cool shadows, contrast up, vibrance down) |
| `warm-only look` / `palander` | `palander` | Warm-only look (color zones: warm hues up, others down; ex. 4, 5) |
| `keep only the red` | `keep_color` | Keep only the red (color zones chroma; ex. 6) |
| `the wildlife look` / `landscape look` | preset | The lesson steps for that picture type |

`output.json` `analysis.crop_suggestion` is a rule-of-thirds 16:9 crop (ex. 1, 8). It is advice only and is never applied.

### Two mask paths

| Path | What it changes | When it runs |
| --- | --- | --- |
| Painted TIFF | `edited.tif` pixels | Ask builds the guide's local grades (subject, background, and the rest). `masks_can_attach()` in `ai_dt/masks_xmp.py` stays false so that run does not replace history. |
| Sidecar splice | One new history module plus its shapes | Accept, and only for the step that is waiting. |

`local_edits` in `output.json` is the measurement. An ellipse there is not on the photo until Accept splices it. Coordinates are 0 to 1, origin at the top left. `cx`, `cy` are the center. `rx`, `ry` are the radii.

### What Accept already writes

Python `commit_waiting_step` (`ai_dt/features.py`) writes exactly one step into the work folder, then Lua applies it. A mask step sets `recipe.step_mask`. Anything else sets `recipe.step_style` and is imported as `step.dtstyle`.

| Waiting step | History module | Shape |
| --- | --- | --- |
| Eyes | exposure, history name `AI eyes` | One ellipse around every detected eye box. `eye_ellipse` unions the boxes and uses the largest `exposure_ev`. |
| Highlight one hue (red aprons, lanterns) | colorzones, name `Highlight the <hue>` | Up to four ellipses from `hue_sample` in `ai_dt/hue_mask.py`. Two or more are a group. `mask.json` holds the blob and the shapes. |
| Recolor (too much of a color, or Gemini asking for a more pleasing color) | channelmixerrgb when warmth or tint moves, otherwise colorbalancergb. Name `Recolor` | None. Whole frame. |
| Subject, foreground, opening, shafts, highlight warmth | exposure, vignette, or colorbalancergb | None, except a subject with a box, which is a vignette hole rather than a drawn mask. |

`ai_dt/drawn_mask.py` is the splicer. Darktable launches it by path. The eye call is `--xmp`, `--xmp-out`, `--cx`, `--cy`, `--rx`, `--ry`, `--ev`, `--name`. A hue call passes `--mask-json`. The script inserts the module. It does not delete rows.

`splice_region_module` accepts `colorzones` only, and a 520-byte parameter blob. A general mask on another module has to extend that function. The name must start with `AI ` or be one of the names in `accepted_step_names()` (`ai_dt/revert.py`), or the splice refuses it. Those names are duplicated in `editor_history_names` in `ai_editor.lua` so Back and Revert can find the row. Lua loads on the next Darktable start. Python loads on the next Accept.

### How a shape is stored

Live sidecars use blend version 14 and a 420-byte blend blob. Styles still say version 13. The splice copies the blend version already in the sidecar and starts from the default 420-byte blob.

Inside that blob, byte 0 is the mask mode (`3`, drawn and enabled), byte 16 is opacity (`100`), and byte 24 is the mask id the module blends through. An id of 0 hides the module.

Shapes go in `darktable:masks_history`. Ids start at `0x10000000`.

- One ellipse is the blend mask. Type `32` (`1 << 5`), version `6`. Points are little-endian `ffffffI`: `cx`, `cy`, `rx`, `ry`, rotation `0`, border (`0.05` for eyes, `0.04` for a hue), flags `1` (proportional).
- Two or more ellipses need a group, type `4`. The blend `mask_id` is the group id, not an ellipse id. Each member is `IIIf`: form id, group id, state `3` for the first member and `131` for each later one, opacity `1`. Every shape for that module shares the history item's `num` in `mask_num`.
- After Revert, an empty mask list is `<rdf:Seq/>` with no closing tag. The splice turns that into a real list. A missing `masks_history` element is created.

### Insert, and the slot Darktable will keep

`history_end` is how many rows are active. Rows after it stay in the stack.

- Insert the new module at that index. Add 1 to every later `darktable:num`, and to every mask `num` that is at least that index. Set `history_end` to the insert index plus 1.
- When `history_end` is already at or past the number of items, append with `max(num) + 1`. Do not renumber. A gap must not turn an active row off. `history_end` sitting past the last row means Darktable counted a module and then dropped it.

`multi_priority` has to match a pipeline slot. The default pipe already lists `exposure,0` and `colorzones,0`. A new instance at priority 1 has no slot unless `iop_order_list` gains that pair, and Darktable drops it on reload. The first instance of an operation that is not already in the history uses priority 0. A later instance uses `max(priority) + 1` and `_pipeline_slot` adds it to `iop_order_list` when that attribute exists. Many 5.4 sidecars have only `iop_order_version="4"` and no list, so priority 0 is what makes the first eye or the first color-zones module stay.

### Work folder

Accept's files live under `<darktable config>/ai-editor-work/<image id>/v<duplicate>/`.

| File | What it is |
| --- | --- |
| `history.xmp` | Sidecar copied before the new module |
| `history-before-mask.xmp` | Live sidecar after the splice succeeded, before `apply_sidecar` |
| `masked.xmp` | Spliced sidecar Accept loads |
| `mask.json` | Hue step: operation, params hex, and ellipse list |
| `step.dtstyle` | Whole-frame step |
| `history-step.xmp` | Sidecar after Back or Next |
| `history-redo.json` | The row Back removed. Names only. The XML inside is not a second stack |

A failed splice does not call `apply_sidecar`. The live sidecar stays as it was. `history-before-mask.xmp` is written only after the splice succeeds.

Back removes the last active editor row from the sidecar and stores it in `history-redo.json`. Lowering `history_end` alone only moves the highlight. The row stays. Next splices that row back. Accept clears the redo file.

The history panel reads the library, not the sidecar. Read it without writing, including while Darktable is open:

```text
.venv/Scripts/python.exe .cursor/skills/darktable-accept/scripts/read_library.py 8182
.venv/Scripts/python.exe .cursor/skills/darktable-accept/scripts/read_accept.py WORKDIR
.venv/Scripts/python.exe .cursor/skills/darktable-history/scripts/read_sidecar.py PATH.xmp
```

Do not write `library.db` or the open photo's sidecar while `darktable.exe` is running. Develop writes its stack back over both.

### For a general mask

Already done, and safe to call: ellipse points, the group wrapper, blend mode and mask id, history insert, priority 0 for the first instance, the Accept leave-and-apply sequence, and revert by history name.

Done on `feature/ai-masks`: path outlines, gradients and groups on exposure, color balance rgb and colorzones (see Drawn-mask steps above). Still open: a rectangle and a brush. A new shape should go through `drawn_mask.py` and the same Accept sequence. A style file still cannot carry it.

## Limits

- Picking a look or accepting a feature opens a new copy of the masked picture. The original file stays unchanged.
- The first eight selected images are processed. A large selection is cut short on purpose. Use `python3 -m ai_dt.batch` for a folder.
- The panel blocks until Python returns. A RAW decode plus the background blur takes longer than the old preview pass. `run.log` in the work folder has the output.
- A style still cannot store a drawn mask. Guide masks are painted into the TIFF. An accepted eye or hue step is spliced into the sidecar. See [Sidecar history and drawn masks](#sidecar-history-and-drawn-masks).
- The TIFF uses the camera white balance, at half resolution unless you pass `--full-res`. A later steer on that opened picture rereads the camera file, so the masks are not applied twice.

### Clear plan

**Clear plan** forgets this photo's saved plan and starts fresh. It deletes only
the plan files in the photo's work folder
(`<darktable config dir>/ai-editor-work/<image id>/v<duplicate>`; on Windows
`%LOCALAPPDATA%\darktable\ai-editor-work\<id>\v0`). The darktable history,
the XMP sidecar and the camera file stay as they are. Then type a request, or
leave it blank, and press **Ask**.

### Complete plan

**Complete plan** (next to Accept and Skip) accepts every remaining step in
order. Before and after each step it reads `darktable:history_end` from the
synced sidecar; a step that reports success but leaves the history unchanged
is a failure and stops the run. A step Accept leaves off with a note (no mask
found) is listed as left off and the run goes on; a worker error stops it. The
status text lists each step (✓ applied with history counts, – left off, ✗
failed) and can be selected and copied. Animal masks drop loose outlines over
water or scenery before joining the group, so a tight animal is not diluted.

## Default editor steps

Every plan from Ask starts with the standard editor steps that the photo needs.
Each one is measured from the preview and shows up as a normal step you can
Accept, Skip or go Back on (`ai_dt/default_steps.py`):

- **Recover clipping** (tone equalizer): runs when at least 0.5 % of pixels
  are clipped (any channel at 250 or more) or at least 1 % are crushed (luma
  at 5 or less). A preview estimate searches for the shadow lift and highlight
  pull that shrink both, and it keeps at least 85 % of the p5–p95 contrast.
- **Remove haze** (haze removal): runs only when the dark-channel haze score
  is 0.2 or more. Strength is 0.1–0.4, scaled to the score.
- **Sharpen the subject** (sharpen, through the subject mask): radius 1.5,
  amount 0.6, threshold 0.9. On noisy photos (noise σ ≥ 2.5, or ISO ≥ 3200)
  it drops to amount 0.35 and threshold 1.5. On very noisy ones (σ ≥ 5, or
  ISO ≥ 12800) it is skipped.
