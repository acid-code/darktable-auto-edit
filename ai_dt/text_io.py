"""User text between darktable's Lua and Python, kept UTF-8.

On Windows, Lua's ``io.popen`` starts ``cmd.exe`` through the ANSI code page,
so the UTF-8 bytes of a typed or Mistral-written description ("duck’s") came
into ``sys.argv`` decoded as cp850/cp437/cp1252 and were stored as "duckÔÇÖs" /
"duckâ€™s". The Lua side now writes the request and the description to UTF-8
files and passes ``--steer-file`` / ``--scene-file``; this module reads them,
repairs text that already went through a wrong code page, and turns smart
quotes into ASCII so nothing downstream depends on the console code page.
"""

from __future__ import annotations

from pathlib import Path

_SMART = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u2032": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u2033": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2026": "...",
    "\u00a0": " ",
}
# Leading characters of a UTF-8 multi-byte sequence after a wrong decode.
_MARKERS = ("Ô", "Γ", "Ã", "â€", "Â", "Ð")


def repair_mojibake(text: str) -> str:
    """Undo UTF-8 that was decoded as cp850, cp437 or cp1252 ("ÔÇÖ" / "â€™" -> "’")."""

    value = str(text or "")
    if not value or not any(marker in value for marker in _MARKERS):
        return value
    for codec in ("cp850", "cp437", "cp1252", "latin-1"):
        try:
            fixed = value.encode(codec).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if fixed != value:
            return fixed
    return value


def plain_quotes(text: str) -> str:
    value = str(text or "")
    for smart, plain in _SMART.items():
        value = value.replace(smart, plain)
    return value


def clean_text(text: str) -> str:
    """Repaired, ASCII-quoted text."""

    return plain_quotes(repair_mojibake(text))


def read_text_arg(value: str, file_path: str | None) -> str:
    """The text from ``file_path`` (UTF-8, BOM allowed) when given, else ``value``; cleaned."""

    if file_path:
        try:
            return clean_text(Path(file_path).read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError):
            pass
    return clean_text(value)
