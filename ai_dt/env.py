"""Load the repo-root ``.env`` without overriding the process environment."""

from __future__ import annotations

from pathlib import Path

_DEFAULT_LOADED = False


def project_root() -> Path:
    """Directory that contains ``ai_processor.py``."""

    return Path(__file__).resolve().parent.parent


def load_project_env(path: Path | None = None) -> bool:
    """Load ``.env`` from the repo root. Variables already in ``os.environ`` win.

    Uses python-dotenv with ``override=False``, so a test or the parent
    process can set ``OLLAMA_HOST=127.0.0.1:9`` and keep it. A missing file
    does nothing. The default path is applied at most once per process, so a
    later call cannot restore a key a test has removed.
    """

    global _DEFAULT_LOADED
    env_file = project_root() / ".env" if path is None else Path(path)
    if path is None:
        if _DEFAULT_LOADED:
            return False
        _DEFAULT_LOADED = True
    if not env_file.is_file():
        return False
    from dotenv import load_dotenv

    return bool(load_dotenv(env_file, override=False))
