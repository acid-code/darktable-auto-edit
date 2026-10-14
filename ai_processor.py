#!/usr/bin/env python3
"""Entry point Darktable runs: python3 ai_processor.py <preview> --output-dir <dir>."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python3 /path/to/ai_processor.py` without an installed package.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ai_dt.env import load_project_env

load_project_env()

from ai_dt.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
