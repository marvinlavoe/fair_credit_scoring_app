from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT_APP = Path(__file__).resolve().parents[1] / "app.py"
PROJECT_ROOT = ROOT_APP.parent


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))
    runpy.run_path(str(ROOT_APP), run_name="__main__")
