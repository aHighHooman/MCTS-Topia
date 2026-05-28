from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GAME_ROOT = Path(r"C:\Users\Umair\OneDrive\Desktop\Work\Self_Projects\TribesTopia\Tribes")


def game_root() -> Path:
    return Path(os.environ.get("TRIBES_GAME_ROOT", DEFAULT_GAME_ROOT)).resolve()


def game_src_root() -> Path:
    return game_root() / "src"


def game_json_jar() -> Path:
    game_jar = game_root() / "lib" / "json.jar"
    return game_jar if game_jar.exists() else PROJECT_ROOT / "lib" / "json.jar"
