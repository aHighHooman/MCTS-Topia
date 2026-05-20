from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_config_defaults(parser: argparse.ArgumentParser, argv: list[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=None, help="JSON config file with argparse option names.")
    config_args, remaining = config_parser.parse_known_args(argv)
    parser.add_argument("--config", type=Path, default=config_args.config, help="JSON config file with argparse option names.")
    if config_args.config is None:
        return parser.parse_args(remaining)

    with config_args.config.open("r", encoding="utf-8") as handle:
        defaults = json.load(handle)
    if not isinstance(defaults, dict):
        raise TypeError(f"profile config must be a JSON object: {config_args.config}")

    normalized: dict[str, Any] = {}
    for key, value in defaults.items():
        normalized[str(key).replace("-", "_")] = value
    parser.set_defaults(**normalized)
    args = parser.parse_args(remaining)
    for action in parser._actions:
        if action.type is Path:
            value = getattr(args, action.dest, None)
            if isinstance(value, str):
                setattr(args, action.dest, Path(value))
    return args
