from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from test_native_mcts import _message_with_capital_capture


ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT / "out" / "native" / "native_static_mcts_bot.exe"


def _require_exe() -> Path:
    if not EXE.exists():
        pytest.skip("native static MCTS executable is not built; run scripts/build_native_static_bot.ps1")
    return EXE


def _compact_board(board: dict) -> dict:
    size = int(board["size"])
    tiles = board["tiles"]
    compact = {
        "size": size,
        "terrain": [[None for _ in range(size)] for _ in range(size)],
        "resource": [[None for _ in range(size)] for _ in range(size)],
        "building": [[None for _ in range(size)] for _ in range(size)],
        "city": [[0 for _ in range(size)] for _ in range(size)],
        "unit": [[0 for _ in range(size)] for _ in range(size)],
        "exp": [[False for _ in range(size)] for _ in range(size)],
        "road": [[False for _ in range(size)] for _ in range(size)],
    }
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            compact["terrain"][y][x] = tile.get("terrain")
            compact["resource"][y][x] = tile.get("resource")
            compact["building"][y][x] = tile.get("building")
            compact["city"][y][x] = int(tile.get("city_id", tile.get("city", 0)) or 0)
            compact["unit"][y][x] = int(tile.get("unit_id", tile.get("unit", 0)) or 0)
            compact["exp"][y][x] = bool(tile.get("explored", tile.get("visible", False)))
            compact["road"][y][x] = bool(tile.get("road", False))
    return compact


def _compact_payload(message: dict) -> dict:
    observation = message["observation"]
    return {
        "type": "action_request",
        "player_id": message.get("player_id", 0),
        "obs": {
            "tick": observation.get("tick", 0),
            "active": observation.get("active_player_id", message.get("player_id", 0)),
            "end": observation.get("can_end_turn", False),
            "lvlup": observation.get("leveling_up", False),
            "mode": observation.get("mode"),
            "board": _compact_board(observation["board"]),
            "units": [
                {
                    "id": unit.get("id"),
                    "p": unit.get("tribe_id"),
                    "c": unit.get("city_id"),
                    "t": unit.get("type"),
                    "x": unit.get("x"),
                    "y": unit.get("y"),
                    "hp": unit.get("current_hp"),
                    "mhp": unit.get("max_hp"),
                    "k": unit.get("kills", 0),
                    "v": unit.get("is_veteran", False),
                    "s": unit.get("status"),
                    "h": unit.get("is_hidden", False),
                    "atk": unit.get("attack", 0),
                    "def": unit.get("defence", 0),
                    "mov": unit.get("movement", 0),
                    "r": unit.get("range", 0),
                }
                for unit in observation.get("units", [])
            ],
            "cities": [
                {
                    "id": city.get("id"),
                    "p": city.get("tribe_id"),
                    "x": city.get("x"),
                    "y": city.get("y"),
                    "lvl": city.get("level"),
                    "pop": city.get("population"),
                    "need": city.get("population_need"),
                    "prod": city.get("production"),
                    "cap": city.get("is_capital", False),
                    "wall": city.get("has_walls", False),
                    "b": city.get("buildings", []),
                }
                for city in observation.get("cities", [])
            ],
            "tribes": [
                {
                    "id": tribe.get("id"),
                    "stars": tribe.get("stars", 0),
                    "score": tribe.get("score", 0),
                    "res": tribe.get("result", tribe.get("res")),
                    "tech": tribe.get("researched_tech_ids", tribe.get("tech", [])),
                    "cap": tribe.get("capital_id", tribe.get("cap", 0)),
                    "cities": tribe.get("cities", tribe.get("city_ids", [])),
                    "extra": tribe.get("extra_units", tribe.get("extra_unit_ids", [])),
                }
                for tribe in observation.get("tribes", [])
            ],
        },
        "actions": [
            {
                "i": index,
                "id": action.get("id", f"A{index}"),
                "t": action.get("type"),
                "u": action.get("unit_id", 0),
                "c": action.get("city_id", 0),
                "p": action.get("tribe_id", 0),
                "tu": action.get("target_unit_id", 0),
                "tc": action.get("target_city_id", 0),
                "tp": action.get("target_player_id", -1),
                "ct": action.get("capture_type"),
                **({"x": action["destination"]["x"], "y": action["destination"]["y"]} if isinstance(action.get("destination"), dict) else {}),
            }
            for index, action in enumerate(message.get("actions", []))
        ],
    }


def test_native_static_mcts_exe_protocol_smoke_without_forward_model() -> None:
    message = _message_with_capital_capture(second_action={"id": "end", "type": "END_TURN"})
    message["type"] = "action_request"
    completed = subprocess.run(
        [str(_require_exe()), "--simulations", "0", "--deterministic"],
        input=json.dumps(message) + "\n",
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )

    response = json.loads(completed.stdout.strip().splitlines()[-1])

    assert response["actionId"] == "capture"
    assert response["rankedActionIds"][:2] == ["capture", "end"]


def test_native_static_mcts_exe_uses_native_tree_without_forward_model_protocol() -> None:
    message = _message_with_capital_capture(second_action={"id": "end", "type": "END_TURN"})
    message["type"] = "action_request"
    completed = subprocess.run(
        [str(_require_exe()), "--simulations", "4", "--search-batch-size", "2", "--deterministic", "--seed", "1"],
        input=json.dumps(message) + "\n",
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )

    output_lines = completed.stdout.strip().splitlines()
    assert len(output_lines) == 1
    response = json.loads(output_lines[-1])
    assert response["actionId"] == "capture"
    assert response["rankedActionIds"][0] == "capture"


def test_native_static_mcts_exe_profile_json_includes_root_action_stats() -> None:
    message = _message_with_capital_capture(second_action={"id": "end", "type": "END_TURN"})
    message["type"] = "action_request"
    completed = subprocess.run(
        [str(_require_exe()), "--simulations", "4", "--search-batch-size", "2", "--deterministic", "--profile-json", "--seed", "1"],
        input=json.dumps(message) + "\n",
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )

    response = json.loads(completed.stdout.strip().splitlines()[-1])
    stats = response["_profile"]["root_action_stats"]

    assert stats
    assert {"action_id", "visits", "visit_share", "value_sum", "q_mean"} <= set(stats[0])



def test_native_static_mcts_exe_accepts_java_compact_protocol() -> None:
    message = _message_with_capital_capture(second_action={"id": "end", "type": "END_TURN"})
    compact = _compact_payload(message)
    completed = subprocess.run(
        [str(_require_exe()), "--simulations", "0", "--deterministic"],
        input=json.dumps(compact) + "\n",
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )

    response = json.loads(completed.stdout.strip().splitlines()[-1])

    assert response["actionId"] == "capture"
    assert response["rankedActionIds"][:2] == ["capture", "end"]


def test_native_static_mcts_exe_turn_cmab_returns_legal_root_action() -> None:
    message = _message_with_capital_capture(second_action={"id": "end", "type": "END_TURN"})
    message["type"] = "action_request"
    completed = subprocess.run(
        [
            str(_require_exe()),
            "--search-mode",
            "turn-cmab",
            "--simulations",
            "16",
            "--turn-cmab-max-turn-depth",
            "1",
            "--turn-cmab-max-primitives-per-turn",
            "8",
            "--deterministic",
            "--profile-json",
            "--seed",
            "13",
        ],
        input=json.dumps(message) + "\n",
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )

    response = json.loads(completed.stdout.strip().splitlines()[-1])
    legal_ids = {action["id"] for action in message["actions"]}

    assert response["actionId"] in legal_ids
    assert response["rankedActionIds"][0] in legal_ids
    assert response["_profile"]["search_mode"] == "turn-cmab"
