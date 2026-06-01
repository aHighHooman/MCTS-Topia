from __future__ import annotations

import sys
from pathlib import Path


PY_ROOT = Path(__file__).resolve().parents[1]
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

BOT_ROOT = PY_ROOT / "bots"
if str(BOT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOT_ROOT))

from simple_bot import choose_action, score_action
from nn.encoding import normalize_message


def _compact_message(actions: list[dict]) -> dict:
    size = 2
    tiles = [[{"x": x, "y": y, "explored": True, "terrain": "PLAIN"} for x in range(size)] for y in range(size)]
    tiles[0][0]["terrain"] = "VILLAGE"
    tiles[1][0]["terrain"] = "FOREST"
    tiles[1][1]["terrain"] = "MOUNTAIN"
    tiles[0][0]["city_id"] = 1
    tiles[1][1]["explored"] = False
    return {
        "type": "action_request",
        "player_id": 0,
        "observation": {
            "tick": 1,
            "active_player_id": 0,
            "can_end_turn": True,
            "leveling_up": False,
            "tribes": [{"id": 0, "stars": 10, "score": 0, "researched_tech_ids": [], "capital_id": 1}],
            "cities": [{"id": 1, "tribe_id": 0, "x": 0, "y": 0, "level": 1, "population": 0, "population_need": 2, "production": 1, "is_capital": True}],
            "units": [],
            "board": {
                "size": size,
                "tiles": tiles,
            },
            "ranking": [0],
            "relationships": [["NEUTRAL"]],
        },
        "actions": actions,
    }


def test_compact_action_request_returns_non_end_action_when_scored_higher() -> None:
    message = normalize_message(
        _compact_message(
            [
                {"id": "A0", "type": "END_TURN", "tribe_id": 0},
                {"id": "A1", "type": "RESEARCH_TECH", "tribe_id": 0, "tech": "FISHING"},
            ]
        )
    )

    assert choose_action(message)["actionId"] == "A1"


def test_research_actions_score_compact_tech_field() -> None:
    message = normalize_message(_compact_message([]))
    view = __import__("simple_bot").ObservationView(message)

    assert score_action({"type": "RESEARCH_TECH", "tech": "FISHING"}, view) == 5
    assert score_action({"type": "RESEARCH_TECH", "tech": "RIDING"}, view) == 4


def test_compact_tiles_without_visible_treat_explored_as_visible() -> None:
    message = normalize_message(_compact_message([]))
    view = __import__("simple_bot").ObservationView(message)

    assert len(view.visible_tiles) == 3
    assert len(view.explored_tiles) == 3


def test_normalize_message_expands_java_compact_payload() -> None:
    message = normalize_message(
        {
            "type": "action_request",
            "player_id": 1,
            "obs": {
                "tick": 2,
                "active": 1,
                "end": True,
                "lvlup": False,
                "tribes": [{"id": 1, "stars": 7, "score": 3, "tech": ["RIDING"], "cap": 10}],
                "cities": [{"id": 10, "p": 1, "x": 0, "y": 0, "lvl": 2, "pop": 1, "need": 3, "prod": 2, "cap": True, "b": [{"t": "SAWMILL", "x": 0, "y": 0}]}],
                "units": [{"id": 20, "p": 1, "c": 10, "t": "RIDER", "x": 1, "y": 0, "hp": 10, "mhp": 10, "k": 0, "v": False, "s": "FRESH", "atk": 2, "def": 1, "mov": 2, "r": 1}],
                "board": {
                    "size": 2,
                    "terrain": [["CITY", "PLAIN"], ["FOREST", "MOUNTAIN"]],
                    "resource": [[None, None], [None, None]],
                    "building": [[None, None], [None, None]],
                    "city": [[10, 0], [0, 0]],
                    "unit": [[0, 20], [0, 0]],
                    "exp": [[True, True], [False, True]],
                    "road": [[False, True], [False, False]],
                },
                "rank": [1],
                "rel": [["NEUTRAL"]],
            },
            "actions": [
                {"i": 0, "t": "MOVE", "u": 20, "x": 1, "y": 1},
                {"i": 1, "t": "BUILD_EMBASSY", "p": 1, "tp": 2},
            ],
            "fm": {"root": "root"},
        }
    )

    assert message["observation"]["active_player_id"] == 1
    assert message["observation"]["can_end_turn"] is True
    assert message["observation"]["tribes"][0]["researched_tech_ids"] == ["RIDING"]
    assert message["observation"]["cities"][0]["tribe_id"] == 1
    assert message["observation"]["cities"][0]["buildings"][0]["type"] == "SAWMILL"
    assert message["observation"]["units"][0]["type"] == "RIDER"
    assert message["observation"]["board"]["tiles"][0][1]["unit_id"] == 20
    assert message["observation"]["board"]["tiles"][0][1]["visible"] is True
    assert message["actions"][0]["id"] == "A0"
    assert message["actions"][0]["type"] == "MOVE"
    assert message["actions"][0]["unit_id"] == 20
    assert message["actions"][1]["target_player_id"] == 2
    assert message["forward_model"]["root"] == "root"
