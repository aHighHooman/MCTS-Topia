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
