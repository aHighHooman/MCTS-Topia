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
    return {
        "type": "action_request",
        "player_id": 0,
        "obs": {
            "tick": 1,
            "active": 0,
            "end": True,
            "lvlup": False,
            "tribes": [{"id": 0, "stars": 10, "score": 0, "tech": [], "cap": 1}],
            "cities": [{"id": 1, "p": 0, "x": 0, "y": 0, "lvl": 1, "pop": 0, "need": 2, "prod": 1, "cap": True}],
            "units": [],
            "board": {
                "size": 2,
                "terrain": [["PLAIN", "VILLAGE"], ["FOREST", "MOUNTAIN"]],
                "resource": [[None, None], [None, None]],
                "building": [[None, None], [None, None]],
                "city": [[1, 0], [0, 0]],
                "unit": [[0, 0], [0, 0]],
                "exp": [[1, 1], [1, 0]],
                "road": [[0, 0], [0, 0]],
            },
            "rank": [0],
            "rel": [["NEUTRAL"]],
        },
        "actions": actions,
    }


def test_compact_action_request_returns_non_end_action_when_scored_higher() -> None:
    message = normalize_message(
        _compact_message(
            [
                {"i": 0, "t": "END_TURN", "p": 0},
                {"i": 1, "t": "RESEARCH_TECH", "p": 0, "tech": "FISHING"},
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
