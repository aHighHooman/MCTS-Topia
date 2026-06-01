from __future__ import annotations

import json
import subprocess
import unittest
import sys
from pathlib import Path


PY_ROOT = Path(__file__).resolve().parents[1]
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from search.config import HybridAgentConfig
from nn.encoding import (
    ACTION_FEATURE_INDEX,
    ACTION_NATIVE_CONTEXT_START,
    BOARD_FEATURE_INDEX,
    BUILDING_TYPES,
    LEVEL_UP_BONUS_TYPES,
    RESOURCE_TYPES,
    SCALAR_MY_TECH_START,
    TECH_TYPES,
    TERRAIN_TYPES,
    UNIT_FEATURE_INDEX,
    UNIT_STATUS_TYPES,
    encode_observation,
    normalize_message,
)
from nn.model import HybridPolicyValueNet
from search.native import run_native_hybrid_mcts, run_native_mcts, run_native_static_mcts
from search.native.hybrid_mcts import _Evaluation, _mix_evaluation
from search.native.cpp_extension import load_native_mcts_extension
from search.native.mcts import NativeSearchParityError, _apply_end_turn_visit_guard, _message_cache_key, _root_priors
from search.native.parity_runner import _canonical_state, run_parity, parse_args


def _message() -> dict:
    size = 4
    return normalize_message({
        "player_id": 0,
        "observation": {
            "active_player_id": 0,
            "tick": 0,
            "can_end_turn": True,
            "board": {
                "size": size,
                "tiles": [
                    [
                        {
                            "x": x,
                            "y": y,
                            "visible": True,
                            "explored": True,
                            "terrain": "PLAIN",
                        }
                        for x in range(size)
                    ]
                    for y in range(size)
                ],
            },
            "units": [],
            "cities": [
                {"id": 10, "tribe_id": 0, "x": 1, "y": 1, "level": 1, "population": 0, "population_need": 2, "production": 2, "is_capital": True}
            ],
            "tribes": [
                {"id": 0, "stars": 10, "score": 0, "researched_tech_ids": ["ROADS"], "cities": [10], "extra_units": []},
                {"id": 1, "stars": 10, "score": 0, "researched_tech_ids": ["ROADS"], "cities": [], "extra_units": []},
            ],
        },
        "actions": [
            {"id": "end", "type": "END_TURN"},
            {"id": "spawn", "type": "SPAWN", "position": {"x": 1, "y": 1}, "x": 1, "y": 1},
            {"id": "road", "type": "BUILD_ROAD", "position": {"x": 1, "y": 2}, "x": 1, "y": 2},
        ],
    })


def _message_with_unit_move() -> dict:
    message = _message()
    for t in message["observation"]["tribes"]:
        t["stars"] = 0
        t["researched_tech_ids"] = []
    message["observation"]["units"] = [
        {
            "id": 1,
            "tribe_id": 0,
            "city_id": 10,
            "type": "WARRIOR",
            "x": 1,
            "y": 1,
            "current_hp": 10,
            "max_hp": 10,
            "kills": 0,
            "is_veteran": False,
            "status": "FRESH",
            "is_hidden": False,
        }
    ]
    message["observation"]["board"]["tiles"][1][1]["unit_id"] = 1
    message["actions"] = [
        {"id": "move", "type": "MOVE", "unit_id": 1, "destination": {"x": 2, "y": 1}, "x": 2, "y": 1},
        {"id": "road", "type": "BUILD_ROAD", "tribe_id": 0, "position": {"x": 1, "y": 2}, "x": 1, "y": 2},
    ]
    return message


def _message_with_infiltrate() -> dict:
    message = _message()
    observation = message["observation"]
    observation["tribes"] = [
        {"id": 0, "stars": 3, "score": 0, "researched_tech_ids": ["DIPLOMACY"], "cities": [10], "extra_units": []},
        {"id": 1, "stars": 8, "score": 0, "researched_tech_ids": [], "cities": [20], "extra_units": []},
    ]
    observation["cities"] = [
        {
            "id": 10,
            "tribe_id": 0,
            "x": 0,
            "y": 0,
            "level": 1,
            "population": 0,
            "population_need": 2,
            "production": 1,
            "is_capital": True,
            "has_walls": False,
            "units": [],
        },
        {
            "id": 20,
            "tribe_id": 1,
            "x": 2,
            "y": 1,
            "level": 3,
            "population": 2,
            "population_need": 5,
            "production": 4,
            "is_capital": True,
            "has_walls": False,
            "units": [2],
        },
    ]
    for y in range(0, 3):
        for x in range(1, 4):
            observation["board"]["tiles"][y][x]["city_id"] = 20
    observation["board"]["tiles"][0][0]["terrain"] = "CITY"
    observation["board"]["tiles"][0][0]["city_id"] = 10
    observation["board"]["tiles"][1][2]["terrain"] = "CITY"
    observation["units"] = [
        {
            "id": 1,
            "tribe_id": 0,
            "city_id": 10,
            "type": "CLOAK",
            "x": 1,
            "y": 1,
            "current_hp": 10,
            "max_hp": 10,
            "kills": 0,
            "is_veteran": False,
            "status": "FRESH",
            "is_hidden": False,
            "attack": 2,
            "range": 1,
        },
        {
            "id": 2,
            "tribe_id": 1,
            "city_id": 20,
            "type": "WARRIOR",
            "x": 2,
            "y": 1,
            "current_hp": 10,
            "max_hp": 10,
            "kills": 0,
            "is_veteran": False,
            "status": "FRESH",
            "is_hidden": False,
        },
    ]
    observation["board"]["tiles"][1][1]["unit_id"] = 1
    observation["board"]["tiles"][1][2]["unit_id"] = 2
    message["actions"] = [{"id": "infiltrate", "type": "INFILTRATE", "unit_id": 1, "u": 1, "target_city_id": 20, "tc": 20}]
    return message


def _message_with_village_and_ruin_choices() -> dict:
    message = _message()
    observation = message["observation"]
    observation["tribes"][0]["stars"] = 10
    observation["tribes"][0]["researched_tech_ids"] = ["ROADS"]
    observation["cities"] = [
        {
            "id": 10,
            "tribe_id": 0,
            "x": 0,
            "y": 0,
            "level": 1,
            "population": 0,
            "population_need": 2,
            "production": 1,
            "is_capital": True,
            "has_walls": False,
        }
    ]
    observation["board"]["tiles"][0][0]["terrain"] = "CITY"
    observation["board"]["tiles"][0][0]["city_id"] = 10
    observation["board"]["tiles"][1][2]["terrain"] = "VILLAGE"
    observation["board"]["tiles"][2][1]["resource"] = "RUINS"
    observation["units"] = [
        {
            "id": 1,
            "tribe_id": 0,
            "city_id": 10,
            "type": "WARRIOR",
            "x": 1,
            "y": 1,
            "current_hp": 10,
            "max_hp": 10,
            "kills": 0,
            "is_veteran": False,
            "status": "FRESH",
            "is_hidden": False,
        },
        {
            "id": 2,
            "tribe_id": 0,
            "city_id": 10,
            "type": "WARRIOR",
            "x": 1,
            "y": 2,
            "current_hp": 10,
            "max_hp": 10,
            "kills": 0,
            "is_veteran": False,
            "status": "FRESH",
            "is_hidden": False,
        },
    ]
    observation["board"]["tiles"][1][1]["unit_id"] = 1
    observation["board"]["tiles"][2][1]["unit_id"] = 2
    message["actions"] = [
        {"id": "move_village", "type": "MOVE", "unit_id": 1, "u": 1, "destination": {"x": 2, "y": 1}, "x": 2, "y": 1},
        {"id": "examine", "type": "EXAMINE", "unit_id": 2, "u": 2},
        {"id": "isolated_road", "type": "BUILD_ROAD", "tribe_id": 0, "p": 0, "position": {"x": 3, "y": 3}, "x": 3, "y": 3},
        {"id": "end", "type": "END_TURN"},
    ]
    return message


def _message_with_village_capture() -> dict:
    message = _message()
    observation = message["observation"]
    observation["tribes"][0]["stars"] = 5
    observation["cities"] = [
        {
            "id": 10,
            "tribe_id": 0,
            "x": 0,
            "y": 0,
            "level": 1,
            "population": 0,
            "population_need": 2,
            "production": 1,
            "is_capital": True,
            "has_walls": False,
        }
    ]
    observation["board"]["tiles"][0][0]["terrain"] = "CITY"
    observation["board"]["tiles"][0][0]["city_id"] = 10
    observation["board"]["tiles"][1][1]["terrain"] = "VILLAGE"
    observation["board"]["tiles"][1][1]["unit_id"] = 1
    observation["units"] = [
        {
            "id": 1,
            "tribe_id": 0,
            "city_id": 10,
            "type": "WARRIOR",
            "x": 1,
            "y": 1,
            "current_hp": 10,
            "max_hp": 10,
            "kills": 0,
            "is_veteran": False,
            "status": "FRESH",
            "is_hidden": False,
        }
    ]
    message["actions"] = [
        {"id": "capture_village", "type": "CAPTURE", "unit_id": 1, "u": 1, "target_city_id": -1, "tc": -1, "capture_type": "VILLAGE", "ct": "VILLAGE"}
    ]
    return message


def _message_with_unlabeled_village_capture() -> dict:
    message = _message_with_village_capture()
    message["actions"] = [{"id": "capture_village", "type": "CAPTURE", "unit_id": 1, "u": 1}]
    return message


def _message_with_territory_labeled_village_capture() -> dict:
    message = _message_with_village_capture()
    observation = message["observation"]
    village_tile = observation["board"]["tiles"][1][1]
    village_tile["city_id"] = 30
    message["actions"] = [
        {
            "id": "capture_village",
            "type": "CAPTURE",
            "unit_id": 1,
            "u": 1,
            "target_city_id": 30,
            "tc": 30,
            "capture_type": "VILLAGE",
            "ct": "VILLAGE",
        }
    ]
    return message


def _message_with_embassy_actions() -> dict:
    message = _message()
    observation = message["observation"]
    observation["tribes"] = [
        {"id": 0, "stars": 10, "score": 0, "researched_tech_ids": ["DIPLOMACY"], "cities": [10], "extra_units": [], "met": [1, 2]},
        {"id": 1, "stars": 10, "score": 0, "researched_tech_ids": [], "cities": [20], "extra_units": [], "met": [0]},
        {"id": 2, "stars": 10, "score": 0, "researched_tech_ids": [], "cities": [30], "extra_units": [], "met": [0]},
    ]
    observation["cities"] = [
        {"id": 10, "tribe_id": 0, "x": 0, "y": 0, "level": 1, "population": 0, "population_need": 2, "production": 2, "is_capital": True},
        {"id": 20, "tribe_id": 1, "x": 2, "y": 0, "level": 1, "population": 0, "population_need": 2, "production": 2, "is_capital": True},
        {"id": 30, "tribe_id": 2, "x": 3, "y": 3, "level": 1, "population": 0, "population_need": 2, "production": 2, "is_capital": True},
    ]
    observation["rel"] = [
        ["PEACE", "PEACE", "PEACE"],
        ["PEACE", "PEACE", "PEACE"],
        ["PEACE", "PEACE", "PEACE"],
    ]
    for city in observation["cities"]:
        tile = observation["board"]["tiles"][city["y"]][city["x"]]
        tile["terrain"] = "CITY"
        tile["city_id"] = city["id"]
    message["actions"] = [
        {"id": "embassy_1", "type": "BUILD_EMBASSY", "tribe_id": 0, "p": 0, "target_player_id": 1, "tp": 1},
        {"id": "embassy_2", "type": "BUILD_EMBASSY", "tribe_id": 0, "p": 0, "target_player_id": 2, "tp": 2},
    ]
    return message


def _message_with_upgrade(action_type: str, unit_type: str, stars: int, techs: list[str], kills: int = 0) -> dict:
    message = _message()
    observation = message["observation"]
    observation["tribes"][0]["stars"] = stars
    observation["tribes"][0]["researched_tech_ids"] = techs
    observation["units"] = [
        {
            "id": 1,
            "tribe_id": 0,
            "city_id": 10,
            "type": unit_type,
            "x": 1,
            "y": 1,
            "current_hp": 10,
            "max_hp": 10,
            "kills": kills,
            "is_veteran": False,
            "status": "FRESH",
            "is_hidden": False,
        }
    ]
    observation["cities"][0]["units"] = [1]
    observation["board"]["tiles"][1][1]["unit_id"] = 1
    message["actions"] = [{"id": action_type.lower(), "type": action_type, "unit_id": 1, "u": 1}]
    return message


def _message_with_end_turn(unit_owner: int | None = None, city_owner: int | None = None) -> dict:
    message = _message()
    message["observation"]["tick"] = 1
    message["actions"] = [{"id": "end", "type": "END_TURN"}]
    message["observation"]["tribes"] = [
        {"id": 0, "stars": 0, "score": 0, "researched_tech_ids": [], "cities": [], "extra_units": []},
        {"id": 1, "stars": 0, "score": 0, "researched_tech_ids": [], "cities": [], "extra_units": []},
        {"id": 2, "stars": 0, "score": 0, "researched_tech_ids": [], "cities": [], "extra_units": []},
    ]
    if unit_owner is not None:
        message["observation"]["units"] = [
            {
                "id": 20 + unit_owner,
                "tribe_id": unit_owner,
                "city_id": 0,
                "type": "WARRIOR",
                "x": 2,
                "y": unit_owner,
                "current_hp": 8,
                "max_hp": 10,
                "kills": 0,
                "is_veteran": False,
                "status": "MOVED",
                "is_hidden": False,
            }
        ]
        message["observation"]["board"]["tiles"][unit_owner][2]["unit_id"] = 20 + unit_owner
    if city_owner is not None:
        message["observation"]["cities"] = [
            {
                "id": 30 + city_owner,
                "tribe_id": city_owner,
                "x": 1,
                "y": city_owner,
                "level": 1,
                "population": 0,
                "population_need": 2,
                "production": 2,
                "is_capital": True,
                "has_walls": False,
            }
        ]
        message["observation"]["board"]["tiles"][city_owner][1]["city_id"] = 30 + city_owner
    return message


def _message_with_two_visible_enemies() -> dict:
    message = _message_with_end_turn()
    message["observation"]["active_player_id"] = 1
    message["observation"]["units"] = [
        {
            "id": 21,
            "tribe_id": 1,
            "city_id": 0,
            "type": "WARRIOR",
            "x": 1,
            "y": 1,
            "current_hp": 10,
            "max_hp": 10,
            "kills": 0,
            "is_veteran": False,
            "status": "MOVED",
            "is_hidden": False,
        },
        {
            "id": 22,
            "tribe_id": 2,
            "city_id": 0,
            "type": "WARRIOR",
            "x": 2,
            "y": 2,
            "current_hp": 10,
            "max_hp": 10,
            "kills": 0,
            "is_veteran": False,
            "status": "MOVED",
            "is_hidden": False,
        },
    ]
    message["observation"]["board"]["tiles"][1][1]["unit_id"] = 21
    message["observation"]["board"]["tiles"][2][2]["unit_id"] = 22
    return message


def _capital_city(city_id: int, owner: int, x: int, y: int) -> dict:
    return {
        "id": city_id,
        "tribe_id": owner,
        "p": owner,
        "x": x,
        "y": y,
        "level": 1,
        "population": 0,
        "population_need": 2,
        "production": 2,
        "is_capital": True,
        "cap": True,
        "has_walls": False,
    }


def _message_with_capital_capture(
    *,
    root_player: int = 0,
    capturer: int = 0,
    target_city_id: int = 20,
    mode: str = "MIGHT",
    extra_capital_owner: int | None = None,
    second_action: dict | None = None,
) -> dict:
    message = _message()
    message["player_id"] = root_player
    observation = message["observation"]
    observation["mode"] = mode
    observation["active_player_id"] = capturer
    tribe_ids = [0, 1] if extra_capital_owner is None else [0, 1, extra_capital_owner]
    observation["tribes"] = [
        {
            "id": tribe_id,
            "stars": 5,
            "score": 0,
            "res": "INCOMPLETE",
            "researched_tech_ids": [],
            "cities": [],
            "extra_units": [],
        }
        for tribe_id in tribe_ids
    ]
    cities = [
        _capital_city(10, 0, 0, 0),
        _capital_city(20, 1, 2, 2),
    ]
    if extra_capital_owner is not None:
        cities.append(_capital_city(30, extra_capital_owner, 3, 3))
    observation["cities"] = cities
    for city in cities:
        tile = observation["board"]["tiles"][city["y"]][city["x"]]
        tile["terrain"] = "CITY"
        tile["city_id"] = city["id"]

    target_city = next(city for city in cities if city["id"] == target_city_id)
    observation["units"] = [
        {
            "id": 7,
            "tribe_id": capturer,
            "p": capturer,
            "city_id": target_city_id,
            "type": "WARRIOR",
            "x": target_city["x"],
            "y": target_city["y"],
            "current_hp": 10,
            "max_hp": 10,
            "kills": 0,
            "is_veteran": False,
            "status": "FRESH",
            "is_hidden": False,
        }
    ]
    observation["board"]["tiles"][target_city["y"]][target_city["x"]]["unit_id"] = 7
    actions = [
        {
            "id": "capture",
            "type": "CAPTURE",
            "unit_id": 7,
            "u": 7,
            "target_city_id": target_city_id,
            "tc": target_city_id,
            "capture_type": "CITY",
            "ct": "CITY",
        }
    ]
    if second_action is not None:
        actions.append(second_action)
    message["actions"] = actions
    return message


class NativeMCTSTest(unittest.TestCase):
    def setUp(self) -> None:
        if load_native_mcts_extension() is None:
            self.skipTest("native MCTS extension is unavailable")

    def test_strict_native_mcts_raises_on_approximate_opponent_turns(self) -> None:
        cfg = HybridAgentConfig()
        cfg.search.num_simulations = 8
        cfg.search.batch_size = 4
        cfg.search.sample_action = False
        cfg.search.dirichlet_epsilon = 0.0
        model = HybridPolicyValueNet(cfg.model).eval()

        res = run_native_mcts(_message(), model, cfg.search, cfg.model, "cpu")
        self.assertIsNotNone(res)

    def test_strict_native_mcts_wall_clock_raises_on_approximate_opponent_turns(self) -> None:
        cfg = HybridAgentConfig()
        cfg.search.num_simulations = 10_000
        cfg.search.batch_size = 2
        cfg.search.sample_action = False
        cfg.search.dirichlet_epsilon = 0.0
        model = HybridPolicyValueNet(cfg.model).eval()

        res = run_native_mcts(_message(), model, cfg.search, cfg.model, "cpu", wall_time_seconds=0.01)
        self.assertIsNotNone(res)

    def test_strict_native_mcts_deterministic_error_on_approximate_opponent_turns(self) -> None:
        cfg = HybridAgentConfig()
        cfg.search.num_simulations = 8
        cfg.search.batch_size = 2
        cfg.search.sample_action = False
        cfg.search.dirichlet_epsilon = 0.0
        cfg.search.seed = 123
        model = HybridPolicyValueNet(cfg.model).eval()

        res = run_native_mcts(_message(), model, cfg.search, cfg.model, "cpu")
        self.assertIsNotNone(res)
        res = run_native_mcts(_message(), model, cfg.search, cfg.model, "cpu")
        self.assertIsNotNone(res)

    def test_cpp_tree_accepts_full_payload_and_serializes_leaf_payload(self) -> None:
        extension = load_native_mcts_extension()
        self.assertIsNotNone(extension)
        message = _message()
        message["observation"]["cities"] = []
        tree = extension.NativeMCTS(message, [1, 2], [0.75, 0.25], 0.1, False, 7, 64)

        with self.assertRaisesRegex(RuntimeError, "unsupported_or_failed_transition:SPAWN"):
            tree.select_leaf(4, 1.5)

    def test_native_spawn_cloak_uses_authoritative_stats(self) -> None:
        extension = load_native_mcts_extension()
        self.assertIsNotNone(extension)
        message = _message()
        message["observation"]["tribes"][0]["researched_tech_ids"] = ["DIPLOMACY"]
        message["actions"] = [
            {"id": "spawn_cloak", "type": "SPAWN", "city_id": 10, "c": 10, "unit_type": "CLOAK", "ut": "CLOAK"}
        ]
        tree = extension.NativeMCTS(message, [0], [1.0], 0.1, False, 7, 64)

        selection = dict(tree.select_leaf(4, 1.5))
        leaf_payload = dict(selection["leaf_payload"])
        unit = next(unit for unit in leaf_payload["observation"]["units"] if unit["type"] == "CLOAK")

        self.assertEqual(unit["max_hp"], 5)
        self.assertEqual(unit["current_hp"], 5)
        self.assertEqual(unit["movement"], 2)

    def test_strict_native_tree_rejects_missing_required_tile_field(self) -> None:
        extension = load_native_mcts_extension()
        self.assertIsNotNone(extension)
        message = _message()
        del message["observation"]["board"]["tiles"][0][0]["city_id"]

        with self.assertRaisesRegex(RuntimeError, "missing city_id/city"):
            extension.NativeMCTS(message, [0], [1.0], 0.1, False, 7, 64)

    def test_strict_native_tree_rejects_unknown_action_type(self) -> None:
        extension = load_native_mcts_extension()
        self.assertIsNotNone(extension)
        message = _message()
        message["actions"] = [{"id": "bogus", "type": "BOGUS"}]
        tree = extension.NativeMCTS(message, [0], [1.0], 0.1, False, 7, 64)

        with self.assertRaisesRegex(RuntimeError, "unsupported_or_failed_transition:BOGUS"):
            tree.select_leaf(4, 1.5)

    def test_strict_native_tree_rejects_invalid_root_action_index(self) -> None:
        extension = load_native_mcts_extension()
        self.assertIsNotNone(extension)

        with self.assertRaisesRegex(IndexError, "Root action index out of range"):
            extension.NativeMCTS(_message(), [99], [1.0], 0.1, False, 7, 64)

    def test_capital_capture_terminal_leaf_has_root_win_value(self) -> None:
        extension = load_native_mcts_extension()
        self.assertIsNotNone(extension)
        for mode in ("MIGHT", "CAPITALS"):
            with self.subTest(mode=mode):
                tree = extension.NativeMCTS(
                    _message_with_capital_capture(mode=mode),
                    [0],
                    [1.0],
                    0.1,
                    False,
                    7,
                    64,
                )

                selection = dict(tree.select_leaf(4, 1.5))

                self.assertTrue(selection["needs_expansion"])
                self.assertTrue(selection["leaf_terminal"])
                self.assertEqual(selection["leaf_value"], 1.0)
                leaf_payload = dict(selection["leaf_payload"])
                self.assertTrue(leaf_payload["is_terminal"])
                self.assertEqual(leaf_payload["winner_id"], 0)
                self.assertEqual(leaf_payload["normalized_terminal_reward"], 1.0)
                self.assertEqual(leaf_payload["native_terminal_reason"], "capital_objective:player_0")

    def test_enemy_capital_capture_terminal_leaf_has_root_loss_value(self) -> None:
        extension = load_native_mcts_extension()
        self.assertIsNotNone(extension)
        tree = extension.NativeMCTS(
            _message_with_capital_capture(capturer=1, target_city_id=10),
            [0],
            [1.0],
            0.1,
            False,
            7,
            64,
        )

        selection = dict(tree.select_leaf(4, 1.5))

        self.assertTrue(selection["needs_expansion"])
        self.assertTrue(selection["leaf_terminal"])
        self.assertEqual(selection["leaf_value"], -1.0)
        leaf_payload = dict(selection["leaf_payload"])
        self.assertTrue(leaf_payload["is_terminal"])
        self.assertEqual(leaf_payload["winner_id"], 1)
        self.assertEqual(leaf_payload["normalized_terminal_reward"], -1.0)
        self.assertEqual(leaf_payload["native_terminal_reason"], "capital_objective:player_1")

    def test_capital_capture_without_all_capitals_regenerates_actions(self) -> None:
        extension = load_native_mcts_extension()
        self.assertIsNotNone(extension)
        tree = extension.NativeMCTS(
            _message_with_capital_capture(extra_capital_owner=2),
            [0],
            [1.0],
            0.1,
            False,
            7,
            64,
        )

        selection = dict(tree.select_leaf(4, 1.5))
        leaf_payload = dict(selection["leaf_payload"])

        self.assertFalse(selection["leaf_terminal"])
        self.assertFalse(leaf_payload["is_terminal"])
        self.assertGreater(len(leaf_payload["actions"]), 0)
        captured_city = next(city for city in leaf_payload["observation"]["cities"] if city["id"] == 20)
        self.assertEqual(captured_city["tribe_id"], 0)
        self.assertEqual(captured_city["p"], 0)
        self.assertEqual(leaf_payload["observation"]["units"][0]["status"], "FINISHED")

    def test_terminal_capital_value_backs_up_as_root_perspective(self) -> None:
        extension = load_native_mcts_extension()
        self.assertIsNotNone(extension)
        message = _message_with_capital_capture(
            capturer=1,
            target_city_id=10,
            second_action={"id": "end", "type": "END_TURN"},
        )
        tree = extension.NativeMCTS(message, [0, 1], [0.95, 0.05], 0.0, False, 7, 64)

        batch = tree.select_leaf_batch_evals_only(1, 4, 1.5)
        self.assertEqual(len(batch), 0)
        selection = dict(tree.select_leaf(4, 1.5))

        self.assertEqual(selection["path_action_indexes"], [1])

    def test_supported_move_regenerates_follow_up_actions(self) -> None:
        extension = load_native_mcts_extension()
        self.assertIsNotNone(extension)
        tree = extension.NativeMCTS(_message_with_unit_move(), [0, 1], [1.0, 0.0], 0.1, False, 7, 64)

        selection = dict(tree.select_leaf(4, 1.5))

        self.assertTrue(selection["needs_expansion"])
        self.assertFalse(selection["leaf_terminal"])
        leaf_payload = dict(selection["leaf_payload"])
        self.assertFalse(leaf_payload["is_terminal"])
        self.assertEqual([action["type"] for action in leaf_payload["actions"]], ["END_TURN"])
        self.assertTrue(str(leaf_payload["actions"][0]["id"]).startswith("sim:p0:t0:"))
        moved_unit = leaf_payload["observation"]["units"][0]
        self.assertEqual((moved_unit["x"], moved_unit["y"]), (2, 1))
        self.assertEqual(moved_unit["status"], "MOVED")
        self.assertEqual(leaf_payload["observation"]["board"]["tiles"][1][1]["unit_id"], 0)
        self.assertEqual(leaf_payload["observation"]["board"]["tiles"][1][2]["unit_id"], 1)

    def test_end_turn_without_visible_enemy_advances_by_live_tribe_order(self) -> None:
        extension = load_native_mcts_extension()
        self.assertIsNotNone(extension)
        tree = extension.NativeMCTS(_message_with_end_turn(), [0], [1.0], 0.1, False, 7, 64)

        selection = dict(tree.select_leaf(4, 1.5))
        self.assertTrue(selection["needs_expansion"])
        self.assertFalse(selection["leaf_terminal"])
        leaf_payload = dict(selection["leaf_payload"])
        self.assertEqual(leaf_payload["active_player_id"], 1)
        self.assertEqual([action["type"] for action in leaf_payload["actions"]], ["END_TURN"])

    def test_end_turn_cycles_back_to_root_when_no_enemy_visible(self) -> None:
        extension = load_native_mcts_extension()
        self.assertIsNotNone(extension)
        first_tree = extension.NativeMCTS(_message_with_end_turn(), [0], [1.0], 0.1, False, 7, 64)
        
        selection = dict(first_tree.select_leaf(4, 1.5))
        self.assertTrue(selection["needs_expansion"])
        leaf_payload = dict(selection["leaf_payload"])
        self.assertEqual(leaf_payload["active_player_id"], 1)
        
        first_tree.expand(selection["parent_node_id"], selection["parent_action_index"], [1.0], 0.0, False)
        
        selection = dict(first_tree.select_leaf(4, 1.5))
        self.assertTrue(selection["needs_expansion"])
        leaf_payload = dict(selection["leaf_payload"])
        self.assertEqual(leaf_payload["active_player_id"], 2)
        
        first_tree.expand(selection["parent_node_id"], selection["parent_action_index"], [1.0], 0.0, False)
        
        selection = dict(first_tree.select_leaf(4, 1.5))
        self.assertTrue(selection["needs_expansion"])
        leaf_payload = dict(selection["leaf_payload"])
        self.assertEqual(leaf_payload["active_player_id"], 0)

    def test_native_attack_transition_is_non_terminal(self) -> None:
        extension = load_native_mcts_extension()
        self.assertIsNotNone(extension)
        message = _message_with_unit_move()
        message["observation"]["units"].append(
            {
                "id": 2,
                "tribe_id": 1,
                "city_id": 0,
                "type": "WARRIOR",
                "x": 2,
                "y": 1,
                "current_hp": 10,
                "max_hp": 10,
                "kills": 0,
                "is_veteran": False,
                "status": "FRESH",
                "is_hidden": False,
            }
        )
        message["observation"]["board"]["tiles"][1][2]["unit_id"] = 2
        message["actions"] = [{"id": "attack", "type": "ATTACK", "unit_id": 1, "target_unit_id": 2, "tu": 2}]
        tree = extension.NativeMCTS(message, [0], [1.0], 0.1, False, 7, 64)

        selection = dict(tree.select_leaf(4, 1.5))
        leaf_payload = dict(selection["leaf_payload"])

        self.assertFalse(selection["leaf_terminal"])
        self.assertFalse(leaf_payload["is_terminal"])
        target = next(unit for unit in leaf_payload["observation"]["units"] if unit["id"] == 2)
        self.assertLess(target["current_hp"], 10)

    def test_native_attack_applies_city_wall_defence_bonus(self) -> None:
        extension = load_native_mcts_extension()
        self.assertIsNotNone(extension)
        message = _message()
        message["observation"]["cities"] = [
            {"id": 10, "tribe_id": 0, "x": 0, "y": 0, "level": 1, "population": 0, "population_need": 2, "production": 1, "is_capital": True, "has_walls": False},
            {"id": 20, "tribe_id": 1, "x": 2, "y": 1, "level": 3, "population": 2, "population_need": 5, "production": 4, "is_capital": True, "has_walls": True},
        ]
        message["observation"]["tribes"][0]["stars"] = 0
        message["observation"]["tribes"][1]["cities"] = [20]
        message["observation"]["board"]["tiles"][0][0]["terrain"] = "CITY"
        message["observation"]["board"]["tiles"][0][0]["city_id"] = 10
        message["observation"]["board"]["tiles"][1][2]["terrain"] = "CITY"
        message["observation"]["board"]["tiles"][1][2]["city_id"] = 20
        message["observation"]["units"] = [
            {"id": 1, "tribe_id": 0, "city_id": 10, "type": "WARRIOR", "x": 1, "y": 1, "current_hp": 10, "max_hp": 10, "kills": 0, "is_veteran": False, "status": "FRESH", "is_hidden": False},
            {"id": 2, "tribe_id": 1, "city_id": 20, "type": "WARRIOR", "x": 2, "y": 1, "current_hp": 10, "max_hp": 10, "kills": 0, "is_veteran": False, "status": "FRESH", "is_hidden": False},
        ]
        message["observation"]["board"]["tiles"][1][1]["unit_id"] = 1
        message["observation"]["board"]["tiles"][1][2]["unit_id"] = 2
        message["actions"] = [{"id": "attack", "type": "ATTACK", "unit_id": 1, "u": 1, "target_unit_id": 2, "tu": 2}]
        tree = extension.NativeMCTS(message, [0], [1.0], 0.1, False, 7, 64)

        selection = dict(tree.select_leaf(4, 1.5))
        leaf_payload = dict(selection["leaf_payload"])
        target = next(unit for unit in leaf_payload["observation"]["units"] if unit["id"] == 2)

        self.assertEqual(target["current_hp"], 8)

    def test_native_attack_applies_forest_archery_defence_bonus(self) -> None:
        extension = load_native_mcts_extension()
        self.assertIsNotNone(extension)
        message = _message_with_unit_move()
        message["observation"]["tribes"][1]["researched_tech_ids"] = ["ARCHERY"]
        message["observation"]["board"]["tiles"][1][2]["terrain"] = "FOREST"
        message["observation"]["units"].append(
            {
                "id": 2,
                "tribe_id": 1,
                "city_id": 0,
                "type": "WARRIOR",
                "x": 2,
                "y": 1,
                "current_hp": 10,
                "max_hp": 10,
                "kills": 0,
                "is_veteran": False,
                "status": "FRESH",
                "is_hidden": False,
            }
        )
        message["observation"]["board"]["tiles"][1][2]["unit_id"] = 2
        message["actions"] = [{"id": "attack", "type": "ATTACK", "unit_id": 1, "u": 1, "target_unit_id": 2, "tu": 2}]
        tree = extension.NativeMCTS(message, [0], [1.0], 0.1, False, 7, 64)

        selection = dict(tree.select_leaf(4, 1.5))
        leaf_payload = dict(selection["leaf_payload"])
        target = next(unit for unit in leaf_payload["observation"]["units"] if unit["id"] == 2)

        self.assertEqual(target["current_hp"], 6)

    def test_native_attack_applies_mountain_climbing_defence_bonus(self) -> None:
        extension = load_native_mcts_extension()
        self.assertIsNotNone(extension)
        message = _message_with_unit_move()
        message["observation"]["tribes"][1]["researched_tech_ids"] = ["CLIMBING"]
        message["observation"]["board"]["tiles"][1][2]["terrain"] = "MOUNTAIN"
        message["observation"]["units"].append(
            {"id": 2, "tribe_id": 1, "city_id": 0, "type": "WARRIOR", "x": 2, "y": 1, "current_hp": 10, "max_hp": 10, "kills": 0, "is_veteran": False, "status": "FRESH", "is_hidden": False}
        )
        message["observation"]["board"]["tiles"][1][2]["unit_id"] = 2
        message["actions"] = [{"id": "attack", "type": "ATTACK", "unit_id": 1, "u": 1, "target_unit_id": 2, "tu": 2}]
        tree = extension.NativeMCTS(message, [0], [1.0], 0.1, False, 7, 64)

        selection = dict(tree.select_leaf(4, 1.5))
        leaf_payload = dict(selection["leaf_payload"])
        target = next(unit for unit in leaf_payload["observation"]["units"] if unit["id"] == 2)

        self.assertEqual(target["current_hp"], 6)

    def test_native_attack_applies_water_aquatism_defence_bonus(self) -> None:
        extension = load_native_mcts_extension()
        self.assertIsNotNone(extension)
        message = _message_with_upgrade("BUILD_ROAD", "BOMBER", 0, [])
        message["observation"]["tribes"][1]["researched_tech_ids"] = ["AQUATISM"]
        message["observation"]["board"]["tiles"][1][2]["terrain"] = "SHALLOW_WATER"
        message["observation"]["units"].append(
            {"id": 2, "tribe_id": 1, "city_id": 0, "type": "BOMBER", "x": 2, "y": 1, "current_hp": 10, "max_hp": 10, "kills": 0, "is_veteran": False, "status": "FRESH", "is_hidden": False, "attack": 3, "defence": 2, "range": 3, "movement": 2}
        )
        message["observation"]["board"]["tiles"][1][2]["unit_id"] = 2
        message["actions"] = [{"id": "attack", "type": "ATTACK", "unit_id": 1, "u": 1, "target_unit_id": 2, "tu": 2}]
        tree = extension.NativeMCTS(message, [0], [1.0], 0.1, False, 7, 64)

        selection = dict(tree.select_leaf(4, 1.5))
        leaf_payload = dict(selection["leaf_payload"])
        target = next(unit for unit in leaf_payload["observation"]["units"] if unit["id"] == 2)

        self.assertEqual(target["current_hp"], 3)

    def test_native_bomber_attack_applies_splash_damage(self) -> None:
        extension = load_native_mcts_extension()
        self.assertIsNotNone(extension)
        message = _message_with_upgrade("BUILD_ROAD", "BOMBER", 0, [])
        message["observation"]["cities"].append(
            {"id": 20, "tribe_id": 1, "x": 2, "y": 2, "level": 3, "population": 2, "population_need": 5, "production": 4, "is_capital": True, "has_walls": True}
        )
        message["observation"]["tribes"][1]["cities"] = [20]
        message["observation"]["units"].append(
            {"id": 2, "tribe_id": 1, "city_id": 0, "type": "WARRIOR", "x": 2, "y": 1, "current_hp": 10, "max_hp": 10, "kills": 0, "is_veteran": False, "status": "FRESH", "is_hidden": False}
        )
        message["observation"]["units"].append(
            {"id": 3, "tribe_id": 1, "city_id": 20, "type": "WARRIOR", "x": 2, "y": 2, "current_hp": 10, "max_hp": 10, "kills": 0, "is_veteran": False, "status": "FRESH", "is_hidden": False}
        )
        message["observation"]["board"]["tiles"][1][2]["unit_id"] = 2
        message["observation"]["board"]["tiles"][2][2]["terrain"] = "CITY"
        message["observation"]["board"]["tiles"][2][2]["city_id"] = 20
        message["observation"]["board"]["tiles"][2][2]["unit_id"] = 3
        message["actions"] = [{"id": "attack", "type": "ATTACK", "unit_id": 1, "u": 1, "target_unit_id": 2, "tu": 2}]
        tree = extension.NativeMCTS(message, [0], [1.0], 0.1, False, 7, 64)

        selection = dict(tree.select_leaf(4, 1.5))
        leaf_payload = dict(selection["leaf_payload"])
        splash_target = next(unit for unit in leaf_payload["observation"]["units"] if unit["id"] == 3)

        self.assertEqual(splash_target["current_hp"], 8)

    def test_native_lethal_attack_increments_attacker_tribe_kills(self) -> None:
        extension = load_native_mcts_extension()
        self.assertIsNotNone(extension)
        message = _message_with_unit_move()
        message["observation"]["tribes"][0]["kills"] = 0
        message["observation"]["units"][0]["type"] = "KNIGHT"
        message["observation"]["units"][0]["attack"] = 3.5
        message["observation"]["units"][0]["movement"] = 3
        message["observation"]["units"].append(
            {"id": 2, "tribe_id": 1, "city_id": 0, "type": "WARRIOR", "x": 2, "y": 1, "current_hp": 1, "max_hp": 10, "kills": 0, "is_veteran": False, "status": "FRESH", "is_hidden": False}
        )
        message["observation"]["board"]["tiles"][1][2]["unit_id"] = 2
        message["actions"] = [{"id": "attack", "type": "ATTACK", "unit_id": 1, "u": 1, "target_unit_id": 2, "tu": 2}]
        tree = extension.NativeMCTS(message, [0], [1.0], 0.1, False, 7, 64)

        selection = dict(tree.select_leaf(4, 1.5))
        leaf_payload = dict(selection["leaf_payload"])
        attacker_tribe = next(tribe for tribe in leaf_payload["observation"]["tribes"] if tribe["id"] == 0)

        self.assertEqual(attacker_tribe["kills"], 1)

    def test_native_knight_persist_after_kill_sets_attacked_status(self) -> None:
        extension = load_native_mcts_extension()
        self.assertIsNotNone(extension)
        message = _message_with_unit_move()
        message["observation"]["units"][0]["type"] = "KNIGHT"
        message["observation"]["units"][0]["attack"] = 3.5
        message["observation"]["units"][0]["movement"] = 3
        message["observation"]["units"].append(
            {"id": 2, "tribe_id": 1, "city_id": 0, "type": "WARRIOR", "x": 2, "y": 1, "current_hp": 1, "max_hp": 10, "kills": 0, "is_veteran": False, "status": "FRESH", "is_hidden": False}
        )
        message["observation"]["board"]["tiles"][1][2]["unit_id"] = 2
        message["actions"] = [{"id": "attack", "type": "ATTACK", "unit_id": 1, "u": 1, "target_unit_id": 2, "tu": 2}]
        tree = extension.NativeMCTS(message, [0], [1.0], 0.1, False, 7, 64)

        selection = dict(tree.select_leaf(4, 1.5))
        leaf_payload = dict(selection["leaf_payload"])
        knight = next(unit for unit in leaf_payload["observation"]["units"] if unit["id"] == 1)

        self.assertEqual(knight["status"], "ATTACKED")

    def test_native_retaliation_kill_increments_defender_kills(self) -> None:
        extension = load_native_mcts_extension()
        self.assertIsNotNone(extension)
        message = _message_with_unit_move()
        message["observation"]["tribes"][1]["kills"] = 0
        message["observation"]["units"][0]["current_hp"] = 1
        message["observation"]["units"][0]["current_hp_exact"] = 1.0
        message["observation"]["units"].append(
            {"id": 2, "tribe_id": 1, "city_id": 0, "type": "DEFENDER", "x": 2, "y": 1, "current_hp": 15, "max_hp": 15, "kills": 0, "is_veteran": False, "status": "FRESH", "is_hidden": False}
        )
        message["observation"]["board"]["tiles"][1][2]["unit_id"] = 2
        message["actions"] = [{"id": "attack", "type": "ATTACK", "unit_id": 1, "u": 1, "target_unit_id": 2, "tu": 2}]
        tree = extension.NativeMCTS(message, [0], [1.0], 0.1, False, 7, 64)

        selection = dict(tree.select_leaf(4, 1.5))
        leaf_payload = dict(selection["leaf_payload"])
        defender = next(unit for unit in leaf_payload["observation"]["units"] if unit["id"] == 2)
        defender_tribe = next(tribe for tribe in leaf_payload["observation"]["tribes"] if tribe["id"] == 1)

        self.assertEqual(defender["kills"], 1)
        self.assertEqual(defender_tribe["kills"], 0)

    def test_native_infiltrate_transition_spawns_daggers(self) -> None:
        extension = load_native_mcts_extension()
        self.assertIsNotNone(extension)
        tree = extension.NativeMCTS(_message_with_infiltrate(), [0], [1.0], 0.1, False, 7, 64)

        selection = dict(tree.select_leaf(4, 1.5))
        leaf_payload = dict(selection["leaf_payload"])

        self.assertFalse(selection["leaf_terminal"])
        self.assertFalse(leaf_payload["is_terminal"])
        city = next(city for city in leaf_payload["observation"]["cities"] if city["id"] == 20)
        self.assertTrue(city["infiltrated"])
        self.assertTrue(city["inf"])
        attacker = next(tribe for tribe in leaf_payload["observation"]["tribes"] if tribe["id"] == 0)
        self.assertEqual(attacker["stars"], 7)
        defender = next(unit for unit in leaf_payload["observation"]["units"] if unit["id"] == 2)
        self.assertEqual(defender["current_hp"], 8)
        self.assertFalse(any(unit["id"] == 1 for unit in leaf_payload["observation"]["units"]))
        spawned = [unit for unit in leaf_payload["observation"]["units"] if unit["tribe_id"] == 0 and unit["type"] == "DAGGER"]
        self.assertEqual(len(spawned), 3)
        self.assertEqual([(unit["x"], unit["y"]) for unit in spawned], [(1, 0), (1, 2), (2, 0)])
        self.assertTrue(all(unit["defence"] == 2 for unit in spawned))

    def test_native_tree_applies_unit_upgrades(self) -> None:
        extension = load_native_mcts_extension()
        self.assertIsNotNone(extension)
        cases = [
            ("MAKE_VETERAN", "WARRIOR", 0, [], "WARRIOR", True, 0, 0, 3),
            ("UPGRADE_RAMMER", "RAFT", 5, ["RAMMING"], "RAMMER", False, 0, 0, 0),
            ("UPGRADE_SCOUT", "RAFT", 5, ["SAILING"], "SCOUT", False, 0, 0, 0),
            ("UPGRADE_BOMBER", "RAFT", 15, ["NAVIGATION"], "BOMBER", False, 0, 0, 0),
            ("UPGRADE_BOMBER", "SCOUT", 15, ["NAVIGATION"], "BOMBER", False, 0, 0, 0),
        ]
        for action_type, unit_type, stars, techs, expected_type, expected_veteran, expected_stars, expected_score, kills in cases:
            with self.subTest(action_type=action_type, unit_type=unit_type):
                tree = extension.NativeMCTS(
                    _message_with_upgrade(action_type, unit_type, stars, techs, kills=kills),
                    [0],
                    [1.0],
                    0.1,
                    False,
                    7,
                    64,
                )

                selection = dict(tree.select_leaf(4, 1.5))
                leaf_payload = dict(selection["leaf_payload"])
                unit = next(unit for unit in leaf_payload["observation"]["units"] if unit["type"] == expected_type)
                city = next(city for city in leaf_payload["observation"]["cities"] if city["id"] == 10)
                tribe = next(tribe for tribe in leaf_payload["observation"]["tribes"] if tribe["id"] == 0)

                self.assertFalse(selection["leaf_terminal"])
                self.assertEqual(unit["is_veteran"], expected_veteran)
                self.assertEqual(tribe["stars"], expected_stars)
                self.assertEqual(tribe["score"], expected_score)
                self.assertIn(unit["id"], city["units"])

    def test_native_generated_actions_include_unit_upgrades(self) -> None:
        extension = load_native_mcts_extension()
        self.assertIsNotNone(extension)
        message = _message_with_upgrade("BUILD_ROAD", "RAFT", 18, ["ROADS", "RAMMING", "SAILING", "NAVIGATION"])
        message["actions"] = [{"id": "road", "type": "BUILD_ROAD", "tribe_id": 0, "p": 0, "x": 0, "y": 0}]

        tree = extension.NativeMCTS(message, [0], [1.0], 0.1, False, 7, 64)
        selection = dict(tree.select_leaf(4, 1.5))
        leaf_payload = dict(selection["leaf_payload"])

        self.assertIn("UPGRADE_RAMMER", [action["type"] for action in leaf_payload["actions"]])
        self.assertIn("UPGRADE_SCOUT", [action["type"] for action in leaf_payload["actions"]])
        self.assertIn("UPGRADE_BOMBER", [action["type"] for action in leaf_payload["actions"]])

    def test_research_tech_identity_is_encoded(self) -> None:
        cfg = HybridAgentConfig()
        message = _message()
        message["actions"] = [
            {"id": "fish", "type": "RESEARCH_TECH", "tech": "FISHING"},
            {"id": "ride", "type": "RESEARCH_TECH", "tech": "RIDING"},
        ]

        encoded = encode_observation(message, cfg.model)
        tech_start = ACTION_FEATURE_INDEX["tech:CLIMBING"]
        fish_features = encoded.action_features[0, 0, tech_start:]
        ride_features = encoded.action_features[0, 1, tech_start:]

        self.assertEqual(int(fish_features.sum().item()), 1)
        self.assertEqual(int(ride_features.sum().item()), 1)
        self.assertNotEqual(int(fish_features.argmax().item()), int(ride_features.argmax().item()))

    def test_level_up_bonus_identity_is_encoded(self) -> None:
        cfg = HybridAgentConfig()
        message = _message()
        message["actions"] = [
            {"id": "workshop", "type": "LEVEL_UP", "city_id": 10, "bonus": "WORKSHOP"},
            {"id": "explorer", "type": "LEVEL_UP", "city_id": 10, "bonus": "EXPLORER"},
        ]

        encoded = encode_observation(message, cfg.model)
        bonus_start = ACTION_FEATURE_INDEX["level_up_bonus:WORKSHOP"]
        workshop_features = encoded.action_features[0, 0, bonus_start : bonus_start + len(LEVEL_UP_BONUS_TYPES)]
        explorer_features = encoded.action_features[0, 1, bonus_start : bonus_start + len(LEVEL_UP_BONUS_TYPES)]

        self.assertEqual(int(workshop_features.sum().item()), 1)
        self.assertEqual(int(explorer_features.sum().item()), 1)
        self.assertEqual(float(workshop_features[LEVEL_UP_BONUS_TYPES.index("WORKSHOP")]), 1.0)
        self.assertEqual(float(explorer_features[LEVEL_UP_BONUS_TYPES.index("EXPLORER")]), 1.0)

    def test_java_resource_building_relationship_and_status_values_are_encoded(self) -> None:
        cfg = HybridAgentConfig()
        message = _message()
        tile = message["observation"]["board"]["tiles"][0][0]
        tile["resource"] = "LIGHTHOUSE"
        tile["building"] = "FOREST_TEMPLE"
        tile["unit_id"] = 99
        message["observation"]["units"].append(
            {
                "id": 99,
                "tribe_id": 0,
                "type": "RIDER",
                "x": 0,
                "y": 0,
                "current_hp": 10,
                "max_hp": 10,
                "status": "ATTACKED",
            }
        )
        message["observation"]["rel"] = [["PEACE", "TREATY"], ["TREATY", "PEACE"]]
        message["actions"] = [
            {"id": "build", "type": "BUILD", "city_id": 10, "x": 0, "y": 0, "building_type": "FOREST_TEMPLE"},
            {"id": "cancel", "type": "CANCEL_TREATY", "tribe_id": 0, "target_player_id": 1},
        ]

        encoded = encode_observation(message, cfg.model)

        self.assertEqual(float(encoded.board[0, BOARD_FEATURE_INDEX["resource:LIGHTHOUSE"], 0, 0]), 1.0)
        self.assertEqual(float(encoded.board[0, BOARD_FEATURE_INDEX["building:FOREST_TEMPLE"], 0, 0]), 1.0)
        self.assertEqual(
            float(encoded.board[0, BOARD_FEATURE_INDEX["visible_unit_status:ATTACKED"], 0, 0]),
            1.0,
        )
        self.assertEqual(float(encoded.unit_features[0, 0, UNIT_FEATURE_INDEX["status:ATTACKED"]]), 1.0)
        self.assertEqual(float(encoded.action_features[0, 0, ACTION_FEATURE_INDEX["building_type:FOREST_TEMPLE"]]), 1.0)
        self.assertEqual(float(encoded.action_features[0, 1, ACTION_FEATURE_INDEX["target_relationship:TREATY"]]), 1.0)

    def test_native_state_fields_are_encoded_for_model(self) -> None:
        cfg = HybridAgentConfig()
        message = _message_with_village_capture()
        message["observation"]["rel"] = [["PEACE", "WAR"], ["WAR", "PEACE"]]
        message["observation"]["tribes"][0]["researched_tech_ids"] = ["ROADS", "RIDING"]
        message["actions"].append({"id": "peace", "type": "PROPOSE_PEACE", "tribe_id": 0, "target_player_id": 1})

        encoded = encode_observation(message, cfg.model)

        tech_slice = encoded.scalar_features[0, SCALAR_MY_TECH_START : SCALAR_MY_TECH_START + len(TECH_TYPES)]
        self.assertEqual(float(tech_slice[TECH_TYPES.index("ROADS")]), 1.0)
        self.assertEqual(float(tech_slice[TECH_TYPES.index("RIDING")]), 1.0)

        capture_context = encoded.action_features[0, 0, ACTION_NATIVE_CONTEXT_START : ACTION_NATIVE_CONTEXT_START + len(TERRAIN_TYPES)]
        self.assertEqual(float(capture_context[TERRAIN_TYPES.index("VILLAGE")]), 1.0)

        self.assertEqual(float(encoded.action_features[0, 1, ACTION_FEATURE_INDEX["target_relationship:WAR"]]), 1.0)
        self.assertEqual(float(encoded.action_features[0, 1, ACTION_FEATURE_INDEX["pending:propose_peace"]]), 1.0)

    def test_end_turn_with_visible_enemy_unit_switches_to_enemy_actions(self) -> None:
        extension = load_native_mcts_extension()
        self.assertIsNotNone(extension)
        tree = extension.NativeMCTS(_message_with_end_turn(unit_owner=1), [0], [1.0], 0.1, False, 7, 64)

        selection = dict(tree.select_leaf(4, 1.5))
        self.assertTrue(selection["needs_expansion"])
        leaf_payload = dict(selection["leaf_payload"])
        self.assertEqual(leaf_payload["active_player_id"], 1)
        self.assertGreater(len(leaf_payload["actions"]), 1)

    def test_end_turn_with_visible_enemy_city_switches_to_enemy_actions(self) -> None:
        extension = load_native_mcts_extension()
        self.assertIsNotNone(extension)
        tree = extension.NativeMCTS(_message_with_end_turn(city_owner=1), [0], [1.0], 0.1, False, 7, 64)

        selection = dict(tree.select_leaf(4, 1.5))
        self.assertTrue(selection["needs_expansion"])
        leaf_payload = dict(selection["leaf_payload"])
        self.assertEqual(leaf_payload["active_player_id"], 1)
        self.assertGreater(len(leaf_payload["actions"]), 1)

    def test_multiple_visible_enemies_use_cyclic_order(self) -> None:
        extension = load_native_mcts_extension()
        self.assertIsNotNone(extension)
        tree = extension.NativeMCTS(_message_with_two_visible_enemies(), [0], [1.0], 0.1, False, 7, 64)

        selection = dict(tree.select_leaf(4, 1.5))
        self.assertTrue(selection["needs_expansion"])
        leaf_payload = dict(selection["leaf_payload"])
        self.assertEqual(leaf_payload["active_player_id"], 2)

    def test_generated_leaf_payload_encodes_successfully(self) -> None:
        extension = load_native_mcts_extension()
        self.assertIsNotNone(extension)
        tree = extension.NativeMCTS(_message_with_end_turn(unit_owner=1), [0], [1.0], 0.1, False, 7, 64)

        selection = dict(tree.select_leaf(4, 1.5))
        self.assertTrue(selection["needs_expansion"])
        leaf_payload = dict(selection["leaf_payload"])
        cfg = HybridAgentConfig()
        encoded = encode_observation(leaf_payload, cfg.model)
        self.assertIsNotNone(encoded)

    def test_strict_native_mcts_raises_before_approximate_follow_up_distribution(self) -> None:
        cfg = HybridAgentConfig()
        cfg.search.num_simulations = 12
        cfg.search.batch_size = 4
        cfg.search.sample_action = False
        cfg.search.dirichlet_epsilon = 0.0
        cfg.search.seed = 123
        model = HybridPolicyValueNet(cfg.model).eval()

        res = run_native_mcts(_message_with_unit_move(), model, cfg.search, cfg.model, "cpu")
        self.assertIsNotNone(res)

    def test_enemy_leaf_value_backs_up_from_root_perspective(self) -> None:
        extension = load_native_mcts_extension()
        self.assertIsNotNone(extension)
        message = _message_with_end_turn(unit_owner=1)
        message["actions"].append({"id": "road", "type": "BUILD_ROAD", "tribe_id": 0, "x": 0, "y": 0, "position": {"x": 0, "y": 0}})
        tree = extension.NativeMCTS(message, [0, 1], [0.5, 0.5], 0.1, False, 7, 64)

        selection = dict(tree.select_leaf_batch(1, 4, 1.5)[0])
        self.assertTrue(selection["needs_expansion"])
        leaf_payload = dict(selection["leaf_payload"])
        self.assertEqual(leaf_payload["active_player_id"], 1)
        tree.expand(selection["parent_node_id"], selection["parent_action_index"], [1.0] * len(leaf_payload["actions"]), 0.8, False)
        tree.complete_selected_paths([selection["selection_id"]], [0.8])

    def test_end_turn_visit_guard_requires_non_end_exploration(self) -> None:
        cfg = HybridAgentConfig()
        cfg.search.min_non_end_turn_visits = 1
        action_id = _apply_end_turn_visit_guard(
            "end",
            {"end": 1.0, "research": 0.0},
            [{"id": "end", "type": "END_TURN"}, {"id": "research", "type": "RESEARCH_TECH"}],
            cfg.search,
        )

        self.assertEqual(action_id, "research")

    def test_eval_cache_key_distinguishes_same_actions_different_state(self) -> None:
        first = _message()
        second = _message()
        second["observation"]["tribes"][0]["stars"] = 5

        self.assertNotEqual(_message_cache_key(first), _message_cache_key(second))

    def test_eval_cache_key_is_stable_for_dict_order(self) -> None:
        first = {"player_id": 0, "observation": {"tick": 1, "active_player_id": 0}, "actions": [{"id": "a", "type": "END_TURN"}]}
        second = {"actions": [{"type": "END_TURN", "id": "a"}], "observation": {"active_player_id": 0, "tick": 1}, "player_id": 0}

        self.assertEqual(_message_cache_key(first), _message_cache_key(second))

    def test_static_eval_returns_normalized_priors(self) -> None:
        extension = load_native_mcts_extension()
        self.assertIsNotNone(extension)

        evaluation = dict(extension.evaluate_static(_message(), 64))
        priors = [float(value) for value in evaluation["priors"]]

        self.assertEqual(len(priors), 3)
        self.assertAlmostEqual(sum(priors), 1.0, places=6)
        self.assertGreaterEqual(float(evaluation["value"]), -1.0)
        self.assertLessEqual(float(evaluation["value"]), 1.0)

    def test_static_eval_accepts_neutral_observed_city(self) -> None:
        extension = load_native_mcts_extension()
        self.assertIsNotNone(extension)
        message = _message()
        message["observation"]["cities"].append(
            {
                "id": 13,
                "tribe_id": -1,
                "x": 2,
                "y": 2,
                "level": 1,
                "population": 0,
                "population_need": 2,
                "production": 2,
                "is_capital": False,
            }
        )
        message["observation"]["board"]["tiles"][2][2]["terrain"] = "VILLAGE"
        message["observation"]["board"]["tiles"][2][2]["city_id"] = 13

        evaluation = dict(extension.evaluate_static(message, 64))

        self.assertEqual(len(list(evaluation["priors"])), len(message["actions"]))

    def test_root_priors_return_current_profile_contract(self) -> None:
        cfg = HybridAgentConfig()
        cfg.search.top_k_actions = 2
        model = HybridPolicyValueNet(cfg.model).eval()

        action_ids, priors, root_value, root_indexes = _root_priors(
            _message(),
            model,
            cfg.search,
            cfg.model,
            "cpu",
        )

        self.assertEqual(len(action_ids), 2)
        self.assertEqual(len(priors), 2)
        self.assertEqual(len(root_indexes), 2)
        self.assertIsInstance(root_value, float)

    def test_static_eval_prioritizes_capture_over_end_turn(self) -> None:
        extension = load_native_mcts_extension()
        self.assertIsNotNone(extension)
        message = _message_with_capital_capture(second_action={"id": "end", "type": "END_TURN"})

        evaluation = dict(extension.evaluate_static(message, 64))
        priors = [float(value) for value in evaluation["priors"]]

        self.assertGreater(priors[0], priors[1])

    def test_static_eval_values_villages_and_ruins_over_isolated_roads(self) -> None:
        extension = load_native_mcts_extension()
        self.assertIsNotNone(extension)
        message = _message_with_village_and_ruin_choices()

        evaluation = dict(extension.evaluate_static(message, 64))
        priors = [float(value) for value in evaluation["priors"]]

        self.assertGreater(priors[0], priors[2])
        self.assertGreater(priors[1], priors[2])

    def test_static_eval_prefers_parsed_unit_attack_over_type_fallback(self) -> None:
        extension = load_native_mcts_extension()
        self.assertIsNotNone(extension)

        def attack_message(parsed_attack: int) -> dict:
            message = _message()
            observation = message["observation"]
            observation["units"] = [
                {
                    "id": 1,
                    "tribe_id": 0,
                    "city_id": 10,
                    "type": "CLOAK",
                    "x": 1,
                    "y": 1,
                    "current_hp": 10,
                    "max_hp": 10,
                    "kills": 0,
                    "is_veteran": False,
                    "status": "FRESH",
                    "is_hidden": False,
                    "attack": parsed_attack,
                    "defence": 1,
                    "range": 1,
                    "movement": 2,
                },
                {
                    "id": 2,
                    "tribe_id": 1,
                    "city_id": 0,
                    "type": "WARRIOR",
                    "x": 2,
                    "y": 1,
                    "current_hp": 10,
                    "max_hp": 10,
                    "kills": 0,
                    "is_veteran": False,
                    "status": "FRESH",
                    "is_hidden": False,
                    "attack": 2,
                    "defence": 2,
                    "range": 1,
                    "movement": 1,
                },
            ]
            observation["board"]["tiles"][1][1]["unit_id"] = 1
            observation["board"]["tiles"][1][2]["unit_id"] = 2
            message["actions"] = [
                {"id": "attack", "type": "ATTACK", "unit_id": 1, "u": 1, "target_unit_id": 2, "tu": 2},
                {"id": "end", "type": "END_TURN"},
            ]
            return message

        low = dict(extension.evaluate_static(attack_message(0), 64))
        high = dict(extension.evaluate_static(attack_message(6), 64))

        self.assertGreater(float(high["priors"][0]), float(low["priors"][0]))

    def test_hybrid_eval_zero_weights_matches_nn(self) -> None:
        nn_eval = _Evaluation([0.8, 0.2], 0.25)
        static_eval = _Evaluation([0.1, 0.9], -0.75)

        mixed = _mix_evaluation(nn_eval, static_eval, 0.0, 0.0)

        self.assertAlmostEqual(mixed.priors[0], 0.8, places=6)
        self.assertAlmostEqual(mixed.priors[1], 0.2, places=6)
        self.assertAlmostEqual(mixed.value, 0.25, places=6)

    def test_hybrid_eval_static_weights_bias_policy_and_value(self) -> None:
        nn_eval = _Evaluation([0.8, 0.2], 0.25)
        static_eval = _Evaluation([0.1, 0.9], -0.75)

        mixed = _mix_evaluation(nn_eval, static_eval, 1.0, 1.0)

        self.assertAlmostEqual(sum(mixed.priors), 1.0, places=6)
        self.assertLess(mixed.priors[0], nn_eval.priors[0])
        self.assertGreater(mixed.priors[1], nn_eval.priors[1])
        self.assertAlmostEqual(mixed.value, -0.75, places=6)

    def test_native_hybrid_mcts_smoke(self) -> None:
        cfg = HybridAgentConfig()
        cfg.search.num_simulations = 8
        cfg.search.batch_size = 4
        cfg.search.sample_action = False
        cfg.search.dirichlet_epsilon = 0.0
        cfg.search.static_policy_weight = 1.0
        cfg.search.static_value_weight = 1.0
        model = HybridPolicyValueNet(cfg.model).eval()

        res = run_native_hybrid_mcts(_message(), model, cfg.search, cfg.model, "cpu")

        self.assertIn(res.action_id, {"end", "spawn", "road"})
        self.assertAlmostEqual(sum(res.visit_target), 1.0, places=6)

    def test_native_tree_simulates_village_capture(self) -> None:
        extension = load_native_mcts_extension()
        self.assertIsNotNone(extension)
        for message in (
            _message_with_village_capture(),
            _message_with_unlabeled_village_capture(),
            _message_with_territory_labeled_village_capture(),
        ):
            with self.subTest(action=message["actions"][0]):
                tree = extension.NativeMCTS(message, [0], [1.0], 0.1, False, 7, 64)

                raw_selection = tree.select_leaf_batch_evals_only(1, 4, 1.0)[0]
                leaf_payload = dict(raw_selection[-1])
                cities = list(leaf_payload["observation"]["cities"])

                self.assertEqual(leaf_payload.get("native_terminal_reason", ""), "")
                self.assertEqual(len(cities), 2)
                self.assertTrue(any(city["x"] == 1 and city["y"] == 1 and city["tribe_id"] == 0 for city in cities))

    def test_native_tree_uses_explicit_build_embassy_targets(self) -> None:
        extension = load_native_mcts_extension()
        self.assertIsNotNone(extension)
        cases = [
            (0, [1.0, 0.0], 20),
            (1, [0.0, 1.0], 30),
        ]
        for action_index, priors, target_city_id in cases:
            with self.subTest(action_index=action_index):
                tree = extension.NativeMCTS(_message_with_embassy_actions(), [0, 1], priors, 0.1, False, 7, 64)

                raw_selection = tree.select_leaf_batch_evals_only(1, 4, 1.0)[0]
                leaf_payload = dict(raw_selection[-1])
                city = next(city for city in leaf_payload["observation"]["cities"] if city["id"] == target_city_id)
                tribe = next(tribe for tribe in leaf_payload["observation"]["tribes"] if tribe["id"] == 0)
                buildings = list(city.get("buildings", city.get("b", [])))

                self.assertEqual(raw_selection[2], action_index)
                self.assertEqual(tribe["stars"], 5)
                self.assertTrue(any(building.get("type", building.get("t")) == "EMBASSY" and building.get("owner_tribe_id", building.get("owner")) == 0 for building in buildings))

    def test_native_tree_rejects_build_embassy_without_target(self) -> None:
        extension = load_native_mcts_extension()
        self.assertIsNotNone(extension)
        message = _message_with_embassy_actions()
        del message["actions"][0]["target_player_id"]
        del message["actions"][0]["tp"]
        tree = extension.NativeMCTS(message, [0], [1.0], 0.1, False, 7, 64)

        with self.assertRaisesRegex(RuntimeError, "unsupported_or_failed_transition:BUILD_EMBASSY"):
            tree.select_leaf(4, 1.5)

    def test_native_tree_simulates_ruin_examine_without_parity_crash(self) -> None:
        extension = load_native_mcts_extension()
        self.assertIsNotNone(extension)
        tree = extension.NativeMCTS(_message_with_village_and_ruin_choices(), [1], [1.0], 0.1, False, 7, 64)

        raw_selection = tree.select_leaf_batch_evals_only(1, 4, 1.0)[0]
        leaf_payload = dict(raw_selection[-1])
        observation = leaf_payload["observation"]
        tribes = {tribe["id"]: tribe for tribe in observation["tribes"]}
        units = {unit["id"]: unit for unit in observation["units"]}
        ruin_tile = observation["board"]["tiles"][2][1]

        self.assertEqual(leaf_payload.get("native_terminal_reason", ""), "")
        self.assertEqual(tribes[0]["stars"], 20)
        self.assertIsNone(ruin_tile.get("resource"))
        self.assertEqual(units[2]["status"], "FINISHED")

    def test_static_eval_preserves_known_terminal_value(self) -> None:
        extension = load_native_mcts_extension()
        self.assertIsNotNone(extension)
        message = _message()
        message["is_terminal"] = True
        message["normalized_terminal_reward"] = 0.75
        message["actions"] = []

        evaluation = dict(extension.evaluate_static(message, 64))

        self.assertEqual(list(evaluation["priors"]), [])
        self.assertAlmostEqual(float(evaluation["value"]), 0.75)

    def test_native_static_mcts_raises_on_approximate_opponent_turns(self) -> None:
        cfg = HybridAgentConfig()
        cfg.search.num_simulations = 12
        cfg.search.batch_size = 4
        cfg.search.sample_action = False
        cfg.search.dirichlet_epsilon = 0.0
        cfg.search.root_temperature = 1e-6
        cfg.search.seed = 123

        res = run_native_static_mcts(_message(), cfg.search, cfg.model)
        self.assertIsNotNone(res)

    def test_native_static_mcts_bot_protocol_smoke(self) -> None:
        bot_path = Path(__file__).resolve().parents[1] / "bots" / "native_static_mcts_bot.py"
        message = _message_with_capital_capture(second_action={"id": "end", "type": "END_TURN"})
        message["type"] = "action_request"
        completed = subprocess.run(
            [
                sys.executable,
                str(bot_path),
                "--simulations",
                "4",
                "--search-batch-size",
                "2",
                "--deterministic",
            ],
            input=json.dumps(message) + "\n",
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )

        response = json.loads(completed.stdout.strip().splitlines()[-1])

        self.assertIn(response.get("actionId"), {"capture", "end"})

    def test_parity_canonical_state_ignores_action_ids_order_and_territory_noise(self) -> None:
        java_state = _message()
        cpp_state = json.loads(json.dumps(java_state))
        cpp_state["actions"] = list(reversed(cpp_state["actions"]))
        for index, action in enumerate(cpp_state["actions"]):
            action["id"] = f"sim:{index}"
            action["i"] = 100 + index
        cpp_state["observation"]["board"]["tiles"][3][1]["city_id"] = 2
        cpp_state["observation"]["board"]["tiles"][3][1]["territory_city_id"] = 2

        self.assertEqual(_canonical_state(java_state, 0), _canonical_state(cpp_state, 0))

    def test_java_parity_superunit_attack_uses_authoritative_stats(self) -> None:
        status = run_parity(
            parse_args(
                [
                    "--fixture",
                    "unitstats:superunit-attack",
                    "--depth",
                    "1",
                    "--max-states",
                    "1",
                    "--max-actions-per-state",
                    "4",
                    "--max-actions",
                    "64",
                ]
            )
        )

        self.assertEqual(status, 0)

    def test_java_parity_milestone4_end_turn_hidden_enemy_city_actions(self) -> None:
        status = run_parity(
            parse_args(
                [
                    "--fixture",
                    "milestone4:units",
                    "--depth",
                    "1",
                    "--max-states",
                    "1",
                    "--max-actions-per-state",
                    "16",
                    "--action-id",
                    "A8",
                    "--max-actions",
                    "256",
                ]
            )
        )

        self.assertEqual(status, 0)

    def test_java_parity_milestone4_research_action_regeneration(self) -> None:
        status = run_parity(
            parse_args(
                [
                    "--fixture",
                    "milestone4:units",
                    "--depth",
                    "2",
                    "--max-states",
                    "12",
                    "--max-actions-per-state",
                    "12",
                    "--action-id",
                    "s4_A4",
                    "--max-actions",
                    "256",
                ]
            )
        )

        self.assertEqual(status, 0)

    def test_java_parity_milestone4_research_preserves_hidden_city_center(self) -> None:
        status = run_parity(
            parse_args(
                [
                    "--fixture",
                    "milestone4:units",
                    "--depth",
                    "2",
                    "--max-states",
                    "12",
                    "--max-actions-per-state",
                    "12",
                    "--action-id",
                    "s6_A7",
                    "--max-actions",
                    "256",
                ]
            )
        )

        self.assertEqual(status, 0)

    def test_java_parity_milestone4_hidden_enemy_spawn_masks_unit(self) -> None:
        status = run_parity(
            parse_args(
                [
                    "--fixture",
                    "milestone4:units",
                    "--depth",
                    "2",
                    "--max-states",
                    "12",
                    "--max-actions-per-state",
                    "12",
                    "--action-id",
                    "s9_A5",
                    "--max-actions",
                    "256",
                ]
            )
        )

        self.assertEqual(status, 0)

    def test_java_parity_milestone4_enemy_attack_regenerates_visible_actions(self) -> None:
        status = run_parity(
            parse_args(
                [
                    "--fixture",
                    "milestone4:units",
                    "--depth",
                    "2",
                    "--max-states",
                    "12",
                    "--max-actions-per-state",
                    "12",
                    "--action-id",
                    "s9_A6",
                    "--max-actions",
                    "256",
                ]
            )
        )

        self.assertEqual(status, 0)

    def test_java_parity_milestone4_enemy_move_scores_exploration(self) -> None:
        status = run_parity(
            parse_args(
                [
                    "--fixture",
                    "milestone4:units",
                    "--depth",
                    "2",
                    "--max-states",
                    "12",
                    "--max-actions-per-state",
                    "12",
                    "--action-id",
                    "s9_A8",
                    "--max-actions",
                    "256",
                ]
            )
        )

        self.assertEqual(status, 0)

    def test_java_parity_milestone4_enemy_move_masks_unseen_units(self) -> None:
        status = run_parity(
            parse_args(
                [
                    "--fixture",
                    "milestone4:units",
                    "--depth",
                    "2",
                    "--max-states",
                    "12",
                    "--max-actions-per-state",
                    "12",
                    "--action-id",
                    "s9_A9",
                    "--max-actions",
                    "256",
                ]
            )
        )

        self.assertEqual(status, 0)

    def test_java_parity_milestone3_city_end_turn_regenerates_enemy_actions(self) -> None:
        status = run_parity(
            parse_args(
                [
                    "--fixture",
                    "milestone3:city",
                    "--depth",
                    "2",
                    "--max-states",
                    "12",
                    "--max-actions-per-state",
                    "16",
                    "--action-id",
                    "A8",
                    "--max-actions",
                    "256",
                ]
            )
        )

        self.assertEqual(status, 0)

    def test_java_parity_milestone3_city_destroy_updates_points_worth(self) -> None:
        status = run_parity(
            parse_args(
                [
                    "--fixture",
                    "milestone3:city",
                    "--depth",
                    "2",
                    "--max-states",
                    "12",
                    "--max-actions-per-state",
                    "16",
                    "--action-id",
                    "s10_A12",
                    "--max-actions",
                    "256",
                ]
            )
        )

        self.assertEqual(status, 0)

    def test_java_parity_milestone3_tribe_war_end_turn_regenerates_enemy_actions(self) -> None:
        status = run_parity(
            parse_args(
                [
                    "--fixture",
                    "milestone3:tribe-war",
                    "--depth",
                    "1",
                    "--max-states",
                    "1",
                    "--max-actions-per-state",
                    "16",
                    "--action-id",
                    "A15",
                    "--max-actions",
                    "256",
                ]
            )
        )

        self.assertEqual(status, 0)

    def test_java_parity_milestone3_level_up_hidden_enemy_spawn_unit_id(self) -> None:
        status = run_parity(
            parse_args(
                [
                    "--fixture",
                    "milestone3:level-up",
                    "--depth",
                    "2",
                    "--max-states",
                    "12",
                    "--max-actions-per-state",
                    "16",
                    "--action-id",
                    "s1_A7",
                    "--max-actions",
                    "256",
                ]
            )
        )

        self.assertEqual(status, 0)

    def test_java_parity_milestone3_tribe_war_hidden_enemy_spawn_unit_id(self) -> None:
        status = run_parity(
            parse_args(
                [
                    "--fixture",
                    "milestone3:tribe-war",
                    "--depth",
                    "2",
                    "--max-states",
                    "12",
                    "--max-actions-per-state",
                    "16",
                    "--action-id",
                    "s1_A9",
                    "--max-actions",
                    "256",
                ]
            )
        )

        self.assertEqual(status, 0)


if __name__ == "__main__":
    unittest.main()
