from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import pytest

from test_mcts import _message, _message_with_capital_capture, _message_with_end_turn, _message_with_simple_unit_action_reuse
from search.native.cpp_extension import load_native_mcts_extension


ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT / "out" / "native" / "static_mcts_bot.exe"


def _require_exe() -> Path:
    if not EXE.exists():
        pytest.skip("native static MCTS executable is not built; run scripts/build_static_bot.ps1")
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


def _dense_payload(message: dict) -> dict:
    compact = _compact_payload(message)
    obs = compact["obs"]
    terrain_codes = {
        "PLAIN": 0,
        "SHALLOW_WATER": 1,
        "DEEP_WATER": 2,
        "MOUNTAIN": 3,
        "VILLAGE": 4,
        "CITY": 5,
        "FOREST": 6,
        "FOG": 7,
    }
    resource_codes = {
        "FISH": 0,
        "FRUIT": 1,
        "ANIMAL": 2,
        "STARFISH": 3,
        "LIGHTHOUSE": 4,
        "ORE": 5,
        "CROPS": 6,
        "RUINS": 7,
    }
    building_codes = {
        "PORT": 0,
        "MINE": 1,
        "FORGE": 2,
        "FARM": 3,
        "WINDMILL": 4,
        "MARKET": 5,
        "LUMBER_HUT": 6,
        "SAWMILL": 7,
        "TEMPLE": 8,
        "WATER_TEMPLE": 9,
        "FOREST_TEMPLE": 10,
        "MOUNTAIN_TEMPLE": 11,
        "ALTAR_OF_PEACE": 12,
        "EMPERORS_TOMB": 13,
        "EYE_OF_GOD": 14,
        "GATE_OF_POWER": 15,
        "GRAND_BAZAR": 16,
        "PARK_OF_FORTUNE": 17,
        "TOWER_OF_WISDOM": 18,
        "EMBASSY": 19,
    }
    unit_codes = {
        "WARRIOR": 0,
        "RIDER": 1,
        "DEFENDER": 2,
        "SWORDMAN": 3,
        "ARCHER": 4,
        "CATAPULT": 5,
        "KNIGHT": 6,
        "MIND_BENDER": 7,
        "RAFT": 8,
        "SCOUT": 9,
        "BOMBER": 10,
        "SUPERUNIT": 11,
        "CLOAK": 12,
        "DAGGER": 13,
        "RAMMER": 14,
        "JUGGERNAUT": 15,
        "DINGHY": 16,
        "PIRATE": 17,
    }
    status_codes = {
        "FRESH": 0,
        "MOVED": 1,
        "ATTACKED": 2,
        "MOVED_AND_ATTACKED": 3,
        "PUSHED": 4,
        "FINISHED": 5,
    }
    result_codes = {"WIN": 0, "LOSS": 1, "INCOMPLETE": 2, None: None}
    action_codes = {
        "BUILD": 0,
        "BURN_FOREST": 1,
        "CLEAR_FOREST": 2,
        "DESTROY": 3,
        "GROW_FOREST": 4,
        "LEVEL_UP": 5,
        "GATHER": 6,
        "RESOURCE_GATHERING": 6,
        "SPAWN": 7,
        "BUILD_ROAD": 8,
        "BUILD_EMBASSY": 9,
        "END_TURN": 10,
        "RESEARCH": 11,
        "RESEARCH_TECH": 11,
        "PROPOSE_PEACE": 12,
        "ACCEPT_PEACE": 13,
        "PROPOSE_TREATY": 14,
        "ACCEPT_TREATY": 15,
        "CANCEL_TREATY": 16,
        "ATTACK": 17,
        "CAPTURE": 18,
        "CONVERT": 19,
        "DISBAND": 20,
        "EXAMINE": 21,
        "HEAL_OTHERS": 22,
        "INFILTRATE": 23,
        "MAKE_VETERAN": 24,
        "MOVE": 25,
        "RECOVER": 26,
        "UPGRADE_RAMMER": 27,
        "UPGRADE_SCOUT": 28,
        "UPGRADE_BOMBER": 29,
    }

    def code_matrix(matrix: list[list[object]], codes: dict[object, object]) -> list[list[object]]:
        return [[codes.get(value, value) for value in row] for row in matrix]

    board = obs["board"]
    dense_board = [
        board["size"],
        code_matrix(board["terrain"], terrain_codes),
        code_matrix(board["resource"], resource_codes),
        code_matrix(board["building"], building_codes),
        board["city"],
        board["unit"],
        [[1 if value else 0 for value in row] for row in board["exp"]],
        [[1 if value else 0 for value in row] for row in board["road"]],
        [],
    ]
    dense_obs = [
        obs["tick"],
        2,
        obs["active"],
        1 if obs["end"] else 0,
        1 if obs["lvlup"] else 0,
        [[tribe["id"], 0, tribe["stars"], tribe["score"], result_codes[tribe["res"]], tribe["cap"], tribe["tech"]] for tribe in obs["tribes"]],
        [
            [
                city["id"],
                city["p"],
                city["x"],
                city["y"],
                city["lvl"],
                city["pop"],
                city["need"],
                city["prod"],
                1 if city["cap"] else 0,
                1 if city["wall"] else 0,
                0,
                [[building_codes.get(item.get("type")), item.get("x"), item.get("y")] for item in city["b"]],
            ]
            for city in obs["cities"]
        ],
        [
            [
                unit["id"],
                unit["p"],
                unit["c"],
                unit_codes[unit["t"]],
                unit["x"],
                unit["y"],
                unit["hp"],
                unit["mhp"],
                unit["k"],
                1 if unit["v"] else 0,
                status_codes.get(unit["s"], unit["s"]),
                1 if unit["h"] else 0,
                0,
                unit["atk"],
                unit["def"],
                unit["mov"],
                unit["r"],
                0,
            ]
            for unit in obs["units"]
        ],
        dense_board,
        [],
        [],
    ]

    dense_actions = []
    for action in compact["actions"]:
        action_type = action["t"]
        if action_type == "CAPTURE":
            dense_actions.append([action_codes[action_type], action["u"], action["tc"], terrain_codes[action["ct"]]])
        elif action_type == "END_TURN":
            dense_actions.append([action_codes[action_type], action["p"]])
        else:
            dense_actions.append([action_codes[action_type], action.get("u", 0)])

    return {
        "type": "action_request",
        "request_id": 7,
        "player_id": compact["player_id"],
        "time_ms": 5000,
        "game_mode": "MIGHT",
        "observation": dense_obs,
        "actions": dense_actions,
        "forward_model": {"root": "root", "cmd": ["inspect", "step", "step_many", "step_batch", "release"]},
    }


def test_static_mcts_exe_protocol_smoke_without_forward_model() -> None:
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


def test_static_mcts_exe_uses_native_tree_without_forward_model_protocol() -> None:
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


def test_static_mcts_exe_profile_json_includes_root_action_stats() -> None:
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



def test_static_mcts_exe_accepts_java_compact_protocol() -> None:
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


def test_static_mcts_exe_accepts_dense_java_protocol() -> None:
    message = _message_with_capital_capture(second_action={"id": "end", "type": "END_TURN"})
    dense = _dense_payload(message)
    completed = subprocess.run(
        [str(_require_exe()), "--simulations", "0", "--deterministic"],
        input=json.dumps(dense) + "\n",
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )

    response = json.loads(completed.stdout.strip().splitlines()[-1])

    assert response["i"] == 0
    assert response["actionId"] == "A0"
    assert response["rankedActionIndexes"][:2] == [0, 1]


def test_static_mcts_exe_turn_macro_exp_returns_legal_root_action() -> None:
    message = _message_with_capital_capture(second_action={"id": "end", "type": "END_TURN"})
    message["type"] = "action_request"
    completed = subprocess.run(
        [
            str(_require_exe()),
            "--search-mode",
            "turn-macro-exp",
            "--simulations",
            "16",
            "--turn-macro-max-primitives-per-turn",
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
    assert response["_profile"]["search_mode"] == "turn-macro-exp"


def test_static_mcts_exe_turn_macro_exp_applies_inner_continuation() -> None:
    message = _message_with_simple_unit_action_reuse("RECOVER")
    message["type"] = "action_request"
    completed = subprocess.run(
        [
            str(_require_exe()),
            "--search-mode",
            "turn-macro-exp",
            "--simulations",
            "8",
            "--turn-macro-inner-simulations",
            "16",
            "--turn-macro-max-primitives-per-turn",
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

    assert response["actionId"] in {"recover-u1", "recover-u2", "move-u1"}
    assert int(response["_profile"]["max_primitives_per_turn_sample"]) >= 3
    assert int(response["_profile"]["inner_partial_action_regenerations"]) > 0


def test_static_mcts_exe_turn_macro_exp_respects_primitive_limit_during_finalization() -> None:
    message = _message_with_simple_unit_action_reuse("RECOVER")
    message["type"] = "action_request"
    completed = subprocess.run(
        [
            str(_require_exe()),
            "--search-mode",
            "turn-macro-exp",
            "--simulations",
            "4",
            "--turn-macro-inner-simulations",
            "16",
            "--turn-macro-max-primitives-per-turn",
            "1",
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

    assert response["actionId"] in {action["id"] for action in message["actions"]}
    assert int(response["_profile"]["max_primitives_per_turn_sample"]) == 1
    assert all(int(plan["primitives_executed"]) == 1 for plan in response["_profile"]["root_turn_plans"])


def test_static_mcts_exe_turn_macro_inner_confidence_matches_conditional_visit_product() -> None:
    message = _message_with_simple_unit_action_reuse("RECOVER")
    message["type"] = "action_request"
    completed = subprocess.run(
        [
            str(_require_exe()),
            "--search-mode",
            "turn-macro-inner-probe",
            "--turn-macro-inner-simulations",
            "64",
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
    candidates = response["_profile"]["inner_candidates"]

    assert candidates
    for candidate in candidates:
        shares = [float(value) for value in candidate["conditional_visit_shares"]]
        assert shares
        assert all(0.0 < value <= 1.0 for value in shares)
        assert float(candidate["log_confidence"]) == pytest.approx(sum(math.log(value) for value in shares))


def test_static_mcts_exe_turn_macro_inner_respects_primitive_depth_limit() -> None:
    message = _message_with_simple_unit_action_reuse("RECOVER")
    message["type"] = "action_request"
    completed = subprocess.run(
        [
            str(_require_exe()),
            "--search-mode",
            "turn-macro-inner-probe",
            "--turn-macro-inner-simulations",
            "64",
            "--turn-macro-max-primitives-per-turn",
            "1",
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
    candidates = response["_profile"]["inner_candidates"]

    assert candidates
    assert all(len(candidate["plan"]) == 1 for candidate in candidates)
    assert int(response["_profile"]["inner_nodes_expanded"]) == 0


def test_static_mcts_exe_turn_macro_exp_inner_stops_at_turn_boundary() -> None:
    message = _message()
    message["type"] = "action_request"
    message["actions"] = [
        {"id": "end-a", "type": "END_TURN"},
        {"id": "end-b", "type": "END_TURN"},
    ]
    completed = subprocess.run(
        [
            str(_require_exe()),
            "--search-mode",
            "turn-macro-exp",
            "--simulations",
            "1",
            "--turn-macro-inner-simulations",
            "8",
            "--turn-macro-max-primitives-per-turn",
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

    assert response["actionId"] in {"end-a", "end-b"}
    assert int(response["_profile"]["inner_nodes_expanded"]) == 0


def _request_after_first_recover_action(first_action_index: int = 0) -> dict:
    message = _message_with_simple_unit_action_reuse("RECOVER")
    extension = load_native_mcts_extension()
    assert extension is not None
    priors = [0.0, 0.0, 0.0, 0.0]
    priors[first_action_index] = 1.0
    tree = extension.NativeMCTS(message, [0, 1, 2, 3], priors, 0.1, False, 7, 64)
    selection = dict(tree.select_leaf(1.5))
    payload = dict(selection["leaf_payload"])
    for key in ("_native_board_tensor", "_native_explored_tiles", "_native_visible_tiles"):
        payload.pop(key, None)
    payload["type"] = "action_request"
    payload["player_id"] = 0
    for action in payload["actions"]:
        action["id"] = "next-" + str(action["id"])
    return payload


def test_static_mcts_exe_turn_macro_exp_continues_selected_plan_across_requests() -> None:
    first = _message_with_simple_unit_action_reuse("RECOVER")
    first["type"] = "action_request"
    second = _request_after_first_recover_action(2)
    completed = subprocess.run(
        [
            str(_require_exe()),
            "--search-mode",
            "turn-macro-exp",
            "--simulations",
            "8",
            "--turn-macro-inner-simulations",
            "16",
            "--turn-macro-max-primitives-per-turn",
            "8",
            "--deterministic",
            "--profile-json",
            "--seed",
            "13",
        ],
        input="\n".join((json.dumps(first), json.dumps(second), json.dumps({"type": "game_over"}))) + "\n",
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )

    responses = [json.loads(line) for line in completed.stdout.splitlines()]
    assert responses[0]["actionId"] == "recover-u2"
    assert responses[1]["actionId"] == "next-recover-u1"
    assert responses[1]["_profile"]["continuation_event"] == "continued"


def test_static_mcts_exe_turn_macro_exp_replans_after_state_invalidation() -> None:
    first = _message_with_simple_unit_action_reuse("RECOVER")
    first["type"] = "action_request"
    second = _request_after_first_recover_action(2)
    second["observation"]["tribes"][0]["stars"] += 1
    completed = subprocess.run(
        [
            str(_require_exe()),
            "--search-mode",
            "turn-macro-exp",
            "--simulations",
            "8",
            "--turn-macro-inner-simulations",
            "16",
            "--deterministic",
            "--profile-json",
            "--seed",
            "13",
        ],
        input="\n".join((json.dumps(first), json.dumps(second), json.dumps({"type": "game_over"}))) + "\n",
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )

    responses = [json.loads(line) for line in completed.stdout.splitlines()]
    assert responses[1]["_profile"]["continuation_event"] == "replanned:state_mismatch"


def test_static_mcts_exe_turn_macro_exp_exposes_confidence_metadata() -> None:
    message = _message()
    message["type"] = "action_request"
    completed = subprocess.run(
        [
            str(_require_exe()),
            "--search-mode",
            "turn-macro-exp",
            "--simulations",
            "16",
            "--turn-macro-inner-simulations",
            "16",
            "--turn-macro-max-edges-per-node",
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
    profile = response["_profile"]
    plans = profile["root_turn_plans"]

    assert plans
    assert len({plan["diversity_key"] for plan in plans}) == len(plans)
    assert all(plan["selection_reasons"] for plan in plans)
    assert any("new_function" in plan["selection_reasons"] for plan in plans)
    assert {"spawn", "road", "end"}.issubset({plan["first_action_id"] for plan in plans})
    assert all(float(plan["log_confidence"]) <= 0.0 for plan in plans)
    assert sum(float(plan["edge_prior"]) for plan in plans) == pytest.approx(1.0)
    assert all(float(plan["edge_prior"]) >= 0.0 for plan in plans)
    assert int(profile["macro_exp_raw_candidates"]) >= len(plans)
    assert int(profile["macro_exp_visited_candidates"]) > 0
    assert int(profile["macro_exp_value_candidates"]) > 0
    assert int(profile["macro_exp_prior_candidates"]) > 0
    assert "macro_exp_exact_duplicates_removed" in profile
    assert "macro_exp_material_duplicates_removed" in profile


def test_static_mcts_exe_turn_macro_temperature_controls_edge_prior_sharpness() -> None:
    message = _message_with_simple_unit_action_reuse("RECOVER")
    message["type"] = "action_request"
    prior_ranges = []
    for temperature in (0.1, 10.0):
        completed = subprocess.run(
            [
                str(_require_exe()),
                "--search-mode",
                "turn-macro-exp",
                "--simulations",
                "1",
                "--turn-macro-inner-simulations",
                "32",
                "--turn-macro-max-edges-per-node",
                "8",
                "--turn-macro-temperature",
                str(temperature),
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
        priors = [float(plan["edge_prior"]) for plan in response["_profile"]["root_turn_plans"]]
        assert len(priors) >= 2
        assert sum(priors) == pytest.approx(1.0)
        prior_ranges.append(max(priors) - min(priors))

    assert prior_ranges[0] > prior_ranges[1]


def test_static_mcts_exe_turn_macro_exp_keeps_distinct_unit_commitments() -> None:
    message = _message_with_simple_unit_action_reuse("RECOVER")
    message["type"] = "action_request"
    completed = subprocess.run(
        [
            str(_require_exe()),
            "--search-mode",
            "turn-macro-exp",
            "--simulations",
            "8",
            "--turn-macro-inner-simulations",
            "16",
            "--turn-macro-max-edges-per-node",
            "8",
            "--turn-macro-max-primitives-per-turn",
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
    plans = response["_profile"]["root_turn_plans"]
    first_actions = {plan["first_action_id"] for plan in plans}

    assert {"recover-u1", "recover-u2", "move-u1", "end"}.issubset(first_actions)
    assert sum(plan["first_action_id"] == "recover-u2" for plan in plans) >= 2
    assert len({plan["diversity_key"] for plan in plans}) == len(plans)


@pytest.mark.parametrize(
    ("mode", "expected_action"),
    [("maximalist", "capture"), ("root-max", None)],
)
def test_static_mcts_exe_turn_macro_exp_opponent_mode_uses_expected_utility(
    mode: str,
    expected_action: str | None,
) -> None:
    message = _message_with_capital_capture(
        root_player=0,
        capturer=1,
        target_city_id=10,
        second_action={"id": "end", "type": "END_TURN", "tribe_id": 1, "p": 1},
    )
    message["actions"][0]["tribe_id"] = 1
    message["actions"][0]["p"] = 1
    message["type"] = "action_request"
    completed = subprocess.run(
        [
            str(_require_exe()),
            "--search-mode",
            "turn-macro-exp",
            "--simulations",
            "8",
            "--turn-macro-inner-simulations",
            "32",
            "--turn-macro-max-primitives-per-turn",
            "8",
            "--turn-macro-opponent-mode",
            mode,
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

    if expected_action is not None:
        assert response["actionId"] == expected_action
        assert response["rankedActionIds"][0] == expected_action
    assert int(response["_profile"]["inner_searches"]) > 0
    assert int(response["_profile"]["root_turn_edges"]) >= 1
    opponent_leaf_evaluations = int(response["_profile"]["opponent_leaf_evaluations"])
    if mode == "maximalist":
        assert opponent_leaf_evaluations > 0
    else:
        assert opponent_leaf_evaluations == 0


def test_static_mcts_exe_turn_macro_exp_wall_clock_returns_legal_root_action() -> None:
    message = _message_with_capital_capture(second_action={"id": "end", "type": "END_TURN"})
    message["type"] = "action_request"
    completed = subprocess.run(
        [
            str(_require_exe()),
            "--search-mode",
            "turn-macro-exp",
            "--wall-clock-per-action-seconds",
            "0.01",
            "--search-batch-size",
            "64",
            "--turn-macro-inner-simulations",
            "8",
            "--turn-macro-max-primitives-per-turn",
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
    assert response["_profile"]["search_mode"] == "turn-macro-exp"
    assert int(response["_profile"]["inner_simulations"]) >= 8
    assert int(response["_profile"]["deadline_stops"]) >= 1


def test_static_mcts_exe_primitive_wall_clock_single_legal_action_fast_path() -> None:
    message = _message_with_end_turn()
    message["type"] = "action_request"
    completed = subprocess.run(
        [
            str(_require_exe()),
            "--search-mode",
            "primitive",
            "--wall-clock-per-action-seconds",
            "1",
            "--search-batch-size",
            "64",
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

    assert response["actionId"] == "end"
    assert response["i"] == 0
    assert response["_profile"]["search_mode"] == "primitive"
    assert response["_profile"]["single_legal_action_fast_path"] is True


@pytest.mark.parametrize("mode", ["root-adversarial", "root-max"])
def test_static_mcts_exe_accepts_native_opponent_mode(mode: str) -> None:
    message = _message_with_capital_capture(second_action={"id": "end", "type": "END_TURN"})
    message["type"] = "action_request"
    completed = subprocess.run(
        [
            str(_require_exe()),
            "--simulations",
            "8",
            "--deterministic",
            "--native-opponent-mode",
            mode,
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
