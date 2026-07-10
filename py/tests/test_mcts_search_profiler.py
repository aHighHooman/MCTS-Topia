from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from nn.encoding import EncodedObservation
from profiling.mcts_search import (
    BranchingCollector,
    _action_breadth_rows,
    _branching_key_stats,
    _branching_pressure_rows,
    _load_mcts_search_config,
    _native_static_exe_command,
)
from search.native.mcts import SearchResult


def _payload() -> dict:
    return {
        "player_id": 0,
        "observation": {"board": {"size": 4}, "tick": 0},
        "actions": [
            {"id": "a", "type": "MOVE", "unit_id": 1, "x": 1, "y": 0},
            {"id": "b", "type": "ATTACK", "unit_id": 1, "x": 2, "y": 0},
            {"id": "c", "type": "BUILD_ROAD", "x": 1, "y": 1},
            {"id": "d", "type": "END_TURN"},
        ],
    }


def test_branching_collector_tracks_caps_search_and_action_rows() -> None:
    collector = BranchingCollector()
    payload = _payload()
    collector.add_root(payload, label="case", model_max_actions=3)
    collector.add_result(
        payload,
        SearchResult(
            action_id="b",
            action_index=1,
            visit_distribution={"a": 0.25, "b": 0.75},
            visit_target=[0.25, 0.75, 0.0],
            value=0.1,
        ),
        label="case",
        model_max_actions=3,
    )

    assert collector.root_action_counts == [4]
    assert collector.root_model_capped_counts == [3]
    assert collector.root_searched_counts == [2]
    assert collector.root_model_cap_drops == [1]
    assert collector.root_top_k_drops == [1]
    assert collector.root_dropped_action_types["BUILD_ROAD"] == 1
    assert len(collector.action_rows) == 4
    assert collector.action_rows[1]["selected"] == 1
    assert collector.action_rows[3]["kept_by_model_cap"] == 0

    pressure = {row["scope"]: row for row in _branching_pressure_rows(collector)}
    assert pressure["root raw legal"]["avg"] == "4.00"
    assert pressure["root searched"]["avg"] == "2.00"
    key_stats = _branching_key_stats(collector)
    assert key_stats["model_cap_dropped_avg"] == "1.00"
    assert key_stats["top_k_dropped_avg"] == "1.00"


def test_action_breadth_includes_searched_dropped_and_visit_share() -> None:
    collector = BranchingCollector()
    payload = _payload()
    collector.add_root(payload, label="case", model_max_actions=4)
    collector.add_eval_messages([payload])
    collector.add_result(
        payload,
        SearchResult("a", 0, {"a": 0.5, "d": 0.5}, [0.5, 0.0, 0.0, 0.5], 0.0),
        label="case",
        model_max_actions=4,
    )

    rows = {row["type"]: row for row in _action_breadth_rows(collector, limit=10)}
    assert rows["MOVE"]["searched_share"] == "50.0%"
    assert rows["MOVE"]["visit_share"] == "50.0%"
    assert rows["ATTACK"]["dropped_top_k_share"] == "50.0%"


def test_encoded_batch_policy_pressure_and_transfer_bytes() -> None:
    collector = BranchingCollector()
    encoded_one = EncodedObservation(
        board=torch.zeros(1, 2, 2, 2),
        unit_features=torch.zeros(1, 1, 3),
        unit_mask=torch.ones(1, 1, dtype=torch.bool),
        city_features=torch.zeros(1, 1, 3),
        city_mask=torch.ones(1, 1, dtype=torch.bool),
        action_features=torch.zeros(1, 2, 5),
        action_mask=torch.ones(1, 2, dtype=torch.bool),
        scalar_features=torch.zeros(1, 4),
        action_ids=["a", "b"],
    )
    encoded_two = EncodedObservation(
        board=torch.zeros(1, 2, 2, 2),
        unit_features=torch.zeros(1, 1, 3),
        unit_mask=torch.ones(1, 1, dtype=torch.bool),
        city_features=torch.zeros(1, 1, 3),
        city_mask=torch.ones(1, 1, dtype=torch.bool),
        action_features=torch.zeros(1, 1, 5),
        action_mask=torch.ones(1, 1, dtype=torch.bool),
        scalar_features=torch.zeros(1, 4),
        action_ids=["c"],
    )
    batch = EncodedObservation(
        board=torch.zeros(2, 2, 2, 2),
        unit_features=torch.zeros(2, 1, 3),
        unit_mask=torch.ones(2, 1, dtype=torch.bool),
        city_features=torch.zeros(2, 1, 3),
        city_mask=torch.ones(2, 1, dtype=torch.bool),
        action_features=torch.zeros(2, 2, 5),
        action_mask=torch.ones(2, 2, dtype=torch.bool),
        scalar_features=torch.zeros(2, 4),
        action_ids=[],
    )

    collector.add_encoded_batch([encoded_one, encoded_two], batch)
    assert collector.policy_action_counts == [2, 1]
    assert collector.policy_padded_action_counts == [2, 2]
    assert collector.policy_padding_waste_slots == 1
    assert collector.policy_total_slots == 4
    assert collector.cpu_to_device_bytes > 0


def test_turn_macro_exp_config_and_static_exe_command() -> None:
    cfg_path = Path("py/profiling/configs/mcts_search_turn_macro_exp.json")
    args = _load_mcts_search_config(cfg_path)
    cfg = SimpleNamespace(
        search=SimpleNamespace(num_simulations=64, top_k_actions=32, batch_size=16, seed=7),
        model=SimpleNamespace(max_actions=128),
    )

    command = _native_static_exe_command(Path("out/native/static_mcts_bot.exe"), cfg, args, using_walltime=True)

    assert args.native_static_search_mode == "turn-macro-exp"
    assert command[command.index("--search-mode") + 1] == "turn-macro-exp"
    assert "--turn-macro-max-turn-depth" not in command
    assert command[command.index("--turn-macro-max-primitives-per-turn") + 1] == "0"
    assert command[command.index("--turn-macro-max-edges-per-node") + 1] == "8"
    assert command[command.index("--turn-macro-inner-simulations") + 1] == "128"
    assert command[command.index("--turn-macro-inner-c-puct") + 1] == "1.5"
    assert command[command.index("--turn-macro-greedy-eval-top-k") + 1] == "1"
    assert command[command.index("--turn-macro-opponent-mode") + 1] == "maximalist"
