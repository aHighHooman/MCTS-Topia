from __future__ import annotations

import sys
import tempfile
import unittest
import re
from pathlib import Path
from unittest.mock import patch

import torch


PY_ROOT = Path(__file__).resolve().parents[1]
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from nn.belief import BeliefTracker
from nn.bot_agent import BotOutputClosed, HybridRLBot
from project_paths import game_src_root
from search.config import HybridAgentConfig
from nn.encoding import (
    ACTION_FEATURE_SCHEMA,
    ACTION_TYPES,
    ACTION_TYPE_ALIASES,
    BOARD_FEATURE_INDEX,
    BOARD_SCHEMA,
    BUILDING_TYPES,
    CITY_FEATURE_SCHEMA,
    RELATIONSHIP_TYPES,
    RESOURCE_TYPES,
    SCALAR_FEATURE_INDEX,
    SCALAR_FEATURE_SCHEMA,
    TECH_TYPES,
    TERRAIN_TYPES,
    UNIT_STATUS_TYPES,
    UNIT_TYPES,
    UNIT_FEATURE_SCHEMA,
    encode_observation,
)
from nn.model import HybridPolicyValueNet
from search.native.mcts import SearchResult


class _BrokenStdout:
    def write(self, _value: str) -> int:
        raise OSError(22, "Invalid argument")

    def flush(self) -> None:
        raise OSError(22, "Invalid argument")


def _message() -> dict:
    size = 5
    tiles = [
        [
            {
                "x": x,
                "y": y,
                "explored": not (x >= 3 and y >= 3),
                "terrain": "PLAIN",
                "city_id": 0,
                "unit_id": 0,
            }
            for x in range(size)
        ]
        for y in range(size)
    ]
    tiles[0][0]["city_id"] = 10
    tiles[0][1]["city_id"] = 10
    tiles[4][0]["city_id"] = 20
    tiles[4][1]["city_id"] = 20
    tiles[1][1]["unit_id"] = 1
    tiles[2][2]["unit_id"] = 2
    return {
        "player_id": 0,
        "observation": {
            "active_player_id": 0,
            "tick": 3,
            "board": {"size": size, "tiles": tiles},
            "units": [
                {"id": 1, "tribe_id": 0, "type": "WARRIOR", "x": 1, "y": 1, "current_hp": 10, "max_hp": 10, "hint": True},
                {"id": 2, "tribe_id": 1, "type": "RIDER", "x": 2, "y": 2, "current_hp": 10, "max_hp": 10, "range": 1, "movement": 2},
            ],
            "cities": [
                {"id": 10, "tribe_id": 0, "x": 0, "y": 0, "is_capital": True, "buildings": []},
                {"id": 20, "tribe_id": 1, "x": 0, "y": 4, "is_capital": True, "buildings": [{"type": "MINE", "x": 1, "y": 4}]},
            ],
            "tribes": [
                {"id": 0, "stars": 2, "score": 100, "researched_tech_ids": [], "cities": [10]},
                {"id": 1, "stars": 1, "score": 70, "researched_tech_ids": [], "cities": [20]},
                {"id": 2, "stars": 1, "score": 40, "researched_tech_ids": [], "cities": []},
            ],
        },
        "actions": [{"id": "end", "type": "END_TURN"}],
    }


def _mcts_message() -> dict:
    message = _message()
    message["actions"] = [
        {"id": "end", "type": "END_TURN"},
        {"id": "research", "type": "RESEARCH_TECH", "tech": "RIDING"},
    ]
    return message


class BeliefBuilderTest(unittest.TestCase):
    def test_planes_and_scalars_are_stable(self) -> None:
        tracker = BeliefTracker()
        annotated = tracker.annotate(_message())
        planes = annotated["observation"]["belief"]["planes"]

        self.assertEqual(planes["own_unit_presence"][1][1], 1.0)
        self.assertEqual(planes["enemy_unit_presence"][2][2], 1.0)
        self.assertEqual(planes["own_city_center"][0][0], 1.0)
        self.assertEqual(planes["enemy_city_center"][4][0], 1.0)
        self.assertEqual(planes["own_territory"][0][1], 1.0)
        self.assertEqual(planes["enemy_territory"][4][1], 1.0)
        self.assertEqual(planes["unexplored"][4][4], 1.0)
        self.assertEqual(planes["frontier_unexplored"][3][3], 1.0)
        self.assertEqual(planes["deep_unexplored"][4][4], 1.0)
        self.assertEqual(planes["possible_enemy_capital"][4][0], 1.0)
        self.assertEqual(planes["possible_enemy_capital"][4][4], 0.25)
        self.assertEqual(planes["hidden_cloak_hint_source"][1][1], 1.0)
        self.assertGreater(planes["possible_hidden_cloak"][1][2], 0.0)
        self.assertEqual(planes["known_enemy_threat"][2][2], 1.0)
        self.assertGreater(planes["possible_hidden_cloak_threat"][1][3], 0.0)
        self.assertEqual(planes["known_enemy_city_zone"][4][1], 1.0)

        scalars = annotated["observation"]["belief"]["opponent_scalars"]
        self.assertEqual(len(scalars), 70)
        self.assertAlmostEqual(scalars[0], 0.007)
        self.assertAlmostEqual(scalars[1], 0.003)
        self.assertEqual(scalars[4], 1.0)
        self.assertEqual(scalars[5], 3.0 / 64.0)
        self.assertEqual(scalars[6], 0.0)
        self.assertEqual(scalars[7], 1.0)

    def test_tech_evidence_persists_and_reset_clears_it(self) -> None:
        tracker = BeliefTracker()
        tracker.annotate(_message())

        second = _message()
        second["observation"]["units"] = []
        second["observation"]["cities"][1]["buildings"] = []
        annotated = tracker.annotate(second)
        self.assertEqual(annotated["observation"]["belief"]["opponent_scalars"][5], 3.0 / 64.0)

        tracker.reset()
        annotated = tracker.annotate(second)
        self.assertEqual(annotated["observation"]["belief"]["opponent_scalars"][5], 0.0)

    def test_last_seen_enemy_memory_reset_and_snapshot_safety(self) -> None:
        tracker = BeliefTracker()
        tracker.annotate(_message())

        hidden = _message()
        hidden["observation"]["tick"] = 4
        hidden["observation"]["units"] = [hidden["observation"]["units"][0]]
        hidden["observation"]["board"]["tiles"][2][2]["unit_id"] = 0
        annotated = tracker.annotate(hidden)
        planes = annotated["observation"]["belief"]["planes"]

        self.assertEqual(planes["enemy_unit_presence"][2][2], 0.0)
        self.assertEqual(planes["last_seen_enemy_unit"][2][2], 0.7)
        self.assertGreater(planes["possible_enemy_unit_position"][2][4], 0.0)
        self.assertGreater(planes["possible_enemy_melee_threat"][2][4], 0.0)

        snapshot = tracker.snapshot(0)
        before = {unit_id: unit.copy() for unit_id, unit in tracker.last_seen_enemy_units.items()}
        leaf = _message()
        leaf["observation"]["tick"] = 9
        leaf["observation"]["units"][1]["x"] = 4
        leaf["observation"]["units"][1]["y"] = 4
        snapshot.annotate_without_update(leaf)
        self.assertEqual(
            {unit_id: unit.__dict__ for unit_id, unit in tracker.last_seen_enemy_units.items()},
            {unit_id: unit.__dict__ for unit_id, unit in before.items()},
        )

        tracker.reset()
        self.assertEqual(tracker.last_seen_enemy_units, {})

    def test_snapshot_does_not_mutate_tracker(self) -> None:
        tracker = BeliefTracker()
        tracker.annotate(_message())
        snapshot = tracker.snapshot(0)
        leaf = _message()
        leaf["observation"]["units"].append({"id": 99, "tribe_id": 2, "type": "KNIGHT", "x": 3, "y": 0})

        annotated = snapshot.annotate_without_update(leaf)

        self.assertIn("belief", annotated["observation"])
        self.assertNotIn("CHIVALRY", tracker.tech_evidence.get(2, set()))


class BeliefEncoderTest(unittest.TestCase):
    def test_encoder_type_lists_cover_java_enums(self) -> None:
        types_source = (game_src_root() / "core" / "Types.java").read_text()

        def enum_names(name: str) -> list[str]:
            marker = f"public enum {name}"
            start = types_source.find(marker)
            self.assertGreaterEqual(start, 0, f"Missing enum {name}")
            brace = types_source.find("{", start)
            self.assertGreaterEqual(brace, 0, f"Missing enum body {name}")
            depth = 0
            end = brace
            while end < len(types_source):
                char = types_source[end]
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        break
                end += 1
            body = types_source[brace + 1 : end]
            constants = body.split(";", 1)[0]
            return re.findall(r"^\s*([A-Z][A-Z0-9_]*)\s*(?:\(|,)", constants, re.M)

        self.assertEqual(TERRAIN_TYPES, enum_names("TERRAIN"))
        self.assertEqual(RESOURCE_TYPES, enum_names("RESOURCE"))
        self.assertEqual(BUILDING_TYPES, enum_names("BUILDING"))
        self.assertEqual(UNIT_TYPES, enum_names("UNIT"))
        self.assertEqual(TECH_TYPES, enum_names("TECHNOLOGY"))
        self.assertEqual(RELATIONSHIP_TYPES[:-1], enum_names("RELATIONSHIP"))
        self.assertEqual(UNIT_STATUS_TYPES, enum_names("TURN_STATUS"))
        java_actions = {ACTION_TYPE_ALIASES.get(name, name) for name in enum_names("ACTION")}
        self.assertEqual(java_actions, set(ACTION_TYPES) - {"STEP_MOVE"})

    def test_model_config_dimensions_match_named_schemas(self) -> None:
        cfg = HybridAgentConfig()

        self.assertEqual(cfg.model.board_channels, len(BOARD_SCHEMA))
        self.assertEqual(cfg.model.unit_feature_dim, len(UNIT_FEATURE_SCHEMA))
        self.assertEqual(cfg.model.city_feature_dim, len(CITY_FEATURE_SCHEMA))
        self.assertEqual(cfg.model.action_feature_dim, len(ACTION_FEATURE_SCHEMA))
        self.assertEqual(cfg.model.scalar_dim, len(SCALAR_FEATURE_SCHEMA))

    def test_belief_uncertainty_is_not_encoded_but_visible_tech_evidence_is(self) -> None:
        cfg = HybridAgentConfig()
        annotated = BeliefTracker().annotate(_message())

        encoded = encode_observation(annotated, cfg.model)

        self.assertEqual(tuple(encoded.scalar_features.shape), (1, cfg.model.scalar_dim))
        self.assertLess(max(BOARD_FEATURE_INDEX.values()), cfg.model.board_channels)
        self.assertEqual(float(encoded.board[0, BOARD_FEATURE_INDEX["visible_unit_owner:enemy"], 2, 2]), 1.0)

        for tech in ("RIDING", "MINING", "CLIMBING"):
            index = SCALAR_FEATURE_INDEX[f"known_opponent_tech_evidence:{tech}"]
            self.assertEqual(float(encoded.scalar_features[0, index]), 1.0)
        self.assertEqual(
            float(encoded.scalar_features[0, SCALAR_FEATURE_INDEX["known_opponent_tech_evidence:CHIVALRY"]]),
            0.0,
        )

    def test_belief_payload_changes_do_not_change_nn_tensors(self) -> None:
        cfg = HybridAgentConfig()
        annotated = BeliefTracker().annotate(_message())
        changed = BeliefTracker().annotate(_message())
        changed["observation"]["belief"]["opponent_scalars"] = [1.0] * 70
        for plane in changed["observation"]["belief"]["planes"].values():
            for row in plane:
                for x in range(len(row)):
                    row[x] = 1.0

        model = HybridPolicyValueNet(cfg.model)
        self.assertFalse(model.empty_board_channel_indices)
        left = encode_observation(annotated, cfg.model)
        right = encode_observation(changed, cfg.model)

        self.assertTrue(torch.equal(left.board, right.board))
        self.assertTrue(torch.equal(left.scalar_features, right.scalar_features))

    def test_unseen_opponent_researched_tech_ids_are_not_encoded(self) -> None:
        cfg = HybridAgentConfig()
        message = _message()
        message["observation"]["tribes"][1]["researched_tech_ids"] = ["CHIVALRY", "NAVIGATION"]

        encoded = encode_observation(message, cfg.model)

        for tech in ("CHIVALRY", "NAVIGATION"):
            index = SCALAR_FEATURE_INDEX[f"known_opponent_tech_evidence:{tech}"]
            self.assertEqual(float(encoded.scalar_features[0, index]), 0.0)


class BeliefBotFlowTest(unittest.TestCase):
    def test_emit_raises_protocol_signal_when_stdout_is_closed(self) -> None:
        with patch("nn.bot_agent.sys.stdout", _BrokenStdout()):
            with self.assertRaises(BotOutputClosed):
                HybridRLBot.emit({"actionId": "A1"})

    def test_choose_action_stores_belief_and_passes_snapshot(self) -> None:
        cfg = HybridAgentConfig()
        cfg.search.num_simulations = 1
        model = HybridPolicyValueNet(cfg.model).eval()
        with tempfile.TemporaryDirectory() as tmp:
            captured: dict[str, object] = {}

            def fake_mcts(root_payload, evaluator, search_cfg, model_cfg, device, root_policy_logits=None, root_value=None, belief_snapshot=None, wall_time_seconds=None):
                captured["root_has_belief"] = "belief" in root_payload["observation"]
                captured["leaf_has_belief"] = "belief" in belief_snapshot.annotate_without_update(_message())["observation"]
                return SearchResult("end", 0, {"end": 1.0}, [1.0], 0.0)

            with patch("nn.bot_agent.run_native_mcts", side_effect=fake_mcts):
                bot = HybridRLBot(cfg, Path("missing.pt"), Path(tmp), model=model, device=torch.device("cpu"), native_available=True, warmup=False)
                response = bot.choose_action(_mcts_message())

        self.assertEqual(response["actionId"], "end")
        self.assertTrue(captured["root_has_belief"])
        self.assertTrue(captured["leaf_has_belief"])
        self.assertIn("belief", bot.records[0].observation)

    def test_choose_action_uses_fresh_wall_clock_action_budget(self) -> None:
        cfg = HybridAgentConfig()
        cfg.selfplay.wall_clock_per_action_seconds = 100.0
        model = HybridPolicyValueNet(cfg.model).eval()
        with tempfile.TemporaryDirectory() as tmp:
            captured: list[float | None] = []

            def fake_mcts(root_payload, evaluator, search_cfg, model_cfg, device, root_policy_logits=None, root_value=None, belief_snapshot=None, wall_time_seconds=None):
                captured.append(wall_time_seconds)
                return SearchResult("end", 0, {"end": 1.0}, [1.0], 0.0)

            with patch("nn.bot_agent.run_native_mcts", side_effect=fake_mcts):
                bot = HybridRLBot(cfg, Path("missing.pt"), Path(tmp), model=model, device=torch.device("cpu"), native_available=True, warmup=False)
                bot.choose_action(_mcts_message())
                bot.choose_action(_mcts_message())
                next_turn = _mcts_message()
                next_turn["player_id"] = 1
                next_turn["observation"]["active_player_id"] = 1
                bot.choose_action(next_turn)

        self.assertEqual(len(captured), 3)
        for budget in captured:
            self.assertIsNotNone(budget)
            self.assertGreater(float(budget), 99.0)
            self.assertLessEqual(float(budget), 100.0)


if __name__ == "__main__":
    unittest.main()
