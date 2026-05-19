from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


PY_ROOT = Path(__file__).resolve().parents[1]
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from tribes_rl.belief import BELIEF_PLANE_CHANNEL_START, BELIEF_PLANE_NAMES, BeliefTracker
from tribes_rl.bot_agent import HybridRLBot
from tribes_rl.config import HybridAgentConfig
from tribes_rl.encoding import encode_observation
from tribes_rl.model import HybridPolicyValueNet
from tribes_rl.native.mcts import SearchResult


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
                {"id": 2, "tribe_id": 1, "type": "RIDER", "x": 2, "y": 2, "current_hp": 10, "max_hp": 10, "range": 1, "mov": 2},
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
    def test_belief_planes_and_scalars_encode(self) -> None:
        cfg = HybridAgentConfig()
        annotated = BeliefTracker().annotate(_message())

        encoded = encode_observation(annotated, cfg.model)

        self.assertEqual(tuple(encoded.scalar_features.shape), (1, 88))
        for idx, name in enumerate(BELIEF_PLANE_NAMES):
            channel = BELIEF_PLANE_CHANNEL_START + idx
            expected = annotated["observation"]["belief"]["planes"][name]
            self.assertTrue(torch.equal(encoded.board[0, channel], torch.tensor(expected, dtype=torch.float32)))
        self.assertAlmostEqual(float(encoded.scalar_features[0, 18]), annotated["observation"]["belief"]["opponent_scalars"][0])

    def test_missing_belief_encodes_as_zeros_and_priors_do_not_overwrite(self) -> None:
        cfg = HybridAgentConfig()
        encoded = encode_observation(_message(), cfg.model)
        for idx in range(len(BELIEF_PLANE_NAMES)):
            self.assertEqual(float(encoded.board[0, BELIEF_PLANE_CHANNEL_START + idx].sum()), 0.0)
        self.assertEqual(float(encoded.scalar_features[0, 18:].sum()), 0.0)

        model = HybridPolicyValueNet(cfg.model)
        with_priors = model._with_empty_board_priors(encoded.board.clone())
        for idx in range(len(BELIEF_PLANE_NAMES)):
            self.assertEqual(float(with_priors[0, BELIEF_PLANE_CHANNEL_START + idx].sum()), 0.0)


class BeliefBotFlowTest(unittest.TestCase):
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

            with patch("tribes_rl.bot_agent.run_native_mcts", side_effect=fake_mcts):
                bot = HybridRLBot(cfg, Path("missing.pt"), Path(tmp), model=model, device=torch.device("cpu"), native_available=True, warmup=False)
                response = bot.choose_action(_message())

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

            with patch("tribes_rl.bot_agent.run_native_mcts", side_effect=fake_mcts):
                bot = HybridRLBot(cfg, Path("missing.pt"), Path(tmp), model=model, device=torch.device("cpu"), native_available=True, warmup=False)
                bot.choose_action(_message())
                bot.choose_action(_message())
                next_turn = _message()
                next_turn["player_id"] = 1
                next_turn["observation"]["active_player_id"] = 1
                bot.choose_action(next_turn)

        self.assertEqual(len(captured), 3)
        self.assertEqual(captured, [100.0, 100.0, 100.0])


if __name__ == "__main__":
    unittest.main()
