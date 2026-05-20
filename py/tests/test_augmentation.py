from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import torch


PY_ROOT = Path(__file__).resolve().parents[1]
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from training.augment_replay import augment_record, materialize_augmented_replay, symmetry_specs
from nn.augmentation import transform_message_symmetry
from nn.belief import BELIEF_PLANE_NAMES
from search.config import HybridAgentConfig
from nn.encoding import normalize_message
from nn.model import HybridPolicyValueNet
from training.replay import ReplayStore, StepRecord, record_to_payload
from training.symmetry_consistency import evaluate_symmetry_consistency
from training.train import _archive_replay_shards, _training_records_for_iteration, _write_augmented_iteration_shard, collate_batch


def _message() -> dict:
    size = 4
    terrain = [[f"T{y}{x}" for x in range(size)] for y in range(size)]
    terrain[1][0] = "MOUNTAIN"
    resource = [[None for _ in range(size)] for _ in range(size)]
    resource[2][1] = "ORE"
    building = [[None for _ in range(size)] for _ in range(size)]
    building[3][2] = "MINE"
    city = [[0 for _ in range(size)] for _ in range(size)]
    city[0][3] = 10
    unit = [[0 for _ in range(size)] for _ in range(size)]
    unit[1][0] = 7
    exp = [[0 for _ in range(size)] for _ in range(size)]
    exp[2][3] = 1
    road = [[0 for _ in range(size)] for _ in range(size)]
    road[3][1] = 1
    return {
        "player_id": 0,
        "observation": {
            "active_player_id": 0,
            "tick": 3,
            "can_end_turn": True,
            "board": {
                "size": size,
                "terrain": terrain,
                "resource": resource,
                "building": building,
                "city": city,
                "unit": unit,
                "exp": exp,
                "road": road,
                "lighthouses": {"0": {"x": 0, "y": 0, "seen": [0]}, "1": None},
                "tiles": [
                    [
                        {
                            "x": x,
                            "y": y,
                            "explored": True,
                            "terrain": "PLAIN",
                            "resource": None,
                            "building": None,
                            "city_id": city[y][x],
                            "unit_id": unit[y][x],
                            "road": bool(road[y][x]),
                        }
                        for x in range(size)
                    ]
                    for y in range(size)
                ],
            },
            "units": [
                {
                    "id": 7,
                    "tribe_id": 0,
                    "city_id": 10,
                    "type": "WARRIOR",
                    "x": 0,
                    "y": 1,
                    "current_hp": 10,
                    "max_hp": 10,
                    "kills": 0,
                    "is_veteran": False,
                    "status": "FRESH",
                    "is_hidden": False,
                }
            ],
            "cities": [
                {
                    "id": 10,
                    "tribe_id": 0,
                    "x": 3,
                    "y": 0,
                    "level": 1,
                    "population": 1,
                    "population_need": 2,
                    "production": 1,
                    "is_capital": True,
                    "has_walls": False,
                    "points_worth": 5,
                    "buildings": [{"type": "MINE", "x": 2, "y": 3}],
                }
            ],
            "tribes": [
                {"id": 0, "stars": 2, "score": 10, "researched_tech_ids": [], "cities": [10]},
                {"id": 1, "stars": 1, "score": 7, "researched_tech_ids": [], "cities": []},
            ],
            "belief": {
                "version": 1,
                "planes": {
                    name: [[1.0 if (name == "unexplored" and x == 1 and y == 2) else 0.0 for x in range(size)] for y in range(size)]
                    for name in BELIEF_PLANE_NAMES
                },
                "opponent_scalars": [float(i) for i in range(70)],
            },
        },
        "actions": [
            {"id": "move", "type": "MOVE", "unit_id": 7, "destination": {"x": 2, "y": 1}, "x": 2, "y": 1},
            {"id": "road", "type": "BUILD_ROAD", "tribe_id": 0, "position": {"x": 1, "y": 2}, "x": 1, "y": 2},
            {"id": "build", "type": "BUILD", "city_id": 10, "building_type": "MINE", "x": 3, "y": 0},
            {"id": "target", "type": "EXAMINE", "target_pos": {"x": 3, "y": 0}, "x": 3, "y": 0},
        ],
    }


def _record() -> StepRecord:
    message = _message()
    return StepRecord(
        observation=message["observation"],
        legal_actions=message["actions"],
        action_index=1,
        action_id="road",
        visit_target=[0.2, 0.7, 0.1, 0.0],
        root_value=0.25,
        reward_delta=0.0,
        player_id=0,
        active_player_id=0,
        tick=3,
        turn_index=0,
        turn_step_index=0,
        value_target=0.5,
    )


class AugmentationTest(unittest.TestCase):
    def test_rotates_asymmetric_payload_and_actions(self) -> None:
        original = _message()
        original_copy = copy.deepcopy(original)

        rotated = transform_message_symmetry(original, rotation=1, mirror=False)

        self.assertEqual(original, original_copy)
        self.assertEqual(rotated["observation"]["board"]["unit"][0][2], 7)
        self.assertEqual(rotated["observation"]["board"]["city"][3][3], 10)
        self.assertEqual(rotated["observation"]["board"]["road"][1][0], 1)
        self.assertEqual((rotated["observation"]["units"][0]["x"], rotated["observation"]["units"][0]["y"]), (2, 0))
        self.assertEqual((rotated["observation"]["cities"][0]["x"], rotated["observation"]["cities"][0]["y"]), (3, 3))
        self.assertEqual((rotated["observation"]["cities"][0]["buildings"][0]["x"], rotated["observation"]["cities"][0]["buildings"][0]["y"]), (0, 2))
        self.assertEqual([action["id"] for action in rotated["actions"]], ["move", "road", "build", "target"])
        self.assertEqual(rotated["actions"][0]["destination"], {"x": 2, "y": 2})
        self.assertEqual(rotated["actions"][1]["position"], {"x": 1, "y": 1})
        self.assertEqual(rotated["actions"][2]["x"], 3)
        self.assertEqual(rotated["actions"][2]["y"], 3)
        self.assertEqual(rotated["actions"][3]["target_pos"], {"x": 3, "y": 3})
        self.assertEqual(rotated["observation"]["belief"]["planes"]["unexplored"][1][1], 1.0)
        self.assertEqual(rotated["observation"]["belief"]["opponent_scalars"], original["observation"]["belief"]["opponent_scalars"])

    def test_four_rotations_and_double_mirror_are_reversible(self) -> None:
        expected = normalize_message(_message())
        rotated = normalize_message(_message())
        for _ in range(4):
            rotated = transform_message_symmetry(rotated, rotation=1, mirror=False)

        mirrored = transform_message_symmetry(_message(), rotation=0, mirror=True)
        mirrored = transform_message_symmetry(mirrored, rotation=0, mirror=True)

        self.assertEqual(rotated, expected)
        self.assertEqual(mirrored, expected)

    def test_collate_preserves_targets_and_shapes_with_augmentation(self) -> None:
        cfg = HybridAgentConfig()
        cfg.training.augment_symmetries = True
        cfg.training.augmentation_prob = 1.0
        cfg.training.augmentation_seed = 7
        batch = collate_batch([_record()], cfg)

        self.assertEqual(tuple(batch["encoded"].board.shape), (1, cfg.model.board_channels, 4, 4))
        self.assertEqual(tuple(batch["encoded"].action_features.shape), (1, cfg.model.max_actions, cfg.model.action_feature_dim))
        self.assertEqual(len(batch["records"]), 1)
        self.assertTrue(torch.equal(batch["policy_targets"][0, :4], torch.tensor([0.2, 0.7, 0.1, 0.0])))
        self.assertAlmostEqual(float(batch["value_targets"][0]), 0.5)

    def test_collate_respects_zero_augmentation_probability(self) -> None:
        cfg = HybridAgentConfig()
        cfg.training.augment_symmetries = True
        cfg.training.augmentation_prob = 0.0
        batch = collate_batch([_record()], cfg)

        self.assertEqual(tuple(batch["encoded"].board.shape), (1, cfg.model.board_channels, 4, 4))
        self.assertEqual(len(batch["records"]), 1)
        self.assertTrue(torch.equal(batch["policy_targets"][0, :4], torch.tensor([0.2, 0.7, 0.1, 0.0])))

    def test_symmetry_consistency_metric_is_finite(self) -> None:
        cfg = HybridAgentConfig()
        model = HybridPolicyValueNet(cfg.model)

        metrics = evaluate_symmetry_consistency(cfg, model, [_record()], torch.device("cpu"), max_records=1)

        self.assertEqual(metrics["symmetry_groups"], 1.0)
        self.assertEqual(metrics["symmetry_samples"], 8.0)
        for key in ("policy_js", "policy_l1", "value_std", "value_range"):
            self.assertTrue(torch.isfinite(torch.tensor(metrics[key])))
            self.assertGreaterEqual(metrics[key], 0.0)

    def test_offline_augment_record_preserves_targets(self) -> None:
        record = _record()

        augmented = augment_record(record, 1, False)

        self.assertEqual(augmented.visit_target, record.visit_target)
        self.assertEqual(augmented.action_index, record.action_index)
        self.assertEqual(augmented.action_id, record.action_id)
        self.assertEqual(augmented.legal_actions[0]["destination"], {"x": 2, "y": 2})

    def test_materialize_augmented_replay_writes_expected_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            torch.save(
                {
                    "version": 2,
                    "created_at": 1.0,
                    "iteration": 0,
                    "player_id": 0,
                    "step_count": 1,
                    "records": [record_to_payload(_record())],
                },
                input_dir / "replay_0000_test.pt",
            )

            summary = materialize_augmented_replay(
                input_dir,
                output_dir,
                symmetry_set="d4",
                include_identity=False,
                max_records_per_shard=3,
            )
            store = ReplayStore(output_dir, capacity_steps=0, shard_prefix="replay")
            records = store.load_records()

            self.assertEqual(summary["input_records"], 1)
            self.assertEqual(summary["symmetry_count"], 7)
            self.assertEqual(summary["augmented_records"], 7)
            self.assertEqual(len(records), 7)
            self.assertEqual(len(store.shards()), 3)
            self.assertEqual(records[0].visit_target, [0.2, 0.7, 0.1, 0.0])

    def test_iteration_offline_augmentation_writes_8x_records_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            replay_dir = root / "replay"
            cfg = HybridAgentConfig()
            cfg.replay.replay_dir = replay_dir
            store = ReplayStore(replay_dir, capacity_steps=0, shard_prefix="replay")
            raw_shard = store.add_episode([_record()], iteration=1, player_id=0)
            self.assertIsNotNone(raw_shard)

            augmented_shard, records, raw_count = _write_augmented_iteration_shard(store, [raw_shard], cfg, 1)
            payload = torch.load(augmented_shard, map_location="cpu", weights_only=False)

            self.assertEqual(raw_count, 1)
            self.assertEqual(len(records), 8)
            self.assertEqual(payload["kind"], "augmented_replay")
            self.assertEqual(payload["iteration"], 1)
            self.assertEqual(payload["symmetry_set"], "d4")
            self.assertTrue(payload["include_identity"])
            self.assertEqual(payload["step_count"], 8)
            self.assertEqual(payload["source_shards"], [str(raw_shard)])

    def test_training_selection_includes_all_fresh_and_equal_old_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            replay_dir = Path(tmp) / "replay"
            cfg = HybridAgentConfig()
            cfg.replay.replay_dir = replay_dir
            store = ReplayStore(replay_dir, capacity_steps=0, shard_prefix="replay")
            raw_old_shard = store.add_episode([_record()], iteration=0, player_id=None)
            self.assertIsNotNone(raw_old_shard)
            old_shard, _, _ = _write_augmented_iteration_shard(store, [raw_old_shard], cfg, 0)
            current_shard = store.add_episode([_record() for _ in range(4)], iteration=1, player_id=None)
            self.assertIsNotNone(old_shard)
            self.assertIsNotNone(current_shard)
            store.refresh()
            current_records = store._load_shard(current_shard)[1]

            records, old_count = _training_records_for_iteration(store, current_records, [current_shard])

            self.assertEqual(len(current_records), 4)
            self.assertEqual(old_count, 4)
            self.assertEqual(len(records), 8)

    def test_training_selection_samples_old_replay_without_loading_every_old_shard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            replay_dir = Path(tmp) / "replay"
            cfg = HybridAgentConfig()
            cfg.replay.replay_dir = replay_dir
            store = ReplayStore(replay_dir, capacity_steps=0, shard_prefix="replay")
            old_shards = []
            for iteration in range(4):
                raw_shard = store.add_episode([_record()], iteration=iteration, player_id=None)
                old_shard, _, _ = _write_augmented_iteration_shard(store, [raw_shard], cfg, iteration)
                old_shards.append(old_shard)
            current_shard = store.add_episode([_record() for _ in range(2)], iteration=10, player_id=None)
            store.refresh()
            current_records = store._load_shard(current_shard)[1]
            original_load_shard = store._load_shard
            loaded_paths = []

            def counted_load_shard(path: Path):
                loaded_paths.append(path)
                return original_load_shard(path)

            store._load_shard = counted_load_shard  # type: ignore[method-assign]

            records, old_count = _training_records_for_iteration(store, current_records, [current_shard])

            old_loads = [path for path in loaded_paths if path in old_shards]
            self.assertEqual(old_count, 2)
            self.assertEqual(len(records), 4)
            self.assertLessEqual(len(set(old_loads)), 2)

    def test_replay_refresh_enforces_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            replay_dir = Path(tmp) / "replay"
            writer = ReplayStore(replay_dir, capacity_steps=0, shard_prefix="replay")
            for iteration in range(3):
                writer.add_episode([_record(), _record()], iteration=iteration, player_id=None)

            store = ReplayStore(replay_dir, capacity_steps=3, shard_prefix="replay")

            self.assertEqual(len(store.shards()), 1)
            self.assertEqual(len(store.load_records()), 2)
            self.assertEqual(store.step_count(), 2)

    def test_raw_shards_archive_excludes_them_from_main_replay_scan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            replay_dir = Path(tmp) / "replay"
            store = ReplayStore(replay_dir, capacity_steps=0, shard_prefix="replay")
            raw_shard = store.add_episode([_record()], iteration=1, player_id=0)
            self.assertIsNotNone(raw_shard)

            archived = _archive_replay_shards([raw_shard], replay_dir / "raw" / "iter_0001")
            store.refresh()

            self.assertEqual(len(archived), 1)
            self.assertTrue(archived[0].exists())
            self.assertFalse(raw_shard.exists())
            self.assertEqual(store.shards(), [])

    def test_symmetry_specs_counts(self) -> None:
        self.assertEqual(len(symmetry_specs("mirror")), 1)
        self.assertEqual(len(symmetry_specs("rotations")), 3)
        self.assertEqual(len(symmetry_specs("d4")), 7)
        self.assertEqual(len(symmetry_specs("mirror", include_identity=True)), 2)
        self.assertEqual(len(symmetry_specs("rotations", include_identity=True)), 4)
        self.assertEqual(len(symmetry_specs("d4", include_identity=True)), 8)


if __name__ == "__main__":
    unittest.main()
