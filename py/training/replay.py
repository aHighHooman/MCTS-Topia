from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List
import os
import random
import time

import torch

from search.config import RewardConfig


@dataclass
class StepRecord:
    observation: Dict[str, Any]
    legal_actions: List[Dict[str, Any]]
    action_index: int
    action_id: str
    visit_target: List[float]
    root_value: float
    reward_delta: float
    player_id: int
    active_player_id: int
    tick: int
    turn_index: int
    turn_step_index: int
    decisive: bool = False
    outcome: Dict[str, Any] | None = None
    value_target: float = 0.0


def record_to_payload(record: StepRecord) -> Dict[str, Any]:
    return {
        "observation": record.observation,
        "legal_actions": record.legal_actions,
        "action_index": record.action_index,
        "action_id": record.action_id,
        "visit_target": record.visit_target,
        "root_value": record.root_value,
        "reward_delta": record.reward_delta,
        "player_id": record.player_id,
        "active_player_id": record.active_player_id,
        "tick": record.tick,
        "turn_index": record.turn_index,
        "turn_step_index": record.turn_step_index,
        "decisive": record.decisive,
        "outcome": record.outcome,
        "value_target": record.value_target,
    }


def record_from_payload(item: Dict[str, Any]) -> StepRecord:
    allowed = StepRecord.__dataclass_fields__.keys()
    payload = {key: value for key, value in item.items() if key in allowed}
    return StepRecord(**payload)


def terminal_reward_for_player(game_over: Dict[str, Any], player_id: int, cfg: RewardConfig) -> float:
    if cfg.use_java_normalized_terminal_reward and "normalized_terminal_reward" in game_over:
        return float(game_over.get("normalized_terminal_reward", 0.0))
    winner_id = game_over.get("winner_id")
    if winner_id is None:
        return cfg.terminal_draw
    return cfg.terminal_win if int(winner_id) == int(player_id) else cfg.terminal_loss


def compute_returns(records: List[StepRecord], terminal_reward: float, cfg: RewardConfig) -> None:
    running = float(terminal_reward)
    for record in reversed(records):
        shaped = float(record.reward_delta) * float(cfg.shaped_reward_weight)
        running = shaped + float(cfg.gamma) * running
        record.value_target = max(-1.0, min(1.0, running))


class ReplayStore:
    def __init__(
        self,
        replay_dir: Path,
        capacity_steps: int = 100_000,
        shard_prefix: str = "replay",
        *,
        load_existing: bool = True,
        cache_records: bool = True,
    ) -> None:
        self.replay_dir = Path(replay_dir)
        self.capacity_steps = max(0, int(capacity_steps))
        self.shard_prefix = shard_prefix
        self.load_existing = bool(load_existing)
        self.cache_records = bool(cache_records)
        self.replay_dir.mkdir(parents=True, exist_ok=True)
        self._shards: list[dict[str, Any]] = []
        self._records: list[StepRecord] = []
        self._records_by_shard: dict[Path, list[StepRecord]] = {}
        self._total_steps = 0
        if self.load_existing:
            self._load_index()

    def shards(self) -> List[Path]:
        if self._shards:
            return [item["path"] for item in self._shards]
        return sorted(self.replay_dir.glob(f"{self.shard_prefix}_*.pt"))

    def _load_shard(self, shard: Path) -> tuple[dict[str, Any], list[StepRecord]]:
        payload = torch.load(shard, map_location="cpu", weights_only=False)
        records = [record_from_payload(item) for item in payload.get("records", [])]
        return payload, records

    def _load_index(self) -> None:
        self._shards.clear()
        self._records.clear()
        self._records_by_shard.clear()
        self._total_steps = 0
        for shard in sorted(self.replay_dir.glob(f"{self.shard_prefix}_*.pt")):
            try:
                payload, records = self._load_shard(shard)
            except Exception:
                continue
            step_count = int(payload.get("step_count", len(records)))
            self._shards.append({
                "path": shard,
                "created_at": float(payload.get("created_at", 0.0)),
                "step_count": step_count,
                "kind": payload.get("kind"),
            })
            if self.cache_records:
                self._records_by_shard[shard] = records
                self._records.extend(records)
            self._total_steps += step_count
        self._enforce_capacity()

    def refresh(self) -> None:
        """Reload the process-local index after another process writes replay shards."""
        if not self.load_existing:
            self._shards.clear()
            self._records.clear()
            self._records_by_shard.clear()
            self._total_steps = 0
            return
        self._load_index()

    def add_episode(self, records: List[StepRecord], *, iteration: int = 0, player_id: int | None = None) -> Path | None:
        if not records:
            return None
        payload = {
            "version": 2,
            "created_at": time.time(),
            "iteration": int(iteration),
            "player_id": player_id,
            "step_count": len(records),
            "records": [record_to_payload(record) for record in records],
        }
        path = self.replay_dir / f"{self.shard_prefix}_{iteration:04d}_{int(time.time() * 1000)}.pt"
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
        if not self.load_existing:
            return path
        try:
            loaded_payload, loaded_records = self._load_shard(path)
        except Exception as exc:
            raise RuntimeError(f"Replay shard was written but could not be loaded for cache validation: {path}") from exc
        step_count = int(loaded_payload.get("step_count", len(loaded_records)))
        self._shards.append({
            "path": path,
            "created_at": float(loaded_payload.get("created_at", payload["created_at"])),
            "step_count": step_count,
            "kind": loaded_payload.get("kind"),
        })
        if self.cache_records:
            self._records_by_shard[path] = loaded_records
            self._records.extend(loaded_records)
        self._total_steps += step_count
        self._enforce_capacity()
        return path

    def load_records(self) -> List[StepRecord]:
        if not self.cache_records:
            records: list[StepRecord] = []
            for shard in self.shards():
                _, shard_records = self._load_shard(shard)
                records.extend(shard_records)
            return records
        return list(self._records)

    def sample(self, count: int) -> List[StepRecord]:
        if count <= 0:
            return []
        if not self.cache_records:
            return self._sample_from_shards(count)
        if not self._records:
            return []
        return random.sample(self._records, k=min(count, len(self._records)))

    def step_count(self) -> int:
        return self._total_steps

    def _enforce_capacity(self) -> None:
        if self.capacity_steps <= 0:
            return
        while len(self._shards) > 1 and self._total_steps > self.capacity_steps:
            meta = self._shards.pop(0)
            shard = meta["path"]
            records = self._records_by_shard.pop(shard, [])
            self._total_steps -= int(meta["step_count"])
            if self.cache_records and records:
                remove_count = len(records)
                self._records = self._records[remove_count:]
            try:
                shard.unlink()
            except FileNotFoundError:
                break

    def _sample_from_shards(self, count: int) -> List[StepRecord]:
        shard_counts = [(meta["path"], int(meta.get("step_count", 0))) for meta in self._shards]
        total_records = sum(max(0, step_count) for _, step_count in shard_counts)
        if total_records <= 0:
            return []

        target_indexes = set(random.sample(range(total_records), k=min(count, total_records)))
        sampled: list[StepRecord] = []
        offset = 0
        for shard, step_count in shard_counts:
            if step_count <= 0:
                continue
            shard_targets = sorted(index - offset for index in target_indexes if offset <= index < offset + step_count)
            offset += step_count
            if not shard_targets:
                continue
            _, records = self._load_shard(shard)
            sampled.extend(records[index] for index in shard_targets if index < len(records))
        return sampled


def visit_target_tensor(records: Iterable[StepRecord], max_actions: int) -> torch.Tensor:
    rows = []
    for record in records:
        row = torch.zeros(max_actions, dtype=torch.float32)
        values = torch.tensor(record.visit_target[:max_actions], dtype=torch.float32)
        if values.numel() and values.sum() > 0:
            row[: values.numel()] = values / values.sum()
        elif record.action_index < max_actions:
            row[record.action_index] = 1.0
        rows.append(row)
    return torch.stack(rows, dim=0) if rows else torch.zeros(0, max_actions)
