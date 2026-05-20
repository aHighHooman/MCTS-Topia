from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import re
import shutil
import socket
import subprocess
import statistics
import time
from collections import Counter
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import Dataset

from .augment_replay import augment_record, symmetry_specs
from nn.encoding import EncodedObservation, encode_observation
from nn.model import HybridPolicyValueNet, count_parameters
from search.config import HybridAgentConfig
from search.device import move_optimizer_state, require_cuda_device
from .replay import ReplayStore, StepRecord, record_to_payload, visit_target_tensor
from .selfplay import run_selfplay


class ReplayDataset(Dataset):
    def __init__(self, records: List[StepRecord], cfg: HybridAgentConfig) -> None:
        self.records = records
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> StepRecord:
        return self.records[idx]


def _stack_encoded(items: List[EncodedObservation]) -> EncodedObservation:
    return EncodedObservation(
        board=torch.cat([item.board for item in items], dim=0),
        unit_features=torch.cat([item.unit_features for item in items], dim=0),
        unit_mask=torch.cat([item.unit_mask for item in items], dim=0),
        city_features=torch.cat([item.city_features for item in items], dim=0),
        city_mask=torch.cat([item.city_mask for item in items], dim=0),
        action_features=torch.cat([item.action_features for item in items], dim=0),
        action_mask=torch.cat([item.action_mask for item in items], dim=0),
        scalar_features=torch.cat([item.scalar_features for item in items], dim=0),
        action_ids=[],
    )


def _append_original_encoded(
    encoded_items: list[EncodedObservation],
    target_records: list[StepRecord],
    record: StepRecord,
    cfg: HybridAgentConfig,
) -> None:
    message = {"player_id": record.player_id, "observation": record.observation, "actions": record.legal_actions}
    encoded_items.append(encode_observation(message, cfg.model))
    target_records.append(record)


def collate_batch(records: List[StepRecord], cfg: HybridAgentConfig) -> Dict[str, object]:
    encoded_items: list[EncodedObservation] = []
    target_records: list[StepRecord] = []
    for record in records:
        _append_original_encoded(encoded_items, target_records, record, cfg)
    encoded = _stack_encoded(encoded_items)
    return {
        "encoded": encoded,
        "records": target_records,
        "policy_targets": visit_target_tensor(target_records, cfg.model.max_actions),
        "value_targets": torch.tensor([record.value_target for record in target_records], dtype=torch.float32),
        "root_values": torch.tensor([record.root_value for record in target_records], dtype=torch.float32),
    }


def _load_checkpoint(model: HybridPolicyValueNet, checkpoint_path: Path, optimizer: torch.optim.Optimizer | None = None) -> int:
    if not checkpoint_path.exists():
        return 0
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("model", payload)
    model.load_state_dict(state_dict)
    if optimizer is not None and isinstance(payload, dict) and payload.get("optimizer") is not None:
        try:
            optimizer.load_state_dict(payload["optimizer"])
        except ValueError:
            pass
    try:
        return int(payload.get("iteration", 0))
    except Exception:
        return 0


def _append_metrics(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row_keys = list(row.keys())
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=row_keys)
            writer.writeheader()
            writer.writerow(row)
        return

    with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle, restkey="_extra")
        existing_rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    missing_keys = [key for key in row_keys if key not in fieldnames]
    if missing_keys:
        fieldnames.extend(missing_keys)
        for existing in existing_rows:
            existing.pop("_extra", None)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(existing_rows)
            writer.writerow(row)
        return

    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writerow(row)


def _append_selfplay_game_metrics(path: Path, row: Dict[str, object]) -> None:
    _append_metrics(path, row)


def _format_seconds(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, rem = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{rem:02d}s"
    return f"{minutes}m{rem:02d}s"


def _action_type(record: StepRecord) -> str:
    selected = str(record.action_id)
    action = next((item for item in record.legal_actions if str(item.get("id")) == selected), None)
    if action is None and 0 <= int(record.action_index) < len(record.legal_actions):
        action = record.legal_actions[int(record.action_index)]
    raw = str((action or {}).get("type") or (action or {}).get("t") or "UNKNOWN")
    normalized = raw.strip().upper().replace(" ", "_")
    aliases = {
        "MOVE": "MOVE",
        "STEP_MOVE": "MOVE",
        "ATTACK": "ATTACK",
        "CAPTURE": "CAPTURE",
        "SPAWN": "SPAWN",
        "END_TURN": "END_TURN",
        "RESEARCH_TECH": "RESEARCH",
        "BUILD": "BUILD",
        "RESOURCE_GATHERING": "RESOURCE",
        "LEVEL_UP": "LEVEL_UP",
    }
    return aliases.get(normalized, normalized or "UNKNOWN")


def _summarize_records(records: List[StepRecord]) -> Dict[str, object]:
    action_counts = Counter(_action_type(record) for record in records)
    outcome_counts = Counter()
    seen_episodes: set[tuple[int, int, int]] = set()
    for idx, record in enumerate(records):
        outcome = record.outcome or {}
        if not outcome:
            continue
        key = (int(record.player_id), int(record.tick), idx)
        if key in seen_episodes:
            continue
        seen_episodes.add(key)
        winner_id = outcome.get("winner_id")
        if winner_id is None:
            outcome_counts["draws"] += 1
        elif int(winner_id) == int(record.player_id):
            outcome_counts["wins"] += 1
        else:
            outcome_counts["losses"] += 1
    return {
        "steps": len(records),
        "actions": dict(sorted(action_counts.items())),
        "wins": outcome_counts["wins"],
        "losses": outcome_counts["losses"],
        "draws": outcome_counts["draws"],
    }


def _summarize_replay_shards(replay_store: ReplayStore, shard_paths: List[Path]) -> Dict[str, object]:
    records: list[StepRecord] = []
    wins = losses = draws = 0
    adjudicated_games = adjudication_wins = adjudication_draws = adjudication_failed = 0
    for shard in shard_paths:
        try:
            _, shard_records = replay_store._load_shard(shard)
        except Exception:
            continue
        records.extend(shard_records)
        if not shard_records:
            continue
        player_id = int(shard_records[0].player_id)
        outcome = shard_records[-1].outcome or {}
        if bool(outcome.get("adjudicated")):
            adjudicated_games += 1
            if bool(outcome.get("adjudication_failed")):
                adjudication_failed += 1
        winner_id = outcome.get("winner_id")
        if winner_id is None:
            draws += 1
            if bool(outcome.get("adjudicated")):
                adjudication_draws += 1
        elif int(winner_id) == player_id:
            wins += 1
            if bool(outcome.get("adjudicated")):
                adjudication_wins += 1
        else:
            losses += 1
    summary = _summarize_records(records)
    summary["wins"] = wins
    summary["losses"] = losses
    summary["draws"] = draws
    summary["adjudicated_games"] = adjudicated_games
    summary["adjudication_wins"] = adjudication_wins
    summary["adjudication_draws"] = adjudication_draws
    summary["adjudication_failed"] = adjudication_failed
    return summary


def _format_action_counts(summary: Dict[str, object]) -> str:
    actions = summary.get("actions", {})
    if not isinstance(actions, dict) or not actions:
        return "none"
    preferred = ["ATTACK", "CAPTURE", "MOVE", "SPAWN", "BUILD", "RESEARCH", "RESOURCE", "LEVEL_UP", "END_TURN"]
    parts = [f"{name.lower()}={int(actions[name])}" for name in preferred if name in actions]
    parts.extend(f"{str(name).lower()}={int(value)}" for name, value in actions.items() if name not in preferred)
    return " ".join(parts)


def _match_end_reason(stdout: str, stderr: str = "") -> str:
    for text in (stdout, stderr):
        for line in reversed(text.splitlines()):
            stripped = line.strip()
            if stripped.startswith("Match End Reason:"):
                return stripped.split(":", 1)[1].strip() or "unknown"
            if "forcing draw" in stripped and "total actions" in stripped:
                return "action_limit"
    return "unknown"


def _adjudication_seconds(stdout: str) -> float:
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("Adjudication Sec:"):
            try:
                return float(stripped.split(":", 1)[1].strip())
            except ValueError:
                return 0.0
    return 0.0


def _unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    for index in range(1, 10_000):
        candidate = parent / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find available path for {path}")


def _archive_replay_shards(shards: list[Path], archive_dir: Path) -> list[Path]:
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived: list[Path] = []
    for shard in shards:
        if not shard.exists():
            continue
        destination = _unique_destination(archive_dir / shard.name)
        shutil.move(str(shard), str(destination))
        archived.append(destination)
    return archived


def _write_augmented_iteration_shard(
    replay_store: ReplayStore,
    raw_shards: list[Path],
    cfg: HybridAgentConfig,
    iteration: int,
    *,
    max_return_records: int | None = None,
    max_records_per_shard: int = 10_000,
) -> tuple[Path | None, list[StepRecord], int]:
    raw_records = 0
    returned_records: list[StepRecord] = []
    pending: list[StepRecord] = []
    written_paths: list[Path] = []
    specs = symmetry_specs("d4", include_identity=True)
    max_records = max(1, int(max_records_per_shard))
    return_limit = None if max_return_records is None else max(0, int(max_return_records))
    total_augmented = 0
    shard_index = 0

    def remember(record: StepRecord) -> None:
        nonlocal returned_records
        if return_limit is None or len(returned_records) < return_limit:
            returned_records.append(record)
            return
        if return_limit <= 0:
            return
        # Reservoir sample so the bounded in-memory training slice is not just
        # the opening segment of the iteration.
        candidate = random.randrange(total_augmented)
        if candidate < return_limit:
            returned_records[candidate] = record

    def flush() -> None:
        nonlocal pending, shard_index
        if not pending:
            return
        path = cfg.replay.replay_dir / (
            f"{cfg.replay.shard_prefix}_{iteration:04d}_aug_{shard_index:03d}_{int(time.time() * 1000)}.pt"
        )
        shard_index += 1
        payload = {
            "version": 2,
            "kind": "augmented_replay",
            "created_at": time.time(),
            "iteration": int(iteration),
            "player_id": None,
            "step_count": len(pending),
            "source_shards": [str(shard) for shard in raw_shards],
            "symmetry_set": "d4",
            "include_identity": True,
            "augmentation": {
                "symmetry_set": "d4",
                "include_identity": True,
                "symmetries": [{"rotation": rotation, "mirror": mirror} for rotation, mirror in specs],
            },
            "records": [record_to_payload(record) for record in pending],
        }
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
        written_paths.append(path)
        pending = []

    for shard in raw_shards:
        _, records = replay_store._load_shard(shard)
        raw_records += len(records)
        for rotation, mirror in specs:
            for record in records:
                augmented = augment_record(record, rotation, mirror)
                total_augmented += 1
                remember(augmented)
                pending.append(augmented)
                if len(pending) >= max_records:
                    flush()
    flush()
    return (written_paths[0] if written_paths else None), returned_records, raw_records


def _load_records_from_shards(replay_store: ReplayStore, shards: list[Path]) -> list[StepRecord]:
    records: list[StepRecord] = []
    for shard in shards:
        _, shard_records = replay_store._load_shard(shard)
        records.extend(shard_records)
    return records


def _is_augmented_replay_shard(replay_store: ReplayStore, shard: Path) -> bool:
    for meta in replay_store._shards:
        if meta.get("path") == shard:
            return meta.get("kind") == "augmented_replay"
    try:
        payload, _ = replay_store._load_shard(shard)
    except Exception:
        return False
    return payload.get("kind") == "augmented_replay"


def _training_records_for_iteration(
    replay_store: ReplayStore,
    current_records: list[StepRecord],
    current_shards: list[Path],
    max_records: int | None = None,
    fresh_fraction: float = 0.75,
) -> tuple[list[StepRecord], int]:
    if not current_records:
        if max_records is not None:
            return replay_store.sample(max_records), 0
        return replay_store.load_records(), 0
    current_shard_set = set(current_shards)
    older_shards = [
        shard
        for shard in replay_store.shards()
        if shard not in current_shard_set and _is_augmented_replay_shard(replay_store, shard)
    ]
    if max_records is None or max_records <= 0:
        fresh_records = list(current_records)
        old_count = min(len(fresh_records), sum(_replay_shard_step_count(replay_store, shard) for shard in older_shards))
    else:
        cap = max(1, int(max_records))
        fresh_target = min(len(current_records), max(1, int(cap * max(0.0, min(1.0, fresh_fraction)))))
        old_available = sum(_replay_shard_step_count(replay_store, shard) for shard in older_shards)
        old_count = min(cap - fresh_target, old_available)
        if old_count <= 0:
            fresh_target = min(len(current_records), cap)
            old_count = 0
        fresh_records = random.sample(current_records, k=fresh_target) if fresh_target < len(current_records) else list(current_records)
    mixed_old = _sample_records_from_shards(replay_store, older_shards, old_count)
    return fresh_records + mixed_old, old_count


def _replay_shard_step_count(replay_store: ReplayStore, shard: Path) -> int:
    for meta in replay_store._shards:
        if meta.get("path") == shard:
            return int(meta.get("step_count", 0))
    try:
        payload, records = replay_store._load_shard(shard)
    except Exception:
        return 0
    return int(payload.get("step_count", len(records)))


def _sample_records_from_shards(replay_store: ReplayStore, shards: list[Path], count: int) -> list[StepRecord]:
    if count <= 0 or not shards:
        return []
    shard_counts = [(shard, _replay_shard_step_count(replay_store, shard)) for shard in shards]
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
        _, records = replay_store._load_shard(shard)
        sampled.extend(records[index] for index in shard_targets if index < len(records))
    return sampled


PROFILE_PATTERN = re.compile(
    r"choose_action .*?encode_ms=(?P<encode>[0-9.]+) .*?policy_ms=(?P<policy>[0-9.]+) "
    r".*?search_ms=(?P<search>[0-9.]+) .*?total_ms=(?P<total>[0-9.]+) .*?actions=(?P<actions>[0-9]+)"
)


def _profile_summary(stderr: str) -> Dict[str, float]:
    samples: list[dict[str, float]] = []
    for line in stderr.splitlines():
        match = PROFILE_PATTERN.search(line)
        if not match:
            continue
        samples.append({key: float(value) for key, value in match.groupdict().items()})
    if not samples:
        return {}
    out: Dict[str, float] = {"profile_actions": float(len(samples))}
    for key in ("encode", "policy", "search", "total", "actions"):
        values = [sample[key] for sample in samples]
        out[f"profile_{key}_avg_ms"] = sum(values) / len(values)
        out[f"profile_{key}_median_ms"] = statistics.median(values)
        out[f"profile_{key}_max_ms"] = max(values)
        out[f"profile_{key}_p95_ms"] = sorted(values)[min(len(values) - 1, int(len(values) * 0.95))]
    return out


def _format_profile(summary: Dict[str, float]) -> str:
    if not summary:
        return ""
    return (
        f" profile actions={int(summary['profile_actions'])} "
        f"avg_ms encode={summary['profile_encode_avg_ms']:.1f} "
        f"policy={summary['profile_policy_avg_ms']:.1f} "
        f"search={summary['profile_search_avg_ms']:.1f} "
        f"total={summary['profile_total_avg_ms']:.1f} "
        f"median_total={summary['profile_total_median_ms']:.1f} "
        f"p95_total={summary['profile_total_p95_ms']:.1f} "
        f"max_total={summary['profile_total_max_ms']:.1f}"
    )


def _selfplay_game_row(
    *,
    iteration: int,
    local_iteration: int,
    game_idx: int,
    games: int,
    seed: int,
    elapsed: float,
    summary: Dict[str, object],
    profile_summary: Dict[str, float],
    simulations: int,
    shard_count: int,
    exact_summary: bool,
) -> Dict[str, object]:
    profile_actions = float(profile_summary.get("profile_actions", 0.0))
    steps = float(summary.get("steps", 0.0) or 0.0)
    actions_sec = profile_actions / elapsed if profile_actions > 0.0 and elapsed > 0.0 else steps / elapsed if elapsed > 0.0 else 0.0
    configured_sims_sec = actions_sec * float(simulations) if simulations else 0.0
    actions = summary.get("actions", {})
    action_metrics = {
        f"action_{str(name).lower()}": count
        for name, count in sorted(actions.items())
    } if isinstance(actions, dict) else {}
    return {
        "iteration": iteration,
        "local_iteration": local_iteration + 1,
        "game": game_idx + 1,
        "games_in_iteration": games,
        "seed": seed,
        "steps": int(steps),
        "game_seconds": elapsed,
        "actions_sec": actions_sec,
        "configured_sims_sec": configured_sims_sec,
        "simulations": simulations,
        "profile_actions": profile_actions,
        "wins": int(summary.get("wins", 0) or 0) if exact_summary else "",
        "losses": int(summary.get("losses", 0) or 0) if exact_summary else "",
        "draws": int(summary.get("draws", 0) or 0) if exact_summary else "",
        "adjudicated_games": int(summary.get("adjudicated_games", 0) or 0) if exact_summary else "",
        "adjudication_wins": int(summary.get("adjudication_wins", 0) or 0) if exact_summary else "",
        "adjudication_draws": int(summary.get("adjudication_draws", 0) or 0) if exact_summary else "",
        "adjudication_failed": int(summary.get("adjudication_failed", 0) or 0) if exact_summary else "",
        "shards": shard_count,
        "exact_summary": int(exact_summary),
        **action_metrics,
    }


def train_round(
    cfg: HybridAgentConfig,
    model: HybridPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    replay_store: ReplayStore,
    device: torch.device,
    records: list[StepRecord] | None = None,
) -> Dict[str, float]:
    sample_started_at = time.perf_counter()
    if records is None:
        records = replay_store.sample(cfg.training.replay_batch_size)
    else:
        records = list(records)
    sample_elapsed = time.perf_counter() - sample_started_at
    if len(records) < cfg.replay.min_train_steps:
        return {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "steps": 0.0,
            "sample_sec": sample_elapsed,
            "fetch_sec": 0.0,
            "optimize_sec": 0.0,
        }
    model.train()
    model.to(device)
    totals = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "steps": 0.0,
        "sample_sec": sample_elapsed,
        "fetch_sec": 0.0,
        "optimize_sec": 0.0,
    }
    batch_size = max(1, int(cfg.training.batch_size))
    for epoch_idx in range(cfg.training.epochs_per_iteration):
        epoch_loss = 0.0
        epoch_steps = 0.0
        epoch_started_at = time.perf_counter()
        shuffled_records = list(records)
        random.shuffle(shuffled_records)
        offset = 0
        batches_completed = 0
        while offset < len(shuffled_records):
            batch_records = shuffled_records[offset : offset + batch_size]
            fetch_started_at = time.perf_counter()
            batch = collate_batch(batch_records, cfg)
            totals["fetch_sec"] += time.perf_counter() - fetch_started_at
            optimize_started_at = time.perf_counter()
            encoded = batch["encoded"].to(device)
            policy_targets = batch["policy_targets"].to(device)
            value_targets = batch["value_targets"].to(device)
            output = model(encoded)
            log_probs = torch.log_softmax(output.policy_logits, dim=-1)
            policy_loss = -(policy_targets * log_probs).sum(dim=-1).mean()
            value_loss = nn.functional.mse_loss(output.value, value_targets)
            loss = (
                cfg.training.policy_loss_weight * policy_loss
                + cfg.training.value_loss_weight * value_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)
            optimizer.step()
            size = float(len(batch_records))
            totals["loss"] += float(loss.item()) * size
            totals["policy_loss"] += float(policy_loss.item()) * size
            totals["value_loss"] += float(value_loss.item()) * size
            totals["steps"] += size
            epoch_loss += float(loss.item()) * size
            epoch_steps += size
            totals["optimize_sec"] += time.perf_counter() - optimize_started_at
            offset += len(batch_records)
            batches_completed += 1
        avg_epoch_loss = epoch_loss / max(1.0, epoch_steps)
        print(
            f"[train] update epoch {epoch_idx + 1}/{cfg.training.epochs_per_iteration} "
            f"samples={int(epoch_steps)} batches={batches_completed} train_batch={batch_size} "
            f"loss={avg_epoch_loss:.4f} "
            f"time={_format_seconds(time.perf_counter() - epoch_started_at)}",
            flush=True,
        )
    if totals["steps"] > 0:
        for key in ("loss", "policy_loss", "value_loss"):
            totals[key] /= totals["steps"]
    model.eval()
    return totals


def count_replay_shards(replay_dir: Path) -> int:
    return sum(1 for _ in replay_dir.glob("replay_*.pt"))


def ensure_successful_selfplay(
    result: subprocess.CompletedProcess[str],
    before_count: int,
    after_count: int,
    label: str = "selfplay",
) -> None:
    if result.returncode != 0:
        raise RuntimeError(
            f"Self-play match failed ({label}).\n"
            f"Return code: {result.returncode}\n"
            f"Stdout tail:\n{chr(10).join(result.stdout.splitlines()[-10:])}\n"
            f"Stderr tail:\n{chr(10).join(result.stderr.splitlines()[-10:])}"
        )
    if after_count <= before_count:
        raise RuntimeError(
            f"Self-play match completed but produced no replay shard ({label}).\n"
            f"Stdout tail:\n{chr(10).join(result.stdout.splitlines()[-10:])}\n"
            f"Stderr tail:\n{chr(10).join(result.stderr.splitlines()[-10:])}"
        )


def _bot_command(
    bot_script: Path,
    checkpoint_path: Path,
    replay_dir: Path,
    cfg: HybridAgentConfig,
) -> list[str]:
    command = [
        "python",
        str(bot_script),
        "--checkpoint",
        str(checkpoint_path),
        "--replay-dir",
        str(replay_dir),
        "--simulations",
        str(cfg.search.num_simulations),
        "--max-depth",
        str(cfg.search.max_depth),
        "--top-k-actions",
        str(cfg.search.top_k_actions),
        "--search-batch-size",
        str(cfg.search.batch_size),
        "--max-game-actions",
        str(max(1, int(cfg.selfplay.max_actions_per_game))),
    ]
    if cfg.selfplay.wall_clock_per_action_seconds is not None:
        command.extend(
            [
                "--wall-clock-per-action-seconds",
                str(max(0.0, float(cfg.selfplay.wall_clock_per_action_seconds))),
            ]
        )
    return command


def _format_wall_clock_per_action(cfg: HybridAgentConfig) -> str:
    budget = cfg.selfplay.wall_clock_per_action_seconds
    if budget is None or float(budget) <= 0.0:
        return "off"
    return f"{float(budget):.3f}s"


def _find_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_port(host: str, port: int, process: subprocess.Popen[str], timeout_seconds: float = 90.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Persistent bot server exited early with code {process.returncode}")
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(0.2)
    raise TimeoutError(f"Persistent bot server did not open {host}:{port}: {last_error}")


def _shutdown_persistent_server(host: str, port: int, process: subprocess.Popen[str], timeout_seconds: float = 20.0) -> None:
    if process.poll() is not None:
        return
    try:
        with socket.create_connection((host, port), timeout=5.0) as sock:
            payload = json.dumps({"type": "shutdown"}) + "\n"
            sock.sendall(payload.encode("utf-8"))
    except OSError:
        pass
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            process.kill()
        process.wait(timeout=10)


def _start_persistent_bot_server(
    cfg: HybridAgentConfig,
    bot_server_script: Path,
    checkpoint_path: Path,
    replay_dir: Path,
    log_dir: Path,
) -> tuple[subprocess.Popen[str], str, int, Path, Path]:
    host = "127.0.0.1"
    port = _find_free_local_port()
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"persistent_bot_{port}.stdout.log"
    stderr_path = log_dir / f"persistent_bot_{port}.stderr.log"
    command = [
        "python",
        str(bot_server_script),
        "--host",
        host,
        "--port",
        str(port),
        "--checkpoint",
        str(checkpoint_path),
        "--replay-dir",
        str(replay_dir),
        "--simulations",
        str(cfg.search.num_simulations),
        "--max-depth",
        str(cfg.search.max_depth),
        "--top-k-actions",
        str(cfg.search.top_k_actions),
        "--search-batch-size",
        str(cfg.search.batch_size),
        "--max-game-actions",
        str(max(1, int(cfg.selfplay.max_actions_per_game))),
    ]
    if cfg.selfplay.wall_clock_per_action_seconds is not None:
        command.extend(
            [
                "--wall-clock-per-action-seconds",
                str(max(0.0, float(cfg.selfplay.wall_clock_per_action_seconds))),
            ]
        )
    if cfg.selfplay.profile_selfplay:
        command.append("--profile-selfplay")
    stdout_handle = stdout_path.open("w", encoding="utf-8")
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=stdout_handle,
        stderr=stderr_handle,
        text=True,
    )
    stdout_handle.close()
    stderr_handle.close()
    try:
        _wait_for_port(host, port, process)
    except Exception:
        _shutdown_persistent_server(host, port, process, timeout_seconds=3.0)
        stdout_tail = "\n".join(stdout_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:])
        stderr_tail = "\n".join(stderr_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:])
        raise RuntimeError(f"Persistent bot server failed to start.\nStdout tail:\n{stdout_tail}\nStderr tail:\n{stderr_tail}")
    return process, host, port, stdout_path, stderr_path


def _profile_summary_from_files(paths: list[Path]) -> Dict[str, float]:
    text_parts: list[str] = []
    for path in paths:
        try:
            text_parts.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return _profile_summary("\n".join(text_parts))


def _persistent_bridge_command(bridge_script: Path, host: str, port: int) -> list[str]:
    return ["python", str(bridge_script), "--host", host, "--port", str(port)]


def _run_selfplay_game(
    cfg: HybridAgentConfig,
    bot_script: Path,
    checkpoint_path: Path,
    replay_dir: Path,
    tribes: list[str],
    workdir: Path,
    device: torch.device,
    local_iteration: int,
    iteration: int,
    game_idx: int,
    games: int,
    bot_commands: list[list[str]] | None = None,
) -> dict[str, object]:
    game_cfg = copy.deepcopy(cfg)
    seed = iteration * 10_000 + game_idx
    game_cfg.selfplay.game_seed = seed
    game_cfg.selfplay.agent_seed = seed
    game_cfg.selfplay.level_seed = seed
    label = f"i{local_iteration + 1} g{game_idx + 1}/{games} seed={seed}"
    command = _bot_command(bot_script, checkpoint_path, replay_dir, game_cfg)
    commands = bot_commands if bot_commands is not None else [command, list(command)]
    started_at = time.monotonic()
    result = run_selfplay(
        game_cfg,
        commands,
        tribes,
        workdir,
        checkpoint_path=checkpoint_path,
        replay_store=ReplayStore(replay_dir, game_cfg.replay.capacity_steps, game_cfg.replay.shard_prefix, load_existing=False),
        device=device,
        progress_label=label,
    )
    return {
        "game_idx": game_idx,
        "seed": seed,
        "label": label,
        "elapsed": time.monotonic() - started_at,
        "result": result,
    }


def train(cfg: HybridAgentConfig, *, device: torch.device, iterations: int | None = None, games_per_iteration: int | None = None) -> None:
    cfg.training.device = str(device)
    workdir = Path(__file__).resolve().parents[2]
    checkpoint_path = cfg.training.checkpoint_path
    replay_store = ReplayStore(
        cfg.replay.replay_dir,
        cfg.replay.capacity_steps,
        cfg.replay.shard_prefix,
        cache_records=False,
    )
    model = HybridPolicyValueNet(cfg.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    start_iteration = _load_checkpoint(model, checkpoint_path, optimizer)
    move_optimizer_state(optimizer, device)
    print(
        f"[train] start checkpoint_iter={start_iteration} params={count_parameters(model):,} "
        f"device={device} search=native sims={cfg.search.num_simulations} "
        f"wall_clock_per_action={_format_wall_clock_per_action(cfg)} "
        f"search_batch={cfg.search.batch_size} train_batch={cfg.training.batch_size} "
        f"replay_batch={cfg.training.replay_batch_size} selfplay_workers={cfg.selfplay.workers} "
        f"persistent_bot={cfg.selfplay.persistent_bot} "
        f"offline_augment_symmetries={cfg.training.augment_symmetries} "
        f"augmentation_prob={cfg.training.augmentation_prob}",
        flush=True,
    )
    bot_script = workdir / "py" / "bots" / "hybrid_nn_bot.py"
    bot_server_script = workdir / "py" / "training" / "persistent_bot_server.py"
    bot_bridge_script = workdir / "py" / "training" / "persistent_bot_bridge.py"
    total_iterations = iterations if iterations is not None else cfg.training.num_iterations
    games = games_per_iteration if games_per_iteration is not None else cfg.training.selfplay_games_per_iteration
    tribes = ["Xin Xi", "Imperius"]
    if cfg.selfplay.workers > 2 and not cfg.selfplay.allow_unsafe_workers:
        raise ValueError("selfplay workers above 2 require allow_unsafe_workers=True")
    if cfg.selfplay.workers < 1:
        raise ValueError("selfplay workers must be at least 1")
    if cfg.selfplay.persistent_bot and cfg.selfplay.workers != 1:
        raise ValueError("persistent bot mode currently requires selfplay_workers=1")

    for local_iteration in range(total_iterations):
        iteration_started_at = time.monotonic()
        selfplay_elapsed_total = 0.0
        iteration = start_iteration + local_iteration + 1
        print(
            f"\n[iter {local_iteration + 1}/{total_iterations} | global {iteration}] "
            f"self-play games={games} mode={cfg.selfplay.game_mode} max_turns={cfg.selfplay.max_turns_capitals} "
            f"max_actions={cfg.selfplay.max_actions_per_game} "
            f"wall_clock_per_action={_format_wall_clock_per_action(cfg)} "
            f"action_timeout={cfg.selfplay.external_action_timeout_ms}ms workers={cfg.selfplay.workers} "
            f"persistent_bot={cfg.selfplay.persistent_bot}",
            flush=True,
        )
        iteration_action_counts: Counter[str] = Counter()
        iteration_wins = iteration_losses = iteration_draws = 0
        iteration_adjudicated = iteration_adjudication_wins = iteration_adjudication_draws = iteration_adjudication_failed = 0
        iteration_adjudication_sec = 0.0
        iteration_profile_actions = 0.0
        before_iteration_shards = set(replay_store.shards())
        before_count = len(before_iteration_shards)
        workers = max(1, int(cfg.selfplay.workers))
        persistent_process: subprocess.Popen[str] | None = None
        persistent_host = "127.0.0.1"
        persistent_port = 0
        persistent_log_paths: list[Path] = []
        persistent_profile_summary: Dict[str, float] = {}
        try:
            if cfg.selfplay.persistent_bot:
                persistent_process, persistent_host, persistent_port, stdout_path, stderr_path = _start_persistent_bot_server(
                    cfg,
                    bot_server_script,
                    checkpoint_path,
                    cfg.replay.replay_dir,
                    Path(cfg.training.output_dir) / "persistent_bot_logs",
                )
                persistent_log_paths = [stdout_path, stderr_path]
                print(f"[persistent bot] started host={persistent_host} port={persistent_port}", flush=True)
            if workers == 1:
                game_results = []
                for game_idx in range(games):
                    seed = iteration * 10_000 + game_idx
                    before_game_shards = set(replay_store.shards())
                    before = len(before_game_shards)
                    print(f"[game {game_idx + 1}/{games}] seed={seed} start shards={before}", flush=True)
                    bridge_commands = None
                    if cfg.selfplay.persistent_bot:
                        bridge = _persistent_bridge_command(bot_bridge_script, persistent_host, persistent_port)
                        bridge_commands = [bridge, list(bridge)]
                    game_result = _run_selfplay_game(
                        cfg, bot_script, checkpoint_path, cfg.replay.replay_dir, tribes, workdir, device,
                        local_iteration, iteration, game_idx, games, bot_commands=bridge_commands,
                    )
                    result = game_result["result"]
                    after = count_replay_shards(cfg.replay.replay_dir)
                    ensure_successful_selfplay(result, before, after, str(game_result["label"]))
                    replay_store.refresh()
                    new_shards = sorted(set(replay_store.shards()) - before_game_shards)
                    game_result["new_shards"] = new_shards
                    game_results.append(game_result)
            else:
                game_results = []
        finally:
            if persistent_process is not None:
                _shutdown_persistent_server(persistent_host, persistent_port, persistent_process)
                persistent_profile_summary = _profile_summary_from_files(persistent_log_paths)
                print("[persistent bot] stopped", flush=True)

        if workers > 1:
            print(f"[selfplay] parallel start games={games} workers={workers} start_shards={before_count}", flush=True)
            game_results = []
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="selfplay") as executor:
                futures = [
                    executor.submit(
                        _run_selfplay_game,
                        cfg,
                        bot_script,
                        checkpoint_path,
                        cfg.replay.replay_dir,
                        tribes,
                        workdir,
                        device,
                        local_iteration,
                        iteration,
                        game_idx,
                        games,
                    )
                    for game_idx in range(games)
                ]
                done, pending = wait(futures, return_when=FIRST_EXCEPTION)
                first_error: BaseException | None = None
                for future in done:
                    try:
                        game_results.append(future.result())
                    except BaseException as exc:
                        first_error = exc
                        break
                if first_error is not None:
                    for future in pending:
                        future.cancel()
                    wait(pending, timeout=10)
                    if not cfg.selfplay.allow_partial_selfplay:
                        raise RuntimeError("Parallel self-play failed; partial iteration will not train.") from first_error
                for future in pending:
                    if future.cancelled():
                        continue
                    game_results.append(future.result())
            replay_store.refresh()
            after_count = count_replay_shards(cfg.replay.replay_dir)
            for game_result in game_results:
                ensure_successful_selfplay(game_result["result"], before_count, after_count, str(game_result["label"]))
            all_new_shards = sorted(set(replay_store.shards()) - before_iteration_shards)
            expected_min_shards = 2 * games
            if len(all_new_shards) < expected_min_shards and not cfg.selfplay.allow_partial_selfplay:
                raise RuntimeError(
                    f"Parallel self-play produced {len(all_new_shards)} new shards, expected at least {expected_min_shards}; "
                    "partial iteration will not train."
                )
            for game_result in game_results:
                game_result["new_shards"] = all_new_shards

        if workers == 1:
            for game_result in sorted(game_results, key=lambda item: int(item["game_idx"])):
                result = game_result["result"]
                elapsed = float(game_result["elapsed"])
                selfplay_elapsed_total += elapsed
                game_summary = _summarize_replay_shards(replay_store, list(game_result["new_shards"]))
                profile_summary = _profile_summary(result.stderr)
                iteration_profile_actions += float(profile_summary.get("profile_actions", 0.0))
                iteration_action_counts.update(game_summary.get("actions", {}))
                iteration_wins += int(game_summary["wins"])
                iteration_losses += int(game_summary["losses"])
                iteration_draws += int(game_summary["draws"])
                iteration_adjudicated += int(game_summary.get("adjudicated_games", 0) or 0)
                iteration_adjudication_wins += int(game_summary.get("adjudication_wins", 0) or 0)
                iteration_adjudication_draws += int(game_summary.get("adjudication_draws", 0) or 0)
                iteration_adjudication_failed += int(game_summary.get("adjudication_failed", 0) or 0)
                iteration_adjudication_sec += _adjudication_seconds(result.stdout)
                game_idx = int(game_result["game_idx"])
                _append_selfplay_game_metrics(
                    cfg.diagnostics.selfplay_games_csv,
                    _selfplay_game_row(
                        iteration=iteration,
                        local_iteration=local_iteration,
                        game_idx=game_idx,
                        games=games,
                        seed=int(game_result["seed"]),
                        elapsed=elapsed,
                        summary=game_summary,
                        profile_summary=profile_summary,
                        simulations=int(cfg.search.num_simulations),
                        shard_count=len(game_result["new_shards"]),
                        exact_summary=True,
                    ),
                )
                adj_note = ""
                ag = int(game_summary.get("adjudicated_games", 0) or 0)
                if ag:
                    adj_note = (
                        f" adjudication(shards={ag} failed={int(game_summary.get('adjudication_failed', 0) or 0)} "
                        f"adj_wins={int(game_summary.get('adjudication_wins', 0) or 0)})"
                    )
                print(
                    f"[game {game_idx + 1}/{games}] done time={_format_seconds(elapsed)} "
                    f"steps={game_summary['steps']} shards=+{len(game_result['new_shards'])} "
                    f"ended={_match_end_reason(result.stdout, result.stderr)} "
                    f"outcomes W/L/D={game_summary['wins']}/{game_summary['losses']}/{game_summary['draws']}"
                    f"{adj_note} "
                    f"actions {_format_action_counts(game_summary)}"
                    f"{_format_profile(profile_summary)}",
                    flush=True,
                )
        else:
            selfplay_elapsed_total = time.monotonic() - iteration_started_at
            aggregate_shards = sorted(set(replay_store.shards()) - before_iteration_shards)
            aggregate_summary = _summarize_replay_shards(replay_store, aggregate_shards)
            iteration_action_counts.update(aggregate_summary.get("actions", {}))
            iteration_wins += int(aggregate_summary["wins"])
            iteration_losses += int(aggregate_summary["losses"])
            iteration_draws += int(aggregate_summary["draws"])
            iteration_adjudicated += int(aggregate_summary.get("adjudicated_games", 0) or 0)
            iteration_adjudication_wins += int(aggregate_summary.get("adjudication_wins", 0) or 0)
            iteration_adjudication_draws += int(aggregate_summary.get("adjudication_draws", 0) or 0)
            iteration_adjudication_failed += int(aggregate_summary.get("adjudication_failed", 0) or 0)
            for game_result in sorted(game_results, key=lambda item: int(item["game_idx"])):
                game_idx = int(game_result["game_idx"])
                profile_summary = _profile_summary(game_result["result"].stderr)
                iteration_adjudication_sec += _adjudication_seconds(game_result["result"].stdout)
                iteration_profile_actions += float(profile_summary.get("profile_actions", 0.0))
                _append_selfplay_game_metrics(
                    cfg.diagnostics.selfplay_games_csv,
                    _selfplay_game_row(
                        iteration=iteration,
                        local_iteration=local_iteration,
                        game_idx=game_idx,
                        games=games,
                        seed=int(game_result["seed"]),
                        elapsed=float(game_result["elapsed"]),
                        summary={},
                        profile_summary=profile_summary,
                        simulations=int(cfg.search.num_simulations),
                        shard_count=0,
                        exact_summary=False,
                    ),
                )
                print(
                    f"[game {game_idx + 1}/{games}] done time={_format_seconds(float(game_result['elapsed']))} "
                    f"seed={game_result['seed']} "
                    f"ended={_match_end_reason(game_result['result'].stdout, game_result['result'].stderr)}"
                    f"{_format_profile(profile_summary)}",
                    flush=True,
                )
            print(
                f"[selfplay] parallel done time={_format_seconds(selfplay_elapsed_total)} "
                f"steps={aggregate_summary['steps']} shards=+{len(aggregate_shards)} "
                f"outcomes W/L/D={aggregate_summary['wins']}/{aggregate_summary['losses']}/{aggregate_summary['draws']} "
                f"actions {_format_action_counts(aggregate_summary)}",
                flush=True,
            )

        raw_iteration_shards = sorted(set(replay_store.shards()) - before_iteration_shards)
        current_train_records: list[StepRecord]
        current_train_shards: list[Path]
        old_replay_records = 0
        if cfg.training.augment_symmetries:
            augmented_shard, current_train_records, raw_record_count = _write_augmented_iteration_shard(
                replay_store,
                raw_iteration_shards,
                cfg,
                iteration,
                max_return_records=max(1, int(cfg.training.replay_batch_size)),
            )
            archive_dir = cfg.replay.replay_dir / "raw" / f"iter_{iteration:04d}"
            archived_shards = _archive_replay_shards(raw_iteration_shards, archive_dir)
            replay_store.refresh()
            current_train_shards = [
                shard for shard in replay_store.shards()
                if shard not in before_iteration_shards and _is_augmented_replay_shard(replay_store, shard)
            ]
            print(
                f"[replay] offline_augmentation raw_shards={len(raw_iteration_shards)} "
                f"raw_records={raw_record_count} augmented_shards={len(current_train_shards)} "
                f"augmented_records={len(current_train_records)} archived_raw_shards={len(archived_shards)}",
                flush=True,
            )
        else:
            current_train_records = _load_records_from_shards(replay_store, raw_iteration_shards)
            current_train_shards = raw_iteration_shards
            print(
                f"[replay] offline_augmentation disabled raw_shards={len(raw_iteration_shards)} "
                f"train_records={len(current_train_records)}",
                flush=True,
            )
        training_records, old_replay_records = _training_records_for_iteration(
            replay_store,
            current_train_records,
            current_train_shards,
            max_records=max(1, int(cfg.training.replay_batch_size)),
            fresh_fraction=float(cfg.replay.fresh_fraction),
        )
        print(
            f"[train] update start replay_steps={replay_store.step_count()} "
            f"fresh_train_records={len(current_train_records)} old_train_records={old_replay_records} "
            f"total_train_records={len(training_records)} "
            f"iteration_actions {_format_action_counts({'actions': dict(iteration_action_counts)})} "
            f"outcomes W/L/D={iteration_wins}/{iteration_losses}/{iteration_draws}",
            flush=True,
        )
        training_started_at = time.monotonic()
        metrics = train_round(cfg, model, optimizer, replay_store, device, records=training_records)
        training_elapsed = time.monotonic() - training_started_at
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_started_at = time.monotonic()
        payload = {
            "version": 2,
            "iteration": iteration,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
        }
        torch.save(payload, checkpoint_path)
        torch.save(payload, cfg.training.iteration_checkpoint_path(iteration))
        checkpoint_elapsed = time.monotonic() - checkpoint_started_at
        action_metrics = {f"action_{name.lower()}": count for name, count in sorted(iteration_action_counts.items())}
        if iteration_profile_actions <= 0.0 and persistent_profile_summary:
            iteration_profile_actions = float(persistent_profile_summary.get("profile_actions", 0.0))
        completed_games = max(1, len(game_results))
        seconds_per_game = selfplay_elapsed_total / completed_games
        action_decisions_sec = iteration_profile_actions / selfplay_elapsed_total if selfplay_elapsed_total > 0 else 0.0
        configured_sims_sec = (
            iteration_profile_actions * float(cfg.search.num_simulations) / selfplay_elapsed_total
            if selfplay_elapsed_total > 0 else 0.0
        )
        row = {
            "iteration": iteration,
            "replay_steps": replay_store.step_count(),
            "replay_shards": count_replay_shards(cfg.replay.replay_dir),
            "games_completed": len(game_results),
            "wins": iteration_wins,
            "losses": iteration_losses,
            "draws": iteration_draws,
            "adjudicated_games": iteration_adjudicated,
            "adjudication_wins": iteration_adjudication_wins,
            "adjudication_draws": iteration_adjudication_draws,
            "adjudication_failed": iteration_adjudication_failed,
            "adjudication_sec": iteration_adjudication_sec,
            "seconds_per_game": seconds_per_game,
            "action_decisions": iteration_profile_actions,
            "action_decisions_sec": action_decisions_sec,
            "configured_sims_sec": configured_sims_sec,
            "selfplay_sec": selfplay_elapsed_total,
            "train_sec": training_elapsed,
            "checkpoint_sec": checkpoint_elapsed,
            "total_sec": time.monotonic() - iteration_started_at,
            **action_metrics,
            **metrics,
        }
        _append_metrics(cfg.diagnostics.metrics_csv, row)
        print(
            f"[iter {iteration}] done loss={metrics['loss']:.4f} policy={metrics['policy_loss']:.4f} "
            f"value={metrics['value_loss']:.4f} replay_steps={row['replay_steps']} "
            f"sec/game={seconds_per_game:.1f} actions/sec={action_decisions_sec:.2f} sims/sec={configured_sims_sec:.1f} "
            f"time selfplay={_format_seconds(selfplay_elapsed_total)} train={_format_seconds(training_elapsed)} "
            f"checkpoint={_format_seconds(checkpoint_elapsed)} total={_format_seconds(time.monotonic() - iteration_started_at)}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the benchmark-style Tribes hybrid RL agent.")
    parser.add_argument("--iterations", "--rounds", dest="iterations", type=int, default=None)
    parser.add_argument("--games-per-iteration", "--games-per-round", dest="games_per_iteration", type=int, default=None)
    parser.add_argument("--simulations", "--mcts-sims", dest="simulations", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--top-k-actions", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--search-batch-size", type=int, default=None)
    parser.add_argument("--replay-batch-size", type=int, default=None)
    parser.add_argument("--replay-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--max-turns-capitals", type=int, default=None)
    parser.add_argument("--max-actions-per-turn", type=int, default=None)
    parser.add_argument("--max-actions-per-game", type=int, default=None)
    parser.add_argument("--adjudicate-incomplete-games", action="store_true")
    parser.add_argument("--adjudicator-bot", type=str, default=None)
    parser.add_argument("--adjudication-max-turns-capitals", type=int, default=None)
    parser.add_argument("--adjudication-max-actions-per-game", type=int, default=None)
    parser.add_argument("--match-timeout-seconds", type=int, default=None)
    parser.add_argument("--external-action-timeout-ms", type=int, default=None)
    parser.add_argument("--wall-clock-per-action-seconds", type=float, default=None)
    parser.add_argument("--wall-clock-per-turn-seconds", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--progress-interval-seconds", type=int, default=None)
    parser.add_argument("--selfplay-workers", type=int, default=None)
    parser.add_argument("--allow-unsafe-workers", action="store_true")
    parser.add_argument("--allow-partial-selfplay", action="store_true")
    parser.add_argument("--profile-selfplay", action="store_true")
    parser.add_argument("--persistent-bot", action="store_true")
    parser.add_argument("--no-persistent-bot", action="store_true")
    parser.add_argument("--augment-symmetries", action="store_true", help="Enable default offline D4 replay augmentation.")
    parser.add_argument("--no-augment-symmetries", action="store_true", help="Disable offline D4 replay augmentation.")
    parser.add_argument("--augmentation-prob", type=float, default=None)
    parser.add_argument("--augmentation-seed", type=int, default=None)
    args = parser.parse_args()

    cfg = HybridAgentConfig()
    if args.simulations is not None:
        cfg.search.num_simulations = args.simulations
    if args.max_depth is not None:
        cfg.search.max_depth = args.max_depth
    if args.top_k_actions is not None:
        cfg.search.top_k_actions = args.top_k_actions
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.search_batch_size is not None:
        cfg.search.batch_size = args.search_batch_size
    if args.replay_batch_size is not None:
        cfg.training.replay_batch_size = args.replay_batch_size
    if args.replay_dir is not None:
        cfg.replay.replay_dir = args.replay_dir
    if args.checkpoint is not None:
        cfg.training.checkpoint_path = args.checkpoint
    if args.max_turns_capitals is not None:
        cfg.selfplay.max_turns_capitals = args.max_turns_capitals
    if args.max_actions_per_turn is not None:
        cfg.selfplay.max_actions_per_turn = args.max_actions_per_turn
    if args.max_actions_per_game is not None:
        cfg.selfplay.max_actions_per_game = args.max_actions_per_game
    if args.adjudicate_incomplete_games:
        cfg.selfplay.adjudicate_incomplete_games = True
    if args.adjudicator_bot is not None:
        cfg.selfplay.adjudicator_bot = args.adjudicator_bot
    if args.adjudication_max_turns_capitals is not None:
        cfg.selfplay.adjudication_max_turns_capitals = args.adjudication_max_turns_capitals
    if args.adjudication_max_actions_per_game is not None:
        cfg.selfplay.adjudication_max_actions_per_game = args.adjudication_max_actions_per_game
    if args.match_timeout_seconds is not None:
        cfg.selfplay.timeout_seconds = args.match_timeout_seconds
    if args.external_action_timeout_ms is not None:
        cfg.selfplay.external_action_timeout_ms = args.external_action_timeout_ms
    wall_clock_per_action = args.wall_clock_per_action_seconds
    if wall_clock_per_action is None:
        wall_clock_per_action = args.wall_clock_per_turn_seconds
    if wall_clock_per_action is not None:
        cfg.selfplay.wall_clock_per_action_seconds = wall_clock_per_action
    if args.progress_interval_seconds is not None:
        cfg.selfplay.progress_interval_seconds = args.progress_interval_seconds
    if args.selfplay_workers is not None:
        cfg.selfplay.workers = args.selfplay_workers
    if args.allow_unsafe_workers:
        cfg.selfplay.allow_unsafe_workers = True
    if args.allow_partial_selfplay:
        cfg.selfplay.allow_partial_selfplay = True
    if args.profile_selfplay:
        cfg.selfplay.profile_selfplay = True
    if args.persistent_bot:
        cfg.selfplay.persistent_bot = True
    if args.no_persistent_bot:
        cfg.selfplay.persistent_bot = False
    if args.augment_symmetries:
        cfg.training.augment_symmetries = True
    if args.no_augment_symmetries:
        cfg.training.augment_symmetries = False
    if args.augmentation_prob is not None:
        cfg.training.augmentation_prob = args.augmentation_prob
        if cfg.training.augment_symmetries and args.augmentation_prob < 1.0:
            print(
                "[train] warning: --augmentation-prob is deprecated for offline augmentation and will be ignored; "
                "offline D4 augmentation materializes all 8 variants.",
                flush=True,
            )
    if args.augmentation_seed is not None:
        cfg.training.augmentation_seed = args.augmentation_seed
    if not 0.0 <= cfg.training.augmentation_prob <= 1.0:
        parser.error("--augmentation-prob must be between 0.0 and 1.0")
    if cfg.selfplay.workers > 2 and not cfg.selfplay.allow_unsafe_workers:
        parser.error("--selfplay-workers above 2 requires --allow-unsafe-workers")
    if cfg.selfplay.workers < 1:
        parser.error("--selfplay-workers must be at least 1")
    if cfg.selfplay.persistent_bot and cfg.selfplay.workers != 1:
        parser.error("--persistent-bot currently requires --selfplay-workers 1")
    device = require_cuda_device()
    train(cfg, device=device, iterations=args.iterations, games_per_iteration=args.games_per_iteration)


if __name__ == "__main__":
    main()
