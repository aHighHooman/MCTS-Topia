from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import statistics
import time
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader

from .augment_replay import augment_record
from .benchmark_selfplay import _bot_command, _fallback_summary, _new_replay_step_counts, _utc_now
from .config import HybridAgentConfig
from .device import move_optimizer_state
from .model import HybridPolicyValueNet, count_parameters
from .replay import ReplayStore, StepRecord, record_to_payload
from .selfplay import run_selfplay
from .symmetry_consistency import evaluate_symmetry_consistency
from .train import ReplayDataset, _load_checkpoint, collate_batch, train_round


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _write_replay(samples: list[StepRecord], replay_dir: Path, prefix: str, max_samples_per_shard: int) -> None:
    if replay_dir.exists():
        shutil.rmtree(replay_dir)
    replay_dir.mkdir(parents=True, exist_ok=True)
    max_samples = max(1, int(max_samples_per_shard))
    for shard_index, offset in enumerate(range(0, len(samples), max_samples)):
        chunk = samples[offset : offset + max_samples]
        payload = {
            "version": 2,
            "created_at": time.time(),
            "iteration": 0,
            "player_id": None,
            "step_count": len(chunk),
            "records": [record_to_payload(record) for record in chunk],
        }
        torch.save(payload, replay_dir / f"{prefix}_{shard_index:04d}_{int(time.time() * 1000)}.pt")


def _split_samples(
    samples: list[StepRecord],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[StepRecord], list[StepRecord]]:
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, int(round(len(shuffled) * validation_fraction))) if len(shuffled) > 1 else 0
    validation = shuffled[:validation_count]
    train = shuffled[validation_count:] or shuffled
    return train, validation


def _symmetry_eval_samples(samples: Iterable[StepRecord]) -> list[StepRecord]:
    out: list[StepRecord] = []
    specs = [(rotation, mirror) for mirror in (False, True) for rotation in range(4)]
    for record in samples:
        for rotation, mirror in specs:
            if rotation == 0 and not mirror:
                out.append(record)
            else:
                out.append(augment_record(record, rotation, mirror))
    return out


def _d4_expanded_samples(samples: Iterable[StepRecord]) -> list[StepRecord]:
    out: list[StepRecord] = []
    for record in samples:
        for mirror in (False, True):
            for rotation in range(4):
                if rotation == 0 and not mirror:
                    out.append(record)
                else:
                    out.append(augment_record(record, rotation, mirror))
    return out


@torch.no_grad()
def _evaluate_samples(
    cfg: HybridAgentConfig,
    model: HybridPolicyValueNet,
    samples: list[StepRecord],
    device: torch.device,
) -> dict[str, float]:
    if not samples:
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "steps": 0.0}
    eval_cfg = deepcopy(cfg)
    eval_cfg.training.augment_symmetries = False
    model.eval()
    loader = DataLoader(
        ReplayDataset(samples, eval_cfg),
        batch_size=eval_cfg.training.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, eval_cfg),
    )
    totals = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "steps": 0.0}
    for batch in loader:
        encoded = batch["encoded"].to(device)
        policy_targets = batch["policy_targets"].to(device)
        value_targets = batch["value_targets"].to(device)
        output = model(encoded)
        log_probs = torch.log_softmax(output.policy_logits, dim=-1)
        policy_loss = -(policy_targets * log_probs).sum(dim=-1).mean()
        value_loss = nn.functional.mse_loss(output.value, value_targets)
        loss = (
            eval_cfg.training.policy_loss_weight * policy_loss
            + eval_cfg.training.value_loss_weight * value_loss
        )
        size = float(len(batch["records"]))
        totals["loss"] += float(loss.item()) * size
        totals["policy_loss"] += float(policy_loss.item()) * size
        totals["value_loss"] += float(value_loss.item()) * size
        totals["steps"] += size
    for key in ("loss", "policy_loss", "value_loss"):
        totals[key] /= max(1.0, totals["steps"])
    return totals


def _generate_selfplay_replay(
    cfg: HybridAgentConfig,
    *,
    games: int,
    seed_base: int,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    if games <= 0:
        return {"games_requested": 0, "games_completed": 0, "replay_dir": str(cfg.replay.replay_dir)}
    workdir = Path(__file__).resolve().parents[2]
    replay_dir = output_dir / "generated_replay"
    if replay_dir.exists():
        shutil.rmtree(replay_dir)
    replay_dir.mkdir(parents=True, exist_ok=True)
    cfg.replay.replay_dir = replay_dir
    cfg.training.output_dir = output_dir / "selfplay_artifacts"
    cfg.training.checkpoint_dir = cfg.training.output_dir / "checkpoints"
    cfg.training.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    command = _bot_command(cfg, workdir, cfg.training.checkpoint_path, device)
    tribes = ["Xin Xi", "Imperius"]
    started = time.perf_counter()
    replay_steps_per_game: list[int] = []
    fallback_count = 0
    returncodes: list[int] = []
    for game_idx in range(games):
        seed = seed_base + game_idx
        cfg.selfplay.game_seed = seed
        cfg.selfplay.agent_seed = seed
        cfg.selfplay.level_seed = seed
        store = ReplayStore(replay_dir, cfg.replay.capacity_steps, cfg.replay.shard_prefix)
        before = set(store.shards())
        result = run_selfplay(
            cfg,
            [command, list(command)],
            tribes,
            workdir,
            checkpoint_path=cfg.training.checkpoint_path,
            replay_store=store,
            device=device,
            progress_label=f"augment-bench selfplay {game_idx + 1}/{games}",
        )
        returncodes.append(int(result.returncode))
        fallback_count += int(_fallback_summary(result)["fallback_count"])
        replay_steps_per_game.append(sum(_new_replay_step_counts(replay_dir, cfg.replay.shard_prefix, before)))
        if result.returncode != 0:
            break
    store = ReplayStore(replay_dir, cfg.replay.capacity_steps, cfg.replay.shard_prefix)
    return {
        "games_requested": int(games),
        "games_completed": len(returncodes),
        "returncodes": returncodes,
        "selfplay_sec": time.perf_counter() - started,
        "replay_dir": str(replay_dir),
        "replay_shards": len(store.shards()),
        "replay_steps": store.step_count(),
        "mean_replay_steps_game": statistics.fmean(replay_steps_per_game) if replay_steps_per_game else 0.0,
        "fallback_count": fallback_count,
    }


def _run_variant(
    base_cfg: HybridAgentConfig,
    *,
    name: str,
    train_samples: list[StepRecord],
    validation_samples: list[StepRecord],
    symmetry_validation_samples: list[StepRecord],
    output_dir: Path,
    device: torch.device,
    seed: int,
    training_iterations: int,
    augment: bool,
    augmentation_prob: float,
    max_samples_per_shard: int,
    replay_batch_size: int | None = None,
) -> dict[str, Any]:
    cfg = deepcopy(base_cfg)
    cfg.training.device = str(device)
    cfg.training.augment_symmetries = augment
    cfg.training.augmentation_prob = float(augmentation_prob if augment else 0.0)
    cfg.training.augmentation_seed = seed
    if replay_batch_size is not None:
        cfg.training.replay_batch_size = max(1, int(replay_batch_size))
    run_dir = output_dir / name
    cfg.replay.replay_dir = run_dir / "replay_train"
    cfg.training.output_dir = run_dir
    cfg.training.checkpoint_dir = run_dir / "checkpoints"
    cfg.diagnostics.metrics_csv = run_dir / "metrics.csv"
    _write_replay(train_samples, cfg.replay.replay_dir, cfg.replay.shard_prefix, max_samples_per_shard)
    cfg.training.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = HybridPolicyValueNet(cfg.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    checkpoint_iter = _load_checkpoint(model, base_cfg.training.checkpoint_path, optimizer)
    move_optimizer_state(optimizer, device)
    replay_store = ReplayStore(cfg.replay.replay_dir, cfg.replay.capacity_steps, cfg.replay.shard_prefix)

    eval_before = _evaluate_samples(cfg, model, validation_samples, device)
    symmetry_eval_before = _evaluate_samples(cfg, model, symmetry_validation_samples, device)
    consistency_before = evaluate_symmetry_consistency(
        cfg,
        model,
        validation_samples,
        device,
        max_records=len(validation_samples),
    )
    train_metrics_by_iteration: list[dict[str, float]] = []
    started = time.perf_counter()
    for iteration_idx in range(training_iterations):
        random.seed(seed + iteration_idx)
        torch.manual_seed(seed + iteration_idx)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed + iteration_idx)
        metrics = train_round(cfg, model, optimizer, replay_store, device)
        train_metrics_by_iteration.append({key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))})
    train_sec = time.perf_counter() - started
    eval_after = _evaluate_samples(cfg, model, validation_samples, device)
    symmetry_eval_after = _evaluate_samples(cfg, model, symmetry_validation_samples, device)
    consistency_after = evaluate_symmetry_consistency(
        cfg,
        model,
        validation_samples,
        device,
        max_records=len(validation_samples),
    )
    return {
        "schema_version": 1,
        "benchmark": "augmentation_training_ab",
        "variant": name,
        "seed": int(seed),
        "checkpoint_iter": int(checkpoint_iter),
        "train_sec": train_sec,
        "training_iterations": int(training_iterations),
        "augment_symmetries": bool(augment),
        "augmentation_prob": float(cfg.training.augmentation_prob),
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "symmetry_validation_samples": len(symmetry_validation_samples),
        "params": count_parameters(model),
        "eval_before": eval_before,
        "eval_after": eval_after,
        "symmetry_eval_before": symmetry_eval_before,
        "symmetry_eval_after": symmetry_eval_after,
        "d4_eval_before": symmetry_eval_before,
        "d4_eval_after": symmetry_eval_after,
        "symmetry_consistency_before": consistency_before,
        "symmetry_consistency_after": consistency_after,
        "eval_loss_delta": eval_after["loss"] - eval_before["loss"],
        "symmetry_eval_loss_delta": symmetry_eval_after["loss"] - symmetry_eval_before["loss"],
        "d4_eval_loss_delta": symmetry_eval_after["loss"] - symmetry_eval_before["loss"],
        "symmetry_gap_after": symmetry_eval_after["loss"] - eval_after["loss"],
        "d4_eval_gap_after": symmetry_eval_after["loss"] - eval_after["loss"],
        "symmetry_policy_js_delta": consistency_after["policy_js"] - consistency_before["policy_js"],
        "train_metrics_by_iteration": train_metrics_by_iteration,
        "config": _jsonable(cfg),
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows) if rows else 0.0


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    variant_names = {str(row["variant"]) for row in rows}
    variants = [name for name in ("standard", "d4_expanded") if name in variant_names]
    variants.extend(sorted(variant_names - set(variants)))
    out: list[dict[str, Any]] = []
    for variant in variants:
        group = [row for row in rows if row["variant"] == variant]
        out.append(
            {
                "variant": variant,
                "train_sec_mean": _mean(group, "train_sec"),
                "eval_loss_after_mean": statistics.fmean(float(row["eval_after"]["loss"]) for row in group),
                "eval_loss_delta_mean": _mean(group, "eval_loss_delta"),
                "symmetry_eval_loss_after_mean": statistics.fmean(float(row["symmetry_eval_after"]["loss"]) for row in group),
                "d4_eval_loss_after_mean": statistics.fmean(float(row["symmetry_eval_after"]["loss"]) for row in group),
                "symmetry_eval_loss_delta_mean": _mean(group, "symmetry_eval_loss_delta"),
                "symmetry_gap_after_mean": _mean(group, "symmetry_gap_after"),
                "d4_eval_gap_after_mean": _mean(group, "symmetry_gap_after"),
                "symmetry_policy_js_after_mean": statistics.fmean(
                    float(row["symmetry_consistency_after"]["policy_js"]) for row in group
                ),
                "symmetry_policy_l1_after_mean": statistics.fmean(
                    float(row["symmetry_consistency_after"]["policy_l1"]) for row in group
                ),
            }
        )
    baseline = next((item for item in out if item["variant"] == "standard"), None)
    if baseline:
        for item in out:
            item["eval_loss_after_vs_standard"] = item["eval_loss_after_mean"] - baseline["eval_loss_after_mean"]
            item["symmetry_loss_after_vs_standard"] = (
                item["symmetry_eval_loss_after_mean"] - baseline["symmetry_eval_loss_after_mean"]
            )
            item["d4_loss_after_vs_standard"] = (
                item["d4_eval_loss_after_mean"] - baseline["d4_eval_loss_after_mean"]
            )
            item["train_sec_vs_standard_pct"] = (
                100.0 * (item["train_sec_mean"] / baseline["train_sec_mean"] - 1.0)
                if baseline["train_sec_mean"] > 0
                else 0.0
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A/B benchmark standard replay training vs expanded rotation+mirroring replay augmentation."
    )
    parser.add_argument("--replay-dir", type=Path, default=Path("rl/replay"))
    parser.add_argument("--output-dir", type=Path, default=Path("rl/augmentation_benchmarks"))
    parser.add_argument("--jsonl", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--training-iterations", type=int, default=3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--training-batch-size", type=int, default=None)
    parser.add_argument("--replay-batch-size", type=int, default=None)
    parser.add_argument("--epochs-per-iteration", type=int, default=None)
    parser.add_argument("--no-d4-expanded", action="store_true")
    parser.add_argument("--expanded-replay-batch-multiplier", type=int, default=8)
    parser.add_argument("--max-samples-per-shard", type=int, default=5000)
    parser.add_argument("--generate-selfplay-games", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--require-checkpoint", action="store_true")
    parser.add_argument("--simulations", type=int, default=None)
    parser.add_argument("--search-batch-size", type=int, default=None)
    parser.add_argument("--max-turns-capitals", type=int, default=None)
    parser.add_argument("--match-timeout-seconds", type=int, default=None)
    args = parser.parse_args()

    if not 0.0 <= args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be >= 0.0 and < 1.0")
    cfg = HybridAgentConfig()
    cfg.replay.replay_dir = args.replay_dir
    if args.checkpoint is not None:
        cfg.training.checkpoint_path = args.checkpoint
    if args.training_batch_size is not None:
        cfg.training.batch_size = args.training_batch_size
    if args.replay_batch_size is not None:
        cfg.training.replay_batch_size = args.replay_batch_size
    if args.epochs_per_iteration is not None:
        cfg.training.epochs_per_iteration = args.epochs_per_iteration
    if args.simulations is not None:
        cfg.search.num_simulations = args.simulations
    if args.search_batch_size is not None:
        cfg.search.batch_size = args.search_batch_size
    if args.max_turns_capitals is not None:
        cfg.selfplay.max_turns_capitals = args.max_turns_capitals
    if args.match_timeout_seconds is not None:
        cfg.selfplay.timeout_seconds = args.match_timeout_seconds

    device = _device(args.device)
    if not cfg.training.checkpoint_path.exists():
        message = (
            f"checkpoint not found: {cfg.training.checkpoint_path}; benchmark will start from random initialization"
        )
        if args.require_checkpoint:
            raise RuntimeError(message)
        print(f"[augment-bench] warning {message}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.jsonl or args.output_dir / f"augmentation_benchmark_{int(time.time())}.jsonl"
    summary_csv = args.summary_csv or args.output_dir / "augmentation_benchmark_summary.csv"

    selfplay_summary = _generate_selfplay_replay(
        cfg,
        games=args.generate_selfplay_games,
        seed_base=args.seed,
        output_dir=args.output_dir,
        device=device,
    )

    store = ReplayStore(cfg.replay.replay_dir, cfg.replay.capacity_steps, cfg.replay.shard_prefix)
    samples = store.load_records()
    if len(samples) < 2:
        raise RuntimeError(f"Need at least 2 replay samples for benchmark, found {len(samples)} in {cfg.replay.replay_dir}")

    train_samples, validation_samples = _split_samples(
        samples,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    symmetry_validation_samples = _symmetry_eval_samples(validation_samples)
    print(
        "[augment-bench] "
        f"samples train={len(train_samples)} validation={len(validation_samples)} "
        f"symmetry_validation={len(symmetry_validation_samples)} device={device} "
        f"checkpoint={cfg.training.checkpoint_path} checkpoint_exists={cfg.training.checkpoint_path.exists()} "
        f"training_iterations={args.training_iterations} jsonl={jsonl_path}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    variants: list[tuple[str, bool, list[StepRecord], int | None]] = [
        ("standard", False, train_samples, None),
    ]
    if not args.no_d4_expanded:
        expanded_samples = _d4_expanded_samples(train_samples)
        expanded_replay_batch = cfg.training.replay_batch_size * max(1, int(args.expanded_replay_batch_multiplier))
        variants.append(("d4_expanded", False, expanded_samples, expanded_replay_batch))
    with jsonl_path.open("a", encoding="utf-8") as handle:
        for variant, augment, variant_train_samples, variant_replay_batch_size in variants:
            row = _run_variant(
                cfg,
                name=variant,
                train_samples=variant_train_samples,
                validation_samples=validation_samples,
                symmetry_validation_samples=symmetry_validation_samples,
                output_dir=args.output_dir,
                device=device,
                seed=args.seed,
                training_iterations=args.training_iterations,
                augment=augment,
                augmentation_prob=0.0,
                max_samples_per_shard=args.max_samples_per_shard,
                replay_batch_size=variant_replay_batch_size,
            )
            row["created_at_utc"] = _utc_now()
            row["selfplay_summary"] = selfplay_summary
            rows.append(row)
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            print(
                "[augment-bench] "
                f"variant={variant} train_sec={row['train_sec']:.2f} "
                f"eval_loss={row['eval_after']['loss']:.4f} "
                f"d4_eval_loss={row['symmetry_eval_after']['loss']:.4f} "
                f"d4_eval_gap={row['symmetry_gap_after']:.4f} "
                f"policy_js={row['symmetry_consistency_after']['policy_js']:.5f}",
                flush=True,
            )

    summary = _summarize(rows)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print("[augment-bench] summary", flush=True)
    for row in summary:
        print(
            "  "
            f"{row['variant']}: eval_loss={row['eval_loss_after_mean']:.4f} "
            f"d4_eval_loss={row['d4_eval_loss_after_mean']:.4f} "
            f"d4_vs_standard={row.get('d4_loss_after_vs_standard', 0.0):+.4f} "
            f"train_overhead={row.get('train_sec_vs_standard_pct', 0.0):+.1f}%"
        )
    print(f"[augment-bench] wrote jsonl={jsonl_path} summary_csv={summary_csv}", flush=True)


if __name__ == "__main__":
    main()
