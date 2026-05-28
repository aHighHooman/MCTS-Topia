from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
import sysconfig
import time
from pathlib import Path
from typing import Any, Dict, Iterable

import torch
from torch import nn

from nn.model import HybridPolicyValueNet, count_parameters
from search.device import move_optimizer_state, require_cuda_device
from .config import HybridAgentConfig
from .replay import ReplayStore, StepRecord
from .train import (
    _align_policy_targets_to_logits,
    _append_metrics,
    _format_seconds,
    _load_checkpoint,
    collate_batch,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_project_path(path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _tensorboard_executable() -> str | None:
    found = shutil.which("tensorboard")
    if found:
        return found
    scripts_dir = sysconfig.get_path("scripts")
    if scripts_dir:
        candidate = Path(scripts_dir) / ("tensorboard.exe" if sys.platform.startswith("win") else "tensorboard")
        if candidate.exists():
            return str(candidate)
    return None


def _stop_tensorboard_on_port(port: int) -> None:
    if not sys.platform.startswith("win"):
        return
    try:
        command = (
            f"Get-NetTCPConnection -LocalPort {int(port)} -State Listen -ErrorAction SilentlyContinue "
            "| Select-Object -ExpandProperty OwningProcess -Unique"
        )
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        for line in completed.stdout.splitlines():
            line = line.strip()
            if not line.isdigit():
                continue
            pid = int(line)
            try:
                proc = subprocess.run(
                    [
                        "powershell",
                        "-NoProfile",
                        "-Command",
                        f"(Get-CimInstance Win32_Process -Filter \"ProcessId={pid}\").CommandLine",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                if "tensorboard" not in proc.stdout.lower():
                    continue
                subprocess.run(
                    ["powershell", "-NoProfile", "-Command", f"Stop-Process -Id {pid} -Force"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
            except Exception:
                continue
    except Exception as exc:
        print(f"[static pretrain] warning: failed to stop existing tensorboard on port {port}: {exc}", flush=True)


def _prepare_tensorboard_logdir(cfg: HybridAgentConfig) -> None:
    logdir = cfg.static_pretrain.tensorboard_log_dir
    if logdir is None:
        return
    if cfg.static_pretrain.tensorboard_clear_on_start and logdir.exists():
        shutil.rmtree(logdir)
        print(f"[static pretrain] cleared tensorboard logdir={logdir}", flush=True)
    logdir.mkdir(parents=True, exist_ok=True)


def _start_tensorboard(cfg: HybridAgentConfig) -> subprocess.Popen[str] | None:
    if not cfg.static_pretrain.tensorboard_auto_start or cfg.static_pretrain.tensorboard_log_dir is None:
        return None
    if cfg.static_pretrain.tensorboard_restart_on_start:
        _stop_tensorboard_on_port(int(cfg.static_pretrain.tensorboard_port))
    executable = _tensorboard_executable()
    if executable is None:
        print("[static pretrain] tensorboard executable not found; live PNG plot is still enabled", flush=True)
        return None
    logdir = cfg.static_pretrain.tensorboard_log_dir
    logdir.mkdir(parents=True, exist_ok=True)
    command = [
        executable,
        "--logdir",
        str(logdir),
        "--port",
        str(int(cfg.static_pretrain.tensorboard_port)),
        "--reload_interval",
        "2",
    ]
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform.startswith("win") else 0,
        )
        print(
            f"[static pretrain] tensorboard started http://localhost:{cfg.static_pretrain.tensorboard_port} "
            f"logdir={logdir}",
            flush=True,
        )
        return process
    except Exception as exc:
        print(f"[static pretrain] warning: failed to start tensorboard: {type(exc).__name__}: {exc}", flush=True)
        return None


class LiveLossPlot:
    def __init__(
        self,
        plot_paths: dict[str, Path],
        update_interval_batches: int = 1,
        tensorboard_log_dir: Path | None = None,
    ) -> None:
        self.plot_paths = {name: Path(path) for name, path in plot_paths.items()}
        self.update_interval_batches = max(1, int(update_interval_batches))
        self.train_steps: list[int] = []
        self.val_steps: list[int] = []
        self.train_losses: dict[str, list[float]] = {name: [] for name in self.plot_paths}
        self.val_losses: dict[str, list[float]] = {name: [] for name in self.plot_paths}
        self.train_writer: Any | None = None
        self.val_writer: Any | None = None
        if tensorboard_log_dir is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.train_writer = SummaryWriter(str(tensorboard_log_dir / "train"))
                self.val_writer = SummaryWriter(str(tensorboard_log_dir / "validation"))
            except Exception as exc:
                print(f"[static pretrain] tensorboard disabled: {type(exc).__name__}: {exc}", flush=True)

    def add_train_batch(self, step: int, losses: dict[str, float]) -> None:
        self.train_steps.append(int(step))
        for name in self.plot_paths:
            self.train_losses[name].append(float(losses[name]))
        if self.train_writer is not None:
            for name, loss in losses.items():
                self.train_writer.add_scalar(f"loss/{name}", float(loss), int(step))
        if int(step) % self.update_interval_batches == 0:
            self.render()

    def add_validation_epoch(self, step: int, losses: dict[str, float], epoch: int) -> None:
        self.val_steps.append(int(step))
        for name in self.plot_paths:
            self.val_losses[name].append(float(losses[name]))
        if self.val_writer is not None:
            for name, loss in losses.items():
                self.val_writer.add_scalar(f"loss/{name}", float(loss), int(step))
        self.render()

    def _render_one(self, name: str, path: Path) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            titles = {
                "total": "Static Bootstrap Pretraining Total Loss",
                "policy": "Static Bootstrap Pretraining Policy Loss",
                "value": "Static Bootstrap Pretraining Value Loss",
            }
            ylabels = {
                "total": "Total loss",
                "policy": "Policy cross-entropy",
                "value": "Value MSE",
            }
            train_colors = {
                "total": "#0f62fe",
                "policy": "#7c3aed",
                "value": "#0891b2",
            }
            val_colors = {
                "total": "#ef4444",
                "policy": "#f59e0b",
                "value": "#16a34a",
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(12, 6.75), dpi=150)
            fig.patch.set_facecolor("#f6f8fb")
            ax.set_facecolor("#fffefa")
            if self.train_steps:
                ax.plot(
                    self.train_steps,
                    self.train_losses[name],
                    color=train_colors.get(name, "#0f62fe"),
                    linewidth=1.8,
                    alpha=0.92,
                    label="Training loss (per batch)",
                )
            if self.val_steps:
                ax.step(
                    self.val_steps,
                    self.val_losses[name],
                    where="post",
                    color=val_colors.get(name, "#ef4444"),
                    linewidth=2.8,
                    label="Validation loss (per epoch)",
                )
                ax.scatter(self.val_steps, self.val_losses[name], color=val_colors.get(name, "#ef4444"), s=34, zorder=3)
            ax.set_title(titles.get(name, name), fontsize=16, fontweight="bold", pad=14)
            ax.set_xlabel("Optimizer batch step", fontsize=11)
            ax.set_ylabel(ylabels.get(name, "Loss"), fontsize=11)
            ax.grid(True, which="major", color="#d7dee9", linewidth=0.95)
            ax.grid(True, which="minor", color="#edf2f7", linewidth=0.55)
            ax.minorticks_on()
            ax.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1")
            for spine in ax.spines.values():
                spine.set_color("#94a3b8")
            fig.tight_layout()
            tmp_path = path.with_suffix(path.suffix + ".tmp.png")
            fig.savefig(tmp_path)
            plt.close(fig)
            tmp_path.replace(path)
        except Exception as exc:
            print(f"[static pretrain] warning: failed to update {name} loss plot {path}: {exc}", flush=True)

    def render(self) -> None:
        for name, path in self.plot_paths.items():
            self._render_one(name, path)
        if self.train_writer is not None:
            self.train_writer.flush()
        if self.val_writer is not None:
            self.val_writer.flush()

    def close(self) -> None:
        self.render()
        if self.train_writer is not None:
            self.train_writer.close()
        if self.val_writer is not None:
            self.val_writer.close()


def _resolve_pretrain_device(cfg: HybridAgentConfig) -> torch.device:
    device = torch.device(str(cfg.static_pretrain.device))
    if device.type == "cuda":
        return require_cuda_device()
    return device


def _static_bootstrap_store(cfg: HybridAgentConfig) -> ReplayStore:
    return ReplayStore(
        cfg.static_pretrain.replay_dir,
        capacity_steps=0,
        shard_prefix=cfg.static_pretrain.shard_prefix,
        load_existing=False,
        cache_records=False,
    )


def _static_bootstrap_shards(cfg: HybridAgentConfig) -> list[Path]:
    shards = sorted(cfg.static_pretrain.replay_dir.glob(f"{cfg.static_pretrain.shard_prefix}_*.pt"))
    if not shards:
        raise RuntimeError(
            "Static pretraining requested, but no replay shards were found at "
            f"{cfg.static_pretrain.replay_dir} with prefix {cfg.static_pretrain.shard_prefix!r}."
        )
    return shards


def _record_goes_to_train(shard: Path, index: int, train_fraction: float, seed: int) -> bool:
    key = f"{seed}:{shard.name}:{index}"
    # Built-in hash is salted per process; random.Random with a string seed is stable enough for this split.
    return random.Random(key).random() < max(0.01, min(0.99, float(train_fraction)))


def _shard_step_count(store: ReplayStore, shard: Path) -> int:
    try:
        payload = torch.load(shard, map_location="cpu", weights_only=False)
    except Exception:
        return 0
    try:
        return int(payload.get("step_count", len(payload.get("records", []) or [])))
    finally:
        del payload


def _count_split_records(
    store: ReplayStore,
    shards: list[Path],
    train_fraction: float,
    seed: int,
) -> tuple[int, int]:
    train_count = 0
    val_count = 0
    for shard in shards:
        step_count = _shard_step_count(store, shard)
        for index in range(step_count):
            if _record_goes_to_train(shard, index, train_fraction, seed):
                train_count += 1
            else:
                val_count += 1
    return train_count, val_count


def _load_static_bootstrap_records(cfg: HybridAgentConfig) -> list[StepRecord]:
    """Small-corpus helper retained for tests; production pretraining streams shards."""
    store = ReplayStore(
        cfg.static_pretrain.replay_dir,
        capacity_steps=0,
        shard_prefix=cfg.static_pretrain.shard_prefix,
        cache_records=False,
    )
    shards = list(store.shards())
    if not shards:
        raise RuntimeError(
            "Static pretraining requested, but no replay shards were found at "
            f"{cfg.static_pretrain.replay_dir} with prefix {cfg.static_pretrain.shard_prefix!r}."
        )
    records = store.load_records()
    if not records:
        raise RuntimeError(
            "Static pretraining requested, but the selected replay shards contained no records: "
            f"shards={len(shards)} replay_dir={cfg.static_pretrain.replay_dir}."
        )
    print(
        f"[static pretrain] loaded records={len(records)} shards={len(shards)} "
        f"replay_dir={cfg.static_pretrain.replay_dir} shard_prefix={cfg.static_pretrain.shard_prefix}",
        flush=True,
    )
    return records


def _split_records(records: list[StepRecord], train_fraction: float, seed: int) -> tuple[list[StepRecord], list[StepRecord]]:
    if len(records) < 2:
        raise RuntimeError("Static pretraining needs at least two records to create a train/validation split.")
    rng = random.Random(int(seed))
    shuffled = list(records)
    rng.shuffle(shuffled)
    fraction = max(0.01, min(0.99, float(train_fraction)))
    train_count = int(round(len(shuffled) * fraction))
    train_count = max(1, min(len(shuffled) - 1, train_count))
    return shuffled[:train_count], shuffled[train_count:]


def _iter_batches(records: list[StepRecord], batch_size: int, *, shuffle: bool, rng: random.Random) -> Iterable[list[StepRecord]]:
    items = list(records)
    if shuffle:
        rng.shuffle(items)
    size = max(1, int(batch_size))
    for offset in range(0, len(items), size):
        yield items[offset : offset + size]


def _iter_split_batches(
    store: ReplayStore,
    shards: list[Path],
    cfg: HybridAgentConfig,
    *,
    split: str,
    shuffle: bool,
    rng: random.Random,
) -> Iterable[list[StepRecord]]:
    shard_order = list(shards)
    if shuffle:
        rng.shuffle(shard_order)
    batch_size = max(1, int(cfg.static_pretrain.batch_size))
    max_pending = max(batch_size, int(cfg.static_pretrain.max_records_in_memory))
    pending: list[StepRecord] = []
    want_train = split == "train"
    for shard in shard_order:
        _, records = store._load_shard(shard)
        indexes = list(range(len(records)))
        if shuffle:
            rng.shuffle(indexes)
        for index in indexes:
            is_train = _record_goes_to_train(
                shard,
                index,
                cfg.static_pretrain.train_fraction,
                cfg.static_pretrain.seed,
            )
            if is_train != want_train:
                continue
            pending.append(records[index])
            if len(pending) >= batch_size:
                yield pending[:batch_size]
                pending = pending[batch_size:]
            if len(pending) > max_pending:
                pending = pending[-batch_size:]
    if pending:
        yield pending


def _record_value_target(record: StepRecord, source: str) -> float:
    if source == "outcome":
        return float(record.value_target)
    if source == "static":
        static_value = getattr(record, "static_value_target", None)
        if static_value is not None:
            return float(static_value)
        return float(record.root_value)
    raise ValueError(f"Unsupported static pretrain value target source: {source!r}")


def _batch_losses(
    cfg: HybridAgentConfig,
    model: HybridPolicyValueNet,
    records: list[StepRecord],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = collate_batch(records, cfg)
    encoded = batch["encoded"].to(device)
    policy_targets = batch["policy_targets"].to(device)
    value_target_source = str(getattr(cfg.static_pretrain, "value_target_source", "outcome")).strip().lower()
    value_targets = torch.tensor(
        [_record_value_target(record, value_target_source) for record in records],
        dtype=torch.float32,
        device=device,
    )
    output = model(encoded)
    log_probs = torch.log_softmax(output.policy_logits, dim=-1)
    policy_targets = _align_policy_targets_to_logits(policy_targets, int(log_probs.shape[-1]))
    policy_targets = policy_targets / policy_targets.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    policy_loss = -(policy_targets * log_probs).sum(dim=-1).mean()
    value_loss = nn.functional.mse_loss(output.value.flatten(), value_targets)
    loss = (
        float(cfg.static_pretrain.policy_loss_weight) * policy_loss
        + float(cfg.static_pretrain.value_loss_weight) * value_loss
    )
    return loss, policy_loss, value_loss


def _run_epoch(
    cfg: HybridAgentConfig,
    model: HybridPolicyValueNet,
    store: ReplayStore,
    shards: list[Path],
    device: torch.device,
    *,
    split: str,
    optimizer: torch.optim.Optimizer | None,
    rng: random.Random,
    plot: LiveLossPlot | None = None,
    global_batch_step: int = 0,
    epoch: int = 0,
    expected_records: int = 0,
) -> tuple[Dict[str, float], int]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "steps": 0.0}
    local_batch = 0
    log_interval = max(1, int(cfg.static_pretrain.console_log_interval_batches))
    phase = "train" if training else "val"
    for batch_records in _iter_split_batches(
        store,
        shards,
        cfg,
        split=split,
        shuffle=training,
        rng=rng,
    ):
        if training:
            loss, policy_loss, value_loss = _batch_losses(cfg, model, batch_records, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.static_pretrain.gradient_clip)
            optimizer.step()
            global_batch_step += 1
            if plot is not None:
                plot.add_train_batch(
                    global_batch_step,
                    {
                        "total": float(loss.item()),
                        "policy": float(policy_loss.item()),
                        "value": float(value_loss.item()),
                    },
                )
        else:
            with torch.no_grad():
                loss, policy_loss, value_loss = _batch_losses(cfg, model, batch_records, device)
        count = float(len(batch_records))
        totals["loss"] += float(loss.item()) * count
        totals["policy_loss"] += float(policy_loss.item()) * count
        totals["value_loss"] += float(value_loss.item()) * count
        totals["steps"] += count
        local_batch += 1
        if local_batch == 1 or local_batch % log_interval == 0:
            avg_loss = totals["loss"] / max(1.0, totals["steps"])
            percent = 100.0 * totals["steps"] / max(1.0, float(expected_records))
            step_note = f" global_step={global_batch_step}" if training else ""
            print(
                f"[static pretrain epoch {epoch} {phase}] "
                f"batch={local_batch} records={int(totals['steps'])}/{expected_records} "
                f"({percent:.1f}%) loss={float(loss.item()):.4f} avg_loss={avg_loss:.4f}{step_note}",
                flush=True,
            )
    if totals["steps"] > 0:
        for key in ("loss", "policy_loss", "value_loss"):
            totals[key] /= totals["steps"]
    return totals, global_batch_step


def _save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: HybridPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    metrics: Dict[str, float],
    best_val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 2,
            "kind": "static_bootstrap_pretrain",
            "epoch": int(epoch),
            "iteration": int(epoch),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "best_val_loss": float(best_val_loss),
        },
        path,
    )


def pretrain_static(cfg: HybridAgentConfig, *, device: torch.device | None = None) -> None:
    if device is None:
        device = _resolve_pretrain_device(cfg)
    cfg.training.device = str(device)
    cfg.static_pretrain.device = str(device)
    cfg.static_pretrain.replay_dir = _resolve_project_path(cfg.static_pretrain.replay_dir).resolve()
    cfg.static_pretrain.checkpoint_dir = _resolve_project_path(cfg.static_pretrain.checkpoint_dir).resolve()
    cfg.static_pretrain.metrics_csv = _resolve_project_path(cfg.static_pretrain.metrics_csv).resolve()
    cfg.static_pretrain.plot_path = _resolve_project_path(cfg.static_pretrain.plot_path).resolve()
    cfg.static_pretrain.policy_plot_path = _resolve_project_path(cfg.static_pretrain.policy_plot_path).resolve()
    cfg.static_pretrain.value_plot_path = _resolve_project_path(cfg.static_pretrain.value_plot_path).resolve()
    if cfg.static_pretrain.tensorboard_log_dir is not None:
        cfg.static_pretrain.tensorboard_log_dir = _resolve_project_path(cfg.static_pretrain.tensorboard_log_dir).resolve()
    _prepare_tensorboard_logdir(cfg)

    startup_started_at = time.monotonic()
    store = _static_bootstrap_store(cfg)
    shards = _static_bootstrap_shards(cfg)
    shard_discovery_sec = time.monotonic() - startup_started_at
    print(
        f"[static pretrain] discovered shards={len(shards)} "
        f"time={_format_seconds(shard_discovery_sec)}",
        flush=True,
    )
    split_count_started_at = time.monotonic()
    train_count, val_count = _count_split_records(
        store,
        shards,
        cfg.static_pretrain.train_fraction,
        cfg.static_pretrain.seed,
    )
    split_count_sec = time.monotonic() - split_count_started_at
    if train_count <= 0 or val_count <= 0:
        raise RuntimeError(
            "Static pretraining split produced an empty subset: "
            f"train={train_count} val={val_count} train_fraction={cfg.static_pretrain.train_fraction}."
        )
    total_records = train_count + val_count
    print(
        f"[static pretrain] indexed records={total_records} shards={len(shards)} "
        f"train={train_count} val={val_count} replay_dir={cfg.static_pretrain.replay_dir} "
        f"time={_format_seconds(split_count_sec)}",
        flush=True,
    )
    model = HybridPolicyValueNet(cfg.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.static_pretrain.learning_rate),
        weight_decay=float(cfg.static_pretrain.weight_decay),
    )
    start_epoch = _load_checkpoint(model, cfg.static_pretrain.checkpoint_path, optimizer)
    move_optimizer_state(optimizer, device)

    print(
        f"[static pretrain] start checkpoint_epoch={start_epoch} params={count_parameters(model):,} "
        f"device={device} train={train_count} val={val_count} "
        f"value_target_source={cfg.static_pretrain.value_target_source} "
        f"split={cfg.static_pretrain.train_fraction:.2f}/{1.0 - cfg.static_pretrain.train_fraction:.2f} "
        f"batch={cfg.static_pretrain.batch_size} max_epochs={cfg.static_pretrain.max_epochs} "
        f"patience={cfg.static_pretrain.patience} max_records_in_memory={cfg.static_pretrain.max_records_in_memory}",
        flush=True,
    )

    rng = random.Random(int(cfg.static_pretrain.seed) + start_epoch)
    best_val_loss = float("inf")
    bad_epochs = 0
    max_epochs = max(1, int(cfg.static_pretrain.max_epochs))
    patience = max(0, int(cfg.static_pretrain.patience))
    min_delta = max(0.0, float(cfg.static_pretrain.min_delta))
    global_batch_step = 0
    tensorboard_process = _start_tensorboard(cfg)
    plot = LiveLossPlot(
        {
            "total": cfg.static_pretrain.plot_path,
            "policy": cfg.static_pretrain.policy_plot_path,
            "value": cfg.static_pretrain.value_plot_path,
        },
        update_interval_batches=cfg.static_pretrain.plot_update_interval_batches,
        tensorboard_log_dir=cfg.static_pretrain.tensorboard_log_dir,
    )

    try:
        for local_epoch in range(max_epochs):
            epoch = start_epoch + local_epoch + 1
            started_at = time.monotonic()
            train_metrics, global_batch_step = _run_epoch(
                cfg,
                model,
                store,
                shards,
                device,
                split="train",
                optimizer=optimizer,
                rng=rng,
                plot=plot,
                global_batch_step=global_batch_step,
                epoch=epoch,
                expected_records=train_count,
            )
            val_metrics, global_batch_step = _run_epoch(
                cfg,
                model,
                store,
                shards,
                device,
                split="val",
                optimizer=None,
                rng=rng,
                global_batch_step=global_batch_step,
                epoch=epoch,
                expected_records=val_count,
            )
            plot.add_validation_epoch(
                global_batch_step,
                {
                    "total": val_metrics["loss"],
                    "policy": val_metrics["policy_loss"],
                    "value": val_metrics["value_loss"],
                },
                epoch,
            )
            improved = val_metrics["loss"] < best_val_loss - min_delta
            if improved:
                best_val_loss = val_metrics["loss"]
                bad_epochs = 0
            else:
                bad_epochs += 1

            row = {
                "epoch": epoch,
                "global_batch_step": global_batch_step,
                "train_records": train_count,
                "val_records": val_count,
                "train_loss": train_metrics["loss"],
                "train_policy_loss": train_metrics["policy_loss"],
                "train_value_loss": train_metrics["value_loss"],
                "val_loss": val_metrics["loss"],
                "val_policy_loss": val_metrics["policy_loss"],
                "val_value_loss": val_metrics["value_loss"],
                "best_val_loss": best_val_loss,
                "bad_epochs": bad_epochs,
                "improved": int(improved),
                "epoch_sec": time.monotonic() - started_at,
            }
            _append_metrics(cfg.static_pretrain.metrics_csv, row)
            checkpoint_metrics = {
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            _save_checkpoint(
                cfg.static_pretrain.checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                metrics=checkpoint_metrics,
                best_val_loss=best_val_loss,
            )
            if improved:
                _save_checkpoint(
                    cfg.static_pretrain.best_checkpoint_path(),
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    metrics=checkpoint_metrics,
                    best_val_loss=best_val_loss,
                )
            print(
                f"[static pretrain epoch {epoch}] train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} best_val={best_val_loss:.4f} "
                f"bad_epochs={bad_epochs}/{patience} plot={cfg.static_pretrain.plot_path} "
                f"time={_format_seconds(row['epoch_sec'])}",
                flush=True,
            )
            if patience > 0 and bad_epochs >= patience:
                print(f"[static pretrain] early stop epoch={epoch} best_val_loss={best_val_loss:.4f}", flush=True)
                break
    finally:
        plot.close()
        if tensorboard_process is not None and tensorboard_process.poll() is not None:
            print("[static pretrain] tensorboard process exited during training", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain the policy/value model from static bootstrap replay.")
    parser.add_argument(
        "--value-target-source",
        choices=("outcome", "static"),
        default="outcome",
        help="Train the value head against match returns or per-node static evaluator values.",
    )
    parser.add_argument(
        "--static-value-targets",
        action="store_true",
        help="Alias for --value-target-source static.",
    )
    args = parser.parse_args()

    cfg = HybridAgentConfig()
    cfg.static_pretrain.value_target_source = "static" if args.static_value_targets else args.value_target_source
    pretrain_static(cfg)


if __name__ == "__main__":
    main()
