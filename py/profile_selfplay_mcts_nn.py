from __future__ import annotations

import argparse
import cProfile
import csv
import json
import socket
import subprocess
import re
import statistics
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch

PY_ROOT = Path(__file__).resolve().parent
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from tribes_rl.config import HybridAgentConfig
from tribes_rl.replay import ReplayStore
from tribes_rl.selfplay import run_selfplay
from tribes_rl.train import (
    _persistent_bridge_command,
    _shutdown_persistent_server,
    _wait_for_port,
)


PROFILE_PATTERN = re.compile(
    r"\[tribes_rl\.profile\]\s+choose_action\s+"
    r"encode_ms=(?P<encode_ms>[-+0-9.]+)\s+"
    r"policy_ms=(?P<policy_ms>[-+0-9.]+)\s+"
    r"search_ms=(?P<search_ms>[-+0-9.]+)\s+"
    r"total_ms=(?P<total_ms>[-+0-9.]+)\s+"
    r"actions=(?P<actions>\d+)\s+"
    r"device=(?P<device>\S+)\s+"
    r"(?:action_budget_sec|turn_budget_remaining_sec)=(?P<action_budget_sec>[-+0-9.]+)"
)
SEARCH_PATTERN = re.compile(
    r"\[tribes_rl\.search_profile\]\s+mcts\s+"
    r"sims=(?P<sims>\d+)\s+"
    r"batch=(?P<batch>\d+)\s+"
    r"max_depth=(?P<max_depth>-?\d+)\s+"
    r"(?:paths=(?P<paths>\d+)\s+)?"
    r"eval_batches=(?P<eval_batches>\d+)\s+"
    r"eval_positions=(?P<eval_positions>\d+)\s+"
    r"(?:eval_cache_hits=\d+\s+)?"
    r"(?:eval_cache_size=\d+\s+)?"
    r"select_ms=(?P<select_ms>[-+0-9.]+)\s+"
    r"eval_ms=(?P<eval_ms>[-+0-9.]+)\s+"
    r"expand_ms=(?P<expand_ms>[-+0-9.]+)\s+"
    r"total_inner_ms=(?P<total_inner_ms>[-+0-9.]+)"
)
WARMUP_PATTERN = re.compile(r"\[tribes_rl\.warmup\]\s+warmup_ms=(?P<warmup_ms>[-+0-9.]+)")
FALLBACK_PATTERN = re.compile(r"(invalid[-_ ]action|fallback)", re.IGNORECASE)


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


def _format_seconds(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.2f}s"
    minutes, rem = divmod(seconds, 60.0)
    return f"{int(minutes)}m{rem:04.1f}s"


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return float(ordered[index])


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": float(len(values)),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "p95": _percentile(values, 95.0),
        "max": float(max(values)),
    }


def _parse_profile(stderr: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[float]]:
    action_rows: list[dict[str, Any]] = []
    search_rows: list[dict[str, Any]] = []
    warmup_ms: list[float] = []
    for line in stderr.splitlines():
        match = PROFILE_PATTERN.search(line)
        if match:
            row: dict[str, Any] = {}
            for key, value in match.groupdict().items():
                row[key] = int(value) if key == "actions" else value if key == "device" else float(value)
            action_rows.append(row)
            continue
        match = SEARCH_PATTERN.search(line)
        if match:
            row = {}
            for key, value in match.groupdict().items():
                if value is None:
                    row[key] = 0
                else:
                    row[key] = int(value) if key in {"sims", "batch", "max_depth", "paths", "eval_batches", "eval_positions"} else float(value)
            search_rows.append(row)
            continue
        match = WARMUP_PATTERN.search(line)
        if match:
            warmup_ms.append(float(match.group("warmup_ms")))
    return action_rows, search_rows, warmup_ms


def _new_replay_step_count(replay_dir: Path, prefix: str, before: set[Path]) -> int:
    store = ReplayStore(replay_dir, capacity_steps=0, shard_prefix=prefix)
    total = 0
    for shard in store.shards():
        if shard in before:
            continue
        try:
            payload, records = store._load_shard(shard)
        except Exception:
            continue
        total += int(payload.get("step_count", len(records)))
    return total


def _bot_command(args: argparse.Namespace, cfg: HybridAgentConfig, workdir: Path) -> list[str]:
    command = [
        sys.executable,
        str(PY_ROOT / "hybrid_rl_bot.py"),
        "--checkpoint",
        str(args.checkpoint),
        "--replay-dir",
        str(cfg.replay.replay_dir),
        "--simulations",
        str(cfg.search.num_simulations),
        "--max-depth",
        str(cfg.search.max_depth),
        "--top-k-actions",
        str(cfg.search.top_k_actions),
        "--search-batch-size",
        str(cfg.search.batch_size),
        "--max-game-actions",
        str(cfg.selfplay.max_actions_per_game),
        "--wall-clock-per-action-seconds",
        str(cfg.selfplay.wall_clock_per_action_seconds),
    ]
    if args.deterministic:
        command.append("--deterministic")
    return command


def _format_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "(no rows)"
    widths = {
        key: max(len(label), *(len(str(row.get(key, ""))) for row in rows))
        for key, label in columns
    }
    header = "  ".join(label.ljust(widths[key]) for key, label in columns)
    divider = "  ".join("-" * widths[key] for key, _label in columns)
    body = ["  ".join(str(row.get(key, "")).ljust(widths[key]) for key, _label in columns) for row in rows]
    return "\n".join([header, divider, *body])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["name"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _profile_rows(profile_path: Path, *, root: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not profile_path.exists():
        return []
    import pstats

    stats = pstats.Stats(str(profile_path))
    entries = []
    for (filename, line, func_name), stat in stats.stats.items():
        primitive_calls, total_calls, total_time, cumulative_time, _callers = stat
        if total_time <= 0.0 and cumulative_time <= 0.0:
            continue
        try:
            display_file = str(Path(filename).resolve().relative_to(root))
        except (OSError, ValueError):
            display_file = filename
        entries.append(
            {
                "function": f"{display_file}:{line}:{func_name}",
                "primitive_calls": primitive_calls,
                "calls": total_calls,
                "self_ms": total_time * 1000.0,
                "cum_ms": cumulative_time * 1000.0,
                "avg_self_us": total_time * 1_000_000.0 / max(1, primitive_calls),
            }
        )
    entries.sort(key=lambda row: row["cum_ms"], reverse=True)
    rows = [
        {
            "function": row["function"],
            "primitive_calls": row["primitive_calls"],
            "calls": row["calls"],
            "self_ms": f"{row['self_ms']:.3f}",
            "cum_ms": f"{row['cum_ms']:.3f}",
            "avg_self_us": f"{row['avg_self_us']:.3f}",
        }
        for row in entries
    ]
    return rows if limit is None else rows[:limit]


def _find_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start_profiled_persistent_bot_server(
    cfg: HybridAgentConfig,
    bot_server_script: Path,
    checkpoint_path: Path,
    replay_dir: Path,
    log_dir: Path,
    profile_output: Path,
) -> tuple[subprocess.Popen[str], str, int, Path, Path]:
    host = "127.0.0.1"
    port = _find_free_local_port()
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"persistent_bot_{port}.stdout.log"
    stderr_path = log_dir / f"persistent_bot_{port}.stderr.log"
    command = [
        sys.executable,
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
        "--wall-clock-per-action-seconds",
        str(max(0.0, float(cfg.selfplay.wall_clock_per_action_seconds))),
        "--profile-output",
        str(profile_output),
    ]
    if cfg.selfplay.profile_selfplay:
        command.append("--profile-selfplay")
    stdout_handle = stdout_path.open("w", encoding="utf-8")
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    process = subprocess.Popen(command, stdout=stdout_handle, stderr=stderr_handle, text=True)
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


def _read_new_text(paths: list[Path], offsets: dict[Path, int]) -> str:
    parts: list[str] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        offset = offsets.get(path, 0)
        if offset < len(text):
            parts.append(text[offset:])
            offsets[path] = len(text)
    return "\n".join(parts)


def _configure(args: argparse.Namespace) -> HybridAgentConfig:
    cfg = HybridAgentConfig()
    cfg.search.num_simulations = int(args.simulations)
    cfg.search.batch_size = int(args.search_batch_size)
    cfg.search.max_depth = int(args.max_depth)
    cfg.search.top_k_actions = int(args.top_k_actions)
    cfg.search.seed = int(args.seed_base)
    if args.no_dirichlet:
        cfg.search.dirichlet_epsilon = 0.0
    cfg.selfplay.profile_selfplay = True
    cfg.selfplay.persistent_bot = bool(args.persistent_bot)
    cfg.selfplay.run_mode = str(args.run_mode)
    cfg.selfplay.game_mode = str(args.game_mode)
    cfg.selfplay.level_file = str(args.level_file)
    cfg.selfplay.map_type = str(args.map_type)
    cfg.selfplay.map_size = str(args.map_size)
    cfg.selfplay.max_turns_capitals = int(args.max_turns_capitals)
    cfg.selfplay.max_actions_per_turn = int(args.max_actions_per_turn)
    cfg.selfplay.max_actions_per_game = int(args.max_actions_per_game)
    cfg.selfplay.adjudicate_incomplete_games = bool(getattr(args, "adjudicate_incomplete_games", False))
    if getattr(args, "adjudicator_bot", None):
        cfg.selfplay.adjudicator_bot = str(args.adjudicator_bot)
    cfg.selfplay.adjudication_max_turns_capitals = int(getattr(args, "adjudication_max_turns_capitals", cfg.selfplay.adjudication_max_turns_capitals))
    cfg.selfplay.adjudication_max_actions_per_game = int(getattr(args, "adjudication_max_actions_per_game", cfg.selfplay.adjudication_max_actions_per_game))
    cfg.selfplay.timeout_seconds = int(args.match_timeout_seconds)
    cfg.selfplay.external_action_timeout_ms = int(args.external_action_timeout_ms)
    cfg.selfplay.progress_interval_seconds = int(args.progress_interval_seconds)
    cfg.selfplay.wall_clock_per_action_seconds = float(args.wall_clock_per_action_seconds)
    if args.java_executable:
        cfg.selfplay.java_executable = str(args.java_executable)
    if args.java_classpath:
        cfg.selfplay.java_classpath = str(args.java_classpath)
    if args.java_main_class:
        cfg.selfplay.java_main_class = str(args.java_main_class)
    cfg.replay.replay_dir = args.output_dir / "replay"
    cfg.training.output_dir = args.output_dir
    cfg.training.checkpoint_path = args.checkpoint
    return cfg


def _game_summary(index: int, seed: int, elapsed_sec: float, returncode: int, replay_steps: int, stderr: str) -> dict[str, Any]:
    action_rows, search_rows, warmup_ms = _parse_profile(stderr)
    total_ms = [float(row["total_ms"]) for row in action_rows]
    encode_ms = [float(row["encode_ms"]) for row in action_rows]
    policy_ms = [float(row["policy_ms"]) for row in action_rows]
    search_ms = [float(row["search_ms"]) for row in action_rows]
    action_counts = [float(row["actions"]) for row in action_rows]
    configured_sims = [float(row["sims"]) for row in search_rows]
    paths = [float(row.get("paths", row["sims"])) for row in search_rows]
    eval_positions = [float(row["eval_positions"]) for row in search_rows]
    total_configured_simulations = int(sum(configured_sims))
    total_simulations = int(sum(paths))
    fallback_count = len([line for line in stderr.splitlines() if FALLBACK_PATTERN.search(line)])
    return {
        "game": index,
        "seed": seed,
        "returncode": returncode,
        "elapsed_sec": elapsed_sec,
        "actions_profiled": len(action_rows),
        "replay_steps": replay_steps,
        "actions_per_sec": len(action_rows) / elapsed_sec if elapsed_sec > 0.0 else 0.0,
        "replay_steps_per_sec": replay_steps / elapsed_sec if elapsed_sec > 0.0 else 0.0,
        "simulations": total_simulations,
        "simulations_per_sec": total_simulations / elapsed_sec if elapsed_sec > 0.0 else 0.0,
        "configured_simulations": total_configured_simulations,
        "mean_legal_actions": statistics.fmean(action_counts) if action_counts else 0.0,
        "encode_mean_ms": _summarize(encode_ms)["mean"],
        "policy_mean_ms": _summarize(policy_ms)["mean"],
        "search_mean_ms": _summarize(search_ms)["mean"],
        "total_mean_ms": _summarize(total_ms)["mean"],
        "total_p95_ms": _summarize(total_ms)["p95"],
        "total_max_ms": _summarize(total_ms)["max"],
        "search_profile_rows": len(search_rows),
        "eval_positions": int(sum(eval_positions)),
        "eval_positions_per_sec": sum(eval_positions) / elapsed_sec if elapsed_sec > 0.0 else 0.0,
        "warmup_ms": statistics.fmean(warmup_ms) if warmup_ms else 0.0,
        "fallback_count": fallback_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile full Tribes self-play games using the MCTS neural-network bot."
    )
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=12345)
    parser.add_argument("--checkpoint", type=Path, default=Path("rl/checkpoints/latest.pt"))
    parser.add_argument("--device", default=None, help="Device metadata passed to the self-play wrapper. Defaults to cuda if available, otherwise cpu.")
    parser.add_argument("--output-dir", type=Path, default=Path("debug-logs/selfplay-mcts-nn-profile"))
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--tribes", nargs=2, default=["Xin Xi", "Imperius"])
    parser.add_argument("--run-mode", default="PlayLG")
    parser.add_argument("--game-mode", default="Capitals")
    parser.add_argument("--level-file", default="levels/MinimalLevel2.csv")
    parser.add_argument("--map-type", default="Continents")
    parser.add_argument("--map-size", default="Small")
    parser.add_argument("--simulations", type=int, default=192)
    parser.add_argument("--search-batch-size", type=int, default=64)
    parser.add_argument("--max-depth", type=int, default=0)
    parser.add_argument("--top-k-actions", type=int, default=64)
    parser.add_argument("--max-turns-capitals", type=int, default=20)
    parser.add_argument("--max-actions-per-turn", type=int, default=80)
    parser.add_argument("--max-actions-per-game", type=int, default=128)
    parser.add_argument("--adjudicate-incomplete-games", action="store_true")
    parser.add_argument("--adjudicator-bot", default="py/bots/simple_bot.py")
    parser.add_argument("--adjudication-max-turns-capitals", type=int, default=80)
    parser.add_argument("--adjudication-max-actions-per-game", type=int, default=2048)
    parser.add_argument("--match-timeout-seconds", type=int, default=600)
    parser.add_argument("--external-action-timeout-ms", type=int, default=120000)
    parser.add_argument("--progress-interval-seconds", type=int, default=30)
    parser.add_argument("--wall-clock-per-action-seconds", type=float, default=None)
    parser.add_argument("--wall-clock-per-turn-seconds", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--persistent-bot", dest="persistent_bot", action="store_true", default=True, help="Use the same persistent shared bot server setup as RL training.")
    parser.add_argument("--no-persistent-bot", dest="persistent_bot", action="store_false", help="Launch one full bot process per player, matching the older non-persistent path.")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-dirichlet", action="store_true")
    parser.add_argument("--java-executable", default=None)
    parser.add_argument("--java-classpath", default=None)
    parser.add_argument("--java-main-class", default=None)
    parser.add_argument("--game-csv", type=Path, default=None)
    parser.add_argument("--action-csv", type=Path, default=None)
    parser.add_argument("--search-csv", type=Path, default=None)
    parser.add_argument("--function-csv", type=Path, default=None)
    parser.add_argument("--profile-stats", type=Path, default=None, help="Raw cProfile .prof output for the persistent bot process.")
    parser.add_argument("--function-limit", type=int, default=40)
    parser.add_argument("--jsonl", type=Path, default=None)
    args = parser.parse_args()
    if args.wall_clock_per_action_seconds is None:
        args.wall_clock_per_action_seconds = (
            args.wall_clock_per_turn_seconds
            if args.wall_clock_per_turn_seconds is not None
            else 2.0
        )

    workdir = Path(args.workdir).resolve() if args.workdir is not None else PY_ROOT.parent.resolve()
    args.output_dir = Path(args.output_dir).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = _configure(args)
    cfg.replay.replay_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    replay_store = ReplayStore(cfg.replay.replay_dir, cfg.replay.capacity_steps, cfg.replay.shard_prefix)

    command = _bot_command(args, cfg, workdir)
    bot_commands = [command, list(command)]
    game_rows: list[dict[str, Any]] = []
    all_action_rows: list[dict[str, Any]] = []
    all_search_rows: list[dict[str, Any]] = []
    profile = cProfile.Profile()
    benchmark_started_at = time.perf_counter()
    persistent_process = None
    persistent_host = "127.0.0.1"
    persistent_port = 0
    persistent_log_paths: list[Path] = []
    persistent_offsets: dict[Path, int] = {}
    bot_profile_path = args.profile_stats or (args.output_dir / "persistent_bot.prof")

    print(
        "Self-play MCTS NN profile: "
        f"games={args.games} sims={cfg.search.num_simulations} batch={cfg.search.batch_size} "
        f"map={cfg.selfplay.map_type}/{cfg.selfplay.map_size} max_turns={cfg.selfplay.max_turns_capitals} "
        f"persistent_bot={cfg.selfplay.persistent_bot} checkpoint={args.checkpoint}",
        flush=True,
    )

    try:
        if cfg.selfplay.persistent_bot:
            bot_server_script = workdir / "py" / "persistent_bot_server.py"
            bot_bridge_script = workdir / "py" / "persistent_bot_bridge.py"
            persistent_process, persistent_host, persistent_port, stdout_path, stderr_path = _start_profiled_persistent_bot_server(
                cfg,
                bot_server_script,
                args.checkpoint,
                cfg.replay.replay_dir,
                args.output_dir / "persistent_bot_logs",
                bot_profile_path,
            )
            persistent_log_paths = [stdout_path, stderr_path]
            persistent_offsets = {path: len(path.read_text(encoding="utf-8", errors="replace")) for path in persistent_log_paths if path.exists()}
            bridge = _persistent_bridge_command(bot_bridge_script, persistent_host, persistent_port)
            bot_commands = [bridge, list(bridge)]
            print(f"[persistent bot] started host={persistent_host} port={persistent_port}", flush=True)

        profile.enable()
        for index in range(1, max(1, int(args.games)) + 1):
            seed = int(args.seed_base) + index - 1
            cfg.selfplay.game_seed = seed
            cfg.selfplay.agent_seed = seed
            cfg.selfplay.level_seed = seed
            before_shards = set(ReplayStore(cfg.replay.replay_dir, 0, cfg.replay.shard_prefix).shards())
            started_at = time.perf_counter()
            print(f"[game {index}/{args.games}] seed={seed}", flush=True)
            result = run_selfplay(
                cfg,
                bot_commands,
                list(args.tribes),
                workdir,
                checkpoint_path=args.checkpoint,
                replay_store=replay_store,
                device=device,
                progress_label=f"profile-selfplay-game-{index}",
            )
            elapsed_sec = time.perf_counter() - started_at
            profile_text = result.stderr
            if cfg.selfplay.persistent_bot:
                profile_text = "\n".join([profile_text, _read_new_text(persistent_log_paths, persistent_offsets)])
            replay_steps = _new_replay_step_count(cfg.replay.replay_dir, cfg.replay.shard_prefix, before_shards)
            action_rows, search_rows, _warmup_ms = _parse_profile(profile_text)
            for row in action_rows:
                all_action_rows.append({"game": index, "seed": seed, **row})
            for row in search_rows:
                all_search_rows.append({"game": index, "seed": seed, **row})
            row = _game_summary(index, seed, elapsed_sec, int(result.returncode), replay_steps, profile_text)
            game_rows.append(row)
            print(
                "  "
                f"elapsed={_format_seconds(elapsed_sec)} returncode={result.returncode} "
                f"actions={row['actions_profiled']} actions/s={row['actions_per_sec']:.2f} "
                f"sims/s={row['simulations_per_sec']:.1f} "
                f"eval_pos/s={row['eval_positions_per_sec']:.1f} "
                f"avg_ms encode={row['encode_mean_ms']:.1f} policy={row['policy_mean_ms']:.1f} "
                f"search={row['search_mean_ms']:.1f} total={row['total_mean_ms']:.1f} "
                f"p95_total={row['total_p95_ms']:.1f}",
                flush=True,
            )
            if result.returncode != 0:
                print(profile_text[-4000:], file=sys.stderr, flush=True)
                break
        profile.disable()
    finally:
        if profile.getstats():
            try:
                profile.disable()
            except RuntimeError:
                pass
        if persistent_process is not None:
            _shutdown_persistent_server(persistent_host, persistent_port, persistent_process)
            print("[persistent bot] stopped", flush=True)

    total_elapsed = time.perf_counter() - benchmark_started_at
    total_actions = sum(int(row["actions_profiled"]) for row in game_rows)
    total_sims = sum(int(row["simulations"]) for row in game_rows)
    total_replay_steps = sum(int(row["replay_steps"]) for row in game_rows)
    print(
        "\nAggregate: "
        f"games={len(game_rows)} elapsed={_format_seconds(total_elapsed)} "
        f"actions={total_actions} actions/s={(total_actions / total_elapsed) if total_elapsed > 0 else 0.0:.2f} "
        f"replay_steps={total_replay_steps} "
        f"simulations={total_sims} sims/s={(total_sims / total_elapsed) if total_elapsed > 0 else 0.0:.1f}"
    )

    print("\nPer-game throughput")
    display_rows = [
        {
            "game": row["game"],
            "seed": row["seed"],
            "sec": f"{row['elapsed_sec']:.2f}",
            "actions": row["actions_profiled"],
            "actions/s": f"{row['actions_per_sec']:.2f}",
            "sims/s": f"{row['simulations_per_sec']:.1f}",
            "eval_pos/s": f"{row['eval_positions_per_sec']:.1f}",
            "encode_ms": f"{row['encode_mean_ms']:.1f}",
            "policy_ms": f"{row['policy_mean_ms']:.1f}",
            "search_ms": f"{row['search_mean_ms']:.1f}",
            "total_ms": f"{row['total_mean_ms']:.1f}",
            "p95_ms": f"{row['total_p95_ms']:.1f}",
            "fallbacks": row["fallback_count"],
        }
        for row in game_rows
    ]
    print(
        _format_table(
            display_rows,
            [
                ("game", "#"),
                ("seed", "seed"),
                ("sec", "sec"),
                ("actions", "actions"),
                ("actions/s", "actions/s"),
                ("sims/s", "sims/s"),
                ("eval_pos/s", "eval_pos/s"),
                ("encode_ms", "enc_ms"),
                ("policy_ms", "nn_ms"),
                ("search_ms", "mcts_ms"),
                ("total_ms", "total_ms"),
                ("p95_ms", "p95_ms"),
                ("fallbacks", "fallbacks"),
            ],
        )
    )

    timing_rows = []
    for key, label in [
        ("encode_ms", "encode"),
        ("policy_ms", "policy_forward"),
        ("search_ms", "mcts_search"),
        ("total_ms", "choose_action_total"),
    ]:
        values = [float(row[key]) for row in all_action_rows]
        summary = _summarize(values)
        timing_rows.append(
            {
                "name": label,
                "calls": int(summary["count"]),
                "mean_ms": f"{summary['mean']:.3f}",
                "median_ms": f"{summary['median']:.3f}",
                "p95_ms": f"{summary['p95']:.3f}",
                "max_ms": f"{summary['max']:.3f}",
                "pct_action_time": f"{(sum(values) / max(0.001, sum(float(row['total_ms']) for row in all_action_rows)) * 100.0):.1f}",
            }
        )
    print("\nAction timing breakdown")
    print(_format_table(timing_rows, [("name", "name"), ("calls", "calls"), ("mean_ms", "mean_ms"), ("median_ms", "median_ms"), ("p95_ms", "p95_ms"), ("max_ms", "max_ms"), ("pct_action_time", "% action")]))

    all_function_rows = _profile_rows(bot_profile_path, root=workdir)
    function_rows = all_function_rows[: max(1, int(args.function_limit))]
    project_function_rows = [
        row for row in all_function_rows
        if str(row.get("function", "")).startswith("py\\") or str(row.get("function", "")).startswith("py/")
    ][: max(1, int(args.function_limit))]
    print("\nTop persistent-bot functions by cumulative time")
    print(
        _format_table(
            function_rows,
            [
                ("function", "function"),
                ("calls", "calls"),
                ("cum_ms", "cum_ms"),
                ("self_ms", "self_ms"),
                ("avg_self_us", "avg_self_us"),
            ],
        )
    )
    print("\nTop project functions by cumulative time")
    print(
        _format_table(
            project_function_rows,
            [
                ("function", "function"),
                ("calls", "calls"),
                ("cum_ms", "cum_ms"),
                ("self_ms", "self_ms"),
                ("avg_self_us", "avg_self_us"),
            ],
        )
    )

    args.game_csv = args.game_csv or args.output_dir / "selfplay_games.csv"
    args.action_csv = args.action_csv or args.output_dir / "action_timings.csv"
    args.search_csv = args.search_csv or args.output_dir / "search_timings.csv"
    args.function_csv = args.function_csv or args.output_dir / "function_profile.csv"
    args.jsonl = args.jsonl or args.output_dir / "profile_runs.jsonl"
    _write_csv(args.game_csv, game_rows)
    _write_csv(args.action_csv, all_action_rows)
    _write_csv(args.search_csv, all_search_rows)
    _write_csv(args.function_csv, all_function_rows)
    _write_jsonl(
        args.jsonl,
        [
            {
                "schema_version": 1,
                "elapsed_sec": total_elapsed,
                "config": _jsonable(cfg),
                "games": game_rows,
                "timing_summary": timing_rows,
                "function_profile_top": function_rows,
                "project_function_profile_top": project_function_rows,
                "profile_stats": str(bot_profile_path),
                "cprofile_total_calls": sum(stat[1] for stat in profile.getstats()),
            }
        ],
    )
    print(f"\nWrote game CSV: {args.game_csv}")
    print(f"Wrote action CSV: {args.action_csv}")
    print(f"Wrote search CSV: {args.search_csv}")
    print(f"Wrote function CSV: {args.function_csv}")
    print(f"Wrote raw cProfile stats: {bot_profile_path}")
    print(f"Wrote JSONL: {args.jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
