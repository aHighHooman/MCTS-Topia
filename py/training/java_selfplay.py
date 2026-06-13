from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Sequence

from .config import HybridAgentConfig


def _terminate_process_tree(process: subprocess.Popen[str], grace_seconds: float = 3.0) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        try:
            os.killpg(process.pid, 15)
        except OSError:
            process.terminate()
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            try:
                os.killpg(process.pid, 9)
            except OSError:
                process.kill()
        else:
            process.kill()
        process.wait(timeout=grace_seconds)


def _tail_file(path: Path, lines: int = 40) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return "\n".join(text.splitlines()[-lines:])


def _external_log_tail(log_dir: Path, lines: int = 40) -> str:
    parts: list[str] = []
    for path in sorted(log_dir.glob("*.stderr.log")):
        tail = _tail_file(path, lines)
        if tail:
            parts.append(f"--- {path.name} ---\n{tail}")
    return "\n".join(parts)


def _external_log_text(log_dir: Path) -> str:
    parts: list[str] = []
    for path in sorted(log_dir.glob("*.stderr.log")):
        text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        if text:
            parts.append(f"--- {path.name} ---\n{text}")
    return "\n".join(parts)


def _resolve_java_executable(config: HybridAgentConfig) -> str:
    if config.selfplay.java_executable:
        return config.selfplay.java_executable
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        candidate = Path(java_home) / "bin" / ("java.exe" if os.name == "nt" else "java")
        if candidate.exists():
            return str(candidate)
    if os.name == "nt":
        jdk21_candidate = Path.home() / "AppData" / "Local" / "Programs" / "Java" / "jdk-21" / "bin" / "java.exe"
        if jdk21_candidate.exists():
            return str(jdk21_candidate)
    return "java"


def run_selfplay_match(
    config: HybridAgentConfig,
    bot_commands: Sequence[Sequence[str]],
    tribes: Sequence[str],
    workdir: Path,
    progress_label: str | None = None,
) -> subprocess.CompletedProcess[str]:
    play_config = {
        "Run Mode": config.selfplay.run_mode,
        "Game Mode": config.selfplay.game_mode,
        "Map Type": config.selfplay.map_type,
        "Map Size": config.selfplay.map_size,
        "Players": ["External" for _ in bot_commands],
        "External Commands": [list(command) for command in bot_commands],
        "Tribes": list(tribes),
        "Verbose": False,
        "Rollouts": config.selfplay.rollouts,
        "Force End": config.selfplay.force_end,
        "Population Size": config.selfplay.population_size,
        "Progressive Bias": config.selfplay.progressive_bias,
        "Pruning": config.selfplay.pruning,
        "K init mult": config.selfplay.k_init_mult,
        "T mult": config.selfplay.t_mult,
        "A mult": config.selfplay.a_mult,
        "B": config.selfplay.b_mult,
        "Game Seed": str(config.selfplay.game_seed),
        "Agents Seed": str(config.selfplay.agent_seed),
        "Level Seed": str(config.selfplay.level_seed),
        "Max Turns Capitals": max(1, int(config.selfplay.max_turns_capitals)),
        "Max Actions Per Turn": max(1, int(config.selfplay.max_actions_per_turn)),
        "Max Actions Per Game": max(1, int(config.selfplay.max_actions_per_game)),
        "Adjudicate Incomplete Games": bool(config.selfplay.adjudicate_incomplete_games),
        "Adjudicator Bot": str(config.selfplay.adjudicator_bot),
        "Adjudication Max Turns Capitals": max(1, int(config.selfplay.adjudication_max_turns_capitals)),
        "Adjudication Max Actions Per Game": max(1, int(config.selfplay.adjudication_max_actions_per_game)),
        "External Startup Timeout Ms": max(1, int(config.selfplay.external_startup_timeout_ms)),
        "External Action Timeout Ms": max(1, int(config.selfplay.external_action_timeout_ms)),
        "External Shutdown Timeout Ms": max(1, int(config.selfplay.external_shutdown_timeout_ms)),
        "External Match Timeout Ms": max(1, int(config.selfplay.timeout_seconds)) * 1000,
    }
    external_env: dict[str, str] = {}
    if config.selfplay.profile_selfplay:
        external_env["MCTS_NN_PROFILE"] = "1"
    if external_env:
        play_config["External Env"] = external_env
    playfile_root = (Path(config.training.output_dir) / "playfiles").resolve()
    playfile_root.mkdir(parents=True, exist_ok=True)
    run_id = f"{int(time.time() * 1000)}_{uuid.uuid4().hex}"
    play_path = playfile_root / f"play_{run_id}.json"
    stdout_path = playfile_root / f"headless_{run_id}.stdout.log"
    stderr_path = playfile_root / f"headless_{run_id}.stderr.log"
    external_log_dir = playfile_root / f"external_{run_id}"
    play_config["External Log Dir"] = str(external_log_dir)
    play_path.write_text(json.dumps(play_config, indent=2), encoding="utf-8")
    classpath_parts = [workdir / part for part in config.selfplay.java_classpath.split(";")]
    resolved_classpath = ";".join(str(path) for path in classpath_parts)
    java_executable = _resolve_java_executable(config)
    command = [java_executable, "-cp", resolved_classpath, config.selfplay.java_main_class, str(play_path)]
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
            process = subprocess.Popen(
                command,
                cwd=str(workdir),
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                start_new_session=(os.name != "nt"),
                creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0),
            )
            started_at = time.monotonic()
            progress_interval = int(config.selfplay.progress_interval_seconds)
            next_progress_at = started_at + progress_interval if progress_interval > 0 else float("inf")
            timeout_seconds = max(1, int(config.selfplay.timeout_seconds))
            label = progress_label or "selfplay"
            while process.poll() is None:
                now = time.monotonic()
                elapsed = now - started_at
                if elapsed > timeout_seconds:
                    _terminate_process_tree(process)
                    stdout_handle.flush()
                    stderr_handle.flush()
                    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
                    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
                    bot_tail = _external_log_tail(external_log_dir)
                    if bot_tail:
                        stderr += f"\n[mcts_nn] external bot stderr tails:\n{bot_tail}"
                    return subprocess.CompletedProcess(
                        args=command,
                        returncode=124,
                        stdout=stdout,
                        stderr=stderr + f"\n[mcts_nn] {label} timed out after {timeout_seconds}s",
                    )
                if now >= next_progress_at:
                    print(
                        f"[game progress] {label} elapsed={elapsed:.0f}s "
                        f"limit={timeout_seconds}s max_turns={config.selfplay.max_turns_capitals}",
                        flush=True,
                    )
                    next_progress_at = now + progress_interval
                time.sleep(0.25)
            stdout_handle.flush()
            stderr_handle.flush()
        stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
        if config.selfplay.profile_selfplay:
            bot_logs = _external_log_text(external_log_dir)
            if bot_logs:
                stderr += f"\n[mcts_nn] external bot stderr logs:\n{bot_logs}"
        if process.returncode != 0:
            bot_tail = _external_log_tail(external_log_dir)
            if bot_tail:
                stderr += f"\n[mcts_nn] external bot stderr tails:\n{bot_tail}"
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    finally:
        for path in (play_path, stdout_path, stderr_path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            for log_path in external_log_dir.glob("*"):
                log_path.unlink(missing_ok=True)
            external_log_dir.rmdir()
        except OSError:
            pass
