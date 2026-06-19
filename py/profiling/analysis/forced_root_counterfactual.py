from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from profiling.config import load_config_defaults
from project_paths import game_json_jar, game_root

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "debug-logs" / "analysis" / "counterfactuals"
DEFAULT_CONFIG = PROJECT_ROOT / "py" / "profiling" / "configs" / "forced_root_counterfactual.json"


def _repo_relative(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _command(value: str) -> list[str]:
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise RuntimeError("command JSON must be a string array")
    return parsed


def _force_wrapper_command(delegate: list[str], forced_action: str, force_request_id: int | None) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "py" / "bots" / "force_once_bot.py"),
        "--force-action",
        forced_action,
    ]
    if force_request_id is not None:
        command.extend(["--force-request-id", str(force_request_id)])
    command.append("--")
    command.extend(delegate)
    return command


def _classify(original_result: str, forced_result: str, original_margin: float, forced_margin: float) -> str:
    if original_result != "WIN" and forced_result == "WIN":
        return "rescue"
    improvement = forced_margin - original_margin
    if forced_result != "WIN" and improvement >= max(50.0, abs(original_margin) * 0.15):
        return "improvement"
    if improvement <= -max(50.0, abs(original_margin) * 0.15):
        return "bad"
    return "neutral"


def run(args: argparse.Namespace) -> Path:
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    output_dir = _repo_relative(Path(args.output_dir)) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config = _load_json(Path(args.base_config))
    baseline_command = _command(args.baseline_command_json)
    experimental_command = _command(args.experimental_command_json)
    actor = int(args.actor_player)
    if actor < 0 or actor > 1:
        raise RuntimeError("forced_root_counterfactual currently supports two-player games")
    commands = [baseline_command, experimental_command]
    commands[actor] = _force_wrapper_command(
        experimental_command if actor == int(args.experimental_player) else baseline_command,
        args.forced_action,
        args.force_request_id,
    )
    config = dict(base_config)
    config["Run Mode"] = "Replay"
    config["Replay File Name"] = str(Path(args.snapshot).resolve())
    config["Players"] = ["External", "External"]
    config["External Commands"] = commands
    config["External Log Dir"] = str(output_dir / "external-logs")
    config["Track Stats"] = True
    config["Stats Report Path"] = str(output_dir / "stats")
    config["Analysis Action Snapshots"] = True
    config_path = output_dir / "counterfactual_play.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    cp = os.pathsep.join([str(PROJECT_ROOT / "out"), str(game_json_jar())])
    java = str(Path(args.java_exe)) if args.java_exe else str(Path(os.environ.get("JAVA_HOME", "")) / "bin" / "java.exe")
    if args.java_exe is None and not Path(java).exists():
        java = "java"
    command = [java, "-cp", cp, "HeadlessPlay", str(config_path)]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=int(args.timeout_sec),
        check=False,
    )
    (output_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    summary = {
        "snapshot": str(Path(args.snapshot).resolve()),
        "forced_action": args.forced_action,
        "force_request_id": args.force_request_id,
        "actor_player": actor,
        "experimental_player": int(args.experimental_player),
        "returncode": completed.returncode,
        "classification": "unknown" if completed.returncode == 0 else "invalid",
        "config": str(config_path),
        "stdout": str(output_dir / "stdout.log"),
        "stderr": str(output_dir / "stderr.log"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {output_dir}")
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay a save snapshot with one forced root action.")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--actor-player", type=int)
    parser.add_argument("--experimental-player", type=int, default=1)
    parser.add_argument("--forced-action")
    parser.add_argument("--force-request-id", type=int, default=None)
    parser.add_argument("--baseline-command-json")
    parser.add_argument("--experimental-command-json")
    parser.add_argument("--java-exe", type=Path, default=None)
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = load_config_defaults(parser, default_config=DEFAULT_CONFIG)
    for name in ("snapshot", "base_config", "actor_player", "forced_action", "baseline_command_json", "experimental_command_json"):
        if getattr(args, name) is None:
            raise RuntimeError(f"forced_root_counterfactual requires --{name.replace('_', '-')}")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
