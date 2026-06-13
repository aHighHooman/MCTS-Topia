from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from profiling.analysis.position_analyzer import analyze_position, parse_target, position_to_dict


def run_local_target(payload: dict[str, Any], payload_hash: str, label: str, target_spec: str, **kwargs: Any) -> dict[str, Any]:
    return position_to_dict(
        analyze_position(
            payload,
            payload_hash=payload_hash,
            label=label,
            target=parse_target(target_spec),
            **kwargs,
        )
    )


def run_command_target(
    payload_path: Path,
    target_spec: str,
    *,
    cwd: Path,
    python: str = sys.executable,
    env: dict[str, str] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    command = [
        python,
        "-m",
        "profiling.analysis.position_analyzer_cli",
        "--payload",
        str(payload_path),
        "--target",
        target_spec,
    ]
    for key, value in kwargs.items():
        command.extend([f"--{key.replace('_', '-')}", str(value)])
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = str(cwd / "py")
    if env:
        child_env.update(env)
    completed = subprocess.run(command, cwd=cwd, env=child_env, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)

