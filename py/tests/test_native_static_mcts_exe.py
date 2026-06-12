from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from test_native_mcts import _message_with_capital_capture


ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT / "out" / "native" / "native_static_mcts_bot.exe"


def _require_exe() -> Path:
    if not EXE.exists():
        pytest.skip("native static MCTS executable is not built; run scripts/build_native_static_bot.ps1")
    return EXE


def test_native_static_mcts_exe_protocol_smoke_without_forward_model() -> None:
    message = _message_with_capital_capture(second_action={"id": "end", "type": "END_TURN"})
    message["type"] = "action_request"
    completed = subprocess.run(
        [str(_require_exe()), "--simulations", "0", "--deterministic"],
        input=json.dumps(message) + "\n",
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )

    response = json.loads(completed.stdout.strip().splitlines()[-1])

    assert response["actionId"] == "capture"
    assert response["rankedActionIds"][:2] == ["capture", "end"]


def test_native_static_mcts_exe_uses_native_tree_without_forward_model_protocol() -> None:
    message = _message_with_capital_capture(second_action={"id": "end", "type": "END_TURN"})
    message["type"] = "action_request"
    completed = subprocess.run(
        [str(_require_exe()), "--simulations", "4", "--search-batch-size", "2", "--deterministic", "--seed", "1"],
        input=json.dumps(message) + "\n",
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )

    output_lines = completed.stdout.strip().splitlines()
    assert len(output_lines) == 1
    response = json.loads(output_lines[-1])
    assert response["actionId"] == "capture"
    assert response["rankedActionIds"][0] == "capture"
