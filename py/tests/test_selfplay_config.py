from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PY_ROOT = Path(__file__).resolve().parents[1]
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from tribes_rl.config import HybridAgentConfig
from tribes_rl.bot_agent import HybridRLBot
from tribes_rl.java_selfplay import run_selfplay_match
from tribes_rl.persistent_bot_server import _configure
from tribes_rl.train import _bot_command, _match_end_reason


class SelfPlayConfigTest(unittest.TestCase):
    def test_training_bot_command_omits_wall_clock_action_budget_by_default(self) -> None:
        cfg = HybridAgentConfig()

        command = _bot_command(Path("bot.py"), Path("latest.pt"), Path("replay"), cfg)

        self.assertNotIn("--wall-clock-per-action-seconds", command)

    def test_training_bot_command_passes_explicit_wall_clock_action_budget(self) -> None:
        cfg = HybridAgentConfig()
        cfg.selfplay.wall_clock_per_action_seconds = 7.5

        command = _bot_command(Path("bot.py"), Path("latest.pt"), Path("replay"), cfg)

        self.assertIn("--wall-clock-per-action-seconds", command)
        index = command.index("--wall-clock-per-action-seconds")
        self.assertEqual(command[index + 1], "7.5")

    def test_persistent_server_config_leaves_wall_clock_action_budget_unset_by_default(self) -> None:
        args = type(
            "Args",
            (),
            {
                "simulations": 64,
                "max_depth": 0,
                "top_k_actions": 32,
                "search_batch_size": 16,
                "max_game_actions": 512,
                "wall_clock_per_action_seconds": None,
                "profile_selfplay": False,
            },
        )()

        cfg = _configure(args)

        self.assertIsNone(cfg.selfplay.wall_clock_per_action_seconds)

    def test_persistent_server_config_reads_explicit_wall_clock_action_budget(self) -> None:
        args = type(
            "Args",
            (),
            {
                "simulations": 64,
                "max_depth": 0,
                "top_k_actions": 32,
                "search_batch_size": 16,
                "max_game_actions": 512,
                "wall_clock_per_action_seconds": 3.25,
                "profile_selfplay": False,
            },
        )()

        cfg = _configure(args)

        self.assertEqual(cfg.selfplay.wall_clock_per_action_seconds, 3.25)

    def test_wall_clock_budget_is_per_action(self) -> None:
        cfg = HybridAgentConfig()
        cfg.selfplay.wall_clock_per_action_seconds = 2.0
        bot = HybridRLBot.__new__(HybridRLBot)
        bot.config = cfg

        self.assertEqual(bot._action_budget_seconds(), 2.0)
        self.assertEqual(bot._action_budget_seconds(), 2.0)

    def test_match_end_reason_reads_headless_summary(self) -> None:
        stdout = "Turn 3; Game Results:\nMatch End Reason: capital_objective: player 0 controls all capitals\n"

        self.assertEqual(_match_end_reason(stdout), "capital_objective: player 0 controls all capitals")

    def test_selfplay_defaults_emit_generated_continents_tiny_with_seeds(self) -> None:
        cfg = HybridAgentConfig()
        cfg.selfplay.game_seed = 11
        cfg.selfplay.agent_seed = 12
        cfg.selfplay.level_seed = 13
        cfg.training.output_dir = Path(tempfile.mkdtemp()) / "rl"
        captured: dict[str, object] = {}

        class FakeProcess:
            returncode = 0

            def poll(self) -> int:
                return 0

        def fake_popen(command, **kwargs):
            play_path = Path(command[-1])
            captured.update(json.loads(play_path.read_text(encoding="utf-8")))
            return FakeProcess()

        with patch("tribes_rl.java_selfplay.subprocess.Popen", side_effect=fake_popen):
            result = run_selfplay_match(
                cfg,
                [["python", "bot.py"], ["python", "bot.py"]],
                ["Xin Xi", "Imperius"],
                Path.cwd(),
            )

        self.assertEqual(result.returncode, 0)
        self.assertEqual(captured["Run Mode"], "PlayLG")
        self.assertEqual(captured["Map Type"], "Continents")
        self.assertEqual(captured["Map Size"], "Tiny")
        self.assertEqual(captured["Game Seed"], "11")
        self.assertEqual(captured["Agents Seed"], "12")
        self.assertEqual(captured["Level Seed"], "13")

    def test_selfplay_passes_absolute_playfile_when_java_cwd_differs(self) -> None:
        cfg = HybridAgentConfig()
        captured: dict[str, object] = {}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            py_cwd = root / "py"
            java_cwd = root / "repo"
            py_cwd.mkdir()
            java_cwd.mkdir()
            cfg.training.output_dir = Path("rl")

            class FakeProcess:
                returncode = 0

                def poll(self) -> int:
                    return 0

            def fake_popen(command, **kwargs):
                play_path = Path(command[-1])
                captured["play_path"] = play_path
                captured["java_cwd"] = Path(kwargs["cwd"])
                self.assertTrue(play_path.is_absolute())
                self.assertTrue(play_path.exists())
                json.loads(play_path.read_text(encoding="utf-8"))
                return FakeProcess()

            old_cwd = Path.cwd()
            try:
                os.chdir(py_cwd)
                with patch("tribes_rl.java_selfplay.subprocess.Popen", side_effect=fake_popen):
                    result = run_selfplay_match(
                        cfg,
                        [["python", "bot.py"], ["python", "bot.py"]],
                        ["Xin Xi", "Imperius"],
                        java_cwd,
                    )
            finally:
                os.chdir(old_cwd)

        self.assertEqual(result.returncode, 0)
        self.assertEqual(captured["java_cwd"], java_cwd)
        self.assertEqual(Path(captured["play_path"]).parent, py_cwd / "rl" / "playfiles")


if __name__ == "__main__":
    unittest.main()
