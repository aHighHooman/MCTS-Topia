from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PY_ROOT = Path(__file__).resolve().parents[1]
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from nn.bot_agent import BotOutputClosed, HybridRLBot
from search.config import HybridAgentConfig
from training.config import rl_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Protocol-facing hybrid RL bot for Tribes.")
    parser.add_argument("--checkpoint", type=Path, default=rl_path("checkpoints", "latest.pt"))
    parser.add_argument("--replay-dir", type=Path, default=rl_path("replay"))
    parser.add_argument("--simulations", type=int, default=None)
    parser.add_argument("--top-k-actions", type=int, default=None)
    parser.add_argument("--search-batch-size", type=int, default=None)
    parser.add_argument("--max-game-actions", type=int, default=None)
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--wall-clock-per-action-seconds", type=float, default=None)
    parser.add_argument("--wall-clock-per-turn-seconds", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--reuse-tree", action="store_true", help="Experimentally reuse/promote the selected native MCTS subtree between same-turn action requests.")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = HybridAgentConfig()
    if args.simulations is not None:
        cfg.search.num_simulations = args.simulations
    if args.top_k_actions is not None:
        cfg.search.top_k_actions = args.top_k_actions
    if args.search_batch_size is not None:
        cfg.search.batch_size = args.search_batch_size
    if args.max_game_actions is not None:
        cfg.selfplay.max_actions_per_game = args.max_game_actions
    if args.max_actions is not None:
        cfg.model.max_actions = int(args.max_actions)
    if args.seed is not None:
        # SearchConfig may not declare seed as a dataclass field, but the native
        # search wrappers read it with getattr(..., "seed", ...), so attaching it
        # here is intentional and compatible.
        cfg.search.seed = int(args.seed)
    if args.reuse_tree:
        cfg.search.reuse_tree = True
    wall_clock_per_action = args.wall_clock_per_action_seconds
    if wall_clock_per_action is None:
        wall_clock_per_action = args.wall_clock_per_turn_seconds
    if wall_clock_per_action is not None:
        cfg.selfplay.wall_clock_per_action_seconds = wall_clock_per_action
    if args.deterministic:
        cfg.search.sample_action = False
        cfg.search.root_temperature = 1e-6
        cfg.search.dirichlet_epsilon = 0.0

    bot = HybridRLBot(
        cfg,
        args.checkpoint,
        args.replay_dir,
    )
    while True:
        try:
            raw_line = input()
        except (EOFError, OSError):
            break
        line = raw_line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue

        msg_type = message.get("type")
        if msg_type == "action_request":
            try:
                HybridRLBot.emit(bot.choose_action(message))
            except BotOutputClosed:
                try:
                    sys.stdout = open(os.devnull, "w", encoding="utf-8")
                except OSError:
                    pass
                break
        elif msg_type == "game_over":
            bot.finish_episode(message)
            break
        elif msg_type == "shutdown":
            break


if __name__ == "__main__":
    main()
