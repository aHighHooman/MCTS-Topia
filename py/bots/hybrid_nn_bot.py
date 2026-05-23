from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PY_ROOT = Path(__file__).resolve().parents[1]
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from nn.bot_agent import HybridRLBot
from search.config import HybridAgentConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Protocol-facing hybrid RL bot for Tribes.")
    parser.add_argument("--checkpoint", type=Path, default=Path("rl/checkpoints/latest.pt"))
    parser.add_argument("--replay-dir", type=Path, default=Path("rl/replay"))
    parser.add_argument("--simulations", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--top-k-actions", type=int, default=None)
    parser.add_argument("--search-batch-size", type=int, default=None)
    parser.add_argument("--static-policy-weight", type=float, default=None)
    parser.add_argument("--static-value-weight", type=float, default=None)
    parser.add_argument("--max-game-actions", type=int, default=None)
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--wall-clock-per-action-seconds", type=float, default=None)
    parser.add_argument("--wall-clock-per-turn-seconds", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--static-eval-variant", choices=("baseline", "tuned"), default=None)
    parser.add_argument(
        "--static-only-bootstrap",
        action="store_true",
        help="Use native static-eval MCTS while keeping HybridRLBot replay recording; skips NN init/warmup/forward passes.",
    )
    args = parser.parse_args()

    cfg = HybridAgentConfig()
    if args.simulations is not None:
        cfg.search.num_simulations = args.simulations
    if args.max_depth is not None:
        cfg.search.max_depth = args.max_depth
    if args.top_k_actions is not None:
        cfg.search.top_k_actions = args.top_k_actions
    if args.search_batch_size is not None:
        cfg.search.batch_size = args.search_batch_size
    if args.static_policy_weight is not None:
        cfg.search.static_policy_weight = max(0.0, min(1.0, float(args.static_policy_weight)))
    if args.static_value_weight is not None:
        cfg.search.static_value_weight = max(0.0, min(1.0, float(args.static_value_weight)))
    if args.max_game_actions is not None:
        cfg.selfplay.max_actions_per_game = args.max_game_actions
    if args.max_actions is not None:
        cfg.model.max_actions = int(args.max_actions)
    if args.seed is not None:
        # SearchConfig may not declare seed as a dataclass field, but the native
        # search wrappers read it with getattr(..., "seed", ...), so attaching it
        # here is intentional and compatible.
        cfg.search.seed = int(args.seed)
    if args.static_eval_variant is not None:
        os.environ["TRIBES_STATIC_EVAL_VARIANT"] = args.static_eval_variant
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
        warmup=not args.static_only_bootstrap,
        static_only_bootstrap=args.static_only_bootstrap,
    )
    while True:
        try:
            raw_line = input()
        except EOFError:
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
            HybridRLBot.emit(bot.choose_action(message))
        elif msg_type == "game_over":
            bot.finish_episode(message)
            break


if __name__ == "__main__":
    main()
