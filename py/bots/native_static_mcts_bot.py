from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional


PY_ROOT = Path(__file__).resolve().parents[2]
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from nn.encoding import normalize_message
from search.config import HybridAgentConfig
from search.native import run_native_static_mcts
from search.native.mcts import NativeSearchParityError


def _configure(args: argparse.Namespace) -> HybridAgentConfig:
    cfg = HybridAgentConfig()
    cfg.search.num_simulations = int(args.simulations)
    cfg.search.max_depth = int(args.max_depth)
    cfg.search.top_k_actions = int(args.top_k_actions)
    cfg.search.batch_size = int(args.search_batch_size)
    cfg.search.seed = int(args.seed)
    cfg.model.max_actions = int(args.max_actions)
    if args.deterministic:
        cfg.search.sample_action = False
        cfg.search.root_temperature = 1e-6
        cfg.search.dirichlet_epsilon = 0.0
    return cfg


def choose_action(
    message: Dict[str, Any],
    cfg: HybridAgentConfig,
    wall_clock_per_action_seconds: Optional[float],
) -> Dict[str, Optional[str]]:
    message = normalize_message(message)
    result = run_native_static_mcts(
        message,
        cfg.search,
        cfg.model,
        wall_time_seconds=wall_clock_per_action_seconds,
    )
    legal_action_ids = {str(action.get("id")) for action in message.get("actions", [])}
    selected = str(result.action_id) if result.action_id is not None else None
    if selected not in legal_action_ids:
        raise NativeSearchParityError(
            "Native static MCTS selected action not present in legal action ids "
            f"selected={selected}"
        )
    ranked = [
        action_id
        for action_id, _prob in sorted(result.visit_distribution.items(), key=lambda item: item[1], reverse=True)
        if action_id in legal_action_ids
    ]
    seen = set(ranked)
    ranked.extend(str(action.get("id")) for action in message.get("actions", []) if str(action.get("id")) not in seen)
    return {"actionId": selected, "rankedActionIds": ranked}


def main() -> None:
    parser = argparse.ArgumentParser(description="Protocol-facing native static-eval MCTS bot for Tribes.")
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--wall-clock-per-action-seconds", type=float, default=None)
    parser.add_argument("--max-depth", type=int, default=0)
    parser.add_argument("--top-k-actions", type=int, default=64)
    parser.add_argument("--max-actions", type=int, default=512, help="Maximum actions to parse/search; use -1 for no cap.")
    parser.add_argument("--search-batch-size", type=int, default=64)
    parser.add_argument("--static-eval-variant", choices=("baseline", "tuned"), default="tuned")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    os.environ["TRIBES_STATIC_EVAL_VARIANT"] = args.static_eval_variant
    cfg = _configure(args)

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if message.get("type") == "action_request":
            sys.stdout.write(json.dumps(choose_action(message, cfg, args.wall_clock_per_action_seconds)) + "\n")
            sys.stdout.flush()
        elif message.get("type") == "game_over":
            break


if __name__ == "__main__":
    main()
