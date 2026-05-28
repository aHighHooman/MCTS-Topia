from __future__ import annotations

import argparse
import cProfile
import json
import os
import socketserver
import sys
import threading
from pathlib import Path
from typing import Any

import torch

PY_ROOT = Path(__file__).resolve().parents[1]
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from nn.bot_agent import HybridRLBot
from nn.model import HybridPolicyValueNet
from training.config import HybridAgentConfig
from search.device import require_cuda_device
from search.native.cpp_extension import load_native_mcts_extension


def _load_checkpoint(model: HybridPolicyValueNet, checkpoint_path: Path) -> None:
    if not checkpoint_path.exists():
        return
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("model", payload)
    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and getattr(value, "shape", None) == getattr(model_state[key], "shape", None)
    }
    model.load_state_dict(compatible, strict=False)


def _configure(args: argparse.Namespace) -> HybridAgentConfig:
    cfg = HybridAgentConfig()
    cfg.search.num_simulations = int(args.simulations)
    cfg.search.max_depth = int(args.max_depth)
    cfg.search.top_k_actions = int(args.top_k_actions)
    cfg.search.batch_size = int(args.search_batch_size)
    cfg.search.static_policy_weight = max(0.0, min(1.0, float(getattr(args, "static_policy_weight", 0.0) or 0.0)))
    cfg.search.static_value_weight = max(0.0, min(1.0, float(getattr(args, "static_value_weight", 0.0) or 0.0)))
    cfg.selfplay.max_actions_per_game = int(args.max_game_actions)
    if args.wall_clock_per_action_seconds is not None:
        cfg.selfplay.wall_clock_per_action_seconds = float(args.wall_clock_per_action_seconds)
    cfg.selfplay.profile_selfplay = bool(args.profile_selfplay)
    return cfg


class PersistentBotServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[socketserver.BaseRequestHandler],
        *,
        cfg: HybridAgentConfig,
        checkpoint_path: Path,
        replay_dir: Path,
        model: HybridPolicyValueNet,
        device: torch.device,
        native_available: bool,
        profiler: cProfile.Profile | None = None,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.cfg = cfg
        self.checkpoint_path = checkpoint_path
        self.replay_dir = replay_dir
        self.model = model
        self.device = device
        self.native_available = native_available
        self.profiler = profiler
        self.inference_lock = threading.Lock()
        self.shutdown_requested = threading.Event()


class BotRequestHandler(socketserver.StreamRequestHandler):
    server: PersistentBotServer

    def handle(self) -> None:
        bot: HybridRLBot | None = None
        while True:
            raw = self.rfile.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                message: dict[str, Any] = json.loads(line)
                if message.get("type") == "shutdown":
                    self._write({"ok": True})
                    self.server.shutdown_requested.set()
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                    break
                if bot is None:
                    bot = HybridRLBot(
                        self.server.cfg,
                        self.server.checkpoint_path,
                        self.server.replay_dir,
                        model=self.server.model,
                        device=self.server.device,
                        native_available=self.server.native_available,
                        warmup=False,
                    )
                if message.get("type") == "action_request":
                    with self.server.inference_lock:
                        if self.server.profiler is not None:
                            self.server.profiler.enable()
                        try:
                            response = bot.choose_action(message)
                        finally:
                            if self.server.profiler is not None:
                                self.server.profiler.disable()
                    self._write(response)
                elif message.get("type") == "game_over":
                    with self.server.inference_lock:
                        if self.server.profiler is not None:
                            self.server.profiler.enable()
                        try:
                            bot.finish_episode(message)
                        finally:
                            if self.server.profiler is not None:
                                self.server.profiler.disable()
                    self._write({"ok": True})
                    break
                else:
                    self._write({"error": f"unsupported message type: {message.get('type')}"})
            except Exception as exc:
                print(f"[persistent_bot_server] request failed: {exc}", file=sys.stderr, flush=True)
                self._write({"error": str(exc)})
                break

    def _write(self, payload: dict[str, Any]) -> None:
        self.wfile.write((json.dumps(payload) + "\n").encode("utf-8"))
        self.wfile.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-iteration persistent Tribes RL bot server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--simulations", type=int, required=True)
    parser.add_argument("--max-depth", type=int, required=True)
    parser.add_argument("--top-k-actions", type=int, required=True)
    parser.add_argument("--search-batch-size", type=int, required=True)
    parser.add_argument("--static-policy-weight", type=float, default=0.0)
    parser.add_argument("--static-value-weight", type=float, default=0.0)
    parser.add_argument("--max-game-actions", type=int, default=512)
    parser.add_argument("--wall-clock-per-action-seconds", type=float, default=None)
    parser.add_argument("--wall-clock-per-turn-seconds", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--profile-selfplay", action="store_true")
    parser.add_argument("--profile-output", type=Path, default=None)
    args = parser.parse_args()
    if args.wall_clock_per_action_seconds is None:
        args.wall_clock_per_action_seconds = args.wall_clock_per_turn_seconds

    profiler: cProfile.Profile | None = cProfile.Profile() if args.profile_output is not None else None
    cfg = _configure(args)
    try:
        if cfg.selfplay.profile_selfplay:
            os.environ["MCTS_NN_PROFILE"] = "1"
        device = require_cuda_device()
        model = HybridPolicyValueNet(cfg.model).to(device)
        _load_checkpoint(model, args.checkpoint)
        model.eval()
        native_available = load_native_mcts_extension() is not None
        if not native_available:
            raise RuntimeError("Native MCTS extension is required but could not be loaded.")
        warmup_bot = HybridRLBot(
            cfg,
            args.checkpoint,
            args.replay_dir,
            model=model,
            device=device,
            native_available=True,
            warmup=True,
        )
        warmup_bot.reset_episode()
        with PersistentBotServer(
            (args.host, args.port),
            BotRequestHandler,
            cfg=cfg,
            checkpoint_path=args.checkpoint,
            replay_dir=args.replay_dir,
            model=model,
            device=device,
            native_available=True,
            profiler=profiler,
        ) as server:
            print(f"[persistent_bot_server] ready host={args.host} port={args.port}", flush=True)
            server.serve_forever(poll_interval=0.2)
        print("[persistent_bot_server] stopped", flush=True)
    finally:
        if profiler is not None:
            profiler.disable()
            args.profile_output.parent.mkdir(parents=True, exist_ok=True)
            function_count = len(profiler.getstats())
            profiler.dump_stats(str(args.profile_output))
            print(
                f"[persistent_bot_server] wrote cProfile stats path={args.profile_output} "
                f"functions={function_count}",
                file=sys.stderr,
                flush=True,
            )


if __name__ == "__main__":
    main()
