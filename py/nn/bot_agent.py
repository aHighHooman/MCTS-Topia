from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .belief import BeliefTracker
from .encoding import EncodedObservation, encode_observation, normalize_message


from .model import HybridPolicyValueNet


def compatible_state_dict(model: HybridPolicyValueNet, state_dict: dict[str, Any]) -> dict[str, Any]:
    """Fallback compatibility filter for repos whose nn.model lacks compatible_state_dict."""
    model_state = model.state_dict()
    filtered: dict[str, Any] = {}
    for key, value in state_dict.items():
        if key in model_state and getattr(value, "shape", None) == getattr(model_state[key], "shape", None):
            filtered[key] = value
    return filtered



from search.config import HybridAgentConfig
from search.device import require_cuda_device
from search.native import NativeSearchUnavailable, run_native_mcts, run_native_static_mcts
from search.native.cpp_extension import load_native_mcts_extension
from search.native.mcts import SearchResult
from training.replay import (
    ReplayStore,
    StepRecord,
    compute_returns,
    terminal_reward_for_player,
)


def _profile_enabled() -> bool:
    return os.environ.get("TRIBES_RL_PROFILE", "").strip().lower() in {"1", "true", "yes", "on"}


def _profile_log(message: str) -> None:
    if _profile_enabled():
        print(f"[tribes_rl.profile] {message}", file=sys.stderr, flush=True)


def _compact_message(message: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "player_id": int(message["player_id"]),
        "observation": message["observation"],
        "actions": list(message.get("actions", [])),
    }


def _ranked_action_ids(message: Dict[str, Any], result: Any) -> list[str]:
    actions = list(message.get("actions", []))
    action_ids = [str(action.get("id")) for action in actions]
    visit_target = [float(value) for value in list(getattr(result, "visit_target", []) or [])]
    if len(visit_target) == len(action_ids):
        ranked_indexes = sorted(range(len(action_ids)), key=lambda idx: visit_target[idx], reverse=True)
        return [action_ids[idx] for idx in ranked_indexes]
    distribution = getattr(result, "visit_distribution", {}) or {}
    ranked_pairs = sorted(
        ((str(action_id), float(score)) for action_id, score in distribution.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    ranked = [action_id for action_id, _ in ranked_pairs if action_id in action_ids]
    seen = set(ranked)
    ranked.extend(action_id for action_id in action_ids if action_id not in seen)
    return ranked


def _write_invalid_action_debug(message: Dict[str, Any], selected_action_id: str | None, ranked_action_ids: list[str]) -> Path | None:
    try:
        debug_root = Path("debug-logs") / "invalid-action-fallbacks"
        debug_root.mkdir(parents=True, exist_ok=True)
        now_ms = int(time.time() * 1000)
        path = debug_root / f"tribes_rl_invalid_action_{now_ms}_{os.getpid()}.json"
        payload = {
            "source": "tribes_rl_bot",
            "created_at_ms": now_ms,
            "selected_action_id": selected_action_id,
            "ranked_action_ids": ranked_action_ids,
            "player_id": message.get("player_id"),
            "active_player_id": message.get("observation", {}).get("active_player_id"),
            "tick": message.get("observation", {}).get("tick"),
            "legal_action_ids": [str(action.get("id")) for action in message.get("actions", [])],
            "message": message,
            "stack": traceback.format_stack(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path
    except Exception:
        return None


class HybridRLBot:
    def __init__(
        self,
        config: HybridAgentConfig,
        checkpoint_path: Path,
        replay_dir: Path,
        model: HybridPolicyValueNet | None = None,
        device: torch.device | None = None,
        native_available: bool | None = None,
        warmup: bool = True,
        static_only_bootstrap: bool = False,
    ) -> None:
        self.config = config
        self.static_only_bootstrap = bool(static_only_bootstrap)
        self.replay = ReplayStore(replay_dir, config.replay.capacity_steps, config.replay.shard_prefix, load_existing=False)
        self.belief_tracker = BeliefTracker()
        self.records: list[StepRecord] = []
        self.turn_index = -1
        self.turn_step_index = 0
        self.last_active_player: int | None = None
        self.turn_budget_started_at: float | None = None

        # Static bootstrap mode intentionally skips every NN cost: no CUDA requirement,
        # no checkpoint load, no model construction, no warmup, no root/leaf forward passes.
        if self.static_only_bootstrap:
            self.device = device or torch.device("cpu")
            self.config.training.device = str(self.device)
            self.model: HybridPolicyValueNet | None = None
            self._native_available = native_available if native_available is not None else self._initialize_native_search()
            return

        self.device = device or require_cuda_device()
        self.config.training.device = str(self.device)
        self.model = model or HybridPolicyValueNet(config.model).to(self.device)
        self.model.eval()
        if model is None:
            self._load_checkpoint(checkpoint_path)
        self._native_available = native_available if native_available is not None else self._initialize_native_search()
        if warmup:
            self._warmup_model()

    def _initialize_native_search(self) -> bool:
        available = load_native_mcts_extension() is not None
        if not available:
            raise NativeSearchUnavailable("Native MCTS extension is required but could not be loaded.")
        return available

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        if self.model is None or not checkpoint_path.exists():
            return
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = payload.get("model", payload)
        self.model.load_state_dict(compatible_state_dict(self.model, state_dict), strict=False)

    def _warmup_model(self) -> None:
        if self.model is None:
            return
        started_at = time.perf_counter()
        native_loaded = bool(self._native_available)
        cuda_name = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else str(self.device)
        try:
            cfg = self.config.model
            # Some older configs only expose entity_feature_dim; newer ones split unit/city widths.
            unit_width = int(getattr(cfg, "unit_feature_dim", getattr(cfg, "entity_feature_dim", 0)))
            city_width = int(getattr(cfg, "city_feature_dim", getattr(cfg, "entity_feature_dim", unit_width)))
            encoded = EncodedObservation(
                board=torch.zeros((1, cfg.board_channels, cfg.board_size, cfg.board_size), device=self.device),
                unit_features=torch.zeros((1, cfg.max_units, unit_width), device=self.device),
                unit_mask=torch.zeros((1, cfg.max_units), dtype=torch.bool, device=self.device),
                city_features=torch.zeros((1, cfg.max_cities, city_width), device=self.device),
                city_mask=torch.zeros((1, cfg.max_cities), dtype=torch.bool, device=self.device),
                action_features=torch.zeros((1, cfg.max_actions, cfg.action_feature_dim), device=self.device),
                action_mask=torch.zeros((1, cfg.max_actions), dtype=torch.bool, device=self.device),
                scalar_features=torch.zeros((1, cfg.scalar_dim), device=self.device),
                action_ids=[],
            )
            encoded.action_mask[:, 0] = True
            with torch.inference_mode():
                self.model(encoded)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        finally:
            warmup_ms = (time.perf_counter() - started_at) * 1000.0
            print(
                "[tribes_rl.warmup] "
                f"warmup_ms={warmup_ms:.1f} native_extension_loaded={str(native_loaded).lower()} "
                f"cuda_device={cuda_name}",
                file=sys.stderr,
                flush=True,
            )

    def reset_episode(self) -> None:
        self.records.clear()
        self.belief_tracker.reset()
        self.turn_index = -1
        self.turn_step_index = 0
        self.last_active_player = None
        self.turn_budget_started_at = None

    def _action_budget_seconds(self) -> float | None:
        budget = float(getattr(self.config.selfplay, "wall_clock_per_action_seconds", 0.0) or 0.0)
        if budget <= 0.0:
            return None
        return budget

    def _fallback_result(self, message: Dict[str, Any], output: Any) -> SearchResult:
        actions = list(message.get("actions", []))
        if not actions:
            return SearchResult("", 0, {}, [], 0.0)
        action_count = len(actions)
        logits = output.policy_logits[0, :action_count].detach()
        probs = torch.softmax(logits, dim=-1).cpu().tolist()
        action_index = int(max(range(action_count), key=lambda idx: probs[idx]))
        action_id = str(actions[action_index].get("id"))
        visit_target = [float(prob) for prob in probs]
        total = sum(max(0.0, value) for value in visit_target)
        if total > 0.0:
            visit_target = [max(0.0, value) / total for value in visit_target]
        else:
            visit_target = [1.0 / action_count] * action_count
        return SearchResult(
            action_id=action_id,
            action_index=action_index,
            visit_distribution={str(action.get("id")): visit_target[index] for index, action in enumerate(actions)},
            visit_target=visit_target,
            value=float(output.value[0].detach().flatten()[0].item()),
        )

    def choose_action(self, message: Dict[str, Any]) -> Dict[str, Optional[str]]:
        message = self.belief_tracker.annotate(normalize_message(message))
        choose_started_at = time.perf_counter()
        actions = list(message.get("actions", []) or [])
        if len(actions) == 1:
            result = SearchResult(
                action_id=str(actions[0].get("id")),
                action_index=0,
                visit_distribution={str(actions[0].get("id")): 1.0},
                visit_target=[1.0],
                value=0.0,
            )
            search_finished_at = time.perf_counter()
            _profile_log(
                "choose_action "
                "encode_ms=0.0 "
                "policy_ms=0.0 "
                "search_ms=0.0 "
                f"total_ms={(search_finished_at - choose_started_at) * 1000.0:.1f} "
                f"actions={len(actions)} device={self.device} static_only={str(self.static_only_bootstrap).lower()} "
                f"action_budget_sec={self._action_budget_seconds() if self._action_budget_seconds() is not None else -1.0:.3f}"
            )
            return self._finish_action_choice(message, result)

        if not self._native_available:
            raise NativeSearchUnavailable("Native MCTS extension is unavailable.")

        if self.static_only_bootstrap:
            result = run_native_static_mcts(
                message,
                self.config.search,
                self.config.model,
                wall_time_seconds=self._action_budget_seconds(),
            )
            search_finished_at = time.perf_counter()
            _profile_log(
                "choose_action "
                "encode_ms=0.0 "
                "policy_ms=0.0 "
                f"search_ms={(search_finished_at - choose_started_at) * 1000.0:.1f} "
                f"total_ms={(search_finished_at - choose_started_at) * 1000.0:.1f} "
                f"actions={len(message.get('actions', []))} device={self.device} static_only=true "
                f"action_budget_sec={self._action_budget_seconds() if self._action_budget_seconds() is not None else -1.0:.3f}"
            )
            return self._finish_action_choice(message, result)

        if self.model is None:
            raise NativeSearchUnavailable("Hybrid NN mode requires a model, but model is None.")

        belief_snapshot = self.belief_tracker.snapshot(int(message["player_id"]))
        encode_started_at = time.perf_counter()
        encoded = encode_observation(message, self.config.model, compact=True).to(self.device)
        encode_finished_at = time.perf_counter()
        with torch.inference_mode():
            output = self.model(encoded)
        inference_finished_at = time.perf_counter()

        action_budget = self._action_budget_seconds()
        search_budget = None if action_budget is None else max(0.0, action_budget - (inference_finished_at - choose_started_at))
        if search_budget is not None and search_budget <= 0.0:
            result = self._fallback_result(message, output)
        else:
            result = run_native_mcts(
                message,
                self.model,
                self.config.search,
                self.config.model,
                self.device,
                output.policy_logits[0],
                output.value[0],
                belief_snapshot,
                wall_time_seconds=search_budget,
            )
        search_finished_at = time.perf_counter()
        _profile_log(
            "choose_action "
            f"encode_ms={(encode_finished_at - encode_started_at) * 1000.0:.1f} "
            f"policy_ms={(inference_finished_at - encode_finished_at) * 1000.0:.1f} "
            f"search_ms={(search_finished_at - inference_finished_at) * 1000.0:.1f} "
            f"total_ms={(search_finished_at - choose_started_at) * 1000.0:.1f} "
            f"actions={len(message.get('actions', []))} device={self.device} static_only=false "
            f"action_budget_sec={action_budget if action_budget is not None else -1.0:.3f}"
        )
        return self._finish_action_choice(message, result)

    def _finish_action_choice(self, message: Dict[str, Any], result: SearchResult) -> Dict[str, Optional[str]]:
        active_player = int(message["observation"].get("active_player_id", message["player_id"]))
        compact = _compact_message(message)
        ranked_action_ids = _ranked_action_ids(message, result)
        selected_action_id = str(result.action_id) if result.action_id is not None else None
        legal_action_ids = {str(action.get("id")) for action in message.get("actions", [])}
        if selected_action_id not in legal_action_ids:
            fallback_action_id = next((action_id for action_id in ranked_action_ids if action_id in legal_action_ids), None)
            debug_path = _write_invalid_action_debug(message, selected_action_id, ranked_action_ids)
            print(
                "[tribes_rl.invalid_action] selected action not present in legal action ids "
                f"selected={selected_action_id} fallback={fallback_action_id} debug={debug_path}",
                file=sys.stderr,
                flush=True,
            )
            selected_action_id = fallback_action_id

        if self.last_active_player != active_player:
            self.turn_index += 1
            self.turn_step_index = 0
            self.last_active_player = active_player

        record_kwargs = {
            "observation": compact["observation"],
            "legal_actions": compact["actions"],
            "action_index": int(result.action_index),
            "action_id": str(selected_action_id),
            "visit_target": list(result.visit_target),
            "root_value": float(result.value),
            "reward_delta": 0.0,
            "player_id": int(message["player_id"]),
            "active_player_id": active_player,
            "tick": int(message["observation"].get("tick", 0)),
            "turn_index": self.turn_index,
            "turn_step_index": self.turn_step_index,
        }
        if "encoded_observation" in getattr(StepRecord, "__dataclass_fields__", {}):
            record_kwargs["encoded_observation"] = None
        self.records.append(StepRecord(**record_kwargs))
        self.turn_step_index += 1
        return {"actionId": selected_action_id, "rankedActionIds": ranked_action_ids}

    def finish_episode(self, message: Dict[str, Any]) -> Optional[Path]:
        message = normalize_message(message)
        if not self.records:
            self.reset_episode()
            return None
        player_id = int(message.get("player_id", self.records[-1].player_id))
        terminal_reward = terminal_reward_for_player(message, player_id, self.config.reward)
        decisive = bool(message.get("winner_id") is not None)
        outcome = {
            "winner_id": message.get("winner_id"),
            "ranking": message.get("ranking"),
            "final_scores": message.get("final_scores"),
            "terminal_reward": terminal_reward,
        }
        compute_returns(self.records, terminal_reward, self.config.reward)
        for record in self.records:
            record.decisive = decisive
            record.outcome = outcome
        path = self.replay.add_episode(self.records, player_id=player_id)
        self.reset_episode()
        return path

    @staticmethod
    def emit(payload: Dict[str, Any]) -> None:
        print(json.dumps(payload), flush=True)
