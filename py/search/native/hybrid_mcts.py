from __future__ import annotations

from typing import Any, Dict, List
import math
import threading

import torch

from search.config import ModelConfig, SearchConfig
from nn.encoding import normalize_message
from nn.model import HybridPolicyValueNet
from .mcts import ReusableNativeMCTSSession, SearchResult, _Evaluation, _evaluate_messages, run_native_mcts
from .static_mcts import _evaluate_static_messages


_STATIC_PRIOR_EPS = 1e-8
_HYBRID_PATCH_LOCK = threading.Lock()


def _clamp_weight(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_priors(priors: List[float]) -> List[float]:
    total = sum(max(0.0, float(prior)) for prior in priors)
    if total > 0.0:
        return [max(0.0, float(prior)) / total for prior in priors]
    return [1.0 / len(priors)] * len(priors) if priors else []


def _mix_evaluation(nn_eval: _Evaluation, static_eval: _Evaluation, policy_weight: float, value_weight: float) -> _Evaluation:
    if len(nn_eval.priors) != len(static_eval.priors):
        raise ValueError(
            f"Hybrid eval prior/action mismatch: nn={len(nn_eval.priors)} static={len(static_eval.priors)}"
        )
    policy_weight = _clamp_weight(policy_weight)
    value_weight = _clamp_weight(value_weight)
    if policy_weight <= 0.0:
        priors = _normalize_priors(nn_eval.priors)
    else:
        logits = [
            math.log(max(_STATIC_PRIOR_EPS, float(nn_prior)))
            + policy_weight * math.log(max(_STATIC_PRIOR_EPS, float(static_prior)))
            for nn_prior, static_prior in zip(nn_eval.priors, static_eval.priors)
        ]
        max_logit = max(logits, default=0.0)
        exp_values = [math.exp(logit - max_logit) for logit in logits]
        priors = _normalize_priors(exp_values)
    value = (1.0 - value_weight) * float(nn_eval.value) + value_weight * float(static_eval.value)
    return _Evaluation(priors=priors, value=max(-1.0, min(1.0, value)))


def _evaluate_hybrid_messages(
    messages: List[Dict[str, Any]],
    model: HybridPolicyValueNet,
    search_cfg: SearchConfig,
    model_cfg: ModelConfig,
    device: torch.device | str,
    belief_snapshot: Any | None = None,
) -> List[_Evaluation]:
    if not messages:
        return []
    nn_evaluations = _evaluate_messages(messages, model, model_cfg, device, belief_snapshot)
    static_evaluations = _evaluate_static_messages(messages, model_cfg.max_actions)
    return [
        _mix_evaluation(
            nn_eval,
            static_eval,
            float(search_cfg.static_policy_weight),
            float(search_cfg.static_value_weight),
        )
        for nn_eval, static_eval in zip(nn_evaluations, static_evaluations)
    ]


def _root_hybrid_priors(
    message: Dict[str, Any],
    model: HybridPolicyValueNet,
    search_cfg: SearchConfig,
    model_cfg: ModelConfig,
    device: torch.device | str,
    root_policy_logits: torch.Tensor | None = None,
    root_value: torch.Tensor | float | None = None,
    belief_snapshot: Any | None = None,
) -> tuple[List[str], List[float], float, List[int]]:
    root_actions = list(message.get("actions", []))[: model_cfg.max_actions]
    if not root_actions:
        return [], [], 0.0, []
    if root_policy_logits is None or root_value is None:
        nn_eval = _evaluate_messages(
            [{"player_id": int(message["player_id"]), "observation": message["observation"], "actions": root_actions}],
            model,
            model_cfg,
            device,
            belief_snapshot,
        )[0]
    else:
        action_count = len(root_actions)
        logits = root_policy_logits.detach()
        if logits.ndim > 1:
            logits = logits[0]
        priors = torch.softmax(logits[:action_count], dim=-1).detach().cpu().tolist() if action_count else []
        value = float(root_value.detach().flatten()[0].item()) if torch.is_tensor(root_value) else float(root_value)
        nn_eval = _Evaluation([float(prob) for prob in priors], value)
    static_eval = _evaluate_static_messages(
        [{"player_id": int(message["player_id"]), "observation": message["observation"], "actions": root_actions}],
        model_cfg.max_actions,
    )[0]
    root_eval = _mix_evaluation(
        nn_eval,
        static_eval,
        float(search_cfg.static_policy_weight),
        float(search_cfg.static_value_weight),
    )
    indexes = list(range(len(root_actions)))
    if len(indexes) > search_cfg.top_k_actions:
        ranked = sorted(indexes, key=lambda idx: root_eval.priors[idx], reverse=True)[: search_cfg.top_k_actions]
        indexes = sorted(ranked)
    action_ids = [str(root_actions[index].get("id")) for index in indexes]
    priors = _normalize_priors([root_eval.priors[index] for index in indexes])
    return action_ids, priors, root_eval.value, indexes


class ReusableNativeHybridMCTSSession(ReusableNativeMCTSSession):
    def _root_priors(
        self,
        root_payload: Dict[str, Any],
        evaluator: HybridPolicyValueNet,
        search_cfg: SearchConfig,
        model_cfg: ModelConfig,
        device: torch.device | str,
        root_policy_logits: torch.Tensor | None,
        root_value: torch.Tensor | float | None,
        belief_snapshot: Any | None,
    ) -> tuple[List[str], List[float], float, List[int]]:
        return _root_hybrid_priors(
            root_payload,
            evaluator,
            search_cfg,
            model_cfg,
            device,
            root_policy_logits,
            root_value,
            belief_snapshot,
        )

    def _evaluate_messages(
        self,
        messages: List[Dict[str, Any]],
        evaluator: HybridPolicyValueNet,
        model_cfg: ModelConfig,
        device: torch.device | str,
    ) -> List[_Evaluation]:
        return _evaluate_hybrid_messages(messages, evaluator, self._search_cfg, model_cfg, device, None)

    def search(
        self,
        root_payload: Dict[str, Any],
        evaluator: HybridPolicyValueNet,
        search_cfg: SearchConfig,
        model_cfg: ModelConfig,
        device: torch.device | str,
        root_policy_logits: torch.Tensor | None = None,
        root_value: torch.Tensor | float | None = None,
        belief_snapshot: Any | None = None,
        wall_time_seconds: float | None = None,
    ) -> SearchResult:
        self._search_cfg = search_cfg
        self._belief_snapshot = belief_snapshot
        try:
            return super().search(
                root_payload,
                evaluator,
                search_cfg,
                model_cfg,
                device,
                root_policy_logits,
                root_value,
                belief_snapshot,
                wall_time_seconds,
            )
        finally:
            self._belief_snapshot = None


def run_native_hybrid_mcts(
    root_payload: Dict[str, Any],
    evaluator: HybridPolicyValueNet,
    search_cfg: SearchConfig,
    model_cfg: ModelConfig,
    device: torch.device | str,
    root_policy_logits: torch.Tensor | None = None,
    root_value: torch.Tensor | float | None = None,
    belief_snapshot: Any | None = None,
    wall_time_seconds: float | None = None,
) -> SearchResult:
    search_cfg.static_policy_weight = _clamp_weight(float(search_cfg.static_policy_weight))
    search_cfg.static_value_weight = _clamp_weight(float(search_cfg.static_value_weight))
    if search_cfg.static_policy_weight <= 0.0 and search_cfg.static_value_weight <= 0.0:
        return run_native_mcts(
            root_payload,
            evaluator,
            search_cfg,
            model_cfg,
            device,
            root_policy_logits,
            root_value,
            belief_snapshot,
            wall_time_seconds,
        )

    import search.native.mcts as mcts_module

    original_root_priors = mcts_module._root_priors
    original_evaluate_messages = mcts_module._evaluate_messages

    with _HYBRID_PATCH_LOCK:
        try:
            mcts_module._root_priors = _root_hybrid_priors
            mcts_module._evaluate_messages = (
                lambda messages, model, model_cfg_arg, device_arg, belief_snapshot_arg=None: _evaluate_hybrid_messages(
                    messages,
                    model,
                    search_cfg,
                    model_cfg_arg,
                    device_arg,
                    belief_snapshot_arg,
                )
            )
            return mcts_module.run_native_mcts(
                normalize_message(root_payload),
                evaluator,
                search_cfg,
                model_cfg,
                device,
                root_policy_logits,
                root_value,
                belief_snapshot,
                wall_time_seconds,
            )
        finally:
            mcts_module._root_priors = original_root_priors
            mcts_module._evaluate_messages = original_evaluate_messages
