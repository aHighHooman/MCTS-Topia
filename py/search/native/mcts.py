from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import json
import math
import os
import random
import sys
import time

import torch

from search.config import ModelConfig, SearchConfig
from nn.encoding import EncodedObservation, encode_observation, normalize_message
from nn.model import HybridPolicyValueNet
from .cpp_extension import load_native_mcts_extension

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


class NativeSearchUnavailable(RuntimeError):
    pass


class NativeSearchParityError(RuntimeError):
    pass


def _message_cache_key(message: Dict[str, Any]) -> str:
    return json.dumps(message, sort_keys=True, separators=(",", ":"), default=str)


def _profile_search_enabled() -> bool:
    return os.environ.get("MCTS_NN_PROFILE_SEARCH", os.environ.get("MCTS_NN_PROFILE", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _profile_search_log(message: str) -> None:
    if _profile_search_enabled():
        print(f"[mcts_nn.search_profile] {message}", file=sys.stderr, flush=True)


@dataclass
class SearchResult:
    action_id: str
    action_index: int
    visit_distribution: Dict[str, float]
    visit_target: List[float]
    value: float


@dataclass
class TreeReuseStats:
    attempts: int = 0
    hits: int = 0
    misses: int = 0
    reused_nodes: int = 0
    promoted_subtree_nodes: int = 0
    last_miss_reason: str = ""


@dataclass
class _TreeSearchResult:
    result: SearchResult
    native_action_id: str
    selected_local_index: int
    selected_paths: int
    expanded_nodes: int
    max_selected_depth: int


@dataclass
class _SearchTelemetry:
    unsupported: int = 0
    invalid: int = 0
    approximate: int = 0

    def observe(self, payload: Dict[str, Any]) -> None:
        kind = str(payload.get("native_transition_kind", ""))
        reason = str(payload.get("native_terminal_reason", ""))
        if kind == "native_unsupported_transition" or reason.startswith("unsupported_native_transition:"):
            self.unsupported += 1
        elif kind == "native_invalid_transition" or reason.startswith("invalid_"):
            self.invalid += 1
        if bool(payload.get("native_approximate_transition", False)):
            self.approximate += 1

    def log_fields(self) -> str:
        return (
            f"native_unsupported_transition={self.unsupported} "
            f"native_invalid_transition={self.invalid} "
            f"native_approximate_transition={self.approximate}"
        )


@dataclass
class _Evaluation:
    priors: List[float]
    value: float


_ACTION_COUNT_MASK_CACHE: Dict[tuple[str, tuple[int, ...]], torch.Tensor] = {}
_ACTION_COUNT_MASK_CACHE_MAX = 128


def _action_count_mask(action_counts: List[int], max_action_count: int, device: torch.device) -> torch.Tensor:
    key = (str(device), tuple(int(count) for count in action_counts))
    cached = _ACTION_COUNT_MASK_CACHE.get(key)
    if cached is not None:
        return cached
    counts = torch.tensor(action_counts, device=device)
    mask = torch.arange(max_action_count, device=device).unsqueeze(0) >= counts.unsqueeze(1)
    if len(_ACTION_COUNT_MASK_CACHE) >= _ACTION_COUNT_MASK_CACHE_MAX:
        _ACTION_COUNT_MASK_CACHE.clear()
    _ACTION_COUNT_MASK_CACHE[key] = mask
    return mask


def _native_call(fn: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except RuntimeError as exc:
        message = str(exc)
        if (
            "Native strict forward model parity failure" in message
            or "Native strict payload parse failure" in message
        ):
            raise NativeSearchParityError(message) from exc
        raise


def _stack_encoded(items: List[EncodedObservation]) -> EncodedObservation:
    max_units = max((item.unit_features.shape[1] for item in items), default=0)
    max_cities = max((item.city_features.shape[1] for item in items), default=0)
    max_actions = max((item.action_features.shape[1] for item in items), default=0)

    def pad_slots(tensor: torch.Tensor, size: int) -> torch.Tensor:
        if tensor.shape[1] >= size:
            return tensor
        pad_shape = list(tensor.shape)
        pad_shape[1] = size - tensor.shape[1]
        return torch.cat([tensor, tensor.new_zeros(pad_shape)], dim=1)

    return EncodedObservation(
        board=torch.cat([item.board for item in items], dim=0),
        unit_features=torch.cat([pad_slots(item.unit_features, max_units) for item in items], dim=0),
        unit_mask=torch.cat([pad_slots(item.unit_mask, max_units) for item in items], dim=0),
        city_features=torch.cat([pad_slots(item.city_features, max_cities) for item in items], dim=0),
        city_mask=torch.cat([pad_slots(item.city_mask, max_cities) for item in items], dim=0),
        action_features=torch.cat([pad_slots(item.action_features, max_actions) for item in items], dim=0),
        action_mask=torch.cat([pad_slots(item.action_mask, max_actions) for item in items], dim=0),
        scalar_features=torch.cat([item.scalar_features for item in items], dim=0),
        action_ids=[],
    )


def _evaluate_messages(
    messages: List[Dict[str, Any]],
    model: HybridPolicyValueNet,
    model_cfg: ModelConfig,
    device: torch.device | str,
    belief_snapshot: Any | None = None,
) -> List[_Evaluation]:
    if not messages:
        return []
    if belief_snapshot is not None:
        messages = [belief_snapshot.annotate_without_update(message) for message in messages]
    encoded_items = [encode_observation(message, model_cfg, compact=True, normalized=True) for message in messages]
    batch = encoded_items[0] if len(encoded_items) == 1 else _stack_encoded(encoded_items)
    batch = batch.to(device)
    with torch.inference_mode():
        output = model(batch)

    action_counts = [len(encoded.action_ids) for encoded in encoded_items]
    max_action_count = max(action_counts, default=0)
    if len(action_counts) == 1:
        action_count = action_counts[0]
        probs_rows = [
            torch.softmax(output.policy_logits[0, :action_count], dim=-1).detach().cpu().tolist()
            if action_count
            else []
        ]
    elif max_action_count:
        logits = output.policy_logits[:, :max_action_count]
        mask = _action_count_mask(action_counts, max_action_count, logits.device)
        probs_rows = torch.softmax(logits.masked_fill(mask, float("-inf")), dim=-1).detach().cpu().tolist()
    else:
        probs_rows = [[] for _ in encoded_items]
    values = output.value.detach().flatten().cpu().tolist()
    evaluations: List[_Evaluation] = []
    for index, action_count in enumerate(action_counts):
        evaluations.append(_Evaluation([float(prob) for prob in probs_rows[index][:action_count]], float(values[index])))
    return evaluations


def _raise_on_unsupported_leaf(leaf_payload: Dict[str, Any], search_cfg: SearchConfig) -> None:
    kind = str(leaf_payload.get("native_transition_kind", ""))
    reason = str(leaf_payload.get("native_terminal_reason", ""))
    if kind == "native_unsupported_transition" or reason.startswith("unsupported_native_transition:"):
        raise NativeSearchParityError(
            "Native MCTS encountered unsupported transition "
            f"kind={kind or '<empty>'} reason={reason or '<empty>'}"
        )
    if bool(leaf_payload.get("native_approximate_transition", False)):
        raise NativeSearchParityError(
            "Native MCTS encountered approximate transition "
            f"reason={reason or '<empty>'}"
        )


def _apply_end_turn_visit_guard(
    action_id: str,
    visit_distribution: Dict[str, float],
    root_actions: List[Dict[str, Any]],
    search_cfg: SearchConfig,
) -> str:
    min_visits = int(getattr(search_cfg, "min_non_end_turn_visits", 0) or 0)
    if min_visits <= 0:
        return action_id
    id_to_type = {str(action.get("id")): str(action.get("type")) for action in root_actions}
    if id_to_type.get(action_id) != "END_TURN":
        return action_id
    non_end = [candidate for candidate in visit_distribution if id_to_type.get(candidate) != "END_TURN"]
    if not non_end:
        return action_id
    if any(visit_distribution.get(candidate, 0.0) > 0.0 for candidate in non_end):
        return action_id
    return max(non_end, key=lambda candidate: visit_distribution.get(candidate, 0.0))


def _visit_entropy_bits(visit_distribution: Dict[str, float]) -> float:
    return -sum(
        max(0.0, float(share)) * math.log2(max(1e-12, float(share)))
        for share in visit_distribution.values()
        if float(share) > 0.0
    )


_ROOT_TACTICAL_KEEP_TYPES = {
    "CAPTURE",
    "MAKE_VETERAN",
    "ATTACK",
    "CONVERT",
    "EXAMINE",
    "RESOURCE_GATHERING",
    "LEVEL_UP",
    "RESEARCH_TECH",
    "BUILD",
    "SPAWN",
    "BUILD_ROAD",
    "HEAL_OTHERS",
    "UPGRADE_SHIP",
    "UPGRADE_BOAT",
    "UPGRADE_RAMMER",
    "UPGRADE_SCOUT",
    "UPGRADE_BOMBER",
}


def _action_type(action: Dict[str, Any]) -> str:
    payload = action.get("payload") if isinstance(action.get("payload"), dict) else {}
    return str(action.get("type") or payload.get("type") or "")


def _move_signature(action: Dict[str, Any]) -> tuple[Any, Any, Any] | None:
    if _action_type(action) != "MOVE":
        return None
    payload = action.get("payload") if isinstance(action.get("payload"), dict) else {}
    unit_id = action.get("unit_id", payload.get("unit_id", payload.get("unitId")))
    x = action.get("x", payload.get("x"))
    y = action.get("y", payload.get("y"))
    return unit_id, x, y


def _select_heuristic_root_indexes(root_actions: List[Dict[str, Any]], top_k_actions: int) -> List[int]:
    indexes = list(range(len(root_actions)))
    if top_k_actions <= 0 or len(indexes) <= top_k_actions:
        return indexes

    selected: set[int] = {
        index for index, action in enumerate(root_actions) if _action_type(action) in _ROOT_TACTICAL_KEEP_TYPES
    }
    budget = max(int(top_k_actions), len(selected))
    seen_moves: set[tuple[Any, Any, Any]] = set()
    for index, action in enumerate(root_actions):
        if len(selected) >= budget:
            break
        signature = _move_signature(action)
        if signature is not None:
            if signature in seen_moves:
                continue
            seen_moves.add(signature)
        selected.add(index)
    return sorted(selected)


def _root_priors(
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
    if root_policy_logits is None or root_value is None:
        root_eval = _evaluate_messages(
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
        if torch.is_tensor(root_value):
            value = float(root_value.detach().flatten()[0].item())
        else:
            value = float(root_value)
        root_eval = _Evaluation([float(prob) for prob in priors], value)
    indexes = _select_heuristic_root_indexes(root_actions, int(search_cfg.top_k_actions))
    action_ids = [str(root_actions[index].get("id")) for index in indexes]
    priors = [root_eval.priors[index] for index in indexes]
    total = sum(max(0.0, prior) for prior in priors)
    if total > 0.0:
        priors = [max(0.0, prior) / total for prior in priors]
    elif priors:
        priors = [1.0 / len(priors)] * len(priors)
    return action_ids, priors, root_eval.value, indexes


_ACTION_SIGNATURE_IGNORED_KEYS = {"id", "i", "rank", "score", "prior"}
_ACTION_SIGNATURE_ALIAS_KEYS = {
    "t": "type",
    "u": "unit_id",
    "c": "city_id",
    "p": "tribe_id",
    "tu": "target_unit_id",
    "tc": "target_city_id",
    "tp": "target_player_id",
    "ut": "unit_type",
    "bt": "building_type",
    "rt": "resource_type",
    "ct": "capture_type",
    "b": "bonus",
}


def _freeze_signature_value(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(
            (str(key), _freeze_signature_value(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in _ACTION_SIGNATURE_IGNORED_KEYS
        )
    if isinstance(value, list):
        return tuple(_freeze_signature_value(item) for item in value)
    return value


def action_signature(action: Dict[str, Any]) -> Any:
    normalized = dict(action)
    for alias, canonical in _ACTION_SIGNATURE_ALIAS_KEYS.items():
        if canonical in normalized and alias in normalized:
            normalized.pop(alias, None)
    return _freeze_signature_value(normalized)


def observation_signature(message: Dict[str, Any]) -> str:
    observation = message.get("observation", {})
    return json.dumps(observation, sort_keys=True, separators=(",", ":"), default=str)


def _unique_action_mapping(
    native_actions: List[Dict[str, Any]],
    current_actions: List[Dict[str, Any]],
) -> tuple[List[str], str]:
    current_by_sig: Dict[Any, List[str]] = {}
    for action in current_actions:
        current_by_sig.setdefault(action_signature(action), []).append(str(action.get("id")))
    mapped: List[str] = []
    for native_action in native_actions:
        candidates = current_by_sig.get(action_signature(native_action), [])
        if len(candidates) != 1:
            return [], "ambiguous_action_signature" if candidates else "missing_action_signature"
        mapped.append(candidates[0])
    return mapped, ""


def _tree_root_payload(tree: Any) -> Dict[str, Any]:
    root_payload = getattr(tree, "root_payload", None)
    if root_payload is None:
        return {}
    return dict(_native_call(root_payload))


def _tree_root_action_payloads(tree: Any) -> List[Dict[str, Any]]:
    root_action_payloads = getattr(tree, "root_action_payloads", None)
    if root_action_payloads is None:
        return []
    return [dict(action) for action in list(_native_call(root_action_payloads))]


def _root_visit_distribution_by_index(tree: Any, temperature: float) -> List[float]:
    by_index = getattr(tree, "root_visit_distribution_by_index", None)
    if by_index is not None:
        return [float(value) for value in list(_native_call(by_index, float(temperature)))]
    distribution = {
        str(action_id): float(prob)
        for action_id, prob in _native_call(tree.root_visit_distribution, temperature).items()
    }
    native_actions = _tree_root_action_payloads(tree)
    return [float(distribution.get(str(action.get("id")), 0.0)) for action in native_actions]


def _node_count(tree: Any) -> int:
    node_count = getattr(tree, "node_count", None)
    if node_count is None:
        return 0
    return int(_native_call(node_count))


def _run_native_tree_search(
    tree: Any,
    root_payload: Dict[str, Any],
    root_actions: List[Dict[str, Any]],
    searched_action_ids: List[str],
    native_action_ids: List[str],
    root_value: float,
    evaluator: HybridPolicyValueNet,
    search_cfg: SearchConfig,
    model_cfg: ModelConfig,
    device: torch.device | str,
    belief_snapshot: Any | None = None,
    wall_time_seconds: float | None = None,
    evaluate_fn: Any = None,
) -> _TreeSearchResult:
    evaluate_fn = evaluate_fn or _evaluate_messages
    wall_time_budget = 0.0 if wall_time_seconds is None else max(0.0, float(wall_time_seconds))
    deadline = time.perf_counter() + wall_time_budget if wall_time_budget > 0.0 else None
    simulation_budget = max(0, int(search_cfg.num_simulations))
    batch_size = max(1, int(search_cfg.batch_size))
    reserve_tree_capacity = getattr(tree, "reserve_tree_capacity", None)
    if reserve_tree_capacity is not None:
        reserve_target = simulation_budget + _node_count(tree) + 1 if deadline is None else max(4096, batch_size * 1024)
        _native_call(reserve_tree_capacity, int(reserve_target))

    select_sec = 0.0
    eval_sec = 0.0
    expand_sec = 0.0
    eval_batches = 0
    eval_positions = 0
    eval_cache_hits = 0
    eval_cache: Dict[Any, _Evaluation] = {}
    selected_paths = 0
    expanded_node_ids: set[int] = set()
    depth_sum = 0
    max_selected_depth = 0
    telemetry = _SearchTelemetry()

    while (deadline is not None and time.perf_counter() < deadline) or (deadline is None and len(expanded_node_ids) < simulation_budget):
        selection_frontier = batch_size * 2 if deadline is not None else batch_size
        frontier = selection_frontier if deadline is not None else min(selection_frontier, max(1, simulation_budget - len(expanded_node_ids)))
        selections: List[Any] = []
        eval_messages: List[Dict[str, Any]] = []
        eval_index_by_key: Dict[Any, int] = {}
        select_started_at = time.perf_counter()
        evals_only_batch = getattr(tree, "select_leaf_batch_evals_only", None)
        evals_only_batches = getattr(tree, "select_leaf_batches_evals_only", None)
        completed_frontier = frontier
        if evals_only_batches is not None:
            raw_selections, completed_frontier = _native_call(evals_only_batches, frontier, 1, float(search_cfg.c_puct))
            batch_stats = _native_call(tree.last_batch_stats)
            batch_depth_sum, batch_max_depth = batch_stats[:2]
            depth_sum += int(batch_depth_sum)
            max_selected_depth = max(max_selected_depth, int(batch_max_depth))
        else:
            select_leaf_batch = evals_only_batch or getattr(tree, "select_leaf_batch_compact", tree.select_leaf_batch)
            raw_selections = _native_call(select_leaf_batch, frontier, float(search_cfg.c_puct))
            if evals_only_batch is not None:
                batch_stats = _native_call(tree.last_batch_stats)
                batch_depth_sum, batch_max_depth = batch_stats[:2]
                depth_sum += int(batch_depth_sum)
                max_selected_depth = max(max_selected_depth, int(batch_max_depth))
        for raw_selection in raw_selections:
            if evals_only_batch is not None:
                if len(raw_selection) == 7:
                    selection_id, parent_node_id, parent_action_index, state_key, _selection_depth, _turn_depth, raw_leaf_payload = raw_selection
                elif len(raw_selection) == 6:
                    selection_id, parent_node_id, parent_action_index, state_key, _selection_depth, raw_leaf_payload = raw_selection
                else:
                    selection_id, parent_node_id, parent_action_index, state_key, raw_leaf_payload = raw_selection
                eval_key: Any = int(state_key)
                cached = eval_cache.get(eval_key)
                if cached is not None:
                    eval_cache_hits += 1
                    eval_index = -1
                else:
                    eval_index = eval_index_by_key.get(eval_key, -1)
                    if eval_index < 0:
                        leaf_payload = dict(raw_leaf_payload or {})
                        telemetry.observe(leaf_payload)
                        _raise_on_unsupported_leaf(leaf_payload, search_cfg)
                        if belief_snapshot is not None:
                            leaf_payload = belief_snapshot.annotate_without_update(leaf_payload)
                        eval_index = len(eval_messages)
                        eval_index_by_key[eval_key] = eval_index
                        eval_messages.append(leaf_payload)
                selections.append((int(selection_id), True, int(parent_node_id), int(parent_action_index), 0.0, False, eval_index, cached))
                continue
            if isinstance(raw_selection, tuple):
                (
                    selection_id,
                    needs_expansion,
                    parent_node_id,
                    parent_action_index,
                    leaf_value,
                    leaf_terminal,
                    state_key,
                    raw_leaf_payload,
                ) = raw_selection
                eval_index = -1
                cached: _Evaluation | None = None
                if needs_expansion and not leaf_terminal:
                    eval_key = int(state_key)
                    cached = eval_cache.get(eval_key)
                    if cached is not None:
                        eval_cache_hits += 1
                    else:
                        eval_index = eval_index_by_key.get(eval_key, -1)
                        if eval_index < 0:
                            leaf_payload = dict(raw_leaf_payload or {})
                            telemetry.observe(leaf_payload)
                            _raise_on_unsupported_leaf(leaf_payload, search_cfg)
                            if belief_snapshot is not None:
                                leaf_payload = belief_snapshot.annotate_without_update(leaf_payload)
                            eval_index = len(eval_messages)
                            eval_index_by_key[eval_key] = eval_index
                            eval_messages.append(leaf_payload)
                selections.append((int(selection_id), bool(needs_expansion), int(parent_node_id), int(parent_action_index), float(leaf_value), bool(leaf_terminal), eval_index, cached))
                continue
            selection = dict(raw_selection)
            if selection.get("needs_expansion") and not selection.get("leaf_terminal"):
                eval_key = int(selection.get("state_key", selection.get("selection_id", 0)))
                cached = eval_cache.get(eval_key)
                if cached is not None:
                    selection["_cached_eval"] = cached
                    eval_cache_hits += 1
                else:
                    if eval_key not in eval_index_by_key:
                        leaf_payload = dict(selection.get("leaf_payload") or {})
                        telemetry.observe(leaf_payload)
                        _raise_on_unsupported_leaf(leaf_payload, search_cfg)
                        if belief_snapshot is not None:
                            leaf_payload = belief_snapshot.annotate_without_update(leaf_payload)
                        eval_index_by_key[eval_key] = len(eval_messages)
                        eval_messages.append(leaf_payload)
                    selection["_eval_index"] = eval_index_by_key[eval_key]
                    selection["_eval_key"] = eval_key
            selections.append(selection)
        select_sec += time.perf_counter() - select_started_at
        selected_paths += int(completed_frontier) if evals_only_batch is not None else len(selections)

        eval_started_at = time.perf_counter()
        evaluations = evaluate_fn(eval_messages, evaluator, model_cfg, device)
        eval_sec += time.perf_counter() - eval_started_at
        if eval_messages:
            eval_batches += 1
            eval_positions += len(eval_messages)
            for eval_key, eval_index in eval_index_by_key.items():
                eval_cache[eval_key] = evaluations[eval_index]
        expand_started_at = time.perf_counter()
        completed_selection_ids: List[int] = []
        completed_leaf_values: List[float] = []
        expanded_before_batch = len(expanded_node_ids)
        for selection in selections:
            if isinstance(selection, tuple):
                (
                    selection_id,
                    needs_expansion,
                    parent_node_id,
                    parent_action_index,
                    leaf_value,
                    leaf_terminal,
                    eval_index,
                    cached_eval,
                ) = selection
                if needs_expansion and not leaf_terminal:
                    evaluation = cached_eval if cached_eval is not None else evaluations[int(eval_index)]
                    leaf_value = evaluation.value
                    child_node_id = int(_native_call(tree.expand, int(parent_node_id), int(parent_action_index), evaluation.priors, float(evaluation.value), False))
                    expanded_node_ids.add(child_node_id)
                completed_selection_ids.append(int(selection_id))
                completed_leaf_values.append(float(leaf_value))
                continue
            leaf_value = float(selection.get("leaf_value", 0.0))
            if selection.get("needs_expansion") and not selection.get("leaf_terminal"):
                evaluation = selection.get("_cached_eval")
                if evaluation is None:
                    evaluation = evaluations[int(selection["_eval_index"])]
                leaf_value = evaluation.value
                child_node_id = int(_native_call(tree.expand, int(selection["parent_node_id"]), int(selection["parent_action_index"]), evaluation.priors, float(evaluation.value), False))
                expanded_node_ids.add(child_node_id)
            completed_selection_ids.append(int(selection["selection_id"]))
            completed_leaf_values.append(float(leaf_value))
        if completed_selection_ids:
            _native_call(tree.complete_selected_paths, completed_selection_ids, completed_leaf_values)
        expand_sec += time.perf_counter() - expand_started_at
        if deadline is None and len(expanded_node_ids) == expanded_before_batch:
            break
        if deadline is not None and completed_frontier <= 0 and not selections:
            break

    if selected_paths <= 0:
        visit_probs = [1.0 / len(searched_action_ids)] * len(searched_action_ids) if searched_action_ids else []
    else:
        visit_probs = _root_visit_distribution_by_index(tree, search_cfg.root_temperature)
    visit_distribution = {
        str(action_id): float(visit_probs[index])
        for index, action_id in enumerate(searched_action_ids)
        if index < len(visit_probs)
    }
    if search_cfg.sample_action:
        candidates = list(visit_distribution.keys())
        weights = [visit_distribution[action_id] for action_id in candidates]
        action_id = random.choices(candidates, weights=weights, k=1)[0]
    else:
        action_id = max(visit_distribution, key=visit_distribution.get)
    action_id = _apply_end_turn_visit_guard(action_id, visit_distribution, root_actions, search_cfg)

    selected_local_index = searched_action_ids.index(action_id) if action_id in searched_action_ids else 0
    native_action_id = native_action_ids[selected_local_index] if selected_local_index < len(native_action_ids) else action_id
    root_action_ids = [str(action.get("id")) for action in root_actions]
    action_index = root_action_ids.index(action_id) if action_id in root_action_ids else 0
    visit_target = [float(visit_distribution.get(candidate_id, 0.0)) for candidate_id in root_action_ids]
    avg_depth = float(depth_sum) / max(1, selected_paths)
    expanded_nodes = len(expanded_node_ids)
    root_visit_entropy = _visit_entropy_bits(visit_distribution)
    top_visit_share = max((float(share) for share in visit_distribution.values()), default=0.0)
    _profile_search_log(
        f"mcts sims={int(search_cfg.num_simulations)} batch={batch_size} "
        f"paths={selected_paths} nodes={expanded_nodes} eval_batches={eval_batches} eval_positions={eval_positions} "
        f"eval_cache_hits={eval_cache_hits} eval_cache_size={len(eval_cache)} "
        f"avg_depth={avg_depth:.3f} selected_max_depth={max_selected_depth} "
        f"root_visit_entropy={root_visit_entropy:.6f} top_visit_share={top_visit_share:.6f} "
        f"select_ms={select_sec * 1000.0:.1f} eval_ms={eval_sec * 1000.0:.1f} "
        f"expand_ms={expand_sec * 1000.0:.1f} total_inner_ms={(select_sec + eval_sec + expand_sec) * 1000.0:.1f} "
        f"root_actions={len(root_actions)} searched_actions={len(searched_action_ids)} {telemetry.log_fields()}"
    )
    return _TreeSearchResult(
        SearchResult(action_id=action_id, action_index=action_index, visit_distribution=visit_distribution, visit_target=visit_target, value=float(root_value)),
        native_action_id=str(native_action_id),
        selected_local_index=int(selected_local_index),
        selected_paths=int(selected_paths),
        expanded_nodes=int(expanded_nodes),
        max_selected_depth=int(max_selected_depth),
    )


class ReusableNativeMCTSSession:
    def __init__(self) -> None:
        self.tree: Any | None = None
        self.root_value: float = 0.0
        self.stats = TreeReuseStats()

    def reset(self, reason: str = "") -> None:
        self.tree = None
        if reason:
            self.stats.last_miss_reason = reason

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
        return _root_priors(root_payload, evaluator, search_cfg, model_cfg, device, root_policy_logits, root_value, belief_snapshot)

    def _evaluate_messages(
        self,
        messages: List[Dict[str, Any]],
        evaluator: HybridPolicyValueNet,
        model_cfg: ModelConfig,
        device: torch.device | str,
    ) -> List[_Evaluation]:
        return _evaluate_messages(messages, evaluator, model_cfg, device)

    def _try_reuse(self, root_payload: Dict[str, Any], root_actions: List[Dict[str, Any]]) -> tuple[List[str], List[str]] | None:
        if self.tree is None:
            return None
        self.stats.attempts += 1
        promoted_payload = _tree_root_payload(self.tree)
        if not promoted_payload:
            self.stats.misses += 1
            self.reset("missing_promoted_payload")
            return None
        if int(promoted_payload.get("player_id", -1)) != int(root_payload.get("player_id", -2)):
            self.stats.misses += 1
            self.reset("player_changed")
            return None
        if observation_signature(promoted_payload) != observation_signature(root_payload):
            self.stats.misses += 1
            self.reset("observation_mismatch")
            return None
        native_actions = _tree_root_action_payloads(self.tree)
        mapped_ids, reason = _unique_action_mapping(native_actions, root_actions)
        if reason:
            self.stats.misses += 1
            self.reset(reason)
            return None
        self.stats.hits += 1
        self.stats.reused_nodes += _node_count(self.tree)
        native_ids = [str(action.get("id")) for action in native_actions]
        return mapped_ids, native_ids

    def _construct_tree(
        self,
        extension: Any,
        root_payload: Dict[str, Any],
        root_indexes: List[int],
        priors: List[float],
        root_value: float,
        search_cfg: SearchConfig,
        model_cfg: ModelConfig,
    ) -> Any:
        seed = int(getattr(search_cfg, "seed", 0) or int(time.time_ns() & 0xFFFFFFFF))
        tree = _native_call(
            extension.NativeMCTS,
            root_payload,
            root_indexes,
            priors,
            float(root_value),
            bool(root_payload.get("is_terminal", False) or root_payload.get("terminal", False)),
            seed,
            int(model_cfg.max_actions),
            bool(getattr(search_cfg, "use_progressive_widening", True)),
        )
        tree.add_root_dirichlet_noise(float(search_cfg.dirichlet_alpha), float(search_cfg.dirichlet_epsilon))
        return tree

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
        root_payload = normalize_message(root_payload)
        extension = load_native_mcts_extension()
        if extension is None:
            raise NativeSearchUnavailable("Native MCTS extension is unavailable.")
        root_actions = list(root_payload.get("actions", []))[: model_cfg.max_actions]
        if not root_actions:
            self.reset("no_actions")
            return SearchResult("", 0, {}, [], 0.0)

        reuse_mapping = self._try_reuse(root_payload, root_actions) if bool(getattr(search_cfg, "reuse_tree", False)) else None
        if reuse_mapping is None:
            action_ids, priors, self.root_value, root_indexes = self._root_priors(
                root_payload,
                evaluator,
                search_cfg,
                model_cfg,
                device,
                root_policy_logits,
                root_value,
                belief_snapshot,
            )
            if not action_ids:
                self.reset("no_searched_actions")
                return SearchResult("", 0, {}, [0.0] * len(root_actions), self.root_value)
            if len(action_ids) == 1:
                self.reset("single_action")
                only_action_id = action_ids[0]
                root_action_ids = [str(action.get("id")) for action in root_actions]
                action_index = root_action_ids.index(only_action_id) if only_action_id in root_action_ids else root_indexes[0]
                return SearchResult(only_action_id, action_index, {only_action_id: 1.0}, [1.0 if candidate_id == only_action_id else 0.0 for candidate_id in root_action_ids], float(self.root_value))
            self.tree = self._construct_tree(extension, root_payload, root_indexes, priors, float(self.root_value), search_cfg, model_cfg)
            searched_action_ids = action_ids
            native_action_ids = [str(root_actions[index].get("id")) for index in root_indexes]
        else:
            searched_action_ids, native_action_ids = reuse_mapping

        tree_result = _run_native_tree_search(
            self.tree,
            root_payload,
            root_actions,
            searched_action_ids,
            native_action_ids,
            float(self.root_value),
            evaluator,
            search_cfg,
            model_cfg,
            device,
            belief_snapshot,
            wall_time_seconds,
            self._evaluate_messages,
        )
        if bool(getattr(search_cfg, "reuse_tree", False)) and self.tree is not None:
            selected_action = root_actions[tree_result.result.action_index] if tree_result.result.action_index < len(root_actions) else {}
            if str(selected_action.get("type")) == "END_TURN":
                self.reset("selected_end_turn")
            else:
                promote = getattr(self.tree, "promote_root_child_by_action_id", None)
                if promote is None:
                    self.reset("native_promotion_unavailable")
                else:
                    info = dict(_native_call(promote, tree_result.native_action_id))
                    if bool(info.get("ok")):
                        self.stats.promoted_subtree_nodes += int(info.get("promoted_subtree_nodes", 0) or 0)
                    else:
                        self.reset(str(info.get("reason") or "promotion_failed"))
        else:
            self.reset()
        return tree_result.result


def run_native_mcts(
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
    root_payload = normalize_message(root_payload)
    extension = load_native_mcts_extension()
    if extension is None:
        raise NativeSearchUnavailable("Native MCTS extension is unavailable.")

    root_actions = list(root_payload.get("actions", []))[: model_cfg.max_actions]
    if not root_actions:
        return SearchResult("", 0, {}, [], 0.0)

    action_ids, priors, root_value, root_indexes = _root_priors(
        root_payload,
        evaluator,
        search_cfg,
        model_cfg,
        device,
        root_policy_logits,
        root_value,
        belief_snapshot,
    )
    if not action_ids:
        return SearchResult("", 0, {}, [0.0] * len(root_actions), root_value)
    if len(action_ids) == 1:
        only_action_id = action_ids[0]
        root_action_ids = [str(action.get("id")) for action in root_actions]
        action_index = root_action_ids.index(only_action_id) if only_action_id in root_action_ids else root_indexes[0]
        visit_distribution = {only_action_id: 1.0}
        visit_target = [1.0 if candidate_id == only_action_id else 0.0 for candidate_id in root_action_ids]
        _profile_search_log(
            f"mcts sims=0 batch={int(search_cfg.batch_size)} "
            f"paths=0 eval_batches=0 eval_positions=0 eval_cache_hits=0 eval_cache_size=0 "
            f"avg_depth=0.000 selected_max_depth=0 root_visit_entropy=0.000000 top_visit_share=1.000000 "
            f"select_ms=0.0 eval_ms=0.0 expand_ms=0.0 total_inner_ms=0.0 "
            f"root_actions={len(root_actions)} searched_actions=1 native_unsupported_transition=0 "
            f"native_invalid_transition=0 native_approximate_transition=0"
        )
        return SearchResult(
            action_id=only_action_id,
            action_index=action_index,
            visit_distribution=visit_distribution,
            visit_target=visit_target,
            value=float(root_value),
        )

    seed = int(getattr(search_cfg, "seed", 0) or int(time.time_ns() & 0xFFFFFFFF))
    tree = _native_call(
        extension.NativeMCTS,
        root_payload,
        root_indexes,
        priors,
        float(root_value),
        bool(root_payload.get("is_terminal", False) or root_payload.get("terminal", False)),
        seed,
        int(model_cfg.max_actions),
        bool(getattr(search_cfg, "use_progressive_widening", True)),
    )
    tree.add_root_dirichlet_noise(float(search_cfg.dirichlet_alpha), float(search_cfg.dirichlet_epsilon))
    wall_time_budget = 0.0 if wall_time_seconds is None else max(0.0, float(wall_time_seconds))
    deadline = time.perf_counter() + wall_time_budget if wall_time_budget > 0.0 else None
    simulation_budget = max(0, int(search_cfg.num_simulations))
    batch_size = max(1, int(search_cfg.batch_size))
    reserve_tree_capacity = getattr(tree, "reserve_tree_capacity", None)
    if reserve_tree_capacity is not None:
        reserve_target = simulation_budget + 1 if deadline is None else max(4096, batch_size * 1024)
        _native_call(reserve_tree_capacity, int(reserve_target))
    select_sec = 0.0
    eval_sec = 0.0
    expand_sec = 0.0
    eval_batches = 0
    eval_positions = 0
    eval_cache_hits = 0
    eval_cache: Dict[Any, _Evaluation] = {}
    selected_paths = 0
    expanded_node_ids: set[int] = set()
    depth_sum = 0
    max_selected_depth = 0
    telemetry = _SearchTelemetry()

    while (deadline is not None and time.perf_counter() < deadline) or (deadline is None and len(expanded_node_ids) < simulation_budget):
        selection_frontier = batch_size * 2 if deadline is not None else batch_size
        if deadline is not None:
            frontier = selection_frontier
        else:
            frontier = min(selection_frontier, max(1, simulation_budget - len(expanded_node_ids)))
        selections: List[Any] = []
        eval_messages: List[Dict[str, Any]] = []
        eval_index_by_key: Dict[Any, int] = {}
        select_started_at = time.perf_counter()
        evals_only_batch = getattr(tree, "select_leaf_batch_evals_only", None)
        evals_only_batches = getattr(tree, "select_leaf_batches_evals_only", None)
        completed_frontier = frontier
        if evals_only_batches is not None:
            raw_selections, completed_frontier = _native_call(evals_only_batches, frontier, 1, float(search_cfg.c_puct))
            batch_stats = _native_call(tree.last_batch_stats)
            batch_depth_sum, batch_max_depth = batch_stats[:2]
            depth_sum += int(batch_depth_sum)
            max_selected_depth = max(max_selected_depth, int(batch_max_depth))
        else:
            select_leaf_batch = evals_only_batch or getattr(tree, "select_leaf_batch_compact", tree.select_leaf_batch)
            raw_selections = _native_call(select_leaf_batch, frontier, float(search_cfg.c_puct))
            if evals_only_batch is not None:
                batch_stats = _native_call(tree.last_batch_stats)
                batch_depth_sum, batch_max_depth = batch_stats[:2]
                depth_sum += int(batch_depth_sum)
                max_selected_depth = max(max_selected_depth, int(batch_max_depth))
        for raw_selection in raw_selections:
            if evals_only_batch is not None:
                if len(raw_selection) == 7:
                    selection_id, parent_node_id, parent_action_index, state_key, _selection_depth, _turn_depth, raw_leaf_payload = raw_selection
                elif len(raw_selection) == 6:
                    selection_id, parent_node_id, parent_action_index, state_key, _selection_depth, raw_leaf_payload = raw_selection
                else:
                    selection_id, parent_node_id, parent_action_index, state_key, raw_leaf_payload = raw_selection
                eval_key: Any = int(state_key)
                cached = eval_cache.get(eval_key)
                if cached is not None:
                    eval_cache_hits += 1
                    eval_index = -1
                else:
                    eval_index = eval_index_by_key.get(eval_key, -1)
                    if eval_index < 0:
                        leaf_payload = dict(raw_leaf_payload or {})
                        telemetry.observe(leaf_payload)
                        _raise_on_unsupported_leaf(leaf_payload, search_cfg)
                        if belief_snapshot is not None:
                            leaf_payload = belief_snapshot.annotate_without_update(leaf_payload)
                        eval_index = len(eval_messages)
                        eval_index_by_key[eval_key] = eval_index
                        eval_messages.append(leaf_payload)
                selections.append(
                    (
                        int(selection_id),
                        True,
                        int(parent_node_id),
                        int(parent_action_index),
                        0.0,
                        False,
                        eval_index,
                        cached,
                    )
                )
                continue
            if isinstance(raw_selection, tuple):
                (
                    selection_id,
                    needs_expansion,
                    parent_node_id,
                    parent_action_index,
                    leaf_value,
                    leaf_terminal,
                    state_key,
                    raw_leaf_payload,
                ) = raw_selection
                eval_index = -1
                cached: _Evaluation | None = None
                if needs_expansion and not leaf_terminal:
                    eval_key: Any = int(state_key)
                    cached = eval_cache.get(eval_key)
                    if cached is not None:
                        eval_cache_hits += 1
                    else:
                        eval_index = eval_index_by_key.get(eval_key, -1)
                        if eval_index < 0:
                            leaf_payload = dict(raw_leaf_payload or {})
                            telemetry.observe(leaf_payload)
                            _raise_on_unsupported_leaf(leaf_payload, search_cfg)
                            if belief_snapshot is not None:
                                leaf_payload = belief_snapshot.annotate_without_update(leaf_payload)
                            eval_index = len(eval_messages)
                            eval_index_by_key[eval_key] = eval_index
                            eval_messages.append(leaf_payload)
                selections.append(
                    (
                        int(selection_id),
                        bool(needs_expansion),
                        int(parent_node_id),
                        int(parent_action_index),
                        float(leaf_value),
                        bool(leaf_terminal),
                        eval_index,
                        cached,
                    )
                )
                continue
            else:
                selection = dict(raw_selection)
            if selection.get("needs_expansion") and not selection.get("leaf_terminal"):
                eval_key: Any = int(selection.get("state_key", selection.get("selection_id", 0)))
                cached = eval_cache.get(eval_key)
                if cached is not None:
                    selection["_cached_eval"] = cached
                    eval_cache_hits += 1
                else:
                    if eval_key not in eval_index_by_key:
                        leaf_payload = dict(selection.get("leaf_payload") or {})
                        telemetry.observe(leaf_payload)
                        _raise_on_unsupported_leaf(leaf_payload, search_cfg)
                        if belief_snapshot is not None:
                            leaf_payload = belief_snapshot.annotate_without_update(leaf_payload)
                        eval_index_by_key[eval_key] = len(eval_messages)
                        eval_messages.append(leaf_payload)
                        selection["_eval_index"] = eval_index_by_key[eval_key]
                    elif eval_key in eval_index_by_key:
                        selection["_eval_index"] = eval_index_by_key[eval_key]
                    selection["_eval_key"] = eval_key
            selections.append(selection)
        select_sec += time.perf_counter() - select_started_at
        selected_paths += int(completed_frontier) if evals_only_batch is not None else len(selections)

        eval_started_at = time.perf_counter()
        evaluations = _evaluate_messages(eval_messages, evaluator, model_cfg, device)
        eval_sec += time.perf_counter() - eval_started_at
        if eval_messages:
            eval_batches += 1
            eval_positions += len(eval_messages)
            for eval_key, eval_index in eval_index_by_key.items():
                eval_cache[eval_key] = evaluations[eval_index]
        expand_started_at = time.perf_counter()
        completed_selection_ids: List[int] = []
        completed_leaf_values: List[float] = []
        expanded_before_batch = len(expanded_node_ids)
        for selection in selections:
            if isinstance(selection, tuple):
                (
                    selection_id,
                    needs_expansion,
                    parent_node_id,
                    parent_action_index,
                    leaf_value,
                    leaf_terminal,
                    eval_index,
                    cached_eval,
                ) = selection
                if needs_expansion and not leaf_terminal:
                    evaluation = cached_eval if cached_eval is not None else evaluations[int(eval_index)]
                    leaf_value = evaluation.value
                    child_node_id = int(_native_call(
                        tree.expand,
                        int(parent_node_id),
                        int(parent_action_index),
                        evaluation.priors,
                        float(evaluation.value),
                        False,
                    ))
                    expanded_node_ids.add(child_node_id)
                completed_selection_ids.append(int(selection_id))
                completed_leaf_values.append(float(leaf_value))
                continue

            leaf_value = float(selection.get("leaf_value", 0.0))
            if selection.get("needs_expansion") and not selection.get("leaf_terminal"):
                evaluation = selection.get("_cached_eval")
                if evaluation is None:
                    evaluation = evaluations[int(selection["_eval_index"])]
                leaf_value = evaluation.value
                child_node_id = int(_native_call(
                    tree.expand,
                    int(selection["parent_node_id"]),
                    int(selection["parent_action_index"]),
                    evaluation.priors,
                    float(evaluation.value),
                    False,
                ))
                expanded_node_ids.add(child_node_id)
            completed_selection_ids.append(int(selection["selection_id"]))
            completed_leaf_values.append(float(leaf_value))
        _native_call(tree.complete_selected_paths, completed_selection_ids, completed_leaf_values)
        expand_sec += time.perf_counter() - expand_started_at
        if deadline is None and len(expanded_node_ids) == expanded_before_batch:
            break
        if deadline is not None and completed_frontier <= 0 and not selections:
            break

    if selected_paths <= 0:
        visit_distribution = {str(action_id): float(prior) for action_id, prior in zip(action_ids, priors)}
    else:
        visit_distribution = {str(action_id): float(prob) for action_id, prob in _native_call(tree.root_visit_distribution, search_cfg.root_temperature).items()}
    if search_cfg.sample_action:
        candidates = list(visit_distribution.keys())
        weights = [visit_distribution[action_id] for action_id in candidates]
        action_id = random.choices(candidates, weights=weights, k=1)[0]
    else:
        action_id = max(visit_distribution, key=visit_distribution.get)
    action_id = _apply_end_turn_visit_guard(action_id, visit_distribution, root_actions, search_cfg)

    root_action_ids = [str(action.get("id")) for action in root_actions]
    action_index = root_action_ids.index(action_id) if action_id in root_action_ids else 0
    visit_target = [float(visit_distribution.get(candidate_id, 0.0)) for candidate_id in root_action_ids]
    avg_depth = float(depth_sum) / max(1, selected_paths)
    expanded_nodes = len(expanded_node_ids)
    root_visit_entropy = _visit_entropy_bits(visit_distribution)
    top_visit_share = max((float(share) for share in visit_distribution.values()), default=0.0)
    _profile_search_log(
        f"mcts sims={int(search_cfg.num_simulations)} batch={batch_size} "
        f"paths={selected_paths} nodes={expanded_nodes} eval_batches={eval_batches} eval_positions={eval_positions} "
        f"eval_cache_hits={eval_cache_hits} eval_cache_size={len(eval_cache)} "
        f"avg_depth={avg_depth:.3f} selected_max_depth={max_selected_depth} "
        f"root_visit_entropy={root_visit_entropy:.6f} top_visit_share={top_visit_share:.6f} "
        f"select_ms={select_sec * 1000.0:.1f} eval_ms={eval_sec * 1000.0:.1f} "
        f"expand_ms={expand_sec * 1000.0:.1f} total_inner_ms={(select_sec + eval_sec + expand_sec) * 1000.0:.1f} "
        f"root_actions={len(root_actions)} searched_actions={len(root_indexes)} {telemetry.log_fields()}"
    )
    return SearchResult(
        action_id=action_id,
        action_index=action_index,
        visit_distribution=visit_distribution,
        visit_target=visit_target,
        value=float(root_value),
    )
