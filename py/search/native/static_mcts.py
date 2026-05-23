from __future__ import annotations

from typing import Any, Dict, List
import random
import time

from search.config import ModelConfig, SearchConfig
from nn.encoding import normalize_message
from .cpp_extension import load_native_mcts_extension
from .mcts import (
    NativeSearchParityError,
    NativeSearchUnavailable,
    SearchResult,
    _Evaluation,
    _SearchTelemetry,
    _apply_end_turn_visit_guard,
    _native_call,
)


def _coerce_static_evaluation(raw: Dict[str, Any], action_count: int) -> _Evaluation:
    raw_priors = list(raw.get("priors", []))
    priors = [float(value) for value in raw_priors[:action_count]]
    if len(priors) != action_count:
        raise NativeSearchParityError(
            f"Static eval prior/action mismatch: {len(raw_priors)} priors for {action_count} actions"
        )
    total = sum(max(0.0, prior) for prior in priors)
    if total > 0.0:
        priors = [max(0.0, prior) / total for prior in priors]
    else:
        raise NativeSearchParityError("Static eval returned non-positive total prior mass.")
    return _Evaluation(priors=priors, value=float(raw.get("value", 0.0)))


def _evaluate_static_messages(messages: List[Dict[str, Any]], max_actions: int) -> List[_Evaluation]:
    if not messages:
        return []
    extension = load_native_mcts_extension()
    if extension is None:
        raise NativeSearchUnavailable("Native MCTS extension is unavailable.")
    raw_evaluations = list(extension.evaluate_static_batch(messages, int(max_actions)))
    if len(raw_evaluations) != len(messages):
        raise NativeSearchUnavailable(
            f"Static evaluator returned {len(raw_evaluations)} evaluations for {len(messages)} messages."
        )
    evaluations: List[_Evaluation] = []
    for message, raw in zip(messages, raw_evaluations):
        evaluations.append(_coerce_static_evaluation(dict(raw), len(list(message.get("actions", []))) if message else 0))
    return evaluations


_ROOT_ALWAYS_KEEP_TYPES = {
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
    "RECOVER",
    "HEAL_OTHERS",
    "UPGRADE_SHIP",
    "UPGRADE_BOAT",
    "UPGRADE_RAMMER",
    "UPGRADE_SCOUT",
    "UPGRADE_BOMBER",
}


def _action_type(action: Dict[str, Any]) -> str:
    payload = action.get("payload") if isinstance(action.get("payload"), dict) else {}
    return str(action.get("type") or action.get("t") or payload.get("type") or payload.get("t") or "")


def _select_root_indexes(root_actions: List[Dict[str, Any]], priors: List[float], top_k_actions: int) -> List[int]:
    indexes = list(range(len(root_actions)))
    if top_k_actions <= 0 or len(indexes) <= top_k_actions:
        return indexes

    forced = {idx for idx, action in enumerate(root_actions) if _action_type(action) in _ROOT_ALWAYS_KEEP_TYPES}
    budget = max(int(top_k_actions), len(forced))
    ranked = sorted(indexes, key=lambda idx: priors[idx], reverse=True)
    selected = set(forced)
    for idx in ranked:
        if len(selected) >= budget:
            break
        selected.add(idx)
    return sorted(selected)


def _root_static_priors(
    message: Dict[str, Any],
    search_cfg: SearchConfig,
    model_cfg: ModelConfig,
) -> tuple[List[str], List[float], float, List[int]]:
    all_root_actions = list(message.get("actions", []))
    root_actions = all_root_actions if int(model_cfg.max_actions) < 0 else all_root_actions[: model_cfg.max_actions]
    if not root_actions:
        return [], [], 0.0, []
    root_eval = _evaluate_static_messages(
        [{"player_id": int(message["player_id"]), "observation": message["observation"], "actions": root_actions}],
        model_cfg.max_actions,
    )[0]
    indexes = _select_root_indexes(root_actions, root_eval.priors, int(search_cfg.top_k_actions))
    action_ids = [str(root_actions[index].get("id")) for index in indexes]
    priors = [root_eval.priors[index] for index in indexes]
    total = sum(max(0.0, prior) for prior in priors)
    if total > 0.0:
        priors = [max(0.0, prior) / total for prior in priors]
    elif priors:
        priors = [1.0 / len(priors)] * len(priors)
    return action_ids, priors, root_eval.value, indexes


def run_native_static_mcts(
    root_payload: Dict[str, Any],
    search_cfg: SearchConfig,
    model_cfg: ModelConfig,
    wall_time_seconds: float | None = None,
) -> SearchResult:
    root_payload = normalize_message(root_payload)
    extension = load_native_mcts_extension()
    if extension is None:
        raise NativeSearchUnavailable("Native MCTS extension is unavailable.")

    all_root_actions = list(root_payload.get("actions", []))
    root_actions = all_root_actions if int(model_cfg.max_actions) < 0 else all_root_actions[: model_cfg.max_actions]
    if not root_actions:
        return SearchResult("", 0, {}, [], 0.0)

    action_ids, priors, root_value, root_indexes = _root_static_priors(root_payload, search_cfg, model_cfg)
    if not action_ids:
        return SearchResult("", 0, {}, [0.0] * len(root_actions), root_value)
    if len(action_ids) == 1:
        only_action_id = action_ids[0]
        root_action_ids = [str(action.get("id")) for action in root_actions]
        action_index = root_action_ids.index(only_action_id) if only_action_id in root_action_ids else root_indexes[0]
        return SearchResult(
            action_id=only_action_id,
            action_index=action_index,
            visit_distribution={only_action_id: 1.0},
            visit_target=[1.0 if candidate_id == only_action_id else 0.0 for candidate_id in root_action_ids],
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
    )
    tree.add_root_dirichlet_noise(float(search_cfg.dirichlet_alpha), float(search_cfg.dirichlet_epsilon))
    max_depth = -1 if int(search_cfg.max_depth) <= 0 else int(search_cfg.max_depth)
    wall_time_budget = 0.0 if wall_time_seconds is None else max(0.0, float(wall_time_seconds))
    deadline = time.perf_counter() + wall_time_budget if wall_time_budget > 0.0 else None
    simulation_budget = max(0, int(search_cfg.num_simulations))
    batch_size = max(1, int(search_cfg.batch_size))
    eval_cache: Dict[Any, _Evaluation] = {}
    selected_paths = 0
    expanded_node_ids: set[int] = set()
    telemetry = _SearchTelemetry()

    while (deadline is not None and time.perf_counter() < deadline) or (deadline is None and len(expanded_node_ids) < simulation_budget):
        frontier = batch_size if deadline is not None else min(batch_size, max(1, simulation_budget - len(expanded_node_ids)))
        selections: List[tuple[int, int, int, bool, _Evaluation | None]] = []
        eval_messages: List[Dict[str, Any]] = []
        eval_index_by_key: Dict[Any, int] = {}
        evals_only_batches = getattr(tree, "select_leaf_batches_evals_only", None)
        completed_frontier = frontier
        if evals_only_batches is not None:
            raw_selections, completed_frontier = _native_call(evals_only_batches, frontier, 1, max_depth, float(search_cfg.c_puct))
        else:
            raw_selections = _native_call(tree.select_leaf_batch_evals_only, frontier, max_depth, float(search_cfg.c_puct))

        for raw_selection in raw_selections:
            if len(raw_selection) == 6:
                selection_id, parent_node_id, parent_action_index, state_key, _selection_depth, raw_leaf_payload = raw_selection
            else:
                selection_id, parent_node_id, parent_action_index, state_key, raw_leaf_payload = raw_selection
            leaf_payload = dict(raw_leaf_payload or {})
            leaf_is_terminal = bool(leaf_payload.get("is_terminal", False) or leaf_payload.get("terminal", False))
            eval_key: Any = int(state_key)
            cached = eval_cache.get(eval_key)
            eval_index = -1
            if cached is None:
                eval_index = eval_index_by_key.get(eval_key, -1)
                if eval_index < 0:
                    telemetry.observe(leaf_payload)
                    kind = str(leaf_payload.get("native_transition_kind", ""))
                    reason = str(leaf_payload.get("native_terminal_reason", ""))
                    if kind == "native_unsupported_transition" or reason.startswith("unsupported_native_transition:"):
                        raise NativeSearchParityError(
                            "Native static MCTS encountered unsupported transition "
                            f"kind={kind or '<empty>'} reason={reason or '<empty>'}"
                        )
                    if bool(leaf_payload.get("native_approximate_transition", False)):
                        raise NativeSearchParityError(
                            "Native static MCTS encountered approximate transition "
                            f"reason={reason or '<empty>'}"
                        )
                    if not bool(leaf_payload.get("observation_perspective_valid", True)):
                        raise NativeSearchParityError(
                            "Native static MCTS encountered invalid opponent perspective leaf "
                            f"root_player_id={leaf_payload.get('root_player_id')} "
                            f"active_player_id={leaf_payload.get('active_player_id')}"
                        )
                    eval_index = len(eval_messages)
                    eval_index_by_key[eval_key] = eval_index
                    eval_messages.append(leaf_payload)
            selections.append(
                (
                    int(selection_id),
                    int(parent_node_id),
                    int(parent_action_index),
                    int(eval_index),
                    leaf_is_terminal,
                    cached,
                )
            )

        selected_paths += int(completed_frontier)
        evaluations = _evaluate_static_messages(eval_messages, model_cfg.max_actions)
        for eval_key, eval_index in eval_index_by_key.items():
            eval_cache[eval_key] = evaluations[eval_index]

        completed_selection_ids: List[int] = []
        completed_leaf_values: List[float] = []
        expanded_before_batch = len(expanded_node_ids)
        for selection_id, parent_node_id, parent_action_index, eval_index, leaf_is_terminal, cached_eval in selections:
            evaluation = cached_eval if cached_eval is not None else evaluations[int(eval_index)]
            child_node_id = int(_native_call(
                tree.expand,
                int(parent_node_id),
                int(parent_action_index),
                evaluation.priors,
                float(evaluation.value),
                bool(leaf_is_terminal),
            ))
            expanded_node_ids.add(child_node_id)
            completed_selection_ids.append(int(selection_id))
            completed_leaf_values.append(float(evaluation.value))
        if completed_selection_ids:
            _native_call(tree.complete_selected_paths, completed_selection_ids, completed_leaf_values)

        if deadline is None and len(expanded_node_ids) == expanded_before_batch:
            break
        if deadline is not None and completed_frontier <= 0 and not selections:
            break

    if selected_paths <= 0:
        visit_distribution = {str(action_id): float(prior) for action_id, prior in zip(action_ids, priors)}
    else:
        visit_distribution = {
            str(action_id): float(prob)
            for action_id, prob in _native_call(tree.root_visit_distribution, search_cfg.root_temperature).items()
        }
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
    return SearchResult(
        action_id=action_id,
        action_index=action_index,
        visit_distribution=visit_distribution,
        visit_target=visit_target,
        value=float(root_value),
    )
