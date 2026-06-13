from __future__ import annotations

import os
import time
from dataclasses import asdict
from typing import Any

from profiling.analysis import ANALYSIS_VERSION
from profiling.analysis.actions import action_field, action_fingerprint, action_id, action_type, top_mass_keys
from profiling.analysis.distributions import rank_map
from profiling.analysis.schema import ActionAnalysis, PositionAnalysis
from search.config import ModelConfig, SearchConfig
from search.native.static_mcts import _evaluate_static_messages, run_native_static_mcts
from search.native.cpp_extension import load_native_mcts_extension


def parse_target(spec: str) -> dict[str, str]:
    name, _, rest = spec.partition(":")
    target = {"name": name or spec, "variant": "baseline", "mcts_impl": "native_static"}
    for part in rest.split(","):
        if not part:
            continue
        key, _, value = part.partition("=")
        if key and value:
            target[key.strip()] = value.strip()
    return target


def _static_breakdown(payload: dict[str, Any], max_actions: int) -> dict[str, Any] | None:
    extension = load_native_mcts_extension()
    fn = getattr(extension, "evaluate_static_breakdown", None) if extension is not None else None
    if fn is None:
        return None
    return dict(fn(payload, int(max_actions))).get("value_breakdown")


def analyze_position(
    payload: dict[str, Any],
    *,
    payload_hash: str,
    label: str,
    target: dict[str, str],
    simulations: int,
    batch_size: int,
    top_k_actions: int,
    max_actions: int,
    seed: int = 0,
    c_puct: float = 1.5,
    include_breakdown: bool = True,
) -> PositionAnalysis:
    previous_variant = os.environ.get("TRIBES_STATIC_EVAL_VARIANT")
    os.environ["TRIBES_STATIC_EVAL_VARIANT"] = str(target.get("variant", "baseline"))
    model_cfg = ModelConfig(max_actions=int(max_actions))
    search_cfg = SearchConfig()
    search_cfg.num_simulations = int(simulations)
    search_cfg.batch_size = int(batch_size)
    search_cfg.c_puct = float(c_puct)
    search_cfg.top_k_actions = int(top_k_actions)
    search_cfg.sample_action = False
    search_cfg.root_temperature = 1.0
    search_cfg.dirichlet_epsilon = 0.0
    setattr(search_cfg, "seed", int(seed))
    try:
        actions = list(payload.get("actions", [])) if max_actions < 0 else list(payload.get("actions", []))[:max_actions]
        priors = _evaluate_static_messages(
            [{"player_id": int(payload["player_id"]), "observation": payload["observation"], "actions": actions}],
            int(max_actions),
        )[0].priors
        started_at = time.perf_counter()
        result = run_native_static_mcts(payload, search_cfg, model_cfg)
        search_sec = time.perf_counter() - started_at
        stats_by_id = {str(row.get("action_id")): row for row in (result.root_stats or [])}
        visit_scores = [float(result.visit_distribution.get(action_id(action), 0.0)) for action in actions]
        prior_scores = [float(priors[index]) if index < len(priors) else 0.0 for index in range(len(actions))]
        visit_ranks = rank_map(visit_scores)
        prior_ranks = rank_map(prior_scores)
        top95 = top_mass_keys({action_id(action): visit_scores[index] for index, action in enumerate(actions)})
        selected_action = next((action for action in actions if action_id(action) == result.action_id), {})
        action_rows: list[ActionAnalysis] = []
        for index, action in enumerate(actions):
            aid = action_id(action)
            stat = stats_by_id.get(aid, {})
            action_rows.append(
                ActionAnalysis(
                    action_id=aid,
                    action_fingerprint=action_fingerprint(action),
                    action_index=index,
                    action_type=action_type(action),
                    prior=prior_scores[index],
                    prior_rank=prior_ranks.get(index, 0),
                    visits=int(stat["visits"]) if "visits" in stat else None,
                    visit_share=float(stat.get("visit_share", visit_scores[index])),
                    visit_rank=visit_ranks.get(index, 0),
                    q_mean=float(stat["q_mean"]) if "q_mean" in stat else None,
                    value_sum=float(stat["value_sum"]) if "value_sum" in stat else None,
                    in_top95=aid in top95 or aid == result.action_id,
                    unit_id=action_field(action, "unit_id", "unitId", "u"),
                    city_id=action_field(action, "city_id", "cityId", "c"),
                    x=action_field(action, "x"),
                    y=action_field(action, "y"),
                    target=action_field(action, "target_unit_id", "targetUnitId", "target_id", "targetId", "tu"),
                )
            )
        return PositionAnalysis(
            analysis_version=ANALYSIS_VERSION,
            payload_hash=payload_hash,
            label=label,
            target_name=str(target.get("name", "")),
            evaluator="static",
            mcts_impl=str(target.get("mcts_impl", "native_static")),
            static_eval_variant=str(target.get("variant", "baseline")),
            seed=int(seed),
            simulations=int(simulations),
            c_puct=float(c_puct),
            top_k_actions=int(top_k_actions),
            max_actions=int(max_actions),
            root_value=float(result.value),
            selected_action_id=str(result.action_id),
            selected_action_fingerprint=action_fingerprint(selected_action) if selected_action else "",
            action_count_raw=len(list(payload.get("actions", []))),
            action_count_analyzed=len(actions),
            search_sec=search_sec,
            actions=action_rows,
            value_breakdown=_static_breakdown(payload, int(max_actions)) if include_breakdown else None,
        )
    finally:
        if previous_variant is None:
            os.environ.pop("TRIBES_STATIC_EVAL_VARIANT", None)
        else:
            os.environ["TRIBES_STATIC_EVAL_VARIANT"] = previous_variant


def position_to_dict(position: PositionAnalysis) -> dict[str, Any]:
    return asdict(position)
