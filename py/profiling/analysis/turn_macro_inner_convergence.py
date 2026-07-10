"""Measure stability of turn-macro decisions as the inner MCTS budget grows.

The metric deliberately fixes the payload, outer turn-MCTS budget, evaluator, and
seed.  For each requested inner budget N, it asks whether the final deterministic
macro decision is unchanged at 2N and 5N.  This measures the practical effect of
the inner planner on the action returned to the protocol, rather than only the
inner tree's private ranking.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from profiling.analysis.payload_store import load_payload, payload_hash
from profiling.analysis.position_analyzer import _ensure_native_static_exe
from profiling.config import load_config_defaults

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "py" / "profiling" / "configs" / "turn_macro_inner_convergence.json"


def _repo_path(value: Path | str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _wilson_lower(successes: int, total: int, z: float = 1.959963984540054) -> float:
    if total <= 0:
        return 0.0
    proportion = successes / total
    denom = 1.0 + z * z / total
    centre = proportion + z * z / (2.0 * total)
    spread = z * math.sqrt((proportion * (1.0 - proportion) + z * z / (4.0 * total)) / total)
    return max(0.0, (centre - spread) / denom)


def _action_id(response: dict[str, Any]) -> str:
    return str(response.get("actionId") or response.get("action_id") or "")


def _response_features(response: dict[str, Any]) -> dict[str, float | int]:
    profile = response.get("_profile") if isinstance(response.get("_profile"), dict) else {}
    visits = profile.get("root_first_action_visits") if isinstance(profile.get("root_first_action_visits"), dict) else {}
    visit_counts = sorted((int(value) for value in visits.values()), reverse=True)
    total = sum(visit_counts)
    top_share = visit_counts[0] / total if total else 0.0
    second_share = visit_counts[1] / total if len(visit_counts) > 1 and total else 0.0
    candidates = profile.get("inner_candidates") if isinstance(profile.get("inner_candidates"), list) else []
    candidate_visits = [max(0, int(item.get("visits", 0) or 0)) for item in candidates if isinstance(item, dict)]
    candidate_scores = [float(item.get("score", 0.0) or 0.0) for item in candidates if isinstance(item, dict)]
    inner_total = sum(candidate_visits)
    inner_top_share = candidate_visits[0] / inner_total if inner_total else 0.0
    inner_second_share = candidate_visits[1] / inner_total if len(candidate_visits) > 1 and inner_total else 0.0
    return {
        "elapsed_sec": float(profile.get("elapsed_sec", 0.0) or 0.0),
        "inner_searches": int(profile.get("inner_searches", 0) or 0),
        "inner_simulations": int(profile.get("inner_simulations", 0) or 0),
        "inner_nodes_expanded": int(profile.get("inner_nodes_expanded", 0) or 0),
        "root_turn_edges": int(profile.get("root_turn_edges", 0) or 0),
        "outer_simulations": int(profile.get("outer_simulations", 0) or 0),
        "root_top_visit_share": top_share,
        "root_visit_margin": top_share - second_share,
        "inner_candidates": len(candidates),
        "inner_top_visit_share": inner_top_share,
        "inner_visit_margin": inner_top_share - inner_second_share,
        "inner_best_score_gap": candidate_scores[0] - candidate_scores[1] if len(candidate_scores) > 1 else float("inf"),
    }


def _payload_features(payload: dict[str, Any]) -> dict[str, int]:
    actions = list(payload.get("actions", []) or [])
    counts = Counter(str(action.get("type") or action.get("t") or "UNKNOWN") for action in actions if isinstance(action, dict))
    observation = payload.get("observation") if isinstance(payload.get("observation"), dict) else {}
    units = list(observation.get("units", []) or [])
    cities = list(observation.get("cities", []) or [])
    player_id = int(payload.get("player_id", 0) or 0)
    enemy_units = sum(1 for unit in units if isinstance(unit, dict) and int(unit.get("tribe_id", unit.get("p", player_id)) or player_id) != player_id)
    return {
        "tick": int(observation.get("tick", observation.get("turn", 0)) or 0),
        "actions": len(actions),
        "action_types": len(counts),
        "tactical_actions": sum(counts[kind] for kind in ("ATTACK", "MOVE", "STEP_MOVE", "CONVERT", "INFILTRATE", "RECOVER")),
        "research_actions": counts["RESEARCH_TECH"],
        "cities": len(cities),
        "units": len(units),
        "enemy_units": enemy_units,
    }


def _condition_bucket(row: dict[str, Any], key: str) -> str:
    value = float(row.get(key, 0) or 0)
    if key == "actions":
        return "<=24" if value <= 24 else "25-48" if value <= 48 else ">48"
    if key in {"root_top_visit_share", "inner_top_visit_share"}:
        return "<0.60" if value < 0.60 else "0.60-0.85" if value < 0.85 else ">=0.85"
    if key in {"root_visit_margin", "inner_visit_margin"}:
        return "<0.20" if value < 0.20 else "0.20-0.60" if value < 0.60 else ">=0.60"
    if key == "root_turn_edges":
        return "<=1" if value <= 1 else "2" if value == 2 else ">=3"
    if key == "inner_candidates":
        return "<=4" if value <= 4 else "5-12" if value <= 12 else ">12"
    return str(value)


def _run(
    exe: Path,
    payload: dict[str, Any],
    *,
    outer_simulations: int,
    measurement_mode: str,
    inner_simulations: int,
    max_edges: int,
    max_primitives: int,
    inner_c_puct: float,
    outer_c: float,
    macro_c: float,
    prior_weight: float,
    temperature: float,
    greedy_eval_top_k: int,
    opponent_mode: str,
    max_actions: int,
    seed: int,
    static_eval_variant: str,
    timeout_sec: float,
) -> dict[str, Any]:
    command = [
        str(exe), "--search-mode", measurement_mode,
        "--simulations", str(outer_simulations),
        "--max-actions", str(max_actions),
        "--seed", str(seed),
        "--static-eval-variant", static_eval_variant,
        "--turn-macro-max-edges-per-node", str(max_edges),
        "--turn-macro-max-primitives-per-turn", str(max_primitives),
        "--turn-macro-inner-simulations", str(inner_simulations),
        "--turn-macro-inner-c-puct", str(inner_c_puct),
        "--turn-macro-outer-c", str(outer_c),
        "--turn-macro-c", str(macro_c),
        "--turn-macro-prior-weight", str(prior_weight),
        "--turn-macro-temperature", str(temperature),
        "--turn-macro-greedy-eval-top-k", str(greedy_eval_top_k),
        "--turn-macro-opponent-mode", opponent_mode,
        "--profile-json", "--deterministic",
    ]
    request = dict(payload)
    request["type"] = "action_request"
    completed = subprocess.run(
        command,
        input=json.dumps(request, separators=(",", ":")) + "\n",
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"turn-macro run failed (inner={inner_simulations}): {completed.stderr[-2500:]}")
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"turn-macro run produced no response (inner={inner_simulations}): {completed.stderr[-2500:]}")
    response = json.loads(lines[-1])
    if not isinstance(response, dict) or not _action_id(response):
        raise RuntimeError(f"turn-macro run returned no action (inner={inner_simulations}): {response}")
    return response


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload-dir", type=Path, default=PROJECT_ROOT / "debug-logs" / "mcts-profile-payloads")
    parser.add_argument("--position-offset", type=int, default=0, help="Skip this many sorted payloads; useful for resumable corpus chunks.")
    parser.add_argument("--max-positions", type=int, default=0)
    parser.add_argument("--inner-budgets", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--top-k-values", type=int, nargs="+", default=[1, 2, 4, 8], help="Candidate-plan counts evaluated against the 5x reference action.")
    parser.add_argument("--outer-simulations", type=int, default=128)
    parser.add_argument("--measurement-mode", choices=("turn-macro-inner-probe", "turn-macro-exp"), default="turn-macro-inner-probe")
    parser.add_argument("--max-edges", type=int, default=4)
    parser.add_argument("--max-primitives", type=int, default=0)
    parser.add_argument("--inner-c-puct", type=float, default=1.5)
    parser.add_argument("--outer-c", type=float, default=1.4)
    parser.add_argument("--macro-c", type=float, default=1.0)
    parser.add_argument("--prior-weight", type=float, default=0.35)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--greedy-eval-top-k", type=int, default=1)
    parser.add_argument("--opponent-mode", choices=("root-max", "maximalist"), default="maximalist")
    parser.add_argument("--max-actions", type=int, default=512)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--static-eval-variant", default="baseline")
    parser.add_argument("--native-static-exe", type=Path, default=PROJECT_ROOT / "out" / "native" / "static_mcts_bot.exe")
    parser.add_argument("--build-native-static-exe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--target-match-rate", type=float, default=0.90)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "debug-logs" / "analysis" / "turn-macro-inner-convergence")
    args = load_config_defaults(parser, default_config=DEFAULT_CONFIG)

    payload_dir = _repo_path(args.payload_dir)
    output_dir = _repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_paths = sorted(path for path in payload_dir.glob("*.json") if path.name != "mcts_profile_position_selection_summary.json")
    if args.position_offset > 0:
        payload_paths = payload_paths[args.position_offset :]
    if args.max_positions > 0:
        payload_paths = payload_paths[: args.max_positions]
    payloads = [(path, load_payload(path)) for path in payload_paths]
    payloads = [(path, payload) for path, payload in payloads if payload is not None]
    if not payloads:
        raise RuntimeError(f"No valid payloads found in {payload_dir}")
    budgets = sorted({int(value) for value in args.inner_budgets if int(value) > 0})
    if not budgets:
        raise RuntimeError("inner_budgets must contain at least one positive budget")
    top_k_values = sorted({int(value) for value in args.top_k_values if int(value) > 0})
    if not top_k_values:
        raise RuntimeError("top_k_values must contain at least one positive value")
    run_budgets = sorted({multiple * budget for budget in budgets for multiple in (1, 2, 5)})
    exe = _ensure_native_static_exe(args.native_static_exe, bool(args.build_native_static_exe))
    print(f"[turn_macro_inner_convergence] positions={len(payloads)} outer={args.outer_simulations} inner_runs={run_budgets}", flush=True)

    raw_rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    started = time.perf_counter()
    for position_index, (path, payload) in enumerate(payloads, start=1):
        assert payload is not None
        features = _payload_features(payload)
        responses: dict[int, tuple[str, list[str], dict[str, float | int]]] = {}
        for inner_budget in run_budgets:
            response = _run(
                exe, payload, outer_simulations=args.outer_simulations, inner_simulations=inner_budget,
                measurement_mode=args.measurement_mode,
                max_edges=args.max_edges, max_primitives=args.max_primitives, inner_c_puct=args.inner_c_puct,
                outer_c=args.outer_c, macro_c=args.macro_c, prior_weight=args.prior_weight,
                temperature=args.temperature, greedy_eval_top_k=args.greedy_eval_top_k,
                opponent_mode=args.opponent_mode, max_actions=args.max_actions, seed=args.seed,
                static_eval_variant=args.static_eval_variant, timeout_sec=args.timeout_sec,
            )
            action = _action_id(response)
            ranked = [str(value) for value in response.get("rankedActionIds", []) if value is not None]
            if action not in ranked:
                ranked.insert(0, action)
            response_features = _response_features(response)
            responses[inner_budget] = (action, ranked, response_features)
            raw_rows.append({
                "position": position_index, "payload": path.name, "payload_hash": payload_hash(payload),
                "inner_budget": inner_budget, "action_id": action, "ranked_action_ids": json.dumps(ranked), **features, **response_features,
            })
        for budget in budgets:
            action, ranked, run_features = responses[budget]
            action_2x, _, _ = responses[2 * budget]
            action_5x, _, _ = responses[5 * budget]
            comparison = {
                "position": position_index, "payload": path.name, "payload_hash": payload_hash(payload),
                "inner_budget": budget, "action_id": action, "action_2x": action_2x, "action_5x": action_5x,
                "same_2x": int(action == action_2x), "same_5x": int(action == action_5x),
                "stable_2x_5x": int(action_2x == action_5x), "stable_all": int(action == action_2x == action_5x),
                **features, **run_features,
            }
            for top_k in top_k_values:
                comparison[f"top_{top_k}_contains_5x"] = int(action_5x in ranked[:top_k])
            comparisons.append(comparison)
        print(f"  {position_index}/{len(payloads)} {path.name}", flush=True)

    summary_rows: list[dict[str, Any]] = []
    for budget in budgets:
        rows = [row for row in comparisons if row["inner_budget"] == budget]
        total = len(rows)
        successes = sum(int(row["stable_all"]) for row in rows)
        summary = {
            "inner_budget": budget,
            "positions": total,
            "same_2x_rate": sum(int(row["same_2x"]) for row in rows) / total,
            "same_5x_rate": sum(int(row["same_5x"]) for row in rows) / total,
            "stable_2x_5x_rate": sum(int(row["stable_2x_5x"]) for row in rows) / total,
            "stable_all_rate": successes / total,
            "stable_all_wilson_lower_95": _wilson_lower(successes, total),
            "mean_elapsed_sec": sum(float(row["elapsed_sec"]) for row in rows) / total,
            "mean_inner_searches": sum(int(row["inner_searches"]) for row in rows) / total,
            "mean_inner_nodes_per_search": sum(int(row["inner_nodes_expanded"]) / max(1, int(row["inner_searches"])) for row in rows) / total,
        }
        for top_k in top_k_values:
            summary[f"top_{top_k}_contains_5x_rate"] = sum(int(row[f"top_{top_k}_contains_5x"]) for row in rows) / total
        summary_rows.append(summary)

    condition_rows: list[dict[str, Any]] = []
    for budget in budgets:
        budget_rows = [row for row in comparisons if row["inner_budget"] == budget]
        for key in ("actions", "inner_top_visit_share", "inner_visit_margin", "inner_candidates"):
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in budget_rows:
                grouped[_condition_bucket(row, key)].append(row)
            for bucket, rows in sorted(grouped.items()):
                total = len(rows)
                successes = sum(int(row["stable_all"]) for row in rows)
                condition_rows.append({
                    "inner_budget": budget, "condition": key, "bucket": bucket, "positions": total,
                    "stable_all_rate": successes / total,
                    "stable_all_wilson_lower_95": _wilson_lower(successes, total),
                })

    recommended = next((row for row in summary_rows if row["stable_all_rate"] >= args.target_match_rate), None)
    result = {
        "method": {
            "decision": "deterministic selected first action from the inner primitive MCTS" if args.measurement_mode == "turn-macro-inner-probe" else "deterministic final action returned by turn-macro MCTS",
            "measurement_mode": args.measurement_mode,
            "criterion": "N action equals both 2N and 5N actions at fixed payload, outer budget, evaluator, and seed",
            "target_match_rate": args.target_match_rate,
            "outer_simulations": args.outer_simulations,
            "inner_budgets": budgets,
            "top_k_values": top_k_values,
            "top_k_definition": "whether the action ranked first at 5N is present in N's top-K ranked inner candidates",
            "run_budgets": run_budgets,
            "positions": len(payloads),
        },
        "recommended_inner_budget_by_point_estimate": recommended["inner_budget"] if recommended else None,
        "summary": summary_rows,
        "elapsed_sec": time.perf_counter() - started,
        "artifacts": {"runs": "runs.csv", "comparisons": "comparisons.csv", "conditions": "conditions.csv"},
    }
    _write_csv(output_dir / "runs.csv", raw_rows)
    _write_csv(output_dir / "comparisons.csv", comparisons)
    _write_csv(output_dir / "conditions.csv", condition_rows)
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
