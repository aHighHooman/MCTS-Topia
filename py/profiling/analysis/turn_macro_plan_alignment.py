"""Compare lower-budget generated macro plans to a fixed high-budget baseline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from profiling.analysis.payload_store import load_payload, payload_hash
from profiling.analysis.position_analyzer import _ensure_native_static_exe
from profiling.analysis.turn_macro_inner_convergence import PROJECT_ROOT, _repo_path, _run
from profiling.config import load_config_defaults

DEFAULT_CONFIG = PROJECT_ROOT / "py" / "profiling" / "configs" / "turn_macro_plan_alignment.json"


def _plans(response: dict[str, Any], plan_count: int, max_depth: int) -> list[list[str]]:
    profile = response.get("_profile") if isinstance(response.get("_profile"), dict) else {}
    raw = profile.get("root_turn_plans") if isinstance(profile.get("root_turn_plans"), list) else []
    plans: list[list[str]] = []
    for item in raw[:plan_count]:
        if not isinstance(item, dict):
            continue
        signatures = item.get("action_signatures")
        if not isinstance(signatures, list):
            continue
        plans.append([str(signature) for signature in signatures[:max_depth]])
    return plans


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _run_plans(exe: Path, payload: dict[str, Any], args: argparse.Namespace, inner_simulations: int) -> list[list[str]]:
    response = _run(
        exe,
        payload,
        outer_simulations=1,
        measurement_mode="turn-macro-exp",
        inner_simulations=inner_simulations,
        max_edges=args.plan_count,
        max_primitives=args.max_primitives,
        inner_c_puct=args.inner_c_puct,
        outer_c=args.outer_c,
        prior_weight=args.prior_weight,
        temperature=args.temperature,
        opponent_mode=args.opponent_mode,
        max_actions=args.max_actions,
        seed=args.seed,
        static_eval_variant=args.static_eval_variant,
        timeout_sec=args.timeout_sec,
    )
    return _plans(response, args.plan_count, args.max_depth)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload-dir", type=Path, default=PROJECT_ROOT / "debug-logs" / "mcts-profile-payloads")
    parser.add_argument("--position-offset", type=int, default=0)
    parser.add_argument("--max-positions", type=int, default=0)
    parser.add_argument("--baseline-inner-simulations", type=int, default=7500)
    parser.add_argument("--inner-budgets", type=int, nargs="+", default=[32, 64, 128, 256, 512, 1024, 2048])
    parser.add_argument("--plan-count", type=int, default=4)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--max-primitives", type=int, default=0)
    parser.add_argument("--inner-c-puct", type=float, default=1.5)
    parser.add_argument("--outer-c", type=float, default=1.4)
    parser.add_argument("--prior-weight", type=float, default=0.35)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--opponent-mode", choices=("root-max", "maximalist"), default="maximalist")
    parser.add_argument("--max-actions", type=int, default=512)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--static-eval-variant", default="baseline")
    parser.add_argument("--native-static-exe", type=Path, default=PROJECT_ROOT / "out" / "native" / "static_mcts_bot.exe")
    parser.add_argument("--build-native-static-exe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout-sec", type=float, default=180.0)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "debug-logs" / "analysis" / "turn-macro-plan-alignment")
    args = load_config_defaults(parser, default_config=DEFAULT_CONFIG)
    if args.baseline_inner_simulations <= 0 or args.plan_count <= 0 or args.max_depth <= 0:
        raise ValueError("baseline_inner_simulations, plan_count, and max_depth must be positive")
    budgets = sorted({int(value) for value in args.inner_budgets if int(value) > 0 and int(value) != args.baseline_inner_simulations})
    if not budgets:
        raise ValueError("inner_budgets must include at least one positive budget other than the baseline")

    payload_dir = _repo_path(args.payload_dir)
    output_dir = _repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(path for path in payload_dir.glob("*.json") if path.name != "mcts_profile_position_selection_summary.json")
    paths = paths[max(0, args.position_offset) :]
    if args.max_positions > 0:
        paths = paths[: args.max_positions]
    payloads = [(path, load_payload(path)) for path in paths]
    payloads = [(path, payload) for path, payload in payloads if payload is not None]
    if not payloads:
        raise RuntimeError(f"No valid payloads found in {payload_dir}")
    exe = _ensure_native_static_exe(args.native_static_exe, bool(args.build_native_static_exe))
    print(f"[turn_macro_plan_alignment] positions={len(payloads)} baseline={args.baseline_inner_simulations} plans={args.plan_count}", flush=True)

    rows: list[dict[str, Any]] = []
    aggregate: dict[tuple[int, int], dict[str, int]] = {
        (budget, depth): {"baseline_slots": 0, "rank_matches": 0, "coverage_matches": 0, "positions": 0, "full_rank_positions": 0, "full_coverage_positions": 0}
        for budget in budgets for depth in range(1, args.max_depth + 1)
    }
    for position, (path, payload) in enumerate(payloads, start=1):
        assert payload is not None
        baseline = _run_plans(exe, payload, args, args.baseline_inner_simulations)
        for budget in budgets:
            candidate = _run_plans(exe, payload, args, budget)
            row: dict[str, Any] = {
                "position": position,
                "payload": path.name,
                "payload_hash": payload_hash(payload),
                "inner_budget": budget,
                "baseline_inner_simulations": args.baseline_inner_simulations,
                "baseline_plans": json.dumps(baseline),
                "candidate_plans": json.dumps(candidate),
                "baseline_plan_count": len(baseline),
                "candidate_plan_count": len(candidate),
            }
            for depth in range(1, args.max_depth + 1):
                index = depth - 1
                baseline_slots = [(rank, plan[index]) for rank, plan in enumerate(baseline) if len(plan) > index]
                candidate_actions = {plan[index] for plan in candidate if len(plan) > index}
                rank_matches = sum(
                    int(rank < len(candidate) and len(candidate[rank]) > index and candidate[rank][index] == action)
                    for rank, action in baseline_slots
                )
                coverage_matches = sum(int(action in candidate_actions) for _, action in baseline_slots)
                row[f"step_{depth}_baseline_slots"] = len(baseline_slots)
                row[f"step_{depth}_rank_matches"] = rank_matches
                row[f"step_{depth}_coverage_matches"] = coverage_matches
                row[f"step_{depth}_rank_rate"] = rank_matches / len(baseline_slots) if baseline_slots else 1.0
                row[f"step_{depth}_coverage_rate"] = coverage_matches / len(baseline_slots) if baseline_slots else 1.0
                stats = aggregate[(budget, depth)]
                stats["baseline_slots"] += len(baseline_slots)
                stats["rank_matches"] += rank_matches
                stats["coverage_matches"] += coverage_matches
                stats["positions"] += 1
                stats["full_rank_positions"] += int(rank_matches == len(baseline_slots))
                stats["full_coverage_positions"] += int(coverage_matches == len(baseline_slots))
            rows.append(row)
        print(f"  {position}/{len(payloads)} {path.name}", flush=True)

    summary: list[dict[str, Any]] = []
    for budget in budgets:
        for depth in range(1, args.max_depth + 1):
            stats = aggregate[(budget, depth)]
            slots = stats["baseline_slots"]
            positions = stats["positions"]
            summary.append({
                "inner_budget": budget,
                "step": depth,
                "baseline_action_slots": slots,
                "rank_aligned_action_rate": stats["rank_matches"] / slots if slots else 1.0,
                "top_four_coverage_action_rate": stats["coverage_matches"] / slots if slots else 1.0,
                "positions": positions,
                "positions_full_rank_rate": stats["full_rank_positions"] / positions if positions else 1.0,
                "positions_full_coverage_rate": stats["full_coverage_positions"] / positions if positions else 1.0,
            })
    result = {
        "method": {
            "baseline_inner_simulations": args.baseline_inner_simulations,
            "generated_plans": args.plan_count,
            "max_plan_depth": args.max_depth,
            "rank_aligned": "candidate plan rank r action at step d equals baseline plan rank r action at step d",
            "top_four_coverage": "baseline plan action at step d appears in any candidate top-four plan at that same step",
            "positions": len(payloads),
        },
        "summary": summary,
        "artifacts": {"alignment": "alignment.csv", "summary": "summary.csv"},
    }
    _write_csv(output_dir / "alignment.csv", rows)
    _write_csv(output_dir / "summary.csv", summary)
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
