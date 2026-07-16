"""Measure convergence of complete turn-macro plan sets as inner MCTS grows.

For every position and candidate inner budget N, generate the actual root macro
plans selected by the current visit/value policy. Compare those plans with 2N,
5N, and a fixed high-budget reference. Exact ordered action sequences are the
primary identity. An unordered continuation identity is also reported as a
looser diagnostic. A shared first action alone is never a match.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

from profiling.analysis.payload_store import load_payload, payload_hash
from profiling.analysis.position_analyzer import _ensure_native_static_exe
from profiling.config import load_config_defaults

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "py" / "profiling" / "configs" / "turn_macro_inner_convergence.json"
Plan = tuple[str, ...]


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


def _plans(response: dict[str, Any], plan_count: int, comparison_depth: int) -> list[Plan]:
    profile = response.get("_profile") if isinstance(response.get("_profile"), dict) else {}
    raw_plans = profile.get("root_turn_plans") if isinstance(profile.get("root_turn_plans"), list) else []
    plans: list[Plan] = []
    for item in raw_plans[:plan_count]:
        if not isinstance(item, dict) or not isinstance(item.get("action_signatures"), list):
            continue
        signatures = [str(value) for value in item["action_signatures"]]
        if comparison_depth > 0:
            signatures = signatures[:comparison_depth]
        if signatures:
            plans.append(tuple(signatures))
    return plans


def _material_plans(response: dict[str, Any], plan_count: int) -> list[Plan]:
    """Return a loose first-action-plus-unordered-continuation diagnostic."""
    exact_plans = _plans(response, plan_count, comparison_depth=0)
    return [(plan[0], *sorted(plan[1:])) for plan in exact_plans]


def _is_ordered_subsequence(shorter: Plan, longer: Plan) -> bool:
    if len(shorter) > len(longer):
        return False
    cursor = 0
    for action in longer:
        if cursor < len(shorter) and action == shorter[cursor]:
            cursor += 1
    return cursor == len(shorter)


def _plans_containment_aligned(left: Plan, right: Plan, identity: str) -> bool:
    shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
    if identity == "material":
        return bool(shorter) and bool(longer) and shorter[0] == longer[0] and set(shorter[1:]).issubset(longer[1:])
    if identity == "exact_sequence":
        return bool(shorter) and bool(longer) and shorter[0] == longer[0] and _is_ordered_subsequence(shorter, longer)
    raise ValueError(f"unknown plan identity: {identity}")


def _maximum_containment_matching(candidate: list[Plan], reference: list[Plan], identity: str) -> int:
    """Maximum one-to-one matching, so one generic short plan cannot cover many plans."""
    adjacency = [
        [index for index, other in enumerate(reference) if _plans_containment_aligned(plan, other, identity)]
        for plan in candidate
    ]
    matched_candidate_by_reference = [-1] * len(reference)

    def augment(candidate_index: int, seen: set[int]) -> bool:
        for reference_index in adjacency[candidate_index]:
            if reference_index in seen:
                continue
            seen.add(reference_index)
            previous = matched_candidate_by_reference[reference_index]
            if previous < 0 or augment(previous, seen):
                matched_candidate_by_reference[reference_index] = candidate_index
                return True
        return False

    return sum(augment(index, set()) for index in range(len(candidate)))


def _compare(candidate: list[Plan], reference: list[Plan], identity: str = "exact_sequence") -> dict[str, float | int]:
    candidate_set = set(candidate)
    reference_set = set(reference)
    union = candidate_set | reference_set
    intersection = candidate_set & reference_set
    slots = len(reference)
    rank_matches = sum(
        int(rank < len(candidate) and candidate[rank] == plan)
        for rank, plan in enumerate(reference)
    )
    ranked_containment_matches = sum(
        int(rank < len(candidate) and _plans_containment_aligned(candidate[rank], plan, identity))
        for rank, plan in enumerate(reference)
    )
    containment_matches = _maximum_containment_matching(candidate, reference, identity)
    largest_count = max(len(candidate), len(reference))
    return {
        "reference_plans": slots,
        "candidate_plans": len(candidate),
        "ranked_plan_match_rate": rank_matches / slots if slots else float(not candidate),
        "set_recall": len(intersection) / len(reference_set) if reference_set else float(not candidate_set),
        "set_precision": len(intersection) / len(candidate_set) if candidate_set else float(not reference_set),
        "set_jaccard": len(intersection) / len(union) if union else 1.0,
        "containment_alignment_rate": containment_matches / largest_count if largest_count else 1.0,
        "containment_candidate_rate": containment_matches / len(candidate) if candidate else float(not reference),
        "containment_reference_rate": containment_matches / len(reference) if reference else float(not candidate),
        "ranked_containment_rate": ranked_containment_matches / slots if slots else float(not candidate),
        "exact_ranked_plans": int(candidate == reference),
        "exact_plan_set": int(candidate_set == reference_set),
    }


def _run(
    exe: Path,
    payload: dict[str, Any],
    *,
    inner_simulations: int,
    plan_count: int,
    max_primitives: int,
    inner_c_puct: float,
    outer_c: float,
    opponent_mode: str,
    max_actions: int,
    seed: int,
    static_eval_variant: str,
    timeout_sec: float,
) -> dict[str, Any]:
    command = [
        str(exe), "--search-mode", "turn-macro-exp",
        "--simulations", "1",
        "--max-actions", str(max_actions),
        "--seed", str(seed),
        "--static-eval-variant", static_eval_variant,
        "--turn-macro-max-edges-per-node", str(plan_count),
        "--turn-macro-max-primitives-per-turn", str(max_primitives),
        "--turn-macro-inner-simulations", str(inner_simulations),
        "--turn-macro-inner-c-puct", str(inner_c_puct),
        "--turn-macro-outer-c", str(outer_c),
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
    profile = response.get("_profile") if isinstance(response, dict) else None
    if not isinstance(profile, dict) or not isinstance(profile.get("root_turn_plans"), list):
        raise RuntimeError(f"turn-macro run returned no root plans (inner={inner_simulations}): {response}")
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
    parser.add_argument("--position-offset", type=int, default=0)
    parser.add_argument("--max-positions", type=int, default=0)
    parser.add_argument("--inner-budgets", type=int, nargs="+", default=[32, 64, 128, 256, 512, 1024, 2048])
    parser.add_argument("--comparison-multipliers", type=int, nargs="+", default=[2, 5])
    parser.add_argument("--reference-inner-simulations", type=int, default=7500)
    parser.add_argument("--plan-count", type=int, default=8)
    parser.add_argument("--comparison-depth", type=int, default=0, help="Actions compared per plan; 0 compares the complete plan.")
    parser.add_argument("--max-primitives", type=int, default=0)
    parser.add_argument("--inner-c-puct", type=float, default=1.5)
    parser.add_argument("--outer-c", type=float, default=1.4)
    parser.add_argument("--opponent-mode", choices=("root-max", "maximalist"), default="maximalist")
    parser.add_argument("--max-actions", type=int, default=512)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--static-eval-variant", default="baseline")
    parser.add_argument("--native-static-exe", type=Path, default=PROJECT_ROOT / "out" / "native" / "static_mcts_bot.exe")
    parser.add_argument("--build-native-static-exe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout-sec", type=float, default=180.0)
    parser.add_argument("--target-set-jaccard", type=float, default=0.90)
    parser.add_argument("--target-containment-alignment", type=float, default=0.90)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "debug-logs" / "analysis" / "turn-macro-inner-convergence")
    args = load_config_defaults(parser, default_config=DEFAULT_CONFIG)

    budgets = sorted({int(value) for value in args.inner_budgets if int(value) > 0})
    multipliers = sorted({int(value) for value in args.comparison_multipliers if int(value) > 1})
    if not budgets or args.reference_inner_simulations <= 0 or args.plan_count <= 0 or args.comparison_depth < 0:
        raise ValueError("budgets, reference_inner_simulations, and plan_count must be positive; comparison_depth cannot be negative")
    run_budgets = sorted(
        {args.reference_inner_simulations}
        | set(budgets)
        | {budget * multiplier for budget in budgets for multiplier in multipliers}
    )
    payload_dir = _repo_path(args.payload_dir)
    output_dir = _repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(path for path in payload_dir.glob("*.json") if path.name != "mcts_profile_position_selection_summary.json")
    paths = paths[max(0, args.position_offset):]
    if args.max_positions > 0:
        paths = paths[:args.max_positions]
    payloads = [(path, load_payload(path)) for path in paths]
    payloads = [(path, payload) for path, payload in payloads if payload is not None]
    if not payloads:
        raise RuntimeError(f"No valid payloads found in {payload_dir}")

    exe = _ensure_native_static_exe(args.native_static_exe, bool(args.build_native_static_exe))
    print(
        f"[turn_macro_inner_convergence] positions={len(payloads)} plans={args.plan_count} "
        f"depth={'full' if args.comparison_depth == 0 else args.comparison_depth} inner_runs={run_budgets}",
        flush=True,
    )
    run_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for position, (path, payload) in enumerate(payloads, start=1):
        assert payload is not None
        exact_plans_by_budget: dict[int, list[Plan]] = {}
        material_plans_by_budget: dict[int, list[Plan]] = {}
        for run_budget in run_budgets:
            response = _run(
                exe,
                payload,
                inner_simulations=run_budget,
                plan_count=args.plan_count,
                max_primitives=args.max_primitives,
                inner_c_puct=args.inner_c_puct,
                outer_c=args.outer_c,
                opponent_mode=args.opponent_mode,
                max_actions=args.max_actions,
                seed=args.seed,
                static_eval_variant=args.static_eval_variant,
                timeout_sec=args.timeout_sec,
            )
            exact_plans = _plans(response, args.plan_count, args.comparison_depth)
            material_plans = _material_plans(response, args.plan_count)
            exact_plans_by_budget[run_budget] = exact_plans
            material_plans_by_budget[run_budget] = material_plans
            profile = response["_profile"]
            run_rows.append({
                "position": position,
                "payload": path.name,
                "payload_hash": payload_hash(payload),
                "inner_budget": run_budget,
                "plan_count": len(exact_plans),
                "exact_sequence_plans": json.dumps(exact_plans),
                "material_plans": json.dumps(material_plans),
                "inner_nodes_expanded": int(profile.get("inner_nodes_expanded", 0) or 0),
                "elapsed_sec": float(profile.get("elapsed_sec", 0.0) or 0.0),
            })
        for budget in budgets:
            row: dict[str, Any] = {
                "position": position,
                "payload": path.name,
                "payload_hash": payload_hash(payload),
                "inner_budget": budget,
                "exact_sequence_plans": json.dumps(exact_plans_by_budget[budget]),
                "material_plans": json.dumps(material_plans_by_budget[budget]),
            }
            references = {"reference": args.reference_inner_simulations}
            references.update({f"{multiplier}x": budget * multiplier for multiplier in multipliers})
            for label, reference_budget in references.items():
                for identity, plans_by_budget in (
                    ("material", material_plans_by_budget),
                    ("exact_sequence", exact_plans_by_budget),
                ):
                    for key, value in _compare(
                        plans_by_budget[budget], plans_by_budget[reference_budget], identity
                    ).items():
                        row[f"{identity}_{label}_{key}"] = value
            comparison_rows.append(row)
        print(f"  {position}/{len(payloads)} {path.name}", flush=True)

    summary_rows: list[dict[str, Any]] = []
    labels = ["reference", *(f"{multiplier}x" for multiplier in multipliers)]
    for budget in budgets:
        rows = [row for row in comparison_rows if row["inner_budget"] == budget]
        summary: dict[str, Any] = {"inner_budget": budget, "positions": len(rows)}
        for identity in ("material", "exact_sequence"):
            for label in labels:
                for metric in (
                    "ranked_plan_match_rate",
                    "set_recall",
                    "set_precision",
                    "set_jaccard",
                    "containment_alignment_rate",
                    "containment_candidate_rate",
                    "containment_reference_rate",
                    "ranked_containment_rate",
                ):
                    key = f"{identity}_{label}_{metric}"
                    summary[f"mean_{key}"] = sum(float(row[key]) for row in rows) / len(rows)
                exact_sets = sum(int(row[f"{identity}_{label}_exact_plan_set"]) for row in rows)
                exact_ranked = sum(int(row[f"{identity}_{label}_exact_ranked_plans"]) for row in rows)
                summary[f"{identity}_{label}_exact_plan_set_rate"] = exact_sets / len(rows)
                summary[f"{identity}_{label}_exact_plan_set_wilson_lower_95"] = _wilson_lower(exact_sets, len(rows))
                summary[f"{identity}_{label}_exact_ranked_plans_rate"] = exact_ranked / len(rows)
        summary_rows.append(summary)

    strict_recommended = next((
        row for row in summary_rows
        if float(row["mean_material_reference_set_jaccard"]) >= args.target_set_jaccard
        and all(float(row[f"mean_material_{multiplier}x_set_jaccard"]) >= args.target_set_jaccard for multiplier in multipliers)
    ), None)
    containment_recommendations: dict[str, int | None] = {}
    for identity in ("exact_sequence", "material"):
        recommended = next((
            row for row in summary_rows
            if float(row[f"mean_{identity}_reference_containment_alignment_rate"]) >= args.target_containment_alignment
            and all(
                float(row[f"mean_{identity}_{multiplier}x_containment_alignment_rate"])
                >= args.target_containment_alignment
                for multiplier in multipliers
            )
        ), None)
        containment_recommendations[identity] = recommended["inner_budget"] if recommended else None
    result = {
        "method": {
            "target": "complete root macro plans selected by the current visit/value policy",
            "material_plan_identity": "diagnostic exact first action plus the unordered set of remaining action signatures",
            "strict_diagnostic": "live exact-dedup identity: complete ordered action-signature sequence",
            "length_invariant_alignment": "maximum one-to-one matching; ordered subsequence for exact sequences, or same first action plus subset of remaining commitments for material plans",
            "comparison_depth": "complete plan" if args.comparison_depth == 0 else args.comparison_depth,
            "plan_count": args.plan_count,
            "max_primitives_per_plan": args.max_primitives,
            "inner_budgets": budgets,
            "comparison_multipliers": multipliers,
            "reference_inner_simulations": args.reference_inner_simulations,
            "target_set_jaccard": args.target_set_jaccard,
            "target_containment_alignment": args.target_containment_alignment,
            "strict_criterion": "mean plan-set Jaccard reaches the target against every multiplier and the fixed high-budget reference",
            "containment_criterion": "mean one-to-one containment alignment reaches the target against every multiplier and the fixed high-budget reference",
            "positions": len(payloads),
            "run_budgets": run_budgets,
        },
        "recommended_inner_budget_by_point_estimate": strict_recommended["inner_budget"] if strict_recommended else None,
        "recommended_inner_budget_by_ordered_containment": containment_recommendations["exact_sequence"],
        "recommended_inner_budget_by_unordered_containment": containment_recommendations["material"],
        "summary": summary_rows,
        "elapsed_sec": time.perf_counter() - started,
        "artifacts": {"runs": "runs.csv", "comparisons": "comparisons.csv", "summary": "summary.csv"},
    }
    _write_csv(output_dir / "runs.csv", run_rows)
    _write_csv(output_dir / "comparisons.csv", comparison_rows)
    _write_csv(output_dir / "summary.csv", summary_rows)
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
