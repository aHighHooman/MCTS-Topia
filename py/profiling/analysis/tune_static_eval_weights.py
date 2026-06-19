from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

from profiling.config import load_config_defaults
from profiling.analysis.report_html import write_report

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "debug-logs" / "analysis" / "tuning-proposals"
DEFAULT_CONFIG = PROJECT_ROOT / "py" / "profiling" / "configs" / "tune_static_eval_weights.json"


def _repo_relative(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _truthy(value: Any, default: bool = True) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "n", "off"}


def _matrix(rows: list[dict[str, str]]) -> tuple[dict[tuple[str, str], dict[str, float]], dict[str, dict[str, Any]]]:
    features: dict[tuple[str, str], dict[str, float]] = {}
    specs: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not _truthy(row.get("contributes"), True) or not _truthy(row.get("tunable"), True):
            continue
        key = (str(row.get("payload_hash", "")), str(row.get("action_id", "")))
        term = str(row.get("name", ""))
        if not key[0] or not key[1] or not term:
            continue
        features.setdefault(key, {})[term] = _float(row.get("feature_value"))
        specs.setdefault(
            term,
            {
                "term": term,
                "parent": str(row.get("parent", "")),
                "initial": _float(row.get("weight")),
            },
        )
    return features, specs


def _constraint_delta(
    features: dict[tuple[str, str], dict[str, float]],
    payload_hash: str,
    desired: str,
    rival: str,
) -> dict[str, float] | None:
    desired_features = features.get((payload_hash, desired))
    rival_features = features.get((payload_hash, rival))
    if desired_features is None or rival_features is None:
        return None
    return {
        term: desired_features.get(term, 0.0) - rival_features.get(term, 0.0)
        for term in set(desired_features) | set(rival_features)
    }


def _dot(weights: dict[str, float], delta: dict[str, float]) -> float:
    return sum(weights.get(term, 0.0) * value for term, value in delta.items())


def _bounds_for(
    initial: float,
    *,
    max_abs_delta: float,
    max_relative_delta: float,
    preserve_sign: bool,
    min_weight: float | None,
    max_weight: float | None,
) -> tuple[float, float]:
    rel_cap = abs(initial) * max(0.0, max_relative_delta)
    cap = max_abs_delta if abs(initial) <= 1e-12 else min(max_abs_delta, rel_cap)
    lower = initial - cap
    upper = initial + cap
    if preserve_sign:
        if initial > 0.0:
            lower = max(lower, 0.0)
        elif initial < 0.0:
            upper = min(upper, 0.0)
        else:
            lower = max(lower, 0.0)
    if min_weight is not None:
        lower = max(lower, min_weight)
    if max_weight is not None:
        upper = min(upper, max_weight)
    if lower > upper:
        lower = upper = initial
    return lower, upper


def _project(weights: dict[str, float], bounds: dict[str, tuple[float, float]]) -> dict[str, float]:
    out = dict(weights)
    for term, (lower, upper) in bounds.items():
        out[term] = min(upper, max(lower, out.get(term, 0.0)))
    return out


def _parse_constraints(
    features: dict[tuple[str, str], dict[str, float]],
    raw_constraints: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    constraints: list[dict[str, Any]] = []
    for row in raw_constraints:
        payload_hash = str(row.get("payload_hash", ""))
        desired = str(row.get("desired_action_id") or row.get("baseline_action_id") or "")
        rival = str(row.get("rival_action_id") or row.get("experimental_action_id") or "")
        delta = _constraint_delta(features, payload_hash, desired, rival)
        if delta is None:
            continue
        kind = str(row.get("kind", row.get("classification", "repair")))
        constraints.append(
            {
                "payload_hash": payload_hash,
                "desired_action_id": desired,
                "rival_action_id": rival,
                "kind": kind,
                "margin": _float(row.get("margin"), 1.0),
                "weight": _float(row.get("constraint_weight"), 2.0 if kind.lower() == "protect" else 1.0),
                "delta": delta,
            }
        )
    return constraints


def optimize(
    terms_csv: Path,
    constraints_jsonl: Path,
    *,
    steps: int,
    learning_rate: float,
    l2: float,
    l1: float = 0.0,
    max_abs_delta: float = 0.75,
    max_relative_delta: float = 0.5,
    preserve_sign: bool = True,
    min_weight: float | None = None,
    max_weight: float | None = None,
    min_satisfied_rate: float = 1.0,
    require_all_repairs: bool = True,
    protect_satisfied_before_weight: float = 4.0,
) -> dict[str, Any]:
    features, specs = _matrix(_read_csv(terms_csv))
    initial = {term: float(spec["initial"]) for term, spec in specs.items()}
    constraints = _parse_constraints(features, _read_jsonl(constraints_jsonl))
    if not constraints:
        raise RuntimeError("no constraints matched the terms matrix")
    for constraint in constraints:
        before = _dot(initial, constraint["delta"])
        constraint["satisfied_before"] = before >= float(constraint["margin"])
        if constraint["satisfied_before"] and str(constraint["kind"]).lower() != "protect":
            constraint["kind"] = "protect"
            constraint["weight"] = float(constraint["weight"]) * protect_satisfied_before_weight

    bounds = {
        term: _bounds_for(
            initial_weight,
            max_abs_delta=max_abs_delta,
            max_relative_delta=max_relative_delta,
            preserve_sign=preserve_sign,
            min_weight=min_weight,
            max_weight=max_weight,
        )
        for term, initial_weight in initial.items()
    }
    weights = dict(initial)
    constraint_norm = max(1, len(constraints))
    for _ in range(max(1, int(steps))):
        grad = {
            term: 2.0 * l2 * (weights.get(term, 0.0) - initial.get(term, 0.0))
            for term in weights
        }
        if l1 > 0.0:
            for term in weights:
                delta_weight = weights.get(term, 0.0) - initial.get(term, 0.0)
                if delta_weight > 0.0:
                    grad[term] = grad.get(term, 0.0) + l1
                elif delta_weight < 0.0:
                    grad[term] = grad.get(term, 0.0) - l1
        for constraint in constraints:
            score = _dot(weights, constraint["delta"])
            miss = float(constraint["margin"]) - score
            if miss <= 0.0:
                continue
            scale = -2.0 * float(constraint["weight"]) * miss / constraint_norm
            for term, value in constraint["delta"].items():
                if term not in weights:
                    continue
                grad[term] = grad.get(term, 0.0) + scale * value
        for term, value in grad.items():
            weights[term] = weights.get(term, 0.0) - learning_rate * value
            if not math.isfinite(weights[term]):
                weights[term] = initial.get(term, 0.0)
        weights = _project(weights, bounds)

    evaluations = []
    for constraint in constraints:
        before = _dot(initial, constraint["delta"])
        after = _dot(weights, constraint["delta"])
        evaluations.append(
            {
                "payload_hash": constraint["payload_hash"],
                "desired_action_id": constraint["desired_action_id"],
                "rival_action_id": constraint["rival_action_id"],
                "kind": constraint["kind"],
                "margin": constraint["margin"],
                "before": before,
                "after": after,
                "satisfied_before": before >= constraint["margin"],
                "satisfied_after": after >= constraint["margin"],
            }
        )
    changes = [
        {
            "term": term,
            "parent": specs.get(term, {}).get("parent", ""),
            "before": initial.get(term, 0.0),
            "after": weights.get(term, 0.0),
            "delta": weights.get(term, 0.0) - initial.get(term, 0.0),
            "lower_bound": bounds[term][0],
            "upper_bound": bounds[term][1],
        }
        for term in sorted(weights)
        if abs(weights.get(term, 0.0) - initial.get(term, 0.0)) > 1e-9
    ]
    changes.sort(key=lambda row: abs(float(row["delta"])), reverse=True)
    satisfied = sum(1 for row in evaluations if row["satisfied_after"])
    satisfied_rate = satisfied / len(evaluations) if evaluations else 0.0
    repair_rows = [row for row in evaluations if str(row["kind"]).lower() != "protect"]
    all_repairs_satisfied = all(row["satisfied_after"] for row in repair_rows)
    sign_flips = [
        row for row in changes
        if float(row["before"]) != 0.0 and float(row["after"]) != 0.0 and math.copysign(1.0, float(row["before"])) != math.copysign(1.0, float(row["after"]))
    ]
    accepted = satisfied_rate >= min_satisfied_rate and (all_repairs_satisfied or not require_all_repairs) and not sign_flips
    rejection_reasons: list[str] = []
    if satisfied_rate < min_satisfied_rate:
        rejection_reasons.append(f"satisfied_rate {satisfied_rate:.3f} < {min_satisfied_rate:.3f}")
    if require_all_repairs and not all_repairs_satisfied:
        rejection_reasons.append("not all repair constraints satisfied")
    if sign_flips:
        rejection_reasons.append("sign flip detected")
    changed_override_env = ",".join(f"{row['term']}={float(row['after']):.12g}" for row in changes)
    return {
        "accepted": accepted,
        "rejection_reasons": rejection_reasons,
        "weights": weights,
        "initial_weights": initial,
        "changes": changes,
        "constraints": evaluations,
        "satisfied_after": satisfied,
        "satisfied_rate": satisfied_rate,
        "override_env": ",".join(f"{term}={weights[term]:.12g}" for term in sorted(weights)),
        "changed_override_env": changed_override_env,
        "bounds": {term: {"lower": lower, "upper": upper} for term, (lower, upper) in bounds.items()},
        "optimizer": {
            "steps": steps,
            "learning_rate": learning_rate,
            "l2": l2,
            "l1": l1,
            "max_abs_delta": max_abs_delta,
            "max_relative_delta": max_relative_delta,
            "preserve_sign": preserve_sign,
            "min_weight": min_weight,
            "max_weight": max_weight,
            "min_satisfied_rate": min_satisfied_rate,
            "require_all_repairs": require_all_repairs,
            "protect_satisfied_before_weight": protect_satisfied_before_weight,
        },
    }


def run(args: argparse.Namespace) -> Path:
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    output_dir = _repo_relative(Path(args.output_dir)) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    result = optimize(
        _repo_relative(Path(args.terms_csv)),
        _repo_relative(Path(args.constraints_jsonl)),
        steps=args.steps,
        learning_rate=args.learning_rate,
        l2=args.l2,
        l1=args.l1,
        max_abs_delta=args.max_abs_delta,
        max_relative_delta=args.max_relative_delta,
        preserve_sign=args.preserve_sign,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
        min_satisfied_rate=args.min_satisfied_rate,
        require_all_repairs=args.require_all_repairs,
        protect_satisfied_before_weight=args.protect_satisfied_before_weight,
    )
    (output_dir / "candidate_weights.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
    with (output_dir / "weight_changes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["term", "parent", "before", "after", "delta", "lower_bound", "upper_bound"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(result["changes"])
    with (output_dir / "constraints.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "payload_hash", "desired_action_id", "rival_action_id", "kind", "margin",
            "before", "after", "satisfied_before", "satisfied_after",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["constraints"])
    summary = {
        "terms_csv": str(args.terms_csv),
        "constraints_jsonl": str(args.constraints_jsonl),
        "constraints": len(result["constraints"]),
        "changed_terms": len(result["changes"]),
        "accepted": result["accepted"],
        "rejection_reasons": result["rejection_reasons"],
        "satisfied_after": result["satisfied_after"],
        "satisfied_rate": result["satisfied_rate"],
        "override_env": result["override_env"],
        "changed_override_env": result["changed_override_env"],
        "optimizer": result["optimizer"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    write_report(output_dir / "report.html", "Static Eval Weight Proposal", summary, result["changes"])
    print(f"wrote {output_dir}")
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Solve a bounded least-change static eval weight proposal from ranking constraints.")
    parser.add_argument("--terms-csv", type=Path)
    parser.add_argument("--constraints-jsonl", type=Path)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--l2", type=float, default=0.5)
    parser.add_argument("--l1", type=float, default=0.0)
    parser.add_argument("--max-abs-delta", type=float, default=0.75)
    parser.add_argument("--max-relative-delta", type=float, default=0.5)
    parser.add_argument("--preserve-sign", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-weight", type=float, default=None)
    parser.add_argument("--max-weight", type=float, default=None)
    parser.add_argument("--min-satisfied-rate", type=float, default=1.0)
    parser.add_argument("--require-all-repairs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--protect-satisfied-before-weight", type=float, default=4.0)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = load_config_defaults(parser, default_config=DEFAULT_CONFIG)
    if args.terms_csv is None or args.constraints_jsonl is None:
        raise RuntimeError("tune_static_eval_weights requires --terms-csv and --constraints-jsonl")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
