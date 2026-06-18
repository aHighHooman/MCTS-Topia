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


def _matrix(rows: list[dict[str, str]]) -> tuple[dict[tuple[str, str], dict[str, float]], dict[str, float]]:
    features: dict[tuple[str, str], dict[str, float]] = {}
    weights: dict[str, list[float]] = {}
    for row in rows:
        key = (str(row.get("payload_hash", "")), str(row.get("action_id", "")))
        term = str(row.get("name", ""))
        if not key[0] or not key[1] or not term:
            continue
        features.setdefault(key, {})[term] = _float(row.get("feature_value"))
        weights.setdefault(term, []).append(_float(row.get("weight")))
    default_weights = {term: values[0] for term, values in weights.items() if values}
    return features, default_weights


def _constraint_delta(
    features: dict[tuple[str, str], dict[str, float]],
    payload_hash: str,
    desired: str,
    rival: str,
) -> dict[str, float] | None:
    d = features.get((payload_hash, desired))
    r = features.get((payload_hash, rival))
    if d is None or r is None:
        return None
    return {term: d.get(term, 0.0) - r.get(term, 0.0) for term in set(d) | set(r)}


def _dot(weights: dict[str, float], delta: dict[str, float]) -> float:
    return sum(weights.get(term, 0.0) * value for term, value in delta.items())


def optimize(
    terms_csv: Path,
    constraints_jsonl: Path,
    *,
    steps: int,
    learning_rate: float,
    l2: float,
) -> dict[str, Any]:
    features, initial = _matrix(_read_csv(terms_csv))
    raw_constraints = _read_jsonl(constraints_jsonl)
    constraints = []
    for row in raw_constraints:
        payload_hash = str(row.get("payload_hash", ""))
        desired = str(row.get("desired_action_id") or row.get("baseline_action_id") or "")
        rival = str(row.get("rival_action_id") or row.get("experimental_action_id") or "")
        delta = _constraint_delta(features, payload_hash, desired, rival)
        if delta is None:
            continue
        constraints.append(
            {
                "payload_hash": payload_hash,
                "desired_action_id": desired,
                "rival_action_id": rival,
                "kind": str(row.get("kind", row.get("classification", "repair"))),
                "margin": _float(row.get("margin"), 1.0),
                "weight": _float(row.get("constraint_weight"), 2.0 if str(row.get("kind", "")).lower() == "protect" else 1.0),
                "delta": delta,
            }
        )
    weights = dict(initial)
    if not constraints:
        raise RuntimeError("no constraints matched the terms matrix")
    for _ in range(max(1, int(steps))):
        grad = {term: 2.0 * l2 * (weights.get(term, 0.0) - initial.get(term, 0.0)) for term in weights}
        for constraint in constraints:
            score = _dot(weights, constraint["delta"])
            miss = float(constraint["margin"]) - score
            if miss <= 0.0:
                continue
            scale = -2.0 * float(constraint["weight"]) * miss
            for term, value in constraint["delta"].items():
                grad[term] = grad.get(term, 0.0) + scale * value
                weights.setdefault(term, initial.get(term, 0.0))
        for term, value in grad.items():
            weights[term] = weights.get(term, 0.0) - learning_rate * value
            if not math.isfinite(weights[term]):
                weights[term] = initial.get(term, 0.0)
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
            "before": initial.get(term, 0.0),
            "after": weights.get(term, 0.0),
            "delta": weights.get(term, 0.0) - initial.get(term, 0.0),
        }
        for term in sorted(weights)
        if abs(weights.get(term, 0.0) - initial.get(term, 0.0)) > 1e-9
    ]
    changes.sort(key=lambda row: abs(float(row["delta"])), reverse=True)
    return {
        "weights": weights,
        "initial_weights": initial,
        "changes": changes,
        "constraints": evaluations,
        "override_env": ",".join(f"{term}={weights[term]:.12g}" for term in sorted(weights)),
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
    )
    (output_dir / "candidate_weights.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
    with (output_dir / "weight_changes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["term", "before", "after", "delta"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["changes"])
    with (output_dir / "constraints.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["payload_hash", "desired_action_id", "rival_action_id", "kind", "margin", "before", "after", "satisfied_before", "satisfied_after"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["constraints"])
    summary = {
        "terms_csv": str(args.terms_csv),
        "constraints_jsonl": str(args.constraints_jsonl),
        "constraints": len(result["constraints"]),
        "changed_terms": len(result["changes"]),
        "satisfied_after": sum(1 for row in result["constraints"] if row["satisfied_after"]),
        "override_env": result["override_env"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    write_report(output_dir / "report.html", "Static Eval Weight Proposal", summary, result["changes"])
    print(f"wrote {output_dir}")
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Solve a least-change static eval weight proposal from ranking constraints.")
    parser.add_argument("--terms-csv", type=Path)
    parser.add_argument("--constraints-jsonl", type=Path)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--l2", type=float, default=0.05)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = load_config_defaults(parser, default_config=DEFAULT_CONFIG)
    if args.terms_csv is None or args.constraints_jsonl is None:
        raise RuntimeError("tune_static_eval_weights requires --terms-csv and --constraints-jsonl")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
