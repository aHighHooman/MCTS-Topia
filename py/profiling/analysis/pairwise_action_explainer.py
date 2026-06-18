from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from profiling.config import load_config_defaults
from profiling.analysis.payload_store import load_payload, payload_hash, store_payload
from profiling.analysis.position_analyzer import parse_target
from profiling.analysis.report_html import write_report
from profiling.analysis.root_child_value_matrix import analyze_payload

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "debug-logs" / "analysis" / "pairwise-action-explainer"
DEFAULT_CONFIG = PROJECT_ROOT / "py" / "profiling" / "configs" / "pairwise_action_explainer.json"


def _repo_relative(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _matches(row: dict[str, Any], selector: str) -> bool:
    return selector in {str(row.get("action_id")), str(row.get("action_fingerprint"))}


def _term_map(rows: list[dict[str, Any]], action_id: str) -> dict[str, dict[str, Any]]:
    return {str(row.get("name")): row for row in rows if str(row.get("action_id")) == action_id}


def explain(result: dict[str, Any], desired_selector: str, rival_selector: str, *, danger_limit: int = 5) -> dict[str, Any]:
    actions = list(result["actions"])
    desired = next((row for row in actions if _matches(row, desired_selector)), None)
    rival = next((row for row in actions if _matches(row, rival_selector)), None)
    if desired is None:
        raise RuntimeError(f"desired action not found: {desired_selector}")
    if rival is None:
        raise RuntimeError(f"rival action not found: {rival_selector}")
    desired_terms = _term_map(result["terms"], str(desired["action_id"]))
    rival_terms = _term_map(result["terms"], str(rival["action_id"]))
    delta_rows: list[dict[str, Any]] = []
    for name in sorted(set(desired_terms) | set(rival_terms)):
        d = desired_terms.get(name, {})
        r = rival_terms.get(name, {})
        d_raw = float(d.get("raw", 0.0) or 0.0)
        r_raw = float(r.get("raw", 0.0) or 0.0)
        delta_rows.append(
            {
                "term": name,
                "desired_raw": d_raw,
                "rival_raw": r_raw,
                "delta_raw_desired_minus_rival": d_raw - r_raw,
                "desired_feature_value": d.get("feature_value", ""),
                "rival_feature_value": r.get("feature_value", ""),
                "desired_weight": d.get("weight", ""),
                "rival_weight": r.get("weight", ""),
            }
        )
    raw_margin = float(desired.get("child_raw", 0.0) or 0.0) - float(rival.get("child_raw", 0.0) or 0.0)
    value_margin = float(desired.get("child_value", 0.0) or 0.0) - float(rival.get("child_value", 0.0) or 0.0)
    directional_terms = sorted(delta_rows, key=lambda row: abs(float(row["delta_raw_desired_minus_rival"])), reverse=True)
    danger_actions = sorted(
        [
            row for row in actions
            if row["action_id"] not in {desired["action_id"], rival["action_id"]}
            and float(row.get("child_raw", 0.0) or 0.0) >= float(desired.get("child_raw", 0.0) or 0.0)
        ],
        key=lambda row: float(row.get("child_raw", 0.0) or 0.0),
        reverse=True,
    )[:danger_limit]
    return {
        "desired_action": desired,
        "rival_action": rival,
        "raw_margin_desired_minus_rival": raw_margin,
        "value_margin_desired_minus_rival": value_margin,
        "raw_margin_needed_for_desired_to_overtake": max(0.0, -raw_margin),
        "terms_favoring_desired": [row for row in directional_terms if float(row["delta_raw_desired_minus_rival"]) > 0.0],
        "terms_blocking_desired": [row for row in directional_terms if float(row["delta_raw_desired_minus_rival"]) < 0.0],
        "danger_actions_already_above_desired": danger_actions,
        "all_term_deltas": delta_rows,
    }


def run(args: argparse.Namespace) -> Path:
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    output_dir = _repo_relative(Path(args.output_dir)) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = load_payload(_repo_relative(Path(args.payload)))
    if payload is None:
        raise RuntimeError(f"invalid payload: {args.payload}")
    digest = store_payload(payload, PROJECT_ROOT / "debug-logs" / "analysis" / "payloads")
    result = analyze_payload(
        payload,
        payload_hash=digest,
        label=Path(args.payload).stem,
        target=parse_target(args.target),
        simulations=args.simulations,
        batch_size=args.batch_size,
        top_k_actions=args.top_k_actions,
        max_actions=args.max_actions,
        seed=args.seed,
        c_puct=args.c_puct,
        native_static_exe=args.native_static_exe,
        build_native_static_exe=args.build_native_static_exe,
        native_static_search_mode=args.native_static_search_mode,
        pair_limit=args.pair_limit,
    )
    explanation = explain(result, args.desired_action, args.rival_action, danger_limit=args.danger_limit)
    (output_dir / "explanation.json").write_text(json.dumps(explanation, indent=2, sort_keys=True, default=str), encoding="utf-8")
    with (output_dir / "term_deltas.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "term", "desired_raw", "rival_raw", "delta_raw_desired_minus_rival",
            "desired_feature_value", "rival_feature_value", "desired_weight", "rival_weight",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(explanation["all_term_deltas"])
    summary = {
        "payload_hash": digest,
        "target": args.target,
        "desired_action": args.desired_action,
        "rival_action": args.rival_action,
        "raw_margin_desired_minus_rival": explanation["raw_margin_desired_minus_rival"],
        "raw_margin_needed_for_desired_to_overtake": explanation["raw_margin_needed_for_desired_to_overtake"],
        "danger_actions_already_above_desired": len(explanation["danger_actions_already_above_desired"]),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    write_report(output_dir / "report.html", "Pairwise Action Explainer", summary, explanation["all_term_deltas"])
    print(f"wrote {output_dir}")
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Explain why one root action scores above another.")
    parser.add_argument("--payload", type=Path)
    parser.add_argument("--target")
    parser.add_argument("--desired-action", required=False)
    parser.add_argument("--rival-action", required=False)
    parser.add_argument("--simulations", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k-actions", type=int, default=0)
    parser.add_argument("--max-actions", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--native-static-exe", type=Path, default=PROJECT_ROOT / "out" / "native" / "native_static_mcts_bot.exe")
    parser.add_argument("--build-native-static-exe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--native-static-search-mode", choices=("primitive", "turn-cmab"), default="primitive")
    parser.add_argument("--pair-limit", type=int, default=5)
    parser.add_argument("--danger-limit", type=int, default=5)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = load_config_defaults(parser, default_config=DEFAULT_CONFIG)
    if args.payload is None:
        raise RuntimeError("pairwise_action_explainer requires --payload")
    if not args.target:
        raise RuntimeError("pairwise_action_explainer requires --target")
    if not args.desired_action or not args.rival_action:
        raise RuntimeError("pairwise_action_explainer requires --desired-action and --rival-action")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
