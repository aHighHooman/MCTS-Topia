from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from profiling.config import load_config_defaults
from profiling.analysis.actions import action_fingerprint, action_id, action_type
from profiling.analysis.distributions import rank_map
from profiling.analysis.payload_store import load_payload, store_payload
from profiling.analysis.position_analyzer import _action_breakdown, analyze_position, parse_target
from profiling.analysis.report_html import write_report

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "debug-logs" / "analysis" / "root-child-value-matrix"
DEFAULT_CONFIG = PROJECT_ROOT / "py" / "profiling" / "configs" / "root_child_value_matrix.json"


def _repo_relative(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _terms_by_name(breakdown: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(breakdown, dict):
        return {}
    return {str(row.get("name")): dict(row) for row in breakdown.get("terms", []) if isinstance(row, dict)}


def _breakdown_value(breakdown: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if not isinstance(breakdown, dict):
        return default
    try:
        return float(breakdown.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _action_lookup(position: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(action.get("action_id")): dict(action) for action in position.get("actions", [])}


def _selected_pairs(rows: list[dict[str, Any]], *, limit: int) -> list[tuple[dict[str, Any], dict[str, Any], str]]:
    if not rows:
        return []
    by_id = {str(row["action_id"]): row for row in rows}
    candidates: list[dict[str, Any]] = []
    for key in ("visit_rank", "child_value_rank", "prior_rank"):
        candidates.extend(sorted(rows, key=lambda row: int(row.get(key, 999999) or 999999))[:limit])
    selected = next((row for row in rows if row.get("selected")), None)
    top_visit = min(rows, key=lambda row: int(row.get("visit_rank", 999999) or 999999))
    rivals = [row for row in (selected, top_visit) if row is not None]
    out: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    seen: set[tuple[str, str]] = set()
    for desired in candidates:
        for rival in rivals:
            if desired["action_id"] == rival["action_id"]:
                continue
            key = (str(desired["action_id"]), str(rival["action_id"]))
            if key in seen:
                continue
            seen.add(key)
            reason = "desired_vs_selected" if rival is selected else "desired_vs_top_visit"
            out.append((desired, rival, reason))
    return out


def analyze_payload(
    payload: dict[str, Any],
    *,
    payload_hash: str,
    label: str,
    target: dict[str, str],
    simulations: int,
    batch_size: int,
    top_k_actions: int,
    max_actions: int,
    seed: int,
    c_puct: float,
    native_static_exe: Path | str | None,
    build_native_static_exe: bool,
    native_static_search_mode: str,
    pair_limit: int = 5,
) -> dict[str, Any]:
    position = analyze_position(
        payload,
        payload_hash=payload_hash,
        label=label,
        target=target,
        simulations=simulations,
        batch_size=batch_size,
        top_k_actions=top_k_actions,
        max_actions=max_actions,
        seed=seed,
        c_puct=c_puct,
        include_breakdown=False,
        native_static_exe=native_static_exe,
        build_native_static_exe=build_native_static_exe,
        native_static_search_mode=native_static_search_mode,
    )
    position_row = asdict(position)
    action_stats = _action_lookup(position_row)
    raw_actions = list(payload.get("actions", [])) if max_actions < 0 else list(payload.get("actions", []))[:max_actions]
    previous_variant = os.environ.get("TRIBES_STATIC_EVAL_VARIANT")
    os.environ["TRIBES_STATIC_EVAL_VARIANT"] = str(target.get("variant", "baseline"))
    try:
        action_rows: list[dict[str, Any]] = []
        term_rows: list[dict[str, Any]] = []
        for index, action in enumerate(raw_actions):
            aid = action_id(action)
            breakdown = _action_breakdown(payload, aid, max_actions)
            value_breakdown = breakdown if isinstance(breakdown, dict) else {}
            child_value = _breakdown_value(value_breakdown, "final_value")
            child_raw = _breakdown_value(value_breakdown, "raw_total")
            stat = action_stats.get(aid, {})
            row = {
                "payload_hash": payload_hash,
                "label": label,
                "target_name": target.get("name", ""),
                "static_eval_variant": target.get("variant", ""),
                "action_id": aid,
                "action_fingerprint": action_fingerprint(action),
                "action_index": index,
                "action_type": action_type(action),
                "selected": aid == position.selected_action_id,
                "prior": float(stat.get("prior", 0.0) or 0.0),
                "prior_rank": int(stat.get("prior_rank", 0) or 0),
                "visit_share": float(stat.get("visit_share", 0.0) or 0.0),
                "visit_rank": int(stat.get("visit_rank", 0) or 0),
                "q_mean": stat.get("q_mean"),
                "child_value": child_value,
                "child_raw": child_raw,
                "child_value_rank": 0,
                "child_raw_rank": 0,
            }
            action_rows.append(row)
            for term in value_breakdown.get("terms", []):
                if isinstance(term, dict):
                    term_rows.append({**row, **term})
        value_ranks = rank_map([float(row["child_value"]) for row in action_rows])
        raw_ranks = rank_map([float(row["child_raw"]) for row in action_rows])
        for index, row in enumerate(action_rows):
            row["child_value_rank"] = value_ranks.get(index, 0)
            row["child_raw_rank"] = raw_ranks.get(index, 0)
        pairwise_rows: list[dict[str, Any]] = []
        terms_by_action: dict[str, dict[str, dict[str, Any]]] = {}
        for row in action_rows:
            terms_by_action[str(row["action_id"])] = {}
        for term in term_rows:
            terms_by_action[str(term["action_id"])][str(term["name"])] = term
        for desired, rival, reason in _selected_pairs(action_rows, limit=pair_limit):
            desired_terms = terms_by_action.get(str(desired["action_id"]), {})
            rival_terms = terms_by_action.get(str(rival["action_id"]), {})
            for name in sorted(set(desired_terms) | set(rival_terms)):
                d_raw = float(desired_terms.get(name, {}).get("raw", 0.0) or 0.0)
                r_raw = float(rival_terms.get(name, {}).get("raw", 0.0) or 0.0)
                pairwise_rows.append(
                    {
                        "payload_hash": payload_hash,
                        "label": label,
                        "target_name": target.get("name", ""),
                        "reason": reason,
                        "desired_action_id": desired["action_id"],
                        "desired_action_fingerprint": desired["action_fingerprint"],
                        "rival_action_id": rival["action_id"],
                        "rival_action_fingerprint": rival["action_fingerprint"],
                        "term": name,
                        "desired_raw": d_raw,
                        "rival_raw": r_raw,
                        "delta_raw_desired_minus_rival": d_raw - r_raw,
                    }
                )
        return {
            "position": position_row,
            "actions": action_rows,
            "terms": term_rows,
            "pairwise": pairwise_rows,
        }
    finally:
        if previous_variant is None:
            os.environ.pop("TRIBES_STATIC_EVAL_VARIANT", None)
        else:
            os.environ["TRIBES_STATIC_EVAL_VARIANT"] = previous_variant


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> Path:
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    output_dir = _repo_relative(Path(args.output_dir)) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_store = PROJECT_ROOT / "debug-logs" / "analysis" / "payloads"
    target = parse_target(args.target)
    payload_paths = [Path(path) for path in getattr(args, "payload", []) or []]
    payload_dir = _repo_relative(Path(args.payload_dir))
    payload_paths = [_repo_relative(path) for path in payload_paths] + sorted(payload_dir.glob("*.json"))
    positions: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    terms: list[dict[str, Any]] = []
    pairwise: list[dict[str, Any]] = []
    loaded = 0
    for path in payload_paths:
        if args.positions is not None and loaded >= int(args.positions):
            break
        payload = load_payload(path)
        if payload is None:
            continue
        loaded += 1
        digest = store_payload(payload, payload_store)
        result = analyze_payload(
            payload,
            payload_hash=digest,
            label=path.stem,
            target=target,
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
        positions.append(result["position"])
        actions.extend(result["actions"])
        terms.extend(result["terms"])
        pairwise.extend(result["pairwise"])
    if not positions:
        raise RuntimeError(f"no valid payload positions loaded from {payload_dir}")
    (output_dir / "positions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in positions),
        encoding="utf-8",
    )
    action_fields = [
        "payload_hash", "label", "target_name", "static_eval_variant", "action_id",
        "action_fingerprint", "action_index", "action_type", "selected", "prior",
        "prior_rank", "visit_share", "visit_rank", "q_mean", "child_value",
        "child_raw", "child_value_rank", "child_raw_rank",
    ]
    term_fields = action_fields + ["name", "feature_value", "weight", "raw", "normalized", "abs_share", "linearized_value"]
    pair_fields = [
        "payload_hash", "label", "target_name", "reason", "desired_action_id",
        "desired_action_fingerprint", "rival_action_id", "rival_action_fingerprint",
        "term", "desired_raw", "rival_raw", "delta_raw_desired_minus_rival",
    ]
    _write_csv(output_dir / "actions.csv", actions, action_fields)
    _write_csv(output_dir / "terms_matrix.csv", terms, term_fields)
    _write_csv(output_dir / "pairwise_deltas.csv", pairwise, pair_fields)
    summary = {
        "target": target,
        "positions": len(positions),
        "actions": len(actions),
        "terms": len(terms),
        "pairwise_deltas": len(pairwise),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    write_report(output_dir / "report.html", "Root Child Value Matrix", summary, actions[:1000])
    print(f"wrote {output_dir}")
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate every root child and write an action x value-term matrix.")
    parser.add_argument("--payload", action="append", default=[])
    parser.add_argument("--payload-dir", type=Path, default=PROJECT_ROOT / "debug-logs" / "mcts-profile-payloads")
    parser.add_argument("--target")
    parser.add_argument("--positions", type=int, default=None)
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
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = load_config_defaults(parser, default_config=DEFAULT_CONFIG)
    if not args.target:
        raise RuntimeError("root_child_value_matrix requires target in config or --target")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
