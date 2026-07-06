from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from profiling.config import load_config_defaults
from profiling.analysis.actions import top_mass_keys
from profiling.analysis.distributions import jaccard, js_bits, normalize, top_overlap
from profiling.analysis.payload_store import load_payload, store_payload
from profiling.analysis.position_analyzer import analyze_position, parse_target
from profiling.analysis.report_html import write_report

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "debug-logs" / "analysis" / "branch-compare"
DEFAULT_CONFIG = PROJECT_ROOT / "py" / "profiling" / "configs" / "compare_branches.json"


def _repo_relative(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _payload_paths(payload_dir: Path, explicit: list[Path]) -> list[Path]:
    payload_dir = _repo_relative(payload_dir)
    paths = [_repo_relative(path) for path in explicit] + sorted(payload_dir.glob("*.json"))
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def _scores(position: dict[str, Any], field: str) -> dict[str, float]:
    return {str(action["action_id"]): float(action.get(field, 0.0) or 0.0) for action in position["actions"]}


def _position_metrics(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    action_ids = sorted(set(_scores(a, "visit_share")) | set(_scores(b, "visit_share")))
    a_visit = _scores(a, "visit_share")
    b_visit = _scores(b, "visit_share")
    a_prior = _scores(a, "prior")
    b_prior = _scores(b, "prior")
    av = [a_visit.get(action_id, 0.0) for action_id in action_ids]
    bv = [b_visit.get(action_id, 0.0) for action_id in action_ids]
    ap = [a_prior.get(action_id, 0.0) for action_id in action_ids]
    bp = [b_prior.get(action_id, 0.0) for action_id in action_ids]
    a_top95 = top_mass_keys(a_visit)
    b_top95 = top_mass_keys(b_visit)
    b_ranks = {str(action["action_id"]): int(action.get("visit_rank", 0) or 0) for action in b["actions"]}
    return {
        "payload_hash": a["payload_hash"],
        "label": a["label"],
        "selected_same": a["selected_action_fingerprint"] == b["selected_action_fingerprint"],
        "visit_top1_same": max(a_visit, key=a_visit.get, default="") == max(b_visit, key=b_visit.get, default=""),
        "prior_top1_same": max(a_prior, key=a_prior.get, default="") == max(b_prior, key=b_prior.get, default=""),
        "selected_rank_in_other": b_ranks.get(str(a["selected_action_id"]), ""),
        "top95_overlap_jaccard": jaccard(a_top95, b_top95),
        "top5_overlap": top_overlap(av, bv, 5),
        "top10_overlap": top_overlap(av, bv, 10),
        "top32_overlap": top_overlap(av, bv, 32),
        "js_visit_bits": js_bits(normalize(av), normalize(bv)),
        "js_prior_bits": js_bits(normalize(ap), normalize(bp)),
    }


def run(args: argparse.Namespace) -> Path:
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    output_dir = _repo_relative(Path(args.output_dir)) / run_id
    payload_store = PROJECT_ROOT / "debug-logs" / "analysis" / "payloads"
    targets = [parse_target(spec) for spec in args.target]
    if len(targets) < 2:
        raise RuntimeError("pass at least two --target values")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "targets.json").write_text(json.dumps(targets, indent=2, sort_keys=True), encoding="utf-8")
    positions: list[dict[str, Any]] = []
    payload_paths = _payload_paths(Path(args.payload_dir), [Path(p) for p in args.payload])
    if not payload_paths:
        raise RuntimeError(
            f"no payload JSON files found under {_repo_relative(Path(args.payload_dir))}; "
            "run from the repo root or pass an absolute --payload-dir"
        )
    with (output_dir / "positions.jsonl").open("w", encoding="utf-8") as jsonl:
        loaded_positions = 0
        skipped_payloads = 0
        for path in payload_paths:
            payload = load_payload(path)
            if payload is None:
                skipped_payloads += 1
                continue
            if args.positions is not None and loaded_positions >= int(args.positions):
                break
            loaded_positions += 1
            digest = store_payload(payload, payload_store)
            for target in targets:
                position = analyze_position(
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
                    native_static_exe=getattr(args, "native_static_exe", PROJECT_ROOT / "out" / "native" / "static_mcts_bot.exe"),
                    build_native_static_exe=bool(getattr(args, "build_native_static_exe", True)),
                    native_static_search_mode=str(getattr(args, "native_static_search_mode", "primitive")),
                )
                row = asdict(position)
                positions.append(row)
                jsonl.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    if not positions:
        raise RuntimeError(
            f"no valid payload positions loaded from {_repo_relative(Path(args.payload_dir))}; "
            f"checked={len(payload_paths)} skipped={skipped_payloads}"
        )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for position in positions:
        grouped.setdefault(str(position["payload_hash"]), []).append(position)
    summary_rows = []
    for rows in grouped.values():
        by_name = {str(row["target_name"]): row for row in rows}
        if len(rows) >= 2:
            summary_rows.append(_position_metrics(rows[0], rows[1]))
    with (output_dir / "positions_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()) if summary_rows else ["payload_hash"])
        writer.writeheader()
        writer.writerows(summary_rows)
    action_headers = ["payload_hash", "label", "action_id", "action_fingerprint", "action_type"]
    for target in targets:
        name = str(target["name"])
        action_headers.extend([f"{name}_prior", f"{name}_prior_rank", f"{name}_visit_share", f"{name}_visit_rank", f"{name}_q_mean", f"{name}_selected"])
    with (output_dir / "actions_wide.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=action_headers)
        writer.writeheader()
        for rows in grouped.values():
            union: dict[str, dict[str, Any]] = {}
            for position in rows:
                for action in position["actions"]:
                    if action.get("in_top95") or action["action_id"] == position["selected_action_id"]:
                        union.setdefault(
                            str(action["action_fingerprint"]),
                            {
                                "payload_hash": position["payload_hash"],
                                "label": position["label"],
                                "action_id": action["action_id"],
                                "action_fingerprint": action["action_fingerprint"],
                                "action_type": action["action_type"],
                            },
                        )
                        name = str(position["target_name"])
                        union[str(action["action_fingerprint"])].update(
                            {
                                f"{name}_prior": action["prior"],
                                f"{name}_prior_rank": action["prior_rank"],
                                f"{name}_visit_share": action["visit_share"],
                                f"{name}_visit_rank": action["visit_rank"],
                                f"{name}_q_mean": action["q_mean"],
                                f"{name}_selected": action["action_id"] == position["selected_action_id"],
                            }
                        )
            writer.writerows(union.values())
    summary = {
        "targets": [target["name"] for target in targets],
        "positions": len(grouped),
        "comparisons": len(summary_rows),
        "js_visit_bits_avg": sum(float(row["js_visit_bits"]) for row in summary_rows) / len(summary_rows) if summary_rows else 0.0,
        "selected_same_rate": sum(1 for row in summary_rows if row["selected_same"]) / len(summary_rows) if summary_rows else 0.0,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_report(output_dir / "report.html", "Branch Distribution Comparison", summary, summary_rows[:200])
    print(f"wrote {output_dir}")
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare root action distributions across static-eval targets.")
    parser.add_argument("--payload", action="append", default=[])
    parser.add_argument("--payload-dir", type=Path, default=PROJECT_ROOT / "debug-logs" / "mcts-profile-payloads")
    parser.add_argument("--target", action="append")
    parser.add_argument("--positions", type=int, default=None)
    parser.add_argument("--simulations", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k-actions", type=int, default=0)
    parser.add_argument("--max-actions", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--native-static-exe", type=Path, default=PROJECT_ROOT / "out" / "native" / "static_mcts_bot.exe")
    parser.add_argument("--build-native-static-exe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--native-static-search-mode", choices=("primitive", "turn-macro-exp"), default="primitive")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = load_config_defaults(parser, default_config=DEFAULT_CONFIG)
    if args.payload is None:
        args.payload = []
    if not args.target:
        raise RuntimeError("compare_branches requires target list in config or --target")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
