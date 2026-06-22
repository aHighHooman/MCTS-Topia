from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from profiling.config import load_config_defaults
from profiling.analysis.distributions import jaccard, js_bits, normalize
from profiling.analysis.payload_store import load_payload
from profiling.analysis.position_analyzer import analyze_position, parse_target
from profiling.analysis.report_html import write_report

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "py" / "profiling" / "configs" / "overlay_match.json"


def _payload_by_hash(digest: str) -> dict | None:
    path = PROJECT_ROOT / "debug-logs" / "analysis" / "payloads" / digest[:2] / digest[2:4] / f"{digest}.json"
    return load_payload(path) if path.exists() else None


def _scores(position: dict, field: str) -> dict[str, float]:
    return {str(action["action_id"]): float(action.get(field, 0.0) or 0.0) for action in position["actions"]}


def run(args: argparse.Namespace) -> Path:
    targets = [parse_target(spec) for spec in args.target]
    if len(targets) < 2:
        raise RuntimeError("pass at least two --target values")
    output_dir = Path(args.output_dir or (Path(args.match_dir) / f"overlay-{time.strftime('%Y%m%d-%H%M%S')}"))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (Path(args.match_dir) / "timeline.csv").open(newline="", encoding="utf-8") as handle:
        timeline = list(csv.DictReader(handle))
    cache: dict[tuple[str, str], dict] = {}
    rows = []
    for item in timeline:
        digest = str(item["payload_hash"])
        payload = _payload_by_hash(digest)
        if payload is None:
            continue
        analyses = []
        for target in targets:
            key = (digest, str(target["name"]))
            if key not in cache:
                cache[key] = analyze_position(
                    payload,
                    payload_hash=digest,
                    label=digest[:12],
                    target=target,
                    simulations=args.simulations,
                    batch_size=args.batch_size,
                    top_k_actions=args.top_k_actions,
                    max_actions=args.max_actions,
                    seed=args.seed,
                    native_static_exe=getattr(args, "native_static_exe", PROJECT_ROOT / "out" / "native" / "static_mcts_bot.exe"),
                    build_native_static_exe=bool(getattr(args, "build_native_static_exe", True)),
                    native_static_search_mode=str(getattr(args, "native_static_search_mode", "primitive")),
                ).__dict__
                cache[key]["actions"] = [action.__dict__ for action in cache[key]["actions"]]
            analyses.append(cache[key])
        action_ids = sorted(set(_scores(analyses[0], "visit_share")) | set(_scores(analyses[1], "visit_share")))
        a_scores = _scores(analyses[0], "visit_share")
        b_scores = _scores(analyses[1], "visit_share")
        chosen = str(item["chosen_action_id"])
        row = {
            **item,
            f"{targets[0]['name']}_selected": analyses[0]["selected_action_id"],
            f"{targets[1]['name']}_selected": analyses[1]["selected_action_id"],
            f"chosen_visit_{targets[0]['name']}": a_scores.get(chosen, 0.0),
            f"chosen_visit_{targets[1]['name']}": b_scores.get(chosen, 0.0),
            f"{targets[0]['name']}_root_value": analyses[0]["root_value"],
            f"{targets[1]['name']}_root_value": analyses[1]["root_value"],
            "js_visit_bits": js_bits(normalize([a_scores.get(aid, 0.0) for aid in action_ids]), normalize([b_scores.get(aid, 0.0) for aid in action_ids])),
            "top95_overlap_jaccard": jaccard(
                {a["action_id"] for a in analyses[0]["actions"] if a.get("in_top95")},
                {a["action_id"] for a in analyses[1]["actions"] if a.get("in_top95")},
            ),
            "actor_choice_bad_by_other_flag": chosen not in {a["action_id"] for a in analyses[1]["actions"] if a.get("in_top95")},
            "causal_limitation": "preference divergence at same observed payload",
        }
        rows.append(row)
    with (output_dir / "overlay.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["match_id"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {"rows": len(rows), "targets": [target["name"] for target in targets], "causal_limitation": "This is not a counterfactual continuation."}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_report(output_dir / "report.html", "Match Overlay", summary, sorted(rows, key=lambda row: float(row.get("js_visit_bits", 0.0)), reverse=True)[:200])
    print(f"wrote {output_dir}")
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Overlay target analyses onto a recorded match timeline.")
    parser.add_argument("--match-dir", type=Path)
    parser.add_argument("--target", action="append")
    parser.add_argument("--simulations", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k-actions", type=int, default=0)
    parser.add_argument("--max-actions", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--native-static-exe", type=Path, default=PROJECT_ROOT / "out" / "native" / "static_mcts_bot.exe")
    parser.add_argument("--build-native-static-exe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--native-static-search-mode", choices=("primitive", "turn-cmab"), default="primitive")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = load_config_defaults(parser, default_config=DEFAULT_CONFIG)
    if args.match_dir is None:
        raise RuntimeError("overlay_match requires match_dir in config or --match-dir")
    if not args.target:
        raise RuntimeError("overlay_match requires target list in config or --target")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
