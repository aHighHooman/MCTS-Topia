from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from profiling.config import load_config_defaults
from profiling.analysis.payload_store import load_payload, store_payload
from profiling.analysis.position_analyzer import analyze_position, parse_target

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "debug-logs" / "analysis" / "matches"
DEFAULT_CONFIG = PROJECT_ROOT / "py" / "profiling" / "configs" / "match_recorder.json"


def run(args: argparse.Namespace) -> Path:
    match_id = args.match_id or f"analysis-{args.seed}-{time.strftime('%Y%m%d-%H%M%S')}"
    match_dir = Path(args.output_dir) / match_id
    payload_store = PROJECT_ROOT / "debug-logs" / "analysis" / "payloads"
    match_dir.mkdir(parents=True, exist_ok=True)
    branches = [parse_target(args.branch0), parse_target(args.branch1)]
    seat_assignments = [(0, 1), (1, 0)] if args.seats == "both" else [(0, 1)]
    manifest = {
        "match_id": match_id,
        "seed": args.seed,
        "branches": branches,
        "seat_assignments": seat_assignments,
        "causal_limitation": "Recorded rows support same-payload preference divergence overlays, not proof of alternate match outcomes.",
    }
    (match_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    rows = []
    payload_paths = sorted(Path(args.payload_dir).glob("*.json")) if args.payload_dir else []
    for seat_index, seats in enumerate(seat_assignments):
        for ply_index, path in enumerate(payload_paths):
            payload = load_payload(path)
            if payload is None:
                continue
            actor_index = int(payload.get("player_id", 0)) % 2
            actor_branch = branches[actor_index]
            digest = store_payload(payload, payload_store)
            analysis = analyze_position(
                payload,
                payload_hash=digest,
                label=path.stem,
                target=actor_branch,
                simulations=args.simulations,
                batch_size=args.batch_size,
                top_k_actions=args.top_k_actions,
                max_actions=args.max_actions,
                seed=args.seed + ply_index,
                native_static_exe=getattr(args, "native_static_exe", PROJECT_ROOT / "out" / "native" / "native_static_mcts_bot.exe"),
                build_native_static_exe=bool(getattr(args, "build_native_static_exe", True)),
                native_static_search_mode=str(getattr(args, "native_static_search_mode", "primitive")),
            )
            rows.append(
                {
                    "match_id": match_id,
                    "seat_assignment": f"{seats[0]}-{seats[1]}",
                    "ply_index": ply_index,
                    "tick": "",
                    "turn_step_index": "",
                    "active_player_id": payload.get("player_id", ""),
                    "actor_branch": actor_branch["name"],
                    "payload_hash": digest,
                    "chosen_action_id": analysis.selected_action_id,
                    "root_value": analysis.root_value,
                }
            )
    with (match_dir / "timeline.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["match_id", "seat_assignment", "ply_index", "tick", "turn_step_index", "active_player_id", "actor_branch", "payload_hash", "chosen_action_id", "root_value"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {match_dir}")
    return match_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Record analysis match timelines or seed them from payloads.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--branch0")
    parser.add_argument("--branch1")
    parser.add_argument("--seats", choices=("both", "forward"), default="both")
    parser.add_argument("--payload-dir", type=Path, default=None, help="Optional payload source for offline recorder seeding.")
    parser.add_argument("--simulations", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k-actions", type=int, default=0)
    parser.add_argument("--max-actions", type=int, default=512)
    parser.add_argument("--native-static-exe", type=Path, default=PROJECT_ROOT / "out" / "native" / "native_static_mcts_bot.exe")
    parser.add_argument("--build-native-static-exe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--native-static-search-mode", choices=("primitive", "turn-cmab"), default="primitive")
    parser.add_argument("--match-id", default="")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = load_config_defaults(parser, default_config=DEFAULT_CONFIG)
    if args.seed is None:
        raise RuntimeError("match_recorder requires seed in config or --seed")
    if not args.branch0 or not args.branch1:
        raise RuntimeError("match_recorder requires branch0 and branch1 in config or --branch0/--branch1")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
