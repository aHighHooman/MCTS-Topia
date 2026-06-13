from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

from profiling.config import load_config_defaults
from profiling.analysis.payload_store import load_payload, store_payload
from profiling.analysis.position_analyzer import analyze_position, parse_target
from profiling.analysis.report_html import write_report

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "debug-logs" / "analysis" / "value-breakdown"
DEFAULT_CONFIG = PROJECT_ROOT / "py" / "profiling" / "configs" / "value_breakdown.json"


def _repo_relative(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def run(args: argparse.Namespace) -> Path:
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    output_dir = _repo_relative(Path(args.output_dir)) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    target = parse_target(args.target)
    payload_store = PROJECT_ROOT / "debug-logs" / "analysis" / "payloads"
    positions = []
    term_rows = []
    payload_dir = _repo_relative(Path(args.payload_dir))
    payload_paths = sorted(payload_dir.glob("*.json"))
    if not payload_paths:
        raise RuntimeError(
            f"no payload JSON files found under {payload_dir}; "
            "run from the repo root or pass an absolute --payload-dir"
        )
    skipped_payloads = 0
    for path in payload_paths:
        payload = load_payload(path)
        if payload is None:
            skipped_payloads += 1
            continue
        if args.positions is not None and len(positions) >= int(args.positions):
            break
        digest = store_payload(payload, payload_store)
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
            include_breakdown=True,
            native_static_exe=getattr(args, "native_static_exe", PROJECT_ROOT / "out" / "native" / "native_static_mcts_bot.exe"),
            build_native_static_exe=bool(getattr(args, "build_native_static_exe", True)),
            native_static_search_mode=str(getattr(args, "native_static_search_mode", "primitive")),
        )
        row = asdict(position)
        positions.append(row)
        for term in (row.get("value_breakdown") or {}).get("terms", []):
            term_rows.append({"payload_hash": digest, "label": path.stem, **term})
    if not positions:
        raise RuntimeError(
            f"no valid payload positions loaded from {payload_dir}; "
            f"checked={len(payload_paths)} skipped={skipped_payloads}"
        )
    (output_dir / "positions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in positions),
        encoding="utf-8",
    )
    with (output_dir / "terms.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["payload_hash", "label", "name", "raw", "normalized", "abs_share", "linearized_value"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(term_rows)
    summary = {
        "target": target,
        "positions": len(positions),
        "terms": len(term_rows),
        "note": "Term rows are exact pre-tanh additive raw-space contributions; linearized_value is local tanh sensitivity, not an ablation.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_report(output_dir / "report.html", "Static Value Breakdown", summary, term_rows[:300])
    print(f"wrote {output_dir}")
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Report native static-eval value term breakdowns.")
    parser.add_argument("--payload-dir", type=Path, default=PROJECT_ROOT / "debug-logs" / "mcts-profile-payloads")
    parser.add_argument("--target")
    parser.add_argument("--positions", type=int, default=100)
    parser.add_argument("--simulations", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k-actions", type=int, default=0)
    parser.add_argument("--max-actions", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--native-static-exe", type=Path, default=PROJECT_ROOT / "out" / "native" / "native_static_mcts_bot.exe")
    parser.add_argument("--build-native-static-exe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--native-static-search-mode", choices=("primitive", "turn-cmab"), default="primitive")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = load_config_defaults(parser, default_config=DEFAULT_CONFIG)
    if not args.target:
        raise RuntimeError("value_breakdown requires target in config or --target")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
