from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from profiling.config import load_config_defaults
from profiling.analysis.payload_store import load_payload, store_payload
from profiling.analysis.position_analyzer import analyze_position, parse_target
from profiling.analysis.report_html import write_report

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "debug-logs" / "analysis" / "value-breakdown"
DEFAULT_CONFIG = PROJECT_ROOT / "py" / "profiling" / "configs" / "value_breakdown.json"
OUTPUT_FILES = ("positions.jsonl", "terms.csv", "summary.json", "report.html")


def _repo_relative(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _path_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _mtime_iso(path: Path) -> str:
    mtime = _path_mtime(path)
    return datetime.fromtimestamp(mtime, timezone.utc).isoformat() if mtime else ""


def _clear_previous_outputs(output_dir: Path) -> None:
    for name in OUTPUT_FILES:
        path = output_dir / name
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _write_run_status(output_dir: Path, status: str, **extra: object) -> None:
    payload = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    (output_dir / "run_status.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> Path:
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    output_dir = _repo_relative(Path(args.output_dir)) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    _clear_previous_outputs(output_dir)
    target = parse_target(args.target)
    payload_store = PROJECT_ROOT / "debug-logs" / "analysis" / "payloads"
    positions = []
    term_rows = []
    processed_payloads = []
    payload_dir = _repo_relative(Path(args.payload_dir))
    payload_paths = sorted(payload_dir.glob("*.json"), key=lambda path: (-_path_mtime(path), path.name))
    _write_run_status(
        output_dir,
        "running",
        run_id=run_id,
        payload_dir=str(payload_dir),
        run_output_dir=str(output_dir),
        target=target,
        payload_order="mtime-desc",
        payload_files_found=len(payload_paths),
    )
    skipped_payloads = 0
    try:
        if not payload_paths:
            raise RuntimeError(
                f"no payload JSON files found under {payload_dir}; "
                "run from the repo root or pass an absolute --payload-dir"
            )
        for path in payload_paths:
            payload = load_payload(path)
            if payload is None:
                skipped_payloads += 1
                continue
            if args.positions is not None and len(positions) >= int(args.positions):
                break
            digest = store_payload(payload, payload_store)
            processed_payloads.append({
                "label": path.stem,
                "path": str(path),
                "payload_hash": digest,
                "last_modified": _mtime_iso(path),
            })
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
                term_rows.append({
                    "payload_hash": digest,
                    "label": path.stem,
                    "action_rank": 0,
                    "action_id": "root",
                    "action_fingerprint": "root",
                    **term,
                })
            for action in row.get("actions", []):
                rank = action.get("visit_rank")
                if rank in {1, 2, 3} and action.get("value_breakdown") is not None:
                    for term in action["value_breakdown"].get("terms", []):
                        term_rows.append({
                            "payload_hash": digest,
                            "label": path.stem,
                            "action_rank": rank,
                            "action_id": action["action_id"],
                            "action_fingerprint": action["action_fingerprint"],
                            **term,
                        })
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
            fieldnames = [
                "payload_hash",
                "label",
                "action_rank",
                "action_id",
                "action_fingerprint",
                "name",
                "parent",
                "feature_value",
                "weight",
                "raw",
                "normalized",
                "abs_share",
                "linearized_value",
                "tunable",
                "contributes",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(term_rows)
        summary = {
            "target": target,
            "positions": len(positions),
            "terms": len(term_rows),
            "payload_dir": str(payload_dir),
            "payload_files_found": len(payload_paths),
            "payload_order": "mtime-desc",
            "skipped_payloads": skipped_payloads,
            "processed_payloads": processed_payloads,
            "note": "Term rows are exact pre-tanh additive raw-space contributions; linearized_value is local tanh sensitivity, not an ablation.",
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        write_report(output_dir / "report.html", "Static Value Breakdown", summary, term_rows[:1200])
        _write_run_status(
            output_dir,
            "complete",
            run_id=run_id,
            payload_dir=str(payload_dir),
            run_output_dir=str(output_dir),
            target=target,
            positions=len(positions),
            terms=len(term_rows),
        )
    except Exception as exc:
        _write_run_status(
            output_dir,
            "failed",
            run_id=run_id,
            payload_dir=str(payload_dir),
            run_output_dir=str(output_dir),
            target=target,
            error=str(exc),
        )
        raise
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
