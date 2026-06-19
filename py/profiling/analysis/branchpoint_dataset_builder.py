from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from profiling.config import load_config_defaults
from profiling.analysis.distributions import js_bits, normalize
from profiling.analysis.report_html import write_report

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "debug-logs" / "analysis" / "branchpoints"
DEFAULT_CONFIG = PROJECT_ROOT / "py" / "profiling" / "configs" / "branchpoint_dataset_builder.json"


def _repo_relative(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _score_map(position: dict[str, Any], field: str) -> dict[str, float]:
    return {str(action["action_id"]): float(action.get(field, 0.0) or 0.0) for action in position.get("actions", [])}


def _rank_map(position: dict[str, Any]) -> dict[str, int]:
    return {str(action["action_id"]): int(action.get("visit_rank", 999999) or 999999) for action in position.get("actions", [])}


def _top_gap(scores: dict[str, float]) -> float:
    values = sorted(scores.values(), reverse=True)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return values[0] - values[1]


def _top_id(scores: dict[str, float]) -> str:
    return max(scores, key=scores.get, default="")


def _fingerprint_for(position: dict[str, Any], action_id: str) -> str:
    for action in position.get("actions", []):
        if str(action.get("action_id")) == action_id:
            return str(action.get("action_fingerprint", ""))
    return ""


def _counterfactual_rescue_ok(row: dict[str, Any]) -> bool:
    baseline_result = str(row.get("baseline_action_result", row.get("baseline_forced_result", ""))).upper()
    experimental_result = str(row.get("experimental_action_result", row.get("experimental_forced_result", ""))).upper()
    if baseline_result or experimental_result:
        return baseline_result == "WIN" and experimental_result != "WIN"
    if "baseline_action_margin" in row and "experimental_action_margin" in row:
        return _float(row.get("baseline_action_margin")) > 0.0 and _float(row.get("experimental_action_margin")) < 0.0
    return False


def _superior_seat_agrees(row: dict[str, Any]) -> bool:
    baseline_seat = row.get("baseline_superior_seat")
    experimental_seat = row.get("experimental_superior_seat")
    return baseline_seat not in (None, "") and str(baseline_seat) == str(experimental_seat)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_candidates(
    positions: list[dict[str, Any]],
    *,
    baseline_name: str,
    experimental_name: str,
    min_js_visit_bits: float,
    min_baseline_top_visit_share: float,
    min_top_gap: float,
    max_experimental_baseline_visit_share: float,
    min_experimental_baseline_rank: int,
    require_counterfactual_rescue: bool = False,
    require_superior_seat_agreement: bool = False,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for position in positions:
        grouped.setdefault(str(position["payload_hash"]), {})[str(position["target_name"])] = position
    out: list[dict[str, Any]] = []
    for digest, by_target in grouped.items():
        baseline = by_target.get(baseline_name)
        experimental = by_target.get(experimental_name)
        if baseline is None or experimental is None:
            continue
        b_visit = _score_map(baseline, "visit_share")
        e_visit = _score_map(experimental, "visit_share")
        action_ids = sorted(set(b_visit) | set(e_visit))
        baseline_top = _top_id(b_visit)
        experimental_top = _top_id(e_visit)
        js_visit = js_bits(
            normalize([b_visit.get(action_id, 0.0) for action_id in action_ids]),
            normalize([e_visit.get(action_id, 0.0) for action_id in action_ids]),
        )
        e_ranks = _rank_map(experimental)
        baseline_under_exp_rank = e_ranks.get(baseline_top, 999999)
        baseline_under_exp_visit = e_visit.get(baseline_top, 0.0)
        selected_same = baseline.get("selected_action_fingerprint") == experimental.get("selected_action_fingerprint")
        top_same = baseline_top == experimental_top
        keep = (
            (not selected_same or not top_same)
            and js_visit >= min_js_visit_bits
            and b_visit.get(baseline_top, 0.0) >= min_baseline_top_visit_share
            and _top_gap(b_visit) >= min_top_gap
            and (
                baseline_under_exp_rank >= min_experimental_baseline_rank
                or baseline_under_exp_visit <= max_experimental_baseline_visit_share
            )
        )
        if not keep:
            continue
        candidate = {
                "payload_hash": digest,
                "label": baseline.get("label", ""),
                "baseline_target": baseline_name,
                "experimental_target": experimental_name,
                "baseline_action_id": baseline_top,
                "baseline_action_fingerprint": _fingerprint_for(baseline, baseline_top),
                "experimental_action_id": experimental_top,
                "experimental_action_fingerprint": _fingerprint_for(experimental, experimental_top),
                "baseline_top_visit_share": b_visit.get(baseline_top, 0.0),
                "experimental_top_visit_share": e_visit.get(experimental_top, 0.0),
                "baseline_action_visit_under_experimental": baseline_under_exp_visit,
                "baseline_action_rank_under_experimental": baseline_under_exp_rank,
                "js_visit_bits": js_visit,
                "baseline_top_gap": _top_gap(b_visit),
                "selected_same": selected_same,
                "top_same": top_same,
                "snapshot_path": "",
                "classification": "candidate",
            }
        if require_counterfactual_rescue and not _counterfactual_rescue_ok(candidate):
            continue
        if require_superior_seat_agreement and not _superior_seat_agrees(candidate):
            continue
        out.append(candidate)
    return sorted(out, key=lambda row: float(row["js_visit_bits"]), reverse=True)


def run(args: argparse.Namespace) -> Path:
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    output_dir = _repo_relative(Path(args.output_dir)) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    positions_path = _repo_relative(Path(args.positions_jsonl))
    positions = _read_jsonl(positions_path)
    rows = build_candidates(
        positions,
        baseline_name=args.baseline_name,
        experimental_name=args.experimental_name,
        min_js_visit_bits=args.min_js_visit_bits,
        min_baseline_top_visit_share=args.min_baseline_top_visit_share,
        min_top_gap=args.min_top_gap,
        max_experimental_baseline_visit_share=args.max_experimental_baseline_visit_share,
        min_experimental_baseline_rank=args.min_experimental_baseline_rank,
        require_counterfactual_rescue=args.require_counterfactual_rescue,
        require_superior_seat_agreement=args.require_superior_seat_agreement,
    )
    (output_dir / "branchpoints.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in rows),
        encoding="utf-8",
    )
    fieldnames = list(rows[0].keys()) if rows else ["payload_hash"]
    with (output_dir / "branchpoints.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "positions_jsonl": str(positions_path),
        "positions": len(positions),
        "branchpoints": len(rows),
        "thresholds": {
            "min_js_visit_bits": args.min_js_visit_bits,
            "min_baseline_top_visit_share": args.min_baseline_top_visit_share,
            "min_top_gap": args.min_top_gap,
            "max_experimental_baseline_visit_share": args.max_experimental_baseline_visit_share,
            "min_experimental_baseline_rank": args.min_experimental_baseline_rank,
            "require_counterfactual_rescue": args.require_counterfactual_rescue,
            "require_superior_seat_agreement": args.require_superior_seat_agreement,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    write_report(output_dir / "report.html", "Branch Point Candidates", summary, rows[:500])
    print(f"wrote {output_dir}")
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Build branch-point candidates from branch comparison positions.jsonl.")
    parser.add_argument("--positions-jsonl", type=Path)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--experimental-name", default="experimental")
    parser.add_argument("--min-js-visit-bits", type=float, default=0.08)
    parser.add_argument("--min-baseline-top-visit-share", type=float, default=0.35)
    parser.add_argument("--min-top-gap", type=float, default=0.05)
    parser.add_argument("--max-experimental-baseline-visit-share", type=float, default=0.10)
    parser.add_argument("--min-experimental-baseline-rank", type=int, default=4)
    parser.add_argument("--require-counterfactual-rescue", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-superior-seat-agreement", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = load_config_defaults(parser, default_config=DEFAULT_CONFIG)
    if args.positions_jsonl is None:
        raise RuntimeError("branchpoint_dataset_builder requires --positions-jsonl")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
