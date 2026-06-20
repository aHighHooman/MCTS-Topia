from __future__ import annotations

import argparse
import csv
import html
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from profiling.config import load_config_defaults
from profiling.analysis.payload_store import load_payload, store_payload
from profiling.analysis.position_analyzer import analyze_position, parse_target

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "debug-logs" / "analysis" / "value-breakdown"
DEFAULT_CONFIG = PROJECT_ROOT / "py" / "profiling" / "configs" / "value_breakdown.json"
OUTPUT_FILES = (
    "positions.jsonl",
    "terms.csv",
    "terms_components.csv",
    "terms_aggregate.csv",
    "terms_comparable_old.csv",
    "term_summary.csv",
    "term_comparison.csv",
    "summary.json",
    "report.html",
)
TERM_FIELDS = [
    "payload_hash",
    "label",
    "action_rank",
    "action_id",
    "action_fingerprint",
    "name",
    "parent",
    "aggregate_term",
    "row_type",
    "feature_value",
    "weight",
    "raw",
    "normalized",
    "abs_share",
    "linearized_value",
    "tunable",
    "contributes",
]
AGGREGATE_FIELDS = [
    "payload_hash",
    "label",
    "action_rank",
    "action_id",
    "action_fingerprint",
    "name",
    "aggregate_term",
    "raw",
    "abs_raw",
    "normalized",
    "abs_share",
    "linearized_value",
    "component_count",
    "row_type",
]
COMPARABLE_OLD_FIELDS = [
    "payload_hash",
    "label",
    "action_rank",
    "action_id",
    "action_fingerprint",
    "name",
    "feature_value",
    "weight",
    "raw",
    "normalized",
    "abs_share",
    "linearized_value",
]
SUMMARY_FIELDS = ["parent_term", "raw_sum", "abs_raw_sum", "abs_share", "normalized_sum"]
COMPARISON_FIELDS = [
    "term",
    "baseline_share",
    "experimental_share",
    "delta_pp",
    "ratio",
    "baseline_raw_sum",
    "experimental_raw_sum",
]
PRIORITY_TERMS = [
    "military.unit_power",
    "threat.city_pressure",
    "score_terminal.score_diff",
    "economy.stars",
    "economy.city_quality",
    "military.own_unit_material",
    "threat.city_threat",
    "threat.vulnerable_units",
]


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


def _float(value: object, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _truthy(value: object, default: bool = True) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _term_row_type(row: dict[str, Any]) -> str:
    if not _truthy(row.get("contributes"), True):
        return "diagnostic"
    return "component" if str(row.get("parent", "") or "") else "aggregate"


def _annotate_term_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    parent = str(out.get("parent", "") or "")
    out["aggregate_term"] = parent or str(out.get("name", "") or "")
    out["row_type"] = _term_row_type(out)
    return out


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_rows(term_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    for row in term_rows:
        if not _truthy(row.get("contributes"), True):
            continue
        aggregate_term = str(row.get("aggregate_term") or row.get("parent") or row.get("name") or "")
        key = (
            str(row.get("payload_hash", "")),
            str(row.get("label", "")),
            str(row.get("action_rank", "")),
            str(row.get("action_id", "")),
            str(row.get("action_fingerprint", "")),
            aggregate_term,
        )
        agg = grouped.setdefault(
            key,
            {
                "payload_hash": key[0],
                "label": key[1],
                "action_rank": key[2],
                "action_id": key[3],
                "action_fingerprint": key[4],
                "name": aggregate_term,
                "aggregate_term": aggregate_term,
                "raw": 0.0,
                "abs_raw": 0.0,
                "normalized": 0.0,
                "abs_share": 0.0,
                "linearized_value": 0.0,
                "component_count": 0,
                "row_type": "aggregate",
            },
        )
        raw = _float(row.get("raw"))
        agg["raw"] = float(agg["raw"]) + raw
        agg["abs_raw"] = float(agg["abs_raw"]) + abs(raw)
        agg["normalized"] = float(agg["normalized"]) + _float(row.get("normalized"))
        agg["linearized_value"] = float(agg["linearized_value"]) + _float(row.get("linearized_value"))
        agg["component_count"] = int(agg["component_count"]) + 1
    totals: dict[tuple[str, str, str, str, str], float] = {}
    for row in grouped.values():
        key = (
            str(row["payload_hash"]),
            str(row["label"]),
            str(row["action_rank"]),
            str(row["action_id"]),
            str(row["action_fingerprint"]),
        )
        totals[key] = totals.get(key, 0.0) + float(row["abs_raw"])
    for row in grouped.values():
        key = (
            str(row["payload_hash"]),
            str(row["label"]),
            str(row["action_rank"]),
            str(row["action_id"]),
            str(row["action_fingerprint"]),
        )
        total = totals.get(key, 0.0)
        row["abs_share"] = float(row["abs_raw"]) / total if total else 0.0
    return sorted(grouped.values(), key=lambda row: (str(row["label"]), int(row["action_rank"] or 0), str(row["name"])))


def _comparable_old_rows(aggregate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "payload_hash": row.get("payload_hash", ""),
            "label": row.get("label", ""),
            "action_rank": row.get("action_rank", ""),
            "action_id": row.get("action_id", ""),
            "action_fingerprint": row.get("action_fingerprint", ""),
            "name": row.get("name", ""),
            "feature_value": "",
            "weight": "",
            "raw": row.get("raw", 0.0),
            "normalized": row.get("normalized", 0.0),
            "abs_share": row.get("abs_share", 0.0),
            "linearized_value": row.get("linearized_value", 0.0),
        }
        for row in aggregate_rows
    ]


def _summary_rows(aggregate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in aggregate_rows:
        term = str(row.get("name", ""))
        out = grouped.setdefault(
            term,
            {"parent_term": term, "raw_sum": 0.0, "abs_raw_sum": 0.0, "abs_share": 0.0, "normalized_sum": 0.0},
        )
        out["raw_sum"] = float(out["raw_sum"]) + _float(row.get("raw"))
        out["abs_raw_sum"] = float(out["abs_raw_sum"]) + _float(row.get("abs_raw"))
        out["normalized_sum"] = float(out["normalized_sum"]) + _float(row.get("normalized"))
    total_abs = sum(float(row["abs_raw_sum"]) for row in grouped.values())
    for row in grouped.values():
        row["abs_share"] = float(row["abs_raw_sum"]) / total_abs if total_abs else 0.0
    return sorted(grouped.values(), key=lambda row: -float(row["abs_raw_sum"]))


def _category_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in summary_rows:
        category = str(row["parent_term"]).split(".", 1)[0]
        out = grouped.setdefault(
            category,
            {"category": category, "raw_sum": 0.0, "abs_raw_sum": 0.0, "abs_share": 0.0, "normalized_sum": 0.0},
        )
        out["raw_sum"] = float(out["raw_sum"]) + _float(row.get("raw_sum"))
        out["abs_raw_sum"] = float(out["abs_raw_sum"]) + _float(row.get("abs_raw_sum"))
        out["normalized_sum"] = float(out["normalized_sum"]) + _float(row.get("normalized_sum"))
    total_abs = sum(float(row["abs_raw_sum"]) for row in grouped.values())
    for row in grouped.values():
        row["abs_share"] = float(row["abs_raw_sum"]) / total_abs if total_abs else 0.0
    return sorted(grouped.values(), key=lambda row: -float(row["abs_raw_sum"]))


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_aggregate_rows(run_dir: Path) -> list[dict[str, Any]]:
    aggregate_path = run_dir / "terms_aggregate.csv"
    if aggregate_path.exists():
        return _read_csv(aggregate_path)
    terms_path = run_dir / "terms.csv"
    if not terms_path.exists():
        raise RuntimeError(f"compare_dir has no terms_aggregate.csv or terms.csv: {run_dir}")
    return _aggregate_rows([_annotate_term_row(row) for row in _read_csv(terms_path)])


def _comparison_rows(baseline_rows: list[dict[str, Any]], experimental_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = {str(row["parent_term"]): row for row in _summary_rows(baseline_rows)}
    experimental = {str(row["parent_term"]): row for row in _summary_rows(experimental_rows)}
    terms = sorted(set(baseline) | set(experimental))
    terms.sort(key=lambda term: (0 if term in PRIORITY_TERMS else 1, PRIORITY_TERMS.index(term) if term in PRIORITY_TERMS else term))
    rows = []
    for term in terms:
        base_share = _float(baseline.get(term, {}).get("abs_share"))
        exp_share = _float(experimental.get(term, {}).get("abs_share"))
        ratio = (exp_share / base_share) if base_share else (float("inf") if exp_share else 0.0)
        rows.append(
            {
                "term": term,
                "baseline_share": base_share,
                "experimental_share": exp_share,
                "delta_pp": (exp_share - base_share) * 100.0,
                "ratio": ratio,
                "baseline_raw_sum": _float(baseline.get(term, {}).get("raw_sum")),
                "experimental_raw_sum": _float(experimental.get(term, {}).get("raw_sum")),
            }
        )
    return rows


def _action_mix_rows(positions: list[dict[str, Any]], rank: int) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], int] = {}
    for position in positions:
        for action in position.get("actions", []):
            if int(action.get("visit_rank", 0) or 0) == rank:
                key = (str(action.get("action_fingerprint", "")), str(action.get("action_type", "")))
                counts[key] = counts.get(key, 0) + 1
                break
    return [
        {"rank": rank, "action_fingerprint": key[0], "action_type": key[1], "count": count}
        for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0][0]))[:25]
    ]


def _saturation_rows(positions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    roots = [_float(position.get("root_value")) for position in positions]
    if not roots:
        return []
    saturated = sum(1 for value in roots if abs(value) >= 0.95)
    return [
        {
            "positions": len(roots),
            "root_value_min": min(roots),
            "root_value_max": max(roots),
            "root_value_avg": sum(roots) / len(roots),
            "abs_ge_0_95": saturated,
            "abs_ge_0_95_share": saturated / len(roots),
        }
    ]


def _visit_confidence_rows(positions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    margins = []
    top1_shares = []
    for position in positions:
        ranked = sorted(position.get("actions", []), key=lambda action: int(action.get("visit_rank", 999999) or 999999))
        if ranked:
            top1 = _float(ranked[0].get("visit_share"))
            top1_shares.append(top1)
            top2 = _float(ranked[1].get("visit_share")) if len(ranked) > 1 else 0.0
            margins.append(top1 - top2)
    if not top1_shares:
        return []
    return [
        {
            "positions": len(top1_shares),
            "top1_visit_share_avg": sum(top1_shares) / len(top1_shares),
            "top1_vs_top2_margin_avg": sum(margins) / len(margins) if margins else 0.0,
            "top1_visit_share_min": min(top1_shares),
            "top1_visit_share_max": max(top1_shares),
        }
    ]


def _hinge_rows(positions: list[dict[str, Any]], aggregate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_action: dict[tuple[str, str], dict[str, float]] = {}
    for row in aggregate_rows:
        by_action.setdefault((str(row.get("payload_hash", "")), str(row.get("action_id", ""))), {})[str(row.get("name", ""))] = _float(row.get("raw"))
    totals: dict[str, dict[str, Any]] = {}
    for position in positions:
        ranked = sorted(position.get("actions", []), key=lambda action: int(action.get("visit_rank", 999999) or 999999))
        if len(ranked) < 2:
            continue
        payload_hash = str(position.get("payload_hash", ""))
        top1 = by_action.get((payload_hash, str(ranked[0].get("action_id", ""))), {})
        top2 = by_action.get((payload_hash, str(ranked[1].get("action_id", ""))), {})
        for term in set(top1) | set(top2):
            delta = top1.get(term, 0.0) - top2.get(term, 0.0)
            out = totals.setdefault(term, {"term": term, "delta_raw_sum": 0.0, "abs_delta_raw_sum": 0.0, "positions": 0})
            out["delta_raw_sum"] = float(out["delta_raw_sum"]) + delta
            out["abs_delta_raw_sum"] = float(out["abs_delta_raw_sum"]) + abs(delta)
            out["positions"] = int(out["positions"]) + 1
    return sorted(totals.values(), key=lambda row: -float(row["abs_delta_raw_sum"]))[:25]


def _table(headers: list[str], rows: list[dict[str, Any]], limit: int = 50) -> str:
    if not rows:
        return "<p>No rows.</p>"
    body = ["<table><thead><tr>"]
    body.extend(f"<th>{html.escape(header)}</th>" for header in headers)
    body.append("</tr></thead><tbody>")
    for row in rows[:limit]:
        body.append("<tr>")
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                value = f"{value:.6g}"
            body.append(f"<td>{html.escape(str(value))}</td>")
        body.append("</tr>")
    body.append("</tbody></table>")
    if len(rows) > limit:
        body.append(f"<p>Showing {limit} of {len(rows)} rows.</p>")
    return "\n".join(body)


def _write_value_report(
    path: Path,
    summary: dict[str, Any],
    validation: dict[str, Any],
    warnings: list[str],
    summary_rows: list[dict[str, Any]],
    category_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    positions: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    term_rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    root_rows = [row for row in summary_rows if row["parent_term"]]
    top1_rows = _action_mix_rows(positions, 1)
    top2_rows = _action_mix_rows(positions, 2)
    component_rows = [row for row in term_rows if row.get("row_type") == "component"]
    body = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Static Value Breakdown</title>",
        "<style>body{font-family:system-ui,Segoe UI,sans-serif;margin:24px;color:#1f2933}table{border-collapse:collapse;margin:12px 0 24px;max-width:100%;font-size:13px}td,th{border:1px solid #d7dde5;padding:4px 8px;text-align:left;vertical-align:top}th{background:#f4f6f8}pre{background:#f7f9fb;padding:12px;overflow:auto}.warn{border:1px solid #b45309;background:#fff7ed;color:#7c2d12;padding:12px;margin:12px 0}.ok{border:1px solid #16803c;background:#f0fff4;color:#14532d;padding:12px;margin:12px 0}h2{margin-top:28px}</style>",
        "<h1>Static Value Breakdown</h1>",
    ]
    if warnings:
        for warning in warnings:
            body.append(f"<div class='warn'><strong>{html.escape('Warning: ' + warning)}</strong></div>")
    else:
        body.append("<div class='ok'><strong>Status:</strong> complete</div>")
    body.extend(
        [
            "<h2>Run Metadata</h2>",
            f"<pre>{html.escape(json.dumps(summary, indent=2, sort_keys=True, default=str))}</pre>",
            "<h2>Validation / Consistency Checks</h2>",
            f"<pre>{html.escape(json.dumps(validation, indent=2, sort_keys=True, default=str))}</pre>",
            "<h2>Quick Verdict</h2>",
            _table(["positions", "root_value_min", "root_value_max", "root_value_avg", "abs_ge_0_95", "abs_ge_0_95_share"], _saturation_rows(positions), 5),
            "<h2>Root Aggregate Term Shares</h2>",
            _table(SUMMARY_FIELDS, root_rows, 30),
            "<h2>Root Category Shares</h2>",
            _table(["category", "raw_sum", "abs_raw_sum", "abs_share", "normalized_sum"], category_rows, 20),
            "<h2>Saturation Stats</h2>",
            _table(["positions", "root_value_min", "root_value_max", "root_value_avg", "abs_ge_0_95", "abs_ge_0_95_share"], _saturation_rows(positions), 5),
            "<h2>Visit Confidence Stats</h2>",
            _table(["positions", "top1_visit_share_avg", "top1_vs_top2_margin_avg", "top1_visit_share_min", "top1_visit_share_max"], _visit_confidence_rows(positions), 5),
            "<h2>Action Mix: Top-1</h2>",
            _table(["rank", "action_fingerprint", "action_type", "count"], top1_rows, 25),
            "<h2>Action Mix: Top-2</h2>",
            _table(["rank", "action_fingerprint", "action_type", "count"], top2_rows, 25),
            "<h2>Top1-vs-Top2 Hinge Terms</h2>",
            _table(["term", "delta_raw_sum", "abs_delta_raw_sum", "positions"], _hinge_rows(positions, aggregate_rows), 25),
        ]
    )
    if comparison_rows:
        body.extend(["<h2>Baseline Comparison</h2>", _table(COMPARISON_FIELDS, comparison_rows, 40)])
    body.extend(
        [
            "<h2>Component Breakdown</h2>",
            _table(TERM_FIELDS, component_rows, 100),
            "<h2>Full Raw Table Preview</h2>",
            _table(TERM_FIELDS, term_rows, 200),
        ]
    )
    path.write_text("\n".join(body), encoding="utf-8")


def _validation(
    *,
    status: str,
    run_id: str,
    positions: list[dict[str, Any]],
    term_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    summary_terms: int,
    target: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": status,
        "run_id": run_id,
        "positions_jsonl_count": len(positions),
        "terms_csv_rows": len(term_rows),
        "components_csv_rows": len(term_rows),
        "aggregate_csv_rows": len(aggregate_rows),
        "summary_terms": summary_terms,
        "report_terms": len(term_rows),
        "terms_match_summary": summary_terms == len(term_rows),
        "target_name": target.get("name", ""),
        "target_variant": target.get("variant", ""),
    }


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
                term_rows.append(_annotate_term_row({
                    "payload_hash": digest,
                    "label": path.stem,
                    "action_rank": 0,
                    "action_id": "root",
                    "action_fingerprint": "root",
                    **term,
                }))
            for action in row.get("actions", []):
                rank = action.get("visit_rank")
                if rank in {1, 2, 3} and action.get("value_breakdown") is not None:
                    for term in action["value_breakdown"].get("terms", []):
                        term_rows.append(_annotate_term_row({
                            "payload_hash": digest,
                            "label": path.stem,
                            "action_rank": rank,
                            "action_id": action["action_id"],
                            "action_fingerprint": action["action_fingerprint"],
                            **term,
                        }))
        if not positions:
            raise RuntimeError(
                f"no valid payload positions loaded from {payload_dir}; "
                f"checked={len(payload_paths)} skipped={skipped_payloads}"
            )
        (output_dir / "positions.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in positions),
            encoding="utf-8",
        )
        aggregate_rows = _aggregate_rows(term_rows)
        comparable_old_rows = _comparable_old_rows(aggregate_rows)
        summary_rows = _summary_rows(aggregate_rows)
        category_rows = _category_rows(summary_rows)
        comparison_rows: list[dict[str, Any]] = []
        compare_dir = str(getattr(args, "compare_dir", "") or "")
        if compare_dir:
            comparison_rows = _comparison_rows(_load_aggregate_rows(_repo_relative(Path(compare_dir))), aggregate_rows)
            _write_csv(output_dir / "term_comparison.csv", COMPARISON_FIELDS, comparison_rows)
        _write_csv(output_dir / "terms.csv", TERM_FIELDS, term_rows)
        _write_csv(output_dir / "terms_components.csv", TERM_FIELDS, term_rows)
        _write_csv(output_dir / "terms_aggregate.csv", AGGREGATE_FIELDS, aggregate_rows)
        _write_csv(output_dir / "terms_comparable_old.csv", COMPARABLE_OLD_FIELDS, comparable_old_rows)
        _write_csv(output_dir / "term_summary.csv", SUMMARY_FIELDS, summary_rows)
        has_components = any(row.get("row_type") == "component" for row in term_rows)
        has_diagnostics = any(row.get("row_type") == "diagnostic" for row in term_rows)
        warnings = []
        if has_components and has_diagnostics:
            warnings.append(
                "This report contains component rows and parent/diagnostic rows. Do not sum all rows directly. Use contributes=True grouped by parent, or terms_aggregate.csv."
            )
        validation_status = "complete"
        validation = _validation(
            status=validation_status,
            run_id=run_id,
            positions=positions,
            term_rows=term_rows,
            aggregate_rows=aggregate_rows,
            summary_terms=len(term_rows),
            target=target,
        )
        if not validation["terms_match_summary"]:
            validation_status = "complete_with_warnings"
            validation["status"] = validation_status
            warnings.append("Generated term counts disagree with summary metadata.")
        summary = {
            "target": target,
            "positions": len(positions),
            "terms": len(term_rows),
            "aggregate_terms": len(aggregate_rows),
            "component_terms": sum(1 for row in term_rows if row.get("row_type") == "component"),
            "diagnostic_terms": sum(1 for row in term_rows if row.get("row_type") == "diagnostic"),
            "payload_dir": str(payload_dir),
            "payload_files_found": len(payload_paths),
            "payload_order": "mtime-desc",
            "skipped_payloads": skipped_payloads,
            "processed_payloads": processed_payloads,
            "compare_dir": compare_dir,
            "validation": validation,
            "note": "Term rows are exact pre-tanh additive raw-space contributions; linearized_value is local tanh sensitivity, not an ablation.",
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        _write_value_report(
            output_dir / "report.html",
            summary,
            validation,
            warnings,
            summary_rows,
            category_rows,
            comparison_rows,
            positions,
            aggregate_rows,
            term_rows,
        )
        _write_run_status(
            output_dir,
            validation_status,
            run_id=run_id,
            payload_dir=str(payload_dir),
            run_output_dir=str(output_dir),
            target=target,
            positions=len(positions),
            terms=len(term_rows),
            validation=validation,
            warnings=warnings,
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
    parser.add_argument("--compare-dir", default="")
    args = load_config_defaults(parser, default_config=DEFAULT_CONFIG)
    if not args.target:
        raise RuntimeError("value_breakdown requires target in config or --target")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
