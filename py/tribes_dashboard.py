from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from xml.sax.saxutils import escape as xml_escape


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable or "python"
RL_ROOT = ROOT / "rl"


@dataclass
class Job:
    id: str
    kind: str
    name: str
    command: list[str]
    started_at: float
    log_path: str
    returncode: int | None = None
    stopped: bool = False
    pid: int | None = None


JOBS: dict[str, Job] = {}
PROCESSES: dict[str, subprocess.Popen[str]] = {}
LOCK = threading.Lock()
GAME_DONE_RE = re.compile(r"\[game (?P<game>\d+)/(?P<games>\d+)\] done time=(?P<time>\S+) steps=(?P<steps>\d+)")
ITER_RE = re.compile(r"\[iter (?P<local>\d+)/(?:\d+) \| global (?P<iteration>\d+)\]")
SIMS_RE = re.compile(r"--simulations\s+(?P<sims>\d+)")


def _json_response(handler: BaseHTTPRequestHandler, payload: Any, status: int = 200) -> None:
    body = json.dumps(payload, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _text_response(handler: BaseHTTPRequestHandler, body: str, content_type: str = "text/html; charset=utf-8") -> None:
    encoded = body.encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(encoded)))
    handler.end_headers()
    handler.wfile.write(encoded)


def _binary_response(handler: BaseHTTPRequestHandler, body: bytes, filename: str, content_type: str) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Content-Disposition", f'attachment; filename="{filename}"')
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _read_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    if length <= 0:
        return {}
    raw = handler.rfile.read(length).decode("utf-8")
    return json.loads(raw or "{}")


def _tail(path: Path, max_lines: int = 240) -> list[str]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError:
        return []
    return [line.rstrip("\n") for line in lines[-max_lines:]]


def _read_csv(path: Path, limit: int = 500) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
            rows = list(csv.DictReader(handle, restkey="_extra"))
    except Exception:
        return []
    if path.name == "metrics.csv":
        rows = [_normalize_metric_row(row) for row in rows]
    return rows[-limit:]


def _normalize_metric_row(row: dict[str, Any]) -> dict[str, str]:
    extra = row.pop("_extra", None) or []
    normalized = {str(key): "" if value is None else str(value) for key, value in row.items()}
    if len(extra) >= 8:
        normalized["wins"] = normalized.get("loss", "")
        normalized["losses"] = normalized.get("policy_loss", "")
        normalized["draws"] = normalized.get("value_loss", "")
        tail = [str(value) for value in extra[-8:]]
        normalized["loss"] = tail[0]
        normalized["policy_loss"] = tail[1]
        normalized["value_loss"] = tail[2]
        normalized["steps"] = tail[4]
        normalized["sample_sec"] = tail[5]
        normalized["fetch_sec"] = tail[6]
        normalized["optimize_sec"] = tail[7]
    return normalized


def _tensorboard_event_files() -> list[dict[str, Any]]:
    root = RL_ROOT
    if not root.exists():
        return []
    files = sorted(root.rglob("events.out.tfevents*"), key=lambda path: path.stat().st_mtime, reverse=True)
    return [_file_info(path) for path in files[:20]]


def _read_jsonl(path: Path, limit: int = 300) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in _tail(path, limit):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _nested_number(row: dict[str, Any], path: list[str]) -> float | str:
    value: Any = row
    for key in path:
        if not isinstance(value, dict):
            return ""
        value = value.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return ""


def _augmentation_event_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "created_at_utc": row.get("created_at_utc", ""),
                "variant": row.get("variant", ""),
                "seed": row.get("seed", ""),
                "train_sec": row.get("train_sec", ""),
                "eval_loss": _nested_number(row, ["eval_after", "loss"]),
                "d4_eval_loss": _nested_number(row, ["d4_eval_after", "loss"]) or _nested_number(row, ["symmetry_eval_after", "loss"]),
                "d4_eval_gap": row.get("d4_eval_gap_after", row.get("symmetry_gap_after", "")),
                "policy_js": _nested_number(row, ["symmetry_consistency_after", "policy_js"]),
                "policy_l1": _nested_number(row, ["symmetry_consistency_after", "policy_l1"]),
                "value_std": _nested_number(row, ["symmetry_consistency_after", "value_std"]),
                "policy_js_delta": row.get("symmetry_policy_js_delta", ""),
            }
        )
    return out


def _parse_duration_seconds(raw: str) -> float:
    text = raw.strip().lower()
    if text.endswith("ms"):
        return float(text[:-2]) / 1000.0
    if text.endswith("s") and "m" not in text and "h" not in text:
        return float(text[:-1])
    match = re.fullmatch(r"(?:(?P<h>\d+)h)?(?:(?P<m>\d+)m)?(?P<s>[0-9.]+)s", text)
    if match:
        return int(match.group("h") or 0) * 3600 + int(match.group("m") or 0) * 60 + float(match.group("s") or 0)
    return 0.0


def _training_game_rows(limit: int = 1000) -> list[dict[str, Any]]:
    game_metrics_path = RL_ROOT / "selfplay_games.csv"
    structured_rows = _read_csv(game_metrics_path, limit)
    if structured_rows:
        for row in structured_rows:
            row.setdefault("source_log", str(game_metrics_path.relative_to(ROOT)))
        return structured_rows[-limit:]

    log_dir = RL_ROOT / "dashboard" / "logs"
    if not log_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(log_dir.glob("*_train_*.log"), key=lambda item: item.stat().st_mtime):
        text = path.read_text(encoding="utf-8", errors="replace")
        command_line = text.splitlines()[0] if text else ""
        sims_match = SIMS_RE.search(command_line)
        simulations = int(sims_match.group("sims")) if sims_match else 0
        iteration = None
        for line in text.splitlines():
            iter_match = ITER_RE.search(line)
            if iter_match:
                iteration = int(iter_match.group("iteration"))
                continue
            game_match = GAME_DONE_RE.search(line)
            if not game_match:
                continue
            seconds = _parse_duration_seconds(game_match.group("time"))
            steps = int(game_match.group("steps"))
            actions_sec = steps / seconds if seconds > 0 else 0.0
            rows.append(
                {
                    "source_log": str(path.relative_to(ROOT)),
                    "iteration": iteration,
                    "game": int(game_match.group("game")),
                    "games_in_iteration": int(game_match.group("games")),
                    "steps": steps,
                    "game_seconds": round(seconds, 3),
                    "actions_sec": round(actions_sec, 4),
                    "configured_sims_sec": round(actions_sec * simulations, 4) if simulations else "",
                    "simulations": simulations,
                }
            )
    return rows[-limit:]


def _latest_file(directory: Path, pattern: str) -> Path | None:
    files = [path for path in directory.glob(pattern) if path.is_file()]
    if not files:
        return None
    return max(files, key=lambda path: path.stat().st_mtime)


def _latest_nonempty_file(directory: Path, pattern: str) -> Path | None:
    if not directory.exists():
        return None
    files = [path for path in directory.rglob(pattern) if path.is_file() and path.stat().st_size > 0]
    if not files:
        return None
    return max(files, key=lambda path: path.stat().st_mtime)


def _file_info(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
        "bytes": stat.st_size,
        "modified": stat.st_mtime,
    }


def _artifact_summary() -> dict[str, Any]:
    replay_dir = RL_ROOT / "replay"
    checkpoint_dir = RL_ROOT / "checkpoints"
    analytics_dir = RL_ROOT / "analytics_benchmarks"
    augmentation_dir = RL_ROOT / "augmentation_benchmarks"
    tournament_dir = RL_ROOT / "tournaments"
    metrics_path = RL_ROOT / "metrics.csv"

    replay_shards = list(replay_dir.glob("replay_*.pt")) if replay_dir.exists() else []
    checkpoints = list(checkpoint_dir.glob("*.pt")) if checkpoint_dir.exists() else []
    analytics_csv = _latest_nonempty_file(analytics_dir, "analytics_benchmark_*.csv")
    analytics_jsonl = _latest_nonempty_file(analytics_dir, "analytics_benchmark_*.jsonl")
    augmentation_csv = _latest_nonempty_file(augmentation_dir, "augmentation_benchmark_summary.csv")
    augmentation_jsonl = _latest_nonempty_file(augmentation_dir, "augmentation_benchmark_*.jsonl")
    augmentation_events = _read_jsonl(augmentation_jsonl) if augmentation_jsonl else []
    tournament_csv = _latest_nonempty_file(tournament_dir, "*summary*.csv")
    tournament_jsonl = _latest_nonempty_file(tournament_dir, "checkpoint_tournament_*.jsonl")

    return {
        "root": str(ROOT),
        "metrics_path": str(metrics_path.relative_to(ROOT)),
        "metrics_rows": _read_csv(metrics_path),
        "training_game_rows": _training_game_rows(),
        "replay_shards": len(replay_shards),
        "replay_bytes": sum(path.stat().st_size for path in replay_shards),
        "checkpoints": [_file_info(path) for path in sorted(checkpoints, key=lambda item: item.stat().st_mtime, reverse=True)[:20]],
        "analytics_csv": _file_info(analytics_csv) if analytics_csv else None,
        "analytics_rows": _read_csv(analytics_csv) if analytics_csv else [],
        "analytics_jsonl": _file_info(analytics_jsonl) if analytics_jsonl else None,
        "analytics_events": _read_jsonl(analytics_jsonl) if analytics_jsonl else [],
        "augmentation_csv": _file_info(augmentation_csv) if augmentation_csv else None,
        "augmentation_rows": _read_csv(augmentation_csv) if augmentation_csv else [],
        "augmentation_jsonl": _file_info(augmentation_jsonl) if augmentation_jsonl else None,
        "augmentation_events": augmentation_events,
        "augmentation_event_rows": _augmentation_event_rows(augmentation_events),
        "tournament_csv": _file_info(tournament_csv) if tournament_csv else None,
        "tournament_rows": _read_csv(tournament_csv) if tournament_csv else [],
        "tournament_jsonl": _file_info(tournament_jsonl) if tournament_jsonl else None,
        "tournament_events": _read_jsonl(tournament_jsonl) if tournament_jsonl else [],
        "tensorboard_events": _tensorboard_event_files(),
        "updated_at": time.time(),
    }


def _sheet_name(name: str) -> str:
    cleaned = "".join(ch for ch in name if ch not in r'[]:*?/\\').strip() or "Sheet"
    return cleaned[:31]


def _ordered_table(rows: list[dict[str, Any]], preferred: list[str] | None = None) -> list[list[Any]]:
    if not rows:
        return [["No data"]]
    keys: list[str] = []
    for key in preferred or []:
        if any(key in row for row in rows):
            keys.append(key)
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    out = [keys]
    for row in rows:
        out.append([row.get(key, "") for key in keys])
    return out


def _cell_ref(row: int, col: int) -> str:
    letters = ""
    while col:
        col, rem = divmod(col - 1, 26)
        letters = chr(65 + rem) + letters
    return f"{letters}{row}"


def _xlsx_cell(value: Any, row: int, col: int) -> str:
    ref = _cell_ref(row, col)
    if value is None:
        return f'<c r="{ref}"/>'
    if isinstance(value, bool):
        return f'<c r="{ref}" t="b"><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{ref}"><v>{value}</v></c>'
    text = str(value)
    try:
        numeric = float(text)
        if text.strip() and text.strip().replace(".", "", 1).replace("-", "", 1).isdigit():
            return f'<c r="{ref}"><v>{xml_escape(text)}</v></c>'
    except ValueError:
        pass
    return f'<c r="{ref}" t="inlineStr"><is><t>{xml_escape(text)}</t></is></c>'


def _worksheet_xml(rows: list[list[Any]]) -> str:
    sheet_rows = []
    for row_idx, row in enumerate(rows, start=1):
        cells = "".join(_xlsx_cell(value, row_idx, col_idx) for col_idx, value in enumerate(row, start=1))
        sheet_rows.append(f'<row r="{row_idx}">{cells}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        '<sheetData>'
        + "".join(sheet_rows)
        + '</sheetData></worksheet>'
    )


def _build_export_workbook() -> bytes:
    state = _artifact_summary()
    summary_rows = [
        ["Metric", "Value"],
        ["Exported at", datetime_from_epoch(time.time())],
        ["Replay shards", state["replay_shards"]],
        ["Replay MB", round(float(state["replay_bytes"]) / (1024 * 1024), 3)],
        ["Checkpoints", len(state["checkpoints"])],
        ["Training rows", len(state["metrics_rows"])],
        ["Training game rows", len(state["training_game_rows"])],
        ["Analytics rows", len(state["analytics_rows"])],
        ["Augmentation rows", len(state["augmentation_event_rows"])],
        ["Tournament rows", len(state["tournament_rows"])],
        ["Metrics source", state["metrics_path"]],
        ["Analytics source", state["analytics_csv"]["path"] if state["analytics_csv"] else ""],
        ["Augmentation source", state["augmentation_jsonl"]["path"] if state["augmentation_jsonl"] else ""],
        ["Tournament source", state["tournament_csv"]["path"] if state["tournament_csv"] else ""],
    ]
    sheets: list[tuple[str, list[list[Any]]]] = [
        ("Summary", summary_rows),
        ("Training", _ordered_table(state["metrics_rows"], ["iteration", "replay_steps", "replay_shards", "games_completed", "wins", "losses", "draws", "loss", "policy_loss", "value_loss", "seconds_per_game", "action_decisions_sec", "configured_sims_sec", "action_decisions", "steps", "selfplay_sec", "train_sec", "checkpoint_sec", "total_sec", "sample_sec", "fetch_sec", "optimize_sec"])),
        ("Training Games", _ordered_table(state["training_game_rows"], ["iteration", "game", "steps", "game_seconds", "actions_sec", "configured_sims_sec", "simulations", "source_log"])),
        ("Analytics", _ordered_table(state["analytics_rows"], ["run_index", "simulations", "search_batch_size", "top_k_actions", "elapsed_sec", "games_completed", "replay_steps_sec", "action_total_ms_mean", "action_search_ms_mean", "configured_simulations_sec", "bottleneck_hint"])),
        ("Augmentation", _ordered_table(state["augmentation_event_rows"], ["created_at_utc", "variant", "train_sec", "eval_loss", "d4_eval_loss", "d4_eval_gap", "policy_js", "policy_l1"])),
        ("Augment Summary", _ordered_table(state["augmentation_rows"], ["variant", "eval_loss_after_mean", "d4_eval_loss_after_mean", "d4_eval_gap_after_mean", "symmetry_policy_js_after_mean", "symmetry_policy_l1_after_mean"])),
        ("Tournament Elo", _ordered_table(state["tournament_rows"], ["name", "iteration", "elo", "games", "score", "score_rate", "checkpoint"])),
        ("Checkpoints", _ordered_table(state["checkpoints"], ["path", "bytes", "modified"])),
        ("TensorBoard Events", _ordered_table(state["tensorboard_events"], ["path", "bytes", "modified"])),
    ]
    return _xlsx_bytes(sheets)


def datetime_from_epoch(value: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(value))


def _xlsx_bytes(sheets: list[tuple[str, list[list[Any]]]]) -> bytes:
    import io

    output = io.BytesIO()
    safe_sheets = [(_sheet_name(name), rows) for name, rows in sheets]
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _content_types_xml(len(safe_sheets)))
        zf.writestr("_rels/.rels", _root_rels_xml())
        zf.writestr("xl/workbook.xml", _workbook_xml([name for name, _ in safe_sheets]))
        zf.writestr("xl/_rels/workbook.xml.rels", _workbook_rels_xml(len(safe_sheets)))
        zf.writestr("xl/styles.xml", _styles_xml())
        for idx, (_, rows) in enumerate(safe_sheets, start=1):
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", _worksheet_xml(rows))
    return output.getvalue()


def _content_types_xml(sheet_count: int) -> str:
    sheets = "".join(f'<Override PartName="/xl/worksheets/sheet{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for idx in range(1, sheet_count + 1))
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        f'{sheets}</Types>'
    )


def _root_rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '</Relationships>'
    )


def _workbook_xml(sheet_names: list[str]) -> str:
    sheets = "".join(f'<sheet name="{xml_escape(name)}" sheetId="{idx}" r:id="rId{idx}"/>' for idx, name in enumerate(sheet_names, start=1))
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets>{sheets}</sheets></workbook>'
    )


def _workbook_rels_xml(sheet_count: int) -> str:
    rels = "".join(
        f'<Relationship Id="rId{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{idx}.xml"/>'
        for idx in range(1, sheet_count + 1)
    )
    rels += f'<Relationship Id="rId{sheet_count + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    return f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{rels}</Relationships>'


def _styles_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
        '</styleSheet>'
    )


def _compact_args(args: dict[str, Any], keys: list[str]) -> list[str]:
    out: list[str] = []
    for key in keys:
        value = args.get(key)
        if value is None or value == "" or value is False:
            continue
        cli_key = "--" + key.replace("_", "-")
        if value is True:
            out.append(cli_key)
        elif isinstance(value, list):
            out.append(cli_key)
            out.extend(str(item) for item in value if str(item) != "")
        else:
            out.extend([cli_key, str(value)])
    return out


def _build_command(kind: str, args: dict[str, Any]) -> list[str]:
    if kind == "train":
        keys = [
            "iterations", "games_per_iteration", "simulations", "max_depth", "top_k_actions",
            "batch_size", "search_batch_size", "replay_batch_size", "max_turns_capitals",
            "max_actions_per_turn", "max_actions_per_game", "match_timeout_seconds",
            "external_action_timeout_ms", "wall_clock_per_action_seconds", "progress_interval_seconds", "selfplay_workers",
            "allow_unsafe_workers", "allow_partial_selfplay", "profile_selfplay",
            "persistent_bot", "no_persistent_bot", "augment_symmetries", "augmentation_prob",
        ]
        return [PYTHON, "-m", "tribes_rl.train", *_compact_args(args, keys)]
    if kind == "analytics":
        keys = [
            "simulations", "search_batch_sizes", "top_k_actions", "max_turns_capitals",
            "external_action_timeout_ms", "games", "seed_base",
            "match_timeout_seconds",
        ]
        return [PYTHON, "-m", "tribes_rl.benchmark_analytics", *_compact_args(args, keys)]
    if kind == "augmentation":
        keys = [
            "training_iterations", "validation_fraction",
            "training_batch_size",
            "replay_batch_size", "epochs_per_iteration", "expanded_replay_batch_multiplier",
            "max_samples_per_shard", "generate_selfplay_games", "simulations",
            "search_batch_size", "max_turns_capitals", "match_timeout_seconds",
            "checkpoint", "require_checkpoint", "no_d4_expanded",
        ]
        return [PYTHON, "-m", "tribes_rl.benchmark_augmentation", *_compact_args(args, keys)]
    if kind == "tournament":
        keys = [
            "checkpoint_dir", "pattern", "limit", "games_per_pair", "seed_base",
            "latest_vs_history", "history_step",
            "initial_elo", "k_factor", "simulations", "search_batch_size", "max_depth",
            "top_k_actions", "max_turns_capitals", "max_actions_per_turn",
            "max_actions_per_game", "match_timeout_seconds", "external_action_timeout_ms",
            "level_file", "deterministic", "stop_on_error",
        ]
        return [PYTHON, "-m", "tribes_rl.checkpoint_tournament", *_compact_args(args, keys)]
    if kind == "java_headless":
        config = args.get("config") or "play.json"
        return ["java", "-cp", "out;lib/json.jar", "HeadlessPlay", str(config)]
    if kind == "tensorboard":
        logdir = args.get("logdir") or "rl"
        port = str(args.get("port") or "6006")
        return [PYTHON, "-m", "tensorboard.main", "--logdir", str(logdir), "--host", "127.0.0.1", "--port", port]
    raise ValueError(f"Unknown job kind: {kind}")


def _watch_job(job_id: str, process: subprocess.Popen[str]) -> None:
    returncode = process.wait()
    with LOCK:
        job = JOBS.get(job_id)
        if job:
            job.returncode = int(returncode)
        PROCESSES.pop(job_id, None)


def _start_job(kind: str, args: dict[str, Any]) -> Job:
    command = _build_command(kind, args)
    log_dir = RL_ROOT / "dashboard" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex[:10]
    log_path = log_dir / f"{int(time.time())}_{kind}_{job_id}.log"
    handle = log_path.open("a", encoding="utf-8", errors="replace")
    handle.write("$ " + " ".join(command) + "\n\n")
    handle.flush()
    env = os.environ.copy()
    py_path = str(ROOT / "py")
    env["PYTHONPATH"] = py_path + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else py_path
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    job = Job(
        id=job_id,
        kind=kind,
        name=str(args.get("name") or kind),
        command=command,
        started_at=time.time(),
        log_path=str(log_path),
        pid=process.pid,
    )
    with LOCK:
        JOBS[job_id] = job
        PROCESSES[job_id] = process
    threading.Thread(target=_watch_job, args=(job_id, process), daemon=True).start()
    return job


def _stop_job(job_id: str) -> bool:
    with LOCK:
        process = PROCESSES.get(job_id)
        job = JOBS.get(job_id)
        if job:
            job.stopped = True
    if process is None:
        return False
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        process.send_signal(signal.SIGTERM)
    return True


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[dashboard] " + fmt % args + "\n")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            _text_response(self, INDEX_HTML)
            return
        if parsed.path == "/api/state":
            with LOCK:
                jobs = [asdict(job) | {"running": job.id in PROCESSES} for job in JOBS.values()]
            payload = _artifact_summary()
            payload["jobs"] = sorted(jobs, key=lambda row: row["started_at"], reverse=True)
            _json_response(self, payload)
            return
        if parsed.path == "/api/export.xlsx":
            filename = f"tribes_rl_analytics_{time.strftime('%Y%m%d_%H%M%S')}.xlsx"
            _binary_response(
                self,
                _build_export_workbook(),
                filename,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            return
        if parsed.path == "/api/log":
            params = parse_qs(parsed.query)
            job_id = (params.get("job") or [""])[0]
            with LOCK:
                job = JOBS.get(job_id)
            if not job:
                _json_response(self, {"lines": []}, 404)
                return
            _json_response(self, {"job": asdict(job), "lines": _tail(Path(job.log_path), 500)})
            return
        _json_response(self, {"error": "not found"}, 404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/jobs":
            body = _read_body(self)
            try:
                job = _start_job(str(body.get("kind")), dict(body.get("args") or {}))
            except Exception as exc:
                _json_response(self, {"error": str(exc)}, 400)
                return
            _json_response(self, {"job": asdict(job)}, 201)
            return
        if parsed.path == "/api/stop":
            body = _read_body(self)
            ok = _stop_job(str(body.get("job_id")))
            _json_response(self, {"stopped": ok})
            return
        _json_response(self, {"error": "not found"}, 404)


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Tribes RL Dashboard</title>
  <style>
    :root { color-scheme: light; --bg:#f3f5f1; --panel:#ffffff; --panel2:#f8faf6; --ink:#17201f; --muted:#65716e; --line:#dce4dd; --accent:#0f766e; --blue:#2563eb; --red:#b42318; --gold:#b7791f; --violet:#7c3aed; --shadow:0 10px 28px rgba(28,45,39,.06); }
    * { box-sizing:border-box; }
    body { margin:0; font:14px/1.45 Inter, Segoe UI, system-ui, sans-serif; color:var(--ink); background:#f3f5f1; }
    header { height:68px; display:flex; align-items:center; justify-content:space-between; padding:0 24px; border-bottom:1px solid var(--line); background:rgba(255,255,255,.88); backdrop-filter: blur(10px); position:sticky; top:0; z-index:3; }
    h1 { font-size:20px; margin:0; letter-spacing:0; }
    main { display:grid; grid-template-columns: 350px minmax(0,1fr); min-height:calc(100vh - 68px); }
    aside { border-right:1px solid var(--line); padding:18px 20px; background:#f9fbf7; }
    section { padding:14px 16px 24px; min-width:0; width:100%; }
    h2 { font-size:15px; margin:0 0 12px; }
    h3 { font-size:13px; margin:18px 0 8px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
    label { display:block; color:var(--muted); font-size:12px; margin:10px 0 4px; }
    input, select { width:100%; height:36px; border:1px solid var(--line); border-radius:7px; background:#fff; color:var(--ink); padding:0 10px; outline:none; }
    input:focus, select:focus { border-color:#7bb7af; box-shadow:0 0 0 3px rgba(15,118,110,.12); }
    button { height:36px; border:1px solid var(--line); border-radius:7px; background:#fff; color:var(--ink); padding:0 12px; cursor:pointer; font-weight:650; }
    button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
    button.danger { color:var(--red); }
    button.ghost { background:transparent; }
    .grid { display:grid; gap:10px; }
    .metrics { grid-template-columns: repeat(5, minmax(120px,1fr)); }
    .metric, .panel { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; box-shadow:var(--shadow); }
    .metric span { color:var(--muted); font-size:12px; display:block; }
    .metric strong { font-size:21px; display:block; margin-top:2px; }
    .metric canvas { width:100%; height:18px; margin-top:5px; }
    .tabs { display:flex; gap:8px; margin:12px 0 10px; overflow:auto; }
    .tabs button.active { background:#dff1ee; border-color:#a8d2cb; color:#075e57; }
    .row { display:flex; align-items:center; justify-content:space-between; gap:10px; }
    .stack { display:grid; gap:10px; align-content:start; }
    .list { display:grid; gap:8px; }
    .job { border:1px solid var(--line); border-radius:8px; padding:10px; background:#fff; }
    .status { display:inline-flex; align-items:center; gap:6px; font-size:12px; color:var(--muted); }
    .dot { width:8px; height:8px; border-radius:99px; background:var(--muted); }
    .running .dot { background:var(--accent); }
    .failed .dot { background:var(--red); }
    table { width:100%; border-collapse:collapse; font-size:12px; }
    th, td { text-align:left; border-bottom:1px solid var(--line); padding:7px 8px; white-space:nowrap; }
    th { color:var(--muted); font-weight:700; background:#fafbf8; position:sticky; top:0; }
    .table-wrap { overflow:auto; max-height:360px; border:1px solid var(--line); border-radius:8px; background:#fff; }
    .chart { width:100%; height:190px; min-height:190px; border:1px solid var(--line); border-radius:8px; background:#fff; }
    .mini-chart { width:100%; height:118px; min-height:118px; border:1px solid var(--line); border-radius:8px; background:#fff; }
    .hero { display:grid; grid-template-columns:minmax(0,1.12fr) minmax(300px,.88fr); gap:10px; margin-bottom:10px; align-items:start; }
    #analytics .hero { grid-template-columns:minmax(0,1fr) minmax(280px,.8fr); }
    #analytics .panel h2 { margin-bottom:8px; }
    .overview-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; margin-bottom:10px; align-items:start; }
    .viz-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; }
    .pill-row { display:flex; flex-wrap:wrap; gap:8px; }
    .pill { display:inline-flex; align-items:center; gap:6px; min-height:28px; padding:4px 9px; border:1px solid var(--line); border-radius:999px; background:var(--panel2); font-size:12px; color:var(--muted); }
    .bar-list { display:grid; gap:8px; margin-top:10px; }
    .bar-row { display:grid; grid-template-columns:minmax(118px,1.35fr) minmax(92px,1.6fr) 52px; align-items:center; gap:9px; font-size:12px; }
    .bar-track { height:9px; border-radius:99px; background:#e9eee9; overflow:hidden; }
    .bar-fill { height:100%; width:0; border-radius:99px; background:var(--accent); }
    .summary-card { border:1px solid var(--line); background:var(--panel2); border-radius:8px; padding:12px; }
    .summary-card strong { display:block; font-size:20px; margin-top:2px; }
    pre { margin:0; max-height:340px; overflow:auto; background:#172020; color:#d6ebe6; border-radius:8px; padding:12px; font:12px/1.45 Consolas, monospace; }
    .muted { color:var(--muted); }
    .split { display:grid; grid-template-columns: 1fr 1fr; gap:14px; }
    @media (max-width: 1180px) { .hero, .overview-grid, .viz-grid { grid-template-columns:1fr; } }
    @media (max-width: 980px) { main { grid-template-columns:1fr; } aside { border-right:0; border-bottom:1px solid var(--line); } .metrics, .split { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>Tribes RL Dashboard</h1>
    <div class="row">
      <button id="exportWorkbook">Export Excel</button>
      <div class="status"><span class="dot"></span><span id="updated">Loading</span></div>
    </div>
  </header>
  <main>
    <aside>
      <h2>Run Control</h2>
      <label>Job type</label>
      <select id="kind">
        <option value="train">Training</option>
        <option value="analytics">Analytics benchmark</option>
        <option value="augmentation">Augmentation benchmark</option>
        <option value="tournament">Checkpoint tournament</option>
        <option value="java_headless">Java headless game</option>
        <option value="tensorboard">TensorBoard</option>
      </select>
      <div id="fields"></div>
      <button class="primary" id="start">Start Job</button>
      <h3>Jobs</h3>
      <div class="list" id="jobs"></div>
    </aside>
    <section>
      <div class="grid metrics">
        <div class="metric"><span>Replay shards</span><strong id="replayShards">0</strong><canvas id="replaySpark"></canvas></div>
        <div class="metric"><span>Replay size</span><strong id="replaySize">0 MB</strong><canvas id="stepSpark"></canvas></div>
        <div class="metric"><span>Checkpoints</span><strong id="checkpointCount">0</strong><div class="muted" id="checkpointMeta">No checkpoints</div></div>
        <div class="metric"><span>Latest iteration</span><strong id="latestIteration">-</strong><div class="muted" id="latestLoss">loss -</div></div>
        <div class="metric"><span>Active jobs</span><strong id="activeJobs">0</strong><div class="muted" id="tensorboardMeta">TensorBoard idle</div></div>
      </div>
      <div class="tabs" style="margin-top:16px">
        <button data-tab="overview" class="active">Overview</button>
        <button data-tab="analytics">Analytics</button>
        <button data-tab="tournaments">Tournaments</button>
        <button data-tab="logs">Logs</button>
      </div>
      <div id="overview" class="tab">
        <div class="hero">
          <div class="panel"><div class="row"><h2>Training trajectory</h2><div class="pill-row" id="lossLegend"></div></div><canvas id="lossChart" class="chart"></canvas></div>
          <div class="panel">
            <div class="row"><h2>Game performance distribution</h2><div class="muted" id="gamePerfNote">Per-game points from training logs</div></div>
            <canvas id="gamePerfChart" class="chart"></canvas>
          </div>
        </div>
        <div class="overview-grid">
          <div class="panel">
            <h2>Self-play outcomes</h2>
            <canvas id="outcomeChart" class="mini-chart"></canvas>
          </div>
          <div class="panel">
            <h2>Performance spread</h2><div class="muted" id="gamePerfBarsNote">Per-game percentile bands from training logs.</div><div class="bar-list" id="gamePerfBars"></div>
          </div>
          <div class="panel">
            <h2 id="actionPanelTitle">Recent action mix</h2><div class="muted" id="actionPanelNote"></div><div class="bar-list" id="actionBars"></div>
          </div>
        </div>
        <h3>Training metrics</h3><div class="table-wrap"><table id="metricsTable"></table></div>
      </div>
      <div id="analytics" class="tab" hidden>
        <div class="hero">
          <div class="panel"><div class="row"><h2>Benchmark speed map</h2><div class="muted" id="analyticsFile">No file yet</div></div><canvas id="analyticsChart" class="chart"></canvas></div>
          <div class="stack">
            <div class="panel"><h2>Bottlenecks</h2><div class="pill-row" id="bottleneckPills"></div></div>
            <div class="panel"><h2>Best runs</h2><div class="bar-list" id="analyticsBars"></div></div>
          </div>
        </div>
        <div class="hero">
          <div class="panel"><div class="row"><h2>Symmetry consistency</h2><div class="muted" id="augmentationFile">No file yet</div></div><canvas id="symmetryChart" class="chart"></canvas></div>
          <div class="stack">
            <div class="panel"><h2>Consistency summary</h2><div class="bar-list" id="symmetryBars"></div></div>
            <div class="panel"><h2>Augmentation loss gap</h2><canvas id="symmetryGapChart" class="mini-chart"></canvas></div>
          </div>
        </div>
        <h3>Runs</h3><div class="table-wrap"><table id="analyticsTable"></table></div>
        <h3>Augmentation runs</h3><div class="table-wrap"><table id="augmentationTable"></table></div>
      </div>
      <div id="tournaments" class="tab" hidden>
        <div class="hero">
          <div class="panel"><div class="row"><h2>Checkpoint Elo ladder</h2><div class="muted" id="tournamentFile">No file yet</div></div><canvas id="eloChart" class="chart"></canvas></div>
          <div class="panel"><h2>Standout checkpoints</h2><div class="bar-list" id="eloBars"></div></div>
        </div>
        <h3>Standings</h3><div class="table-wrap"><table id="tournamentTable"></table></div>
      </div>
      <div id="logs" class="tab" hidden>
        <div class="panel"><h2 id="logTitle">Select a job</h2><pre id="logBox"></pre></div>
      </div>
    </section>
  </main>
<script>
const defaults = {
  train: [["iterations", "1"], ["games_per_iteration", "2"], ["simulations", "64"], ["search_batch_size", "64"], ["max_turns_capitals", "40"], ["profile_selfplay", true]],
  analytics: [["games", "1"], ["simulations", "32 64"], ["search_batch_sizes", "16 32"], ["top_k_actions", "64"], ["max_turns_capitals", "20"]],
  augmentation: [["training_iterations", "1"], ["training_batch_size", "64"], ["replay_batch_size", "256"], ["epochs_per_iteration", "1"]],
  tournament: [["limit", "0"], ["games_per_pair", "1"], ["latest_vs_history", true], ["history_step", "5"], ["simulations", "64"], ["search_batch_size", "32"], ["deterministic", true]],
  java_headless: [["config", "play.json"]],
  tensorboard: [["logdir", "rl"], ["port", "6006"]]
};
let state = null, selectedJob = null;
const $ = id => document.getElementById(id);
function prettyBytes(n){ if(!n) return "0 MB"; return (n/1048576).toFixed(n > 10485760 ? 1 : 2) + " MB"; }
function num(v, fallback=0){ const n = Number(v); return Number.isFinite(n) ? n : fallback; }
function fmt(v, digits=2){ const n = num(v, NaN); return Number.isFinite(n) ? n.toFixed(digits).replace(/\.?0+$/,"") : "-"; }
function esc(v){ return String(v ?? "").replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function fieldInput([name, value]){
  const id = "f_" + name;
  if(typeof value === "boolean") return `<label><input id="${id}" type="checkbox" ${value ? "checked" : ""} style="width:auto;height:auto"> ${name.replaceAll("_"," ")}</label>`;
  return `<label>${name.replaceAll("_"," ")}</label><input id="${id}" value="${value}">`;
}
function renderFields(){ $("fields").innerHTML = defaults[$("kind").value].map(fieldInput).join(""); }
function collectArgs(){
  const args = {};
  for(const [name, value] of defaults[$("kind").value]){
    const el = $("f_" + name);
    if(typeof value === "boolean") args[name] = el.checked;
    else if(["simulations","search_batch_sizes","top_k_actions","max_turns_capitals","external_action_timeout_ms"].includes(name) && el.value.trim().includes(" ")) args[name] = el.value.trim().split(/\s+/);
    else args[name] = el.value.trim();
  }
  return args;
}
async function startJob(){
  const res = await fetch("/api/jobs", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({kind:$("kind").value, args:collectArgs()})});
  if(!res.ok) alert((await res.json()).error || "Failed to start");
  await refresh();
}
async function stopJob(id){ await fetch("/api/stop", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({job_id:id})}); await refresh(); }
function renderTable(id, rows, columns){
  const table = $(id);
  if(!rows || !rows.length){ table.innerHTML = "<tbody><tr><td class='muted'>No rows yet</td></tr></tbody>"; return; }
  const cols = columns || Object.keys(rows[rows.length - 1]).slice(0, 12);
  table.innerHTML = `<thead><tr>${cols.map(c=>`<th>${c}</th>`).join("")}</tr></thead><tbody>` +
    rows.slice().reverse().map(r=>`<tr>${cols.map(c=>`<td>${esc(r[c] ?? "")}</td>`).join("")}</tr>`).join("") + "</tbody>";
}
function setupCanvas(id){
  const canvas = $(id), dpr = devicePixelRatio || 1, rect = canvas.getBoundingClientRect();
  const cssW = Math.max(220, rect.width || canvas.parentElement?.clientWidth || 320);
  const cssH = Math.max(80, rect.height || Number.parseFloat(getComputedStyle(canvas).height) || 220);
  canvas.width = cssW * dpr; canvas.height = cssH * dpr;
  const ctx = canvas.getContext("2d"); ctx.scale(dpr,dpr); ctx.clearRect(0,0,cssW,cssH); return {canvas, ctx, w:cssW, h:cssH};
}
function drawGrid(ctx, w, h, opts={}){
  const left = opts.left ?? 68, right = opts.right ?? 16, top = opts.top ?? 18, bottom = opts.bottom ?? 42;
  const min = opts.min ?? 0, max = opts.max ?? 1, yUnit = opts.yUnit || "", xLabel = opts.xLabel || "";
  ctx.strokeStyle = "#e4ebe4"; ctx.lineWidth = 1; ctx.fillStyle = "#65716e"; ctx.font = "11px Segoe UI";
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for(let i=0;i<4;i++){
    const ratio = i / 3;
    const y = top + ratio*(h-top-bottom);
    const value = max - ratio*(max-min);
    ctx.beginPath(); ctx.moveTo(left,y); ctx.lineTo(w-right,y); ctx.stroke();
    ctx.fillText(formatAxis(value, yUnit), left - 7, y);
  }
  ctx.strokeStyle = "#cfd9d0"; ctx.beginPath(); ctx.moveTo(left,top); ctx.lineTo(left,h-bottom); ctx.lineTo(w-right,h-bottom); ctx.stroke();
  ctx.textAlign = "left"; ctx.textBaseline = "alphabetic"; ctx.fillText(xLabel, left, h - 12);
  return {left, right, top, bottom, plotW:w-left-right, plotH:h-top-bottom};
}
function formatAxis(value, unit){
  if(!Number.isFinite(value)) return "-";
  const abs = Math.abs(value);
  const body = abs >= 1000 ? (value/1000).toFixed(1).replace(/\.0$/,"") + "k" : abs >= 100 ? value.toFixed(0) : abs >= 10 ? value.toFixed(1).replace(/\.0$/,"") : value.toFixed(2).replace(/\.?0+$/,"");
  return `${body}${unit}`;
}
function drawChart(id, rows, yKeys, colors, opts={}){
  const {ctx,w,h} = setupCanvas(id);
  if(!rows || rows.length < 2) return;
  const vals = rows.flatMap(r => yKeys.map(k => num(r[k], NaN))).filter(Number.isFinite);
  const max = Math.max(...vals, 1), min = Math.min(...vals, 0);
  const pad = (max - min) * .08 || 1;
  const domainMin = opts.zeroBase ? 0 : Math.max(0, min - pad);
  const domainMax = max + pad;
  const area = drawGrid(ctx,w,h,{min:domainMin,max:domainMax,yUnit:opts.yUnit || "",xLabel:opts.xLabel || "iteration"});
  ctx.fillStyle = "#65716e"; ctx.font = "11px Segoe UI"; ctx.textAlign = "right"; ctx.fillText(`n=${rows.length}`, w - 14, h - 12);
  yKeys.forEach((key, i) => {
    ctx.strokeStyle = colors[i]; ctx.lineWidth = 2.5; ctx.beginPath();
    rows.forEach((r, idx) => {
      const x = area.left + idx * (area.plotW / Math.max(1, rows.length-1));
      const y = area.top + (1 - ((num(r[key],0)-domainMin) / (domainMax-domainMin || 1))) * area.plotH;
      if(idx === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    });
    ctx.stroke();
  });
}
function drawGroupedLines(id, rows, yKeys, colors, opts={}){
  const {ctx,w,h} = setupCanvas(id);
  if(!rows || !rows.length){
    drawGrid(ctx,w,h,{min:0,max:1,yUnit:opts.yUnit || "",xLabel:opts.xLabel || ""});
    ctx.fillStyle = "#65716e"; ctx.font = "13px Segoe UI"; ctx.textAlign = "center"; ctx.fillText(opts.empty || "No data yet", w/2, h/2);
    return;
  }
  const vals = rows.flatMap(r => yKeys.map(k => num(r[k], NaN))).filter(Number.isFinite);
  const max = Math.max(...vals, 1e-6), min = Math.min(...vals, 0);
  const pad = (max - min) * .08 || Math.max(max * .08, 1e-6);
  const domainMin = opts.zeroBase === false ? min - pad : Math.min(0, min - pad);
  const domainMax = max + pad;
  const area = drawGrid(ctx,w,h,{min:domainMin,max:domainMax,yUnit:opts.yUnit || "",xLabel:opts.xLabel || ""});
  const groups = [...new Set(rows.map(r => String(r.variant || "run")))];
  yKeys.forEach((key, i) => {
    groups.forEach((variant, groupIndex) => {
      const groupRows = rows.map((r, idx) => ({...r, _idx:idx})).filter(r => String(r.variant || "run") === variant && Number.isFinite(num(r[key], NaN)));
      if(!groupRows.length) return;
      ctx.strokeStyle = colors[(i + groupIndex) % colors.length];
      ctx.lineWidth = 2.3;
      ctx.setLineDash(groupIndex % 2 ? [5,4] : []);
      ctx.beginPath();
      groupRows.forEach((r, pointIndex) => {
        const x = area.left + r._idx * (area.plotW / Math.max(1, rows.length-1));
        const y = area.top + (1 - ((num(r[key],0)-domainMin) / (domainMax-domainMin || 1))) * area.plotH;
        if(pointIndex === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
      });
      ctx.stroke();
      ctx.setLineDash([]);
      groupRows.forEach(r => {
        const x = area.left + r._idx * (area.plotW / Math.max(1, rows.length-1));
        const y = area.top + (1 - ((num(r[key],0)-domainMin) / (domainMax-domainMin || 1))) * area.plotH;
        ctx.fillStyle = colors[(i + groupIndex) % colors.length];
        ctx.beginPath(); ctx.arc(x,y,3,0,Math.PI*2); ctx.fill();
      });
    });
  });
  ctx.fillStyle = "#65716e"; ctx.font = "11px Segoe UI"; ctx.textAlign = "right"; ctx.fillText(`n=${rows.length}`, w - 14, h - 12);
}
function drawSpark(id, rows, key, color){
  const {ctx,w,h} = setupCanvas(id);
  const values = rows.map(r => num(r[key], NaN)).filter(Number.isFinite);
  if(values.length < 2) return;
  const max = Math.max(...values), min = Math.min(...values);
  ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
  values.forEach((v, i) => {
    const x = 2 + i*((w-4)/Math.max(1, values.length-1));
    const y = h - 4 - ((v-min)/(max-min || 1))*(h-8);
    if(i === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  });
  ctx.stroke();
}
function renderLegend(id, items){ $(id).innerHTML = items.map(([name,color]) => `<span class="pill"><span class="dot" style="background:${color}"></span>${esc(name)}</span>`).join(""); }
function renderBars(id, rows, labelKey, valueKey, color, limit=6){
  const vals = rows.map(r => ({ label:String(r[labelKey] ?? "-"), value:num(r[valueKey], 0) })).filter(r => Number.isFinite(r.value)).sort((a,b)=>b.value-a.value).slice(0,limit);
  const max = Math.max(...vals.map(r=>r.value), 1);
  $(id).innerHTML = vals.length ? vals.map(r => `<div class="bar-row"><span title="${esc(r.label)}">${esc(r.label.slice(0,22))}</span><span class="bar-track"><span class="bar-fill" style="width:${Math.max(2,(r.value/max)*100)}%;background:${color}"></span></span><strong>${fmt(r.value,1)}</strong></div>`).join("") : "<div class='muted'>No data yet</div>";
}
function renderActionBars(rows, gameRows){
  const exactGames = (gameRows || []).filter(r => String(r.exact_summary || "") === "1");
  const last = (exactGames.length ? exactGames[exactGames.length-1] : rows[rows.length-1]) || {};
  const keys = Object.keys(last).filter(k => k.startsWith("action_") && !["action_mask","action_policy"].includes(k));
  const actionRows = keys.map(k => ({ name:k.replace("action_",""), value:num(last[k],0) })).filter(r => r.value > 0);
  if(actionRows.length) {
    $("actionPanelTitle").textContent = "Recent action mix";
    $("actionPanelNote").textContent = exactGames.length ? "Counts from the latest completed self-play game." : "Counts from the latest training metrics row.";
    renderBars("actionBars", actionRows, "name", "value", "#0f766e", 8);
  } else {
    $("actionPanelTitle").textContent = "Training time split";
    $("actionPanelNote").textContent = exactGames.length ? "No action-count columns found for the latest game; showing seconds from the latest update." : "No action-count columns found in metrics.csv; showing seconds from the latest update.";
    renderBars("actionBars", [
    {name:"sample", value:num(last.sample_sec,0)},
    {name:"fetch", value:num(last.fetch_sec,0)},
    {name:"optimize", value:num(last.optimize_sec,0)}
    ], "name", "value", "#0f766e", 3);
  }
}
function drawScatter(id, rows){
  const {ctx,w,h} = setupCanvas(id);
  if(!rows || !rows.length) return;
  const xKey = "action_total_ms_mean", yKey = "configured_simulations_sec";
  const xs = rows.map(r=>num(r[xKey],NaN)).filter(Number.isFinite), ys = rows.map(r=>num(r[yKey],NaN)).filter(Number.isFinite);
  const xMax = Math.max(...xs, 1), yMax = Math.max(...ys, 1);
  const area = drawGrid(ctx,w,h,{min:0,max:yMax,yUnit:"/s",xLabel:"mean action latency (ms)"});
  ctx.fillStyle = "#65716e"; ctx.font = "11px Segoe UI"; ctx.textAlign = "right"; ctx.fillText(`max ${formatAxis(xMax,"ms")}`, w - 14, h - 12);
  rows.forEach(r => {
    const x = area.left + (num(r[xKey],0)/xMax)*area.plotW;
    const y = area.top + (1 - (num(r[yKey],0)/yMax))*area.plotH;
    const radius = Math.max(4, Math.min(14, num(r.replay_steps_sec,1)));
    ctx.fillStyle = num(r.simulations,0) >= 64 ? "#2563eb" : "#0f766e";
    ctx.globalAlpha = .78; ctx.beginPath(); ctx.arc(x,y,radius,0,Math.PI*2); ctx.fill(); ctx.globalAlpha = 1;
  });
  ctx.fillStyle = "#65716e"; ctx.font = "11px Segoe UI"; ctx.textAlign = "left"; ctx.fillText("y: configured simulations/sec", area.left, 14);
}
function drawGamePerformance(id, gameRows, metricRows){
  const {ctx,w,h} = setupCanvas(id);
  let rows = (gameRows || []).filter(r => num(r.actions_sec, NaN) > 0 && num(r.configured_sims_sec, NaN) > 0);
  let label = "points are games";
  if(!rows.length){
    rows = (metricRows || []).filter(r => num(r.action_decisions_sec, NaN) > 0 && num(r.configured_sims_sec, NaN) > 0)
      .map(r => ({ iteration:r.iteration, actions_sec:r.action_decisions_sec, configured_sims_sec:r.configured_sims_sec }));
    label = "fallback: points are iterations";
  }
  $("gamePerfNote").textContent = label;
  if(!rows.length){
    drawGrid(ctx,w,h,{min:0,max:1,yUnit:" sims/s",xLabel:"actions/sec"});
    ctx.fillStyle = "#65716e"; ctx.font = "13px Segoe UI"; ctx.textAlign = "center"; ctx.fillText("No speed data yet", w/2, h/2);
    return;
  }
  const xMax = Math.max(...rows.map(r => num(r.actions_sec, 0)), 1);
  const yMax = Math.max(...rows.map(r => num(r.configured_sims_sec, 0)), 1);
  const area = drawGrid(ctx,w,h,{min:0,max:yMax,yUnit:" sims/s",xLabel:"actions/sec"});
  ctx.fillStyle = "#65716e"; ctx.font = "11px Segoe UI"; ctx.textAlign = "right"; ctx.fillText(`max ${formatAxis(xMax," actions/s")}`, w - 14, h - 12);
  rows.forEach((r, i) => {
    const x = area.left + (num(r.actions_sec,0)/xMax)*area.plotW;
    const y = area.top + (1 - num(r.configured_sims_sec,0)/yMax)*area.plotH;
    ctx.fillStyle = i >= rows.length - 8 ? "#0f766e" : "#2563eb";
    ctx.globalAlpha = i >= rows.length - 8 ? .86 : .45;
    ctx.beginPath(); ctx.arc(x,y,5.5,0,Math.PI*2); ctx.fill(); ctx.globalAlpha = 1;
  });
  ctx.fillStyle = "#65716e"; ctx.font = "11px Segoe UI"; ctx.textAlign = "left"; ctx.fillText("y: configured simulations/sec", area.left, 14);
}
function renderSymmetry(rows){
  const eventRows = rows || [];
  $("augmentationFile").textContent = state.augmentation_jsonl ? `${state.augmentation_jsonl.path} · ${eventRows.length} runs` : "No file yet";
  const plotRows = eventRows.map((r, index) => ({...r, run:index + 1}));
  drawGroupedLines("symmetryChart", plotRows, ["policy_js","policy_l1"], ["#0f766e","#2563eb","#7c3aed"], { yUnit:"", xLabel:"augmentation benchmark run", empty:"Run an augmentation benchmark to populate consistency metrics." });
  drawGroupedLines("symmetryGapChart", plotRows, ["d4_eval_gap"], ["#b42318","#7c3aed"], { yUnit:" loss", xLabel:"run", empty:"No D4 eval loss gap yet." });
  const latestByVariant = {};
  plotRows.forEach(r => { latestByVariant[String(r.variant || "run")] = r; });
  const summary = Object.entries(latestByVariant).flatMap(([variant, r]) => [
    {name:`${variant} JS`, value:num(r.policy_js,0)},
    {name:`${variant} L1`, value:num(r.policy_l1,0)}
  ]);
  renderBars("symmetryBars", summary, "name", "value", "#7c3aed", 9);
  renderTable("augmentationTable", eventRows, ["created_at_utc","variant","train_sec","eval_loss","d4_eval_loss","d4_eval_gap","policy_js","policy_l1"]);
}
function percentile(values, p){
  const clean = values.map(v => num(v, NaN)).filter(Number.isFinite).sort((a,b)=>a-b);
  if(!clean.length) return 0;
  const idx = (clean.length - 1) * p, lo = Math.floor(idx), hi = Math.ceil(idx);
  return lo === hi ? clean[lo] : clean[lo] + (clean[hi] - clean[lo]) * (idx - lo);
}
function renderGamePerfBars(gameRows, metricRows){
  let rows = (gameRows || []).filter(r => num(r.actions_sec, NaN) > 0 && num(r.configured_sims_sec, NaN) > 0);
  let source = "games";
  if(!rows.length){
    rows = (metricRows || []).filter(r => num(r.action_decisions_sec, NaN) > 0 && num(r.configured_sims_sec, NaN) > 0)
      .map(r => ({ actions_sec:r.action_decisions_sec, configured_sims_sec:r.configured_sims_sec }));
    source = "iterations";
  }
  $("gamePerfBarsNote").textContent = source === "games" ? "Per-game percentile bands from training logs." : "Iteration fallback; no per-game speed logs found.";
  if(!rows.length){ $("gamePerfBars").innerHTML = "<div class='muted'>No speed data yet.</div>"; return; }
  const actionVals = rows.map(r => num(r.actions_sec, 0));
  const simVals = rows.map(r => num(r.configured_sims_sec, 0));
  const summary = [
    {name:"p90 actions/s", value:percentile(actionVals,.9)},
    {name:`median actions/s`, value:percentile(actionVals,.5)},
    {name:"p10 actions/s", value:percentile(actionVals,.1)},
    {name:"p90 sims/s", value:percentile(simVals,.9)},
    {name:`median sims/s`, value:percentile(simVals,.5)},
    {name:"p10 sims/s", value:percentile(simVals,.1)}
  ];
  renderBars("gamePerfBars", summary, "name", "value", "#2563eb", 6);
}
function drawElo(id, rows){
  const {ctx,w,h} = setupCanvas(id);
  const vals = rows.map(r => ({label:r.name || r.iteration || "-", elo:num(r.elo, NaN)})).filter(r => Number.isFinite(r.elo)).sort((a,b)=>b.elo-a.elo).slice(0,10);
  if(!vals.length) return;
  const max = Math.max(...vals.map(r=>r.elo)), min = Math.min(...vals.map(r=>r.elo), 1400);
  const area = drawGrid(ctx,w,h,{min:min,max:max,yUnit:" Elo",xLabel:"checkpoint rank"});
  const barW = area.plotW/vals.length;
  vals.forEach((r,i) => {
    const bh = ((r.elo-min)/(max-min || 1))*area.plotH + 6;
    const x = area.left + i*barW, y = area.top + area.plotH - bh;
    ctx.fillStyle = i === 0 ? "#0f766e" : "#2563eb";
    ctx.fillRect(x, y, Math.max(12, barW-8), bh);
  });
}
function renderJobs(jobs){
  $("jobs").innerHTML = jobs.map(j => {
    const cls = j.running ? "running" : (j.returncode && j.returncode !== 0 ? "failed" : "");
    const status = j.running ? "running" : `exit ${j.returncode ?? "-"}`;
    const tb = j.kind === "tensorboard" && j.running ? `<button onclick="window.open('http://127.0.0.1:6006','_blank')">Open</button>` : "";
    return `<div class="job ${cls}"><div class="row"><strong>${esc(j.kind)}</strong><span class="status"><span class="dot"></span>${esc(status)}</span></div>
      <div class="muted">pid ${esc(j.pid ?? "-")} · ${new Date(j.started_at*1000).toLocaleTimeString()}</div>
      <div class="row" style="margin-top:8px"><button onclick="selectLog('${j.id}')">View log</button><span>${tb}${j.running ? `<button class="danger" onclick="stopJob('${j.id}')">Stop</button>` : ""}</span></div></div>`;
  }).join("") || "<div class='muted'>No jobs started from this dashboard yet.</div>";
}
async function selectLog(id){ selectedJob = id; document.querySelector('[data-tab="logs"]').click(); await refreshLog(); }
async function refreshLog(){
  if(!selectedJob) return;
  const payload = await (await fetch("/api/log?job=" + encodeURIComponent(selectedJob))).json();
  $("logTitle").textContent = `${payload.job.kind} log`;
  $("logBox").textContent = payload.lines.join("\n");
  $("logBox").scrollTop = $("logBox").scrollHeight;
}
async function refresh(){
  state = await (await fetch("/api/state")).json();
  $("updated").textContent = "Updated " + new Date(state.updated_at*1000).toLocaleTimeString();
  $("replayShards").textContent = state.replay_shards;
  $("replaySize").textContent = prettyBytes(state.replay_bytes);
  $("checkpointCount").textContent = state.checkpoints.length;
  $("activeJobs").textContent = state.jobs.filter(j=>j.running).length;
  const rows = state.metrics_rows || [];
  const last = rows[rows.length-1] || {};
  $("latestIteration").textContent = rows.length ? rows[rows.length-1].iteration || "-" : "-";
  $("latestLoss").textContent = "loss " + fmt(last.loss, 3);
  $("checkpointMeta").textContent = state.checkpoints[0] ? state.checkpoints[0].path : "No checkpoints";
  const tbRunning = state.jobs.some(j => j.kind === "tensorboard" && j.running);
  $("tensorboardMeta").textContent = tbRunning ? "TensorBoard live on :6006" : `${state.tensorboard_events.length} event files`;
  renderJobs(state.jobs);
  renderTable("metricsTable", rows, ["iteration","replay_steps","games_completed","action_decisions_sec","configured_sims_sec","wins","losses","draws","loss","policy_loss","value_loss"]);
  renderLegend("lossLegend", [["loss","#0f766e"],["policy","#2563eb"],["value","#b7791f"]]);
  drawChart("lossChart", rows, ["loss","policy_loss","value_loss"], ["#0f766e","#2563eb","#b7791f"], { yUnit:" loss", xLabel:"training iteration" });
  const gameRows = state.training_game_rows || [];
  const outcomeRows = gameRows.filter(r => Number.isFinite(num(r.wins, NaN)) || Number.isFinite(num(r.losses, NaN)) || Number.isFinite(num(r.draws, NaN)));
  drawChart("outcomeChart", outcomeRows.length ? outcomeRows : rows, ["wins","losses","draws"], ["#0f766e","#b42318","#63706d"], { yUnit:" games", xLabel:outcomeRows.length ? "self-play game" : "training iteration", zeroBase:true });
  drawGamePerformance("gamePerfChart", gameRows, rows);
  renderGamePerfBars(gameRows, rows);
  drawSpark("replaySpark", rows, "replay_steps", "#0f766e");
  drawSpark("stepSpark", rows, "steps", "#2563eb");
  renderActionBars(rows, gameRows);
  $("analyticsFile").textContent = state.analytics_csv ? `${state.analytics_csv.path} · ${state.analytics_rows.length} rows` : "No file yet";
  drawScatter("analyticsChart", state.analytics_rows || []);
  const bottlenecks = {};
  (state.analytics_rows || []).forEach(r => { const key = String(r.bottleneck_hint || "unknown").split(";")[0]; bottlenecks[key] = (bottlenecks[key] || 0) + 1; });
  $("bottleneckPills").innerHTML = Object.entries(bottlenecks).sort((a,b)=>b[1]-a[1]).map(([k,v]) => `<span class="pill"><span class="dot" style="background:#b7791f"></span>${esc(k)} · ${v}</span>`).join("") || "<span class='muted'>No benchmark hints yet</span>";
  renderBars("analyticsBars", state.analytics_rows || [], "run_index", "configured_simulations_sec", "#2563eb", 6);
  renderTable("analyticsTable", state.analytics_rows, ["run_index","simulations","search_batch_size","top_k_actions","elapsed_sec","games_completed","replay_steps_sec","action_total_ms_mean","action_search_ms_mean","configured_simulations_sec","bottleneck_hint"]);
  renderSymmetry(state.augmentation_event_rows || []);
  $("tournamentFile").textContent = state.tournament_csv ? `${state.tournament_csv.path} · ${state.tournament_rows.length} rows` : "No file yet";
  drawElo("eloChart", state.tournament_rows || []);
  renderBars("eloBars", state.tournament_rows || [], "name", "elo", "#0f766e", 8);
  renderTable("tournamentTable", state.tournament_rows, ["name","iteration","elo","games","score","score_rate","checkpoint"]);
  await refreshLog();
}
document.querySelectorAll(".tabs button").forEach(btn => btn.addEventListener("click", () => {
  document.querySelectorAll(".tabs button").forEach(b=>b.classList.remove("active")); btn.classList.add("active");
  document.querySelectorAll(".tab").forEach(t=>t.hidden = t.id !== btn.dataset.tab);
  refresh();
}));
$("kind").addEventListener("change", renderFields);
$("start").addEventListener("click", startJob);
$("exportWorkbook").addEventListener("click", () => { window.location.href = "/api/export.xlsx"; });
renderFields(); refresh(); setInterval(refresh, 2500);
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local live dashboard for Tribes RL training, selfplay, benchmarks, and tournaments.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"[dashboard] open http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
