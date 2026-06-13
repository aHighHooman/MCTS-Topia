from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def write_report(path: Path, title: str, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys()) if rows else []
    body = [
        "<!doctype html><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        "<style>body{font-family:system-ui,Segoe UI,sans-serif;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:4px 8px;text-align:left}th{background:#f5f5f5}pre{background:#f7f7f7;padding:12px}</style>",
        f"<h1>{html.escape(title)}</h1>",
        f"<pre>{html.escape(json.dumps(summary, indent=2, sort_keys=True, default=str))}</pre>",
    ]
    if rows:
        body.append("<table><thead><tr>" + "".join(f"<th>{html.escape(str(h))}</th>" for h in headers) + "</tr></thead><tbody>")
        for row in rows:
            body.append("<tr>" + "".join(f"<td>{html.escape(str(row.get(h, '')))}</td>" for h in headers) + "</tr>")
        body.append("</tbody></table>")
    path.write_text("\n".join(body), encoding="utf-8")

