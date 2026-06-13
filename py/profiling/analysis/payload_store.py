from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from nn.encoding import normalize_message


def canonical_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_message(payload)
    return {
        "player_id": int(normalized.get("player_id", 0)),
        "observation": normalized.get("observation", {}),
        "actions": list(normalized.get("actions", [])),
    }


def payload_hash(payload: dict[str, Any]) -> str:
    data = json.dumps(canonical_payload(payload), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def store_payload(payload: dict[str, Any], root: Path) -> str:
    digest = payload_hash(payload)
    path = root / digest[:2] / digest[2:4] / f"{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            json.dumps(canonical_payload(payload), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    return digest


def load_payload(path: Path) -> dict[str, Any] | None:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    if not isinstance(raw, dict) or not isinstance(raw.get("actions"), list) or not isinstance(raw.get("observation"), dict):
        return None
    return canonical_payload(raw)

