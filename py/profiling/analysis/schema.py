from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ActionAnalysis:
    action_id: str
    action_fingerprint: str
    action_index: int
    action_type: str
    prior: float = 0.0
    prior_rank: int = 0
    visits: int | None = None
    visit_share: float = 0.0
    visit_rank: int = 0
    q_mean: float | None = None
    value_sum: float | None = None
    in_top95: bool = False
    unit_id: Any = None
    city_id: Any = None
    x: Any = None
    y: Any = None
    target: Any = None
    value_breakdown: dict[str, Any] | None = None


@dataclass
class PositionAnalysis:
    analysis_version: int
    payload_hash: str
    label: str
    target_name: str
    evaluator: str
    mcts_impl: str
    static_eval_variant: str
    seed: int
    simulations: int
    c_puct: float
    top_k_actions: int
    max_actions: int
    root_value: float
    selected_action_id: str
    selected_action_fingerprint: str
    action_count_raw: int
    action_count_analyzed: int
    search_sec: float
    actions: list[ActionAnalysis] = field(default_factory=list)
    value_breakdown: dict[str, Any] | None = None


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return value


def append_jsonl(path: Path, row: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(row), sort_keys=True, default=str) + "\n")

