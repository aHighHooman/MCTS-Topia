from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from profiling.analysis import branchpoint_dataset_builder as bp
from profiling.analysis import position_analyzer as pa
from profiling.analysis import root_child_value_matrix as rcm
from profiling.analysis import tune_static_eval_weights as tuner
from profiling.analysis.schema import ActionAnalysis, PositionAnalysis


def _payload() -> dict[str, object]:
    return {
        "player_id": 0,
        "observation": {"board": {"size": 2}, "tick": 0},
        "actions": [
            {"id": "a", "type": "END_TURN"},
            {"id": "b", "type": "MOVE", "x": 1, "y": 0},
        ],
    }


def test_analyze_position_fails_when_profile_stats_missing(monkeypatch, tmp_path) -> None:
    exe = tmp_path / "native_static_mcts_bot.exe"
    exe.write_text("", encoding="utf-8")

    monkeypatch.setattr(pa, "_root_static_eval", lambda payload, actions, max_actions: ([0.5, 0.5], 0.0))
    monkeypatch.setattr(
        pa.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=json.dumps({"actionId": "a", "_profile": {}}), stderr=""),
    )

    with pytest.raises(RuntimeError, match="root_action_stats"):
        pa.analyze_position(
            _payload(),
            payload_hash="hash",
            label="case",
            target=pa.parse_target("baseline"),
            simulations=4,
            batch_size=2,
            top_k_actions=0,
            max_actions=512,
            include_breakdown=False,
            native_static_exe=exe,
            build_native_static_exe=False,
        )


def test_analyze_position_continues_when_static_extension_unavailable(monkeypatch, tmp_path) -> None:
    exe = tmp_path / "native_static_mcts_bot.exe"
    exe.write_text("", encoding="utf-8")

    def fail_root(*args, **kwargs):
        raise RuntimeError("extension unavailable")

    monkeypatch.setattr(pa, "_root_static_eval", fail_root)
    monkeypatch.setattr(
        pa.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "actionId": "b",
                    "_profile": {
                        "root_action_stats": [
                            {"action_id": "a", "visits": 1, "visit_share": 0.25},
                            {"action_id": "b", "visits": 3, "visit_share": 0.75},
                        ]
                    },
                }
            ),
            stderr="",
        ),
    )

    row = pa.analyze_position(
        _payload(),
        payload_hash="hash",
        label="case",
        target=pa.parse_target("baseline"),
        simulations=4,
        batch_size=2,
        top_k_actions=0,
        max_actions=512,
        include_breakdown=False,
        native_static_exe=exe,
        build_native_static_exe=False,
    )

    assert row.selected_action_id == "b"
    assert row.static_eval_source == "unavailable"
    assert row.profile_validated is True
    assert row.warnings


def test_root_child_value_matrix_includes_all_actions(monkeypatch, tmp_path) -> None:
    def fake_analyze(*args, **kwargs):
        return PositionAnalysis(
            analysis_version=1,
            payload_hash="hash",
            label="case",
            target_name="experimental",
            evaluator="static",
            mcts_impl="native_static_exe",
            static_eval_variant="experimental",
            seed=0,
            simulations=4,
            c_puct=1.5,
            top_k_actions=0,
            max_actions=512,
            root_value=0.0,
            selected_action_id="b",
            selected_action_fingerprint="MOVE:x=1:y=0",
            action_count_raw=2,
            action_count_analyzed=2,
            search_sec=0.001,
            actions=[
                ActionAnalysis("a", "END_TURN", 0, "END_TURN", prior=0.4, prior_rank=2, visit_share=0.2, visit_rank=2),
                ActionAnalysis("b", "MOVE:x=1:y=0", 1, "MOVE", prior=0.6, prior_rank=1, visit_share=0.8, visit_rank=1),
            ],
        )

    def fake_breakdown(payload, aid, max_actions):
        raw = 1.0 if aid == "a" else 2.0
        return {
            "raw_total": raw,
            "final_value": raw / 10.0,
            "terms": [{"name": "term", "feature_value": raw, "weight": 1.0, "raw": raw}],
        }

    monkeypatch.setattr(rcm, "analyze_position", fake_analyze)
    monkeypatch.setattr(rcm, "_action_breakdown", fake_breakdown)
    result = rcm.analyze_payload(
        _payload(),
        payload_hash="hash",
        label="case",
        target={"name": "experimental", "variant": "experimental"},
        simulations=4,
        batch_size=2,
        top_k_actions=0,
        max_actions=512,
        seed=0,
        c_puct=1.5,
        native_static_exe=tmp_path / "missing.exe",
        build_native_static_exe=False,
        native_static_search_mode="primitive",
    )

    assert [row["action_id"] for row in result["actions"]] == ["a", "b"]
    assert len(result["terms"]) == 2
    assert {row["child_value_rank"] for row in result["actions"]} == {1, 2}


def test_branchpoint_builder_filters_high_confidence_divergence() -> None:
    positions = [
        {
            "payload_hash": "h",
            "label": "case",
            "target_name": "baseline",
            "selected_action_fingerprint": "A",
            "actions": [
                {"action_id": "a", "action_fingerprint": "A", "visit_share": 0.8, "visit_rank": 1},
                {"action_id": "b", "action_fingerprint": "B", "visit_share": 0.2, "visit_rank": 2},
            ],
        },
        {
            "payload_hash": "h",
            "label": "case",
            "target_name": "experimental",
            "selected_action_fingerprint": "B",
            "actions": [
                {"action_id": "a", "action_fingerprint": "A", "visit_share": 0.02, "visit_rank": 5},
                {"action_id": "b", "action_fingerprint": "B", "visit_share": 0.98, "visit_rank": 1},
            ],
        },
    ]

    rows = bp.build_candidates(
        positions,
        baseline_name="baseline",
        experimental_name="experimental",
        min_js_visit_bits=0.08,
        min_baseline_top_visit_share=0.35,
        min_top_gap=0.05,
        max_experimental_baseline_visit_share=0.10,
        min_experimental_baseline_rank=4,
    )

    assert len(rows) == 1
    assert rows[0]["baseline_action_id"] == "a"
    assert rows[0]["experimental_action_id"] == "b"


def test_tuner_moves_weight_toward_constraint(tmp_path) -> None:
    terms_csv = tmp_path / "terms.csv"
    with terms_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["payload_hash", "action_id", "name", "feature_value", "weight"])
        writer.writeheader()
        writer.writerows(
            [
                {"payload_hash": "h", "action_id": "desired", "name": "term", "feature_value": 2.0, "weight": 1.0},
                {"payload_hash": "h", "action_id": "rival", "name": "term", "feature_value": 0.0, "weight": 1.0},
            ]
        )
    constraints = tmp_path / "constraints.jsonl"
    constraints.write_text(
        json.dumps({"payload_hash": "h", "desired_action_id": "desired", "rival_action_id": "rival", "margin": 3.0}) + "\n",
        encoding="utf-8",
    )

    result = tuner.optimize(terms_csv, constraints, steps=100, learning_rate=0.01, l2=0.001)

    assert result["weights"]["term"] > 1.0
    assert result["constraints"][0]["after"] > result["constraints"][0]["before"]
