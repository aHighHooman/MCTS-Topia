from __future__ import annotations

import csv
import json
import os
from types import SimpleNamespace

import pytest

from profiling.analysis import branchpoint_dataset_builder as bp
from profiling.analysis.payload_store import load_payload
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
    exe = tmp_path / "static_mcts_bot.exe"
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
    exe = tmp_path / "static_mcts_bot.exe"
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


def test_native_static_exe_staleness_is_rejected_when_build_disabled(monkeypatch, tmp_path) -> None:
    repo_root = tmp_path / "repo"
    src_dir = repo_root / "py" / "search" / "native" / "src"
    bot_dir = repo_root / "bots"
    scripts_dir = repo_root / "scripts"
    out_dir = repo_root / "out" / "native"
    src_dir.mkdir(parents=True)
    bot_dir.mkdir()
    scripts_dir.mkdir()
    out_dir.mkdir(parents=True)
    exe = out_dir / "static_mcts_bot.exe"
    source = src_dir / "static_eval.cpp"
    bot = bot_dir / "static_mcts_bot.cpp"
    build_script = scripts_dir / "build_static_bot.ps1"
    exe.write_text("exe", encoding="utf-8")
    source.write_text("source", encoding="utf-8")
    bot.write_text("bot", encoding="utf-8")
    build_script.write_text("build", encoding="utf-8")
    os.utime(exe, (1000, 1000))
    os.utime(source, (2000, 2000))
    os.utime(bot, (1000, 1000))
    os.utime(build_script, (1000, 1000))

    monkeypatch.setattr(pa, "PROJECT_ROOT", repo_root)

    with pytest.raises(RuntimeError, match="older than one or more build inputs"):
        pa._ensure_native_static_exe(exe, build_native_static_exe=False)


def test_static_eval_target_env_applies_and_restores_weight_overrides(monkeypatch) -> None:
    monkeypatch.setenv("TRIBES_STATIC_EVAL_VARIANT", "baseline")
    monkeypatch.setenv("TRIBES_STATIC_EVAL_WEIGHT_OVERRIDES", "economy.stars=1")

    target = pa.parse_target("temp:variant=experimental,weight_overrides=military.unit_power=0.25;economy.stars=0.5")
    with pa.static_eval_target_env(target):
        assert os.environ["TRIBES_STATIC_EVAL_VARIANT"] == "experimental"
        assert os.environ["TRIBES_STATIC_EVAL_WEIGHT_OVERRIDES"] == "military.unit_power=0.25,economy.stars=0.5"

    assert os.environ["TRIBES_STATIC_EVAL_VARIANT"] == "baseline"
    assert os.environ["TRIBES_STATIC_EVAL_WEIGHT_OVERRIDES"] == "economy.stars=1"


def test_native_static_exe_receives_weight_overrides(monkeypatch, tmp_path) -> None:
    exe = tmp_path / "static_mcts_bot.exe"
    exe.write_text("", encoding="utf-8")
    captured: dict[str, object] = {}

    monkeypatch.setattr(pa, "_ensure_native_static_exe", lambda *args, **kwargs: exe)

    def fake_run(command, **kwargs):
        captured["command"] = command
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"actionId": "a", "_profile": {"root_action_stats": []}}),
            stderr="",
        )

    monkeypatch.setattr(pa.subprocess, "run", fake_run)
    response = pa._run_native_static_exe(
        _payload(),
        target=pa.parse_target("temp:variant=experimental,weight_overrides=military.unit_power=0.25"),
        simulations=0,
        batch_size=1,
        top_k_actions=0,
        max_actions=512,
        seed=0,
        c_puct=1.5,
        native_static_exe=exe,
        build_native_static_exe=False,
        native_static_search_mode="primitive",
    )

    assert response["actionId"] == "a"
    command = list(captured["command"])
    assert "--static-eval-weight-overrides" in command
    assert command[command.index("--static-eval-weight-overrides") + 1] == "military.unit_power=0.25"


def test_payload_store_loads_dense_tournament_payload(tmp_path) -> None:
    dense_payload = {
        "player_id": 0,
        "observation": [
            3,
            "Drylands",
            0,
            True,
            False,
            [[0, 0, 5, 100, 2, 1, []]],
            [],
            [],
            [
                2,
                [[0, 0], [0, 0]],
                [[None, None], [None, None]],
                [[None, None], [None, None]],
                [[0, 0], [0, 0]],
                [[0, 0], [0, 0]],
                [[1, 1], [1, 1]],
                [[0, 0], [0, 0]],
            ],
            [],
            [[0]],
        ],
        "actions": [[10, 0]],
    }
    path = tmp_path / "payload.json"
    path.write_text(json.dumps(dense_payload), encoding="utf-8")

    payload = load_payload(path)

    assert payload is not None
    assert payload["observation"]["tick"] == 3
    assert payload["actions"][0]["id"] == "A0"
    assert payload["actions"][0]["type"] == "END_TURN"


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


def test_branchpoint_builder_rejects_missing_counterfactual_filter_metadata() -> None:
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
        require_counterfactual_rescue=True,
    )

    assert rows == []


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


def test_tuner_preserves_sign_and_rejects_unsatisfied_candidate(tmp_path) -> None:
    terms_csv = tmp_path / "terms.csv"
    with terms_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["payload_hash", "action_id", "name", "feature_value", "weight", "tunable", "contributes"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"payload_hash": "h", "action_id": "desired", "name": "term", "feature_value": -100.0, "weight": 1.0, "tunable": "true", "contributes": "true"},
                {"payload_hash": "h", "action_id": "rival", "name": "term", "feature_value": 0.0, "weight": 1.0, "tunable": "true", "contributes": "true"},
                {"payload_hash": "h", "action_id": "desired", "name": "summary", "feature_value": 1000.0, "weight": 1.0, "tunable": "false", "contributes": "false"},
                {"payload_hash": "h", "action_id": "rival", "name": "summary", "feature_value": 0.0, "weight": 1.0, "tunable": "false", "contributes": "false"},
            ]
        )
    constraints = tmp_path / "constraints.jsonl"
    constraints.write_text(
        json.dumps({"payload_hash": "h", "desired_action_id": "desired", "rival_action_id": "rival", "margin": 1.0}) + "\n",
        encoding="utf-8",
    )

    result = tuner.optimize(
        terms_csv,
        constraints,
        steps=100,
        learning_rate=0.01,
        l2=0.001,
        max_abs_delta=0.5,
        max_relative_delta=0.5,
        preserve_sign=True,
    )

    assert result["accepted"] is False
    assert result["weights"]["term"] >= 0.0
    assert "summary" not in result["weights"]
