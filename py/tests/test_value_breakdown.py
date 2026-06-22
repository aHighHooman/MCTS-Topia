from __future__ import annotations

import csv
import json
import os
from types import SimpleNamespace

import pytest

from profiling.analysis import value_breakdown as vb
from profiling.analysis.schema import PositionAnalysis, ActionAnalysis


def _payload(action_id: str) -> dict[str, object]:
    return {
        "player_id": 0,
        "observation": {"board": {"size": 2}, "tick": 0},
        "actions": [{"id": action_id, "type": "END_TURN"}],
    }


def _args(repo_root, *, output_dir=None, compare_dir=""):
    return SimpleNamespace(
        run_id="test-run",
        output_dir=output_dir if output_dir is not None else "debug-logs/analysis/value-breakdown",
        payload_dir="debug-logs/mcts-profile-payloads",
        target="baseline",
        positions=1,
        simulations=10000,
        batch_size=64,
        top_k_actions=0,
        max_actions=512,
        seed=0,
        c_puct=1.5,
        native_static_exe=repo_root / "out" / "native" / "static_mcts_bot.exe",
        build_native_static_exe=True,
        native_static_search_mode="primitive",
        compare_dir=compare_dir,
    )


def _analysis(
    *,
    payload_hash="hash",
    label="case",
    action_id="a",
    root_terms=None,
    action_terms=None,
    actions=None,
) -> PositionAnalysis:
    if actions is None:
        actions = [
            ActionAnalysis(
                action_id=action_id,
                action_fingerprint=f"END_TURN|{action_id}",
                action_index=0,
                action_type="END_TURN",
                prior=1.0,
                prior_rank=1,
                visits=10,
                visit_share=1.0,
                visit_rank=1,
                q_mean=0.5,
                value_sum=5.0,
                in_top95=True,
                value_breakdown={"terms": list(action_terms or [])},
            )
        ]
    return PositionAnalysis(
        analysis_version=1,
        payload_hash=payload_hash,
        label=label,
        target_name="baseline",
        evaluator="static",
        mcts_impl="native_static_exe",
        static_eval_variant="baseline",
        seed=0,
        simulations=10000,
        c_puct=1.5,
        top_k_actions=0,
        max_actions=512,
        root_value=0.0,
        selected_action_id=action_id,
        selected_action_fingerprint=f"END_TURN|{action_id}",
        action_count_raw=1,
        action_count_analyzed=1,
        search_sec=0.001,
        value_breakdown={"terms": list(root_terms or [])},
        actions=actions,
    )


def test_value_breakdown_resolves_payload_dir_from_repo_root_and_skips_metadata(monkeypatch, tmp_path) -> None:
    repo_root = tmp_path / "repo"
    payload_dir = repo_root / "debug-logs" / "mcts-profile-payloads"
    payload_dir.mkdir(parents=True)
    (repo_root / "py").mkdir()
    (payload_dir / "manifest.json").write_text(json.dumps({"generated_payloads": 1}), encoding="utf-8")
    (payload_dir / "case-1.json").write_text(json.dumps(_payload("a")), encoding="utf-8")

    calls: list[str] = []

    def analyze(payload, *, payload_hash, label, target, simulations, batch_size, top_k_actions, max_actions, seed, c_puct, include_breakdown, **kwargs):
        calls.append(label)
        return PositionAnalysis(
            analysis_version=1,
            payload_hash=payload_hash,
            label=label,
            target_name=str(target["name"]),
            evaluator="static",
            mcts_impl="native_static_exe",
            static_eval_variant=str(target["variant"]),
            seed=int(seed),
            simulations=int(simulations),
            c_puct=float(c_puct),
            top_k_actions=int(top_k_actions),
            max_actions=int(max_actions),
            root_value=0.0,
            selected_action_id="a",
            selected_action_fingerprint="END_TURN|a",
            action_count_raw=1,
            action_count_analyzed=1,
            search_sec=0.001,
            value_breakdown={
                "terms": [
                    {
                        "name": "term",
                        "raw": 1.0,
                        "normalized": 0.5,
                        "abs_share": 1.0,
                        "linearized_value": 0.25,
                    }
                ]
            },
            actions=[
                ActionAnalysis(
                    action_id="a",
                    action_fingerprint="END_TURN|a",
                    action_index=0,
                    action_type="END_TURN",
                    prior=1.0,
                    prior_rank=1,
                    visits=10,
                    visit_share=1.0,
                    visit_rank=1,
                    q_mean=0.5,
                    value_sum=5.0,
                    in_top95=True,
                    value_breakdown={
                        "terms": [
                            {
                                "name": "term_action_1",
                                "raw": 2.0,
                                "normalized": 1.0,
                                "abs_share": 1.0,
                                "linearized_value": 0.5,
                            }
                        ]
                    }
                )
            ]
        )

    monkeypatch.setattr(vb, "PROJECT_ROOT", repo_root)
    monkeypatch.setattr(vb, "analyze_position", analyze)
    monkeypatch.chdir(repo_root / "py")

    output_dir = vb.run(_args(repo_root))

    assert output_dir == repo_root / "debug-logs" / "analysis" / "value-breakdown" / "test-run"
    assert calls == ["case-1"]
    assert json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))["positions"] == 1
    assert (output_dir / "terms.csv").read_text(encoding="utf-8").count("\n") == 3


def test_value_breakdown_fails_when_payload_dir_is_empty(monkeypatch, tmp_path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setattr(vb, "PROJECT_ROOT", repo_root)

    with pytest.raises(RuntimeError, match="no payload JSON files found"):
        vb.run(
            _args(repo_root, output_dir=tmp_path / "out")
        )


def test_value_breakdown_prefers_newest_payloads_when_limited(monkeypatch, tmp_path) -> None:
    repo_root = tmp_path / "repo"
    payload_dir = repo_root / "debug-logs" / "mcts-profile-payloads"
    payload_dir.mkdir(parents=True)
    old_payload = payload_dir / "a-old.json"
    new_payload = payload_dir / "z-new.json"
    old_payload.write_text(json.dumps(_payload("old")), encoding="utf-8")
    new_payload.write_text(json.dumps(_payload("new")), encoding="utf-8")
    os.utime(old_payload, (1000, 1000))
    os.utime(new_payload, (2000, 2000))

    calls: list[str] = []

    def analyze(payload, *, payload_hash, label, target, simulations, batch_size, top_k_actions, max_actions, seed, c_puct, include_breakdown, **kwargs):
        calls.append(label)
        return PositionAnalysis(
            analysis_version=1,
            payload_hash=payload_hash,
            label=label,
            target_name=str(target["name"]),
            evaluator="static",
            mcts_impl="native_static_exe",
            static_eval_variant=str(target["variant"]),
            seed=int(seed),
            simulations=int(simulations),
            c_puct=float(c_puct),
            top_k_actions=int(top_k_actions),
            max_actions=int(max_actions),
            root_value=0.0,
            selected_action_id="new",
            selected_action_fingerprint="END_TURN|new",
            action_count_raw=1,
            action_count_analyzed=1,
            search_sec=0.001,
            actions=[],
            value_breakdown={"terms": []},
        )

    monkeypatch.setattr(vb, "PROJECT_ROOT", repo_root)
    monkeypatch.setattr(vb, "analyze_position", analyze)

    output_dir = vb.run(_args(repo_root, output_dir=tmp_path / "out"))

    assert calls == ["z-new"]
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["payload_order"] == "mtime-desc"
    assert summary["processed_payloads"][0]["label"] == "z-new"


def test_value_breakdown_clears_stale_outputs_and_marks_failure(monkeypatch, tmp_path) -> None:
    repo_root = tmp_path / "repo"
    payload_dir = repo_root / "debug-logs" / "mcts-profile-payloads"
    payload_dir.mkdir(parents=True)
    (payload_dir / "case-1.json").write_text(json.dumps(_payload("a")), encoding="utf-8")
    output_dir = tmp_path / "out" / "test-run"
    output_dir.mkdir(parents=True)
    for name in (
        "positions.jsonl",
        "terms.csv",
        "terms_components.csv",
        "terms_aggregate.csv",
        "terms_comparable_old.csv",
        "term_summary.csv",
        "term_comparison.csv",
        "summary.json",
        "report.html",
    ):
        (output_dir / name).write_text("stale", encoding="utf-8")

    def analyze(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(vb, "PROJECT_ROOT", repo_root)
    monkeypatch.setattr(vb, "analyze_position", analyze)

    with pytest.raises(RuntimeError, match="boom"):
        vb.run(
            _args(repo_root, output_dir=tmp_path / "out")
        )

    for name in (
        "positions.jsonl",
        "terms.csv",
        "terms_components.csv",
        "terms_aggregate.csv",
        "terms_comparable_old.csv",
        "term_summary.csv",
        "term_comparison.csv",
        "summary.json",
        "report.html",
    ):
        assert not (output_dir / name).exists()
    status = json.loads((output_dir / "run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["error"] == "boom"


def test_value_breakdown_writes_aggregate_component_and_old_style_outputs(monkeypatch, tmp_path) -> None:
    repo_root = tmp_path / "repo"
    payload_dir = repo_root / "debug-logs" / "mcts-profile-payloads"
    payload_dir.mkdir(parents=True)
    (payload_dir / "case-1.json").write_text(json.dumps(_payload("a")), encoding="utf-8")

    root_terms = [
        {"name": "military.unit_power.experimental_formula.own.projection", "parent": "military.unit_power", "raw": 8.0, "normalized": 0.08, "linearized_value": 0.04, "contributes": True},
        {"name": "military.unit_power.experimental_formula.enemy.projection", "parent": "military.unit_power", "raw": -3.0, "normalized": -0.03, "linearized_value": -0.015, "contributes": True},
        {"name": "military.unit_power", "parent": "", "raw": 5.0, "normalized": 0.05, "linearized_value": 0.025, "tunable": False, "contributes": False},
        {"name": "economy.stars", "parent": "", "raw": 2.0, "normalized": 0.02, "linearized_value": 0.01, "contributes": True},
    ]

    def analyze(payload, **kwargs):
        return _analysis(payload_hash=kwargs["payload_hash"], label=kwargs["label"], root_terms=root_terms, action_terms=[])

    monkeypatch.setattr(vb, "PROJECT_ROOT", repo_root)
    monkeypatch.setattr(vb, "analyze_position", analyze)

    output_dir = vb.run(_args(repo_root, output_dir=tmp_path / "out"))

    terms = list(csv.DictReader((output_dir / "terms.csv").open(newline="", encoding="utf-8")))
    aggregate = list(csv.DictReader((output_dir / "terms_aggregate.csv").open(newline="", encoding="utf-8")))
    components = list(csv.DictReader((output_dir / "terms_components.csv").open(newline="", encoding="utf-8")))
    old_rows = list(csv.DictReader((output_dir / "terms_comparable_old.csv").open(newline="", encoding="utf-8")))
    summary_rows = list(csv.DictReader((output_dir / "term_summary.csv").open(newline="", encoding="utf-8")))

    assert {row["row_type"] for row in terms} == {"component", "aggregate", "diagnostic"}
    assert len(components) == len(terms)
    root_aggregate = {
        row["name"]: row
        for row in aggregate
        if row["action_id"] == "root"
    }
    assert float(root_aggregate["military.unit_power"]["raw"]) == pytest.approx(5.0)
    assert float(root_aggregate["military.unit_power"]["abs_raw"]) == pytest.approx(11.0)
    assert float(root_aggregate["economy.stars"]["raw"]) == pytest.approx(2.0)
    assert all(row["name"] in {"military.unit_power", "economy.stars"} for row in old_rows)
    assert {row["parent_term"] for row in summary_rows} == {"military.unit_power", "economy.stars"}
    assert "Warning: This report contains component rows" in (output_dir / "report.html").read_text(encoding="utf-8")


def test_value_breakdown_validation_counts_are_written(monkeypatch, tmp_path) -> None:
    repo_root = tmp_path / "repo"
    payload_dir = repo_root / "debug-logs" / "mcts-profile-payloads"
    payload_dir.mkdir(parents=True)
    (payload_dir / "case-1.json").write_text(json.dumps(_payload("a")), encoding="utf-8")

    def analyze(payload, **kwargs):
        return _analysis(
            payload_hash=kwargs["payload_hash"],
            label=kwargs["label"],
            root_terms=[{"name": "term", "raw": 1.0, "normalized": 0.1, "linearized_value": 0.05}],
            action_terms=[],
        )

    monkeypatch.setattr(vb, "PROJECT_ROOT", repo_root)
    monkeypatch.setattr(vb, "analyze_position", analyze)

    output_dir = vb.run(_args(repo_root, output_dir=tmp_path / "out"))
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    status = json.loads((output_dir / "run_status.json").read_text(encoding="utf-8"))

    assert summary["validation"]["positions_jsonl_count"] == 1
    assert summary["validation"]["terms_csv_rows"] == 1
    assert summary["validation"]["aggregate_csv_rows"] == 1
    assert summary["validation"]["terms_match_summary"] is True
    assert status["validation"] == summary["validation"]


def test_value_breakdown_comparison_output_uses_aggregate_shares(monkeypatch, tmp_path) -> None:
    repo_root = tmp_path / "repo"
    payload_dir = repo_root / "debug-logs" / "mcts-profile-payloads"
    payload_dir.mkdir(parents=True)
    (payload_dir / "case-1.json").write_text(json.dumps(_payload("a")), encoding="utf-8")
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    with (baseline_dir / "terms_aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=vb.AGGREGATE_FIELDS)
        writer.writeheader()
        writer.writerows(
            [
                {"payload_hash": "h", "label": "case", "action_rank": 0, "action_id": "root", "action_fingerprint": "root", "name": "military.unit_power", "aggregate_term": "military.unit_power", "raw": 1.0, "abs_raw": 1.0, "normalized": 0.01, "abs_share": 0.25, "linearized_value": 0.0, "component_count": 1, "row_type": "aggregate"},
                {"payload_hash": "h", "label": "case", "action_rank": 0, "action_id": "root", "action_fingerprint": "root", "name": "economy.stars", "aggregate_term": "economy.stars", "raw": 3.0, "abs_raw": 3.0, "normalized": 0.03, "abs_share": 0.75, "linearized_value": 0.0, "component_count": 1, "row_type": "aggregate"},
            ]
        )

    def analyze(payload, **kwargs):
        return _analysis(
            payload_hash=kwargs["payload_hash"],
            label=kwargs["label"],
            root_terms=[
                {"name": "military.unit_power", "raw": 2.0, "normalized": 0.02, "linearized_value": 0.0},
                {"name": "economy.stars", "raw": 2.0, "normalized": 0.02, "linearized_value": 0.0},
            ],
            action_terms=[],
        )

    monkeypatch.setattr(vb, "PROJECT_ROOT", repo_root)
    monkeypatch.setattr(vb, "analyze_position", analyze)

    output_dir = vb.run(_args(repo_root, output_dir=tmp_path / "out", compare_dir=str(baseline_dir)))
    comparison = {
        row["term"]: row
        for row in csv.DictReader((output_dir / "term_comparison.csv").open(newline="", encoding="utf-8"))
    }

    assert float(comparison["military.unit_power"]["baseline_share"]) == pytest.approx(0.25)
    assert float(comparison["military.unit_power"]["experimental_share"]) == pytest.approx(0.5)
    assert float(comparison["military.unit_power"]["delta_pp"]) == pytest.approx(25.0)
