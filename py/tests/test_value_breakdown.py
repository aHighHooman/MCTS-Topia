from __future__ import annotations

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

    output_dir = vb.run(
        SimpleNamespace(
            run_id="test-run",
            output_dir="debug-logs/analysis/value-breakdown",
            payload_dir="debug-logs/mcts-profile-payloads",
            target="baseline",
            positions=1,
            simulations=10000,
            batch_size=64,
            top_k_actions=0,
            max_actions=512,
            seed=0,
            c_puct=1.5,
            native_static_exe=repo_root / "out" / "native" / "native_static_mcts_bot.exe",
            build_native_static_exe=True,
            native_static_search_mode="primitive",
        )
    )

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
            SimpleNamespace(
                run_id="test-run",
                output_dir=tmp_path / "out",
                payload_dir="debug-logs/mcts-profile-payloads",
                target="baseline",
                positions=1,
                simulations=10000,
                batch_size=64,
                top_k_actions=0,
                max_actions=512,
                seed=0,
                c_puct=1.5,
                native_static_exe=repo_root / "out" / "native" / "native_static_mcts_bot.exe",
                build_native_static_exe=True,
                native_static_search_mode="primitive",
            )
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

    output_dir = vb.run(
        SimpleNamespace(
            run_id="test-run",
            output_dir=tmp_path / "out",
            payload_dir="debug-logs/mcts-profile-payloads",
            target="baseline",
            positions=1,
            simulations=10000,
            batch_size=64,
            top_k_actions=0,
            max_actions=512,
            seed=0,
            c_puct=1.5,
            native_static_exe=repo_root / "out" / "native" / "native_static_mcts_bot.exe",
            build_native_static_exe=True,
            native_static_search_mode="primitive",
        )
    )

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
    for name in ("positions.jsonl", "terms.csv", "summary.json", "report.html"):
        (output_dir / name).write_text("stale", encoding="utf-8")

    def analyze(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(vb, "PROJECT_ROOT", repo_root)
    monkeypatch.setattr(vb, "analyze_position", analyze)

    with pytest.raises(RuntimeError, match="boom"):
        vb.run(
            SimpleNamespace(
                run_id="test-run",
                output_dir=tmp_path / "out",
                payload_dir="debug-logs/mcts-profile-payloads",
                target="baseline",
                positions=1,
                simulations=10000,
                batch_size=64,
                top_k_actions=0,
                max_actions=512,
                seed=0,
                c_puct=1.5,
                native_static_exe=repo_root / "out" / "native" / "native_static_mcts_bot.exe",
                build_native_static_exe=True,
                native_static_search_mode="primitive",
            )
        )

    for name in ("positions.jsonl", "terms.csv", "summary.json", "report.html"):
        assert not (output_dir / name).exists()
    status = json.loads((output_dir / "run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["error"] == "boom"
