from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from profiling.analysis import compare_branches as cb
from profiling.analysis import position_analyzer as pa
from profiling.analysis.schema import ActionAnalysis, PositionAnalysis


def _payload(action_id: str) -> dict[str, object]:
    return {
        "player_id": 0,
        "observation": {"board": {"size": 2}, "tick": 0},
        "actions": [{"id": action_id, "type": "END_TURN"}],
    }


def test_compare_branches_resolves_payload_dir_from_repo_root(monkeypatch, tmp_path) -> None:
    repo_root = tmp_path / "repo"
    payload_dir = repo_root / "debug-logs" / "mcts-profile-payloads"
    payload_dir.mkdir(parents=True)
    (repo_root / "py").mkdir()
    (payload_dir / "case-1.json").write_text(json.dumps(_payload("a")), encoding="utf-8")

    calls: list[tuple[str, str]] = []

    def analyze(payload, *, payload_hash, label, target, simulations, batch_size, top_k_actions, max_actions, seed, c_puct, **kwargs):
        calls.append((label, str(target["name"])))
        return PositionAnalysis(
            analysis_version=1,
            payload_hash=payload_hash,
            label=label,
            target_name=str(target["name"]),
            evaluator="static",
            mcts_impl="native_static",
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
            actions=[
                ActionAnalysis(
                    action_id="a",
                    action_fingerprint="END_TURN|a",
                    action_index=0,
                    action_type="END_TURN",
                    prior=1.0,
                    prior_rank=1,
                    visit_share=1.0,
                    visit_rank=1,
                    in_top95=True,
                )
            ],
        )

    monkeypatch.setattr(cb, "PROJECT_ROOT", repo_root)
    monkeypatch.setattr(cb, "analyze_position", analyze)
    monkeypatch.chdir(repo_root / "py")

    output_dir = cb.run(
        SimpleNamespace(
            run_id="test-run",
            output_dir="debug-logs/analysis/branch-compare",
            payload_dir="debug-logs/mcts-profile-payloads",
            payload=[],
            target=["baseline:variant=baseline", "experimental:variant=experimental"],
            positions=1,
            simulations=10000,
            batch_size=64,
            top_k_actions=0,
            max_actions=512,
            seed=0,
            c_puct=1.5,
        )
    )

    assert output_dir == repo_root / "debug-logs" / "analysis" / "branch-compare" / "test-run"
    assert calls == [("case-1", "baseline"), ("case-1", "experimental")]
    assert (output_dir / "positions.jsonl").read_text(encoding="utf-8").count("\n") == 2


def test_compare_branches_fails_when_payload_dir_is_empty(monkeypatch, tmp_path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setattr(cb, "PROJECT_ROOT", repo_root)

    with pytest.raises(RuntimeError, match="no payload JSON files found"):
        cb.run(
            SimpleNamespace(
                run_id="test-run",
                output_dir=tmp_path / "out",
                payload_dir="debug-logs/mcts-profile-payloads",
                payload=[],
                target=["baseline:variant=baseline", "experimental:variant=experimental"],
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
            )
        )


def test_parse_target_branch_aliases_route_to_native_static_exe() -> None:
    assert pa.parse_target("baseline") == {
        "name": "baseline",
        "variant": "baseline",
        "mcts_impl": "native_static_exe",
    }
    assert pa.parse_target("experimental") == {
        "name": "experimental",
        "variant": "experimental",
        "mcts_impl": "native_static_exe",
    }
    assert pa.parse_target("static-baseline:variant=baseline") == {
        "name": "static-baseline",
        "variant": "baseline",
        "mcts_impl": "native_static_exe",
    }


def test_analyze_position_native_static_exe_maps_profile_stats(monkeypatch, tmp_path) -> None:
    payload = {
        "player_id": 0,
        "observation": {"board": {"size": 2}, "tick": 0},
        "actions": [{"id": "a", "type": "END_TURN"}, {"id": "b", "type": "MOVE", "x": 1, "y": 0}],
    }
    exe = tmp_path / "static_mcts_bot.exe"
    exe.write_text("", encoding="utf-8")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "actionId": "b",
                    "rankedActionIds": ["b", "a"],
                    "_profile": {
                        "root_action_stats": [
                            {"action_id": "a", "visits": 1, "visit_share": 0.25, "value_sum": 0.1, "q_mean": 0.1},
                            {"action_id": "b", "visits": 3, "visit_share": 0.75, "value_sum": 1.2, "q_mean": 0.4},
                        ]
                    },
                }
            )
            + "\n",
            stderr="",
        )

    monkeypatch.setattr(pa.subprocess, "run", fake_run)
    monkeypatch.setattr(pa, "_root_static_eval", lambda payload, actions, max_actions: ([0.2, 0.8], 0.5))

    row = pa.analyze_position(
        payload,
        payload_hash="hash",
        label="case",
        target=pa.parse_target("baseline"),
        simulations=4,
        batch_size=2,
        top_k_actions=0,
        max_actions=512,
        seed=7,
        include_breakdown=False,
        native_static_exe=exe,
        build_native_static_exe=False,
    )

    assert calls
    assert "--static-eval-variant" in calls[0]
    assert calls[0][calls[0].index("--static-eval-variant") + 1] == "baseline"
    assert row.mcts_impl == "native_static_exe"
    assert row.selected_action_id == "b"
    assert row.root_value == 0.5
    by_id = {action.action_id: action for action in row.actions}
    assert by_id["b"].visits == 3
    assert by_id["b"].visit_share == 0.75
    assert by_id["b"].q_mean == 0.4
