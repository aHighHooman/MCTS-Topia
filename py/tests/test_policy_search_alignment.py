from __future__ import annotations

import os
import csv
from pathlib import Path
from types import SimpleNamespace

from profiling import policy_search_alignment as psa


def test_policy_search_alignment_threads_static_eval_variant(monkeypatch, tmp_path) -> None:
    payload = {
        "player_id": 0,
        "observation": {"board": {"size": 2}, "tick": 0},
        "actions": [
            {"id": "a", "type": "MOVE"},
            {"id": "b", "type": "END_TURN"},
        ],
    }
    case = psa.PayloadCase(label="case-1", path=Path("case-1.json"), payload=payload)
    captured_summary: dict[str, object] = {}

    monkeypatch.setenv("TRIBES_STATIC_EVAL_VARIANT", "baseline")
    monkeypatch.setattr(psa, "_load_cases", lambda *_args, **_kwargs: [case])
    monkeypatch.setattr(
        psa,
        "_policy",
        lambda *_args, **_kwargs: psa.PolicyResult(priors=[0.9, 0.1], value=0.25),
    )
    monkeypatch.setattr(
        psa,
        "_run_search",
        lambda *_args, **_kwargs: SimpleNamespace(visit_distribution={"a": 0.2, "b": 0.8}, value=0.5),
    )
    monkeypatch.setattr(
        psa,
        "_write_json",
        lambda _path, payload: captured_summary.update(payload),
    )

    args = SimpleNamespace(
        evaluator="static",
        payload=[],
        payload_dir=tmp_path,
        no_payload_dir=True,
        positions=None,
        simulations=20000,
        batch_size=64,
        top_k_actions=0,
        max_actions=512,
        static_eval_variant="experimental",
        root_temperature=1.0,
        dirichlet_epsilon=0.0,
        checkpoint=tmp_path / "checkpoint.pt",
        device="cpu",
        static_policy_weight=0.5,
        static_value_weight=0.5,
        output_dir=tmp_path / "out",
    )

    psa.run(args)

    assert os.environ["TRIBES_STATIC_EVAL_VARIANT"] == "experimental"
    assert captured_summary["static_eval_variant"] == "experimental"
    assert captured_summary["simulations"] == 20000
    assert str(captured_summary["outputs"]["positions_csv"]).endswith(
        r"out\experimental\experimental_sims20000_all-actions_positions.csv"
    )
    assert captured_summary["policy_entropy_bits_avg"] > 0.0
    assert captured_summary["visit_entropy_bits_avg"] > 0.0
    assert captured_summary["kl_policy_visit_bits_avg"] > 0.0

    positions_csv = tmp_path / "out" / "experimental" / "experimental_sims20000_all-actions_positions.csv"
    with positions_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert "spearman_rank" not in (rows[0].keys() if rows else set())
    assert rows[0]["policy_entropy_bits"]
    assert rows[0]["visit_entropy_bits"]
    assert rows[0]["kl_policy_visit_bits"]
    assert rows[0]["kl_visit_policy_bits"]


def test_policy_search_alignment_all_variants_waits_between_runs(monkeypatch, tmp_path) -> None:
    seen_variants: list[str] = []
    cleanup_calls = 0
    sleep_calls: list[float] = []

    monkeypatch.setattr(
        psa,
        "_run_alignment",
        lambda args: seen_variants.append(str(args.static_eval_variant)),
    )
    monkeypatch.setattr(psa.time, "sleep", lambda seconds: sleep_calls.append(float(seconds)))

    def cleanup() -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1

    monkeypatch.setattr(psa, "_release_variant_memory", cleanup)

    args = SimpleNamespace(
        static_eval_variant="all",
        variant_delay_sec=5.0,
    )

    psa.run(args)

    assert seen_variants == ["baseline", "experimental"]
    assert cleanup_calls == 2
    assert sleep_calls == [5.0]
