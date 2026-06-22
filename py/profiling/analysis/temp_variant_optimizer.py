from __future__ import annotations

import argparse
import re
import json
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from profiling.config import load_config_defaults
from profiling.analysis import branchpoint_dataset_builder, compare_branches, root_child_value_matrix, tune_static_eval_weights
from project_paths import game_json_jar

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "debug-logs" / "analysis" / "temp-variant-optimizer"
DEFAULT_CONFIG = PROJECT_ROOT / "py" / "profiling" / "configs" / "temp_variant_optimizer.json"


def _repo_relative(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _command_with_arg(command: list[str], flag: str, value: str) -> list[str]:
    out = list(command)
    if flag in out:
        index = out.index(flag)
        if index + 1 < len(out):
            out[index + 1] = value
        else:
            out.append(value)
        return out
    out.extend([flag, value])
    return out


def _participant_command(participant: dict[str, Any]) -> list[str]:
    command = participant.get("External Command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise RuntimeError(f"participant has no External Command string array: {participant.get('Name', '')}")
    return list(command)


def _write_tournament_config(
    base_config: dict[str, Any],
    *,
    output_dir: Path,
    level_seed: int,
    temp_overrides: str,
    temp_bot_seed: int | None,
    temp_name: str,
) -> Path:
    config = dict(base_config)
    participants = [dict(row) for row in base_config.get("Participants", [])]
    if len(participants) < 2:
        raise RuntimeError("base tournament config requires at least two Participants")
    temp = dict(participants[1])
    command = _participant_command(temp)
    command = _command_with_arg(command, "--static-eval-variant", "experimental")
    command = _command_with_arg(command, "--static-eval-weight-overrides", temp_overrides)
    if temp_bot_seed is not None:
        command = _command_with_arg(command, "--seed", str(temp_bot_seed))
    temp["Name"] = temp_name if temp_bot_seed is None else f"{temp_name}_botseed{temp_bot_seed}"
    temp["External Command"] = command
    participants[1] = temp
    config["Participants"] = participants
    config["Level Seeds"] = [int(level_seed)]
    config["Game Count"] = 1
    config["Parallel Games"] = 1
    config["Balance Seats"] = True
    config["Concise"] = True
    config["Match Retry Limit"] = 0
    config["Stop When Best P(Strongest) At Least"] = 1.0
    config["Write Match Logs"] = bool(config.get("Write Match Logs", False))
    config["External Log Dir"] = str(output_dir / f"logs_seed{level_seed}_botseed{temp_bot_seed if temp_bot_seed is not None else 'default'}")
    config["Tournament Summary Log Path"] = str(output_dir / f"summary_seed{level_seed}_botseed{temp_bot_seed if temp_bot_seed is not None else 'default'}.log")
    config["Stats Report Path"] = str(output_dir / f"stats_seed{level_seed}_botseed{temp_bot_seed if temp_bot_seed is not None else 'default'}")
    path = output_dir / f"tournament_seed{level_seed}_botseed{temp_bot_seed if temp_bot_seed is not None else 'default'}.json"
    path.write_text(json.dumps(config, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


def _write_multiseed_tournament_config(
    base_config: dict[str, Any],
    *,
    output_dir: Path,
    level_seeds: list[int],
    temp_overrides: str,
    temp_bot_seed: int | None,
    temp_name: str,
    candidate_index: int,
    parallel_games: int | None,
) -> Path:
    config = dict(base_config)
    participants = [dict(row) for row in base_config.get("Participants", [])]
    if len(participants) < 2:
        raise RuntimeError("base tournament config requires at least two Participants")
    temp = dict(participants[1])
    command = _participant_command(temp)
    command = _command_with_arg(command, "--static-eval-variant", "experimental")
    command = _command_with_arg(command, "--static-eval-weight-overrides", temp_overrides)
    if temp_bot_seed is not None:
        command = _command_with_arg(command, "--seed", str(temp_bot_seed))
    temp["Name"] = temp_name if temp_bot_seed is None else f"{temp_name}_botseed{temp_bot_seed}"
    temp["External Command"] = command
    participants[1] = temp
    config["Participants"] = participants
    config["Level Seeds"] = [int(seed) for seed in level_seeds]
    config["Game Count"] = len(level_seeds)
    if parallel_games is not None:
        config["Parallel Games"] = max(1, int(parallel_games))
    config["Balance Seats"] = True
    config["Concise"] = True
    config["Match Retry Limit"] = 0
    config["Stop When Best P(Strongest) At Least"] = 1.0
    config["Write Match Logs"] = bool(config.get("Write Match Logs", False))
    label = f"candidate{candidate_index:03d}_botseed{temp_bot_seed if temp_bot_seed is not None else 'default'}"
    config["External Log Dir"] = str(output_dir / f"logs_{label}")
    config["Tournament Summary Log Path"] = str(output_dir / f"summary_{label}.log")
    config["Stats Report Path"] = str(output_dir / f"stats_{label}")
    path = output_dir / f"tournament_{label}.json"
    path.write_text(json.dumps(config, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


def _run_tournament(config_path: Path, *, java_exe: Path | None, timeout_sec: int) -> dict[str, Any]:
    java = str(java_exe) if java_exe is not None else str(Path(os.environ.get("JAVA_HOME", "")) / "bin" / "java.exe")
    if java_exe is None and not Path(java).exists():
        java = "java"
    command = [java, "-cp", os.pathsep.join([str(PROJECT_ROOT / "out"), str(game_json_jar())]), "Tournament", str(config_path)]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    return {
        "config": str(config_path),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "winner_none": "WINNER: NONE" in completed.stdout,
        "temp_winner": "WINNER: Temp" in completed.stdout or "WINNER: TempVariant" in completed.stdout,
        "baseline_winner": "WINNER: Baseline" in completed.stdout,
    }


_SEED_WINNER_RE = re.compile(r"SEED:(?P<seed>\d+)\s+WINNER:\s+(?P<winner>[^\r\n]+)")


def _parse_seed_winners(stdout: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for match in _SEED_WINNER_RE.finditer(stdout):
        winner = match.group("winner").strip()
        rows.append(
            {
                "seed": int(match.group("seed")),
                "winner": winner,
                "baseline_sweep": winner == "Baseline",
                "temp_not_swept": winner != "Baseline",
            }
        )
    return rows


def _verification_passed(rows: list[dict[str, Any]]) -> bool:
    return all(row["returncode"] == 0 and not row["baseline_winner"] for row in rows)


def _run_verification_suite(
    args: argparse.Namespace,
    *,
    base_config: dict[str, Any],
    output_dir: Path,
    candidate_overrides: str,
    abort_on_baseline: bool = False,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    bot_seeds = [None] + [int(seed) for seed in (args.verification_bot_seed or [])]
    verification = []
    for level_seed in [int(seed) for seed in args.level_seed]:
        for bot_seed in bot_seeds:
            config_path = _write_tournament_config(
                base_config,
                output_dir=output_dir,
                level_seed=level_seed,
                temp_overrides=candidate_overrides,
                temp_bot_seed=bot_seed,
                temp_name=args.temp_name,
            )
            result = _run_tournament(config_path, java_exe=args.java_exe, timeout_sec=args.timeout_sec)
            log_prefix = config_path.with_suffix("")
            (log_prefix.with_suffix(".stdout.log")).write_text(result["stdout"], encoding="utf-8")
            (log_prefix.with_suffix(".stderr.log")).write_text(result["stderr"], encoding="utf-8")
            result.pop("stdout")
            result.pop("stderr")
            result["level_seed"] = level_seed
            result["temp_bot_seed"] = bot_seed
            verification.append(result)
            if abort_on_baseline and result["baseline_winner"]:
                return verification
    return verification


def _run_candidate_search(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    candidate_overrides: list[str],
) -> dict[str, Any]:
    if args.base_tournament_config is None:
        raise RuntimeError("candidate search requires --base-tournament-config")
    if not args.level_seed:
        raise RuntimeError("candidate search requires at least one --level-seed")
    base_config = _load_json(_repo_relative(Path(args.base_tournament_config)))
    search_dir = output_dir / "candidate_search"
    search_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, overrides in enumerate(candidate_overrides):
        config_path = _write_multiseed_tournament_config(
            base_config,
            output_dir=search_dir,
            level_seeds=[int(seed) for seed in args.level_seed],
            temp_overrides=overrides,
            temp_bot_seed=args.search_bot_seed,
            temp_name=args.temp_name,
            candidate_index=index,
            parallel_games=args.search_parallel_games,
        )
        result = _run_tournament(config_path, java_exe=args.java_exe, timeout_sec=args.timeout_sec)
        log_prefix = config_path.with_suffix("")
        (log_prefix.with_suffix(".stdout.log")).write_text(result["stdout"], encoding="utf-8")
        (log_prefix.with_suffix(".stderr.log")).write_text(result["stderr"], encoding="utf-8")
        seed_results = _parse_seed_winners(result["stdout"])
        baseline_sweeps = [row["seed"] for row in seed_results if row["baseline_sweep"]]
        screen_passed = result["returncode"] == 0 and len(seed_results) == len(args.level_seed) and not baseline_sweeps
        candidate = {
            "candidate_index": index,
            "overrides": overrides,
            "config": str(config_path),
            "returncode": result["returncode"],
            "seed_results": seed_results,
            "seeds_checked": len(seed_results),
            "baseline_sweep_count": len(baseline_sweeps),
            "baseline_sweep_seeds": baseline_sweeps,
            "screen_passed": screen_passed,
            "passed": screen_passed,
        }
        should_verify = args.search_verification == "all" or (args.search_verification == "passing" and screen_passed)
        if should_verify:
            verification_dir = search_dir / f"verification_candidate{index:03d}"
            verification = _run_verification_suite(
                args,
                base_config=base_config,
                output_dir=verification_dir,
                candidate_overrides=overrides,
                abort_on_baseline=args.abort_search_verification_on_baseline,
            )
            verification_baseline_sweeps = sorted(
                {
                    int(row["level_seed"])
                    for row in verification
                    if row["returncode"] == 0 and row["baseline_winner"]
                }
            )
            candidate["verification"] = verification
            candidate["verification_passed"] = _verification_passed(verification)
            candidate["verification_baseline_sweep_seeds"] = verification_baseline_sweeps
            candidate["passed"] = bool(candidate["verification_passed"])
            candidate["baseline_sweep_count"] = len(verification_baseline_sweeps)
            candidate["baseline_sweep_seeds"] = verification_baseline_sweeps
        rows.append(candidate)
        if candidate["passed"] and args.stop_on_first_passing_candidate:
            break
    passed_rows = [row for row in rows if row.get("passed")]
    best_pool = passed_rows or rows
    best = min(
        best_pool,
        key=lambda row: (
            int(row["baseline_sweep_count"]),
            -int(row["seeds_checked"]),
            int(row["candidate_index"]),
        ),
        default=None,
    )
    return {
        "candidates": rows,
        "best_candidate": best,
        "passed": bool(best and best.get("passed")),
        "search_dir": str(search_dir),
    }


def _candidate_grid(args: argparse.Namespace) -> list[str]:
    candidates = [str(value) for value in (args.search_candidate_overrides or [])]
    if args.include_empty_candidate:
        candidates.insert(0, "")
    if args.candidate_preset == "unit_power":
        for value in args.unit_power_grid:
            candidates.append(f"military.unit_power={float(value):.12g}")
    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        out.append(candidate)
    return out


def run(args: argparse.Namespace) -> Path:
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    output_dir = _repo_relative(Path(args.output_dir)) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"run_id": run_id, "output_dir": str(output_dir)}

    search_candidates = _candidate_grid(args)
    if search_candidates:
        search_summary = _run_candidate_search(args, output_dir=output_dir, candidate_overrides=search_candidates)
        summary["candidate_search"] = search_summary
        if search_summary.get("passed") and search_summary.get("best_candidate") is not None:
            summary["candidate_overrides"] = search_summary["best_candidate"].get("overrides", "")

    if args.payload_dir is not None:
        compare_dir = compare_branches.run(
            SimpleNamespace(
                payload=[],
                payload_dir=_repo_relative(Path(args.payload_dir)),
                target=["baseline", "experimental"],
                positions=args.positions,
                simulations=args.simulations,
                batch_size=args.batch_size,
                top_k_actions=args.top_k_actions,
                max_actions=args.max_actions,
                seed=args.seed,
                c_puct=args.c_puct,
                native_static_exe=_repo_relative(Path(args.native_static_exe)),
                build_native_static_exe=args.build_native_static_exe,
                native_static_search_mode=args.native_static_search_mode,
                run_id="branch_compare",
                output_dir=output_dir,
            )
        )
        branchpoint_dir = branchpoint_dataset_builder.run(
            SimpleNamespace(
                positions_jsonl=compare_dir / "positions.jsonl",
                baseline_name="baseline",
                experimental_name="experimental",
                min_js_visit_bits=args.min_js_visit_bits,
                min_baseline_top_visit_share=args.min_baseline_top_visit_share,
                min_top_gap=args.min_top_gap,
                max_experimental_baseline_visit_share=args.max_experimental_baseline_visit_share,
                min_experimental_baseline_rank=args.min_experimental_baseline_rank,
                require_counterfactual_rescue=args.require_counterfactual_rescue,
                require_superior_seat_agreement=args.require_superior_seat_agreement,
                run_id="branchpoints",
                output_dir=output_dir,
            )
        )
        branchpoints = _load_jsonl(branchpoint_dir / "branchpoints.jsonl")
        if not branchpoints:
            summary.update({"status": "rejected", "reason": "no branchpoint candidates", "branch_compare": str(compare_dir), "branchpoints": str(branchpoint_dir)})
        else:
            payloads = [
                _repo_relative(Path(args.payload_dir)) / f"{row['label']}.json"
                for row in branchpoints[: int(args.max_branchpoints)]
            ]
            matrix_dir = root_child_value_matrix.run(
                SimpleNamespace(
                    payload=payloads,
                    payload_dir=_repo_relative(Path(args.payload_dir)),
                    target="experimental",
                    positions=len(payloads),
                    simulations=args.simulations,
                    batch_size=args.batch_size,
                    top_k_actions=args.top_k_actions,
                    max_actions=args.max_actions,
                    seed=args.seed,
                    c_puct=args.c_puct,
                    native_static_exe=_repo_relative(Path(args.native_static_exe)),
                    build_native_static_exe=args.build_native_static_exe,
                    native_static_search_mode=args.native_static_search_mode,
                    pair_limit=args.pair_limit,
                    run_id="root_child_matrix",
                    output_dir=output_dir,
                )
            )
            tuning_dir = tune_static_eval_weights.run(
                SimpleNamespace(
                    terms_csv=matrix_dir / "terms_matrix.csv",
                    constraints_jsonl=branchpoint_dir / "branchpoints.jsonl",
                    steps=args.tuner_steps,
                    learning_rate=args.tuner_learning_rate,
                    l2=args.tuner_l2,
                    l1=args.tuner_l1,
                    max_abs_delta=args.tuner_max_abs_delta,
                    max_relative_delta=args.tuner_max_relative_delta,
                    preserve_sign=args.tuner_preserve_sign,
                    min_weight=args.tuner_min_weight,
                    max_weight=args.tuner_max_weight,
                    min_satisfied_rate=args.tuner_min_satisfied_rate,
                    require_all_repairs=args.tuner_require_all_repairs,
                    protect_satisfied_before_weight=args.tuner_protect_satisfied_before_weight,
                    run_id="tuning",
                    output_dir=output_dir,
                )
            )
            tuning_summary = _load_json(tuning_dir / "summary.json")
            summary.update(
                {
                    "status": "candidate" if tuning_summary.get("accepted") else "rejected",
                    "branch_compare": str(compare_dir),
                    "branchpoints": str(branchpoint_dir),
                    "root_child_matrix": str(matrix_dir),
                    "tuning": str(tuning_dir),
                    "candidate_overrides": tuning_summary.get("changed_override_env") or tuning_summary.get("override_env", ""),
                    "tuning_summary": tuning_summary,
                }
            )

    candidate_overrides = str(args.candidate_overrides or summary.get("candidate_overrides") or "")
    should_run_final_verification = bool(args.candidate_overrides or args.payload_dir is not None)
    if args.base_tournament_config is not None and candidate_overrides:
        if not should_run_final_verification:
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
            print(f"wrote {output_dir}")
            return output_dir
        base_config = _load_json(_repo_relative(Path(args.base_tournament_config)))
        verification = _run_verification_suite(
            args,
            base_config=base_config,
            output_dir=output_dir / "verification",
            candidate_overrides=candidate_overrides,
        )
        summary["verification"] = verification
        summary["verification_passed"] = _verification_passed(verification)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(f"wrote {output_dir}")
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Automate temp static-eval variant proposal and verification.")
    parser.add_argument("--payload-dir", type=Path, default=None)
    parser.add_argument("--base-tournament-config", type=Path, default=None)
    parser.add_argument("--candidate-overrides", default="")
    parser.add_argument("--search-candidate-overrides", action="append", default=[])
    parser.add_argument("--candidate-preset", choices=("none", "unit_power"), default="none")
    parser.add_argument("--unit-power-grid", action="append", type=float, default=[0.25, 0.35, 0.5, 0.75, 1.0])
    parser.add_argument("--include-empty-candidate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--search-bot-seed", type=int, default=None)
    parser.add_argument("--search-parallel-games", type=int, default=None)
    parser.add_argument("--search-verification", choices=("none", "passing", "all"), default="passing")
    parser.add_argument("--abort-search-verification-on-baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-on-first-passing-candidate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--level-seed", action="append", type=int, default=[])
    parser.add_argument("--verification-bot-seed", action="append", type=int, default=[])
    parser.add_argument("--temp-name", default="TempVariant")
    parser.add_argument("--positions", type=int, default=None)
    parser.add_argument("--simulations", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k-actions", type=int, default=32)
    parser.add_argument("--max-actions", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--native-static-exe", type=Path, default=PROJECT_ROOT / "out" / "native" / "static_mcts_bot.exe")
    parser.add_argument("--build-native-static-exe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--native-static-search-mode", choices=("primitive", "turn-cmab"), default="primitive")
    parser.add_argument("--min-js-visit-bits", type=float, default=0.08)
    parser.add_argument("--min-baseline-top-visit-share", type=float, default=0.35)
    parser.add_argument("--min-top-gap", type=float, default=0.05)
    parser.add_argument("--max-experimental-baseline-visit-share", type=float, default=0.10)
    parser.add_argument("--min-experimental-baseline-rank", type=int, default=4)
    parser.add_argument("--require-counterfactual-rescue", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-superior-seat-agreement", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-branchpoints", type=int, default=16)
    parser.add_argument("--pair-limit", type=int, default=5)
    parser.add_argument("--tuner-steps", type=int, default=3000)
    parser.add_argument("--tuner-learning-rate", type=float, default=0.0001)
    parser.add_argument("--tuner-l2", type=float, default=0.5)
    parser.add_argument("--tuner-l1", type=float, default=0.0)
    parser.add_argument("--tuner-max-abs-delta", type=float, default=0.75)
    parser.add_argument("--tuner-max-relative-delta", type=float, default=0.5)
    parser.add_argument("--tuner-preserve-sign", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tuner-min-weight", type=float, default=None)
    parser.add_argument("--tuner-max-weight", type=float, default=None)
    parser.add_argument("--tuner-min-satisfied-rate", type=float, default=1.0)
    parser.add_argument("--tuner-require-all-repairs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tuner-protect-satisfied-before-weight", type=float, default=4.0)
    parser.add_argument("--java-exe", type=Path, default=None)
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = load_config_defaults(parser, default_config=DEFAULT_CONFIG if DEFAULT_CONFIG.exists() else None)
    has_candidate_search = bool(_candidate_grid(args))
    if args.payload_dir is None and not args.candidate_overrides and not has_candidate_search:
        raise RuntimeError("pass --payload-dir to propose a temp variant, --candidate-overrides to verify one, or search candidates")
    if args.base_tournament_config is not None and not args.level_seed:
        raise RuntimeError("verification/search requires at least one --level-seed")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
