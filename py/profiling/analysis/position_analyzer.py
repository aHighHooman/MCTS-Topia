from __future__ import annotations

import json
import os
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from profiling.analysis import ANALYSIS_VERSION
from profiling.analysis.actions import action_field, action_fingerprint, action_id, action_type, top_mass_keys
from profiling.analysis.distributions import rank_map
from profiling.analysis.schema import ActionAnalysis, PositionAnalysis
from search.native.cpp_extension import load_native_mcts_extension

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_NATIVE_STATIC_EXE = PROJECT_ROOT / "out" / "native" / "native_static_mcts_bot.exe"


def parse_target(spec: str) -> dict[str, str]:
    name, _, rest = spec.partition(":")
    if not rest and (name or spec) in {"baseline", "experimental"}:
        variant = name or spec
        return {"name": variant, "variant": variant, "mcts_impl": "native_static_exe"}
    target = {"name": name or spec, "variant": "baseline", "mcts_impl": "native_static_exe"}
    for part in rest.split(","):
        if not part:
            continue
        key, _, value = part.partition("=")
        if key and value:
            target[key.strip()] = value.strip()
    if target["name"] in {"baseline", "experimental"}:
        target["variant"] = target["name"]
    return target


def _target_weight_overrides(target: dict[str, str]) -> str:
    raw = str(target.get("weight_overrides") or target.get("static_eval_weight_overrides") or target.get("overrides") or "")
    return raw.replace(";", ",")


@contextmanager
def static_eval_target_env(target: dict[str, str]):
    previous_variant = os.environ.get("TRIBES_STATIC_EVAL_VARIANT")
    previous_overrides = os.environ.get("TRIBES_STATIC_EVAL_WEIGHT_OVERRIDES")
    os.environ["TRIBES_STATIC_EVAL_VARIANT"] = str(target.get("variant", "baseline"))
    overrides = _target_weight_overrides(target)
    if overrides:
        os.environ["TRIBES_STATIC_EVAL_WEIGHT_OVERRIDES"] = overrides
    else:
        os.environ.pop("TRIBES_STATIC_EVAL_WEIGHT_OVERRIDES", None)
    try:
        yield
    finally:
        if previous_variant is None:
            os.environ.pop("TRIBES_STATIC_EVAL_VARIANT", None)
        else:
            os.environ["TRIBES_STATIC_EVAL_VARIANT"] = previous_variant
        if previous_overrides is None:
            os.environ.pop("TRIBES_STATIC_EVAL_WEIGHT_OVERRIDES", None)
        else:
            os.environ["TRIBES_STATIC_EVAL_WEIGHT_OVERRIDES"] = previous_overrides


def _repo_path(path: Path | str | None, default: Path) -> Path:
    out = Path(path) if path is not None else default
    return out if out.is_absolute() else PROJECT_ROOT / out


def _native_static_build_inputs(build_script: Path) -> list[Path]:
    native_src = PROJECT_ROOT / "py" / "search" / "native" / "src"
    paths = [
        PROJECT_ROOT / "bots" / "native_static_mcts_bot.cpp",
        build_script,
    ]
    paths.extend(sorted(native_src.glob("*.cpp")))
    paths.extend(sorted(native_src.glob("*.hpp")))
    return [path for path in paths if path.exists()]


def _is_native_static_exe_fresh(exe: Path, inputs: list[Path]) -> bool:
    if not exe.exists():
        return False
    try:
        exe_mtime = exe.stat().st_mtime
    except OSError:
        return False
    return all(exe_mtime >= path.stat().st_mtime for path in inputs)


def _ensure_native_static_exe(native_static_exe: Path | str | None, build_native_static_exe: bool) -> Path:
    exe = _repo_path(native_static_exe, DEFAULT_NATIVE_STATIC_EXE)
    build_script = PROJECT_ROOT / "scripts" / "build_native_static_bot.ps1"
    build_inputs = _native_static_build_inputs(build_script)
    if _is_native_static_exe_fresh(exe, build_inputs):
        return exe
    if not build_native_static_exe:
        if exe.exists():
            raise RuntimeError(
                f"Native static executable is older than one or more build inputs: {exe}. "
                "Rebuild it or enable build_native_static_exe."
            )
        raise FileNotFoundError(f"Native static executable not found: {exe}")
    if not build_script.exists():
        raise FileNotFoundError(f"Native static executable build script not found: {build_script}")
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(build_script)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Failed to build native static executable.\n"
            f"stdout:\n{completed.stdout[-4000:]}\n"
            f"stderr:\n{completed.stderr[-4000:]}"
        )
    if not exe.exists():
        raise FileNotFoundError(f"Native static executable build completed but output was not found: {exe}")
    return exe


def _run_native_static_exe(
    payload: dict[str, Any],
    *,
    target: dict[str, str],
    simulations: int,
    batch_size: int,
    top_k_actions: int,
    max_actions: int,
    seed: int,
    c_puct: float,
    native_static_exe: Path | str | None,
    build_native_static_exe: bool,
    native_static_search_mode: str,
) -> dict[str, Any]:
    exe = _ensure_native_static_exe(native_static_exe, build_native_static_exe)
    command = [
        str(exe),
        "--search-mode",
        str(native_static_search_mode),
        "--simulations",
        str(int(simulations)),
        "--top-k-actions",
        str(int(top_k_actions)),
        "--max-actions",
        str(int(max_actions)),
        "--search-batch-size",
        str(int(batch_size)),
        "--c-puct",
        str(float(c_puct)),
        "--static-eval-variant",
        str(target.get("variant", "baseline")),
        "--seed",
        str(int(seed)),
        "--profile-json",
        "--deterministic",
    ]
    overrides = _target_weight_overrides(target)
    if overrides:
        command.extend(["--static-eval-weight-overrides", overrides])
    request = dict(payload)
    request["type"] = "action_request"
    completed = subprocess.run(
        command,
        input=json.dumps(request, separators=(",", ":")) + "\n",
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Native static executable analysis run failed.\n"
            f"command={command}\nstdout:\n{completed.stdout[-4000:]}\nstderr:\n{completed.stderr[-4000:]}"
        )
    output_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not output_lines:
        raise RuntimeError("Native static executable produced no stdout response.")
    response = json.loads(output_lines[-1])
    if not isinstance(response, dict):
        raise RuntimeError("Native static executable response was not a JSON object.")
    return response


def _static_breakdown(payload: dict[str, Any], max_actions: int) -> dict[str, Any] | None:
    extension = load_native_mcts_extension()
    fn = getattr(extension, "evaluate_static_breakdown", None) if extension is not None else None
    if fn is None:
        return None
    return dict(fn(payload, int(max_actions))).get("value_breakdown")


def _action_breakdown(payload: dict[str, Any], action_id: str, max_actions: int) -> dict[str, Any] | None:
    extension = load_native_mcts_extension()
    fn = getattr(extension, "evaluate_action_breakdown", None) if extension is not None else None
    if fn is None:
        return None
    res = fn(payload, action_id, int(max_actions))
    return dict(res).get("value_breakdown") if res else None


def _root_static_eval(payload: dict[str, Any], actions: list[dict[str, Any]], max_actions: int) -> tuple[list[float], float]:
    extension = load_native_mcts_extension()
    fn = getattr(extension, "evaluate_static_batch", None) if extension is not None else None
    if fn is None:
        raise RuntimeError("Native static evaluator extension is unavailable.")
    messages = [{"player_id": int(payload["player_id"]), "observation": payload["observation"], "actions": actions}]
    raw_evaluations = list(fn(messages, int(max_actions)))
    if len(raw_evaluations) != 1:
        raise RuntimeError(f"Static evaluator returned {len(raw_evaluations)} root evaluations.")
    raw = dict(raw_evaluations[0])
    priors = [float(value) for value in list(raw.get("priors", []))[: len(actions)]]
    if len(priors) != len(actions):
        raise RuntimeError(f"Static evaluator returned {len(priors)} priors for {len(actions)} actions.")
    total = sum(max(0.0, prior) for prior in priors)
    if total > 0.0:
        priors = [max(0.0, prior) / total for prior in priors]
    elif priors:
        priors = [1.0 / len(priors)] * len(priors)
    return priors, float(raw.get("value", 0.0))


def _native_extension_path() -> str:
    try:
        extension = load_native_mcts_extension()
    except Exception:
        return ""
    return str(getattr(extension, "__file__", "") or "")


def _try_root_static_eval(
    payload: dict[str, Any],
    actions: list[dict[str, Any]],
    max_actions: int,
) -> tuple[list[float], float, str, str, str]:
    try:
        priors, root_value = _root_static_eval(payload, actions, max_actions)
        return priors, root_value, "native_extension", "", ""
    except Exception as exc:
        return [0.0 for _ in actions], 0.0, "unavailable", str(exc), ""


def _response_action_id(row: dict[str, Any]) -> str:
    return str(row.get("action_id") or row.get("actionId") or row.get("id") or "")


def _selected_action_id(response: dict[str, Any]) -> str:
    return str(response.get("actionId") or response.get("action_id") or response.get("selected_action_id") or "")


def analyze_position(
    payload: dict[str, Any],
    *,
    payload_hash: str,
    label: str,
    target: dict[str, str],
    simulations: int,
    batch_size: int,
    top_k_actions: int,
    max_actions: int,
    seed: int = 0,
    c_puct: float = 1.5,
    include_breakdown: bool = True,
    native_static_exe: Path | str | None = None,
    build_native_static_exe: bool = True,
    native_static_search_mode: str = "primitive",
) -> PositionAnalysis:
    with static_eval_target_env(target):
        actions = list(payload.get("actions", [])) if max_actions < 0 else list(payload.get("actions", []))[:max_actions]
        warnings: list[str] = []
        priors, root_value, static_eval_source, static_eval_error, native_extension_path = _try_root_static_eval(
            payload,
            actions,
            int(max_actions),
        )
        if static_eval_error:
            warnings.append(f"root static eval unavailable: {static_eval_error}")
        started_at = time.perf_counter()
        if str(target.get("mcts_impl", "native_static_exe")) != "native_static_exe":
            raise ValueError("Static position analysis only supports mcts_impl=native_static_exe.")
        exe_path = _ensure_native_static_exe(native_static_exe, build_native_static_exe)
        response = _run_native_static_exe(
            payload,
            target=target,
            simulations=simulations,
            batch_size=batch_size,
            top_k_actions=top_k_actions,
            max_actions=max_actions,
            seed=seed,
            c_puct=c_puct,
            native_static_exe=exe_path,
            build_native_static_exe=False,
            native_static_search_mode=native_static_search_mode,
        )
        profile = response.get("_profile") if isinstance(response.get("_profile"), dict) else {}
        root_stats = list(profile.get("root_action_stats", [])) if isinstance(profile, dict) else []
        if int(simulations) > 0 and not root_stats:
            raise RuntimeError(
                "Native static executable response did not include _profile.root_action_stats. "
                "Make sure --profile-json is supported by the executable being analyzed."
            )
        stats_by_id = {_response_action_id(row): row for row in root_stats if isinstance(row, dict) and _response_action_id(row)}
        visit_distribution = {
            _response_action_id(row): float(row.get("visit_share", 0.0) or 0.0)
            for row in root_stats
            if isinstance(row, dict) and _response_action_id(row)
        }
        selected_action_id = _selected_action_id(response)
        search_sec = time.perf_counter() - started_at
        visit_scores = [float(visit_distribution.get(action_id(action), 0.0)) for action in actions]
        prior_scores = [float(priors[index]) if index < len(priors) else 0.0 for index in range(len(actions))]
        visit_ranks = rank_map(visit_scores)
        prior_ranks = rank_map(prior_scores)
        top95 = top_mass_keys({action_id(action): visit_scores[index] for index, action in enumerate(actions)})
        selected_action = next((action for action in actions if action_id(action) == selected_action_id), {})
        action_rows: list[ActionAnalysis] = []
        for index, action in enumerate(actions):
            aid = action_id(action)
            stat = stats_by_id.get(aid, {})
            v_rank = visit_ranks.get(index, 0)
            act_breakdown = None
            if v_rank in {1, 2, 3} and include_breakdown:
                act_breakdown = _action_breakdown(payload, aid, max_actions)
            action_rows.append(
                ActionAnalysis(
                    action_id=aid,
                    action_fingerprint=action_fingerprint(action),
                    action_index=index,
                    action_type=action_type(action),
                    prior=prior_scores[index],
                    prior_rank=prior_ranks.get(index, 0),
                    visits=int(stat["visits"]) if "visits" in stat else None,
                    visit_share=float(stat.get("visit_share", visit_scores[index])),
                    visit_rank=v_rank,
                    q_mean=float(stat["q_mean"]) if "q_mean" in stat else None,
                    value_sum=float(stat["value_sum"]) if "value_sum" in stat else None,
                    in_top95=aid in top95 or aid == selected_action_id,
                    unit_id=action_field(action, "unit_id", "unitId", "u"),
                    city_id=action_field(action, "city_id", "cityId", "c"),
                    x=action_field(action, "x"),
                    y=action_field(action, "y"),
                    target=action_field(action, "target_unit_id", "targetUnitId", "target_id", "targetId", "tu"),
                    value_breakdown=act_breakdown,
                )
            )
        return PositionAnalysis(
            analysis_version=ANALYSIS_VERSION,
            payload_hash=payload_hash,
            label=label,
            target_name=str(target.get("name", "")),
            evaluator="static",
            mcts_impl=str(target.get("mcts_impl", "native_static_exe")),
            static_eval_variant=str(target.get("variant", "baseline")),
            seed=int(seed),
            simulations=int(simulations),
            c_puct=float(c_puct),
            top_k_actions=int(top_k_actions),
            max_actions=int(max_actions),
            root_value=root_value,
            selected_action_id=selected_action_id,
            selected_action_fingerprint=action_fingerprint(selected_action) if selected_action else "",
            action_count_raw=len(list(payload.get("actions", []))),
            action_count_analyzed=len(actions),
            search_sec=search_sec,
            actions=action_rows,
            value_breakdown=_static_breakdown(payload, int(max_actions)) if include_breakdown and static_eval_source != "unavailable" else None,
            static_eval_source=static_eval_source,
            static_eval_error=static_eval_error,
            profile_validated=bool(root_stats) or int(simulations) <= 0,
            native_exe_path=str(exe_path),
            native_extension_path=native_extension_path,
            warnings=warnings,
        )


def position_to_dict(position: PositionAnalysis) -> dict[str, Any]:
    return asdict(position)
