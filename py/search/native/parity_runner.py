from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PY_ROOT = Path(__file__).resolve().parents[2]
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from nn.encoding import normalize_message
from search.native.cpp_extension import load_native_mcts_extension


REPO_ROOT = Path(__file__).resolve().parents[3]


class ParityFailure(AssertionError):
    pass


def _run_java_oracle(fixture: Path, player: int | None, compile_java: bool) -> dict[str, Any]:
    java_exe = _java_executable()
    if compile_java:
        subprocess.run(
            [
                "javac",
                "-cp",
                str(REPO_ROOT / "lib" / "json.jar"),
                "-sourcepath",
                str(REPO_ROOT / "src"),
                "-d",
                str(REPO_ROOT / "out"),
                str(REPO_ROOT / "src" / "core" / "game" / "NativeParityOracle.java"),
            ],
            cwd=REPO_ROOT,
            check=True,
        )

    classpath = f"{REPO_ROOT / 'out'};{REPO_ROOT / 'lib' / 'json.jar'}"
    command = [
        java_exe,
        "-cp",
        classpath,
        "core.game.NativeParityOracle",
        "--fixture",
        str(fixture),
    ]
    if player is not None:
        command.extend(["--player", str(player)])
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


def _java_executable() -> str:
    javac = shutil.which("javac")
    if javac:
        candidate = Path(javac).with_name("java.exe" if sys.platform.startswith("win") else "java")
        if candidate.exists():
            return str(candidate)
    return "java"


def _canonical_state(state: dict[str, Any], player_id: int) -> dict[str, Any]:
    message = dict(state)
    message.setdefault("player_id", player_id)
    normalized = normalize_message(message)
    observation = dict(normalized.get("observation", {}))
    return {
        "observation": observation,
        "actions": [_canonical_action(action) for action in normalized.get("actions", [])],
        "is_terminal": bool(normalized.get("is_terminal", False)),
        "active_player_id": int(
            normalized.get("active_player_id", observation.get("active_player_id", player_id)) or 0
        ),
        "winner_id": normalized.get("winner_id"),
        "final_scores": normalized.get("final_scores"),
        "ranking": normalized.get("ranking", observation.get("ranking")),
        "normalized_terminal_reward": normalized.get("normalized_terminal_reward"),
    }


def _canonical_action(action: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("type", "u", "c", "p", "x", "y", "tu", "tc", "ct", "ut", "bt", "rt", "b", "tech", "tp", "bonus"):
        value = action.get(key)
        if value is None:
            continue
        if key in {"u", "c", "p", "tu", "tc", "tp"} and int(value or 0) == 0:
            continue
        out[key] = value
    return out


def _first_diff(left: Any, right: Any, path: str = "$") -> tuple[str, Any, Any] | None:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)) and not isinstance(left, bool) and not isinstance(right, bool):
        return None if float(left) == float(right) else (path, left, right)
    if type(left) is not type(right):
        return path, left, right
    if isinstance(left, dict):
        left_keys = set(left)
        right_keys = set(right)
        for key in sorted(left_keys | right_keys):
            if key not in left and right.get(key) is None:
                continue
            if key not in right and left.get(key) is None:
                continue
            if key not in left:
                return f"{path}.{key}", None, right[key]
            if key not in right:
                return f"{path}.{key}", left[key], None
            diff = _first_diff(left[key], right[key], f"{path}.{key}")
            if diff is not None:
                return diff
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}.length", len(left), len(right)
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            diff = _first_diff(left_item, right_item, f"{path}[{index}]")
            if diff is not None:
                return diff
        return None
    if left != right:
        return path, left, right
    return None


def _dump_trace(trace_dir: Path | None, action_id: str, root: dict[str, Any], java_child: dict[str, Any], cpp_child: dict[str, Any] | None) -> None:
    if trace_dir is None:
        return
    trace_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "root": root,
        "java_child": java_child,
        "cpp_child": cpp_child,
    }
    (trace_dir / f"{action_id}.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_parity(args: argparse.Namespace) -> int:
    oracle = _run_java_oracle(args.fixture, args.player, not args.no_compile_java)
    if int(oracle.get("protocol_version", -1)) != 1:
        raise RuntimeError(f"Unsupported Java oracle protocol version: {oracle.get('protocol_version')}")

    player_id = int(oracle["player_id"])
    java_root_state = dict(oracle["root"])
    java_root = normalize_message({"player_id": player_id, **java_root_state})
    root_actions = list(java_root.get("actions", []))
    id_to_action = {str(action.get("id")): action for action in root_actions}

    extension = load_native_mcts_extension()
    if extension is None:
        raise RuntimeError("Native MCTS extension is unavailable.")

    failures = 0
    for child in oracle.get("children", []):
        action_id = str(child.get("action_id", ""))
        if args.action_id and action_id != args.action_id:
            continue
        action = id_to_action.get(action_id, {})
        action_type = str(action.get("type", action.get("t", "")))
        action_index = int(child.get("action_index", -1))
        state_id = str(child.get("source_state_id", "root"))

        if not bool(child.get("ok", False)):
            failures += 1
            print(
                f"PARITY FAIL state={state_id} action={action_id} type={action_type} "
                f"path=$.java_oracle java={child.get('error')} cpp=<not-run>",
                file=sys.stderr,
            )
            if not args.keep_going:
                return 1
            continue

        try:
            cpp_root = copy.deepcopy(java_root)
            tree = extension.NativeMCTS(
                cpp_root,
                [action_index],
                [1.0],
                0.0,
                bool(java_root.get("is_terminal", False)),
                int(args.seed),
                int(args.max_actions),
            )
            selection = dict(tree.select_leaf(1, 1.0))
            cpp_payload = dict(selection.get("leaf_payload") or {})
            java_canonical = _canonical_state(dict(child["state"]), player_id)
            cpp_canonical = _canonical_state(cpp_payload, player_id)
            diff = _first_diff(java_canonical, cpp_canonical)
            _dump_trace(args.trace_dir, action_id, cpp_root, dict(child["state"]), cpp_payload)
            if diff is not None:
                path, java_value, cpp_value = diff
                raise ParityFailure(
                    f"PARITY FAIL state={state_id} action={action_id} type={action_type} "
                    f"path={path} java={json.dumps(java_value, sort_keys=True)} "
                    f"cpp={json.dumps(cpp_value, sort_keys=True)}"
                )
            print(f"PARITY OK state={state_id} action={action_id} type={action_type}")
        except Exception as exc:
            failures += 1
            if not isinstance(exc, ParityFailure):
                _dump_trace(args.trace_dir, action_id, java_root, dict(child.get("state", {})), None)
                print(
                    f"PARITY FAIL state={state_id} action={action_id} type={action_type} "
                    f"path=$.cpp_exception java=<state> cpp={exc}",
                    file=sys.stderr,
                )
            else:
                print(str(exc), file=sys.stderr)
            if not args.keep_going:
                return 1

    return 1 if failures else 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diff Java oracle children against native strict C++ transitions.")
    parser.add_argument("--fixture", type=Path, required=True, help="Path to a saved game.json fixture.")
    parser.add_argument("--player", type=int, default=None, help="Observer player id. Defaults to the fixture active player.")
    parser.add_argument("--action-id", default=None, help="Only check one root action id.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after the first parity failure.")
    parser.add_argument("--no-compile-java", action="store_true", help="Skip javac before running the Java oracle.")
    parser.add_argument("--seed", type=int, default=7, help="Native tree seed.")
    parser.add_argument("--max-actions", type=int, default=256, help="Native max action cap.")
    parser.add_argument("--trace-dir", type=Path, default=None, help="Optional directory for root/java/cpp JSON traces.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run_parity(parse_args(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
