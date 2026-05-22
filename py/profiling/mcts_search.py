from __future__ import annotations

import argparse
import cProfile
import csv
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import pstats
import random
import sys
import time
from collections import Counter
from typing import Any, Callable

import torch

PY_ROOT = Path(__file__).resolve().parents[1]
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

DEFAULT_AUTORESEARCH_CHECKPOINT = PY_ROOT / "profiling" / "mcts_search_profiling_random_init_model.pt"

from nn.encoding import EncodedObservation, encode_observation
from nn.model import HybridPolicyValueNet
from profiling.config import load_config_defaults
from search.config import HybridAgentConfig
from search.native import mcts as native_mcts
from search.native import static_mcts as native_static_mcts
from search.native.cpp_extension import load_native_mcts_extension

_STATIC_TREE_DEPTH_SUM = 0
_STATIC_TREE_MAX_DEPTH = 0
_STATIC_TREE_SELECTED_PATHS = 0
_STATIC_TREE_EXPANDED_NODE_IDS: set[int] = set()


@dataclass
class TimingRow:
    name: str
    calls: int = 0
    total_sec: float = 0.0
    child_sec: float = 0.0
    items: int = 0

    @property
    def self_sec(self) -> float:
        return max(0.0, self.total_sec - self.child_sec)


@dataclass
class TimingCollector:
    rows: dict[str, TimingRow] = field(default_factory=dict)

    def add(self, name: str, elapsed: float, *, child_sec: float = 0.0, items: int = 0) -> None:
        row = self.rows.setdefault(name, TimingRow(name))
        row.calls += 1
        row.total_sec += elapsed
        row.child_sec += child_sec
        row.items += items

    def sorted_rows(self) -> list[TimingRow]:
        return sorted(self.rows.values(), key=lambda row: row.total_sec, reverse=True)


@dataclass
class SearchStats:
    mode: str
    elapsed_sec: float
    simulations: int = 0
    selected_paths: int = 0
    expanded_nodes: int = 0
    eval_batches: int = 0
    eval_positions: int = 0
    eval_cache_hits: int = 0
    eval_cache_size: int = 0
    depth_sum: int = 0
    max_depth: int = 0

    @property
    def average_depth(self) -> float:
        return float(self.depth_sum) / max(1, self.selected_paths)


@dataclass
class PayloadCase:
    label: str
    payload: dict[str, Any]
    seed: int | None = None
    path: Path | None = None


@dataclass
class BranchingCollector:
    root_action_counts: list[int] = field(default_factory=list)
    eval_action_counts: list[int] = field(default_factory=list)
    root_action_types: Counter[str] = field(default_factory=Counter)
    eval_action_types: Counter[str] = field(default_factory=Counter)
    root_visit_by_type: Counter[str] = field(default_factory=Counter)
    root_visit_entropy_sum: float = 0.0
    root_visit_samples: int = 0
    selected_action_types: Counter[str] = field(default_factory=Counter)

    def add_root(self, payload: dict[str, Any]) -> None:
        actions = _payload_actions(payload)
        self.root_action_counts.append(len(actions))
        self.root_action_types.update(_action_type(action) for action in actions)

    def add_eval_messages(self, messages: list[dict[str, Any]]) -> None:
        for message in messages:
            actions = _payload_actions(message)
            self.eval_action_counts.append(len(actions))
            self.eval_action_types.update(_action_type(action) for action in actions)

    def add_result(self, payload: dict[str, Any], result: native_mcts.SearchResult | None) -> None:
        if result is None:
            return
        action_by_id = {str(action.get("id")): action for action in _payload_actions(payload)}
        selected = action_by_id.get(str(result.action_id))
        if selected is not None:
            self.selected_action_types[_action_type(selected)] += 1
        for action_id, visit_share in result.visit_distribution.items():
            share = max(0.0, float(visit_share))
            if share <= 0.0:
                continue
            action = action_by_id.get(str(action_id))
            self.root_visit_by_type[_action_type(action) if action is not None else "UNKNOWN"] += share
        if result.visit_distribution:
            entropy = -sum(
                max(0.0, float(share)) * math.log2(max(1e-12, float(share)))
                for share in result.visit_distribution.values()
                if float(share) > 0.0
            )
            self.root_visit_entropy_sum += entropy
            self.root_visit_samples += 1


def _sync_if_needed(device: torch.device | str) -> None:
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device_obj)


def _time_call(
    collector: TimingCollector,
    name: str,
    fn: Callable[[], Any],
    *,
    device: torch.device | str,
    items: int = 0,
    sync_cuda: bool = True,
) -> tuple[Any, float]:
    started_at = time.perf_counter()
    result = fn()
    if sync_cuda:
        _sync_if_needed(device)
    elapsed = time.perf_counter() - started_at
    collector.add(name, elapsed, items=items)
    return result, elapsed


def _stack_encoded(items: list[EncodedObservation]) -> EncodedObservation:
    max_units = max((item.unit_features.shape[1] for item in items), default=0)
    max_cities = max((item.city_features.shape[1] for item in items), default=0)
    max_actions = max((item.action_features.shape[1] for item in items), default=0)

    def pad_slots(tensor: torch.Tensor, size: int) -> torch.Tensor:
        if tensor.shape[1] >= size:
            return tensor
        pad_shape = list(tensor.shape)
        pad_shape[1] = size - tensor.shape[1]
        return torch.cat([tensor, tensor.new_zeros(pad_shape)], dim=1)

    return EncodedObservation(
        board=torch.cat([item.board for item in items], dim=0),
        unit_features=torch.cat([pad_slots(item.unit_features, max_units) for item in items], dim=0),
        unit_mask=torch.cat([pad_slots(item.unit_mask, max_units) for item in items], dim=0),
        city_features=torch.cat([pad_slots(item.city_features, max_cities) for item in items], dim=0),
        city_mask=torch.cat([pad_slots(item.city_mask, max_cities) for item in items], dim=0),
        action_features=torch.cat([pad_slots(item.action_features, max_actions) for item in items], dim=0),
        action_mask=torch.cat([pad_slots(item.action_mask, max_actions) for item in items], dim=0),
        scalar_features=torch.cat([item.scalar_features for item in items], dim=0),
        action_ids=[],
    )


def _install_timed_evaluator(collector: TimingCollector, branching: BranchingCollector) -> Callable[..., list[native_mcts._Evaluation]]:
    original = native_mcts._evaluate_messages

    def timed_evaluate_messages(
        messages: list[dict[str, Any]],
        model: HybridPolicyValueNet,
        model_cfg,
        device: torch.device | str,
        belief_snapshot: Any | None = None,
    ) -> list[native_mcts._Evaluation]:
        if not messages:
            return []
        branching.add_eval_messages(messages)

        child_sec = 0.0
        if belief_snapshot is not None:
            messages, elapsed = _time_call(
                collector,
                "nn_eval.belief_annotate",
                lambda: [belief_snapshot.annotate_without_update(message) for message in messages],
                device=device,
                items=len(messages),
                sync_cuda=False,
            )
            child_sec += elapsed

        encoded_items: list[EncodedObservation] = []
        for message in messages:
            encoded, elapsed = _time_call(
                collector,
                "nn_eval.encode_observation",
                lambda message=message: encode_observation(message, model_cfg, compact=True),
                device=device,
                items=1,
                sync_cuda=False,
            )
            child_sec += elapsed
            encoded_items.append(encoded)

        if len(encoded_items) == 1:
            batch = encoded_items[0]
        else:
            batch, elapsed = _time_call(
                collector,
                "nn_eval.stack_encoded_batch",
                lambda: _stack_encoded(encoded_items),
                device=device,
                items=len(encoded_items),
                sync_cuda=False,
            )
            child_sec += elapsed
        batch, elapsed = _time_call(
            collector,
            "nn_eval.transfer_batch",
            lambda: batch.to(device),
            device=device,
            items=len(messages),
        )
        child_sec += elapsed

        def forward() -> Any:
            with torch.inference_mode():
                return model(batch)

        output, elapsed = _time_call(collector, "nn_eval.model_forward", forward, device=device, items=len(messages))
        child_sec += elapsed

        def postprocess() -> list[native_mcts._Evaluation]:
            action_counts = [len(encoded.action_ids) for encoded in encoded_items]
            max_action_count = max(action_counts, default=0)
            if max_action_count:
                logits = output.policy_logits[:, :max_action_count]
                mask = torch.arange(max_action_count, device=logits.device).unsqueeze(0) >= torch.tensor(
                    action_counts,
                    device=logits.device,
                ).unsqueeze(1)
                probs_rows = torch.softmax(logits.masked_fill(mask, float("-inf")), dim=-1).detach().cpu().tolist()
            else:
                probs_rows = [[] for _ in encoded_items]
            values = output.value.detach().flatten().cpu().tolist()
            evaluations: list[native_mcts._Evaluation] = []
            for index, action_count in enumerate(action_counts):
                evaluations.append(native_mcts._Evaluation([float(prob) for prob in probs_rows[index][:action_count]], float(values[index])))
            return evaluations

        evaluations, elapsed = _time_call(collector, "nn_eval.postprocess_output", postprocess, device=device, items=len(messages))
        child_sec += elapsed
        collector.add("nn_eval.total", child_sec, child_sec=child_sec, items=len(messages))
        return evaluations

    native_mcts._evaluate_messages = timed_evaluate_messages
    return original


def _install_timed_static_evaluator(collector: TimingCollector, branching: BranchingCollector) -> Callable[..., list[native_mcts._Evaluation]]:
    original = native_static_mcts._evaluate_static_messages

    def timed_evaluate_static_messages(
        messages: list[dict[str, Any]],
        max_actions: int,
    ) -> list[native_mcts._Evaluation]:
        if not messages:
            return []
        branching.add_eval_messages(messages)
        started_at = time.perf_counter()
        result = original(messages, max_actions)
        elapsed = time.perf_counter() - started_at
        collector.add("static_eval.total", elapsed, items=len(messages))
        return result

    native_static_mcts._evaluate_static_messages = timed_evaluate_static_messages
    return original


def _install_timed_tree(extension: object, collector: TimingCollector, device: torch.device | str) -> type:
    original_cls = extension.NativeMCTS

    class TimedNativeMCTS:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._tree, _elapsed = _time_call(
                collector,
                "native_tree.construct",
                lambda: original_cls(*args, **kwargs),
                device=device,
                sync_cuda=False,
            )

        def add_root_dirichlet_noise(self, *args: Any, **kwargs: Any) -> Any:
            result, _elapsed = _time_call(
                collector,
                "native_tree.add_root_dirichlet_noise",
                lambda: self._tree.add_root_dirichlet_noise(*args, **kwargs),
                device=device,
                sync_cuda=False,
            )
            return result

        def select_leaf_batch(self, *args: Any, **kwargs: Any) -> Any:
            _sync_if_needed(device)
            started_at = time.perf_counter()
            result = list(self._tree.select_leaf_batch(*args, **kwargs))
            _sync_if_needed(device)
            collector.add("native_tree.select_leaf_batch", time.perf_counter() - started_at, items=len(result))
            return result

        def select_leaf_batch_compact(self, *args: Any, **kwargs: Any) -> Any:
            _sync_if_needed(device)
            started_at = time.perf_counter()
            result = list(self._tree.select_leaf_batch_compact(*args, **kwargs))
            _sync_if_needed(device)
            collector.add("native_tree.select_leaf_batch_compact", time.perf_counter() - started_at, items=len(result))
            return result

        def select_leaf_batch_evals_only(self, *args: Any, **kwargs: Any) -> Any:
            global _STATIC_TREE_DEPTH_SUM, _STATIC_TREE_MAX_DEPTH, _STATIC_TREE_SELECTED_PATHS
            started_at = time.perf_counter()
            result = list(self._tree.select_leaf_batch_evals_only(*args, **kwargs))
            frontier = int(args[0]) if args else len(result)
            collector.add("native_tree.select_leaf_batch_evals_only", time.perf_counter() - started_at, items=frontier)
            batch_depth_sum, batch_max_depth = self._tree.last_batch_stats()
            _STATIC_TREE_DEPTH_SUM += int(batch_depth_sum)
            _STATIC_TREE_MAX_DEPTH = max(_STATIC_TREE_MAX_DEPTH, int(batch_max_depth))
            _STATIC_TREE_SELECTED_PATHS += len(result)
            return result

        def select_leaf_batches_evals_only(self, *args: Any, **kwargs: Any) -> Any:
            global _STATIC_TREE_DEPTH_SUM, _STATIC_TREE_MAX_DEPTH, _STATIC_TREE_SELECTED_PATHS
            started_at = time.perf_counter()
            result, completed = self._tree.select_leaf_batches_evals_only(*args, **kwargs)
            result = list(result)
            collector.add("native_tree.select_leaf_batches_evals_only", time.perf_counter() - started_at, items=int(completed))
            batch_depth_sum, batch_max_depth = self._tree.last_batch_stats()
            _STATIC_TREE_DEPTH_SUM += int(batch_depth_sum)
            _STATIC_TREE_MAX_DEPTH = max(_STATIC_TREE_MAX_DEPTH, int(batch_max_depth))
            _STATIC_TREE_SELECTED_PATHS += int(completed)
            return result, int(completed)

        def expand(self, *args: Any, **kwargs: Any) -> Any:
            global _STATIC_TREE_EXPANDED_NODE_IDS
            result, _elapsed = _time_call(
                collector,
                "native_tree.expand",
                lambda: self._tree.expand(*args, **kwargs),
                device=device,
                items=1,
                sync_cuda=False,
            )
            try:
                _STATIC_TREE_EXPANDED_NODE_IDS.add(int(result))
            except (TypeError, ValueError):
                pass
            return result

        def complete_selected_paths(self, *args: Any, **kwargs: Any) -> Any:
            result, _elapsed = _time_call(
                collector,
                "native_tree.complete_selected_paths",
                lambda: self._tree.complete_selected_paths(*args, **kwargs),
                device=device,
                sync_cuda=False,
            )
            return result

        def root_visit_distribution(self, *args: Any, **kwargs: Any) -> Any:
            result, _elapsed = _time_call(
                collector,
                "native_tree.root_visit_distribution",
                lambda: self._tree.root_visit_distribution(*args, **kwargs),
                device=device,
                sync_cuda=False,
            )
            return result

        def __getattr__(self, name: str) -> Any:
            return getattr(self._tree, name)

    extension.NativeMCTS = TimedNativeMCTS
    return original_cls


def _save_random_initialized_checkpoint(model: HybridPolicyValueNet, checkpoint: Path) -> None:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "metadata": {
                "source": "profiling.mcts_search",
                "description": "Random initialized model checkpoint used by the MCTS search profiler.",
            },
        },
        tmp_path,
    )
    tmp_path.replace(checkpoint)


def _load_checkpoint(model: HybridPolicyValueNet, checkpoint: Path, device: torch.device) -> bool:
    if not checkpoint.exists():
        return False
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = payload.get("model", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    return True


def _load_or_initialize_checkpoint(
    model: HybridPolicyValueNet,
    checkpoint: Path,
    device: torch.device,
    *,
    allow_reinitialize: bool,
) -> str:
    if checkpoint.exists():
        try:
            _load_checkpoint(model, checkpoint, device)
        except RuntimeError as exc:
            if not allow_reinitialize:
                raise
            _save_random_initialized_checkpoint(model, checkpoint)
            model.to(device)
            return f"reinitialized_incompatible:{checkpoint}:{exc.__class__.__name__}"
        return f"loaded:{checkpoint}"

    if not allow_reinitialize:
        model.to(device)
        return f"missing_random_init:{checkpoint}"

    _save_random_initialized_checkpoint(model, checkpoint)
    model.to(device)
    return f"initialized_missing:{checkpoint}"


def _synthetic_payload() -> dict[str, Any]:
    size = 4
    return {
        "player_id": 0,
        "observation": {
            "active_player_id": 0,
            "tick": 0,
            "can_end_turn": True,
            "board": {
                "size": size,
                "tiles": [
                    [
                        {
                            "x": x,
                            "y": y,
                            "visible": True,
                            "explored": True,
                            "terrain": "PLAIN",
                            "unit_id": 1 if (x, y) == (1, 1) else 0,
                        }
                        for x in range(size)
                    ]
                    for y in range(size)
                ],
            },
            "units": [
                {
                    "id": 1,
                    "tribe_id": 0,
                    "city_id": 10,
                    "type": "WARRIOR",
                    "x": 1,
                    "y": 1,
                    "current_hp": 10,
                    "max_hp": 10,
                    "kills": 0,
                    "is_veteran": False,
                    "status": "FRESH",
                    "is_hidden": False,
                }
            ],
            "cities": [],
            "tribes": [
                {"id": 0, "stars": 0, "score": 0, "researched_tech_ids": [], "cities": [], "extra_units": []},
                {"id": 1, "stars": 0, "score": 0, "researched_tech_ids": [], "cities": [], "extra_units": []},
            ],
        },
        "actions": [
            {"id": "move_e", "type": "MOVE", "unit_id": 1, "destination": {"x": 2, "y": 1}, "x": 2, "y": 1},
            {"id": "move_s", "type": "MOVE", "unit_id": 1, "destination": {"x": 1, "y": 2}, "x": 1, "y": 2},
            {"id": "road", "type": "BUILD_ROAD", "tribe_id": 0, "position": {"x": 1, "y": 2}, "x": 1, "y": 2},
            {"id": "end", "type": "END_TURN"},
        ],
    }


def _load_payload(path: Path | None) -> dict[str, Any]:
    if path is None:
        return _synthetic_payload()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        if not payload:
            raise ValueError(f"payload list is empty: {path}")
        payload = payload[0]
    if not isinstance(payload, dict):
        raise TypeError(f"payload must be a JSON object or a non-empty list of objects: {path}")
    return payload



_CAPTURE_BOT_SOURCE = 'from __future__ import annotations\n\nimport argparse\nimport json\nimport sys\nfrom pathlib import Path\nfrom typing import Any\n\n\ndef _pick_action_id(actions: list[dict[str, Any]]) -> str | None:\n    if not actions:\n        return None\n    for action in actions:\n        action_type = str(action.get("type") or action.get("t") or "").upper()\n        if action_type == "END_TURN":\n            return str(action.get("id"))\n    return str(actions[0].get("id"))\n\n\ndef main() -> int:\n    parser = argparse.ArgumentParser(description="Capture the first real self-play action_request payload.")\n    parser.add_argument("--output", type=Path, required=True)\n    parser.add_argument("--py-root", type=Path, required=True)\n    args = parser.parse_args()\n\n    py_root = str(args.py_root.resolve())\n    if py_root not in sys.path:\n        sys.path.insert(0, py_root)\n\n    from nn.belief import BeliefTracker\n    from nn.encoding import normalize_message\n\n    tracker = BeliefTracker()\n    wrote_payload = False\n    args.output.parent.mkdir(parents=True, exist_ok=True)\n\n    for raw_line in sys.stdin:\n        line = raw_line.strip()\n        if not line:\n            continue\n        try:\n            message = json.loads(line)\n        except json.JSONDecodeError:\n            continue\n\n        msg_type = message.get("type")\n        if msg_type == "action_request":\n            normalized = tracker.annotate(normalize_message(message))\n            payload = {\n                "player_id": int(normalized.get("player_id", 0) or 0),\n                "observation": normalized["observation"],\n                "actions": list(normalized.get("actions", []) or []),\n            }\n            if not wrote_payload:\n                tmp = args.output.with_suffix(args.output.suffix + ".tmp")\n                tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")\n                tmp.replace(args.output)\n                wrote_payload = True\n\n            action_ids = [str(action.get("id")) for action in payload["actions"]]\n            selected = _pick_action_id(payload["actions"])\n            print(json.dumps({"actionId": selected, "rankedActionIds": action_ids}), flush=True)\n            continue\n\n        if msg_type == "game_over":\n            print(json.dumps({"ok": True}), flush=True)\n            break\n\n        print(json.dumps({"error": f"unsupported message type: {msg_type}"}), flush=True)\n\n    return 0\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n'


def _write_capture_bot(script_path: Path) -> None:
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(_CAPTURE_BOT_SOURCE, encoding="utf-8")


def _capture_selfplay_start_payload(args: argparse.Namespace, seed: int | None = None) -> tuple[dict[str, Any], Path]:
    """Ask the real Java self-play environment for its first action_request.

    Defaults to PlayLG so the payload comes from the Java level generator rather
    than the old synthetic 4x4 payload or a fixed CSV file. The Java side treats
    PlayLG as the generated-map path; Map Type/Map Size are optional because the
    Java runner has defaults, but we set them when the Python wrapper supports
    these dynamic attributes.
    """
    from training.replay import ReplayStore
    from training.selfplay import run_selfplay

    run_seed = int(args.selfplay_seed if seed is None else seed)

    cfg = HybridAgentConfig()
    cfg.selfplay.run_mode = str(args.selfplay_run_mode)
    cfg.selfplay.game_mode = str(args.selfplay_game_mode)
    cfg.selfplay.level_file = str(args.selfplay_level_file)
    cfg.selfplay.game_seed = run_seed
    cfg.selfplay.agent_seed = run_seed
    cfg.selfplay.level_seed = run_seed
    cfg.selfplay.max_turns_capitals = max(1, int(args.capture_max_turns))
    cfg.selfplay.max_actions_per_turn = max(1, int(args.capture_max_actions_per_turn))
    cfg.selfplay.max_actions_per_game = max(1, int(args.capture_max_actions))
    cfg.selfplay.timeout_seconds = max(1, int(args.capture_timeout_sec))
    cfg.selfplay.persistent_bot = False
    cfg.selfplay.profile_selfplay = False

    # These fields are not present in older HybridAgentConfig versions, but
    # setting them is harmless and lets newer java_selfplay wrappers emit
    # "Map Type" and "Map Size" for PlayLG.
    setattr(cfg.selfplay, "map_type", str(args.selfplay_map_type))
    setattr(cfg.selfplay, "map_size", str(args.selfplay_map_size))

    if args.java_executable:
        cfg.selfplay.java_executable = str(args.java_executable)
    if args.java_classpath:
        cfg.selfplay.java_classpath = str(args.java_classpath)
    if args.java_main_class:
        cfg.selfplay.java_main_class = str(args.java_main_class)

    workdir = Path(args.workdir).resolve() if args.workdir is not None else PY_ROOT.parent.resolve()
    capture_root = Path(args.captured_payload_dir or (workdir / "debug-logs" / "mcts-profile-payloads")).resolve()
    safe_mode = str(args.selfplay_run_mode).replace("/", "_").replace("\\", "_")
    safe_map_type = str(args.selfplay_map_type).replace(" ", "_")
    safe_map_size = str(args.selfplay_map_size).replace(" ", "_")
    out_path = capture_root / f"{safe_mode}_{safe_map_type}_{safe_map_size}_seed{run_seed}.json"

    if out_path.exists() and bool(args.reuse_captured_payloads):
        return _load_payload(out_path), out_path
    if out_path.exists():
        out_path.unlink()

    capture_script = out_path.parent / "_capture_first_action_request_bot.py"
    _write_capture_bot(capture_script)
    command = [sys.executable, str(capture_script), "--output", str(out_path), "--py-root", str(PY_ROOT)]
    tribes = list(args.selfplay_tribes)
    replay_dir = out_path.parent / "_capture_replay"
    replay_store = ReplayStore(replay_dir, capacity_steps=0, shard_prefix="capture", load_existing=False)

    try:
        run_selfplay(
            cfg,
            [command, list(command)],
            tribes,
            workdir,
            checkpoint_path=Path("rl/checkpoints/latest.pt"),
            replay_store=replay_store,
            device=torch.device("cpu"),
            progress_label=f"profile-capture-{safe_mode}-seed{run_seed}",
        )
    except Exception as exc:
        if not out_path.exists():
            raise RuntimeError(
                "Self-play capture failed before any action_request was written. "
                f"workdir={workdir} run_mode={cfg.selfplay.run_mode} map_type={getattr(cfg.selfplay, 'map_type', '?')} "
                f"map_size={getattr(cfg.selfplay, 'map_size', '?')} level={cfg.selfplay.level_file} seed={cfg.selfplay.level_seed}"
            ) from exc
        print(f"[profile_mcts_search] self-play capture ended with {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    if not out_path.exists():
        raise RuntimeError(f"Self-play capture completed but did not write a payload: {out_path}")
    return _load_payload(out_path), out_path


def _run_native_mcts_walltime(
    root_payload: dict[str, Any],
    evaluator: HybridPolicyValueNet,
    search_cfg,
    model_cfg,
    device: torch.device | str,
    wall_time_sec: float,
    belief_snapshot: Any | None = None,
    selected_path_budget: int | None = None,
) -> tuple[native_mcts.SearchResult, SearchStats]:
    extension = load_native_mcts_extension()
    if extension is None:
        raise RuntimeError("Native MCTS extension is unavailable.")

    started_at = time.perf_counter()
    mode = "paths" if selected_path_budget is not None else "walltime"
    deadline = None if selected_path_budget is not None else started_at + max(0.0, float(wall_time_sec))
    root_actions = list(root_payload.get("actions", []))[: model_cfg.max_actions]
    if not root_actions:
        return native_mcts.SearchResult("", 0, {}, [], 0.0), SearchStats(mode, time.perf_counter() - started_at)

    root_prior_result = native_mcts._root_priors(
        root_payload,
        evaluator,
        search_cfg,
        model_cfg,
        device,
    )
    if len(root_prior_result) == 4:
        action_ids, priors, root_value, root_indexes = root_prior_result
    else:
        action_ids, _action_types, _unit_ids, _city_ids, priors, root_value, root_indexes = root_prior_result
    if not action_ids:
        return (
            native_mcts.SearchResult("", 0, {}, [0.0] * len(root_actions), root_value),
            SearchStats(mode, time.perf_counter() - started_at),
        )
    if len(action_ids) == 1:
        only_action_id = action_ids[0]
        root_action_ids = [str(action.get("id")) for action in root_actions]
        action_index = root_action_ids.index(only_action_id) if only_action_id in root_action_ids else root_indexes[0]
        return (
            native_mcts.SearchResult(
                action_id=only_action_id,
                action_index=action_index,
                visit_distribution={only_action_id: 1.0},
                visit_target=[1.0 if candidate_id == only_action_id else 0.0 for candidate_id in root_action_ids],
                value=float(root_value),
            ),
            SearchStats(mode, time.perf_counter() - started_at, simulations=0, selected_paths=0, expanded_nodes=0),
        )

    seed = int(getattr(search_cfg, "seed", 0) or int(time.time_ns() & 0xFFFFFFFF))
    tree = extension.NativeMCTS(
        root_payload,
        root_indexes,
        priors,
        float(root_value),
        bool(root_payload.get("is_terminal", False) or root_payload.get("terminal", False)),
        seed,
        int(model_cfg.max_actions),
    )
    tree.add_root_dirichlet_noise(float(search_cfg.dirichlet_alpha), float(search_cfg.dirichlet_epsilon))

    stats = SearchStats(mode, 0.0, expanded_nodes=0)
    max_depth = -1 if int(search_cfg.max_depth) <= 0 else int(search_cfg.max_depth)
    batch_size = max(1, int(search_cfg.batch_size))
    eval_cache: dict[Any, native_mcts._Evaluation] = {}
    expanded_node_ids: set[int] = set()
    node_depths: dict[int, int] = {0: 0}

    while (
        (deadline is not None and time.perf_counter() < deadline)
        or (selected_path_budget is not None and stats.selected_paths < selected_path_budget)
    ):
        remaining_paths = None if selected_path_budget is None else max(0, selected_path_budget - stats.selected_paths)
        if remaining_paths == 0:
            break
        frontier = batch_size if remaining_paths is None else min(batch_size, remaining_paths)
        selections: list[Any] = []
        eval_messages: list[dict[str, Any]] = []
        eval_index_by_key: dict[Any, int] = {}
        evals_only_batch = getattr(tree, "select_leaf_batch_evals_only", None)
        evals_only_batches = getattr(tree, "select_leaf_batches_evals_only", None)
        completed_frontier = frontier
        if evals_only_batches is not None:
            max_batches = 128 if remaining_paths is None else 1
            raw_selections, completed_frontier = evals_only_batches(frontier, max_batches, max_depth, float(search_cfg.c_puct))
        else:
            select_leaf_batch = evals_only_batch or getattr(tree, "select_leaf_batch_compact", tree.select_leaf_batch)
            raw_selections = select_leaf_batch(frontier, max_depth, float(search_cfg.c_puct))
        for raw_selection in raw_selections:
            if evals_only_batch is not None:
                if len(raw_selection) == 6:
                    selection_id, parent_node_id, parent_action_index, state_key, selection_depth, raw_leaf_payload = raw_selection
                else:
                    selection_id, parent_node_id, parent_action_index, state_key, raw_leaf_payload = raw_selection
                    selection_depth = 0
                parent_node_id = int(parent_node_id)
                selection_depth = int(selection_depth)
                leaf_payload = dict(raw_leaf_payload or {})
                if belief_snapshot is not None:
                    leaf_payload = belief_snapshot.annotate_without_update(leaf_payload)
                    eval_key: Any = native_mcts._message_cache_key(leaf_payload)
                else:
                    eval_key = int(state_key)
                cached = eval_cache.get(eval_key)
                if cached is not None:
                    stats.eval_cache_hits += 1
                    eval_index = -1
                else:
                    eval_index = eval_index_by_key.get(eval_key, -1)
                    if eval_index < 0:
                        eval_index = len(eval_messages)
                        eval_index_by_key[eval_key] = eval_index
                        eval_messages.append(leaf_payload)
                selections.append(
                    (
                        int(selection_id),
                        True,
                        parent_node_id,
                        int(parent_action_index),
                        0.0,
                        False,
                        eval_index,
                        cached,
                    )
                )
                continue
            if isinstance(raw_selection, tuple):
                (
                    selection_id,
                    needs_expansion,
                    parent_node_id,
                    parent_action_index,
                    leaf_value,
                    leaf_terminal,
                    state_key,
                    raw_leaf_payload,
                ) = raw_selection
                parent_node_id = int(parent_node_id)
                selection_depth = node_depths.get(parent_node_id, 0) + (1 if parent_node_id >= 0 else 0)
                stats.depth_sum += selection_depth
                stats.max_depth = max(stats.max_depth, selection_depth)
                eval_index = -1
                cached: native_mcts._Evaluation | None = None
                if needs_expansion and not leaf_terminal:
                    leaf_payload = dict(raw_leaf_payload or {})
                    if belief_snapshot is not None:
                        leaf_payload = belief_snapshot.annotate_without_update(leaf_payload)
                        eval_key: Any = native_mcts._message_cache_key(leaf_payload)
                    else:
                        eval_key = int(state_key)
                    cached = eval_cache.get(eval_key)
                    if cached is not None:
                        stats.eval_cache_hits += 1
                    else:
                        eval_index = eval_index_by_key.get(eval_key, -1)
                        if eval_index < 0:
                            eval_index = len(eval_messages)
                            eval_index_by_key[eval_key] = eval_index
                            eval_messages.append(leaf_payload)
                selections.append(
                    (
                        int(selection_id),
                        bool(needs_expansion),
                        parent_node_id,
                        int(parent_action_index),
                        float(leaf_value),
                        bool(leaf_terminal),
                        eval_index,
                        cached,
                    )
                )
                continue

            selection = dict(raw_selection)
            parent_node_id = int(selection.get("parent_node_id", -1))
            selection_depth = node_depths.get(parent_node_id, 0) + (1 if parent_node_id >= 0 else 0)
            stats.depth_sum += selection_depth
            stats.max_depth = max(stats.max_depth, selection_depth)
            if selection.get("needs_expansion") and not selection.get("leaf_terminal"):
                leaf_payload = dict(selection.get("leaf_payload") or {})
                if belief_snapshot is not None:
                    leaf_payload = belief_snapshot.annotate_without_update(leaf_payload)
                    eval_key: Any = native_mcts._message_cache_key(leaf_payload)
                else:
                    eval_key = int(selection.get("state_key", selection.get("selection_id", 0)))
                cached = eval_cache.get(eval_key)
                if cached is not None:
                    selection["_cached_eval"] = cached
                    stats.eval_cache_hits += 1
                else:
                    if eval_key not in eval_index_by_key:
                        eval_index_by_key[eval_key] = len(eval_messages)
                        eval_messages.append(leaf_payload)
                    selection["_eval_index"] = eval_index_by_key[eval_key]
                    selection["_eval_key"] = eval_key
            selections.append(selection)

        if evals_only_batch is not None:
            batch_depth_sum, batch_max_depth = tree.last_batch_stats()
            stats.depth_sum += int(batch_depth_sum)
            stats.max_depth = max(stats.max_depth, int(batch_max_depth))
            stats.selected_paths += int(completed_frontier)
        if not selections:
            if evals_only_batch is not None:
                continue
            break
        if evals_only_batch is None:
            stats.selected_paths += len(selections)

        evaluations = native_mcts._evaluate_messages(eval_messages, evaluator, model_cfg, device)
        if eval_messages:
            stats.eval_batches += 1
            stats.eval_positions += len(eval_messages)
            for eval_key, eval_index in eval_index_by_key.items():
                eval_cache[eval_key] = evaluations[eval_index]

        completed_selection_ids: list[int] = []
        completed_leaf_values: list[float] = []
        for selection in selections:
            if isinstance(selection, tuple):
                (
                    selection_id,
                    needs_expansion,
                    parent_node_id,
                    parent_action_index,
                    leaf_value,
                    leaf_terminal,
                    eval_index,
                    cached_eval,
                ) = selection
                if needs_expansion:
                    if leaf_terminal:
                        leaf_value = 0.0
                    else:
                        evaluation = cached_eval if cached_eval is not None else evaluations[int(eval_index)]
                        leaf_value = evaluation.value
                        child_node_id = int(
                            tree.expand(
                                int(parent_node_id),
                                int(parent_action_index),
                                evaluation.priors,
                                float(evaluation.value),
                                False,
                            )
                        )
                        expanded_node_ids.add(child_node_id)
                        node_depths[child_node_id] = node_depths.get(int(parent_node_id), 0) + 1
                completed_selection_ids.append(int(selection_id))
                completed_leaf_values.append(float(leaf_value))
                continue

            leaf_value = float(selection.get("leaf_value", 0.0))
            if selection.get("needs_expansion"):
                if selection.get("leaf_terminal"):
                    leaf_value = 0.0
                else:
                    evaluation = selection.get("_cached_eval")
                    if evaluation is None:
                        evaluation = evaluations[int(selection["_eval_index"])]
                    leaf_value = evaluation.value
                    child_node_id = int(
                        tree.expand(
                            int(selection["parent_node_id"]),
                            int(selection["parent_action_index"]),
                            evaluation.priors,
                            float(evaluation.value),
                            False,
                        )
                    )
                    expanded_node_ids.add(child_node_id)
                    parent_node_id = int(selection["parent_node_id"])
                    node_depths[child_node_id] = node_depths.get(parent_node_id, 0) + 1
            completed_selection_ids.append(int(selection["selection_id"]))
            completed_leaf_values.append(float(leaf_value))
        tree.complete_selected_paths(completed_selection_ids, completed_leaf_values)

    stats.expanded_nodes += len(expanded_node_ids)
    stats.simulations = stats.expanded_nodes
    stats.eval_cache_size = len(eval_cache)
    stats.elapsed_sec = time.perf_counter() - started_at

    visit_distribution = {str(action_id): float(prob) for action_id, prob in tree.root_visit_distribution(search_cfg.root_temperature).items()}
    if search_cfg.sample_action:
        candidates = list(visit_distribution.keys())
        weights = [visit_distribution[action_id] for action_id in candidates]
        action_id = random.choices(candidates, weights=weights, k=1)[0]
    else:
        action_id = max(visit_distribution, key=visit_distribution.get)

    root_action_ids = [str(action.get("id")) for action in root_actions]
    action_index = root_action_ids.index(action_id) if action_id in root_action_ids else 0
    visit_target = [float(visit_distribution.get(candidate_id, 0.0)) for candidate_id in root_action_ids]
    return (
        native_mcts.SearchResult(
            action_id=action_id,
            action_index=action_index,
            visit_distribution=visit_distribution,
            visit_target=visit_target,
            value=float(root_value),
        ),
        stats,
    )


def _format_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    widths = []
    for key, label in columns:
        widths.append(max([len(label), *(len(str(row.get(key, ""))) for row in rows)]))
    lines = ["  ".join(label.ljust(widths[index]) for index, (_, label) in enumerate(columns))]
    lines.append("  ".join("-" * width for width in widths))
    for row in rows:
        lines.append("  ".join(str(row.get(key, "")).ljust(widths[index]) for index, (key, _) in enumerate(columns)))
    return "\n".join(lines)


def _timing_rows(collector: TimingCollector, total_sec: float) -> list[dict[str, Any]]:
    rows = []
    for row in collector.sorted_rows():
        calls = max(1, row.calls)
        rows.append(
            {
                "name": row.name,
                "calls": row.calls,
                "items": row.items,
                "total_ms": f"{row.total_sec * 1000.0:.3f}",
                "self_ms": f"{row.self_sec * 1000.0:.3f}",
                "avg_ms": f"{row.total_sec * 1000.0 / calls:.3f}",
                "pct": f"{(row.total_sec / total_sec * 100.0) if total_sec > 0.0 else 0.0:.1f}",
            }
        )
    return rows


def _profile_rows(profile: cProfile.Profile, *, root: Path) -> list[dict[str, Any]]:
    stats = pstats.Stats(profile)
    entries = []
    for (filename, line, func_name), stat in stats.stats.items():
        primitive_calls, total_calls, total_time, cumulative_time, _callers = stat
        if total_time <= 0.0 and cumulative_time <= 0.0:
            continue
        try:
            display_file = str(Path(filename).resolve().relative_to(root))
        except ValueError:
            display_file = filename
        entries.append(
            {
                "function": f"{display_file}:{line}:{func_name}",
                "calls": total_calls,
                "self_ms": total_time * 1000.0,
                "cum_ms": cumulative_time * 1000.0,
                "avg_self_us": total_time * 1_000_000.0 / max(1, primitive_calls),
            }
        )
    entries.sort(key=lambda row: row["cum_ms"], reverse=True)
    return [
        {
            "function": row["function"],
            "calls": row["calls"],
            "self_ms": f"{row['self_ms']:.3f}",
            "cum_ms": f"{row['cum_ms']:.3f}",
            "avg_self_us": f"{row['avg_self_us']:.3f}",
        }
        for row in entries
    ]


def _filtered_timing_rows(rows: list[dict[str, Any]], *, min_ms: float, min_pct: float, limit: int) -> list[dict[str, Any]]:
    filtered = [
        row
        for row in rows
        if float(row["total_ms"]) >= min_ms and float(row["pct"]) >= min_pct
    ]
    return filtered[:limit]


def _filtered_profile_rows(rows: list[dict[str, Any]], *, min_ms: float, total_sec: float, min_pct: float, limit: int) -> list[dict[str, Any]]:
    total_ms = max(1e-9, total_sec * 1000.0)
    filtered = []
    for row in rows:
        cum_ms = float(row["cum_ms"])
        pct = cum_ms / total_ms * 100.0
        if cum_ms < min_ms or pct < min_pct:
            continue
        filtered.append(
            {
                "function": row["function"],
                "calls": row["calls"],
                "cum_ms": row["cum_ms"],
                "self_ms": row["self_ms"],
                "pct": f"{pct:.1f}",
            }
        )
    return filtered[:limit]


def _hotspot_rows(
    timing_rows: list[dict[str, Any]],
    profile_rows: list[dict[str, Any]],
    *,
    total_sec: float,
    min_ms: float,
    min_pct: float,
    limit: int,
) -> list[dict[str, Any]]:
    total_ms = max(1e-9, total_sec * 1000.0)
    rows: list[dict[str, Any]] = []
    for row in timing_rows:
        total = float(row["total_ms"])
        pct = float(row["pct"])
        if total < min_ms or pct < min_pct:
            continue
        rows.append(
            {
                "source": "phase",
                "name": row["name"],
                "calls": row["calls"],
                "items": row["items"],
                "time_ms": f"{total:.3f}",
                "self_ms": row["self_ms"],
                "pct": row["pct"],
            }
        )
    for row in profile_rows:
        if str(row["function"]).startswith("py\\profiling\\mcts_search.py:") or str(row["function"]).startswith("py/profiling/mcts_search.py:"):
            continue
        total = float(row["cum_ms"])
        pct = total / total_ms * 100.0
        if total < min_ms or pct < min_pct:
            continue
        rows.append(
            {
                "source": "function",
                "name": row["function"],
                "calls": row["calls"],
                "items": "",
                "time_ms": row["cum_ms"],
                "self_ms": row["self_ms"],
                "pct": f"{pct:.1f}",
            }
        )
    rows.sort(key=lambda row: float(row["time_ms"]), reverse=True)
    return rows[:limit]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["name"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _add_stats(total: SearchStats, item: SearchStats) -> None:
    total.simulations += int(item.simulations)
    total.selected_paths += int(item.selected_paths)
    total.expanded_nodes += int(item.expanded_nodes)
    total.eval_batches += int(item.eval_batches)
    total.eval_positions += int(item.eval_positions)
    total.eval_cache_hits += int(item.eval_cache_hits)
    total.eval_cache_size += int(item.eval_cache_size)
    total.depth_sum += int(item.depth_sum)
    total.max_depth = max(int(total.max_depth), int(item.max_depth))


def _payload_actions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    actions = payload.get("actions", []) if isinstance(payload, dict) else []
    return [action for action in actions if isinstance(action, dict)]


def _action_type(action: dict[str, Any]) -> str:
    return str(action.get("type") or action.get("t") or "UNKNOWN").upper()


def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return float(ordered[index])


def _branching_summary(values: list[int]) -> dict[str, str]:
    if not values:
        return {"samples": "0", "avg": "0.00", "p50": "0", "p90": "0", "max": "0"}
    return {
        "samples": str(len(values)),
        "avg": f"{sum(values) / len(values):.2f}",
        "p50": f"{_percentile(values, 0.50):.0f}",
        "p90": f"{_percentile(values, 0.90):.0f}",
        "max": str(max(values)),
    }


def _action_breadth_rows(branching: BranchingCollector, *, limit: int = 12) -> list[dict[str, Any]]:
    root_total = sum(branching.root_action_types.values())
    leaf_total = sum(branching.eval_action_types.values())
    visit_total = max(1, branching.root_visit_samples)
    names = set(branching.root_action_types) | set(branching.eval_action_types) | set(branching.root_visit_by_type)

    def rank(name: str) -> float:
        return (
            float(branching.root_action_types.get(name, 0))
            + float(branching.eval_action_types.get(name, 0))
            + float(branching.root_visit_by_type.get(name, 0)) * 100.0
        )

    rows = []
    for name in sorted(names, key=rank, reverse=True)[:limit]:
        root_count = int(branching.root_action_types.get(name, 0))
        leaf_count = int(branching.eval_action_types.get(name, 0))
        visit_share = float(branching.root_visit_by_type.get(name, 0.0)) / float(visit_total)
        rows.append(
            {
                "type": name,
                "root_count": root_count,
                "root_share": f"{(root_count / root_total * 100.0) if root_total else 0.0:.1f}%",
                "leaf_count": leaf_count,
                "leaf_share": f"{(leaf_count / leaf_total * 100.0) if leaf_total else 0.0:.1f}%",
                "visit_share": f"{visit_share * 100.0:.1f}%",
            }
        )
    return rows


def _top_root_action_rows(
    payload: dict[str, Any],
    result: native_mcts.SearchResult | None,
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    if result is None:
        return []
    actions = _payload_actions(payload)
    action_by_id = {str(action.get("id")): action for action in actions}
    rows = []
    for action_id, share in sorted(result.visit_distribution.items(), key=lambda item: float(item[1]), reverse=True)[:limit]:
        action = action_by_id.get(str(action_id), {})
        rows.append(
            {
                "action_id": action_id,
                "type": _action_type(action),
                "visit_share": f"{float(share) * 100.0:.1f}%",
                "unit": action.get("unit_id", ""),
                "city": action.get("city_id", ""),
                "x": action.get("x", ""),
                "y": action.get("y", ""),
            }
        )
    return rows


def _payload_summary(payload: dict[str, Any]) -> dict[str, Any]:
    observation = payload.get("observation", {}) if isinstance(payload, dict) else {}
    board = observation.get("board", {}) if isinstance(observation, dict) else {}
    return {
        "actions": len(_payload_actions(payload)),
        "board_size": board.get("size", "?"),
        "tick": observation.get("tick", "?"),
        "player_id": payload.get("player_id", "?") if isinstance(payload, dict) else "?",
        "active_player_id": observation.get("active_player_id", "?"),
    }


def _format_rate(numerator: float, elapsed_sec: float) -> str:
    return f"{(float(numerator) / elapsed_sec) if elapsed_sec > 0.0 else 0.0:.2f}"


def _build_payload_cases(args: argparse.Namespace) -> list[PayloadCase]:
    if args.payload is not None and args.synthetic:
        raise ValueError("Use either --payload or --synthetic, not both.")
    if args.payload is not None:
        return [PayloadCase(label=f"payload:{args.payload}", payload=_load_payload(args.payload), path=args.payload)]
    if args.synthetic:
        return [PayloadCase(label="synthetic_4x4", payload=_synthetic_payload())]

    if args.selfplay_seeds:
        seeds = [int(seed) for seed in args.selfplay_seeds]
    else:
        seeds = [int(args.selfplay_seed_start) + offset for offset in range(max(1, int(args.positions)))]

    cases: list[PayloadCase] = []
    for index, seed in enumerate(seeds, start=1):
        print(
            f"[capture {index}/{len(seeds)}] run_mode={args.selfplay_run_mode} "
            f"map={args.selfplay_map_type}/{args.selfplay_map_size} seed={seed}",
            flush=True,
        )
        payload, path = _capture_selfplay_start_payload(args, seed=seed)
        summary = _payload_summary(payload)
        print(
            f"  captured actions={summary['actions']} board_size={summary['board_size']} "
            f"tick={summary['tick']} path={path}",
            flush=True,
        )
        cases.append(PayloadCase(label=f"{args.selfplay_run_mode}:seed{seed}", payload=payload, seed=seed, path=path))
    return cases


def _default_mcts_search_output_path(args: argparse.Namespace, filename: str) -> Path:
    return Path("debug-logs") / f"mcts-search-{args.evaluator}" / filename


def _run_one_profile_case(
    case: PayloadCase,
    *,
    evaluator_mode: str,
    model: HybridPolicyValueNet | None,
    cfg: HybridAgentConfig,
    device: torch.device,
    using_walltime: bool,
    wall_time_sec: float,
    repeats: int,
) -> tuple[native_mcts.SearchResult | None, SearchStats, float]:
    global _STATIC_TREE_DEPTH_SUM, _STATIC_TREE_MAX_DEPTH, _STATIC_TREE_SELECTED_PATHS, _STATIC_TREE_EXPANDED_NODE_IDS
    last_result: native_mcts.SearchResult | None = None
    stats = SearchStats("walltime" if using_walltime else "simulations", 0.0)
    if evaluator_mode == "static":
        _STATIC_TREE_DEPTH_SUM = 0
        _STATIC_TREE_MAX_DEPTH = 0
        _STATIC_TREE_SELECTED_PATHS = 0
        _STATIC_TREE_EXPANDED_NODE_IDS = set()
    started_at = time.perf_counter()
    for _ in range(max(1, int(repeats))):
        if evaluator_mode == "static":
            depth_sum_before = _STATIC_TREE_DEPTH_SUM
            max_depth_before = _STATIC_TREE_MAX_DEPTH
            selected_paths_before = _STATIC_TREE_SELECTED_PATHS
            expanded_before = len(_STATIC_TREE_EXPANDED_NODE_IDS)
            last_result = native_static_mcts.run_native_static_mcts(
                case.payload,
                cfg.search,
                cfg.model,
                wall_time_seconds=wall_time_sec if using_walltime else None,
            )
            selected_delta = max(0, _STATIC_TREE_SELECTED_PATHS - selected_paths_before)
            expanded_delta = max(0, len(_STATIC_TREE_EXPANDED_NODE_IDS) - expanded_before)
            stats.depth_sum += max(0, _STATIC_TREE_DEPTH_SUM - depth_sum_before)
            stats.max_depth = max(stats.max_depth, _STATIC_TREE_MAX_DEPTH)
            stats.selected_paths += selected_delta
            stats.expanded_nodes += expanded_delta
            stats.simulations += expanded_delta
            continue
        if model is None:
            raise RuntimeError("NN evaluator mode requires a model.")
        if using_walltime:
            last_result, run_stats = _run_native_mcts_walltime(
                case.payload,
                model,
                cfg.search,
                cfg.model,
                device,
                wall_time_sec,
            )
            _add_stats(stats, run_stats)
        else:
            last_result, run_stats = _run_native_mcts_walltime(
                case.payload,
                model,
                cfg.search,
                cfg.model,
                device,
                0.0,
                selected_path_budget=int(cfg.search.num_simulations),
            )
            _add_stats(stats, run_stats)
    _sync_if_needed(device)
    elapsed = time.perf_counter() - started_at
    stats.elapsed_sec = elapsed
    return last_result, stats, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Profile native MCTS tree search with neural-network or native static evaluation. "
            "By default this captures generated PlayLG self-play starts and benchmarks 10 positions x 10 sec."
        )
    )
    parser.add_argument("--evaluator", choices=["nn", "static"], default="nn", help="Evaluator backend to profile. Default: nn.")
    parser.add_argument("--payload", type=Path, default=None, help="Profile one existing JSON root payload instead of generated self-play starts.")
    parser.add_argument("--synthetic", action="store_true", help="Use the old tiny synthetic 4x4 root instead of real generated starts.")

    parser.add_argument("--positions", type=int, default=10, help="Number of generated starting positions to capture/profile by default.")
    parser.add_argument("--selfplay-seed-start", type=int, default=0, help="First seed for generated starting positions.")
    parser.add_argument("--selfplay-seeds", nargs="*", type=int, default=None, help="Explicit generated-map seeds. Overrides --positions/--selfplay-seed-start.")
    parser.add_argument("--selfplay-run-mode", default="PlayLG", help="Java run mode for capture. Default: PlayLG generated maps.")
    parser.add_argument("--selfplay-level-file", default="levels/MinimalLevel2.csv", help="Level CSV used only when --selfplay-run-mode=PlayFile.")
    parser.add_argument("--selfplay-game-mode", default="Capitals", help="Game mode used for capture.")
    parser.add_argument("--selfplay-map-type", default="Drylands", help="Generated map type for PlayLG when supported by the Python Java wrapper.")
    parser.add_argument("--selfplay-map-size", default="Tiny", help="Generated map size for PlayLG when supported by the Python Java wrapper.")
    parser.add_argument("--selfplay-seed", type=int, default=0, help="Compatibility alias for single-start capture helpers; normally use --selfplay-seed-start or --selfplay-seeds.")
    parser.add_argument("--selfplay-tribes", nargs=2, default=["Xin Xi", "Imperius"], help="Two tribes used for capture.")
    parser.add_argument("--capture-max-turns", type=int, default=1, help="Short cap for the temporary capture game.")
    parser.add_argument("--capture-max-actions-per-turn", type=int, default=1, help="Short cap for the temporary capture game.")
    parser.add_argument("--capture-max-actions", type=int, default=4, help="Short cap for the temporary capture game.")
    parser.add_argument("--capture-timeout-sec", type=int, default=60, help="Timeout for each temporary capture game.")
    parser.add_argument("--captured-payload-dir", type=Path, default=None, help="Directory for captured generated-start payload JSON files.")
    parser.add_argument("--reuse-captured-payloads", action="store_true", help="Reuse captured payload files if they already exist.")
    parser.add_argument("--workdir", type=Path, default=None, help="Repo root/workdir for Java self-play. Defaults to parent of this script directory.")
    parser.add_argument("--java-executable", default=None, help="Optional Java executable override for self-play capture.")
    parser.add_argument("--java-classpath", default=None, help="Optional Java classpath override for self-play capture.")
    parser.add_argument("--java-main-class", default=None, help="Optional Java main class override for self-play capture.")

    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_AUTORESEARCH_CHECKPOINT, help="Model checkpoint used by the NN evaluator.")
    parser.add_argument("--device", default=None, help="Torch device, for example cpu or cuda. Defaults to cuda when available, otherwise cpu.")
    parser.add_argument(
        "--simulations",
        type=int,
        default=None,
        help="Legacy name for fixed selected-path budget per position. If omitted, use wall-clock mode.",
    )
    parser.add_argument("--wall-time-sec", "--walltime", type=float, default=10.0, help="Wall-clock budget per starting position. Default: 10 sec.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=1, help="Repeats per position. In walltime mode, each repeat gets --wall-time-sec.")
    parser.add_argument("--warmup", type=int, default=0, help="Warmup runs per captured payload; not included in reported benchmark time.")
    parser.add_argument("--top-k-actions", type=int, default=None)
    parser.add_argument("--no-dirichlet", action="store_true", help="Disable root Dirichlet noise for deterministic profiling.")
    parser.add_argument("--function-limit", type=int, default=15, help="Maximum hotspot rows printed to the console.")
    parser.add_argument("--min-hotspot-ms", type=float, default=1.0, help="Hide timing/function rows below this cumulative millisecond threshold.")
    parser.add_argument("--min-hotspot-pct", type=float, default=1.0, help="Hide timing/function rows below this percent-of-run threshold.")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV path for custom timing rows.")
    parser.add_argument("--profile-csv", type=Path, default=None, help="Optional CSV path for cProfile rows.")
    parser.add_argument("--position-csv", type=Path, default=None, help="Optional CSV path for per-position search throughput rows.")
    args = load_config_defaults(parser)

    if args.position_csv is None:
        args.position_csv = _default_mcts_search_output_path(args, "positions.csv")
    if args.csv is None:
        args.csv = _default_mcts_search_output_path(args, "timing.csv")
    if args.profile_csv is None:
        args.profile_csv = _default_mcts_search_output_path(args, "functions.csv")

    extension = load_native_mcts_extension()
    if extension is None:
        raise RuntimeError("Native MCTS extension is unavailable; build prerequisites may be missing.")

    cfg = HybridAgentConfig()
    using_walltime = args.simulations is None
    if args.simulations is not None:
        cfg.search.num_simulations = int(args.simulations)
    if args.batch_size is not None:
        cfg.search.batch_size = int(args.batch_size)
    cfg.search.seed = 123
    if args.top_k_actions is not None:
        cfg.search.top_k_actions = int(args.top_k_actions)
    if args.no_dirichlet:
        cfg.search.dirichlet_epsilon = 0.0

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model: HybridPolicyValueNet | None = None
    checkpoint_status = "random_init"
    if args.evaluator == "nn":
        model = HybridPolicyValueNet(cfg.model).eval().to(device)
        if args.checkpoint is not None:
            checkpoint_path = Path(args.checkpoint)
            allow_reinitialize = checkpoint_path.resolve() == DEFAULT_AUTORESEARCH_CHECKPOINT.resolve()
            checkpoint_status = _load_or_initialize_checkpoint(
                model,
                checkpoint_path,
                device,
                allow_reinitialize=allow_reinitialize,
            )
            model.eval()
    else:
        checkpoint_status = "static_eval"

    cases = _build_payload_cases(args)
    if not cases:
        raise RuntimeError("No payload cases were captured/loaded.")

    collector = TimingCollector()
    branching = BranchingCollector()
    original_evaluator = _install_timed_evaluator(collector, branching) if args.evaluator == "nn" else None
    original_static_evaluator = _install_timed_static_evaluator(collector, branching) if args.evaluator == "static" else None
    original_tree_cls = _install_timed_tree(extension, collector, device)
    profile = cProfile.Profile()
    per_position_rows: list[dict[str, Any]] = []
    last_result: native_mcts.SearchResult | None = None
    total_stats = SearchStats("walltime" if using_walltime else "simulations", 0.0)

    try:
        if args.warmup > 0:
            print(f"[warmup] {args.warmup} run(s) per payload; not included in benchmark timing", flush=True)
        for case in cases:
            for _ in range(max(0, int(args.warmup))):
                if using_walltime:
                    if args.evaluator == "static":
                        native_static_mcts.run_native_static_mcts(
                            case.payload,
                            cfg.search,
                            cfg.model,
                            wall_time_seconds=min(float(args.wall_time_sec), 0.2),
                        )
                    else:
                        if model is None:
                            raise RuntimeError("NN evaluator mode requires a model.")
                        _run_native_mcts_walltime(case.payload, model, cfg.search, cfg.model, device, min(float(args.wall_time_sec), 0.2))
                else:
                    if args.evaluator == "static":
                        native_static_mcts.run_native_static_mcts(case.payload, cfg.search, cfg.model)
                    else:
                        if model is None:
                            raise RuntimeError("NN evaluator mode requires a model.")
                        native_mcts.run_native_mcts(case.payload, model, cfg.search, cfg.model, device)

        collector = TimingCollector()
        branching = BranchingCollector()
        if original_evaluator is not None:
            native_mcts._evaluate_messages = original_evaluator
        if original_static_evaluator is not None:
            native_static_mcts._evaluate_static_messages = original_static_evaluator
        extension.NativeMCTS = original_tree_cls
        original_evaluator = _install_timed_evaluator(collector, branching) if args.evaluator == "nn" else None
        original_static_evaluator = _install_timed_static_evaluator(collector, branching) if args.evaluator == "static" else None
        _install_timed_tree(extension, collector, device)

        _sync_if_needed(device)
        benchmark_started_at = time.perf_counter()
        profile.enable()
        for index, case in enumerate(cases, start=1):
            summary = _payload_summary(case.payload)
            branching.add_root(case.payload)
            print(
                f"[benchmark {index}/{len(cases)}] {case.label} "
                f"actions={summary['actions']} board_size={summary['board_size']} tick={summary['tick']}",
                flush=True,
            )
            last_result, run_stats, elapsed = _run_one_profile_case(
                case,
                evaluator_mode=args.evaluator,
                model=model,
                cfg=cfg,
                device=device,
                using_walltime=using_walltime,
                wall_time_sec=float(args.wall_time_sec),
                repeats=max(1, int(args.repeats)),
            )
            _add_stats(total_stats, run_stats)
            branching.add_result(case.payload, last_result)
            row = {
                "index": index,
                "label": case.label,
                "seed": "" if case.seed is None else case.seed,
                "payload_path": "" if case.path is None else str(case.path),
                "actions": summary["actions"],
                "board_size": summary["board_size"],
                "tick": summary["tick"],
                "elapsed_sec": f"{elapsed:.3f}",
                "simulations": run_stats.simulations,
                "simulations_per_sec": _format_rate(run_stats.simulations, elapsed),
                "selected_paths": run_stats.selected_paths,
                "selected_paths_per_sec": _format_rate(run_stats.selected_paths, elapsed),
                "avg_depth": f"{run_stats.average_depth:.2f}",
                "max_depth": run_stats.max_depth,
                "eval_batches": run_stats.eval_batches,
                "eval_positions": run_stats.eval_positions,
                "eval_cache_hits": run_stats.eval_cache_hits,
            }
            per_position_rows.append(row)
        _sync_if_needed(device)
        profile.disable()
        elapsed = time.perf_counter() - benchmark_started_at
        total_stats.elapsed_sec = elapsed
    finally:
        if original_evaluator is not None:
            native_mcts._evaluate_messages = original_evaluator
        if original_static_evaluator is not None:
            native_static_mcts._evaluate_static_messages = original_static_evaluator
        extension.NativeMCTS = original_tree_cls

    print(
        f"MCTS profile: positions={len(cases)} repeats_per_position={max(1, int(args.repeats))} "
        f"mode={'walltime' if using_walltime else 'fixed_paths'} "
        f"selected_path_budget={'walltime' if using_walltime else cfg.search.num_simulations} "
        f"wall_time_sec_per_position={args.wall_time_sec if using_walltime else 'n/a'} "
        f"batch={cfg.search.batch_size} evaluator={args.evaluator} device={device} checkpoint={checkpoint_status} "
        f"source={'synthetic' if args.synthetic else 'payload' if args.payload else args.selfplay_run_mode} "
        f"map={args.selfplay_map_type}/{args.selfplay_map_size} "
        f"total_ms={elapsed * 1000.0:.3f}"
    )
    if last_result is not None:
        print(f"Selected action from final position: id={last_result.action_id} index={last_result.action_index} value={last_result.value:.4f}")

    print("\nPer-position throughput")
    print(
        _format_table(
            per_position_rows,
            [
                ("index", "#"),
                ("seed", "seed"),
                ("actions", "actions"),
                ("board_size", "board"),
                ("elapsed_sec", "sec"),
                ("simulations", "sims"),
                ("simulations_per_sec", "sims/s"),
                ("selected_paths", "paths"),
                ("selected_paths_per_sec", "paths/s"),
                ("avg_depth", "avg_depth"),
                ("max_depth", "max_depth"),
            ],
        )
    )

    if using_walltime:
        print(
            "\nAggregate search work: "
            f"simulations={total_stats.simulations} "
            f"simulations_per_sec={_format_rate(total_stats.simulations, elapsed)} "
            f"selected_paths={total_stats.selected_paths} "
            f"selected_paths_per_sec={_format_rate(total_stats.selected_paths, elapsed)} "
            f"avg_depth={total_stats.average_depth:.2f} "
            f"max_depth={total_stats.max_depth} "
            f"eval_batches={total_stats.eval_batches} "
            f"eval_positions={total_stats.eval_positions} "
            f"eval_cache_hits={total_stats.eval_cache_hits}"
        )

    print("\nBranching and action breadth")
    branching_rows = [
        {"scope": "root legal actions", **_branching_summary(branching.root_action_counts)},
        {"scope": "evaluated leaf actions", **_branching_summary(branching.eval_action_counts)},
    ]
    print(_format_table(branching_rows, [("scope", "scope"), ("samples", "samples"), ("avg", "avg"), ("p50", "p50"), ("p90", "p90"), ("max", "max")]))
    if branching.root_visit_samples:
        print(f"Average root visit entropy: {branching.root_visit_entropy_sum / branching.root_visit_samples:.3f} bits")

    action_breadth_rows = _action_breadth_rows(branching, limit=10)
    if action_breadth_rows:
        print("\nAction type breadth")
        print(
            _format_table(
                action_breadth_rows,
                [
                    ("type", "type"),
                    ("root_count", "root"),
                    ("root_share", "root_%"),
                    ("leaf_count", "leaf"),
                    ("leaf_share", "leaf_%"),
                    ("visit_share", "root_visit_%"),
                ],
            )
        )

    top_action_rows = _top_root_action_rows(cases[-1].payload, last_result, limit=10)
    if top_action_rows:
        print("\nTop root actions in final position")
        print(
            _format_table(
                top_action_rows,
                [("action_id", "action_id"), ("type", "type"), ("visit_share", "visits"), ("unit", "unit"), ("city", "city"), ("x", "x"), ("y", "y")],
            )
        )

    rows = _timing_rows(collector, elapsed)
    if args.csv is not None:
        _write_csv(args.csv, rows)

    profile_table = _profile_rows(profile, root=PY_ROOT.parent)
    if args.profile_csv is not None:
        _write_csv(args.profile_csv, profile_table)

    hotspot_rows = _hotspot_rows(
        rows,
        profile_table,
        total_sec=elapsed,
        min_ms=max(0.0, float(args.min_hotspot_ms)),
        min_pct=max(0.0, float(args.min_hotspot_pct)),
        limit=max(1, int(args.function_limit)),
    )
    print(
        "\nHotspots "
        f"(>= {float(args.min_hotspot_ms):.3g} ms and >= {float(args.min_hotspot_pct):.3g}% of run)"
    )
    if hotspot_rows:
        print(_format_table(hotspot_rows, [("source", "source"), ("name", "name"), ("calls", "calls"), ("items", "items"), ("time_ms", "time_ms"), ("self_ms", "self_ms"), ("pct", "%")]))
    else:
        print("No phase or function rows crossed the reporting threshold.")

    written = []
    if args.position_csv is not None:
        _write_csv(args.position_csv, per_position_rows)
        written.append(f"positions={args.position_csv}")
    if args.csv is not None:
        written.append(f"timing={args.csv}")
    if args.profile_csv is not None:
        written.append(f"functions={args.profile_csv}")
    if written:
        print("\nCSV: " + " ".join(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
