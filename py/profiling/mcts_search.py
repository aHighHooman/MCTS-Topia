from __future__ import annotations

import argparse
import cProfile
import csv
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import pstats
import random
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from typing import Any, Callable

import torch

try:
    import psutil
except ImportError:  # pragma: no cover - optional profiling dependency
    psutil = None

PY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PY_ROOT.parent
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

DEFAULT_AUTORESEARCH_CHECKPOINT = PY_ROOT / "profiling" / "mcts_search_profiling_random_init_model.pt"
DEFAULT_MCTS_SEARCH_CONFIG = PY_ROOT / "profiling" / "configs" / "mcts_search.json"

_MCTS_SEARCH_DEFAULTS: dict[str, Any] = {
    "evaluator": "nn",
    "payload": None,
    "positions": 10,
    "selfplay_seed_start": 0,
    "selfplay_seeds": None,
    "selfplay_run_mode": "PlayLG",
    "selfplay_game_mode": "Capitals",
    "selfplay_map_type": "Drylands",
    "selfplay_map_size": "Tiny",
    "selfplay_seed": 0,
    "selfplay_tribes": ["Xin Xi", "Imperius"],
    "capture_max_turns": 1,
    "capture_max_actions_per_turn": 1,
    "capture_max_actions": 4,
    "capture_timeout_sec": 60,
    "captured_payload_dir": None,
    "reuse_captured_payloads": False,
    "workdir": None,
    "java_executable": None,
    "java_classpath": None,
    "java_main_class": None,
    "checkpoint": str(DEFAULT_AUTORESEARCH_CHECKPOINT),
    "static_eval_variant": "baseline",
    "native_static_search_mode": "primitive",
    "native_static_exe": "out/native/static_mcts_bot.exe",
    "build_native_static_exe": True,
    "turn_macro_simulations": None,
    "turn_macro_max_primitives_per_turn": 0,
    "turn_macro_max_edges_per_node": 4,
    "turn_macro_outer_c": 1.4,
    "turn_macro_c": 1.0,
    "turn_macro_prior_weight": 0.35,
    "turn_macro_temperature": 1.0,
    "turn_macro_inner_simulations": 128,
    "turn_macro_inner_c_puct": 1.5,
    "turn_macro_greedy_eval_top_k": 1,
    "turn_macro_opponent_mode": "maximalist",
    "device": None,
    "simulations": None,
    "wall_time_sec": 10.0,
    "batch_size": 64,
    "repeats": 1,
    "warmup": 0,
    "top_k_actions": None,
    "no_dirichlet": False,
    "section_limit": 10,
    "function_limit": 15,
    "min_hotspot_ms": 1.0,
    "min_hotspot_pct": 1.0,
    "csv": None,
    "function_profile": False,
    "profile_csv": None,
    "position_csv": None,
    "branching_csv": None,
    "action_csv": None,
    "hardware_csv": None,
    "nn_module_csv": None,
    "details_json": None,
    "hardware_profile": False,
    "hardware_sample_interval_ms": 200,
    "nn_module_profile": False,
}

_PATH_CONFIG_KEYS = {
    "payload",
    "captured_payload_dir",
    "workdir",
    "checkpoint",
    "native_static_exe",
    "csv",
    "profile_csv",
    "position_csv",
    "branching_csv",
    "action_csv",
    "hardware_csv",
    "nn_module_csv",
    "details_json",
}

from nn.encoding import EncodedObservation, encode_observation
from nn.bot_agent import HybridRLBot
from nn.model import HybridPolicyValueNet
from search.config import HybridAgentConfig
from search.native import mcts as native_mcts
from search.native.cpp_extension import load_native_mcts_extension
from training.config import rl_path

_STATIC_TREE_DEPTH_SUM = 0
_STATIC_TREE_MAX_DEPTH = 0
_STATIC_TREE_TURN_DEPTH_SUM = 0
_STATIC_TREE_MAX_TURN_DEPTH = 0
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
    turn_depth_sum: int = 0
    max_turn_depth: int = 0
    inner_searches: int = 0
    inner_simulations: int = 0
    inner_nodes_expanded: int = 0
    static_eval_calls: int = 0
    greedy_static_calls: int = 0
    greedy_static_candidates_considered: int = 0
    greedy_static_child_evals: int = 0
    greedy_static_child_eval_skips: int = 0

    @property
    def average_depth(self) -> float:
        return float(self.depth_sum) / max(1, self.selected_paths)

    @property
    def average_turn_depth(self) -> float:
        return float(self.turn_depth_sum) / max(1, self.selected_paths)


@dataclass
class PayloadCase:
    label: str
    payload: dict[str, Any]
    seed: int | None = None
    path: Path | None = None


@dataclass
class BranchingCollector:
    root_action_counts: list[int] = field(default_factory=list)
    root_model_capped_counts: list[int] = field(default_factory=list)
    root_searched_counts: list[int] = field(default_factory=list)
    root_model_cap_drops: list[int] = field(default_factory=list)
    root_top_k_drops: list[int] = field(default_factory=list)
    eval_action_counts: list[int] = field(default_factory=list)
    policy_action_counts: list[int] = field(default_factory=list)
    policy_padded_action_counts: list[int] = field(default_factory=list)
    policy_padding_waste_slots: int = 0
    policy_total_slots: int = 0
    cpu_to_device_bytes: int = 0
    device_to_cpu_bytes: int = 0
    root_action_types: Counter[str] = field(default_factory=Counter)
    root_capped_action_types: Counter[str] = field(default_factory=Counter)
    root_searched_action_types: Counter[str] = field(default_factory=Counter)
    root_dropped_action_types: Counter[str] = field(default_factory=Counter)
    eval_action_types: Counter[str] = field(default_factory=Counter)
    root_visit_by_type: Counter[str] = field(default_factory=Counter)
    root_visit_entropy_sum: float = 0.0
    root_visit_samples: int = 0
    selected_action_types: Counter[str] = field(default_factory=Counter)
    action_rows: list[dict[str, Any]] = field(default_factory=list)
    branching_rows: list[dict[str, Any]] = field(default_factory=list)

    def add_root(self, payload: dict[str, Any], *, label: str = "", model_max_actions: int | None = None) -> None:
        actions = _payload_actions(payload)
        raw_count = len(actions)
        capped_count = _capped_action_count(raw_count, model_max_actions)
        capped_actions = actions[:capped_count]
        self.root_action_counts.append(raw_count)
        self.root_model_capped_counts.append(capped_count)
        self.root_model_cap_drops.append(max(0, raw_count - capped_count))
        self.root_action_types.update(_action_type(action) for action in actions)
        self.root_capped_action_types.update(_action_type(action) for action in capped_actions)
        self.branching_rows.append(
            {
                "label": label,
                "scope": "root",
                "raw_actions": raw_count,
                "model_capped_actions": capped_count,
                "searched_actions": "",
                "model_cap_dropped": max(0, raw_count - capped_count),
                "top_k_dropped": "",
                "visit_entropy_bits": "",
                "effective_branching": "",
            }
        )

    def add_eval_messages(self, messages: list[dict[str, Any]]) -> None:
        for message in messages:
            actions = _payload_actions(message)
            self.eval_action_counts.append(len(actions))
            self.eval_action_types.update(_action_type(action) for action in actions)

    def add_encoded_batch(self, encoded_items: list[EncodedObservation], batch: EncodedObservation) -> None:
        action_counts = [len(encoded.action_ids) for encoded in encoded_items]
        padded_actions = int(batch.action_features.shape[1]) if batch.action_features.ndim >= 2 else 0
        self.policy_action_counts.extend(action_counts)
        if action_counts:
            self.policy_padded_action_counts.extend([padded_actions] * len(action_counts))
            self.policy_padding_waste_slots += sum(max(0, padded_actions - count) for count in action_counts)
            self.policy_total_slots += padded_actions * len(action_counts)
        self.cpu_to_device_bytes += _encoded_nbytes(batch)

    def add_device_to_cpu_bytes(self, *tensors: torch.Tensor) -> None:
        self.device_to_cpu_bytes += sum(_tensor_nbytes(tensor) for tensor in tensors if torch.is_tensor(tensor))

    def add_result(
        self,
        payload: dict[str, Any],
        result: native_mcts.SearchResult | None,
        *,
        label: str = "",
        model_max_actions: int | None = None,
    ) -> None:
        if result is None:
            return
        actions = _payload_actions(payload)
        capped_count = _capped_action_count(len(actions), model_max_actions)
        capped_actions = actions[:capped_count]
        searched_ids = {str(action_id) for action_id in result.visit_distribution}
        searched_count = len(searched_ids)
        top_k_drop_count = max(0, capped_count - searched_count)
        self.root_searched_counts.append(searched_count)
        self.root_top_k_drops.append(top_k_drop_count)
        self.root_searched_action_types.update(
            _action_type(action) for action in capped_actions if str(action.get("id")) in searched_ids
        )
        self.root_dropped_action_types.update(
            _action_type(action) for action in capped_actions if str(action.get("id")) not in searched_ids
        )
        action_by_id = {str(action.get("id")): action for action in actions}
        stats_by_id = {
            str(row.get("action_id")): row
            for row in (result.root_stats or [])
            if isinstance(row, dict) and row.get("action_id") not in (None, "")
        }
        selected = action_by_id.get(str(result.action_id))
        if selected is not None:
            self.selected_action_types[_action_type(selected)] += 1
        entropy = 0.0
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
        effective_branching = 2.0 ** entropy if result.visit_distribution else 0.0
        self.branching_rows.append(
            {
                "label": label,
                "scope": "root_result",
                "raw_actions": len(actions),
                "model_capped_actions": capped_count,
                "searched_actions": searched_count,
                "model_cap_dropped": max(0, len(actions) - capped_count),
                "top_k_dropped": top_k_drop_count,
                "visit_entropy_bits": f"{entropy:.6f}" if result.visit_distribution else "0.000000",
                "effective_branching": f"{effective_branching:.6f}",
            }
        )
        for index, action in enumerate(actions):
            action_id = str(action.get("id"))
            visit_share = float(result.visit_distribution.get(action_id, 0.0))
            root_stat = stats_by_id.get(action_id, {})
            self.action_rows.append(
                {
                    "label": label,
                    "scope": "root",
                    "action_index": index,
                    "action_id": action_id,
                    "type": _action_type(action),
                    "category": _action_category(action),
                    "unit_id": action.get("unit_id", ""),
                    "city_id": action.get("city_id", ""),
                    "x": action.get("x", ""),
                    "y": action.get("y", ""),
                    "kept_by_model_cap": int(index < capped_count),
                    "kept_by_search": int(action_id in searched_ids),
                    "selected": int(action_id == str(result.action_id)),
                    "visit_share": f"{visit_share:.8f}",
                    "prior": root_stat.get("prior", ""),
                    "visits": root_stat.get("visits", ""),
                    "q_mean": root_stat.get("q_mean", ""),
                    "child_expanded": (
                        int(root_stat.get("child_node_id", -1) not in ("", None, -1))
                        if root_stat
                        else ""
                    ),
                }
            )


@dataclass
class HardwareSample:
    elapsed_sec: float
    process_rss_mb: float = 0.0
    process_cpu_pct: float = 0.0
    system_cpu_pct: float = 0.0
    system_ram_pct: float = 0.0
    threads: int = 0
    cuda_allocated_mb: float = 0.0
    cuda_reserved_mb: float = 0.0
    cuda_peak_allocated_mb: float = 0.0
    cuda_peak_reserved_mb: float = 0.0
    gpu_util_pct: float | str = ""
    gpu_memory_util_pct: float | str = ""
    gpu_memory_used_mb: float | str = ""
    gpu_power_watts: float | str = ""


class HardwareSampler:
    def __init__(self, *, device: torch.device, interval_sec: float) -> None:
        self.device = device
        self.interval_sec = max(0.05, float(interval_sec))
        self.samples: list[HardwareSample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0
        self._process = psutil.Process() if psutil is not None else None

    def __enter__(self) -> "HardwareSampler":
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        if self._process is not None:
            self._process.cpu_percent(interval=None)
            psutil.cpu_percent(interval=None)
        self._started_at = time.perf_counter()
        self._thread = threading.Thread(target=self._run, name="mcts-profile-hardware", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.2, self.interval_sec * 2.0))
        self.sample()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_sec):
            self.sample()

    def sample(self) -> None:
        elapsed = time.perf_counter() - self._started_at if self._started_at else 0.0
        sample = HardwareSample(elapsed_sec=elapsed)
        if self._process is not None and psutil is not None:
            try:
                memory = self._process.memory_info()
                sample.process_rss_mb = memory.rss / (1024.0 * 1024.0)
                sample.process_cpu_pct = float(self._process.cpu_percent(interval=None))
                sample.system_cpu_pct = float(psutil.cpu_percent(interval=None))
                sample.system_ram_pct = float(psutil.virtual_memory().percent)
                sample.threads = int(self._process.num_threads())
            except psutil.Error:
                pass
        if self.device.type == "cuda" and torch.cuda.is_available():
            sample.cuda_allocated_mb = torch.cuda.memory_allocated(self.device) / (1024.0 * 1024.0)
            sample.cuda_reserved_mb = torch.cuda.memory_reserved(self.device) / (1024.0 * 1024.0)
            sample.cuda_peak_allocated_mb = torch.cuda.max_memory_allocated(self.device) / (1024.0 * 1024.0)
            sample.cuda_peak_reserved_mb = torch.cuda.max_memory_reserved(self.device) / (1024.0 * 1024.0)
            nvidia = _query_nvidia_smi()
            if nvidia:
                sample.gpu_util_pct = nvidia.get("gpu_util_pct", "")
                sample.gpu_memory_util_pct = nvidia.get("gpu_memory_util_pct", "")
                sample.gpu_memory_used_mb = nvidia.get("gpu_memory_used_mb", "")
                sample.gpu_power_watts = nvidia.get("gpu_power_watts", "")
        self.samples.append(sample)

    def rows(self) -> list[dict[str, Any]]:
        has_process_metrics = self._process is not None
        return [
            {
                "elapsed_sec": f"{sample.elapsed_sec:.3f}",
                "process_rss_mb": f"{sample.process_rss_mb:.3f}" if has_process_metrics else "",
                "process_cpu_pct": f"{sample.process_cpu_pct:.1f}" if has_process_metrics else "",
                "system_cpu_pct": f"{sample.system_cpu_pct:.1f}" if has_process_metrics else "",
                "system_ram_pct": f"{sample.system_ram_pct:.1f}" if has_process_metrics else "",
                "threads": sample.threads if has_process_metrics else "",
                "cuda_allocated_mb": f"{sample.cuda_allocated_mb:.3f}",
                "cuda_reserved_mb": f"{sample.cuda_reserved_mb:.3f}",
                "cuda_peak_allocated_mb": f"{sample.cuda_peak_allocated_mb:.3f}",
                "cuda_peak_reserved_mb": f"{sample.cuda_peak_reserved_mb:.3f}",
                "gpu_util_pct": sample.gpu_util_pct,
                "gpu_memory_util_pct": sample.gpu_memory_util_pct,
                "gpu_memory_used_mb": sample.gpu_memory_used_mb,
                "gpu_power_watts": sample.gpu_power_watts,
            }
            for sample in self.samples
        ]


@dataclass
class NNModuleProfiler:
    collector: TimingCollector
    device: torch.device
    handles: list[Any] = field(default_factory=list)
    _starts: dict[int, float] = field(default_factory=dict)

    def install(self, model: HybridPolicyValueNet) -> None:
        modules = {
            "nn.board_encoder": model.board_encoder,
            "nn.unit_proj": model.unit_proj,
            "nn.city_proj": model.city_proj,
            "nn.action_proj": model.action_proj,
            "nn.scalar_value_proj": model.scalar_value_proj,
            "nn.scalar_summary_proj": model.scalar_summary_proj,
            "nn.transformer_core": model.core,
            "nn.action_attention": model.action_attention,
            "nn.value_action_attention": model.value_action_attention,
            "nn.policy_head": model.policy_head,
            "nn.value_head": model.value_head,
        }
        for name, module in modules.items():
            self.handles.append(module.register_forward_pre_hook(self._make_pre_hook(name)))
            self.handles.append(module.register_forward_hook(self._make_post_hook(name)))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self._starts.clear()

    def _make_pre_hook(self, name: str) -> Callable[..., None]:
        def hook(module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
            _sync_if_needed(self.device)
            self._starts[id(module)] = time.perf_counter()

        return hook

    def _make_post_hook(self, name: str) -> Callable[..., None]:
        def hook(module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            _sync_if_needed(self.device)
            started_at = self._starts.pop(id(module), None)
            if started_at is None:
                return
            self.collector.add(name, time.perf_counter() - started_at, items=_first_tensor_batch_size(inputs))

        return hook


def _result_from_bot_response(payload: dict[str, Any], response: dict[str, Any] | None) -> native_mcts.SearchResult:
    actions = _payload_actions(payload)
    action_ids = [str(action.get("id")) for action in actions]
    selected_action_id = str((response or {}).get("actionId") or "")
    try:
        action_index = action_ids.index(selected_action_id)
    except ValueError:
        action_index = 0

    root_stats: list[dict[str, Any]] = []
    profile = (response or {}).get("_profile") if isinstance(response, dict) else None
    if isinstance(profile, dict):
        raw_root_stats = profile.get("root_action_stats")
        if isinstance(raw_root_stats, list):
            root_stats = [dict(row) for row in raw_root_stats if isinstance(row, dict)]
        if not root_stats:
            raw_first_visits = profile.get("root_first_action_visits")
            if isinstance(raw_first_visits, dict):
                total = 0.0
                visits_by_id: dict[str, float] = {}
                for action_id, raw_visits in raw_first_visits.items():
                    try:
                        visits = max(0.0, float(raw_visits or 0.0))
                    except (TypeError, ValueError):
                        visits = 0.0
                    if visits <= 0.0:
                        continue
                    visits_by_id[str(action_id)] = visits
                    total += visits
                if total > 0.0:
                    root_stats = [
                        {
                            "action_id": action_id,
                            "visits": visits,
                            "visit_share": visits / total,
                            "q_mean": "",
                            "prior": "",
                        }
                        for action_id, visits in sorted(visits_by_id.items(), key=lambda item: item[1], reverse=True)
                    ]

    visit_distribution: dict[str, float] = {}
    if root_stats:
        total_visits = 0.0
        for row in root_stats:
            try:
                total_visits += max(0.0, float(row.get("visits", 0.0) or 0.0))
            except (TypeError, ValueError):
                pass
        for row in root_stats:
            action_id = str(row.get("action_id") or "")
            if not action_id:
                continue
            try:
                share = max(0.0, float(row.get("visit_share", 0.0) or 0.0))
            except (TypeError, ValueError):
                share = 0.0
            if share <= 0.0 and total_visits > 0.0:
                try:
                    share = max(0.0, float(row.get("visits", 0.0) or 0.0)) / total_visits
                except (TypeError, ValueError):
                    share = 0.0
            visit_distribution[action_id] = share
    if not visit_distribution and selected_action_id:
        visit_distribution = {selected_action_id: 1.0}

    visit_target = [float(visit_distribution.get(action_id, 0.0)) for action_id in action_ids]
    if not any(visit_target) and selected_action_id in action_ids:
        visit_target[action_index] = 1.0
    return native_mcts.SearchResult(
        action_id=selected_action_id,
        action_index=action_index,
        visit_distribution=visit_distribution,
        visit_target=visit_target,
        value=0.0,
        root_stats=root_stats,
    )


def _result_from_bot_record(payload: dict[str, Any], bot: HybridRLBot, response: dict[str, Any] | None) -> native_mcts.SearchResult:
    if not bot.records:
        return _result_from_bot_response(payload, response)
    record = bot.records[-1]
    actions = _payload_actions(payload)
    action_ids = [str(action.get("id")) for action in actions]
    selected_action_id = str(getattr(record, "action_id", "") or (response or {}).get("actionId") or "")
    action_index = int(getattr(record, "action_index", 0) or 0)
    visit_target = [float(value) for value in list(getattr(record, "visit_target", []) or [])]
    if len(visit_target) != len(action_ids):
        visit_target = [0.0] * len(action_ids)
        if selected_action_id in action_ids:
            visit_target[action_ids.index(selected_action_id)] = 1.0
    visit_distribution = {
        action_id: float(visit_target[index])
        for index, action_id in enumerate(action_ids[: len(visit_target)])
        if float(visit_target[index]) > 0.0
    }
    return native_mcts.SearchResult(
        action_id=selected_action_id,
        action_index=action_index,
        visit_distribution=visit_distribution,
        visit_target=visit_target,
        value=float(getattr(record, "root_value", 0.0) or 0.0),
    )


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
                lambda message=message: encode_observation(message, model_cfg, compact=True, normalized=True),
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
        branching.add_encoded_batch(encoded_items, batch)
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
            branching.add_device_to_cpu_bytes(output.policy_logits, output.value)
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


def _tree_batch_stats(tree: Any) -> tuple[int, int, int, int]:
    raw = tuple(tree.last_batch_stats())
    depth_sum = int(raw[0]) if len(raw) >= 1 else 0
    max_depth = int(raw[1]) if len(raw) >= 2 else 0
    turn_depth_sum = int(raw[2]) if len(raw) >= 3 else 0
    max_turn_depth = int(raw[3]) if len(raw) >= 4 else 0
    return depth_sum, max_depth, turn_depth_sum, max_turn_depth


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
            if hasattr(self._tree, "set_static_timing_enabled"):
                self._tree.set_static_timing_enabled(True)

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
            global _STATIC_TREE_DEPTH_SUM, _STATIC_TREE_MAX_DEPTH, _STATIC_TREE_TURN_DEPTH_SUM, _STATIC_TREE_MAX_TURN_DEPTH, _STATIC_TREE_SELECTED_PATHS
            started_at = time.perf_counter()
            result = list(self._tree.select_leaf_batch_evals_only(*args, **kwargs))
            frontier = int(args[0]) if args else len(result)
            collector.add("native_tree.select_leaf_batch_evals_only", time.perf_counter() - started_at, items=frontier)
            batch_depth_sum, batch_max_depth, batch_turn_depth_sum, batch_max_turn_depth = _tree_batch_stats(self._tree)
            _STATIC_TREE_DEPTH_SUM += int(batch_depth_sum)
            _STATIC_TREE_MAX_DEPTH = max(_STATIC_TREE_MAX_DEPTH, int(batch_max_depth))
            _STATIC_TREE_TURN_DEPTH_SUM += int(batch_turn_depth_sum)
            _STATIC_TREE_MAX_TURN_DEPTH = max(_STATIC_TREE_MAX_TURN_DEPTH, int(batch_max_turn_depth))
            _STATIC_TREE_SELECTED_PATHS += len(result)
            return result

        def select_leaf_batches_evals_only(self, *args: Any, **kwargs: Any) -> Any:
            global _STATIC_TREE_DEPTH_SUM, _STATIC_TREE_MAX_DEPTH, _STATIC_TREE_TURN_DEPTH_SUM, _STATIC_TREE_MAX_TURN_DEPTH, _STATIC_TREE_SELECTED_PATHS
            started_at = time.perf_counter()
            result, completed = self._tree.select_leaf_batches_evals_only(*args, **kwargs)
            result = list(result)
            collector.add("native_tree.select_leaf_batches_evals_only", time.perf_counter() - started_at, items=int(completed))
            batch_depth_sum, batch_max_depth, batch_turn_depth_sum, batch_max_turn_depth = _tree_batch_stats(self._tree)
            _STATIC_TREE_DEPTH_SUM += int(batch_depth_sum)
            _STATIC_TREE_MAX_DEPTH = max(_STATIC_TREE_MAX_DEPTH, int(batch_max_depth))
            _STATIC_TREE_TURN_DEPTH_SUM += int(batch_turn_depth_sum)
            _STATIC_TREE_MAX_TURN_DEPTH = max(_STATIC_TREE_MAX_TURN_DEPTH, int(batch_max_turn_depth))
            _STATIC_TREE_SELECTED_PATHS += int(completed)
            return result, int(completed)

        def run_static_search_batch(self, *args: Any, **kwargs: Any) -> Any:
            global _STATIC_TREE_DEPTH_SUM, _STATIC_TREE_MAX_DEPTH, _STATIC_TREE_TURN_DEPTH_SUM, _STATIC_TREE_MAX_TURN_DEPTH, _STATIC_TREE_SELECTED_PATHS, _STATIC_TREE_EXPANDED_NODE_IDS
            started_at = time.perf_counter()
            expanded, completed = self._tree.run_static_search_batch(*args, **kwargs)
            expanded = int(expanded)
            completed = int(completed)
            collector.add("native_tree.run_static_search_batch", time.perf_counter() - started_at, items=completed)
            if hasattr(self._tree, "last_static_timing"):
                for name, elapsed_ms in dict(self._tree.last_static_timing()).items():
                    collector.add(f"native_static.{name}", float(elapsed_ms) / 1000.0, items=completed)
            batch_depth_sum, batch_max_depth, batch_turn_depth_sum, batch_max_turn_depth = _tree_batch_stats(self._tree)
            _STATIC_TREE_DEPTH_SUM += int(batch_depth_sum)
            _STATIC_TREE_MAX_DEPTH = max(_STATIC_TREE_MAX_DEPTH, int(batch_max_depth))
            _STATIC_TREE_TURN_DEPTH_SUM += int(batch_turn_depth_sum)
            _STATIC_TREE_MAX_TURN_DEPTH = max(_STATIC_TREE_MAX_TURN_DEPTH, int(batch_max_turn_depth))
            _STATIC_TREE_SELECTED_PATHS += completed
            start_id = len(_STATIC_TREE_EXPANDED_NODE_IDS)
            for offset in range(expanded):
                _STATIC_TREE_EXPANDED_NODE_IDS.add(start_id + offset)
            return expanded, completed

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


def _load_payload(path: Path | None) -> dict[str, Any]:
    if path is None:
        raise ValueError("payload path is required when profiling an explicit payload")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        if not payload:
            raise ValueError(f"payload list is empty: {path}")
        payload = payload[0]
    if not isinstance(payload, dict):
        raise TypeError(f"payload must be a JSON object or a non-empty list of objects: {path}")
    return payload



_CAPTURE_BOT_SOURCE = 'from __future__ import annotations\n\nimport argparse\nimport json\nimport sys\nfrom pathlib import Path\nfrom typing import Any\n\n\ndef _pick_action_id(actions: list[dict[str, Any]]) -> str | None:\n    if not actions:\n        return None\n    for action in actions:\n        action_type = str(action.get("type") or "").upper()\n        if action_type == "END_TURN":\n            return str(action.get("id"))\n    return str(actions[0].get("id"))\n\n\ndef main() -> int:\n    parser = argparse.ArgumentParser(description="Capture the first real self-play action_request payload.")\n    parser.add_argument("--output", type=Path, required=True)\n    parser.add_argument("--py-root", type=Path, required=True)\n    args = parser.parse_args()\n\n    py_root = str(args.py_root.resolve())\n    if py_root not in sys.path:\n        sys.path.insert(0, py_root)\n\n    from nn.belief import BeliefTracker\n    from nn.encoding import normalize_message\n\n    tracker = BeliefTracker()\n    wrote_payload = False\n    args.output.parent.mkdir(parents=True, exist_ok=True)\n\n    for raw_line in sys.stdin:\n        line = raw_line.strip()\n        if not line:\n            continue\n        try:\n            message = json.loads(line)\n        except json.JSONDecodeError:\n            continue\n\n        msg_type = message.get("type")\n        if msg_type == "action_request":\n            normalized = tracker.annotate(normalize_message(message))\n            payload = {\n                "player_id": int(normalized.get("player_id", 0) or 0),\n                "observation": normalized["observation"],\n                "actions": list(normalized.get("actions", []) or []),\n            }\n            if not wrote_payload:\n                tmp = args.output.with_suffix(args.output.suffix + ".tmp")\n                tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")\n                tmp.replace(args.output)\n                wrote_payload = True\n\n            action_ids = [str(action.get("id")) for action in payload["actions"]]\n            selected = _pick_action_id(payload["actions"])\n            print(json.dumps({"actionId": selected, "rankedActionIds": action_ids}), flush=True)\n            continue\n\n        if msg_type == "game_over":\n            print(json.dumps({"ok": True}), flush=True)\n            break\n\n        print(json.dumps({"error": f"unsupported message type: {msg_type}"}), flush=True)\n\n    return 0\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n'

def _write_capture_bot(script_path: Path) -> None:
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(_CAPTURE_BOT_SOURCE, encoding="utf-8")


def _capture_selfplay_start_payload(args: argparse.Namespace, seed: int | None = None) -> tuple[dict[str, Any], Path]:
    """Ask the real Java self-play environment for its first action_request.

    Defaults to PlayLG so the payload comes from the Java level generator rather
    than a fixed CSV file. The Java side treats PlayLG as the generated-map path;
    Map Type/Map Size are optional because the
    Java runner has defaults, but we set them when the Python wrapper supports
    these dynamic attributes.
    """
    from training.replay import ReplayStore
    from training.selfplay import run_selfplay

    run_seed = int(args.selfplay_seed if seed is None else seed)

    cfg = HybridAgentConfig()
    cfg.selfplay.run_mode = str(args.selfplay_run_mode)
    cfg.selfplay.game_mode = str(args.selfplay_game_mode)
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

    try:
        run_selfplay(
            cfg,
            [command, list(command)],
            tribes,
            workdir,
            progress_label=f"profile-capture-{safe_mode}-seed{run_seed}",
        )
    except Exception as exc:
        if not out_path.exists():
            raise RuntimeError(
                "Self-play capture failed before any action_request was written. "
                f"workdir={workdir} run_mode={cfg.selfplay.run_mode} map_type={getattr(cfg.selfplay, 'map_type', '?')} "
                f"map_size={getattr(cfg.selfplay, 'map_size', '?')} seed={cfg.selfplay.level_seed}"
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
    node_budget: int | None = None,
) -> tuple[native_mcts.SearchResult, SearchStats]:
    extension = load_native_mcts_extension()
    if extension is None:
        raise RuntimeError("Native MCTS extension is unavailable.")

    started_at = time.perf_counter()
    mode = "nodes" if node_budget is not None else "walltime"
    deadline = None if node_budget is not None else started_at + max(0.0, float(wall_time_sec))
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
    batch_size = max(1, int(search_cfg.batch_size))
    reserve_tree_capacity = getattr(tree, "reserve_tree_capacity", None)
    if reserve_tree_capacity is not None:
        reserve_target = (int(node_budget) + 1) if node_budget is not None else max(4096, batch_size * 1024)
        reserve_tree_capacity(int(reserve_target))
    eval_cache: dict[Any, native_mcts._Evaluation] = {}
    expanded_node_ids: set[int] = set()
    node_depths: dict[int, int] = {0: 0}

    while (
        (deadline is not None and time.perf_counter() < deadline)
        or (node_budget is not None and len(expanded_node_ids) < node_budget)
    ):
        remaining_nodes = None if node_budget is None else max(0, node_budget - len(expanded_node_ids))
        if remaining_nodes == 0:
            break
        selection_frontier = batch_size * 2 if remaining_nodes is None else batch_size
        frontier = selection_frontier if remaining_nodes is None else min(selection_frontier, remaining_nodes)
        selections: list[Any] = []
        eval_messages: list[dict[str, Any]] = []
        eval_index_by_key: dict[Any, int] = {}
        evals_only_batch = getattr(tree, "select_leaf_batch_evals_only", None)
        evals_only_batches = getattr(tree, "select_leaf_batches_evals_only", None)
        completed_frontier = frontier
        if evals_only_batches is not None:
            max_batches = 128 if remaining_nodes is None else 1
            raw_selections, completed_frontier = evals_only_batches(frontier, max_batches, float(search_cfg.c_puct))
        else:
            select_leaf_batch = evals_only_batch or getattr(tree, "select_leaf_batch_compact", tree.select_leaf_batch)
            raw_selections = select_leaf_batch(frontier, float(search_cfg.c_puct))
        for raw_selection in raw_selections:
            if evals_only_batch is not None:
                if len(raw_selection) == 7:
                    selection_id, parent_node_id, parent_action_index, state_key, selection_depth, _turn_depth, raw_leaf_payload = raw_selection
                elif len(raw_selection) == 6:
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
            batch_depth_sum, batch_max_depth, batch_turn_depth_sum, batch_max_turn_depth = _tree_batch_stats(tree)
            stats.depth_sum += int(batch_depth_sum)
            stats.max_depth = max(stats.max_depth, int(batch_max_depth))
            stats.turn_depth_sum += int(batch_turn_depth_sum)
            stats.max_turn_depth = max(stats.max_turn_depth, int(batch_max_turn_depth))
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
        expanded_before_batch = len(expanded_node_ids)
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
        if node_budget is not None and len(expanded_node_ids) == expanded_before_batch:
            break

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
        if str(row["name"]) in _NATIVE_STATIC_EXE_CONTAINER_ROWS:
            continue
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
        self_ms = float(row["self_ms"])
        pct = self_ms / total_ms * 100.0
        if self_ms < min_ms or pct < min_pct:
            continue
        rows.append(
            {
                "source": "function_self",
                "name": row["function"],
                "calls": row["calls"],
                "items": "",
                "time_ms": row["self_ms"],
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_mcts_search_config(path: Path = DEFAULT_MCTS_SEARCH_CONFIG) -> argparse.Namespace:
    if not path.exists():
        raise FileNotFoundError(f"MCTS search profiler config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"MCTS search profiler config must be a JSON object: {path}")
    normalized = {str(key).replace("-", "_"): value for key, value in raw.items()}
    unknown = sorted(set(normalized) - set(_MCTS_SEARCH_DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown mcts_search config key(s): {', '.join(unknown)}")
    values = dict(_MCTS_SEARCH_DEFAULTS)
    values.update(normalized)
    input_mode = "payload" if values.get("payload") is not None else "selfplay"
    values["input_mode"] = input_mode
    values["synthetic"] = False
    if str(values.get("evaluator")) not in {"nn", "static_exe", "bot"}:
        raise ValueError("mcts_search config evaluator must be one of: nn, static_exe, bot")
    if str(values.get("static_eval_variant")) not in {"baseline", "experimental", "experimental-2", "experimental-training"}:
        raise ValueError("mcts_search config static_eval_variant must be one of: baseline, experimental, experimental-2, experimental-training")
    if str(values.get("native_static_search_mode")) not in {"primitive", "turn-macro-exp"}:
        raise ValueError("mcts_search config native_static_search_mode must be one of: primitive, turn-macro-exp")
    if str(values.get("turn_macro_opponent_mode")) not in {"root-max", "maximalist"}:
        raise ValueError("mcts_search config turn_macro_opponent_mode must be one of: root-max, maximalist")
    for key in _PATH_CONFIG_KEYS:
        value = values.get(key)
        if isinstance(value, str) and value:
            parsed_path = Path(value)
            values[key] = parsed_path if parsed_path.is_absolute() else PROJECT_ROOT / parsed_path
    return argparse.Namespace(**values)


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
    total.turn_depth_sum += int(item.turn_depth_sum)
    total.max_turn_depth = max(int(total.max_turn_depth), int(item.max_turn_depth))
    total.inner_searches += int(item.inner_searches)
    total.inner_simulations += int(item.inner_simulations)
    total.inner_nodes_expanded += int(item.inner_nodes_expanded)
    total.static_eval_calls += int(item.static_eval_calls)
    total.greedy_static_calls += int(item.greedy_static_calls)
    total.greedy_static_candidates_considered += int(item.greedy_static_candidates_considered)
    total.greedy_static_child_evals += int(item.greedy_static_child_evals)
    total.greedy_static_child_eval_skips += int(item.greedy_static_child_eval_skips)


def _payload_actions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    actions = payload.get("actions", []) if isinstance(payload, dict) else []
    return [action for action in actions if isinstance(action, dict)]


def _capped_action_count(raw_count: int, model_max_actions: int | None) -> int:
    if model_max_actions is None or int(model_max_actions) < 0:
        return int(raw_count)
    return min(int(raw_count), int(model_max_actions))


def _action_type(action: dict[str, Any]) -> str:
    return str(action.get("type") or "UNKNOWN").upper()


def _action_category(action: dict[str, Any]) -> str:
    action_type = _action_type(action)
    if action.get("unit_id") not in (None, "") or action_type in {
        "MOVE",
        "STEP_MOVE",
        "ATTACK",
        "CAPTURE",
        "CONVERT",
        "RECOVER",
        "HEAL_OTHERS",
        "MAKE_VETERAN",
        "INFILTRATE",
        "DISBAND",
        "UPGRADE_RAMMER",
        "UPGRADE_SCOUT",
        "UPGRADE_BOMBER",
    }:
        return "unit"
    if action.get("city_id") not in (None, "") or action_type in {"SPAWN", "LEVEL_UP"}:
        return "city"
    if action_type in {"RESEARCH", "RESEARCH_TECH"}:
        return "tech"
    if action_type in {
        "BUILD",
        "RESOURCE_GATHERING",
        "BUILD_ROAD",
        "BUILD_EMBASSY",
        "BURN_FOREST",
        "CLEAR_FOREST",
        "GROW_FOREST",
        "DESTROY",
    }:
        return "economy"
    if action_type in {"PROPOSE_PEACE", "ACCEPT_PEACE", "PROPOSE_TREATY", "ACCEPT_TREATY", "CANCEL_TREATY"}:
        return "diplomacy"
    if action_type == "END_TURN":
        return "turn"
    return "other"


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _encoded_nbytes(encoded: EncodedObservation) -> int:
    return sum(
        _tensor_nbytes(tensor)
        for tensor in (
            encoded.board,
            encoded.unit_features,
            encoded.unit_mask,
            encoded.city_features,
            encoded.city_mask,
            encoded.action_features,
            encoded.action_mask,
            encoded.scalar_features,
        )
    )


def _first_tensor_batch_size(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.shape[0]) if value.ndim else 1
    if isinstance(value, (list, tuple)):
        for item in value:
            size = _first_tensor_batch_size(item)
            if size:
                return size
    if isinstance(value, dict):
        for item in value.values():
            size = _first_tensor_batch_size(item)
            if size:
                return size
    return 0


def _query_nvidia_smi() -> dict[str, float] | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    line = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 4:
        return None
    try:
        return {
            "gpu_util_pct": float(parts[0]),
            "gpu_memory_util_pct": float(parts[1]),
            "gpu_memory_used_mb": float(parts[2]),
            "gpu_power_watts": float(parts[3]),
        }
    except ValueError:
        return None


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


def _branching_pressure_rows(branching: BranchingCollector) -> list[dict[str, Any]]:
    rows = [
        {
            "scope": "root raw legal",
            **_branching_summary(branching.root_action_counts),
        },
        {
            "scope": "root model-capped",
            **_branching_summary(branching.root_model_capped_counts),
        },
        {
            "scope": "root searched",
            **_branching_summary(branching.root_searched_counts),
        },
        {
            "scope": "evaluated leaf legal",
            **_branching_summary(branching.eval_action_counts),
        },
        {
            "scope": "policy action slots",
            **_branching_summary(branching.policy_action_counts),
        },
        {
            "scope": "policy padded width",
            **_branching_summary(branching.policy_padded_action_counts),
        },
    ]
    return rows


def _branching_key_stats(branching: BranchingCollector) -> dict[str, str]:
    entropy = branching.root_visit_entropy_sum / max(1, branching.root_visit_samples)
    padding_waste_pct = (
        branching.policy_padding_waste_slots / float(branching.policy_total_slots) * 100.0
        if branching.policy_total_slots
        else 0.0
    )
    return {
        "model_cap_dropped_avg": f"{(sum(branching.root_model_cap_drops) / len(branching.root_model_cap_drops)) if branching.root_model_cap_drops else 0.0:.2f}",
        "top_k_dropped_avg": f"{(sum(branching.root_top_k_drops) / len(branching.root_top_k_drops)) if branching.root_top_k_drops else 0.0:.2f}",
        "avg_root_visit_entropy_bits": f"{entropy:.3f}",
        "avg_effective_root_branching": f"{(2.0 ** entropy) if branching.root_visit_samples else 0.0:.2f}",
        "policy_padding_waste_slots": str(branching.policy_padding_waste_slots),
        "policy_padding_waste_pct": f"{padding_waste_pct:.1f}%",
        "cpu_to_device_mb_est": f"{branching.cpu_to_device_bytes / (1024.0 * 1024.0):.3f}",
        "device_to_cpu_mb_est": f"{branching.device_to_cpu_bytes / (1024.0 * 1024.0):.3f}",
    }


def _action_breadth_rows(branching: BranchingCollector, *, limit: int = 12) -> list[dict[str, Any]]:
    root_total = sum(branching.root_action_types.values())
    leaf_total = sum(branching.eval_action_types.values())
    searched_total = sum(branching.root_searched_action_types.values())
    dropped_total = sum(branching.root_dropped_action_types.values())
    visit_total = max(1, branching.root_visit_samples)
    names = (
        set(branching.root_action_types)
        | set(branching.eval_action_types)
        | set(branching.root_searched_action_types)
        | set(branching.root_dropped_action_types)
        | set(branching.root_visit_by_type)
    )

    def rank(name: str) -> float:
        return (
            float(branching.root_action_types.get(name, 0))
            + float(branching.eval_action_types.get(name, 0))
            + float(branching.root_dropped_action_types.get(name, 0))
            + float(branching.root_visit_by_type.get(name, 0)) * 100.0
        )

    rows = []
    for name in sorted(names, key=rank, reverse=True)[:limit]:
        root_count = int(branching.root_action_types.get(name, 0))
        leaf_count = int(branching.eval_action_types.get(name, 0))
        searched_count = int(branching.root_searched_action_types.get(name, 0))
        dropped_count = int(branching.root_dropped_action_types.get(name, 0))
        visit_share = float(branching.root_visit_by_type.get(name, 0.0)) / float(visit_total)
        rows.append(
            {
                "type": name,
                "root_count": root_count,
                "root_share": f"{(root_count / root_total * 100.0) if root_total else 0.0:.1f}%",
                "leaf_count": leaf_count,
                "leaf_share": f"{(leaf_count / leaf_total * 100.0) if leaf_total else 0.0:.1f}%",
                "searched_share": f"{(searched_count / searched_total * 100.0) if searched_total else 0.0:.1f}%",
                "dropped_top_k_share": f"{(dropped_count / dropped_total * 100.0) if dropped_total else 0.0:.1f}%",
                "visit_share": f"{visit_share * 100.0:.1f}%",
            }
        )
    return rows


def _hardware_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    metrics = [
        "process_rss_mb",
        "process_cpu_pct",
        "system_cpu_pct",
        "system_ram_pct",
        "cuda_allocated_mb",
        "cuda_reserved_mb",
        "cuda_peak_allocated_mb",
        "cuda_peak_reserved_mb",
        "gpu_util_pct",
        "gpu_memory_util_pct",
        "gpu_memory_used_mb",
        "gpu_power_watts",
    ]
    out = []
    for metric in metrics:
        values = []
        for row in rows:
            try:
                values.append(float(row.get(metric, "")))
            except (TypeError, ValueError):
                continue
        if not values:
            continue
        out.append(
            {
                "metric": metric,
                "avg": f"{sum(values) / len(values):.2f}",
                "max": f"{max(values):.2f}",
                "min": f"{min(values):.2f}",
            }
        )
    return out


def _nn_module_rows(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    nn_rows = [row for row in rows if str(row["name"]).startswith("nn.")]
    return sorted(nn_rows, key=lambda row: float(row["total_ms"]), reverse=True)[:limit]


def _exclusive_phase_rows(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    phase_rows = [
        row
        for row in rows
        if str(row["name"]) not in _NATIVE_STATIC_EXE_CONTAINER_ROWS
        and float(row.get("total_ms", 0.0)) > 0.0
    ]
    return sorted(phase_rows, key=lambda row: float(row["total_ms"]), reverse=True)[:limit]


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
    stats_by_id = {
        str(row.get("action_id")): row
        for row in (result.root_stats or [])
        if isinstance(row, dict) and row.get("action_id") not in (None, "")
    }
    rows = []
    for action_id, share in sorted(result.visit_distribution.items(), key=lambda item: float(item[1]), reverse=True)[:limit]:
        action = action_by_id.get(str(action_id), {})
        root_stat = stats_by_id.get(str(action_id), {})
        rows.append(
            {
                "action_id": action_id,
                "type": _action_type(action),
                "visit_share": f"{float(share) * 100.0:.1f}%",
                "prior": _format_float(root_stat.get("prior"), digits=4),
                "visits": root_stat.get("visits", ""),
                "q_mean": _format_float(root_stat.get("q_mean"), digits=4),
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


def _format_float(value: Any, *, digits: int = 3) -> str:
    if value in ("", None):
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _build_payload_cases(args: argparse.Namespace) -> list[PayloadCase]:
    if args.payload is not None and args.synthetic:
        raise ValueError("Use either payload input or synthetic input, not both.")
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
    return PROJECT_ROOT / "debug-logs" / f"mcts-search-{args.evaluator}" / filename


def _ensure_native_static_exe(args: argparse.Namespace) -> Path:
    exe = Path(args.native_static_exe)
    if not exe.is_absolute():
        exe = PROJECT_ROOT / exe
    if exe.exists():
        return exe
    if not bool(getattr(args, "build_native_static_exe", True)):
        raise FileNotFoundError(f"Native static executable not found: {exe}")
    build_script = PROJECT_ROOT / "scripts" / "build_static_bot.ps1"
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


def _native_static_exe_command(exe: Path, cfg: HybridAgentConfig, args: argparse.Namespace, *, using_walltime: bool) -> list[str]:
    command = [
        str(exe),
        "--search-mode",
        str(getattr(args, "native_static_search_mode", "primitive")),
        "--simulations",
        str(int(cfg.search.num_simulations)),
        "--top-k-actions",
        str(int(cfg.search.top_k_actions)),
        "--max-actions",
        str(int(cfg.model.max_actions)),
        "--search-batch-size",
        str(int(cfg.search.batch_size)),
        "--static-eval-variant",
        str(args.static_eval_variant),
        "--seed",
        str(int(getattr(cfg.search, "seed", 123))),
        "--profile-json",
    ]
    if using_walltime:
        command.extend(["--wall-clock-per-action-seconds", str(max(0.0, float(args.wall_time_sec)))])
    if str(getattr(args, "native_static_search_mode", "primitive")) == "turn-macro-exp":
        turn_macro_simulations = getattr(args, "turn_macro_simulations", None)
        if turn_macro_simulations is not None:
            command.extend(["--turn-macro-simulations", str(int(turn_macro_simulations))])
        command.extend(
            [
                "--turn-macro-max-primitives-per-turn",
                str(int(getattr(args, "turn_macro_max_primitives_per_turn", 0))),
                "--turn-macro-max-edges-per-node",
                str(int(getattr(args, "turn_macro_max_edges_per_node", 4))),
                "--turn-macro-outer-c",
                str(float(getattr(args, "turn_macro_outer_c", 1.4))),
                "--turn-macro-c",
                str(float(getattr(args, "turn_macro_c", 1.0))),
                "--turn-macro-prior-weight",
                str(float(getattr(args, "turn_macro_prior_weight", 0.35))),
                "--turn-macro-temperature",
                str(float(getattr(args, "turn_macro_temperature", 1.0))),
                "--turn-macro-inner-simulations",
                str(int(getattr(args, "turn_macro_inner_simulations", 128))),
                "--turn-macro-inner-c-puct",
                str(float(getattr(args, "turn_macro_inner_c_puct", 1.5))),
                "--turn-macro-greedy-eval-top-k",
                str(int(getattr(args, "turn_macro_greedy_eval_top_k", 1))),
                "--turn-macro-opponent-mode",
                str(getattr(args, "turn_macro_opponent_mode", "maximalist")),
            ]
        )
    if bool(args.no_dirichlet):
        command.append("--deterministic")
    return command


_NATIVE_STATIC_EXE_TIMING_ROWS = {
    "select_ms": ("native_static_exe.select", "paths"),
    "apply_action_ms": ("native_static_exe.apply_action", "paths"),
    "static_eval_ms": ("native_static_exe.static_eval", "paths"),
    "node_allocation_ms": ("native_static_exe.node_allocation", "paths"),
    "backup_ms": ("native_static_exe.backup", "paths"),
    "transition_state_copy_ms": ("native_static_exe.transition.state_copy", "paths"),
    "transition_observation_copy_ms": ("native_static_exe.transition.observation_copy", "paths"),
    "transition_action_mutation_ms": ("native_static_exe.transition.action_mutation", "paths"),
    "transition_reveal_sync_ms": ("native_static_exe.transition.reveal_sync", "paths"),
    "transition_regenerate_actions_ms": ("native_static_exe.transition.regenerate_actions", "paths"),
    "transition_hidden_enemy_ms": ("native_static_exe.transition.hidden_enemy", "paths"),
    "search_loop_ms": ("native_static_exe.search_loop", "paths"),
    "macro_exp_factor_build_ms": ("native_static_exe.turn_macro.factor_build", "paths"),
    "macro_exp_select_ms": ("native_static_exe.turn_macro.select", "paths"),
    "inner_search_ms": ("native_static_exe.turn_macro.inner_search", "paths"),
    "root_static_eval_ms": ("native_static_exe.root_static_eval", "root"),
    "root_setup_ms": ("native_static_exe.root_setup", "root"),
    "result_distribution_ms": ("native_static_exe.result_distribution", "root"),
}


_NATIVE_STATIC_EXE_CONTAINER_ROWS = {
    "native_static_exe.process.total",
    "native_static_exe.search",
    "native_static_exe.search_loop",
    "native_static_exe.root_setup",
    "native_static_exe.apply_action",
}


def _profile_timing_ms(profile: dict[str, Any], key: str) -> float:
    raw_timing = profile.get("timing_ms")
    if not isinstance(raw_timing, dict):
        return 0.0
    try:
        return max(0.0, float(raw_timing.get(key, 0.0) or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _add_native_static_exe_timing_rows(
    collector: TimingCollector,
    profile: dict[str, Any],
    *,
    selected_paths: int,
) -> None:
    raw_timing = profile.get("timing_ms")
    if not isinstance(raw_timing, dict):
        return
    for key, (row_name, item_mode) in _NATIVE_STATIC_EXE_TIMING_ROWS.items():
        try:
            elapsed_sec = float(raw_timing.get(key, 0.0) or 0.0) / 1000.0
        except (TypeError, ValueError):
            continue
        items = max(0, int(selected_paths)) if item_mode == "paths" else 1
        collector.add(row_name, elapsed_sec, items=items)

    search_loop_ms = _profile_timing_ms(profile, "search_loop_ms")
    search_accounted_ms = sum(
        _profile_timing_ms(profile, key)
        for key in (
            "select_ms",
            "apply_action_ms",
            "static_eval_ms",
            "node_allocation_ms",
            "backup_ms",
            "macro_exp_select_ms",
            "macro_exp_factor_build_ms",
            "inner_search_ms",
        )
    )
    if search_loop_ms > 0.0:
        collector.add(
            "native_static_exe.search_loop.unattributed",
            max(0.0, search_loop_ms - search_accounted_ms) / 1000.0,
            items=max(0, int(selected_paths)),
        )

    apply_action_ms = _profile_timing_ms(profile, "apply_action_ms")
    transition_accounted_ms = sum(
        _profile_timing_ms(profile, key)
        for key in (
            "transition_state_copy_ms",
            "transition_observation_copy_ms",
            "transition_action_mutation_ms",
            "transition_reveal_sync_ms",
            "transition_regenerate_actions_ms",
            "transition_hidden_enemy_ms",
        )
    )
    if apply_action_ms > 0.0:
        collector.add(
            "native_static_exe.apply_action.unattributed",
            max(0.0, apply_action_ms - transition_accounted_ms) / 1000.0,
            items=max(0, int(selected_paths)),
        )

    root_setup_ms = _profile_timing_ms(profile, "root_setup_ms")
    root_static_eval_ms = _profile_timing_ms(profile, "root_static_eval_ms")
    if root_setup_ms > 0.0:
        collector.add(
            "native_static_exe.root_setup.unattributed",
            max(0.0, root_setup_ms - root_static_eval_ms) / 1000.0,
            items=1,
        )


def _run_static_exe_profile_case(
    case: PayloadCase,
    *,
    exe: Path,
    cfg: HybridAgentConfig,
    args: argparse.Namespace,
    collector: TimingCollector,
    using_walltime: bool,
    repeats: int,
) -> tuple[native_mcts.SearchResult | None, SearchStats, float]:
    stats = SearchStats("walltime" if using_walltime else "simulations", 0.0)
    last_result: native_mcts.SearchResult | None = None
    started_at = time.perf_counter()
    command = _native_static_exe_command(exe, cfg, args, using_walltime=using_walltime)
    payload = dict(case.payload)
    payload["type"] = "action_request"
    input_text = json.dumps(payload, separators=(",", ":")) + "\n"
    for _ in range(max(1, int(repeats))):
        completed, elapsed = _time_call(
            collector,
            "native_static_exe.process.total",
            lambda: subprocess.run(
                command,
                input=input_text,
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
                timeout=max(30.0, float(args.wall_time_sec or 0.0) + 30.0),
                check=False,
            ),
            device=torch.device("cpu"),
            items=1,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Native static executable profile run failed.\n"
                f"command={command}\nstdout:\n{completed.stdout[-4000:]}\nstderr:\n{completed.stderr[-4000:]}"
            )
        output_lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if not output_lines:
            raise RuntimeError("Native static executable produced no stdout response.")
        response = json.loads(output_lines[-1])
        last_result = _result_from_bot_response(case.payload, response)
        profile = response.get("_profile") if isinstance(response, dict) else {}
        if isinstance(profile, dict):
            stats.simulations += int(profile.get("simulations", 0) or 0)
            stats.selected_paths += int(profile.get("selected_paths", 0) or 0)
            stats.expanded_nodes += int(profile.get("expanded_nodes", 0) or 0)
            stats.depth_sum += int(profile.get("depth_sum", 0) or 0)
            stats.max_depth = max(stats.max_depth, int(profile.get("max_depth", 0) or 0))
            stats.turn_depth_sum += int(profile.get("turn_depth_sum", 0) or 0)
            stats.max_turn_depth = max(stats.max_turn_depth, int(profile.get("max_turn_depth", 0) or 0))
            stats.inner_searches += int(profile.get("inner_searches", 0) or 0)
            stats.inner_simulations += int(profile.get("inner_simulations", 0) or 0)
            stats.inner_nodes_expanded += int(profile.get("inner_nodes_expanded", 0) or 0)
            stats.static_eval_calls += int(profile.get("static_eval_calls", 0) or 0)
            stats.greedy_static_calls += int(profile.get("greedy_static_calls", 0) or 0)
            stats.greedy_static_candidates_considered += int(profile.get("greedy_static_candidates_considered", 0) or 0)
            stats.greedy_static_child_evals += int(profile.get("greedy_static_child_evals", 0) or 0)
            stats.greedy_static_child_eval_skips += int(profile.get("greedy_static_child_eval_skips", 0) or 0)
            selected_paths = int(profile.get("selected_paths", 0) or profile.get("completed_paths", 0) or 0)
            search_elapsed_sec = float(profile.get("elapsed_sec", elapsed) or 0.0)
            collector.add("native_static_exe.search", search_elapsed_sec, items=selected_paths)
            collector.add("native_static_exe.process.overhead", max(0.0, elapsed - search_elapsed_sec), items=1)
            _add_native_static_exe_timing_rows(collector, profile, selected_paths=selected_paths)
    stats.elapsed_sec = time.perf_counter() - started_at
    return last_result, stats, stats.elapsed_sec


def _run_bot_profile_case(
    case: PayloadCase,
    *,
    bot: HybridRLBot,
    collector: TimingCollector,
    using_walltime: bool,
    repeats: int,
) -> tuple[native_mcts.SearchResult | None, SearchStats, float]:
    global _STATIC_TREE_DEPTH_SUM, _STATIC_TREE_MAX_DEPTH, _STATIC_TREE_TURN_DEPTH_SUM, _STATIC_TREE_MAX_TURN_DEPTH, _STATIC_TREE_SELECTED_PATHS, _STATIC_TREE_EXPANDED_NODE_IDS
    _STATIC_TREE_DEPTH_SUM = 0
    _STATIC_TREE_MAX_DEPTH = 0
    _STATIC_TREE_TURN_DEPTH_SUM = 0
    _STATIC_TREE_MAX_TURN_DEPTH = 0
    _STATIC_TREE_SELECTED_PATHS = 0
    _STATIC_TREE_EXPANDED_NODE_IDS = set()
    last_result: native_mcts.SearchResult | None = None
    stats = SearchStats("walltime" if using_walltime else "simulations", 0.0)
    started_at = time.perf_counter()
    for _ in range(max(1, int(repeats))):
        depth_sum_before = _STATIC_TREE_DEPTH_SUM
        turn_depth_sum_before = _STATIC_TREE_TURN_DEPTH_SUM
        selected_paths_before = _STATIC_TREE_SELECTED_PATHS
        expanded_before = len(_STATIC_TREE_EXPANDED_NODE_IDS)
        bot.reset_episode()
        response, _elapsed = _time_call(
            collector,
            "bot.choose_action.total",
            lambda: bot.choose_action(case.payload),
            device=bot.device,
            items=1,
        )
        last_result = _result_from_bot_record(case.payload, bot, response)
        selected_delta = max(0, _STATIC_TREE_SELECTED_PATHS - selected_paths_before)
        expanded_delta = max(0, len(_STATIC_TREE_EXPANDED_NODE_IDS) - expanded_before)
        stats.depth_sum += max(0, _STATIC_TREE_DEPTH_SUM - depth_sum_before)
        stats.max_depth = max(stats.max_depth, _STATIC_TREE_MAX_DEPTH)
        stats.turn_depth_sum += max(0, _STATIC_TREE_TURN_DEPTH_SUM - turn_depth_sum_before)
        stats.max_turn_depth = max(stats.max_turn_depth, _STATIC_TREE_MAX_TURN_DEPTH)
        stats.selected_paths += selected_delta
        stats.expanded_nodes += expanded_delta
        stats.simulations += expanded_delta
    _sync_if_needed(bot.device)
    elapsed = time.perf_counter() - started_at
    stats.elapsed_sec = elapsed
    return last_result, stats, elapsed


def _run_one_profile_case(
    case: PayloadCase,
    *,
    evaluator_mode: str,
    model: HybridPolicyValueNet | None,
    bot: HybridRLBot | None,
    native_static_exe: Path | None,
    collector: TimingCollector,
    cfg: HybridAgentConfig,
    args: argparse.Namespace,
    device: torch.device,
    using_walltime: bool,
    wall_time_sec: float,
    repeats: int,
) -> tuple[native_mcts.SearchResult | None, SearchStats, float]:
    global _STATIC_TREE_DEPTH_SUM, _STATIC_TREE_MAX_DEPTH, _STATIC_TREE_SELECTED_PATHS, _STATIC_TREE_EXPANDED_NODE_IDS
    if evaluator_mode == "static_exe":
        if native_static_exe is None:
            raise RuntimeError("static_exe evaluator mode requires a native static executable path.")
        return _run_static_exe_profile_case(
            case,
            exe=native_static_exe,
            cfg=cfg,
            args=args,
            collector=collector,
            using_walltime=using_walltime,
            repeats=repeats,
        )

    if evaluator_mode == "bot":
        if bot is None:
            raise RuntimeError("Bot evaluator mode requires a HybridRLBot instance.")
        return _run_bot_profile_case(
            case,
            bot=bot,
            collector=collector,
            using_walltime=using_walltime,
            repeats=repeats,
        )

    last_result: native_mcts.SearchResult | None = None
    stats = SearchStats("walltime" if using_walltime else "simulations", 0.0)
    started_at = time.perf_counter()
    for _ in range(max(1, int(repeats))):
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
                node_budget=int(cfg.search.num_simulations),
            )
            _add_stats(stats, run_stats)
    _sync_if_needed(device)
    elapsed = time.perf_counter() - started_at
    stats.elapsed_sec = elapsed
    return last_result, stats, elapsed


def main() -> int:
    config_path = DEFAULT_MCTS_SEARCH_CONFIG
    if len(sys.argv) in {3} and sys.argv[1] == "--config":
        config_path = Path(sys.argv[2])
    elif len(sys.argv) > 1:
        raise SystemExit(
            "mcts_search is config-only. Edit py/profiling/configs/mcts_search.json, "
            "or pass --config PATH, then run: python -m profiling.mcts_search"
        )
    args = _load_mcts_search_config(config_path)

    if args.position_csv is None:
        args.position_csv = _default_mcts_search_output_path(args, "positions.csv")
    if args.csv is None:
        args.csv = _default_mcts_search_output_path(args, "timing.csv")
    if args.branching_csv is None:
        args.branching_csv = _default_mcts_search_output_path(args, "branching.csv")
    if args.action_csv is None:
        args.action_csv = _default_mcts_search_output_path(args, "actions.csv")
    if args.hardware_csv is None:
        args.hardware_csv = _default_mcts_search_output_path(args, "hardware.csv")
    if args.nn_module_csv is None:
        args.nn_module_csv = _default_mcts_search_output_path(args, "nn_modules.csv")
    if args.details_json is None:
        args.details_json = _default_mcts_search_output_path(args, "profile.json")
    if not args.function_profile:
        args.profile_csv = None
    elif args.profile_csv is None:
        args.profile_csv = _default_mcts_search_output_path(args, "functions.csv")

    os.environ["TRIBES_STATIC_EVAL_VARIANT"] = str(args.static_eval_variant)

    extension = None if args.evaluator == "static_exe" else load_native_mcts_extension()
    if extension is None and args.evaluator != "static_exe":
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
    if using_walltime:
        cfg.selfplay.wall_clock_per_action_seconds = max(0.0, float(args.wall_time_sec))

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model: HybridPolicyValueNet | None = None
    bot: HybridRLBot | None = None
    native_static_exe: Path | None = None
    bot_replay_tmp: tempfile.TemporaryDirectory[str] | None = None
    checkpoint_status = "random_init"
    if args.evaluator in {"nn", "bot"}:
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
        if args.evaluator == "bot":
            bot_replay_tmp = tempfile.TemporaryDirectory(prefix="tribes_mcts_profile_bot_replay_")
            bot = HybridRLBot(
                cfg,
                Path(args.checkpoint) if args.checkpoint is not None else Path(),
                Path(bot_replay_tmp.name),
                model=model,
                device=device,
                native_available=True,
                warmup=False,
            )
            checkpoint_status = f"bot:{checkpoint_status}"
    elif args.evaluator == "static_exe":
        native_static_exe = _ensure_native_static_exe(args)
        checkpoint_status = f"static_exe:{native_static_exe}"
    else:
        checkpoint_status = "unknown"

    cases = _build_payload_cases(args)
    if not cases:
        raise RuntimeError("No payload cases were captured/loaded.")

    collector = TimingCollector()
    branching = BranchingCollector()
    original_evaluator = _install_timed_evaluator(collector, branching) if args.evaluator in {"nn", "bot"} else None
    profile_device = bot.device if bot is not None else device
    original_tree_cls = _install_timed_tree(extension, collector, profile_device) if extension is not None else None
    profile = cProfile.Profile()
    per_position_rows: list[dict[str, Any]] = []
    last_result: native_mcts.SearchResult | None = None
    total_stats = SearchStats("walltime" if using_walltime else "simulations", 0.0)
    nn_profiler: NNModuleProfiler | None = None
    hardware_sampler: HardwareSampler | None = None

    try:
        if args.warmup > 0:
            print(f"[warmup] {args.warmup} run(s) per payload; not included in benchmark timing", flush=True)
        for case in cases:
            for _ in range(max(0, int(args.warmup))):
                if args.evaluator == "static_exe":
                    if native_static_exe is None:
                        raise RuntimeError("static_exe evaluator mode requires a native static executable path.")
                    _run_static_exe_profile_case(
                        case,
                        exe=native_static_exe,
                        cfg=cfg,
                        args=args,
                        collector=collector,
                        using_walltime=using_walltime,
                        repeats=1,
                    )
                    continue
                if args.evaluator == "bot":
                    if bot is None:
                        raise RuntimeError("Bot evaluator mode requires a HybridRLBot instance.")
                    bot.reset_episode()
                    bot.choose_action(case.payload)
                    continue
                if using_walltime:
                    if model is None:
                        raise RuntimeError("NN evaluator mode requires a model.")
                    _run_native_mcts_walltime(case.payload, model, cfg.search, cfg.model, device, min(float(args.wall_time_sec), 0.2))
                else:
                    if model is None:
                        raise RuntimeError("NN evaluator mode requires a model.")
                    native_mcts.run_native_mcts(case.payload, model, cfg.search, cfg.model, device)

        collector = TimingCollector()
        branching = BranchingCollector()
        if original_evaluator is not None:
            native_mcts._evaluate_messages = original_evaluator
        if extension is not None and original_tree_cls is not None:
            extension.NativeMCTS = original_tree_cls
        original_evaluator = _install_timed_evaluator(collector, branching) if args.evaluator in {"nn", "bot"} else None
        if extension is not None:
            _install_timed_tree(extension, collector, profile_device)
        if args.nn_module_profile and model is not None:
            nn_profiler = NNModuleProfiler(collector, device)
            nn_profiler.install(model)

        _sync_if_needed(device)
        benchmark_started_at = time.perf_counter()
        if args.hardware_profile:
            hardware_sampler = HardwareSampler(
                device=profile_device if isinstance(profile_device, torch.device) else torch.device(profile_device),
                interval_sec=max(1, int(args.hardware_sample_interval_ms)) / 1000.0,
            )
            hardware_sampler.__enter__()
        if args.function_profile:
            profile.enable()
        for index, case in enumerate(cases, start=1):
            summary = _payload_summary(case.payload)
            branching.add_root(case.payload, label=case.label, model_max_actions=cfg.model.max_actions)
            print(
                f"[benchmark {index}/{len(cases)}] {case.label} "
                f"actions={summary['actions']} board_size={summary['board_size']} tick={summary['tick']}",
                flush=True,
            )
            last_result, run_stats, elapsed = _run_one_profile_case(
                case,
                evaluator_mode=args.evaluator,
                model=model,
                bot=bot,
                native_static_exe=native_static_exe,
                collector=collector,
                cfg=cfg,
                args=args,
                device=device,
                using_walltime=using_walltime,
                wall_time_sec=float(args.wall_time_sec) if using_walltime else 0.0,
                repeats=max(1, int(args.repeats)),
            )
            _add_stats(total_stats, run_stats)
            branching.add_result(case.payload, last_result, label=case.label, model_max_actions=cfg.model.max_actions)
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
                "expanded_nodes": run_stats.expanded_nodes,
                "expanded_nodes_per_sec": _format_rate(run_stats.expanded_nodes, elapsed),
                "avg_depth": f"{run_stats.average_depth:.2f}",
                "max_depth": run_stats.max_depth,
                "avg_turn_depth": f"{run_stats.average_turn_depth:.2f}",
                "max_turn_depth": run_stats.max_turn_depth,
                "inner_searches": run_stats.inner_searches,
                "inner_simulations": run_stats.inner_simulations,
                "inner_simulations_per_sec": _format_rate(run_stats.inner_simulations, elapsed),
                "inner_nodes_expanded": run_stats.inner_nodes_expanded,
                "inner_nodes_expanded_per_sec": _format_rate(run_stats.inner_nodes_expanded, elapsed),
                "static_eval_calls": run_stats.static_eval_calls,
                "static_eval_calls_per_sec": _format_rate(run_stats.static_eval_calls, elapsed),
                "greedy_static_calls": run_stats.greedy_static_calls,
                "greedy_static_candidates_considered": run_stats.greedy_static_candidates_considered,
                "greedy_static_child_evals": run_stats.greedy_static_child_evals,
                "greedy_static_child_eval_skips": run_stats.greedy_static_child_eval_skips,
                "eval_batches": run_stats.eval_batches,
                "eval_positions": run_stats.eval_positions,
                "eval_cache_hits": run_stats.eval_cache_hits,
            }
            per_position_rows.append(row)
        _sync_if_needed(device)
        if args.function_profile:
            profile.disable()
        if hardware_sampler is not None:
            hardware_sampler.__exit__(None, None, None)
        elapsed = time.perf_counter() - benchmark_started_at
        total_stats.elapsed_sec = elapsed
    finally:
        if nn_profiler is not None:
            nn_profiler.remove()
        if hardware_sampler is not None and not hardware_sampler._stop.is_set():
            hardware_sampler.__exit__(None, None, None)
        if original_evaluator is not None:
            native_mcts._evaluate_messages = original_evaluator
        if extension is not None and original_tree_cls is not None:
            extension.NativeMCTS = original_tree_cls
        if bot_replay_tmp is not None:
            bot_replay_tmp.cleanup()

    print(
        f"MCTS profile: positions={len(cases)} repeats_per_position={max(1, int(args.repeats))} "
        f"mode={'walltime' if using_walltime else 'fixed_nodes'} "
        f"node_budget={'walltime' if using_walltime else cfg.search.num_simulations} "
        f"wall_time_sec_per_position={args.wall_time_sec if using_walltime else 'n/a'} "
        f"batch={cfg.search.batch_size} evaluator={args.evaluator} static_eval_variant={args.static_eval_variant} "
        f"device={device} checkpoint={checkpoint_status} "
        f"source={'payload' if args.payload else args.selfplay_run_mode} "
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
                ("expanded_nodes", "nodes"),
                ("expanded_nodes_per_sec", "nodes/s"),
                ("avg_depth", "avg_depth"),
                ("max_depth", "max_depth"),
                ("avg_turn_depth", "avg_turn_depth"),
                ("max_turn_depth", "max_turn_depth"),
                ("inner_simulations", "inner_sims"),
                ("inner_simulations_per_sec", "inner_sims/s"),
                ("inner_nodes_expanded", "inner_nodes"),
                ("static_eval_calls", "static_evals"),
                ("static_eval_calls_per_sec", "static_evals/s"),
                ("greedy_static_child_evals", "greedy_evals"),
                ("greedy_static_child_eval_skips", "greedy_skips"),
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
            f"avg_turn_depth={total_stats.average_turn_depth:.2f} "
            f"max_turn_depth={total_stats.max_turn_depth} "
            f"inner_simulations={total_stats.inner_simulations} "
            f"inner_simulations_per_sec={_format_rate(total_stats.inner_simulations, elapsed)} "
            f"inner_searches={total_stats.inner_searches} "
            f"inner_nodes_expanded={total_stats.inner_nodes_expanded} "
            f"static_eval_calls={total_stats.static_eval_calls} "
            f"static_eval_calls_per_sec={_format_rate(total_stats.static_eval_calls, elapsed)} "
            f"greedy_static_child_evals={total_stats.greedy_static_child_evals} "
            f"greedy_static_child_eval_skips={total_stats.greedy_static_child_eval_skips} "
            f"eval_batches={total_stats.eval_batches} "
            f"eval_positions={total_stats.eval_positions} "
            f"eval_cache_hits={total_stats.eval_cache_hits}"
        )

    print("\nTree search efficiency")
    eval_batch_fill = (
        total_stats.eval_positions / float(total_stats.eval_batches * max(1, int(cfg.search.batch_size)))
        if total_stats.eval_batches
        else 0.0
    )
    cache_total = total_stats.eval_positions + total_stats.eval_cache_hits
    cache_hit_rate = total_stats.eval_cache_hits / float(cache_total) * 100.0 if cache_total else 0.0
    print(
        f"expanded_nodes={total_stats.expanded_nodes} "
        f"expanded_nodes_per_sec={_format_rate(total_stats.expanded_nodes, elapsed)} "
        f"expanded_per_selected_path={(total_stats.expanded_nodes / max(1, total_stats.selected_paths)):.3f} "
        f"eval_cache_hit_rate={cache_hit_rate:.1f}% "
        f"eval_batch_fill={eval_batch_fill * 100.0:.1f}%"
    )

    print("\nBranching and action-space pressure")
    branching_rows = _branching_pressure_rows(branching)
    print(_format_table(branching_rows, [("scope", "scope"), ("samples", "samples"), ("avg", "avg"), ("p50", "p50"), ("p90", "p90"), ("max", "max")]))
    key_stats = _branching_key_stats(branching)
    print(
        "Key pressure: "
        + " ".join(f"{name}={value}" for name, value in key_stats.items())
    )

    action_breadth_rows = _action_breadth_rows(branching, limit=max(1, int(args.section_limit)))
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
                    ("searched_share", "searched_%"),
                    ("dropped_top_k_share", "dropped_%"),
                    ("visit_share", "root_visit_%"),
                ],
            )
        )

    top_action_rows = _top_root_action_rows(cases[-1].payload, last_result, limit=max(1, int(args.section_limit)))
    if top_action_rows:
        print("\nTop root actions in final position")
        print(
            _format_table(
                top_action_rows,
                [
                    ("action_id", "action_id"),
                    ("type", "type"),
                    ("visit_share", "visit_%"),
                    ("prior", "prior"),
                    ("visits", "visits"),
                    ("q_mean", "q_mean"),
                    ("unit", "unit"),
                    ("city", "city"),
                    ("x", "x"),
                    ("y", "y"),
                ],
            )
        )

    rows = _timing_rows(collector, elapsed)
    if args.csv is not None:
        _write_csv(args.csv, rows)
    exclusive_phase_rows = _exclusive_phase_rows(rows, limit=max(1, int(args.function_limit)))
    if exclusive_phase_rows:
        print("\nExclusive phase time")
        print(
            _format_table(
                exclusive_phase_rows,
                [("name", "phase"), ("calls", "calls"), ("items", "items"), ("total_ms", "total_ms"), ("avg_ms", "avg_ms"), ("pct", "%")],
            )
        )
    nn_module_rows = _nn_module_rows(rows, limit=max(1, int(args.section_limit)))
    if args.nn_module_csv is not None:
        _write_csv(args.nn_module_csv, [row for row in rows if str(row["name"]).startswith("nn.")])
    if nn_module_rows:
        print("\nNN module time")
        print(_format_table(nn_module_rows, [("name", "module"), ("calls", "calls"), ("items", "items"), ("total_ms", "total_ms"), ("avg_ms", "avg_ms"), ("pct", "%")]))

    hardware_rows = hardware_sampler.rows() if hardware_sampler is not None else []
    hardware_summary_rows = _hardware_summary_rows(hardware_rows)
    if args.hardware_csv is not None and args.hardware_profile:
        _write_csv(args.hardware_csv, hardware_rows)
    if hardware_summary_rows:
        print("\nHardware utilization")
        print(_format_table(hardware_summary_rows[: max(1, int(args.section_limit))], [("metric", "metric"), ("avg", "avg"), ("max", "max"), ("min", "min")]))

    profile_table = _profile_rows(profile, root=PY_ROOT.parent) if args.function_profile else []
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
    if args.branching_csv is not None:
        _write_csv(args.branching_csv, branching.branching_rows)
        written.append(f"branching={args.branching_csv}")
    if args.action_csv is not None:
        _write_csv(args.action_csv, branching.action_rows)
        written.append(f"actions={args.action_csv}")
    if args.csv is not None:
        written.append(f"timing={args.csv}")
    if args.profile_csv is not None:
        written.append(f"functions={args.profile_csv}")
    if args.nn_module_csv is not None and nn_module_rows:
        written.append(f"nn_modules={args.nn_module_csv}")
    if args.hardware_csv is not None and args.hardware_profile:
        written.append(f"hardware={args.hardware_csv}")
    if args.details_json is not None:
        details = {
            "run": {
                "positions": len(cases),
                "repeats_per_position": max(1, int(args.repeats)),
                "mode": "walltime" if using_walltime else "fixed_nodes",
                "node_budget": "walltime" if using_walltime else cfg.search.num_simulations,
                "wall_time_sec_per_position": args.wall_time_sec if using_walltime else None,
                "batch_size": cfg.search.batch_size,
                "evaluator": args.evaluator,
                "static_eval_variant": args.static_eval_variant,
                "device": str(device),
                "checkpoint": checkpoint_status,
                "elapsed_sec": elapsed,
            },
            "tree_efficiency": {
                "simulations": total_stats.simulations,
                "simulations_per_sec": _format_rate(total_stats.simulations, elapsed),
                "selected_paths": total_stats.selected_paths,
                "selected_paths_per_sec": _format_rate(total_stats.selected_paths, elapsed),
                "expanded_nodes": total_stats.expanded_nodes,
                "expanded_nodes_per_sec": _format_rate(total_stats.expanded_nodes, elapsed),
                "avg_depth": f"{total_stats.average_depth:.3f}",
                "max_depth": total_stats.max_depth,
                "avg_turn_depth": f"{total_stats.average_turn_depth:.3f}",
                "max_turn_depth": total_stats.max_turn_depth,
                "inner_searches": total_stats.inner_searches,
                "inner_simulations": total_stats.inner_simulations,
                "inner_simulations_per_inner_search": (
                    f"{(total_stats.inner_simulations / max(1, total_stats.inner_searches)):.3f}"
                    if total_stats.inner_searches
                    else "0.000"
                ),
                "inner_simulations_per_sec": _format_rate(total_stats.inner_simulations, elapsed),
                "inner_nodes_expanded": total_stats.inner_nodes_expanded,
                "inner_nodes_expanded_per_sec": _format_rate(total_stats.inner_nodes_expanded, elapsed),
                "static_eval_calls": total_stats.static_eval_calls,
                "static_eval_calls_per_sec": _format_rate(total_stats.static_eval_calls, elapsed),
                "greedy_static_calls": total_stats.greedy_static_calls,
                "greedy_static_candidates_considered": total_stats.greedy_static_candidates_considered,
                "greedy_static_child_evals": total_stats.greedy_static_child_evals,
                "greedy_static_child_eval_skips": total_stats.greedy_static_child_eval_skips,
                "greedy_static_child_eval_skip_rate_pct": (
                    f"{(100.0 * total_stats.greedy_static_child_eval_skips / max(1, total_stats.greedy_static_child_evals + total_stats.greedy_static_child_eval_skips)):.3f}"
                ),
                "eval_batches": total_stats.eval_batches,
                "eval_positions": total_stats.eval_positions,
                "eval_cache_hits": total_stats.eval_cache_hits,
                "eval_cache_hit_rate_pct": f"{cache_hit_rate:.3f}",
                "eval_batch_fill_pct": f"{eval_batch_fill * 100.0:.3f}",
            },
            "branching": {
                "summary_rows": branching_rows,
                "key_stats": key_stats,
            },
            "action_type_breadth": action_breadth_rows,
            "top_root_actions": top_action_rows,
            "exclusive_phase_time": exclusive_phase_rows,
            "hotspots": hotspot_rows,
            "nn_module_time": nn_module_rows,
            "hardware": hardware_summary_rows,
            "csv": written,
        }
        _write_json(args.details_json, details)
        written.append(f"details={args.details_json}")
    if written:
        print("\nCSV: " + " ".join(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
