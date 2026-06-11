from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PY_ROOT = PROJECT_ROOT / "py"
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from nn.bot_agent import compatible_state_dict
from nn.encoding import encode_observation, normalize_message
from nn.model import HybridPolicyValueNet
from profiling.config import load_config_defaults
from search.native import run_native_hybrid_mcts, run_native_mcts, run_native_static_mcts
from search.native.hybrid_mcts import _mix_evaluation
from search.native.mcts import _Evaluation
from search.native.static_mcts import _evaluate_static_messages
from training.config import HybridAgentConfig


DEFAULT_PAYLOAD_DIR = PROJECT_ROOT / "debug-logs" / "mcts-profile-payloads"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "debug-logs" / "policy-search-alignment"
DEFAULT_POLICY_SEARCH_ALIGNMENT_CONFIG = PY_ROOT / "profiling" / "configs" / "policy_search_alignment.json"
STATIC_EVAL_VARIANTS = ("baseline", "experimental")

POSITION_FIELDNAMES = [
    "label",
    "payload_path",
    "actions",
    "search_sec",
    "policy_value",
    "search_root_value",
    "policy_top_action_id",
    "policy_top_type",
    "search_top_action_id",
    "search_top_type",
    "search_top_policy_rank",
    "search_top_policy_prior",
    "search_top_visit_share",
    "top1_match",
    "overlap_at_5",
    "overlap_at_10",
    "overlap_at_32",
    "policy_entropy_bits",
    "visit_entropy_bits",
    "kl_policy_visit_bits",
    "kl_visit_policy_bits",
    "js_divergence_bits",
]

ACTION_FIELDNAMES = [
    "label",
    "action_id",
    "policy_rank",
    "search_rank",
    "policy_prior",
    "visit_share",
    "type",
    "unit_id",
    "city_id",
    "x",
    "y",
]


@dataclass
class PayloadCase:
    label: str
    path: Path
    payload: dict[str, Any]


@dataclass
class PolicyResult:
    priors: list[float]
    value: float


@dataclass
class AlignmentCaseMetrics:
    top1_match: bool
    policy_entropy_bits: float
    visit_entropy_bits: float
    kl_policy_visit_bits: float
    kl_visit_policy_bits: float
    js_divergence_bits: float


class _CsvRowWriter:
    def __init__(self, path: Path, fieldnames: list[str]) -> None:
        self.path = path
        self.fieldnames = fieldnames
        self._handle: Any | None = None
        self._writer: csv.DictWriter[str] | None = None

    def __enter__(self) -> "_CsvRowWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.fieldnames)
        self._writer.writeheader()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._handle is not None:
            self._handle.close()
        self._handle = None
        self._writer = None

    def writerow(self, row: dict[str, Any]) -> None:
        if self._writer is None:
            raise RuntimeError("CSV writer is not open.")
        self._writer.writerow(row)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_payload(path: Path) -> dict[str, Any] | None:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        payload = payload[0] if payload else None
    if not isinstance(payload, dict):
        return None
    actions = payload.get("actions")
    observation = payload.get("observation")
    if not isinstance(actions, list) or not isinstance(observation, dict):
        return None
    return normalize_message(payload)


def _load_cases(paths: list[Path], payload_dir: Path | None, limit: int | None) -> list[PayloadCase]:
    candidates: list[Path] = []
    candidates.extend(paths)
    if payload_dir is not None:
        candidates.extend(sorted(payload_dir.glob("*.json")))
    seen: set[Path] = set()
    cases: list[PayloadCase] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        payload = _load_payload(resolved)
        if payload is None:
            continue
        cases.append(PayloadCase(label=resolved.stem, path=resolved, payload=payload))
        if limit is not None and len(cases) >= limit:
            break
    return cases


def _action_id(action: dict[str, Any]) -> str:
    return str(action.get("id"))


def _action_type(action: dict[str, Any]) -> str:
    payload = action.get("payload") if isinstance(action.get("payload"), dict) else {}
    return str(action.get("type") or payload.get("type") or "UNKNOWN").upper()


def _action_summary(action: dict[str, Any]) -> dict[str, Any]:
    payload = action.get("payload") if isinstance(action.get("payload"), dict) else {}
    return {
        "type": _action_type(action),
        "unit_id": action.get("unit_id", payload.get("unit_id", payload.get("unitId", ""))),
        "city_id": action.get("city_id", payload.get("city_id", payload.get("cityId", ""))),
        "x": action.get("x", payload.get("x", "")),
        "y": action.get("y", payload.get("y", "")),
    }


def _rank_map(scores: list[float], *, reverse: bool = True) -> dict[int, int]:
    ordered = sorted(range(len(scores)), key=lambda index: (scores[index], -index if reverse else index), reverse=reverse)
    return {index: rank for rank, index in enumerate(ordered, start=1)}


def _top_indexes(scores: list[float], limit: int) -> list[int]:
    return sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[: max(0, int(limit))]


def _distribution(values: list[float], epsilon: float = 1e-12) -> list[float]:
    clipped = [max(0.0, float(value)) + epsilon for value in values]
    total = sum(clipped)
    return [value / total for value in clipped] if total > 0.0 else []


def _kl_bits(p: list[float], q: list[float]) -> float:
    return sum(pi * math.log(pi / qi, 2.0) for pi, qi in zip(p, q) if pi > 0.0 and qi > 0.0)


def _entropy_bits(distribution: list[float]) -> float:
    return -sum(value * math.log(value, 2.0) for value in distribution if value > 0.0)


def _js_bits(p: list[float], q: list[float]) -> float:
    midpoint = [(pi + qi) * 0.5 for pi, qi in zip(p, q)]
    return 0.5 * _kl_bits(p, midpoint) + 0.5 * _kl_bits(q, midpoint)


def _overlap_at(policy_scores: list[float], search_scores: list[float], k: int) -> float:
    if not policy_scores or not search_scores:
        return 0.0
    limit = min(int(k), len(policy_scores), len(search_scores))
    if limit <= 0:
        return 0.0
    policy_top = set(_top_indexes(policy_scores, limit))
    search_top = set(_top_indexes(search_scores, limit))
    return len(policy_top & search_top) / float(limit)


def _load_model(cfg: HybridAgentConfig, checkpoint: Path, device: torch.device) -> HybridPolicyValueNet:
    model = HybridPolicyValueNet(cfg.model).to(device)
    model.eval()
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state_dict = payload.get("model", payload) if isinstance(payload, dict) else payload
        model.load_state_dict(compatible_state_dict(model, state_dict), strict=False)
    else:
        print(f"[warn] checkpoint not found; using randomly initialized model: {checkpoint}", flush=True)
    return model


def _nn_policy(payload: dict[str, Any], cfg: HybridAgentConfig, model: HybridPolicyValueNet, device: torch.device) -> PolicyResult:
    actions = list(payload.get("actions", []))[: int(cfg.model.max_actions)]
    encoded = encode_observation(
        {"player_id": int(payload["player_id"]), "observation": payload["observation"], "actions": actions},
        cfg.model,
        compact=True,
    ).to(device)
    with torch.inference_mode():
        output = model(encoded)
    logits = output.policy_logits[0, : len(actions)]
    priors = torch.softmax(logits, dim=-1).detach().cpu().tolist() if actions else []
    value = float(output.value[0].detach().flatten()[0].item())
    return PolicyResult([float(prior) for prior in priors], value)


def _static_policy(payload: dict[str, Any], cfg: HybridAgentConfig) -> PolicyResult:
    actions = list(payload.get("actions", []))[: int(cfg.model.max_actions)]
    evaluation = _evaluate_static_messages(
        [{"player_id": int(payload["player_id"]), "observation": payload["observation"], "actions": actions}],
        int(cfg.model.max_actions),
    )[0]
    return PolicyResult([float(prior) for prior in evaluation.priors], float(evaluation.value))


def _policy(
    payload: dict[str, Any],
    cfg: HybridAgentConfig,
    evaluator: str,
    model: HybridPolicyValueNet | None,
    device: torch.device,
) -> PolicyResult:
    if evaluator == "static":
        return _static_policy(payload, cfg)
    if evaluator == "nn":
        if model is None:
            raise RuntimeError("NN evaluator requires a model.")
        return _nn_policy(payload, cfg, model, device)
    if evaluator == "hybrid":
        if model is None:
            raise RuntimeError("Hybrid evaluator requires a model.")
        nn_eval = _nn_policy(payload, cfg, model, device)
        static_eval = _static_policy(payload, cfg)
        mixed = _mix_evaluation(
            _Evaluation(nn_eval.priors, nn_eval.value),
            _Evaluation(static_eval.priors, static_eval.value),
            float(cfg.search.static_policy_weight),
            float(cfg.search.static_value_weight),
        )
        return PolicyResult([float(prior) for prior in mixed.priors], float(mixed.value))
    raise ValueError(f"unknown evaluator: {evaluator}")


def _run_search(
    payload: dict[str, Any],
    cfg: HybridAgentConfig,
    evaluator: str,
    model: HybridPolicyValueNet | None,
    device: torch.device,
):
    if evaluator == "static":
        return run_native_static_mcts(payload, cfg.search, cfg.model)
    if evaluator == "nn":
        if model is None:
            raise RuntimeError("NN evaluator requires a model.")
        return run_native_mcts(payload, model, cfg.search, cfg.model, device)
    if evaluator == "hybrid":
        if model is None:
            raise RuntimeError("Hybrid evaluator requires a model.")
        return run_native_hybrid_mcts(payload, model, cfg.search, cfg.model, device)
    raise ValueError(f"unknown evaluator: {evaluator}")


def _capture_thresholds(ranks: list[int], thresholds: list[int]) -> dict[str, Any]:
    if not ranks:
        return {}
    return {
        str(threshold): {
            "captures": sum(1 for rank in ranks if rank <= threshold),
            "capture_rate": sum(1 for rank in ranks if rank <= threshold) / float(len(ranks)),
        }
        for threshold in thresholds
    }


def _configure(args: argparse.Namespace) -> HybridAgentConfig:
    cfg = HybridAgentConfig()
    cfg.search.num_simulations = int(args.simulations)
    cfg.search.batch_size = int(args.batch_size)
    cfg.search.top_k_actions = int(args.top_k_actions)
    cfg.search.sample_action = False
    cfg.search.root_temperature = float(args.root_temperature)
    cfg.search.dirichlet_epsilon = float(args.dirichlet_epsilon)
    cfg.search.static_policy_weight = float(args.static_policy_weight)
    cfg.search.static_value_weight = float(args.static_value_weight)
    cfg.model.max_actions = int(args.max_actions)
    return cfg


def _run_file_stem(args: argparse.Namespace, cfg: HybridAgentConfig) -> str:
    top_k = int(cfg.search.top_k_actions)
    top_k_label = "all-actions" if top_k <= 0 else f"topk{top_k}"
    return f"{args.static_eval_variant}_sims{int(cfg.search.num_simulations)}_{top_k_label}"


def _run_alignment(args: argparse.Namespace) -> None:
    os.environ["TRIBES_STATIC_EVAL_VARIANT"] = str(args.static_eval_variant)
    cfg = _configure(args)
    payload_paths = [Path(path) for path in args.payload]
    payload_dir = None if args.no_payload_dir else Path(args.payload_dir)
    cases = _load_cases(payload_paths, payload_dir, args.positions)
    if not cases:
        raise RuntimeError("No payloads found. Pass --payload or generate/reuse debug-logs/mcts-profile-payloads.")

    device = torch.device(args.device)
    model = None
    if args.evaluator in {"nn", "hybrid"}:
        model = _load_model(cfg, Path(args.checkpoint), device)

    output_dir = Path(args.output_dir) / str(args.static_eval_variant)
    final_policy_ranks: list[int] = []
    top1_matches = 0
    metric_sums = {
        "policy_entropy_bits": 0.0,
        "visit_entropy_bits": 0.0,
        "kl_policy_visit_bits": 0.0,
        "kl_visit_policy_bits": 0.0,
        "js_divergence_bits": 0.0,
    }
    metric_count = 0
    started_at = time.perf_counter()

    file_stem = _run_file_stem(args, cfg)
    positions_csv = output_dir / f"{file_stem}_positions.csv"
    actions_csv = output_dir / f"{file_stem}_actions.csv"
    summary_json = output_dir / f"{file_stem}_summary.json"

    with _CsvRowWriter(positions_csv, POSITION_FIELDNAMES) as position_writer, _CsvRowWriter(
        actions_csv,
        ACTION_FIELDNAMES,
    ) as action_writer:
        for case_index, case in enumerate(cases, start=1):
            metrics = _run_alignment_case(
                args=args,
                cfg=cfg,
                case=case,
                case_index=case_index,
                case_count=len(cases),
                model=model,
                device=device,
                position_writer=position_writer,
                action_writer=action_writer,
                final_policy_ranks=final_policy_ranks,
            )
            top1_matches += int(metrics.top1_match)
            metric_sums["policy_entropy_bits"] += metrics.policy_entropy_bits
            metric_sums["visit_entropy_bits"] += metrics.visit_entropy_bits
            metric_sums["kl_policy_visit_bits"] += metrics.kl_policy_visit_bits
            metric_sums["kl_visit_policy_bits"] += metrics.kl_visit_policy_bits
            metric_sums["js_divergence_bits"] += metrics.js_divergence_bits
            metric_count += 1

    thresholds = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    elapsed_sec = time.perf_counter() - started_at
    summary = {
        "evaluator": args.evaluator,
        "static_eval_variant": args.static_eval_variant,
        "positions": len(cases),
        "simulations": int(cfg.search.num_simulations),
        "top_k_actions": int(cfg.search.top_k_actions),
        "dirichlet_epsilon": float(cfg.search.dirichlet_epsilon),
        "root_temperature": float(cfg.search.root_temperature),
        "elapsed_sec": elapsed_sec,
        "top1_match_rate": top1_matches / float(len(cases)) if cases else 0.0,
        "policy_entropy_bits_avg": metric_sums["policy_entropy_bits"] / float(metric_count) if metric_count else 0.0,
        "visit_entropy_bits_avg": metric_sums["visit_entropy_bits"] / float(metric_count) if metric_count else 0.0,
        "kl_policy_visit_bits_avg": metric_sums["kl_policy_visit_bits"] / float(metric_count) if metric_count else 0.0,
        "kl_visit_policy_bits_avg": metric_sums["kl_visit_policy_bits"] / float(metric_count) if metric_count else 0.0,
        "js_divergence_bits_avg": metric_sums["js_divergence_bits"] / float(metric_count) if metric_count else 0.0,
        "final_move_policy_rank_max": max(final_policy_ranks) if final_policy_ranks else None,
        "final_move_policy_rank_avg": sum(final_policy_ranks) / float(len(final_policy_ranks)) if final_policy_ranks else None,
        "smallest_top_n_capturing_all_search_best": max(final_policy_ranks) if final_policy_ranks else None,
        "capture_by_policy_top_n": _capture_thresholds(final_policy_ranks, thresholds),
        "outputs": {
            "positions_csv": str(positions_csv),
            "actions_csv": str(actions_csv),
            "summary_json": str(summary_json),
        },
        "notes": {
            "regret_proxy": (
                "Without a trusted optimal action or root Q per action, this reports rank/capture/divergence proxies. "
                "The smallest_top_n field is the smallest policy top-N that would have retained every no-top-k search winner."
            )
        },
    }
    _write_json(summary_json, summary)
    print(
        "Policy/search alignment complete: "
        f"variant={args.static_eval_variant} positions={len(cases)} top1_match={summary['top1_match_rate']:.3f} "
        f"capture_all_top_n={summary['smallest_top_n_capturing_all_search_best']} "
        f"elapsed_sec={elapsed_sec:.1f}",
        flush=True,
    )
    print(f"wrote {positions_csv}")
    print(f"wrote {actions_csv}")
    print(f"wrote {summary_json}")


def _run_alignment_case(
    *,
    args: argparse.Namespace,
    cfg: HybridAgentConfig,
    case: PayloadCase,
    case_index: int,
    case_count: int,
    model: HybridPolicyValueNet | None,
    device: torch.device,
    position_writer: _CsvRowWriter,
    action_writer: _CsvRowWriter,
    final_policy_ranks: list[int],
) -> AlignmentCaseMetrics:
    actions = list(case.payload.get("actions", []))[: int(cfg.model.max_actions)]
    action_ids = [_action_id(action) for action in actions]
    print(
        f"[{case_index}/{case_count}] {case.label} actions={len(actions)} sims={cfg.search.num_simulations}",
        flush=True,
    )
    policy = _policy(case.payload, cfg, args.evaluator, model, device)
    search_started = time.perf_counter()
    result = _run_search(case.payload, cfg, args.evaluator, model, device)
    search_sec = time.perf_counter() - search_started

    search_scores = [float(result.visit_distribution.get(action_id, 0.0)) for action_id in action_ids]
    policy_scores = [float(value) for value in policy.priors[: len(action_ids)]]
    policy_ranks = _rank_map(policy_scores)
    search_ranks = _rank_map(search_scores)
    search_best_index = max(range(len(action_ids)), key=lambda idx: search_scores[idx]) if action_ids else -1
    policy_best_index = max(range(len(action_ids)), key=lambda idx: policy_scores[idx]) if action_ids else -1
    top1_match = search_best_index == policy_best_index and search_best_index >= 0
    final_rank = policy_ranks.get(search_best_index, 0)
    if final_rank:
        final_policy_ranks.append(final_rank)

    policy_dist = _distribution(policy_scores)
    search_dist = _distribution(search_scores)
    policy_entropy_bits = _entropy_bits(policy_dist)
    visit_entropy_bits = _entropy_bits(search_dist)
    kl_policy_visit_bits = _kl_bits(policy_dist, search_dist)
    kl_visit_policy_bits = _kl_bits(search_dist, policy_dist)
    js_divergence_bits = _js_bits(policy_dist, search_dist)
    position_writer.writerow(
        {
            "label": case.label,
            "payload_path": str(case.path),
            "actions": len(action_ids),
            "search_sec": f"{search_sec:.3f}",
            "policy_value": f"{policy.value:.6f}",
            "search_root_value": f"{float(result.value):.6f}",
            "policy_top_action_id": action_ids[policy_best_index] if policy_best_index >= 0 else "",
            "policy_top_type": _action_type(actions[policy_best_index]) if policy_best_index >= 0 else "",
            "search_top_action_id": action_ids[search_best_index] if search_best_index >= 0 else "",
            "search_top_type": _action_type(actions[search_best_index]) if search_best_index >= 0 else "",
            "search_top_policy_rank": final_rank,
            "search_top_policy_prior": f"{policy_scores[search_best_index]:.8f}" if search_best_index >= 0 else "",
            "search_top_visit_share": f"{search_scores[search_best_index]:.8f}" if search_best_index >= 0 else "",
            "top1_match": int(top1_match),
            "overlap_at_5": f"{_overlap_at(policy_scores, search_scores, 5):.4f}",
            "overlap_at_10": f"{_overlap_at(policy_scores, search_scores, 10):.4f}",
            "overlap_at_32": f"{_overlap_at(policy_scores, search_scores, 32):.4f}",
            "policy_entropy_bits": f"{policy_entropy_bits:.6f}",
            "visit_entropy_bits": f"{visit_entropy_bits:.6f}",
            "kl_policy_visit_bits": f"{kl_policy_visit_bits:.6f}",
            "kl_visit_policy_bits": f"{kl_visit_policy_bits:.6f}",
            "js_divergence_bits": f"{js_divergence_bits:.6f}",
        }
    )

    for index, action in enumerate(actions):
        action_writer.writerow(
            {
                "label": case.label,
                "action_id": action_ids[index],
                "policy_rank": policy_ranks[index],
                "search_rank": search_ranks[index],
                "policy_prior": f"{policy_scores[index]:.10f}",
                "visit_share": f"{search_scores[index]:.10f}",
                **_action_summary(action),
            }
        )
    return AlignmentCaseMetrics(
        top1_match=top1_match,
        policy_entropy_bits=policy_entropy_bits,
        visit_entropy_bits=visit_entropy_bits,
        kl_policy_visit_bits=kl_policy_visit_bits,
        kl_visit_policy_bits=kl_visit_policy_bits,
        js_divergence_bits=js_divergence_bits,
    )


def _release_variant_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(args: argparse.Namespace) -> None:
    variant = str(args.static_eval_variant)
    if variant != "all":
        _run_alignment(args)
        return

    delay_sec = max(0.0, float(getattr(args, "variant_delay_sec", 5.0)))
    for index, selected_variant in enumerate(STATIC_EVAL_VARIANTS):
        variant_args = argparse.Namespace(**vars(args))
        variant_args.static_eval_variant = selected_variant
        try:
            _run_alignment(variant_args)
        finally:
            _release_variant_memory()
        if index < len(STATIC_EVAL_VARIANTS) - 1 and delay_sec > 0.0:
            print(
                f"waiting {delay_sec:.1f}s before next static eval variant",
                flush=True,
            )
            time.sleep(delay_sec)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare root policy priors against no-top-k MCTS search visits.")
    parser.add_argument("--evaluator", choices=("static", "nn", "hybrid"), default="static")
    parser.add_argument("--payload", action="append", default=[], help="Explicit payload JSON path. May be repeated.")
    parser.add_argument("--payload-dir", type=Path, default=DEFAULT_PAYLOAD_DIR)
    parser.add_argument("--no-payload-dir", action="store_true", help="Only use explicit --payload files.")
    parser.add_argument("--positions", type=int, default=None, help="Optional max number of payloads to analyze.")
    parser.add_argument("--simulations", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k-actions", type=int, default=0)
    parser.add_argument("--max-actions", type=int, default=512)
    parser.add_argument("--static-eval-variant", choices=("baseline", "experimental", "all"), default="baseline")
    parser.add_argument("--variant-delay-sec", type=float, default=5.0, help="Delay between variants when using --static-eval-variant all.")
    parser.add_argument("--root-temperature", type=float, default=1.0)
    parser.add_argument("--dirichlet-epsilon", type=float, default=0.0)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "py" / "profiling" / "mcts_search_profiling_random_init_model.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--static-policy-weight", type=float, default=0.5)
    parser.add_argument("--static-value-weight", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = load_config_defaults(parser, default_config=DEFAULT_POLICY_SEARCH_ALIGNMENT_CONFIG)
    if args.payload is None:
        args.payload = []
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
