from __future__ import annotations

import argparse
import json
from pathlib import Path

from profiling.analysis.payload_store import load_payload, payload_hash
from profiling.analysis.position_analyzer import analyze_position, parse_target, position_to_dict


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--simulations", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k-actions", type=int, default=0)
    parser.add_argument("--max-actions", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--c-puct", type=float, default=1.5)
    args = parser.parse_args()
    payload = load_payload(args.payload)
    if payload is None:
        raise RuntimeError(f"invalid payload: {args.payload}")
    row = analyze_position(
        payload,
        payload_hash=payload_hash(payload),
        label=args.payload.stem,
        target=parse_target(args.target),
        simulations=args.simulations,
        batch_size=args.batch_size,
        top_k_actions=args.top_k_actions,
        max_actions=args.max_actions,
        seed=args.seed,
        c_puct=args.c_puct,
    )
    print(json.dumps(position_to_dict(row), sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

