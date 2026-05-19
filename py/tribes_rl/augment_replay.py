from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Iterable

import torch

from .augmentation import transform_message_symmetry
from .replay import ReplayStore, StepRecord, record_from_payload, record_to_payload


Symmetry = tuple[int, bool]


def symmetry_specs(name: str, *, include_identity: bool = False) -> list[Symmetry]:
    if name == "mirror":
        specs = [(0, False), (0, True)]
    elif name == "rotations":
        specs = [(0, False), (1, False), (2, False), (3, False)]
    elif name == "d4":
        specs = [(rotation, mirror) for mirror in (False, True) for rotation in range(4)]
    else:
        raise ValueError(f"Unknown symmetry set: {name}")
    if include_identity:
        return specs
    return [spec for spec in specs if spec != (0, False)]


def augment_record(record: StepRecord, rotation: int, mirror: bool) -> StepRecord:
    message = {
        "player_id": record.player_id,
        "observation": record.observation,
        "actions": record.legal_actions,
    }
    transformed = transform_message_symmetry(message, rotation=rotation, mirror=mirror)
    payload = record_to_payload(record)
    payload["observation"] = transformed["observation"]
    payload["legal_actions"] = transformed["actions"]
    return record_from_payload(payload)


def augment_episode_records(
    records: Iterable[StepRecord],
    specs: Iterable[Symmetry],
) -> list[StepRecord]:
    out: list[StepRecord] = []
    for rotation, mirror in specs:
        out.extend(augment_record(record, rotation, mirror) for record in records)
    return out


def materialize_augmented_replay(
    input_dir: Path,
    output_dir: Path,
    *,
    input_prefix: str = "replay",
    output_prefix: str = "replay",
    symmetry_set: str = "d4",
    include_identity: bool = False,
    max_records_per_shard: int = 10_000,
) -> dict[str, Any]:
    specs = symmetry_specs(symmetry_set, include_identity=include_identity)
    input_store = ReplayStore(input_dir, capacity_steps=0, shard_prefix=input_prefix)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_records = max(1, int(max_records_per_shard))
    shard_index = 0
    pending: list[StepRecord] = []
    written_paths: list[Path] = []
    input_records = 0
    augmented_records = 0

    def flush() -> None:
        nonlocal shard_index, pending
        if not pending:
            return
        path = output_dir / f"{output_prefix}_{shard_index:04d}_{int(time.time() * 1000)}.pt"
        shard_index += 1
        payload = {
            "version": 2,
            "kind": "augmented_replay",
            "created_at": time.time(),
            "iteration": 0,
            "player_id": None,
            "step_count": len(pending),
            "source_shards": [str(shard) for shard in input_store.shards()],
            "symmetry_set": symmetry_set,
            "include_identity": include_identity,
            "augmentation": {
                "source_dir": str(Path(input_dir)),
                "symmetry_set": symmetry_set,
                "include_identity": include_identity,
                "symmetries": [{"rotation": rotation, "mirror": mirror} for rotation, mirror in specs],
            },
            "records": [record_to_payload(record) for record in pending],
        }
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
        written_paths.append(path)
        pending = []

    for shard in input_store.shards():
        _, records = input_store._load_shard(shard)
        input_records += len(records)
        for augmented in augment_episode_records(
            records,
            specs,
        ):
            pending.append(augmented)
            augmented_records += 1
            if len(pending) >= max_records:
                flush()
    flush()
    return {
        "input_shards": len(input_store.shards()),
        "input_records": input_records,
        "output_shards": len(written_paths),
        "augmented_records": augmented_records,
        "symmetry_count": len(specs),
        "output_dir": str(output_dir),
        "output_prefix": output_prefix,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize symmetry-augmented Tribes RL replay shards.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-prefix", default="replay")
    parser.add_argument("--output-prefix", default="replay")
    parser.add_argument("--symmetry-set", choices=("mirror", "rotations", "d4"), default="d4")
    parser.add_argument("--include-identity", action="store_true")
    parser.add_argument("--max-records-per-shard", type=int, default=10_000)
    args = parser.parse_args()

    summary = materialize_augmented_replay(
        args.input_dir,
        args.output_dir,
        input_prefix=args.input_prefix,
        output_prefix=args.output_prefix,
        symmetry_set=args.symmetry_set,
        include_identity=args.include_identity,
        max_records_per_shard=args.max_records_per_shard,
    )
    print(
        "augmented replay "
        f"input_shards={summary['input_shards']} input_records={summary['input_records']} "
        f"symmetries={summary['symmetry_count']} output_shards={summary['output_shards']} "
        f"augmented_records={summary['augmented_records']} output_dir={summary['output_dir']} "
        f"output_prefix={summary['output_prefix']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
