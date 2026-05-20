from __future__ import annotations

from typing import Iterable

import torch

from nn.augmentation import transform_message_symmetry
from nn.encoding import EncodedObservation, encode_observation
from nn.model import HybridPolicyValueNet
from search.config import HybridAgentConfig
from .replay import StepRecord


def _d4_symmetry_specs() -> list[tuple[int, bool]]:
    return [(rotation, mirror) for mirror in (False, True) for rotation in range(4)]


def _stack_encoded(items: list[EncodedObservation]) -> EncodedObservation:
    return EncodedObservation(
        board=torch.cat([item.board for item in items], dim=0),
        unit_features=torch.cat([item.unit_features for item in items], dim=0),
        unit_mask=torch.cat([item.unit_mask for item in items], dim=0),
        city_features=torch.cat([item.city_features for item in items], dim=0),
        city_mask=torch.cat([item.city_mask for item in items], dim=0),
        action_features=torch.cat([item.action_features for item in items], dim=0),
        action_mask=torch.cat([item.action_mask for item in items], dim=0),
        scalar_features=torch.cat([item.scalar_features for item in items], dim=0),
        action_ids=[],
    )


@torch.no_grad()
def evaluate_symmetry_consistency(
    cfg: HybridAgentConfig,
    model: HybridPolicyValueNet,
    records: Iterable[StepRecord],
    device: torch.device | str,
    *,
    max_records: int = 32,
) -> dict[str, float]:
    """Measure prediction disagreement across D4-equivalent views.

    Legal action order is preserved by augmentation, so logits at the same
    action index are comparable across transformed observations.
    """
    model.eval()
    groups = 0
    samples = 0
    policy_js_total = 0.0
    policy_l1_total = 0.0
    value_std_total = 0.0
    value_range_total = 0.0
    eps = 1e-8

    for record in records:
        if groups >= max_records:
            break
        action_count = min(len(record.legal_actions), cfg.model.max_actions)
        if action_count <= 0:
            continue
        message = {"player_id": record.player_id, "observation": record.observation, "actions": record.legal_actions}
        encoded_items = [
            encode_observation(
                transform_message_symmetry(message, rotation=rotation, mirror=mirror),
                cfg.model,
            )
            for rotation, mirror in _d4_symmetry_specs()
        ]
        encoded = _stack_encoded(encoded_items).to(device)
        output = model(encoded)
        probs = torch.softmax(output.policy_logits[:, :action_count], dim=-1)
        mean_probs = probs.mean(dim=0)
        policy_js = (
            probs
            * (probs.clamp_min(eps).log() - mean_probs.unsqueeze(0).clamp_min(eps).log())
        ).sum(dim=-1).mean()
        policy_l1 = (probs - mean_probs.unsqueeze(0)).abs().sum(dim=-1).mean()
        values = output.value.detach().float()

        policy_js_total += max(0.0, float(policy_js.item()))
        policy_l1_total += float(policy_l1.item())
        value_std_total += float(values.std(unbiased=False).item())
        value_range_total += float((values.max() - values.min()).item())
        groups += 1
        samples += len(encoded_items)

    if groups <= 0:
        return {
            "symmetry_groups": 0.0,
            "symmetry_samples": 0.0,
            "policy_js": 0.0,
            "policy_l1": 0.0,
            "value_std": 0.0,
            "value_range": 0.0,
        }
    return {
        "symmetry_groups": float(groups),
        "symmetry_samples": float(samples),
        "policy_js": policy_js_total / groups,
        "policy_l1": policy_l1_total / groups,
        "value_std": value_std_total / groups,
        "value_range": value_range_total / groups,
    }
