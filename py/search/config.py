from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ModelConfig:
    board_size: int = 16
    board_channels: int = 73
    entity_feature_dim: int = 45
    unit_feature_dim: int = 45
    city_feature_dim: int = 29
    action_feature_dim: int = 267
    scalar_dim: int = 94
    cnn_channels: int = 64
    board_res_blocks: int = 2
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    ff_mult: int = 4
    dropout: float = 0.0
    max_units: int = 256
    max_cities: int = 64
    max_actions: int = 512
    use_cls_token: bool = True
    use_empty_board_coordinate_priors: bool = False

    def __post_init__(self) -> None:
        try:
            from nn.encoding import (
                ACTION_FEATURE_SCHEMA,
                BOARD_SCHEMA,
                CITY_FEATURE_SCHEMA,
                SCALAR_FEATURE_SCHEMA,
                UNIT_FEATURE_SCHEMA,
            )
        except Exception:
            return
        self.board_channels = len(BOARD_SCHEMA)
        self.entity_feature_dim = len(UNIT_FEATURE_SCHEMA)
        self.unit_feature_dim = len(UNIT_FEATURE_SCHEMA)
        self.city_feature_dim = len(CITY_FEATURE_SCHEMA)
        self.action_feature_dim = len(ACTION_FEATURE_SCHEMA)
        self.scalar_dim = len(SCALAR_FEATURE_SCHEMA)


@dataclass
class SearchConfig:
    num_simulations: int = 64
    batch_size: int = 64
    max_depth: int = 0
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.15
    dirichlet_epsilon: float = 0.2
    root_temperature: float = 1.0
    top_k_actions: int = 64
    sample_action: bool = True
    min_non_end_turn_visits: int = 1
    static_policy_weight: float = 0.0
    static_value_weight: float = 0.0
    use_progressive_widening: bool = True
    reuse_tree: bool = False

_MOVED_TO_TRAINING_CONFIG = {
    "DiagnosticsConfig",
    "HybridAgentConfig",
    "ReplayConfig",
    "RewardConfig",
    "SelfPlayConfig",
    "StaticBootstrapConfig",
    "StaticPretrainConfig",
    "TrainingConfig",
}


def __getattr__(name: str) -> Any:
    if name in _MOVED_TO_TRAINING_CONFIG:
        from training import config as training_config

        value = getattr(training_config, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ModelConfig",
    "SearchConfig",
    *_MOVED_TO_TRAINING_CONFIG,
]
