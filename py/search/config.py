from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ModelConfig:
    board_size: int = 16
    board_channels: int = 73
    entity_feature_dim: int = 45
    unit_feature_dim: int = 45
    city_feature_dim: int = 29
    action_feature_dim: int = 267
    scalar_dim: int = 94
    cnn_channels: int = 96
    d_model: int = 160
    n_heads: int = 8
    n_layers: int = 5
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
class RewardConfig:
    terminal_win: float = 1.0
    terminal_loss: float = -1.0
    terminal_draw: float = 0.0
    use_java_normalized_terminal_reward: bool = True
    shaped_reward_weight: float = 0.0
    gamma: float = 0.997


@dataclass
class ReplayConfig:
    replay_dir: Path = Path("rl/replay")
    capacity_steps: int = 100_000
    shard_prefix: str = "replay"
    min_train_steps: int = 1
    fresh_fraction: float = 0.75


@dataclass
class DiagnosticsConfig:
    metrics_csv: Path = Path("rl/metrics.csv")
    selfplay_games_csv: Path = Path("rl/selfplay_games.csv")


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


@dataclass
class SelfPlayConfig:
    java_executable: Optional[str] = None
    java_classpath: str = "out;lib/json.jar"
    java_main_class: str = "HeadlessPlay"
    run_mode: str = "PlayLG"
    game_mode: str = "Capitals"
    level_file: str = None
    level_seed: int = -1
    map_type: str = "Drylands"
    map_size: str = "Tiny"
    game_seed: int = -1
    agent_seed: int = -1
    max_turns_capitals: int = 40
    max_actions_per_turn: int = 80
    max_actions_per_game: int = 1024
    adjudicate_incomplete_games: bool = False
    adjudicator_bot: str = "py/bots/simple_bot.py"
    adjudication_max_turns_capitals: int = 40
    adjudication_max_actions_per_game: int = 2048
    timeout_seconds: int = 1800
    progress_interval_seconds: int = 0
    profile_selfplay: bool = False
    persistent_bot: bool = True
    external_startup_timeout_ms: int = 10_000
    external_action_timeout_ms: int = 120_000
    external_shutdown_timeout_ms: int = 30_000
    wall_clock_per_action_seconds: Optional[float] = None
    search_depth: int = 0
    force_end: bool = False
    rollouts: bool = False
    population_size: int = 1
    pruning: bool = True
    progressive_bias: bool = True
    k_init_mult: float = 0.5
    t_mult: float = 2.0
    a_mult: float = 1.5
    b_mult: float = 1.3


@dataclass
class TrainingConfig:
    output_dir: Path = Path("rl")
    checkpoint_dir: Path = Path("rl/checkpoints")
    checkpoint_name: str = "latest.pt"
    batch_size: int = 256
    replay_batch_size: int = 512
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 10.0
    epochs_per_iteration: int = 3
    selfplay_games_per_iteration: int = 20
    num_iterations: int = 10
    value_loss_weight: float = 0.5
    policy_loss_weight: float = 1.0
    augment_symmetries: bool = True
    augmentation_seed: Optional[int] = None
    device: str = "cuda"
    static_guidance_bootstrap_iterations: int = 8
    static_guidance_early_iterations: int = 8
    static_guidance_middle_iterations: int = 10
    static_guidance_late_iterations: int = 10
    static_guidance_final_iterations: int = 12

    @property
    def checkpoint_path(self) -> Path:
        return self.checkpoint_dir / self.checkpoint_name

    @checkpoint_path.setter
    def checkpoint_path(self, value: Path | str) -> None:
        path = Path(value)
        self.checkpoint_dir = path.parent
        self.checkpoint_name = path.name

    def iteration_checkpoint_path(self, iteration: int) -> Path:
        stem = Path(self.checkpoint_name).stem
        suffix = Path(self.checkpoint_name).suffix or ".pt"
        return self.checkpoint_dir / f"{stem}_iter_{iteration:04d}{suffix}"


@dataclass
class HybridAgentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
