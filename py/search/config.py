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
    score_delta_scale: float = 0.0
    gamma: float = 0.997
    gae_lambda: float = 1.0


@dataclass
class ReplayConfig:
    replay_dir: Path = Path("rl/replay")
    capacity_steps: int = 100_000
    shard_prefix: str = "replay"
    min_train_steps: int = 1
    fresh_fraction: float = 0.75
    decisive_weight: float = 2.0
    policy_error_weight: float = 0.0


@dataclass
class DiagnosticsConfig:
    enabled: bool = True
    log_dir: Path = Path("rl/runs")
    metrics_csv: Path = Path("rl/metrics.csv")
    selfplay_games_csv: Path = Path("rl/selfplay_games.csv")
    log_every_steps: int = 1
    use_tensorboard: bool = False
    use_wandb: bool = False
    wandb_project: str = "tribes-rl"


@dataclass
class EvaluationConfig:
    games_per_iteration: int = 0
    baseline_bot: str = "py/bots/strong_external_bot.py"
    deterministic: bool = True


@dataclass
class SearchConfig:
    num_simulations: int = 64
    num_simulations_min: int = 32
    num_simulations_max: int = 512
    high_simulation_prob: float = 0.1
    opening_simulations: int = 64
    batch_size: int = 64
    max_depth: int = 0
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.15
    dirichlet_epsilon: float = 0.2
    root_temperature: float = 1.0
    sample_moves: int = 32
    top_k_actions: int = 64
    root_parallel_actions: int = 32
    value_discount: float = 0.997
    sample_action: bool = True
    allow_approximate_opponent_turns: bool = True
    invalid_branch_penalty: float = -0.25
    opponent_uncertainty_penalty: float = -0.05
    min_non_end_turn_visits: int = 1


@dataclass
class SelfPlayConfig:
    java_executable: Optional[str] = None
    java_classpath: str = "out;lib/json.jar"
    java_main_class: str = "HeadlessPlay"
    run_mode: str = "PlayLG"
    game_mode: str = "Capitals"
    level_file: str = "levels/MinimalLevel2.csv"
    level_seed: int = -1
    map_type: str = "Lakes"
    map_size: str = "Small"
    game_seed: int = -1
    agent_seed: int = -1
    max_turns_capitals: int = 60
    max_actions_per_turn: int = 80
    max_actions_per_game: int = 512
    adjudicate_incomplete_games: bool = False
    adjudicator_bot: str = "py/bots/simple_bot.py"
    adjudication_max_turns_capitals: int = 40
    adjudication_max_actions_per_game: int = 2048
    timeout_seconds: int = 1800
    progress_interval_seconds: int = 0
    workers: int = 1
    allow_unsafe_workers: bool = False
    allow_partial_selfplay: bool = False
    profile_selfplay: bool = False
    persistent_bot: bool = True
    external_startup_timeout_ms: int = 10_000
    external_action_timeout_ms: int = 120_000
    external_shutdown_timeout_ms: int = 30_000
    wall_clock_per_action_seconds: Optional[float] = None
    search_depth: int = 20
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
    batch_size: int = 128
    replay_batch_size: int = 512
    learning_rate: float = 2.5e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    epochs_per_iteration: int = 3
    selfplay_games_per_iteration: int = 20
    num_iterations: int = 4
    value_loss_weight: float = 0.5
    policy_loss_weight: float = 1.0
    train_both_sides: bool = True
    vary_mcts_simulations: bool = True
    augment_symmetries: bool = True
    augmentation_prob: float = 1.0
    augmentation_seed: Optional[int] = None
    device: str = "cuda"

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
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

