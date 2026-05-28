from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from search.config import ModelConfig, SearchConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RL_ROOT = PROJECT_ROOT / "rl"


def rl_path(*parts: str) -> Path:
    return RL_ROOT.joinpath(*parts)


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
    replay_dir: Path = rl_path("replay")
    capacity_steps: int = 100_000
    shard_prefix: str = "replay"
    min_train_steps: int = 1
    fresh_fraction: float = 0.75


@dataclass
class DiagnosticsConfig:
    metrics_csv: Path = rl_path("metrics.csv")
    selfplay_games_csv: Path = rl_path("selfplay_games.csv")


@dataclass
class SelfPlayConfig:
    java_executable: Optional[str] = None
    java_classpath: str = "out;lib/json.jar"
    java_main_class: str = "HeadlessPlay"
    run_mode: str = "PlayLG"
    game_mode: str = "Capitals"
    level_file: str | None = None
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
    output_dir: Path = RL_ROOT
    checkpoint_dir: Path = rl_path("checkpoints")
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
class StaticPretrainConfig:
    replay_dir: Path = rl_path("replay_static_bootstrap")
    shard_prefix: str = "replay"
    checkpoint_dir: Path = rl_path("checkpoints")
    checkpoint_name: str = "static_pretrained.pt"
    metrics_csv: Path = rl_path("static_pretrain_metrics.csv")
    plot_path: Path = rl_path("static_pretrain_loss.png")
    policy_plot_path: Path = rl_path("static_pretrain_policy_loss.png")
    value_plot_path: Path = rl_path("static_pretrain_value_loss.png")
    tensorboard_log_dir: Path | None = rl_path("tensorboard", "static_pretrain")
    tensorboard_auto_start: bool = True
    tensorboard_clear_on_start: bool = True
    tensorboard_restart_on_start: bool = True
    tensorboard_port: int = 6006
    max_records_in_memory: int = 20_000
    train_fraction: float = 0.9
    seed: int = 7
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 10.0
    max_epochs: int = 50
    patience: int = 6
    min_delta: float = 1e-4
    plot_update_interval_batches: int = 1
    console_log_interval_batches: int = 10
    value_loss_weight: float = 0.5
    policy_loss_weight: float = 1.0
    value_target_source: str = "outcome"
    device: str = "cuda"

    @property
    def checkpoint_path(self) -> Path:
        return self.checkpoint_dir / self.checkpoint_name

    @checkpoint_path.setter
    def checkpoint_path(self, value: Path | str) -> None:
        path = Path(value)
        self.checkpoint_dir = path.parent
        self.checkpoint_name = path.name

    def best_checkpoint_path(self) -> Path:
        stem = Path(self.checkpoint_name).stem
        suffix = Path(self.checkpoint_name).suffix or ".pt"
        return self.checkpoint_dir / f"{stem}_best{suffix}"


@dataclass
class StaticBootstrapConfig:
    # Bootstrap-only knobs; shared replay/search/self-play settings stay in their existing configs.
    games: int = 1
    checkpoint_path: Path = rl_path("checkpoints", "static_bootstrap_dummy.pt")
    static_bot_script: Path = PROJECT_ROOT / "py" / "bots" / "hybrid_nn_bot.py"
    tribes: tuple[str, ...] = ("Xin Xi", "Imperius")
    start_seed: int = 10_000_000
    deterministic: bool = False
    static_eval_variant: str | None = None
    delete_raw: bool = False
    archive_raw: bool = False
    augmented_iteration: int = 0
    max_records_per_augmented_shard: int = 10_000
    metrics_csv: Path | None = None


@dataclass
class HybridAgentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    static_pretrain: StaticPretrainConfig = field(default_factory=StaticPretrainConfig)
    static_bootstrap: StaticBootstrapConfig = field(default_factory=StaticBootstrapConfig)
