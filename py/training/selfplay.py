from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .config import HybridAgentConfig
from .java_selfplay import run_selfplay_match


def run_selfplay(
    cfg: HybridAgentConfig,
    bot_commands: Sequence[Sequence[str]],
    tribes: Sequence[str],
    workdir: Path,
    *,
    progress_label: str | None = None,
) -> object:
    return run_selfplay_match(cfg, bot_commands, tribes, workdir, progress_label=progress_label)
