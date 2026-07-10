# AGENTS.md

## Repo Map

- Run commands from this repo root.
- The sibling game checkout (`$env:TRIBES_GAME_ROOT`, default `C:\Users\Umair\OneDrive\Desktop\Work\Self_Projects\TribesTopia\Tribes`) owns Java rules, runners, and game/tournament JSON configs.
- `bots/`: supported bot entrypoints; `py/bots/`: Python implementations and test utilities.
- `py/search/native/`: native MCTS wrapper, C++ rules/search/static eval, and Java parity tooling.
- `py/nn/`: model, encoding, belief, and augmentation; `py/training/`: self-play, replay, and training.
- `py/profiling/`: search profilers, analysis tools, and JSON configs; `py/tests/`: Python/native tests.
- `docs/index.md`: subsystem docs; `docs/analysis-tooling.md`: profiling, comparison, and tuning runbook.

## Setup

- Windows PowerShell, JDK 21, and Python with project dependencies are expected.
- Set `$env:PYTHONPATH = "$PWD\py"` before Python module commands.
- Prefer `& "$env:JAVA_HOME\bin\javac.exe"` and `& "$env:JAVA_HOME\bin\java.exe"` for Java commands.
- `scripts/build_java.ps1` reads `$env:TRIBES_GAME_ROOT` or the default sibling checkout and uses its `lib/json.jar`.
- Generated/local state includes `out/`, `save/`, `logs/`, `debug-logs/`, `tmp/`, `rl/`, `.pytest_cache/`, and native build caches.

## Core Commands

```powershell
# Java classes (recreates out/)
.\scripts\build_java.ps1

# Standalone native static bot -> out/native/static_mcts_bot.exe
.\scripts\build_static_bot.ps1

# Core tests; the full suite can take about five minutes
$env:PYTHONPATH = "$PWD\py"
python -m pytest py/tests

# Main profilers
python -m profiling.mcts_search --config py/profiling/configs/mcts_search.json
python -m profiling.selfplay_mcts_nn --config py/profiling/configs/selfplay_mcts_nn.json

# RL training and checkpoint evaluation
python -m training.train
python -m training.checkpoint_tournament

# Java/native transition parity on a saved fixture
python -m search.native.parity_runner --fixture debug-logs\some-fixture\game.json --depth 1
```

- If Java classes and the static bot are both needed, run `build_java.ps1` first because it recreates `out/`.
- The Python native extension auto-builds through PyTorch tooling when `load_native_mcts_extension()` is first used.
- Use the sibling checkout's `HeadlessPlay`, `Tournament`, and JSON configs for Java games and bot-vs-bot evaluation.
- Use `docs/analysis-tooling.md` for branch comparison, position analysis, static-value tuning, and tournament workflows.

## Change Guidance

- Java rules are authoritative; bots and RL consume legal actions from Java.
- External-bot action IDs are request-scoped. Never reuse them across requests.
- Observations and forward-model states are player-specific hidden-information copies, not omniscient state.
- Shared native C++ logic is under `py/search/native/src/`; the standalone protocol bot is `bots/static_mcts_bot.cpp`.
- After changing native transitions, compare against Java and add focused parity/regression coverage where useful.
- Requested bot, search, and training behavior becomes the default unless a flag is explicitly requested.
- Do not preserve backward compatibility unless requested.
- Test functionality, invariants, parity, crashes, and likely regressions; do not lock experimental tuning coefficients merely because they changed.
