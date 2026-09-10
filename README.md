# Tribes MCTS

Monte Carlo tree search bots and reinforcement-learning tools for Tribes, a turn-based strategy game inspired by Polytopia.

This repository contains a native C++ MCTS bot with a hand-written evaluator, a neural-network bot, and Python tools for self-play, training, profiling, and testing. The Java game engine is maintained separately and is required for running games and checking native rules against Java.

## Repository layout

| Path | Contents |
| --- | --- |
| `bots/` | Bot entrypoints, including the standalone C++ bot |
| `py/bots/` | Python bot implementations and test utilities |
| `py/search/` | Search configuration, native MCTS, and C++ game transitions |
| `py/nn/` | Neural-network model, observation encoding, and belief state |
| `py/training/` | Self-play, replay data, training, and checkpoint evaluation |
| `py/profiling/` | Search profilers and analysis tools |
| `py/tests/` | Python and native regression tests |
| `docs/` | Protocol documentation and search, evaluation, and profiling guides |

## Setup

The build scripts use Windows and PowerShell. Run the commands below from this repository's root.

You will need:

- Python with the dependencies required by your workflow, including PyTorch for neural-network and native-extension workflows. This repository does not currently provide a dependency installation manifest.
- Visual Studio C++ Build Tools with the MSVC x64 toolchain for native builds.
- JDK 21 with `JAVA_HOME` set for Java builds and game execution.
- A compatible Tribes Java game checkout containing `src/` and `lib/json.jar`. The Java engine, game runners, and game/tournament configs are not included here.

Set the game checkout path explicitly and make the Python modules available in each new shell:

```powershell
$env:TRIBES_GAME_ROOT = "C:\path\to\Tribes"
$env:PYTHONPATH = "$PWD\py"
```

The game checkout must support the JSON external bot protocol used by these bots. The original Tribes framework alone may not include the integration expected by these tools.

## Build the bots

Compile the Java game classes into `out/`:

```powershell
.\scripts\build_java.ps1
```

Build the standalone MCTS bot:

```powershell
.\scripts\build_static_bot.ps1
```

The executable is written to `out/native/static_mcts_bot.exe`. Build Java first when you need both, since the Java build recreates `out/`.

The neural-network bot entrypoint is `bots/hybrid_nn_bot.py`. Its native Python extension builds through PyTorch's C++ extension tooling on first use.

Both bots communicate with the Java game over JSON on stdin/stdout. Game and tournament configuration belongs in the game checkout. Bot action IDs are request-scoped and must not be reused across requests.

## Run tests

```powershell
python -m pytest py/tests
```

Tests cover bots, encoding, self-play, replay handling, profiling, and native search. Tests that exercise Java or native code require the corresponding build tools and game checkout.

For a Java/native transition comparison, replace the fixture path with a saved game fixture:

```powershell
python -m search.native.parity_runner --fixture debug-logs\some-fixture\game.json --depth 1
```

Java rules are authoritative. Native transition changes should be checked against Java behavior. Observations represent each player's available information, including hidden-information limits.

## Profile search

Edit the profiler JSON configs for your input paths and search settings, then run:

```powershell
python -m profiling.mcts_search --config py/profiling/configs/mcts_search.json
python -m profiling.selfplay_mcts_nn --config py/profiling/configs/selfplay_mcts_nn.json
```

Profiler output normally goes under `debug-logs/`. See the [analysis tooling guide](docs/analysis-tooling.md) for position analysis, branch comparisons, and evaluator tuning.

## Train and evaluate

```powershell
python -m training.train
python -m training.checkpoint_tournament
```

Training defaults live in `py/training/config.py`; model and search settings live in `py/search/config.py`. Check these settings and paths before starting a run. Checkpoints, replay data, and training metrics are stored locally under `rl/`.

Build outputs, logs, training data, and native build caches are generated locally and are not included in the repository.

## Documentation

- [Documentation index](docs/index.md)
- [MCTS search](docs/mcts-search.md)
- [Static evaluation](docs/static-eval.md)
- [Tournament evaluation](docs/tournament-evaluation.md)

## Credits

This project builds on the Tribes research framework:

Diego Perez Liebana, Yu-Jhen Hsu, Stavros Emmanouilidis, Bobby Khaleque, Raluca Gaina, *Tribes: A New Turn-Based Strategy Game for AI*, AIIDE 2020.
