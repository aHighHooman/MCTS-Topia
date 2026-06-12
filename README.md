# Tribes MCTS

`Tribes_MCTS` is the Python, RL, profiling, and native-search workspace for a Polytopia-like bot stack. It does not own the Java game rules source; Java is compiled from the sibling game checkout and this repo layers bots, neural-network encoding, self-play, profiling, and native MCTS work on top.

The default local layout is:

```text
C:\Users\Umair\OneDrive\Desktop\Work\Self_Projects\Tribes_MCTS
C:\Users\Umair\OneDrive\Desktop\Work\Self_Projects\TribesTopia\Tribes
```

Set `$env:TRIBES_GAME_ROOT` if the Java game checkout lives somewhere else.

## What Is Here

- Python external bots under `py/bots/`
- Neural-network model, observation/action encoding, belief state, and augmentation under `py/nn/`
- Search configuration and native/static/hybrid MCTS under `py/search/`
- RL self-play, replay handling, static bootstrap generation, and training under `py/training/`
- MCTS and MCTS+NN profilers under `py/profiling/`
- Native C++ transition/search code under `py/search/native/`
- Focused Python tests under `py/tests/`
- JSON configs for single games, tournaments, profilers, and bot evaluation
- Docs for the external bot protocol and architecture notes under `docs/`

The Java rules, game runners, tournament runner, and broad regression harness are sourced from:

```text
$TRIBES_GAME_ROOT\src
```

By default, that is:

```text
C:\Users\Umair\OneDrive\Desktop\Work\Self_Projects\TribesTopia\Tribes\src
```

## Requirements

- Windows with PowerShell
- JDK 21 with `JAVA_HOME` set
- Python with the project dependencies installed, including PyTorch for NN/native-extension workflows
- `lib/json.jar` in either `$TRIBES_GAME_ROOT\lib\json.jar` or this repo's `lib/json.jar`

Use explicit Java tools when running commands manually:

```powershell
& "$env:JAVA_HOME\bin\javac.exe" ...
& "$env:JAVA_HOME\bin\java.exe" ...
```

Before Python commands in a fresh shell:

```powershell
$env:PYTHONPATH = "$PWD\py"
```

## Build Java

Compile the external Java game source into this repo's `out/` directory:

```powershell
.\scripts\build_java.ps1
```

The script reads Java sources from `$env:TRIBES_GAME_ROOT` when set, otherwise from the default sibling checkout.

## Run Java Entry Points

Single headless game using `play.json`:

```powershell
& "$env:JAVA_HOME\bin\java.exe" -cp "out;lib/json.jar" HeadlessPlay
```

Single headless game with an explicit config:

```powershell
& "$env:JAVA_HOME\bin\java.exe" -cp "out;lib/json.jar" HeadlessPlay play.json
```

Tournament:

```powershell
& "$env:JAVA_HOME\bin\java.exe" -cp "out;lib/json.jar" Tournament tournament.json
```

Java regression harness:

```powershell
& "$env:JAVA_HOME\bin\java.exe" -cp "out;lib/json.jar" core.game.RegressionHarness
```

## Python Tests

```powershell
$env:PYTHONPATH = "$PWD\py"
python -m pytest py/tests
```

These tests cover Python bots, self-play config, observation/belief helpers, MCTS profiler behavior, replay augmentation, and focused native MCTS parity.

## Native MCTS

Native sources live in `py/search/native/`. The extension is built through PyTorch C++ extension tooling when `load_native_mcts_extension()` is first used; build caches are intentionally ignored under `py/search/native/.build*/`.

Useful native files:

- `py/search/native/native_rules.cpp`: native transition/rules implementation
- `py/search/native/native_mcts.cpp`: native MCTS binding/search code
- `py/search/native/static_mcts.py`: Python static MCTS wrapper
- `py/search/native/hybrid_mcts.py`: hybrid NN-guided search wrapper
- `py/search/native/parity_runner.py`: Java-vs-native parity checker
- `py/search/native/java/core/game/NativeParityOracle.java`: Java oracle used by parity checks

Run a parity check against a saved fixture:

```powershell
$env:PYTHONPATH = "$PWD\py"
python -m search.native.parity_runner --fixture debug-logs\some-fixture\game.json --depth 1
```

Common debug flags:

```powershell
python -m search.native.parity_runner --fixture debug-logs\some-fixture\game.json --action-id 12 --trace-dir debug-logs\native-traces
python -m search.native.parity_runner --fixture debug-logs\some-fixture\game.json --depth 2 --keep-going
```

## Profiling

Profiler entrypoints are config-driven. Edit the JSON files, then run the modules without extra CLI flags.

MCTS search profiler:

```powershell
$env:PYTHONPATH = "$PWD\py"
python -m profiling.mcts_search --config py/profiling/configs/mcts_search.json
```

MCTS+NN self-play profiler:

```powershell
$env:PYTHONPATH = "$PWD\py"
python -m profiling.selfplay_mcts_nn --config py/profiling/configs/selfplay_mcts_nn.json
```

Important config files:

- `py/profiling/configs/mcts_search.json`
- `py/profiling/configs/selfplay_mcts_nn.json`

Profiler output normally goes under `debug-logs/`.

## RL And Training

Training state is local and ignored under `rl/`. That directory can contain checkpoints, replay shards, metrics, TensorBoard logs, and generated plots.

Main entrypoints:

```powershell
$env:PYTHONPATH = "$PWD\py"
python -m training.generate_static_bootstrap_replay --games 10 --augment-symmetries
python -m training.pretrain_static
python -m training.train
python -m training.checkpoint_tournament
```

Core training defaults live in `py/training/config.py`. Model/search defaults live in `py/search/config.py`.

When encoded features change, keep these in sync:

- `py/nn/encoding.py`
- `py/search/config.py`
- `py/nn/model.py`
- tests that assert tensor/action dimensions

## External Bots

The Java game talks to Python bots with a JSON stdin/stdout protocol. See:

```text
docs/external-bot-protocol.md
```

Bot scripts currently include:

- `py/bots/random_bot.py`
- `py/bots/simple_bot.py`
- `py/bots/native_static_mcts_bot.py`
- `py/bots/hybrid_nn_bot.py`

External bot action ids are request-scoped. Do not store an action id and reuse it on a later request.

## Important Configs And Data

- `play.json`: single-game config
- `tournament.json`: default tournament config
- `tournament_static_eval_ab.json`: static evaluator A/B tournament config
- `tournament_static_eval_ab_swapped.json`: swapped-side A/B tournament config
- `tournament_static_mcts_iterations.json`: static MCTS iteration tournament config

## Generated Local State

These directories are intentionally ignored and can be regenerated or treated as local run output:

- `out/`
- `save/`
- `logs/`
- `debug-logs/`
- `tmp/`
- `rl/`
- `.pytest_cache/`
- `py/search/native/.build*/`

## Development Notes

- Java rules are authoritative; Python bots and RL should consume legal actions from Java.
- Observation and forward-model states are player-specific hidden-information copies, not omniscient game state.
- Requested bot/search/training behavior should become the default behavior unless a flag is explicitly needed.
- Native transition changes should be compared against Java behavior with focused parity coverage.
- Prefer adding parity fixtures or targeted `py/tests/test_native_mcts.py` coverage when fixing native rules.

## Credits

This project builds on the `Tribes` research framework:

Diego Perez Liebana, Yu-Jhen Hsu, Stavros Emmanouilidis, Bobby Khaleque, Raluca Gaina, *Tribes: A New Turn-Based Strategy Game for AI*, AIIDE 2020.
