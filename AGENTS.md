# AGENTS.md
## Project Shape

- Workspace RL self-play, NN training, profiling, and native/static/hybrid MCTS.
- `C:\Users\Umair\OneDrive\Desktop\Work\Self_Projects\TribesTopia\Tribes` is the authoritative external game repo.
- Python code lives under `py/bots`, `py/nn`, `py/search`, `py/training`, and `py/profiling`.
- Run commands from the repo root: `C:\Users\Umair\OneDrive\Desktop\Work\Self_Projects\Tribes_MCTS`.

## Setup Gotchas

- JDK 21 is expected; prefer explicit Java tools:
  - `& "$env:JAVA_HOME\bin\javac.exe" ...`
  - `& "$env:JAVA_HOME\bin\java.exe" ...`
- Python imports require `$env:PYTHONPATH = "$PWD\py"`.
- Java depends on `lib/json.jar`; prefer `$TRIBES_GAME_ROOT\lib\json.jar` when present, otherwise this repo's `lib/json.jar`.
- Generated/local dirs are ignored: `out/`, `save/`, `logs/`, `debug-logs/`, `tmp/`, `rl/`, `.pytest_cache/`, and native build/debug caches.

## Build And Test Commands
Compile Java from the external game checkout into `out/`:

```powershell
.\scripts\build_java.ps1
```

Run core checks:

```powershell
$env:PYTHONPATH = "$PWD\py"
python -m pytest py/tests
```

- The full `python -m pytest py/tests` suite can take around 5 minutes; wait for it to finish instead of assuming it is hung.

Common Java entrypoints:

```powershell
& "$env:JAVA_HOME\bin\java.exe" -cp "out;lib/json.jar" HeadlessPlay play.json
& "$env:JAVA_HOME\bin\java.exe" -cp "out;lib/json.jar" Tournament tournament.json
```

Common Python entrypoints:

```powershell
$env:PYTHONPATH = "$PWD\py"
python -m profiling.mcts_search --config py/profiling/configs/mcts_search.json
python -m profiling.selfplay_mcts_nn --config py/profiling/configs/selfplay_mcts_nn.json
python -m training.train
python -m training.checkpoint_tournament
```

## Native MCTS Notes

- Native extension sources are in `py/search/native/`; `load_native_mcts_extension()` auto-builds with PyTorch C++ extension tooling.
- Shared native C++ files live in `py/search/native/src/`: `native_rules.cpp`, `native_static_eval.cpp`, `native_mcts.cpp`.
- The standalone full static-eval protocol bot lives in `bots/native_static_mcts_bot.cpp`; build it with `scripts/build_native_static_bot.ps1`.
- Parity/debug entrypoint: `python -m search.native.parity_runner --fixture debug-logs\some-fixture\game.json --depth 1`.
- Java parity oracle: `py/search/native/java/core/game/NativeParityOracle.java`.

## Important Files

- `docs/external-bot-protocol.md`: JSON stdin/stdout protocol and forward-model command loop.
- `play.json`: default single-game config.
- `tournament.json`: default tournament config.
- `py/profiling/`: MCTS profiler implementations and configs.
- `py/training/config.py`: training, replay, diagnostics, and Java self-play defaults.
- `py/search/config.py`: search/model defaults used by bots and training.
- `py/nn/encoding.py`: canonical action/unit/tech lists and observation tensor encoding.
- `py/nn/model.py`, `py/nn/belief.py`, `py/nn/augmentation.py`: model, belief, and replay augmentation.
- `py/search/native/cpp_extension.py`: native extension loader/build behavior.
- `py/bots/`: protocol-facing external bots.

## Coding Guidance

- Keep Java rules authoritative; Python bots and RL should consume legal actions from Java.
- External bot action ids are request-scoped. Do not persist them across requests.
- Observation and forward-model states are player-specific hidden-information copies, not omniscient game state.
- Do not preserve backward compatibility when updating bots, tree searches, training code, or tests unless explicitly requested.
- Requested functionality should become the default behavior; only add explicit flags or toggles when asked.
- When changing native transitions or static evaluation, compare against Java behavior and prefer adding focused parity coverage in `py/tests/test_native_mcts.py` or the parity runner.
- Tests are not required for every change. In particular, do not update or add tests solely because an experimental/static-eval tuning number changed; tests should cover functionality, invariants, parity, crashes, and likely regressions, not lock every exploratory coefficient.
