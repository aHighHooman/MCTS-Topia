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

Head-to-head static bot tournaments:

- Use the Java `Tournament` runner for bot-vs-bot checks, not the profiler or `HeadlessPlay`, because it reports winners, scores, failed matches, and supports `Parallel Games`.
- Generate a temporary tournament config under `debug-logs/` with `Balance Seats: true`, `Parallel Games: 8`, `Match Retry Limit: 0`, and a high per-participant `External Action Timeout Ms` such as `120000`.
- For wall-clock comparisons, pass `--wall-clock-per-action-seconds 1` to each `out/native/static_mcts_bot.exe` command instead of fixed `--simulations`; keep `--search-batch-size`, `--top-k-actions`, `--max-actions`, and `--deterministic` explicit.
- For quick failure scans, use a small deterministic seed range and a short `Turn Limit` such as `8`; larger turn caps can take many minutes even with `Parallel Games: 8` because the game can request many primitive actions per turn.
- After a run, inspect the tournament summary and external bot stderr logs:
  - `Get-Content debug-logs\...\summary.log -Tail 120`
  - `Get-ChildItem debug-logs\...\external_logs -Recurse -Filter *.stderr.log | Where-Object Length -gt 0`

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
- Shared native C++ files live in `py/search/native/src/`: `rules.cpp`, `static_eval.cpp`, `mcts.cpp`.
- The standalone full static-eval protocol bot lives in `bots/static_mcts_bot.cpp`; build it with `scripts/build_static_bot.ps1`.
- `scripts/build_java.ps1` recreates `out/`; if a workflow needs both Java classes and `out/native/static_mcts_bot.exe`, run `build_java.ps1` first and `build_static_bot.ps1` second.
- `scripts/build_static_bot.ps1` default `Release` intentionally avoids MSVC `/GL`/`/LTCG`; use `-Configuration ReleaseLtcg` only when explicitly investigating link-time optimizer behavior, because that profile has reproduced `turn-macro-exp` native crashes.
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
- When changing native transitions or static evaluation, compare against Java behavior and prefer adding focused parity coverage in `py/tests/test_mcts.py` or the parity runner.
- Tests are not required for every change. In particular, do not update or add tests solely because an experimental/static-eval tuning number changed; tests should cover functionality, invariants, parity, crashes, and likely regressions, not lock every exploratory coefficient.
