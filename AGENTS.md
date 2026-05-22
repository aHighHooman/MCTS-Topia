# AGENTS.md
## Project Shape

- Mixed Java/Python project for a Polytopia-like engine, bot evaluation, RL self-play, and native MCTS.
- Java engine/rules live under `src/`; Python code is organized by purpose under `py/bots`, `py/nn`, `py/search`, `py/training`, and `py/profiling`.
- Work from the repo root: `C:\Users\Umair\OneDrive\Desktop\Work\Self_Projects\Tribes_MCTS`.

## Setup Gotchas

- JDK 21 is expected. Prefer explicit Java tools:
  - `& "$env:JAVA_HOME\bin\javac.exe" ...`
  - `& "$env:JAVA_HOME\bin\java.exe" ...`
- Python imports require:
  - `$env:PYTHONPATH = "$PWD\py"`
- Java depends on `lib/json.jar`; include it in compile and run classpaths.
- Generated/local directories are intentionally ignored: `out/`, `save/`, `logs/`, `debug-logs/`, `tmp/`, `rl/`, `.pytest_cache/`, `py/search/native/.build*/`.

## Build And Test Commands

Compile Java:

```powershell
if (Test-Path out) { Remove-Item -LiteralPath out -Recurse -Force }
New-Item -ItemType Directory -Force out | Out-Null
& "$env:JAVA_HOME\bin\javac.exe" -cp "lib/json.jar" -d out (Get-ChildItem -Recurse src -Filter *.java | ForEach-Object { $_.FullName })
```

Run Java regression harness:

```powershell
$sources = Join-Path (Get-Location) 'sources.txt'
Get-ChildItem -Recurse src -Filter *.java | ForEach-Object { $_.FullName } | Set-Content -LiteralPath $sources
if (Test-Path out) { Remove-Item -LiteralPath out -Recurse -Force }
New-Item -ItemType Directory -Force out | Out-Null
& "$env:JAVA_HOME\bin\javac.exe" -cp "lib/json.jar" -d out @$sources
& "$env:JAVA_HOME\bin\java.exe" -cp "out;lib/json.jar" core.game.RegressionHarness
Remove-Item -LiteralPath $sources -Force
```

Run Python tests:

```powershell
$env:PYTHONPATH = "$PWD\py"
python -m pytest py/tests
```

Run a headless game:

```powershell
& "$env:JAVA_HOME\bin\java.exe" -cp "out;lib/json.jar" HeadlessPlay
```

Run tournament:

```powershell
& "$env:JAVA_HOME\bin\java.exe" -cp "out;lib/json.jar" Tournament tournament.json
```

Run MCTS+NN profilers:

```powershell
$env:PYTHONPATH = "$PWD\py"
python -m profiling.mcts_search --config py/profiling/configs/mcts_search.json
python -m profiling.selfplay_mcts_nn --config py/profiling/configs/selfplay_mcts_nn.json
```

## Native MCTS Notes

- Native extension sources are in `py/search/native/`.
- Build cache is `py/search/native/.build*` and is ignored.
- `load_native_mcts_extension()` auto-builds with PyTorch C++ extension tooling.
- Useful parity/debug entrypoint: `python -m search.native.parity_runner` after setting `PYTHONPATH`.

## Important Files

- `docs/external-bot-protocol.md`: JSON stdin/stdout protocol and forward-model command loop.
- `src/core/game/RegressionHarness.java`: broad Java rules regression suite.
- `src/HeadlessPlay.java`: headless Java game runner used by self-play.
- `src/Tournament.java`: tournament runner.
- `py/profiling/mcts_search.py`: primary MCTS search profiler implementation.
- `py/profiling/selfplay_mcts_nn.py`: primary self-play MCTS+NN profiler implementation.
- `py/profiling/configs/`: profiler config JSON files.
- `py/search/config.py`: default RL/self-play/search settings.
- `py/nn/encoding.py`: canonical action/unit/tech lists and observation tensor encoding.
- `py/search/native/cpp_extension.py`: native extension loading/build behavior.

## Coding Guidance

- Keep Java rules authoritative; Python bots and RL should consume legal actions from Java.
- External bot action ids are request-scoped. Do not persist them across requests.
- Observation and forward-model states are player-specific hidden-information copies, not omniscient game state.
- Do not preserve backward compatibility when updating bots, tree searches, or tests.
- Requested functionality should become the default behavior; only add explicit flags or toggles when asked.
- When changing encoded features, keep `ModelConfig` dimensions, `encoding.py`, model input assumptions, and related tests in sync.
- When changing native transitions, compare against Java behavior and prefer adding focused parity coverage in `py/tests/test_native_mcts.py` or the parity runner.
