# TribesTopia for Bots

`TribesTopia-for-bots` is a local Java clone of *The Battle of Polytopia* built for bot development, self-play, and tournament evaluation.

The project started from the open-source `Tribes` codebase and has been heavily modified toward current Polytopia parity for regular-tribe play. The main use case is not shipping a game to end users; it is running fast, reproducible games for AI experiments.

## What Is In This Repo

- A Java game engine for Polytopia-style turn-based play
- GUI and headless single-game runners
- A headless tournament runner with deterministic seeding, retries, and Elo tracking
- External bot process support for Python/RL agents
- MCTS+NN profiling entrypoints under `py/profiling/`
- Regression tests focused on current rules parity

## Current Scope

The repo is closest to current Polytopia for regular tribes and standard shared mechanics.

Important current limitations:

- special tribes are still not exact/current
- map generation is closer to live Polytopia than stock Tribes, but not proven exact in every detail
- this repo intentionally does not implement every official game mode

## Repository Layout

- Java source comes from `C:\Users\Umair\OneDrive\Desktop\Work\Self_Projects\TribesTopia\Tribes\src` by default
- `py/bots/`: launchable external bots
- `py/nn/`: model, observation encoding, belief state, and NN bot agent code
- `py/search/`: MCTS config plus native/static search implementation
- `py/training/`: self-play, replay, checkpoint tournaments, and training loop
- `py/profiling/`: primary MCTS+NN profilers and profiler configs
- `play.json`: config for a single game
- `tournament.json`: config for tournaments
- `levels/`: CSV map files
- `lib/json.jar`: JSON dependency
- `terrainProbs.json`: generated-map terrain/resource probabilities
- external `src/core/game/RegressionHarness.java`: regression suite

Generated directories are ignored by Git:

- `out/`
- `save/`
- `logs/`
- `tmp/`

## Requirements

- Windows with PowerShell
- JDK 21 installed
- `JAVA_HOME` set to the JDK 21 install, for example `C:\Users\Umair\AppData\Local\Programs\Java\jdk-21`

## Standalone RL Repo

This copy is intended to be worked from directly at:

```text
C:\Users\Umair\OneDrive\Desktop\Work\Self_Projects\Tribes_MCTS
```

The RL workflow keeps the original root-relative layout: Python code lives under
`py/`, Java headless self-play builds into this repo's `out/`, and local training
state lives under `rl/`. The Java game source is compiled from:

```text
C:\Users\Umair\OneDrive\Desktop\Work\Self_Projects\TribesTopia\Tribes
```

Set `$env:TRIBES_GAME_ROOT` to point at a different game checkout. The `rl/`
directory is intentionally ignored by Git because it holds large local
checkpoints, replay shards, metrics, and run logs.

Before running Python commands from a fresh PowerShell session:

```powershell
$env:PYTHONPATH = "$PWD\py"
```

Main RL entrypoints:

```powershell
python -m training.train --checkpoint rl/checkpoints/latest.pt
python -m profiling.mcts_search --config py/profiling/configs/mcts_search.json
python -m profiling.selfplay_mcts_nn --config py/profiling/configs/selfplay_mcts_nn.json
python -m pytest py/tests
```

## Build

Run from the repo root:

```powershell
.\scripts\build_java.ps1
```

## Run A Single Game

### GUI

Uses `play.json`:

```powershell
& "$env:JAVA_HOME\bin\java.exe" -cp "out;lib/json.jar" Play
```

### Headless

Also uses `play.json`:

```powershell
& "$env:JAVA_HOME\bin\java.exe" -cp "out;lib/json.jar" HeadlessPlay
```

You can pass a custom config file to either runner:

```powershell
& "$env:JAVA_HOME\bin\java.exe" -cp "out;lib/json.jar" HeadlessPlay my_play_config.json
```

## Run A Tournament

Tournaments are headless only.

Default config:

```powershell
& "$env:JAVA_HOME\bin\java.exe" -cp "out;lib/json.jar" Tournament
```

Custom config:

```powershell
& "$env:JAVA_HOME\bin\java.exe" -cp "out;lib/json.jar" Tournament my_tournament.json
```

## MCTS+NN Profiling

The main performance tools live under `py/profiling/`:

```powershell
python -m profiling.mcts_search --config py/profiling/configs/mcts_search.json --evaluator static
python -m profiling.mcts_search --config py/profiling/configs/mcts_search.json --evaluator nn
python -m profiling.selfplay_mcts_nn --config py/profiling/configs/selfplay_mcts_nn.json
```

Config keys use the same names as CLI flags, with dashes written as underscores.
CLI arguments override config values.

## `play.json`

The main fields are:

- `Run Mode`: `PlayLG`, `PlayFile`, or `Replay`
- `Game Mode`: `Perfection`, `Domination`, `Glory`, or `Might`
- `Map Type`: `Drylands`, `Lakes`, `Continents`, `Pangea`, `Archipelago`, `Water World`
- `Map Size`: `Tiny`, `Small`, `Normal`, `Large`, `Huge`, `Massive`
- `Players`: player types for a single game
- `Tribes`: tribe assignment for those players

Player names currently supported by external `src/Run.java`:

- `Human`
- `External`

## `tournament.json`

The tournament config uses one participant object per bot. Rotation across tribe slots is always enabled.

Example shape:

```json
{
  "Game Mode": "Might",
  "Map Type": "Continents",
  "Map Size": "Tiny",
  "Repetitions": 2,
  "Match Retry Limit": 2,
  "Elo K Factor": 24.0,
  "External Log Dir": "logs/tournament",
  "Verbose": false,
  "Level Seeds": ["93810", "24592"],
  "Participants": [
    {
      "Type": "External",
      "Tribe": "Xin Xi",
      "External Command": ["python", "py/bots/native_static_mcts_bot.py"]
    },
    {
      "Type": "External",
      "Tribe": "Imperius",
      "External Command": ["python", "py/bots/simple_bot.py"]
    }
  ]
}
```

External bot example:

```json
{
  "Type": "External",
  "Tribe": "Xin Xi",
  "External Command": ["python", "py/bots/my_bot.py"]
}
```

Tournament behavior:

- deterministic game and agent seeds per scheduled matchup
- bounded retries for failed matches
- optional external-bot logging under `logs/tournament`
- summary output with win rate, Elo, score, techs, cities, and production

## Regression Tests

Compile and run:

```powershell
$gameRoot = if ($env:TRIBES_GAME_ROOT) { $env:TRIBES_GAME_ROOT } else { 'C:\Users\Umair\OneDrive\Desktop\Work\Self_Projects\TribesTopia\Tribes' }
$sources = Join-Path (Get-Location) 'sources.txt'
Get-ChildItem -Recurse (Join-Path $gameRoot 'src') -Filter *.java | % FullName | Set-Content -LiteralPath $sources
if (Test-Path out) { Remove-Item -LiteralPath out -Recurse -Force }
mkdir out | Out-Null
$jsonJar = Join-Path $gameRoot 'lib\json.jar'
if (-not (Test-Path $jsonJar)) { $jsonJar = 'lib/json.jar' }
& "$env:JAVA_HOME\bin\javac.exe" -cp $jsonJar -d out @$sources
& "$env:JAVA_HOME\bin\java.exe" -cp "out;$jsonJar" core.game.RegressionHarness
Remove-Item -LiteralPath $sources -Force
```

At the time of writing, the harness passes:

- `67/67`

## Notes On Parity

This repo is aiming at the newest Polytopia rules for regular tribes.

Recent parity changes include:

- current regular tech tree
- updated tribe starts and starting stars
- current roads, bridges, ports, markets, and lighthouse behavior
- current ruin rewards
- removal of obsolete custom star-sending
- removal of the non-Polytopia standalone `Declare War` action

Still worth treating as active parity work:

- exact map generation details
- special tribes
- any remaining hidden-information or tournament-quality bot tuning

## Credits

This project builds on the `Tribes` research framework:

- Diego Perez Liebana, Yu-Jhen Hsu, Stavros Emmanouilidis, Bobby Khaleque, Raluca Gaina, *Tribes: A New Turn-Based Strategy Game for AI*, AIIDE 2020
