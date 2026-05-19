# TribesTopia for Bots

`TribesTopia-for-bots` is a local Java clone of *The Battle of Polytopia* built for bot development, self-play, and tournament evaluation.

The project started from the open-source `Tribes` codebase and has been heavily modified toward current Polytopia parity for regular-tribe play. The main use case is not shipping a game to end users; it is running fast, reproducible games for AI experiments.

## What Is In This Repo

- A Java game engine for Polytopia-style turn-based play
- GUI and headless single-game runners
- A headless tournament runner with deterministic seeding, retries, and Elo tracking
- External bot process support for Python/RL agents
- Regression tests focused on current rules parity

## Current Scope

The repo is closest to current Polytopia for regular tribes and standard shared mechanics.

Important current limitations:

- special tribes are still not exact/current
- map generation is closer to live Polytopia than stock Tribes, but not proven exact in every detail
- this repo intentionally does not implement every official game mode

## Repository Layout

- `src/`: Java source
- `play.json`: config for a single game
- `tournament.json`: config for tournaments
- `levels/`: CSV map files
- `lib/json.jar`: JSON dependency
- `terrainProbs.json`: generated-map terrain/resource probabilities
- `src/core/game/RegressionHarness.java`: regression suite

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
`py/`, Java headless self-play builds into `out/`, and local training state lives
under `rl/`. The `rl/` directory is intentionally ignored by Git because it holds
large local checkpoints, replay shards, metrics, and run logs.

Before running Python commands from a fresh PowerShell session:

```powershell
$env:PYTHONPATH = "$PWD\py"
```

Main RL entrypoints:

```powershell
python -m tribes_rl.train --checkpoint rl/checkpoints/latest.pt
python py/tribes_dashboard.py
python -m pytest py/tests
```

## Build

Run from the repo root:

```powershell
if (Test-Path out) { Remove-Item -LiteralPath out -Recurse -Force }
New-Item -ItemType Directory -Force out | Out-Null
& "$env:JAVA_HOME\bin\javac.exe" -cp "lib/json.jar" -d out (Get-ChildItem -Recurse src -Filter *.java | ForEach-Object { $_.FullName })
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

## Live RL Dashboard

The local dashboard gives you a browser UI for starting and watching RL training,
analytics benchmarks, checkpoint tournaments, headless Java games, and an optional
local TensorBoard process without staring at the terminal.

```powershell
python py/tribes_dashboard.py
```

Then open:

```text
http://127.0.0.1:8765
```

The dashboard reads the existing generated artifacts:

- `rl/metrics.csv`
- `rl/replay/`
- `rl/checkpoints/`
- `rl/analytics_benchmarks/`
- `rl/tournaments/`

Jobs launched from the dashboard write combined stdout/stderr logs under
`rl/dashboard/logs/`.

## RL Augmentation Benchmark

Rotation+mirroring replay augmentation can be compared against standard training
with:

```powershell
python py/benchmark_augmentation.py --device auto --max-records 512 --repeats 3 --update-rounds 3 --training-batch-size 64 --replay-batch-size 256 --epochs-per-round 1
```

By default this samples existing self-play replay from `rl/replay`.
Results are written to `rl/augmentation_benchmarks/` as JSONL plus a
summary CSV. Pass `--checkpoint path\to\model.pt --require-checkpoint` when you
want the comparison to start from an existing trained model. The D4 variant uses
materialized 8-way replay, so it actually increases the number of augmented
training examples, and it is enabled by default. Add `--no-d4-expanded` to run
only the standard baseline. Add `--generate-selfplay-games N` to generate a fresh
shared self-play replay set before running the standard-vs-D4 A/B training
comparison.

## `play.json`

The main fields are:

- `Run Mode`: `PlayLG`, `PlayFile`, or `Replay`
- `Game Mode`: `Perfection`, `Domination`, `Glory`, or `Might`
- `Map Type`: `Drylands`, `Lakes`, `Continents`, `Pangea`, `Archipelago`, `Water World`
- `Map Size`: `Tiny`, `Small`, `Normal`, `Large`, `Huge`, `Massive`
- `Players`: player types for a single game
- `Tribes`: tribe assignment for those players

Player names currently supported by `src/Run.java`:

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
      "External Command": ["python", "py/bots/strong_external_bot_v2.py"]
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
$sources = Join-Path (Get-Location) 'sources.txt'
Get-ChildItem -Recurse src -Filter *.java | % FullName | Set-Content -LiteralPath $sources
if (Test-Path out) { Remove-Item -LiteralPath out -Recurse -Force }
mkdir out | Out-Null
& "$env:JAVA_HOME\bin\javac.exe" -cp "lib/json.jar" -d out @$sources
& "$env:JAVA_HOME\bin\java.exe" -cp "out;lib/json.jar" core.game.RegressionHarness
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
