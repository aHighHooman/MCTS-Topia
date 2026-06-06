# Phase 3 PR1 Checkpoint

Date: 2026-04-17

## Scope landed

This checkpoint covers the Phase 2 "PR 1" foundation slice:

- deterministic `GameState.copy()` RNG cloning
- deterministic `TribeResult` tie-breaking
- headless regression runner for baseline engine invariants
- Windows-safe save/load file handling fix in `GameLoader`

## Files changed

- `src/core/game/GameState.java`
- `src/core/game/TribeResult.java`
- `src/core/game/GameLoader.java`
- `src/core/game/RegressionHarness.java`

## What changed

### Deterministic forward-model copies

`GameState.copy(int)` now deep-copies the internal `Random` state instead of creating a fresh RNG. This keeps copied rollouts reproducible and aligned with the authoritative state.

### Deterministic end-of-game ordering

`TribeResult.compareTo()` now uses tribe id as the final tie-break instead of a random coin flip. This removes non-deterministic ranking order.

### Headless regression coverage

`RegressionHarness` is a package-local runner in `core.game` that currently checks:

- tech cost scaling
- city level-up behavior
- copied RNG state
- deterministic tribe-result ties
- combat with retaliation
- city capture
- save/load round-trip

### Save/load stability

`GameLoader.readFile()` now closes its reader with try-with-resources. This fixed a Windows file-handle leak that broke repeated save/load regression runs.

## Verification

Compile:

```powershell
& "$env:JAVA_HOME\bin\javac.exe" -cp 'lib/json.jar' -d out (Get-ChildItem -Recurse -Path 'src' -Filter '*.java' | ForEach-Object { $_.FullName })
```

Run:

```powershell
& "$env:JAVA_HOME\bin\java.exe" '-Djava.awt.headless=true' '-cp' 'out;lib/json.jar' 'core.game.RegressionHarness'
```

Latest result:

```text
[PASS] tech cost scaling
[PASS] city level-up behavior
[PASS] game state copy clones RNG state
[PASS] tribe result tie break is deterministic
[PASS] combat retaliation
[PASS] city capture
[PASS] save/load round-trip
Regression harness passed 7 of 7 checks.
```

## Next recommended step

Start the next PR on tech-tree serialization hardening before adding `STRATEGY` and `DIPLOMACY`. The current tree still serializes by enum ordinal, which is the next major blocker for safe feature expansion.
