# Phase 3 PR4 Checkpoint

Date: 2026-04-17

## Scope landed

This checkpoint replaces the diplomacy foundation from allegiance-only state with an explicit, serializable relationship model while keeping current gameplay compatibility:

- explicit `WAR / PEACE / TREATY` relationship state
- diplomacy metadata for last relation-change turn and pending treaty offers
- diplomacy save/load persistence
- declare-war flow wired to explicit relation state
- regression coverage for diplomacy state and persistence

## Files changed

- `src/core/Types.java`
- `src/core/Diplomacy.java`
- `src/core/game/Board.java`
- `src/core/game/GameSaver.java`
- `src/core/actions/tribeactions/command/DeclareWarCommand.java`
- `src/core/actions/tribeactions/command/SendStarsCommand.java`
- `src/core/actions/tribeactions/factory/DeclareWarFactory.java`
- `src/core/game/RegressionHarness.java`

## What changed

### Explicit relationship model

`Types.RELATIONSHIP` now defines:

- `WAR`
- `PEACE`
- `TREATY`

`Diplomacy` now stores:

- legacy allegiance matrix
- explicit relationship matrix
- last relation-change turn matrix
- pending treaty-offer matrix

This preserves current heuristics that still read allegiance values while giving the engine a first-class relation state for future treaty logic.

### Compatibility rule for this slice

This PR keeps current gameplay stable with one explicit rule:

- allegiance changes may escalate `PEACE -> WAR`
- allegiance changes do **not** implicitly end `WAR`
- `TREATY` is reserved for future actions and persistence, but not yet produced by gameplay

That means war now behaves as a durable state until a later diplomacy action explicitly changes it.

### Persistence

`GameSaver` now writes diplomacy into the board JSON, and `Board` loads it back through a `Diplomacy(JSONObject, size)` path. Older saves still fall back to a blank diplomacy object.

### Action flow updates

- `DeclareWarCommand` now stamps explicit `WAR` plus the relation-change turn
- `DeclareWarFactory` now checks the explicit relation state instead of the old allegiance threshold
- `SendStarsCommand` still updates allegiance for compatibility, but no longer collapses explicit war state

## Regression coverage added

`RegressionHarness` now verifies:

- diplomacy defaults to `PEACE`
- declare war updates explicit relation state
- explicit war survives later allegiance changes
- diplomacy state survives save/load

Total regression coverage now passes 14 checks.

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
[PASS] technology tree stable-id serialization
[PASS] technology tree legacy-array loading
[PASS] strategy and diplomacy research path
[PASS] technology tree completion tracks diplomacy leaf
[PASS] diplomacy relations default to peace
[PASS] declare war updates explicit relationship state
[PASS] diplomacy state survives save load
[PASS] city level-up behavior
[PASS] game state copy clones RNG state
[PASS] tribe result tie break is deterministic
[PASS] combat retaliation
[PASS] city capture
[PASS] save/load round-trip
Regression harness passed 14 of 14 checks.
```

## Next recommended step

The next safe PR is to add explicit diplomacy actions and legality for:

- propose peace / propose treaty
- accept / cancel treaty
- attack restriction when relation is `TREATY`

That will let the new relationship model start driving gameplay rules instead of only persistence and declare-war state.
