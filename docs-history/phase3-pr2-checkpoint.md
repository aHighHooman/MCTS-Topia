# Phase 3 PR2 Checkpoint

Date: 2026-04-17

## Scope landed

This checkpoint covers the next Phase 3 foundation slice after PR1:

- stable technology serialization using explicit tech ids
- backward-compatible loading of legacy ordinal-array tech saves
- regression coverage for both formats

## Files changed

- `src/core/Types.java`
- `src/core/TechnologyTree.java`
- `src/core/game/GameSaver.java`
- `src/core/game/RegressionHarness.java`

## What changed

### Stable technology ids

`Types.TECHNOLOGY` now carries an explicit stable string id such as `climbing`, `roads`, and `philosophy`. The enum also exposes `getId()` and `getTypeById(...)`.

This removes the upcoming risk where adding `STRATEGY` or `DIPLOMACY` would shift enum ordinals and corrupt saved tech state.

### Backward-compatible tech loading

`TechnologyTree(JSONObject)` now supports two formats:

- new format: `researchedTechIds`
- legacy format: `researched`

The constructor allocates storage against the current enum count and safely copies only the overlapping part of legacy arrays. That means older saves still load even after the tech enum grows.

### Save path writes both formats

`GameSaver` now writes:

- `researchedTechIds`
- `researched`

Writing both keeps the new format authoritative while preserving short-term compatibility for any code still expecting the legacy boolean array.

### Regression coverage

`RegressionHarness` now checks:

- tech cost scaling
- technology tree stable-id serialization
- technology tree legacy-array loading
- city level-up behavior
- copied RNG state
- deterministic tribe-result ties
- combat retaliation
- city capture
- save/load round-trip

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
[PASS] city level-up behavior
[PASS] game state copy clones RNG state
[PASS] tribe result tie break is deterministic
[PASS] combat retaliation
[PASS] city capture
[PASS] save/load round-trip
Regression harness passed 9 of 9 checks.
```

## Next recommended step

The next safe PR is to add `STRATEGY` and `DIPLOMACY` to the tech tree and unlock table without gameplay behavior yet, then extend regression coverage around research availability and tech-cost progression. After that, diplomacy state can be upgraded from scalar allegiance to explicit treaty/war relations.
