# Phase 3 PR3 Checkpoint

Date: 2026-04-17

## Scope landed

This checkpoint adds the next tech-tree foundation slice:

- `STRATEGY` added behind `SHIELDS`
- `DIPLOMACY` added behind `STRATEGY`
- generic tree-completion logic replaces hardcoded leaf checks
- regression coverage for research availability and the new leaf

## Files changed

- `src/core/Types.java`
- `src/core/TechnologyTree.java`
- `src/core/game/RegressionHarness.java`

## What changed

### New tech nodes

The regular tech tree now includes:

- `STRATEGY` with parent `SHIELDS`
- `DIPLOMACY` with parent `STRATEGY`

Because research action generation already iterates `Types.TECHNOLOGY.values()`, these nodes automatically appear in legal research actions once their prerequisites are met.

### Generic completion logic

`TechnologyTree.doResearch(...)` no longer hardcodes the previous leaf set. It now checks whether the researched tech has children and only re-evaluates `everythingResearched` when a leaf is completed.

That means new leaves such as `DIPLOMACY` are tracked correctly without more hardcoded maintenance.

### Assumption recorded

For this incremental slice, `DIPLOMACY` is modeled with the current tier-3 cost band while remaining a child of `STRATEGY`, matching the requested "Diplomacy after Strategy" branch without yet widening the rest of the tech-cost model.

If later gameplay validation shows `STRATEGY`/`DIPLOMACY` should use a different cost band, that can be adjusted independently now that serialization is stable.

## Regression coverage added

`RegressionHarness` now verifies:

- `STRATEGY` requires `SHIELDS`
- `DIPLOMACY` requires `STRATEGY`
- `DIPLOMACY` shows up in legal research actions only after `STRATEGY`
- researching the new `DIPLOMACY` leaf can mark the tree complete

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
[PASS] city level-up behavior
[PASS] game state copy clones RNG state
[PASS] tribe result tie break is deterministic
[PASS] combat retaliation
[PASS] city capture
[PASS] save/load round-trip
Regression harness passed 11 of 11 checks.
```

## Next recommended step

The next safe PR is to replace the scalar diplomacy matrix with explicit relationship state and persistence, while still leaving treaties/embassies/cloaks for follow-up slices.
