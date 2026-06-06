# Phase 2 Implementation Plan: PR-Sized Rollout

## Purpose

This document turns the Phase 1 design into an implementation backlog with:

- small PR-sized tasks
- explicit dependencies
- test gates
- migration notes
- recommended rollout order

This plan assumes we continue to optimize for:

1. rules correctness
2. forward-model compatibility
3. bot usability
4. code clarity
5. UI last

## Inputs

- Phase 1 design: [phase1-polytopia-design.md](/C:/Users/Umair/OneDrive/Desktop/Work/Self_Projects/Poly/Tribes/docs/phase1-polytopia-design.md)

## Sequencing Principles

- Do not start with Cloak/Dagger.
- Land determinism and test infrastructure before rule-heavy features.
- Land serialization hardening before changing the tech tree.
- Land explicit diplomacy state before treaty-dependent legality.
- Land visibility refactor before hidden-unit mechanics.
- Keep GUI work behind gameplay and test coverage.

## Delivery Strategy

The safest delivery shape is:

- `Foundation PRs`
  - test harness
  - deterministic simulation
  - serialization hardening
- `Rules Infrastructure PRs`
  - tech metadata
  - diplomacy state machine
  - visibility model
- `Feature PRs`
  - treaties
  - allied roads
  - embassy
  - capital vision
  - cloak
  - infiltration
  - dagger
- `Cleanup PRs`
  - agent compatibility
  - docs
  - lightweight UI support

If we decide to optimize for fewer, slightly larger PRs instead of the safest smallest steps, the first viable merge point is:

- merge PR 1 and PR 2 into a single foundation PR

That compressed option is reasonable because both changes are prerequisite infrastructure, but the default recommendation remains to land them separately to keep regression debugging simpler.

## PR Backlog

### PR 1. Headless Test Harness and Deterministic Baseline

#### Goals

- Add a repeatable headless test harness.
- Make forward-model copies deterministic.
- Add baseline regression tests for already-existing mechanics.

#### Scope

- Introduce a simple test setup for running game-state and command-level tests.
- Fix `GameState.copy(int)` so copied states preserve deterministic RNG behavior.
- Remove or replace random tie-breaking in ranking code.
- Add tests for:
  - tech-cost scaling
  - city level-up behavior
  - combat + retaliation
  - city capture
  - save/load baseline round-trip

#### Files likely touched

- `src/core/game/GameState.java`
- `src/core/game/TribeResult.java`
- new test sources and test runner wiring

#### Dependencies

- none

#### Why first

Every later PR will rely on exact forward-model behavior and easy regression checks.

#### Exit criteria

- Tests run headlessly.
- Copying a state does not change future stochastic outcomes.
- Existing non-diplomacy gameplay still passes regression tests.

### PR 2. Serialization Hardening for Techs and Rule Versions

#### Goals

- Make tech serialization stable across tree changes.
- Introduce save-versioning for future diplomacy/visibility migrations.

#### Scope

- Add stable string codes for technologies.
- Update save/load to persist researched techs by code rather than ordinal-position boolean arrays.
- Add save schema version field.
- Add legacy loader support for existing saves.
- Refactor `TechnologyTree.isEverythingResearched()` logic to derive completion from actual leaves rather than a hard-coded list.

#### Files likely touched

- `src/core/Types.java`
- `src/core/TechnologyTree.java`
- `src/core/game/GameSaver.java`
- `src/core/game/GameLoader.java`

#### Dependencies

- PR 1

#### Why second

`STRATEGY` and `DIPLOMACY` should not be added until enum-order fragility is removed.

#### Exit criteria

- Old saves still load.
- New saves persist researched techs with stable IDs.
- Adding new techs no longer corrupts serialized meaning.

### PR 3. Rule Metadata Layer for Tech/Unit/Building Unlocks

#### Goals

- Reduce scattered enum-condition logic before adding new content.
- Create a cleaner home for branch/category/unlock metadata.

#### Scope

- Add metadata helpers for technology branch/category.
- Add unlock queries for:
  - trainable units
  - buildings
  - diplomacy capabilities
  - vision capabilities
- Update gameplay stats / heuristics to stop depending on fragile ordinal assumptions where practical.

#### Files likely touched

- `src/core/Types.java`
- new `src/core/TechnologyRules.java` or equivalent
- `src/utils/stats/GameplayStats.java`
- selected research heuristics / portfolio code

#### Dependencies

- PR 2

#### Why now

This creates a stable content layer before new techs and diplomacy abilities arrive.

#### Exit criteria

- Tech queries are metadata-driven.
- Existing tech-based behavior still works.
- Stats code no longer relies on enum ordinal structure for branch semantics.

### PR 4. Introduce Strategy and Diplomacy Tech Placeholders

#### Goals

- Add `STRATEGY` and `DIPLOMACY` into the content model without changing gameplay yet.
- Migrate `SHIELDS` behavior cleanly.

#### Scope

- Introduce `STRATEGY`.
- Migrate Defender unlocks from `SHIELDS` to `STRATEGY`.
- Treat old `SHIELDS` saves as `STRATEGY`.
- Add `DIPLOMACY` as a researchable placeholder after `STRATEGY`.
- Ensure tech costs, scoring, and serialization all work.

#### Files likely touched

- `src/core/Types.java`
- `src/core/TechnologyTree.java`
- `src/core/actions/tribeactions/ResearchTech.java`
- `src/core/actions/tribeactions/command/ResearchTechCommand.java`
- serialization files from PR 2

#### Dependencies

- PR 2
- PR 3

#### Why now

The tech tree should exist before diplomacy actions or embassy/cloak units are introduced.

#### Exit criteria

- `STRATEGY` and `DIPLOMACY` can be researched.
- Defender unlock remains correct.
- No diplomacy mechanics are active yet beyond tech presence.

### PR 5. Explicit Diplomacy State Model and Persistence

#### Goals

- Replace diplomacy-as-legality with explicit relation state.
- Persist diplomacy state through save/load.

#### Scope

- Refactor `Diplomacy` to track explicit pair state and metadata.
- Keep optional allegiance score only as heuristic metadata if still useful.
- Add storage for:
  - relation state
  - pending treaty offers
  - treaty/cancel timing metadata
- Persist diplomacy state in save/load.
- Preserve a temporary compatibility accessor for old heuristic code.

#### Files likely touched

- `src/core/Diplomacy.java`
- `src/core/game/Board.java`
- `src/core/game/GameSaver.java`
- `src/core/game/GameLoader.java`

#### Dependencies

- PR 1
- PR 2

#### Why now

Every treaty, embassy, and road-sharing rule depends on explicit relation state.

#### Exit criteria

- Diplomacy no longer relies on scalar allegiance thresholds for core legality.
- Saves round-trip diplomacy state.
- Existing runtime still functions without treaty actions enabled.

### PR 6. Treaty Actions and Hostility Legality

#### Goals

- Add the diplomacy action flow.
- Make hostile actions illegal under treaty state.

#### Scope

- Add tribe actions for:
  - offer peace
  - accept peace
  - cancel treaty
- Add legality helpers for hostile actions.
- Update action generation and `isFeasible()` checks for:
  - attack
  - capture
  - convert
- Add treaty cancellation cooldown / attack lock.

#### Files likely touched

- `src/core/Types.java`
- `src/core/actions/tribeactions/*`
- `src/core/actions/unitactions/Attack.java`
- `src/core/actions/unitactions/Capture.java`
- `src/core/actions/unitactions/Convert.java`
- associated command/factory files

#### Dependencies

- PR 5
- PR 4

#### Why now

This is the first visible diplomacy rules PR and should land before allied movement or embassy.

#### Exit criteria

- Peace can be offered and accepted.
- Allied tribes cannot attack/capture/convert each other.
- Illegal hostile actions are absent from legal-action generation.

### PR 7. Allied Territory and Road Usage

#### Goals

- Implement road sharing and movement access required by treaties.

#### Scope

- Centralize relation-aware movement helpers such as:
  - `canEnterTerritory`
  - `canUseRoad`
  - `canUseTradeLink`
- Update `StepMove` and `TradeNetwork` to treat allied roads and city-road spaces as shared movement/trade infrastructure.
- Keep road building itself conservative unless we later decide treaties should also affect hosted construction.

#### Files likely touched

- `src/core/actions/unitactions/StepMove.java`
- `src/core/game/TradeNetwork.java`
- `src/core/game/Board.java`

#### Dependencies

- PR 6

#### Why now

Treaty movement is a core diplomacy rule and much simpler than embassy/cloak.

#### Exit criteria

- Allied roads are usable for movement.
- Trade network logic does not break existing city connection behavior.
- Non-allied movement remains unchanged.

### PR 8. Exact Resume Save/Load for Turn-Sensitive Diplomacy

#### Goals

- Close remaining replay-vs-resume gaps before sabotage and embassy timing are introduced.

#### Scope

- Persist turn-sensitive tribe fields such as:
  - `starsSent`
  - `hasDeclaredWar`
  - `nWarsDeclared`
  - `nStarsSent`
- Stop advancing active tribe implicitly on load for exact resume mode.
- Add tests covering save/load mid-turn or turn-transition sensitive behavior.

#### Files likely touched

- `src/core/actors/Tribe.java`
- `src/core/game/GameState.java`
- `src/core/game/GameSaver.java`
- `src/core/game/GameLoader.java`

#### Dependencies

- PR 5

#### Why now

Embassy income, betrayal locks, and sabotage windows are all sensitive to exact turn resume semantics.

#### Exit criteria

- Save/load resumes the same authoritative state, not a replay-shifted approximation.

### PR 9. Visibility Refactor: Explored vs VisibleNow

#### Goals

- Replace the current sticky fog model with a true current-visibility model.
- Prepare the observation layer for embassy vision and cloaks.

#### Scope

- Split `obsGrid` into:
  - `exploredGrid`
  - `visibleGrid`
- Recompute `visibleGrid` from current sources.
- Preserve explored terrain memory.
- Update board copy logic to:
  - show `FOG` for unexplored tiles
  - show remembered terrain/buildings/resources for explored but not currently visible tiles
  - omit enemy mutable state when not currently visible
- Recompute or filter observed action lists against `visibleGrid`.

#### Files likely touched

- `src/core/actors/Tribe.java`
- `src/core/game/Board.java`
- `src/core/game/GameState.java`
- selected action factories and helpers

#### Dependencies

- PR 1
- PR 5

#### Why now

Embassy and Cloak both depend on a correct observation model.

#### Exit criteria

- Full-observation mode can still be enabled for debugging.
- Partial-observation mode reflects current vision, not “ever seen”.
- Hidden targets do not leak through legal actions.

### PR 10. Embassy Entity and Hosted Diplomacy Structures

#### Goals

- Add embassy as a diplomacy-side system without breaking normal city-building rules.

#### Scope

- Introduce `Embassy` as a separate entity or actor.
- Add embassy placement action and legality checks.
- Add embassy persistence.
- Add embassy removal on war/capital loss.
- Expose embassy data through bot APIs.

#### Files likely touched

- new `src/core/actors/Embassy.java` or equivalent
- `src/core/Diplomacy.java`
- `src/core/game/Board.java`
- embassy tribe actions and commands
- save/load files

#### Dependencies

- PR 5
- PR 8
- PR 9

#### Why now

Embassy is the first feature that truly requires both diplomacy and the visibility refactor.

#### Exit criteria

- Embassies can be built only in legal host capitals.
- Embassies persist and are removed correctly.

### PR 11. Embassy Vision and Capital Vision

#### Goals

- Implement the two major information features unlocked by Diplomacy.

#### Scope

- Add embassy local reveal as a continuous vision source.
- Add capital-location knowledge channel for met tribes once Diplomacy is researched.
- Expose known capital info cleanly through observed game states.

#### Files likely touched

- `src/core/actors/Tribe.java`
- `src/core/game/Board.java`
- `src/core/game/GameState.java`
- diplomacy/embassy helpers

#### Dependencies

- PR 10
- PR 9
- PR 4

#### Why now

These complete the non-stealth side of the Diplomacy branch before the hidden-information subsystem lands.

#### Exit criteria

- Embassy vision reveals the intended local area.
- Capital vision is queryable without forcing full tile visibility.

### PR 12. Cloak Unit and Concealment Infrastructure

#### Goals

- Add the base stealth unit and concealment rules.

#### Scope

- Add `CLOAK` unit type and training unlock.
- Add concealment metadata to units.
- Implement:
  - visible on training
  - concealed after qualifying movement
  - reveal on hostile action
  - reveal on bump / attempted enemy move into occupied tile
- Add nearby-cloak warning cues to observations.

#### Files likely touched

- `src/core/Types.java`
- `src/core/actors/units/*`
- unit movement / visibility code
- observation projection logic

#### Dependencies

- PR 9
- PR 4
- PR 5

#### Why now

This is the smallest stealth PR that lands the hidden-unit model before infiltration and daggers.

#### Exit criteria

- Concealed cloaks exist in authoritative state and are correctly redacted from observations.

### PR 13. Infiltration and Sabotage State

#### Goals

- Add the city-side effect of cloak infiltration.

#### Scope

- Add `Infiltrate` unit action.
- Add per-city sabotage/infiltration metadata.
- Implement:
  - cloak removal on infiltration
  - immediate infiltrator star gain
  - zero production on the host’s next turn
- Persist sabotage state through save/load.

#### Files likely touched

- `src/core/actors/City.java`
- infiltration action/command/factory files
- `src/core/game/GameState.java`
- save/load files

#### Dependencies

- PR 12
- PR 8

#### Why now

It isolates sabotage timing from dagger combat and keeps debugging manageable.

#### Exit criteria

- Infiltration works without yet requiring full dagger combat rollout.

### PR 14. Dagger Unit, Spawn Rules, and Surprise Attack

#### Goals

- Complete the cloak branch with spawned daggers and their special combat rule.

#### Scope

- Add `DAGGER` unit type.
- Implement dagger spawn count and placement rules.
- Spawn daggers on infiltration with delayed activation.
- Add surprise-attack behavior:
  - defender does not retaliate against dagger attacks
- Persist dagger state normally as units.

#### Files likely touched

- `src/core/Types.java`
- `src/core/actors/units/*`
- combat command logic
- infiltration resolution code

#### Dependencies

- PR 13

#### Why now

Dagger combat is much easier to validate once infiltration and city sabotage are already stable.

#### Exit criteria

- Daggers spawn correctly.
- Dagger attacks suppress retaliation.

### PR 15. Built-In Agent Compatibility Pass

#### Goals

- Keep the bundled agents usable after the new diplomacy model lands.

#### Scope

- Update `SimpleAgent` diplomacy reads to use tribe relations rather than raw allegiance indices.
- Fix the current convert-scoring bug that indexes diplomacy by unit IDs.
- Update portfolio scripts or leave them relation-agnostic, but remove obviously invalid assumptions.

#### Files likely touched

- `src/players/SimpleAgent.java`
- selected portfolio scripts
- stats code if still needed

#### Dependencies

- PR 6
- PR 11
- PR 14

#### Why now

This should happen after the actual gameplay semantics settle.

#### Exit criteria

- Bundled agents compile and behave sensibly with treaties and stealth present.

### PR 16. Lightweight UI and Documentation Cleanup

#### Goals

- Make the new systems inspectable without overbuilding the UI.

#### Scope

- Add simple GUI indicators for:
  - treaty state
  - embassies
  - known capital markers
  - cloak-nearby warnings
- Update README / docs references to the new rules.

#### Files likely touched

- `src/gui/*`
- `README.md`
- docs

#### Dependencies

- PR 11
- PR 14

#### Why last

Gameplay and bot correctness matter more than presentation.

#### Exit criteria

- A human can inspect diplomacy and stealth state functionally from the debug UI.

## Dependency Summary

### Critical path

1. PR 1: tests + determinism
2. PR 2: tech serialization hardening
3. PR 3: tech metadata
4. PR 4: Strategy/Diplomacy tech placeholders
5. PR 5: explicit diplomacy state
6. PR 6: treaties and hostility legality
7. PR 7: allied roads
8. PR 8: exact resume save/load
9. PR 9: visibility refactor
10. PR 10: embassy entity
11. PR 11: embassy vision + capital vision
12. PR 12: cloak concealment
13. PR 13: infiltration
14. PR 14: dagger
15. PR 15: agent compatibility
16. PR 16: UI/docs cleanup

### Parallelizable opportunities

After PR 5 lands:

- PR 6 and parts of PR 8 can overlap if done carefully

After PR 9 lands:

- PR 10 and low-level cloak data plumbing from PR 12 can be prepared in parallel if write scopes are separated

After PR 14 lands:

- PR 15 and PR 16 can proceed in parallel

## Test Gates by Milestone

### Gate A: before diplomacy rules

Must pass after PR 4:

- deterministic copy behavior
- save/load compatibility
- existing combat/city tech loops unchanged

### Gate B: before embassy

Must pass after PR 8:

- treaties exclude hostile actions correctly
- allied roads work
- exact save/load resume preserves diplomacy state

### Gate C: before cloak

Must pass after PR 11:

- explored vs visible current-vision behavior
- no hidden-action leakage
- embassy and capital vision observations correct

### Gate D: before final cleanup

Must pass after PR 14:

- cloak concealment and reveal rules
- infiltration timing
- dagger spawn rules
- surprise attack behavior
- save/load round-trip for stealth state

## Recommended Immediate Next PR

Start with PR 1 only.

That PR should include:

- a headless test harness
- deterministic `GameState.copy()`
- deterministic ranking tie behavior
- baseline regression tests

It should explicitly avoid:

- tech-tree changes
- diplomacy refactors
- visibility refactors
- new units or actions

## Recommended Branching Approach

Use one branch per PR with narrow write scopes.

Suggested branch naming:

- `codex/pr1-test-determinism`
- `codex/pr2-tech-serialization`
- `codex/pr3-tech-metadata`
- `codex/pr4-strategy-diplomacy-tech`
- and so on

## Notes for Multi-Agent Execution

If we use multiple agents during implementation, the safest split is:

- `Foundation worker`
  - tests
  - deterministic copy
  - serialization
- `Diplomacy worker`
  - relation state
  - treaty actions
  - allied roads
- `Visibility/Stealth worker`
  - explored vs visible
  - cloak concealment
  - infiltration
  - dagger combat

Avoid parallel writes to:

- `src/core/Types.java`
- `src/core/game/Board.java`
- `src/core/game/GameState.java`

Those files are on the critical path for nearly every subsystem and should have a single owner per PR.
