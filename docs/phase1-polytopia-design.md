# Phase 1 Technical Design: Polytopia-Like Base Tribes on Tribes

## Purpose

This document defines the target architecture and migration strategy for repurposing the open-source `Tribes` framework into a Polytopia-like bot-play environment focused on:

- regular/base tribes only
- base technology tree and economy
- deterministic forward simulation
- partial observability and hidden-information support
- diplomacy, embassies, capital vision, cloaks, and daggers
- clean AI-facing interfaces

This is a design document only. It does not change rules yet.

## Scope

### In scope

- Xin-Xi, Imperius, Bardur, Oumaji, Kickoo, Hoodrick as the supported starter tribes
- Base economy, cities, population, level-ups, buildings, monuments, units, naval progression
- Score and capitals-style victory
- Strategy -> Diplomacy branch
- Peace treaties, treaty cancellation, road sharing, embassy, capital vision
- Cloak and Dagger stealth/infiltration system
- Save/load and observation support for the above

### Out of scope

- Special tribes and alternate tribe tech trees
- Monetization, online services, account systems
- Commercial asset cloning
- High-fidelity UI polish

## Current-State Assessment

### What already works well

The current codebase already has the right high-level shape for a bot-play engine:

- `src/core/game/GameState.java`
  - central authoritative simulation state
  - legal action computation per city/unit/tribe
  - copyable forward-model state
- `src/core/game/Game.java`
  - authoritative state plus per-player observation copies
- `src/core/game/Board.java`
  - terrain, resources, buildings, units, cities, capitals, trade network, diplomacy
- `src/core/actors/Tribe.java`
  - stars, tech tree, score, monuments, met tribes, observation grid, extra units
- `src/core/actors/City.java`
  - population, level, production, walls, border growth, buildings, units
- `src/core/Types.java`
  - techs, tribes, units, buildings, actions, city level-up choices
- `src/core/levelgen/LevelGenerator.java`
  - Polytopia-inspired map generator with tribe-biased terrain and resources

The existing action model is especially reusable:

- `Action` + `Factory` + `Command`
- city, unit, and tribe actions are clearly separated
- legality is mostly centralized in `isFeasible()`

This is a strong base for incremental extension.

### Current base-game coverage

The framework already contains:

- the regular tribes in code
- the standard pre-diplomacy regular tech tree
- city growth and level-up rewards
- star production and spending
- building placement and monuments
- the standard land and naval unit roster pre-Diplomacy
- combat, retaliation, veteran promotion, conversion, capture
- map generation, villages, capitals, ports, roads
- save/load for most core entities

### Gaps against the target feature set

#### 1. Diplomacy is not yet a rule system

Current diplomacy is a scalar allegiance matrix in `src/core/Diplomacy.java`.

It supports:

- symmetric integer relationship values
- `DeclareWar`
- `SendStars`

It does not support:

- treaty offers and acceptance
- explicit war vs treaty state
- attack illegality while allied
- treaty cancellation rules
- embassy placement or lifecycle
- capital vision
- observed inter-tribe relation visibility

#### 2. Observation is not yet current-vision based

`Tribe.obsGrid` in `src/core/actors/Tribe.java` is only ever cleared from `false` to `true`.

This means the current implementation behaves like:

- persistent exploration memory
- no recomputation of current line-of-sight

That is insufficient for:

- faithful fog of war
- embassy local vision
- capital vision as a distinct knowledge channel
- hidden cloaks
- adjacent-cloak detection cues

Compounding issue:

- the engine defaults to full observability today because `PLAY_WITH_FULL_OBS` and `GUI_FORCE_FULL_OBS` are both `true` in `src/core/Constants.java`

#### 3. Partial-observation copies are not ready for stealth

`Board.copy(boolean partialObs, int playerId)` hides unseen actors and tiles, but it assumes hidden information is only "not currently seen".

It has no support for:

- units that exist in true state but are intentionally invisible
- player-visible warnings that a hidden unit is nearby
- preserving remembered terrain while hiding current units
- recomputing or filtering action lists against currently visible information

At the moment, action leakage is a real risk:

- the authoritative state computes actions before observations are built
- copied game states inherit those actions
- factories like `AttackFactory` scan the real board directly

#### 4. Forward-model determinism is weaker than required

`GameState.copy(int playerIdx)` constructs copies with `new Random()` rather than preserving deterministic game randomness.

This is acceptable for casual play but is a poor fit for:

- reproducible bot evaluation
- deterministic rollout analysis
- regression tests involving ruins, exploration, or any stochastic event

There is also a secondary reproducibility issue in ranking ties:

- `TribeResult.compareTo(...)` uses `new Random()` as a tiebreaker

#### 5. Tech serialization is fragile for tree migration

`TechnologyTree` stores researched techs as a `boolean[]` indexed by enum ordinal.

That makes changes to `Types.TECHNOLOGY` risky because:

- enum insertions reorder existing serialized meaning
- old saves cannot be trusted after tree edits
- adding `STRATEGY` and `DIPLOMACY` can silently corrupt save compatibility

#### 6. Embassy does not fit the current building ownership model

Current buildings are city-owned objects:

- `src/core/actors/Building.java`
- `src/core/actors/City.java`

That works for farms, ports, monuments, and temples, but not for a diplomacy structure that:

- is placed in another tribe's capital
- benefits the builder and the host
- survives independently of city production rules
- reveals vision for the builder, not the host

#### 7. Save/load does not persist diplomacy state

`GameSaver` and `GameLoader` persist tribes, units, cities, terrain, buildings, and networks.

They do not persist:

- diplomacy relation state
- turn-sensitive diplomacy counters like stars sent / declared war state
- pending treaty offers
- embassies
- capital vision knowledge
- cloak concealment state
- infiltrated city state

The current load path is also replay-oriented rather than exact-resume oriented:

- `GameState(Random, gameMode, tribes, board, tick)` advances `activeTribeID` on load
- that is unsafe once diplomacy cancellation locks and sabotage windows become turn-sensitive

## External Rule References and Working Assumptions

The design aims to stay close to the regular-tribe Diplomacy rules described in:

- Official Diplomacy page: [polytopia.io/diplomacy](https://polytopia.io/diplomacy/)
- Supplementary mechanic summaries:
  - [Diplomacy wiki page](https://polytopia.fandom.com/wiki/Diplomacy)
  - [Embassy wiki page](https://polytopia.fandom.com/wiki/Embassy)
  - [Cloak wiki page](https://polytopia.fandom.com/wiki/Cloak)
  - [Dagger wiki page](https://polytopia.fandom.com/wiki/Dagger)

Rules we will treat as target behavior unless implementation cost forces an explicit approximation:

- `Shields` is effectively replaced by `Strategy`
- `Diplomacy` is researched after `Strategy`
- `Strategy` still unlocks Defenders and additionally enables peace treaties
- `Diplomacy` unlocks Cloak, Embassy, Dagger support, and Capital Vision
- Peace treaties prevent attacks in both directions and allow road sharing
- Cancelling a treaty immediately ends allied status, applies a betrayal restriction, and destroys the canceling tribe's units left in former ally territory
- Embassies are buildable only in another tribe's capital while not at war
- Embassies reveal the surrounding eight tiles and create recurring star income
- Cloaks are hidden until they act, with adjacent detection cues
- Daggers spawn from infiltration, cannot be trained normally, and do not trigger retaliation when attacking

Where the original rules are awkward inside the current engine, the implementation should prefer:

1. rule clarity
2. deterministic simulation
3. bot-usable observations
4. minimal architectural disruption

## Design Goals and Invariants

The extended engine should obey the following invariants:

- The authoritative game state is the single source of truth.
- Player observations are derived from authoritative state and may omit or redact information.
- Legal-action generation for the authoritative state is exact.
- Illegal actions caused by treaty rules are filtered out before they reach agents.
- Hidden units remain present in authoritative state even when absent from observations.
- Save/load is stable across tech-tree and diplomacy additions.
- New systems are queryable from bots without needing GUI-only knowledge.

## Proposed Data Model Changes

### 1. Technology model

#### Current

- `Types.TECHNOLOGY` is enum-only
- `TechnologyTree` stores `boolean[] researched`
- tech cost logic lives in the enum

#### Proposed

Keep the enum-based approach for minimal disruption, but make it serialization-safe and more data-driven:

- Add stable string codes to each technology, for example:
  - `climbing`
  - `organization`
  - `strategy`
  - `diplomacy`
- Update `TechnologyTree` serialization to store researched techs by stable code rather than enum ordinal
- Add a small unlock metadata layer, for example `TechnologyRules`, mapping techs to:
  - unit unlocks
  - building unlocks
  - abilities
  - diplomacy capabilities
  - vision capabilities

#### Strategy migration

Introduce `STRATEGY` as the logical successor to `SHIELDS`.

Migration rule:

- old saves that contain `SHIELDS` should deserialize as `STRATEGY`
- Defenders should require `STRATEGY`
- all references to `SHIELDS` in gameplay code should be migrated to `STRATEGY`

### 2. Diplomacy model

Replace the scalar-only diplomacy model with explicit relation state plus metadata.

#### New relation state

```java
enum RelationState {
    WAR,
    TREATY
}
```

The old allegiance score can be retained as optional metadata for heuristics if desired, but it should no longer determine legality.

This compatibility layer is useful because some built-in agents still read allegiance directly today, especially `src/players/SimpleAgent.java`.

#### New diplomacy metadata

Per tribe pair, track:

- `RelationState state`
- `int treatyStartedTick`
- `int attackLockedUntilOwnTurnStartForTribeA`
- `int attackLockedUntilOwnTurnStartForTribeB`
- `boolean pendingPeaceOfferAToB`
- `boolean pendingPeaceOfferBToA`

This supports:

- offer/accept flow
- cancellation penalties
- observed relationship state
- exact legality checks

### 3. Embassy model

Embassy should not be implemented as a normal `City` building.

#### Proposed new entity

Create a new diplomacy-side entity:

```java
class Embassy extends Actor {
    int builderTribeId;
    int hostTribeId;
    int hostCapitalCityId;
    boolean active;
}
```

Store embassies in `Board` alongside actors, with lookup helpers:

- `getEmbassy(builderTribeId, hostTribeId)`
- `getEmbassiesForTribe(tribeId)`
- `removeEmbassy(builderTribeId, hostTribeId)`

Embassies should be independent from `City.buildings` so ownership remains unambiguous.

### 4. Vision model

Split the current single observation grid into:

- `exploredGrid`
- `visibleGrid`

#### exploredGrid

- persistent memory of tiles discovered by the tribe
- used for remembered terrain and buildings

#### visibleGrid

- current turn visibility
- recomputed from current sources
- controls which units are visible now

#### Vision sources

Current implementation already implies vision sources through `clearView()` calls.

Refactor those into explicit recomputation from:

- owned cities
- owned units
- explorer effect
- embassy local reveal
- optional capital-vision marker channel

Important implementation note:

- `exploredGrid` replaces the current sticky meaning of `obsGrid`
- `visibleGrid` must be recomputed every turn and after local state changes such as move, spawn, capture, embark, disembark, and embassy creation/removal

### 5. Hidden-unit support

Add stealth metadata to units rather than creating a second hidden-unit container.

#### Proposed unit flags

```java
boolean concealed;
boolean canUseSurpriseAttack;
int spawnedTick;
```

For cloaks:

- `concealed = true` after a qualifying move
- `concealed = false` when revealed or while newly trained

For daggers:

- `canUseSurpriseAttack = true`

### 6. Infiltration state

Cities need limited sabotage metadata for cloak infiltration effects.

Add to `City`:

- `int infiltratedByTribeId = -1`
- `int sabotageExpiresOnTick = -1`

This supports:

- blocking repeat infiltration during the sabotage window
- zero-star production on the host's next turn
- clean save/load behavior

### 7. Capital knowledge model

Capital Vision should not force full tile visibility.

Add per observing tribe knowledge records:

- `knownCapitalCityId`
- `knownCapitalPosition`
- `knownCurrentCapitalController`

These records are queryable even when the capital tile itself is not currently visible.

## Proposed Rule Changes

### Technology branch

#### Strategy

- replaces the current `Shields` tech logically
- prerequisite branch remains under the Organization line
- unlocks:
  - Defender
  - peace treaty actions

#### Diplomacy

- prerequisite: `Strategy`
- unlocks:
  - Cloak
  - Embassy
  - Capital Vision
  - observed tribe relation visibility

### Treaty rules

#### Actions

Add tribe actions:

- `OfferPeace(targetTribeId)`
- `AcceptPeace(targetTribeId)`
- `CancelTreaty(targetTribeId)`

`DeclareWar` can be retained as a compatibility action name, but its semantics should be updated to operate on explicit relation state rather than allegiance thresholds.

#### Legality

If tribes are in `TREATY`:

- `Attack`, `Capture`, and `Convert` against the ally are illegal
- legal action generation must exclude them
- allied roads can be used for movement

#### Cancellation

When tribe `A` cancels a treaty with tribe `B`:

- relation becomes `WAR`
- `A` receives an attack lock until the start of `A`'s next turn
- all units owned by `A` inside `B` territory are destroyed immediately
- embassies between `A` and `B` are removed

This follows the official cancellation behavior closely while staying simple enough for deterministic simulation.

### Embassy rules

#### Placement

Embassy is buildable only if:

- builder has researched `Diplomacy`
- builder has met the host tribe
- builder is not at war with the host tribe
- host currently controls its original capital
- there is no existing embassy from builder to host
- builder can pay the embassy cost

#### Effects

On each participant tribe's turn start:

- active embassy yields recurring stars to the builder
- active embassy yields recurring stars to the host

Working assumption:

- base income: `+2`
- if the pair is in treaty state: `+4`

#### Vision

Embassy reveals the `3x3` area centered on the host capital to the builder.

This is a continuous visibility source, not a one-time discovery pulse.

#### Removal

Embassy is removed when:

- either party declares war / cancels treaty
- the host loses control of its original capital
- the capital changes owner

### Capital Vision rules

If tribe `A` has researched `Diplomacy`, then for every tribe `B` in `A.tribesMet`:

- `A` knows `B`'s current capital location
- if the capital changes hands, the knowledge updates
- this information is queryable from observation state

Capital Vision should be represented as knowledge metadata, not as full local reveal.

### Cloak rules

#### Unlock and training

- Cloak is trainable only after `Diplomacy`
- it is a normal authoritative unit in the true state
- it starts visible on the turn it is trained

#### Concealment

A cloak becomes concealed after it moves and ends its move without performing a hostile action.

Concealed cloaks:

- are absent from enemy observed unit lists
- still occupy their true tile in authoritative state
- can block movement into that tile

#### Detection

To stay close to base Diplomacy while keeping the observation model explicit:

- enemy units and cities adjacent to a concealed cloak receive an observation cue that an enemy cloak is nearby
- the cue does not reveal the exact tile
- if an enemy attempts to move onto the cloak's tile, the move is cancelled and the cloak becomes revealed

#### Reveal conditions

A cloak becomes revealed when it:

- attacks
- infiltrates
- captures a village
- is directly bumped into by enemy movement
- is targeted after being revealed

### Infiltration rules

Add a new unit action:

- `Infiltrate(targetCityId)`

#### Feasibility

`Infiltrate` is legal only if:

- acting unit is a Cloak
- cloak is on an enemy city center tile
- the city is not already under active sabotage
- the city is not under direct siege by another cloak effect

#### Effects

On infiltration:

- the cloak is removed
- the infiltrating tribe gains immediate stars equal to the infiltrated city's production
- the infiltrated city produces zero stars on its owner's next turn
- daggers are spawned in the city's borders

### Dagger rules

#### Creation

Daggers are not normally trainable.

They spawn only through cloak infiltration.

#### Spawn count

Spawn count is:

- `min(city level, 5)`

#### Spawn placement

Placement priority:

1. city center if available
2. empty defense-bonus tiles in the infiltrated city's borders
3. other empty legal land tiles in city borders

If not enough legal tiles exist, fewer daggers spawn.

#### Activation timing

Spawned daggers should be created with `FINISHED` status and become usable on their owner's next turn.

This matches the base game more closely and avoids immediate burst actions in the infiltrating turn.

#### Surprise attack

Daggers get a dedicated combat flag:

- when a dagger attacks, the defender does not retaliate

This should be implemented explicitly in combat resolution rather than encoded indirectly through stats.

#### Water approximation

The official game can spawn pirate-like naval variants on water.

For the first regular-tribe implementation, we will intentionally approximate this as:

- spawn only on legal land tiles
- if no legal land tile exists, do not spawn a dagger there

This keeps the initial diplomacy rollout smaller and avoids introducing extra naval-only infiltration units in the same milestone.

## State and Observation Model

### Authoritative state

The authoritative state remains the current `GameState` + `Board` + actors model.

It will be extended to additionally track:

- explicit diplomacy pair state
- embassies
- current visible vs explored grids
- capital knowledge
- cloak concealment
- city sabotage/infiltration status

### Observed state

`GameState.copy(playerId)` should remain the public observation API, but the implementation should change.

Observed copies must obey:

- unexplored tiles -> `FOG`
- explored but not currently visible tiles -> remembered terrain/building/resource snapshot
- current units visible only if currently seen and not concealed
- concealed cloaks omitted from board units
- cloak-nearby warning cues exposed as structured observation metadata
- capital vision exposed as known-capital metadata

Observed action lists must also be rebuilt or filtered against observation state.

Otherwise, hidden units will continue to leak through available actions even if the board copy is correct.

### Forward-model note

The authoritative forward model must stay exact.

Observation-based imperfect-information rollouts are a separate concern. The engine should support them by exposing redacted state cleanly, but it does not need to solve determinization on behalf of every bot.

## API Additions for Bots

Add bot-friendly query helpers rather than forcing agents to infer rules from GUI state:

- `GameState.getRelationState(a, b)`
- `GameState.getPendingPeaceOffers(playerId)`
- `GameState.getKnownCapitalLocations(playerId)`
- `GameState.getEmbassiesFor(playerId)`
- `GameState.getDetectedAdjacentCloaks(playerId)`
- `GameState.isCitySabotaged(cityId)`

These APIs should work in both authoritative and observed states.

## Save/Load Changes

Extend save/load to serialize:

- technology by stable tech code rather than ordinal-position arrays
- diplomacy pair state and pending offers
- embassy entities
- capital knowledge if stored persistently
- explored/visible grids as required
- unit concealment state
- city sabotage/infiltration metadata

Backward compatibility target:

- old saves should still load if they only use the pre-diplomacy ruleset
- legacy `SHIELDS` research should map to `STRATEGY`

## Migration Plan

The work should be implemented in the following order.

### Stage 1. Test harness and deterministic simulation

- add a headless test harness
- fix `GameState.copy()` RNG behavior
- remove random tie-breaking from ranking or make it deterministic
- cover baseline regression scenarios before rule changes

### Stage 2. Technology serialization hardening

- add stable technology codes
- migrate `SHIELDS` -> `STRATEGY`
- add `DIPLOMACY`
- centralize unlock metadata

### Stage 3. Explicit diplomacy state machine

- replace legality-critical allegiance logic with `RelationState`
- add treaty offer/accept/cancel flow
- persist diplomacy state
- keep a compatibility wrapper for old heuristic code where useful

### Stage 4. Allied roads and treaty legality

- update movement legality and road handling
- update attack/capture/convert legality
- add betrayal lock behavior

### Stage 5. Embassy and Capital Vision

- add embassy entity, action, persistence, and per-turn income
- add capital knowledge channel
- expose both through bot APIs

### Stage 6. Visibility refactor

- split `exploredGrid` from `visibleGrid`
- recompute current vision from sources
- update observation copy logic

### Stage 7. Cloak support

- add cloak unit and concealment state
- add adjacent detection cues
- add reveal-on-bump behavior

### Stage 8. Infiltration and Dagger support

- add infiltration action
- add sabotage state
- spawn daggers with delayed activation
- add surprise-attack combat handling

### Stage 9. Agent and UI cleanup

- update built-in heuristics that directly read allegiance integers
- fix existing heuristic bugs that index diplomacy by unit IDs instead of tribe IDs
- add lightweight GUI overlays for treaties, embassies, and cloak detection cues

## Risks and Mitigations

### Risk: visibility refactor touches many systems

Mitigation:

- isolate vision recomputation behind helper methods
- do not mix embassy/capital vision implementation into unrelated movement code

### Risk: tech-tree changes break save compatibility

Mitigation:

- move serialization to stable tech codes before inserting new techs

### Risk: embassy ownership fights current city-building assumptions

Mitigation:

- implement embassy as a separate actor/entity, not as a `City` building

### Risk: hidden cloaks create ambiguous observation semantics

Mitigation:

- keep authoritative state exact
- make the observation reducer responsible for redaction
- expose explicit nearby-cloak warning metadata

### Risk: built-in heuristic agents depend on old diplomacy

Mitigation:

- provide a compatibility method during migration
- update `SimpleAgent` in a contained follow-up PR

## Test Plan

Every new mechanic should land with focused headless tests.

### Baseline regression tests

- tech cost scales with city count
- city level-up rewards still match existing behavior
- combat and retaliation remain unchanged for non-dagger units
- score and capitals victory still resolve correctly

### Diplomacy tests

- peace offer becomes pending
- peace acceptance creates treaty state for both sides
- attack/capture/convert actions are excluded while in treaty
- cancel treaty applies attack lock to canceling tribe
- cancel treaty destroys canceling tribe units in former ally territory
- allied roads are usable for movement

### Embassy tests

- embassy can only be built in a met tribe's original capital while not at war
- embassy reveals a `3x3` area around the host capital
- embassy income is paid each turn to both sides
- treaty bonus income is applied if configured
- declaring war removes relevant embassies
- capital capture removes invalid embassies

### Capital Vision tests

- researching Diplomacy reveals met tribes' capital locations
- newly met tribes are added automatically
- captured capitals report updated controller/location metadata

### Visibility tests

- explored tiles remain remembered after leaving vision
- units disappear from observation when no longer visible
- visible tiles update correctly from units, cities, and embassies

### Cloak tests

- newly trained cloak is visible until it moves
- moved cloak becomes concealed
- concealed cloak is absent from enemy observed unit lists
- adjacent enemy unit receives cloak-nearby warning
- moving onto concealed cloak tile reveals the cloak and cancels the move

### Infiltration and Dagger tests

- cloak can infiltrate an enemy city
- infiltration grants stars equal to city production
- infiltrated city produces zero stars on next owner turn
- daggers spawn up to `min(city level, 5)` on legal tiles
- spawned daggers cannot act until next owner turn
- dagger attacks do not trigger retaliation

### Persistence tests

- diplomacy state round-trips through save/load
- embassies round-trip through save/load
- concealed cloaks and sabotage state round-trip through save/load
- legacy pre-diplomacy saves still load

## Recommended First Implementation PR

The first implementation PR should not start with cloaks.

It should do only:

- add a headless test harness
- fix deterministic RNG copying in `GameState.copy()`
- harden tech serialization with stable tech codes
- introduce `STRATEGY` and `DIPLOMACY` as non-functional placeholders

That provides a safe base for the diplomacy and stealth work that follows.
