# Complete Java-to-C++ Native Rules Port

## Summary
Port the authoritative Java forward model into the native C++ MCTS layer in correctness-first phases. The target is exact parity with `GameState.copyForPlayer(playerId)`, `GameState.advance(action, true)`, `computePlayerActions`, legal action order, terminal/winner/ranking fields, fog-of-war observation, and RNG-visible outcomes. Any unported rule path remains a hard crash until the parity harness proves it matches Java.

The current strict native layer stays in place as the safety boundary: no fallback terminal penalties, no approximate opponent turns, and no synthetic “best effort” states.

## Key Changes
- Add a Java oracle parity harness before expanding C++ rules:
  - Add a Java CLI harness that loads deterministic fixtures, serializes a root observed state plus legal actions, applies each legal action with `GameState.advance(action.copy(), true)`, and emits canonical JSON child states.
  - Add a Python parity runner that feeds the same root/action into C++ `NativeMCTS`/`apply_action_strict`, normalizes both outputs, and diffs child observation, actions, terminal fields, active player, winner/ranking, and action ids.
  - Make parity failures print the exact state id, action id/type, first differing JSON path, Java value, and C++ value.

- Replace the prototype C++ state model with a full native model:
  - Extend `NativeTile`, `NativeUnit`, `NativeCity`, `NativeTribe`, and `NativeGameState` to include every serialized Java field needed by rules: unit stats, city buildings/unit ids/points, diplomacy relationships, tribe tech/results/capital, ranking, turn flags, map/game mode, lighthouses, roads/trade-relevant data, and RNG snapshot.
  - Keep `py::dict observation` only as serialization output/cache; C++ rules mutate typed state first, then serialize from typed state.
  - Parser remains strict: missing required fields, unknown enums, duplicate ids, impossible board/entity references, or unsupported version throws.

- Expand Java payloads to be parity-complete:
  - Add a protocol version for the native-parity payload.
  - Include stable action ids for both root and forward-model child states.
  - Include full action payloads for all 30 `Types.ACTION` variants.
  - Add deterministic RNG snapshot/seed data for native-compatible ruin/examine and any randomized rule.
  - Preserve player-observed semantics: payloads represent the same observed copy Java gives the bot, not hidden authoritative state.

- Port rules subsystem-by-subsystem, each gated by oracle tests:
  - Core state/board helpers: lookup by id/position, actor ownership, city/unit membership, roads, terrain/resource/building access, visibility/exploration, push/kill/remove helpers.
  - Static enums/constants: copy Java `Types`, `TribesConfig`, `UnlockRules`, tech tree costs/prereqs, unit stats, building/resource costs, game modes, terrain passability.
  - Action feasibility and commands for all city actions: `SPAWN`, `BUILD`, `RESOURCE_GATHERING`, `LEVEL_UP`, `BURN_FOREST`, `CLEAR_FOREST`, `GROW_FOREST`, `DESTROY`.
  - Action feasibility and commands for all tribe actions: `END_TURN`, `RESEARCH_TECH`, `BUILD_ROAD`, `BUILD_EMBASSY`, peace/treaty propose/accept/cancel.
  - Action feasibility and commands for all unit actions: `MOVE`, `STEP_MOVE`, `ATTACK`, `CAPTURE`, `CONVERT`, `RECOVER`, `HEAL_OTHERS`, `EXAMINE`, `INFILTRATE`, `DISBAND`, `MAKE_VETERAN`, `UPGRADE_RAMMER`, `UPGRADE_SCOUT`, `UPGRADE_BOMBER`.
  - Legal action factories: C++ generation must match Java action order and ids exactly after every step.
  - Turn/terminal logic: port `endTurn`, `initTurn`, production, unit refresh/recover, temple growth, markets/embassies, `gameOver`, capital/glory/might terminal behavior, ranking, and terminal reward.

- Refactor native MCTS integration:
  - `NativeMCTS` continues to call only `apply_action_strict`.
  - Remove any remaining telemetry/bootstrap code that refers to approximate/invalid native states once tests no longer need compatibility checks.
  - Add an optional `TRIBES_NATIVE_PARITY_TRACE=1` mode to dump root/action/child JSON when C++ throws during search.

## Test Plan
- Oracle parity tests:
  - Fixture suite for every action type, including failure-prone cases from `RegressionHarness`: diplomacy, fogged roads/builds, capture, movement/push, cloak/infiltrate, ruins, naval upgrades, markets, temples, capital victory.
  - Recursive parity at depths 1-3 on small deterministic games. For every legal action at each state, Java child and C++ child must match canonical JSON exactly.
  - Random seeded parity smoke: generate small maps and sample legal actions until each action type has coverage.

- Strictness tests:
  - Unknown enum/action type throws.
  - Missing required payload fields throw.
  - Failed feasibility/command parity throws.
  - C++ output never contains `native_unsupported_transition`, `native_invalid_transition`, `native_approximate_transition`, or penalty terminal states.

- Existing regression:
  - `python py/tests/test_native_mcts.py`
  - Java compile: `javac -cp lib/json.jar -d out ...`
  - Java `RegressionHarness` remains green.
  - End-to-end native static MCTS bot smoke with zero invalid-action fallback logs.

## Milestones
- Milestone 1: Oracle harness and canonical JSON diff are in place; current C++ fails loudly on unported paths.
- Milestone 2: Typed native state and payload contract cover all fields needed by Java rules; no rule changes yet.
- Milestone 3: City and tribe actions match Java at depth 1.
- Milestone 4: Unit movement/combat/capture/conversion and generated action order match Java at depth 1.
- Milestone 5: End-turn, visibility, terminal/ranking, diplomacy, and RNG-visible paths match Java at depths 1-3.
- Milestone 6: Native MCTS can run self-play without parity crashes on the supported game configuration; any crash is treated as a real bug.

## Assumptions
- Full parity means parity with Java’s player-observed `GameState` copy.
- The native implementation must be a C++ rules port, not a Java bridge.
- Correctness beats speed until the oracle harness is green.
- `Capitals`/`Might` self-play is the first acceptance target; broader modes remain fail-fast until covered by parity fixtures.
- Existing strict crash behavior remains mandatory throughout the port.