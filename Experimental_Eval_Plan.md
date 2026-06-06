# Experimental Static Eval Recalibration Plan

## Objective

Recalibrate the experimental native static evaluator so military research is chosen only when it unlocks a concrete near-term action or answers a concrete board threat.

This is a policy-prior change for `py/search/native/native_static_eval.cpp`. Keep the tuned economic/resource logic intact and keep experimental research as a stricter overlay on top of tuned scoring:

```cpp
research_score_experimental =
    research_score_tuned
  + stricter_contextual_overlay;
```

Do not change `state_value_experimental`, the external bot protocol, or bot CLI behavior. The only optional runtime surface is debug logging behind an environment variable.

---

## Current Implementation To Rework

The current experimental path is selected with `TRIBES_STATIC_EVAL_VARIANT=experimental`.

Relevant existing code:

- `StaticEvalVariant::Experimental` chooses `action_score_experimental`.
- `action_score_experimental` only changes research scoring and otherwise delegates to tuned scoring.
- `research_score_experimental` adds `military_research_overlay` to `research_score_tuned`.
- `MilitaryResearchContext` is built once per legal research action.
- `road_score_tuned` is shared by tuned and experimental road actions.
- Existing road helpers already reason about city-connected components:
  - `city_ids_connected_by_candidate_road`
  - `candidate_merges_city_connections`
  - `candidate_extends_city_connection`
  - `road_bonus_readiness`

The main issue is that the overlay still rewards broad context such as city count, generic enemy-city proximity, and "has any road action". The recalibration should replace those broad bonuses with explicit gates: contact, threat, exploitability, and opportunity cost.

---

## Workstream 1: Research Context Gates

Extend `MilitaryResearchContext` with concrete board signals:

```cpp
int visible_enemy_units = 0;
int visible_enemy_cities = 0;
bool enemy_can_threaten_owned_city = false;
bool enemy_can_threaten_capital = false;
bool contested_village_exists = false;
int frontline_enemy_city_proximity = 0;
int friendly_unit_pressure_on_enemy_city = 0;
bool urgent_non_research_spend = false;
```

Definitions:

- `visible_enemy_units`: visible, alive, non-hidden enemy units.
- `visible_enemy_cities`: visible enemy-owned cities.
- `enemy_can_threaten_owned_city`: any visible enemy unit can move-plus-attack an owned city tile using `unit_mobility_value + unit_range_value` and Chebyshev distance.
- `enemy_can_threaten_capital`: same as above, but only for the capital.
- `contested_village_exists`: visible village where at least one friendly unit and one visible enemy unit are close enough to plausibly contest capture soon.
- `frontline_enemy_city_proximity`: enemy city is close to owned territory, regardless of whether friendly units can pressure it.
- `friendly_unit_pressure_on_enemy_city`: friendly units can threaten, surround, or approach an enemy city soon.
- `urgent_non_research_spend`: a legal non-research action is more urgent than saving/spending stars on military tech.

Keep existing aggregate fields like `enemy_durable_melee`, `enemy_ranged`, `wall_city_break_need`, and `forest_fronts`, but treat them as supporting signals. They should not create large overlays unless contact/exploitability gates pass.

---

## Workstream 2: Urgent Non-Research Spend

Add:

```cpp
bool has_urgent_non_research_spend(
    const NativeGameState& state,
    const std::vector<NativeAction>& actions,
    const std::vector<int>& legal_action_indexes,
    const MilitaryResearchContext& ctx);
```

Return true when a legal non-research action should normally outrank military research:

- capital is threatened and `SPAWN` or `RECOVER` can respond;
- an owned city is threatened and a defensive `SPAWN` is legal;
- `CAPTURE`, `EXAMINE`, `MAKE_VETERAN`, or a clearly high-value attack is legal;
- `RESOURCE_GATHERING` can level a city now or nearly now;
- a village capture or village-progress move is legal;
- stars after research would block the best available spawn/build/resource action.

Use this as an opportunity-cost penalty, especially for tier-2 and tier-3 military tech. Do not use it to suppress economic/resource tech from tuned scoring.

---

## Workstream 3: Tech-Specific Exploitability

Add:

```cpp
bool can_exploit_researched_tech_soon(
    const std::string& tech,
    const NativeGameState& state,
    const std::vector<NativeAction>& actions,
    const std::vector<int>& legal_action_indexes,
    const MilitaryResearchContext& ctx);
```

This should return true only when the researched tech unlocks a concrete near-term action or response.

### `RIDING`

True only if rider use is plausible soon:

- an owned city can spawn after research;
- stars after research can afford a rider soon;
- and at least one concrete use exists:
  - rider reaches a village/ruin faster than current units;
  - rider helps defend a threatened owned city;
  - rider can safely pressure visible enemy units.

Before contact, allow `RIDING` only for concrete expansion tempo.

### `ROADS`

True only if a concrete road follow-up exists:

- player can afford at least one road after research;
- and at least one legal or near-legal road use exists:
  - candidate road connects two owned city components;
  - candidate road connects an owned city component toward the capital;
  - candidate road extends a city-connected component toward another owned city;
  - road lets an existing unit reach a village, enemy city, or defensive tile earlier.

Reuse the existing road component helpers where possible. Do not count arbitrary adjacency to an existing road unless that component is connected to an owned city/capital network.

### `ARCHERY`

True only if archer use is front-relevant soon:

- archer spawn/use is possible soon;
- and at least one concrete need exists:
  - visible durable melee/defenders;
  - relevant forest-front defense;
  - friendly melee can protect archers;
  - ranged chip is needed against enemy city defense.

Do not boost `ARCHERY` before contact just because it is available.

### `STRATEGY`

True only if a defensive/diplomatic use is concrete:

- owned city or capital is under threat and defender spawn is possible soon;
- or peace/treaty action is legal while enemy pressure is real.

Do not boost `STRATEGY` merely because the tech is available or because generic enemy units are visible.

### `SMITHERY`

True only if at least one follow-up exists:

- swordsman spawn is possible soon;
- forge/mine follow-up exists in owned city territory;
- stronger melee parity is needed against visible durable units;
- friendly units are already pressuring an enemy city and swordsmen improve that attack.

Use `friendly_unit_pressure_on_enemy_city`, not generic enemy-city proximity, for offensive boosts.

### `MATHEMATICS`

True only if siege/economy follow-up is immediate:

- enemy city, walled city, or high-level city is near the front;
- catapult spawn is possible soon;
- friendly units can protect catapults;
- or sawmill follow-up is immediately useful in owned territory.

Do not boost `MATHEMATICS` for a distant enemy city with no protected catapult timing.

### `CHIVALRY`

True only if knight timing is real:

- knight spawn is possible soon;
- and at least one concrete access/target signal exists:
  - multiple damaged or low-HP enemies are visible;
  - exposed ranged/siege units exist;
  - road/mobility path gives knight access;
  - friendly army already controls the front.

Penalize or withhold boosts when enemy defenders/swordsmen/giants dominate and no chain targets are available.

---

## Workstream 4: Overlay Shape

Replace `military_research_overlay` with a stricter shape:

```cpp
overlay =
    tech_specific_need
  + tech_specific_exploitability
  + immediate_tactical_use
  - opportunity_cost;
```

Rules:

- If `can_exploit_researched_tech_soon(...)` is false, do not give large military bonuses.
- If there is no contact:

```cpp
if (ctx.visible_enemy_units == 0 &&
    ctx.visible_enemy_cities == 0 &&
    !ctx.contested_village_exists) {
    // Allow strong overlays only for:
    // - RIDING with concrete expansion tempo;
    // - ROADS with immediate city/capital network value;
    // - tuned economic/resource techs outside this military overlay.
}
```

- Tech-specific need must come from enemy composition, city threat, defended enemy city targets, or immediate expansion tempo.
- Immediate tactical use means the tech enables a spawn/build/upgrade/road/move consequence soon.
- Opportunity cost should reduce expensive military research when urgent spending exists.
- Keep clamp bounds conservative so experimental cannot overwhelm tuned scoring without passing concrete gates.

---

## Workstream 5: Road Research And Road Actions

### `ROADS` Research

Boost `ROADS` only when `can_exploit_researched_tech_soon("ROADS", ...)` passes:

- city/capital connection exists;
- city-to-city connection exists;
- city-connected component makes progress toward another owned city;
- unit reaches a village/enemy city/defensive tile earlier;
- frontline reinforcement route exists.

This should reuse current road network helpers rather than creating separate road adjacency logic.

### `BUILD_ROAD`

Keep road action scoring centered on city-network value:

- connects two owned city components;
- connects a component to the capital network;
- extends a city-connected component toward another owned city;
- enables a current unit to reach a concrete target earlier.

Do not reward roads merely because they branch from or touch an existing road. The existing tests for road branches should remain true for tuned and experimental scoring.

If unit-tempo roads are added, keep them narrow: score only when the road shortens a path to a visible village, enemy city, threatened city, or tactical defensive tile.

---

## Workstream 6: Focused Helpers

Add small helpers only where they remove repeated logic:

```cpp
bool can_spawn_unit_type_soon(state, player_id, unit_type, stars_after_research);
bool has_city_resource_followup(state, player_id, tech);
bool has_safe_catapult_position(state, player_id);
bool has_knight_chain_targets(state, player_id);
bool has_city_connection_road_followup(state, player_id);
bool has_unit_tempo_road_followup(state, player_id);
```

Intended use:

- `SMITHERY`: swordsman spawn, mine/forge follow-up, or melee parity.
- `MATHEMATICS`: protected catapult spawn or sawmill follow-up.
- `CHIVALRY`: knight spawn plus chain/access targets.
- `ROADS`: city connection or unit-tempo follow-up.
- `ARCHERY`: protected/front-relevant archer use.
- `RIDING`: expansion, defense, or safe pressure.

Prefer approximate local checks over expensive legal-action simulation unless a helper can reuse existing legal actions or existing road/component functions.

---

## Workstream 7: Debug Logging

Add optional logging behind:

```text
TRIBES_STATIC_EVAL_DEBUG_RESEARCH=1
```

For each research action, log a compact line with:

```text
tick
stars
tech
base_tuned_score
overlay_score
final_score
can_exploit_soon
urgent_non_research_spend
visible_enemy_units
visible_enemy_cities
enemy_can_threaten_owned_city
enemy_can_threaten_capital
contested_village_exists
frontline_enemy_city_proximity
friendly_unit_pressure_on_enemy_city
reason_flags
```

For road actions, log only when road debug is enabled:

```text
tile
stars
connects_city_component
connects_capital_network
extends_city_network
shortens_unit_target_path
raw_score
readiness
final_score
```

The purpose is to verify that research and road actions are selected because of concrete board context rather than generic bonuses.

---

## Implementation Order

1. Add the new `MilitaryResearchContext` fields and populate them in `build_military_research_context`.
2. Add urgent-spend and tech-exploitability helpers.
3. Rewrite `military_research_overlay` around gated need/exploitability/tactical-use/opportunity-cost terms.
4. Connect `ROADS` research to existing road component helpers.
5. Add optional debug logging.
6. Add focused tests before running broad test suites.

---

## Tests

Add or update focused tests in `py/tests/test_native_mcts.py`:

- no-contact `ARCHERY`, `STRATEGY`, `SMITHERY`, `MATHEMATICS`, and `CHIVALRY` stay at or below tuned priority;
- no-contact `RIDING` can beat tuned only when expansion tempo exists;
- no-contact `ROADS` can beat tuned only when a city/capital road follow-up exists;
- threatened capital/owned city boosts only usable defensive tech;
- `MATHEMATICS` boosts only with protected catapult timing or immediate sawmill follow-up;
- `CHIVALRY` boosts only with knight spawn plus chain/access context;
- `ROADS` research requires a real city-network or unit-tempo follow-up;
- road branch actions stay below spawn/capture/resource actions unless they connect or progress a city network;
- experimental still equals tuned when no research actions are present.

Verification commands:

```powershell
$env:PYTHONPATH = "$PWD\py"
python -m pytest py/tests/test_native_mcts.py -q
python -m pytest py/tests -q
```

Optional smoke:

```powershell
.\scripts\build_java.ps1
& "$env:JAVA_HOME\bin\java.exe" -cp "out;lib/json.jar" Tournament road_debug_tournament.json
```

For A/B validation, run tuned vs experimental with swapped seats and fixed seeds. Track win rate, top action type frequency, research timing, road action frequency, and any debug-reason patterns that show broad bonuses still leaking through.

---

## Non-Goals

- Do not modify Java rules; Java remains authoritative.
- Do not change the external bot JSON protocol.
- Do not change NN encoding, replay format, or training defaults.
- Do not change `state_value_experimental` in this pass.
- Do not preserve existing experimental research behavior when it conflicts with concrete exploitability gates.
