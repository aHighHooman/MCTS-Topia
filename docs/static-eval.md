# Static Evaluation

Static eval lives in `py/search/native/src/static_eval.cpp`. It produces two things:

- action priors for the legal action list;
- a scalar value in `[-1, 1]` from the active player's perspective.

The value is used by static MCTS directly and by analysis tools. The priors are also used to order and bias static search.

## Public Entry Points

The native extension exports:

- `evaluate_static(payload, max_actions)`;
- `evaluate_static_batch(payloads, max_actions)`;
- `evaluate_static_breakdown(payload, max_actions)`;
- `evaluate_static_breakdown_batch(payloads, max_actions)`;
- `evaluate_action_breakdown(payload, action_id, max_actions)`.

The standalone static bot calls the same implementation after translating protocol JSON into the native payload shape.

## Action Priors

Priors are built by scoring each legal action with `action_score_baseline`, then softmaxing scores with temperature `1.5`.

If all scores are tied, the priors are uniform. Negative scores are allowed before softmax; they simply get lower probability.

Important action-score behavior:

| Action type | Current treatment |
| --- | --- |
| `CAPTURE` | High priority, especially villages/cities/capitals. |
| `EXAMINE`, `MAKE_VETERAN` | Very high priority. |
| `ATTACK` | Uses combat forecast and tactical context. |
| `MOVE`, `STEP_MOVE` | Rewards progress toward villages, ruins, enemy cities, and useful frontier positions. |
| `RECOVER` | Rewards healing missing HP, reduced by danger. |
| `BUILD` | Values farms/mines/lumber/economic buildings, level pressure, city level, and star comfort. |
| `SPAWN` | Values unit type and local city threat. |
| `RESOURCE_GATHERING` | Rewards near-level city growth. |
| `LEVEL_UP` | Strong preference for superunit, resources/workshop, border growth. |
| `BUILD_ROAD` | Only positive when connecting/extending useful city networks, with readiness penalties. |
| `RESEARCH_TECH` | Values tech tier, visible resources, city count, and affordability. |
| destructive/diplomacy/star-send actions | Usually low or negative unless explicitly useful. |
| `END_TURN` | Negative baseline score. |

Static eval variants do not currently change action priors. Tests explicitly check that baseline and experimental priors remain stable for several scenarios.

## Value Shape

For non-terminal states:

```text
raw = weighted feature sum
value = tanh(raw / 200)
```

For terminal states with known result:

```text
value = terminal_value
```

The `evaluate_static_breakdown` output reports raw-space additive terms and the local tanh sensitivity. The `linearized_value` field is not an ablation; it is the local derivative approximation around the current raw total.

## Baseline Value Terms

The fast baseline path uses these conceptual terms:

| Group | Direction |
| --- | --- |
| Own unit material | Positive. |
| Enemy unit material | Negative. |
| Own unit power | Positive. |
| Enemy unit power | Negative. |
| Own city quality | Positive. |
| Visible enemy city quality | Negative. |
| Own stars | Positive. |
| Best enemy stars | Negative. |
| Own score | Positive. |
| Best enemy score | Negative. |
| Researched tech | Positive. |
| Visible villages/control | Positive. |
| Exploration ratio | Positive. |
| Pressure on enemy cities | Positive. |
| Vulnerable own units | Negative. |
| Capital threat | Negative. |
| City threat | Negative. |
| Wounded own units | Negative. |

Baseline city quality is roughly:

```text
3.0
+ 1.7 * city level
+ 5.5 * production
+ 0.45 * population
+ 2.0 if walls
+ optional capital bonus, currently passed as 0
```

Baseline unit power uses reach, attack, HP fraction, defense anchor, small veteran bonus, and kill contribution.

## Variants

The active variant is selected by `TRIBES_STATIC_EVAL_VARIANT`.

| Variant | Current behavior |
| --- | --- |
| `baseline` | Baseline unit/material/city/tech formulas and baseline weights. |
| `experimental` | Uses experimental unit-power family and vulnerable-unit formula, excludes separate own/enemy material term, and uses lower unit-power weight. It does not enable the broader experimental city/resource feature path. |
| `experimental-2` | Same experimental family as `experimental`, plus per-tech researched-value terms. |
| `experimental-training` | Uses training-formula unit-power terms, decomposed city-quality terms, per-tech researched terms, resource-potential terms, and unit-capacity terms. Intended for fitting/tuning experiments. |

Important implementation detail: the internal `experimental_eval` flag is currently true only for `experimental-training`. That means `experimental` and `experimental-2` are not "all experimental features"; they mainly switch unit-power/value-head family behavior.

## Experimental Unit Power

Experimental unit power uses:

- attack projection;
- defense projection;
- reach;
- HP fraction;
- estimated incoming damage and survival attacks;
- large veteran bonus;
- kill bonus capped by two kills;
- a projection scale factor.

The training variant exposes separate tunable terms:

- `military.unit_power.training_formula.material`;
- `military.unit_power.training_formula.attack`;
- `military.unit_power.training_formula.defense`;
- `military.unit_power.training_formula.veteran`;
- `military.unit_power.training_formula.kills`.

## Experimental Training City Terms

`experimental-training` decomposes city quality into:

- `economy.city_quality.base_city`;
- `economy.city_quality.income`;
- `economy.city_quality.level`;
- `economy.city_quality.population_progress`;
- `economy.city_quality.walls`.

It also enables resource-potential and unit-capacity features that are otherwise effectively inactive in the fast baseline/experimental paths.

## Weight Overrides

`TRIBES_STATIC_EVAL_WEIGHT_OVERRIDES` accepts comma or semicolon separated `name=value` entries.

Examples:

```text
military.unit_power=0.25
economy.stars=0.5
military.unit_power.training_formula.attack=0.9
```

Overrides can target exact terms. Parent-term overrides can also affect children where parent override is enabled. Analysis targets pass these through as `weight_overrides`, `static_eval_weight_overrides`, or `overrides`.

The standalone static bot exposes the same mechanism with:

```powershell
--static-eval-weight-overrides "military.unit_power=0.25,economy.stars=0.5"
```

## Breakdown and Action Explanation

`evaluate_static_breakdown` explains the root state. `evaluate_action_breakdown` applies one root action first, then explains the resulting child state.

This is what powers:

- value-breakdown reports;
- root-child value matrices;
- pairwise action explainers;
- static weight tuning constraints.

## Practical Caveats

- Static action priors are hand-authored heuristics, not learned policy priors.
- Static value is active-player perspective and must be converted before root-perspective backup.
- Visible enemy values depend on what is in the current hidden-information observation.
- `experimental` is not a broad replacement for baseline value; `experimental-training` is where the larger decomposed feature set appears.
- A high term fit/sign accuracy in static-value-result tuning has not consistently translated into stronger tournament validation.
