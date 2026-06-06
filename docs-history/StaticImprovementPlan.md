**MCTS Static Bot Improvement Test Backlog**

**Summary**
I inspected the native/static MCTS path and the Java rules. The best opportunities are not just “more eval bonuses”; several current static assumptions are mismatched with Java rules, and the search is likely wasting depth on poorly ordered atomic actions. Key reference points: [native_static_eval.cpp](/C:/Users/Umair/OneDrive/Desktop/Work/Self_Projects/Tribes_MCTS/py/search/native/native_static_eval.cpp:171), [native_mcts.cpp](/C:/Users/Umair/OneDrive/Desktop/Work/Self_Projects/Tribes_MCTS/py/search/native/native_mcts.cpp:1354), [static_mcts.py](/C:/Users/Umair/OneDrive/Desktop/Work/Self_Projects/Tribes_MCTS/py/search/native/static_mcts.py:89), [TribesConfig.java](/C:/Users/Umair/OneDrive/Desktop/Work/Self_Projects/TribesTopia/Tribes/src/core/TribesConfig.java:7), and [City.java](/C:/Users/Umair/OneDrive/Desktop/Work/Self_Projects/TribesTopia/Tribes/src/core/actors/City.java:152).

**Direct Ideas To Test**
1. Fix static unit stats and prefer parsed unit fields over hardcoded tables. Current static attack/defence/range/movement differs from Java for several units, including knight, cloak, dagger, rammer, scout, bomber, superunit, and catapult.
2. Replace approximate attack scoring with exact combat forecast: Java’s combat uses HP-scaled attack/defence, city/wall/terrain defence bonuses, retaliation rules, persist/escape/stiff behavior, and special unit rules.
3. Add same-turn tactical pair scoring: move→attack, attack→capture, attack→make veteran, rider attack→escape, knight kill chains, cloak infiltrate→dagger pressure.
4. Make resource/build/level-up scores use actual marginal economy: immediate population, production, next-turn income, city level threshold, unit-cap pressure, and building adjacency from `City.updateBuildingEffects`.
5. Rework research scoring around immediate unlock value: visible resources enabled, available unit/building actions unlocked, city count tech cost, Philosophy discount, current stars, and map type.
6. Sort child legal actions by static prior before progressive widening. Native selection only opens the first `8 + sqrt(visits)` action slots, so high-prior generated child actions can be delayed if generation order puts them late.
7. Replace broad root “always keep” pruning with quotas: keep all urgent tactical actions, but cap low-value roads/builds/research by score while preserving at least one plausible action per unit/city.
8. Make `END_TURN` dynamic: penalize ending with fresh units, useful resources/builds, or level-ups available; reward ending when spending would block a better tech/unit next turn.
9. Make state value mode/stage-aware: MIGHT/Capitals should overweight capital control and lethal threats; score/perfection modes should overweight points, temples, monuments, and turn remaining.
10. Build exact threat maps from legal attack reach rather than Chebyshev approximations, including roads, water movement, city defense, ranged units, and visible enemy unit statuses.

**Bolder Ideas**
11. Add a “turn-completion evaluator”: from each leaf, greedily finish the current turn using static policy before evaluating, so a single atomic action is judged by its natural follow-up sequence.
12. Test macro-action MCTS: generate top action bundles such as research→gather→level-up, move→attack→capture, road chain, or spawn→defend, then return only the first real legal action.
13. Add transposition/canonical state reuse. Many same-turn action orders commute, especially resource gathering, roads, builds, and level-ups; merging equivalent states could massively increase effective depth.
14. Add progressive bias/RAVE-style updates using static action features, so good action types/targets get credit across sibling branches before full visit counts mature.
15. Split planning into tactical MCTS plus economic beam search/knapsack: combat stays tree-searched, while stars/build/research spending is optimized as a constrained one-turn plan.
16. Learn static eval weights from tournaments or self-play data. Keep the fast C++ feature extractor, but fit weights with CMA-ES, Bayesian optimization, or TD targets instead of hand tuning.
17. Make static eval belief-aware: use likely unseen villages/capitals/enemies from explored-map geometry and the existing belief code to improve scouting and fog-of-war decisions.
18. Add adaptive search allocation: more breadth early/exploration turns, more depth during combat, lower exploration when one action dominates, and per-action-type budgets.
19. Incrementalize static eval internals: cache nearest village/ruin/city, threat/support maps, and city-quality deltas per node to raise nodes/sec significantly.
20. Add opponent-response scripts for leaf value: after our candidate turn, let visible enemies take one cheap scripted reply before evaluation to punish exposed units/capitals.

**Test Plan**
Run each idea as an isolated experiment against the current tuned static bot using fixed seeds, swapped sides, and at least static-vs-static plus static-vs-simple/Java baselines. Track win rate, Elo delta, average action time, nodes/sec, average depth, root entropy, and top action type frequency. Add focused parity tests whenever an idea touches native combat, movement, city economy, or action generation.

**Assumptions**
No external bot protocol change is needed. The first implementation batch should prioritize ideas 1, 2, 4, 6, and 13 because they target clear source-level weaknesses and likely give the highest performance-per-risk.
