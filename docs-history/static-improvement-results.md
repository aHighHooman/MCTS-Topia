MCTS Static Improvement Results
================================

Source of truth: `StaticImprovementPlan.md`.

Correctness checks used:
- `python -m pytest py/tests/test_native_mcts.py -q`
- `python -m pytest py/tests -q`
- Java `core.game.RegressionHarness`

Performance smoke used for scoped experiments:
- 4 deterministic fixture payloads, 128 simulations each, batch size 64, top-k 64, repeated across fixed seeds.
- Metric: fixture simulations/sec and selected action stability.

Results
-------

1. Static unit stats from parsed fields: kept. Parsed attack/defence/range/movement now override type fallback tables where present. Focused regression added. Native tests passed. Smoke throughput was acceptable during the isolated run.
2. Exact combat forecast: kept. Attack scoring now uses a local forecast matching the native transition combat formula, including HP-scaled attack/defence, retaliation range, stiff units, and dagger/pirate no-retaliation. Native tests passed. Isolated smoke improved over the preceding kept run.
3. Same-turn tactical pair scoring: reverted. Move/attack/infiltrate follow-up bonuses passed tests but reduced smoke throughput.
4. Marginal economy scoring: reverted. Population/economy scoring changes passed tests but reduced smoke throughput.
5. Immediate-unlock research scoring: reverted. Tech affordability/Philosophy adjustments passed tests but reduced smoke throughput.
6. Child action static-prior sorting: reverted. Non-root prior sorting passed tests but sorting overhead reduced smoke throughput.
7. Quota-based root pruning: reverted. Narrowing root always-keep actions passed tests but did not show stable throughput improvement.
8. Dynamic `END_TURN`: reverted. Legal-action-aware end-turn scoring passed tests but reduced smoke throughput on repeat.
9. Mode/stage-aware value: reverted. Mode-aware value adjustments passed tests but reduced smoke throughput on repeat.
10. Exact threat maps: reverted. A scoped status-aware threat experiment passed tests but reduced smoke throughput; full legal-reach threat maps require a larger movement/action-generation subsystem.
11. Turn-completion evaluator: no code kept. Requires a greedy rollout planner over atomic actions; not safe to add without broader parity coverage and performance budgeting.
12. Macro-action MCTS: no code kept. Requires a macro generator and first-action projection layer; too large for an isolated static-eval experiment.
13. Transposition/canonical state reuse: no code kept. Requires canonical native state hashing and child-node reuse semantics; not safe without extensive collision/parity tests.
14. Progressive bias/RAVE updates: no code kept. Requires new backup statistics and action feature aggregation; not isolated.
15. Tactical MCTS plus economic beam search: no code kept. Requires a second planner and arbitration between tactical/economic policies.
16. Learned static weights: no code kept. Requires tournament/self-play data generation and an optimizer loop, not a code-only local experiment.
17. Belief-aware static eval: no code kept. Requires integrating belief-state estimates into native static evaluation.
18. Adaptive search allocation: no code kept. Requires budget controller changes and tournament validation across phases.
19. Incremental static eval internals: no code kept. Requires cached per-node feature maps and invalidation rules; not safe as a small patch.
20. Opponent-response scripts at leaf value: no code kept. Requires scripted enemy rollout and clear value-perspective handling.

Final kept changes are limited to ideas 1 and 2.
