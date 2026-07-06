# Experiment Record

This document records the experiments visible in the checked-in configs, tests, and local `debug-logs/` artifacts. It is meant to prevent repeating old loops without a reason.

Local generated logs are evidence, not a benchmark standard. Re-run before making final strength claims.

## Saved Position Corpus

Source files:

- Config: `py/profiling/configs/generate_mcts_profile_positions.json`
- Summary: `debug-logs/mcts-profile-payloads/mcts_profile_position_selection_summary.json`

Recorded corpus:

- `generated_payloads`: 36
- payload directory contained 38 files during value-breakdown runs; 2 were skipped in those runs.
- Game setup in config:
  - `PlayLG`
  - `Capitals`
  - `Drylands`
  - `Tiny`
  - tribes `Xin Xi` vs `Imperius`
  - 20 source games
  - 36 selected positions
  - max turns 40
- Selection intentionally favors interesting positions:
  - high branching,
  - combat,
  - economy choices,
  - expansion,
  - enemy visible,
  - damaged units,
  - partial information,
  - action diversity.

The saved corpus is useful for profiler and static-eval inspection, but it is small and biased toward hard/action-rich positions.

## Branch Comparisons

Source:

- Config: `py/profiling/configs/compare_branches.json`
- Tool: `py/profiling/analysis/compare_branches.py`
- Summaries under `debug-logs/analysis/branch-compare/`

Both visible runs compare `baseline` vs `experimental` on 36 positions with static executable search.

| Run | Positions | Avg JS visit bits | Selected same rate |
| --- | ---: | ---: | ---: |
| `20260617-135216` | 36 | 0.04295 | 83.3% |
| `20260617-135631` | 36 | 0.11984 | 66.7% |

Interpretation:

- Baseline and experimental often select the same action, but not always.
- The second run shows larger distribution movement than the first.
- These are root-distribution comparisons, not game-outcome validation.

## Value Breakdown Runs

Source:

- Config: `py/profiling/configs/value_breakdown.json`
- Tool: `py/profiling/analysis/value_breakdown.py`
- Summaries under `debug-logs/analysis/value-breakdown/`

Visible runs:

| Run | Target | Positions | Terms | Notes |
| --- | --- | ---: | ---: | --- |
| `Baseline` | baseline | 36 | 2736 | Older/simple baseline summary. |
| `20260619-215350` | experimental | 36 | 7488 | Complete, 38 payload files found, 2 skipped. |
| `20260619-220915` | experimental | 36 | 7488 | Complete, same corpus shape. |

The experimental runs report:

- 2448 aggregate terms,
- 5040 component terms,
- 1440 diagnostic terms,
- validation status complete,
- term counts matched summary metadata.

Interpretation:

- The breakdown tooling is mature enough to inspect term contribution shape.
- The reports are raw-space accounting. They should not be read as counterfactual ablations.

## Static Value Result Tuning

Source:

- Configs:
  - `py/profiling/configs/tune_static_value_from_results.json`
  - `py/profiling/configs/tune_static_value_experimental_training.json`
- Tools:
  - `tools/static_value_result_driver.cpp`
  - `tools/static_value_result_tuner.cpp`
  - `py/profiling/analysis/tune_static_eval_weights.py`
  - `py/profiling/analysis/temp_variant_optimizer.py`
- Summaries under `debug-logs/analysis/static-value-result-tuning/`

The tuning family tries to fit static-eval weights from game/result records or action-ranking examples, then optionally validates candidate overrides in Java tournaments.

### Notable Visible Outcomes

| Run | Shape | Validation |
| --- | --- | --- |
| `loop-5x25-hybrid-reuse-iter1c` | 5 iterations, final override emitted | Final iteration rejected: 30.0% score rate over 40 games. Iterations were 42.5%, 32.5%, 20.0%, 25.0%, 30.0%. |
| `loop-5x50-all-terms` | 5 iterations, broader terms | Rejected: final 22.5% over 40 games. Iterations stayed below acceptance. |
| `loop-5x25e` | 5 iterations | Rejected: final 27.5% over 40 games. |
| `loop-5x25c` | individual iteration summaries | Iteration 1 accepted at 55.0% over 40 games; iteration 2 accepted at 62.5% over 40 games. Needs rerun/confirmation before treating as stable. |
| `loop-5x25d` | individual iteration summaries | Iteration 1 accepted at 65.0% over 40 games; iteration 2 had no visible validation score in the compact extraction. |
| `allsym-val20` | 180034 examples | Rejected: 45.0% over 40 games; sign accuracy around 51.5%. |
| `experimental-training-500-random-native-all-terms-loose-sign-validation-20-balanced` | 180034 examples | Rejected: 10.0% over 40 games; sign accuracy around 51.7%. |
| smoke validations | tiny runs | Useful only for pipeline health, not strength. |

### Lessons From The Visible Tuning Runs

- Small/medium candidate loops can produce one-off accepted 40-game validations, but later or broader loops often regress badly.
- All-terms or symmetric/global fits with around 180k examples show weak sign accuracy near 51% and poor validation outcomes.
- Training loss or sign accuracy alone is not a reliable strength proxy.
- A candidate should not be adopted because it fits records; it needs balanced tournament validation and ideally a second seed set.
- The latest visible loops around 2026-07-03 lean negative for the broad training-formula direction.

## Branchpoint and Counterfactual Tooling

The branchpoint builder is configured to look for positions where:

- baseline has a confident top action;
- baseline and experimental disagree enough;
- baseline's action ranks poorly under experimental or gets low visit share;
- optional counterfactual rescue metadata can be required.

Defaults:

- minimum JS visit bits: `0.08`
- minimum baseline top visit share: `0.35`
- minimum top gap: `0.05`
- max experimental visit share for baseline action: `0.10`
- minimum baseline-action rank under experimental: `4`

Counterfactual tooling exists to replay a save snapshot with one forced root action, but no completed counterfactual summaries were visible in the top-level analysis folders inspected.

## Turn Macro Search Experiments

Source:

- Search implementation: `py/search/native/src/turn_macro_exp.cpp`
- Static bot wiring: `bots/static_mcts_bot.cpp`
- Profiler configs:
  - `py/profiling/configs/mcts_search_primitive_baseline.json`
  - `py/profiling/configs/mcts_search_turn_macro_exp.json`
- Tournament logs:
  - `debug-logs/headtohead_wallclock_short/`
  - `debug-logs/headtohead_wallclock_patch_smoke/`

### Current Shape

`turn-macro-exp` runs one inner primitive MCTS at a turn node, extracts up to 4 complete first-action-diverse turn plans, and lets the outer tree search over those turn edges. It no longer uses a configured max turn-depth cap. If `max_primitives_per_turn` is reached while still on the same player's turn, the next primitive is forced to `END_TURN`.

### Full Profiler Snapshot

The 36-position static-exe profiler used 3 seconds per position:

| Mode | Positions | Work | Rate | Avg turn depth | Max turn depth | Root searched |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| primitive | 36 | 1,355,136 sims | 10,940 sims/s | 0.598 | 7 | 31.78 |
| turn-macro-exp | 36 | 607,006,336 outer paths | 5,454,976 paths/s | 1.015 | 10 | 4.00 |

Interpretation:

- The raw macro `paths/s` is inflated because many outer iterations repeatedly select existing macro edges.
- The useful improvement was turn-depth: avg `0.598 -> 1.015`, max `7 -> 10`.
- Macro expanded only 239 outer nodes but ran 242 inner searches and 247,808 inner simulations.

### Tournament/Wall-Clock Findings

The correct head-to-head harness is Java `Tournament` with `--wall-clock-per-action-seconds`, not fixed simulation counts.

Visible tournament runs:

| Run | Shape | Result |
| --- | --- | --- |
| `headtohead_wallclock_short` | 4 seeds, balanced seats, 8 games, `Parallel Games: 8`, 1s/action, turn limit 8 | Completed in 256.09s. Primitive 5/8, macro 3/8. External stderr empty. |
| `headtohead_wallclock_patch_smoke` | 1 seed, balanced seats, 2 games, `Parallel Games: 8`, 1s/action, turn limit 4 | Completed in 45.36s. Primitive 2/2, macro 0/2. External stderr empty. |

An attempted 4-seed balanced run with turn limit 20 was actively processing actions but had no completed games after several minutes; one game was already past 100 action requests. That was a throughput problem, not an invalid-action crash.

### 2026-07-05 Wall-Clock Chunk Fix

Before the fix, `choose_action_with_turn_macro_exp_tree` used `tree.run(batch_size)` inside the wall-clock loop. With `--search-batch-size 64`, one time check could include many outer simulations, and macro simulations can trigger expensive 1024-sim inner searches. That made a nominal 1 second action budget overshoot in tournament play.

Patch:

- Fixed-simulation mode is unchanged.
- Macro wall-clock mode now calls `tree.run(1)` per clock check.

Validation:

- `.\scripts\build_static_bot.ps1`
- `python -m pytest py/tests/test_static_mcts_exe.py::test_static_mcts_exe_turn_macro_exp_returns_legal_root_action py/tests/test_static_mcts_exe.py::test_static_mcts_exe_turn_macro_exp_wall_clock_returns_legal_root_action py/tests/test_mcts_search_profiler.py::test_turn_macro_exp_config_and_static_exe_command`
- `debug-logs/headtohead_wallclock_patch_smoke/tournament_1s_parallel8_turn4.json`

Open direction:

- This fix improves wall-clock compliance but does not solve macro throughput. The expensive part remains the inner searches plus greedy plan completion. Next useful measurements should separate inner-search time, greedy completion/static-eval time, and number of generated turn plans per real action in tournament games.

### 2026-07-05 Inner-Search Safety/Throughput Pass

Changes tested:

- Added macro profile counters:
  - `static_eval_calls`
  - `greedy_static_calls`
  - `greedy_static_candidates_considered`
  - `greedy_static_child_evals`
  - `greedy_static_child_eval_skips`
  - `greedy_static_child_eval_limit`
- Greedy plan completion now child-evaluates only the top prior-ranked child by default (`turn_macro_greedy_eval_top_k = 1`), while still considering skipped actions with base-state value plus prior.
- `turn-macro-exp` now runs one inner MCTS per generated turn node and completes plans from that one root candidate set. It no longer runs a new inner MCTS at every primitive prefix inside the same turn plan.
- Fixed a continuation regression:
  - `END_TURN` is applied, not just recorded.
  - full `result_state` is kept for child turn nodes.
  - if no more unique macro plans can be generated, the node is marked expansion-exhausted and existing edges are reused instead of stalling.
- Forced/tactical actions stop greedy continuation and fall through to forced `END_TURN`; this avoids unstable chains such as resource -> level-up -> move -> speculative post-move child evaluation.
- Opponent turn nodes currently use a single forced/pass-style plan, preferring `END_TURN`, instead of running inner MCTS on opponent states.
- Wall-clock macro search now checks time every outer sim and also caps wall-clock actions at 16 outer sims. This is a temporary stability guard; deeper wall-clock traversal still exposes native crashes.

Unsafe findings:

- `turn_macro_inner_simulations = 1024` can crash tournament replay payloads with native access violation `-1073741819`.
- The same game-2 replay stayed clean at inner sims 32, 64, and 128 with greedy top-k 1, but crashed at 256+.
- A 3s/profile position wall-clock run still reached unsafe deep traversal before the 16-outer-sim wall-clock ceiling was added.

Stable profiler snapshot after the safety pass:

| Config | Positions | Outer paths | Inner sims | Static eval calls | Greedy child evals/skips | Avg turn depth | Max turn depth | Root searched |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `turn-macro-exp`, 1s/position, inner=128, greedy top-k=1, wall-clock outer cap=16 | 36 | 560 | 5,248 | 1,794 | 231 / 8,928 | 1.70 | 2 | 3.75 |

Timing:

- Total profiler wall time: 2.69s for 36 positions.
- Inner search: 464ms total, 17.2%.
- Static eval: 37.7ms total, 1.4%.
- Process overhead dominates this small capped run, so this profile is useful for stability/counter metrics, not raw strength.

Tournament smoke:

| Run | Shape | Result |
| --- | --- | --- |
| `headtohead_wallclock_safe128_smoke` | 1 seed, balanced seats, 2 games, `Parallel Games: 8`, 1s/action, turn limit 4, inner=128, wall-clock outer cap=16 | Completed. Primitive 2/2, macro 0/2. External stderr empty. |

Interpretation:

- The current safe version is much cheaper and stable under the small smoke, but it gives back a lot of the intended depth: max turn depth is only 2 under the capped profiler.
- The next real search-quality direction is not more greedy child evals; those were mostly skipped and can trigger unsafe speculative transitions.
- The next useful engineering direction is to isolate the native crash in deeper turn-node/inner-MCTS traversal, then remove or raise the 16-outer-sim wall-clock cap.

## What Seems Worth Avoiding

- Re-running broad all-term static result fitting without a new hypothesis. The visible large-example runs did not validate well.
- Treating `experimental-training` learned weights as automatically better because they expose more features.
- Judging candidates only from branch-distribution movement on the 36-position corpus.
- Reading value-breakdown `linearized_value` as an ablation.
- Treating macro outer `paths/s` as equivalent to primitive expanded nodes/s. Macro `paths/s` can be mostly repeated traversal over already-generated macro edges.

## What Seems Worth Keeping

- The 36-position corpus is useful for quick profiling and qualitative inspection.
- Branch comparison is useful for finding disagreements, not for proving strength.
- Value breakdown is useful for explaining why a candidate likes an action.
- Pairwise/root-child tools are useful when a specific bad choice needs diagnosis.
- Validation must remain the final filter for static-eval weight changes.
- Java `Tournament` with wall-clock static bot commands is the right harness for bot-vs-bot failure checks.
