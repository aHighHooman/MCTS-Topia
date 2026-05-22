# AutoResearch: Self-Play Sims/Sec

Goal: improve full Java self-play NN-MCTS throughput, measured by `profiling.selfplay_mcts_nn`.

Primary metric: `simulations_per_sec`.

Safety metrics: `replay_steps_per_sec`, action latency, eval batch size, depth, root entropy, top visit share, fallback count, native transition counters, and replay validity.

Loop:

```text
baseline -> one change -> smoke test -> self-play profile -> keep/revert
```

Do not optimize ratings, tournaments, training loss, or standalone `mcts_search` throughput unless the change also improves full self-play.

---

## Scope

Likely files:

```text
py/profiling/selfplay_mcts_nn.py
py/profiling/configs/selfplay_mcts_nn.json
py/training/selfplay.py
py/training/java_selfplay.py
py/training/train.py
py/bots/hybrid_nn_bot.py
py/search/native/mcts.py
py/search/config.py
py/nn/encoding.py
py/nn/model.py
docs/external-bot-protocol.md
src/HeadlessPlay.java
src/core/game/ExternalBot.java
src/core/game/RegressionHarness.java
```

Avoid game-rule changes, checkpoint changes, training objective changes, and unrelated bots/scripts.

---

## Baseline

```powershell
$env:PYTHONPATH = "$PWD\py"
python -m pytest py/tests/test_native_mcts.py *> test.log
python -m profiling.selfplay_mcts_nn `
  --config py/profiling/configs/selfplay_mcts_nn.json `
  *> selfplay_profile.log
```

If Java/protocol code changed, also run the Java regression harness from `AGENTS.md`.

Create an untracked results file:

```powershell
"commit`tstatus`tsims_per_sec`treplay_steps_per_sec`tactions`tsimulations`teval_positions_per_sec`tavg_eval_batch`tavg_depth`troot_visit_entropy`tmean_action_ms`tp95_action_ms`ttotal_seconds`tdescription" | Set-Content -LiteralPath selfplay_profile_results.tsv
```

---

## Metrics

Track:

```text
simulations_per_sec
replay_steps_per_sec
actions_per_sec
choose_action total/encode/policy/search ms
mcts select/eval/expand/inner ms
eval_positions_per_sec
avg_eval_batch
eval_cache_hit_rate
avg_depth / max_selected_depth
root_visit_entropy / top_visit_share
root_actions / searched_actions
fallback_count
native_unsupported_transition
native_invalid_transition
native_approximate_transition
```

Write `NA` for unavailable fields.

---

## Experiment

For each candidate:

1. Make one focused throughput change.
2. Run smoke tests.
3. Commit the candidate.
4. Run `profiling.selfplay_mcts_nn`.
5. Append a row to `selfplay_profile_results.tsv`.
6. Keep only if faster and search/replay behavior still looks valid.

Good first areas:

```text
persistent bot startup/reuse
external bot JSON/protocol copying
choose_action encoding overhead
leaf eval batching
search_batch_size GPU utilization
CPU tensor allocation and CPU/GPU transfer
action-mask/postprocess allocation
encoded-observation buffer reuse
replay serialization
native MCTS select/expand overhead
hot-path logging or cuda synchronization
```

Do not start with model architecture or training changes.

---

## Keep / Revert

Keep simple changes around `>= 5%` better `simulations_per_sec`; require more for complex changes.

Revert if tests fail, profiler fails, invalid/fallback actions increase, replay breaks, native transition counters increase, depth/entropy/action counts collapse, or the speedup comes from doing less required work.

Prefer:

```bash
git revert <commit>
```

Do not use `git reset --hard` in a dirty workspace.

---

## Results Row

```tsv
commit	status	sims_per_sec	replay_steps_per_sec	actions	simulations	eval_positions_per_sec	avg_eval_batch	avg_depth	root_visit_entropy	mean_action_ms	p95_action_ms	total_seconds	description
```

---

## Final Summary

Report:

```text
latest commit
experiments tried
kept / reverted / crashed
best sims/sec improvement
best replay steps/sec improvement
main bottleneck before/after
remaining bottlenecks
next experiments
```
