# AutoResearch: NN-MCTS Search Sims/Sec

Goal: improve NN-MCTS search throughput, measured by `profiling.mcts_search`.

Primary metric: `simulations_per_sec` / selected paths per second under the profiler workload.

Safety metrics: eval positions/sec, eval batch size, cache hit rate, depth, root visit entropy, action breadth, top visit share, and native transition counters.

Before choosing experiments, read `py/auto2/nn_mcts_optimization_report.md` and build on the findings already recorded there.

Loop:

```text
baseline -> one change -> smoke test -> MCTS profile -> keep/revert
```

---

## Scope

Likely files:

```text
py/profiling/mcts_search.py
py/profiling/configs/mcts_search.json
py/search/native/mcts.py
py/search/native/static_mcts.py
py/search/config.py
py/nn/encoding.py
py/nn/model.py
py/bots/hybrid_nn_bot.py
py/training/selfplay.py
```

Avoid game-rule changes, checkpoint changes, training objective changes, and unrelated bots/scripts.

---

## Baseline

Default `profiling.mcts_search` mode is wall-clock: if `--simulations` is omitted, each position runs for `--wall-time-sec`.

Do not modify `py/profiling/configs/mcts_search.json` when running profiler experiments; keep the config fixed and pass per-run overrides on the command line.

```powershell
$env:PYTHONPATH = "$PWD\py"
python -m pytest py/tests/test_native_mcts.py *> test.log
python -m profiling.mcts_search `
  --config py/profiling/configs/mcts_search.json `
  --evaluator nn `
  *> profile.log
```

For fixed-simulation profiling, pass `--simulations` explicitly:

```powershell
$env:PYTHONPATH = "$PWD\py"
python -m profiling.mcts_search `
  --config py/profiling/configs/mcts_search.json `
  *> profile.log
```

Create an untracked results file:

```powershell
"commit`tstatus`tsims_per_sec`teval_pos_per_sec`tselected_paths_per_sec`tavg_eval_batch`tcache_hit_rate`tavg_depth`troot_visit_entropy`ttotal_seconds`tdescription" | Set-Content -LiteralPath nn_mcts_profile_results.tsv
```

---

## Metrics

Track:

```text
simulations_per_sec
selected_paths_per_sec
nodes_per_sec
nodes_explored_per_sec
eval_positions_per_sec
avg_eval_batch
eval_cache_hit_rate
avg_depth / max_depth
root legal action counts
evaluated leaf action counts
root visit entropy
action type breadth
top root actions
nn_eval encode/stack/transfer/model_forward/postprocess timings
native_tree select/expand/complete timings
```

Write `NA` for unavailable fields.

---

## Experiment

For each candidate:

1. Make one focused search/profiling change.
2. Run smoke tests.
3. Commit the candidate.
4. Run `profiling.mcts_search` with the same workload as baseline.
5. Append a row to `nn_mcts_profile_results.tsv`.
6. Keep only if faster and search shape still looks valid.

Good first areas:

```text
encode_observation overhead
encoded batch stacking
leaf eval batching
search_batch_size GPU utilization
CPU tensor allocation and CPU/GPU transfer
action-mask/postprocess allocation
eval cache key use
belief annotation reuse
native MCTS select/expand overhead
top_k_actions / max_depth / c_puct throughput stability
hot-path logging or cuda synchronization
```

Do not start with model architecture or training changes.

---

## Keep / Revert

Keep simple changes around `>= 5%` better `simulations_per_sec`, `selected_paths_per_sec`, or `eval_positions_per_sec`; require more for complex changes.

Revert if tests fail, profiler fails, native transition counters increase, depth/action breadth/entropy collapse, legal action filtering changes incorrectly, or the speedup comes from doing less required work.

Prefer:

```bash
git revert <commit>
```

Do not use `git reset --hard` in a dirty workspace.

---

## Results Row

```tsv
commit	status	sims_per_sec	eval_pos_per_sec	selected_paths_per_sec	avg_eval_batch	cache_hit_rate	avg_depth	root_visit_entropy	total_seconds	description
```

---

## Final Summary

Report:

```text
latest commit
experiments tried
kept / reverted / crashed
best sims/sec improvement
best eval positions/sec improvement
main bottleneck before/after
remaining bottlenecks
next experiments
```
