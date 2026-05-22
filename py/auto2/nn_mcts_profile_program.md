# AutoResearch: NN-MCTS Profiling Optimization

This program is an autonomous local experiment loop for improving an NN-MCTS search pipeline.

The goal is to improve profiling metrics for NN-MCTS throughput and search mechanics while preserving correctness.

The loop is:

```text
profile → optimize throughput/search mechanics → smoke test → keep/reset
```

This program does **not** evaluate playing strength. It does **not** run tournaments. It does **not** optimize ratings. It only uses profiling and correctness checks.

---

## High-Level Goal

Improve useful NN-MCTS search throughput under the existing profiling harness.

Primary optimization targets include:

- more evaluated positions per second
- more selected paths / simulations per second
- better NN inference batching
- lower observation-encoding overhead
- lower CPU↔GPU transfer overhead
- lower output-postprocessing overhead
- better cache hit rate where valid
- deeper or equally deep search at lower wall-clock cost
- no regression in native-MCTS correctness or protocol behavior

This is a **search-system optimization loop**, not a model-training loop and not a game-strength loop.

---

## Core Principle

Each experiment should make a clear, motivated change to the NN-MCTS/search/profiling path.

Prefer isolated changes because profiling attribution matters. Larger changes are allowed only when they are clearly explained and still testable.

Keep changes that improve profiler metrics while preserving smoke-test correctness.

Discard changes that improve one metric by damaging the search contract, causing crashes, invalid payloads, broken batching, or suspiciously collapsed search behavior.

---

## Git Setup

Branches and tags are local Git features. GitHub is not required.

If this directory is not already a Git repo:

```bash
git init
git add .
git commit -m "baseline"
```

Create a branch for this profiling run:

```bash
git checkout -b autoresearch/nn-mcts-profile-<tag>
```

Confirm the working tree is clean:

```bash
git status --short
```

Create an untracked results file:

```powershell
"commit`tstatus`teval_pos_per_sec`tselected_paths_per_sec`tavg_eval_batch`tcache_hit_rate`tavg_depth`ttotal_seconds`tdescription" | Set-Content -LiteralPath nn_mcts_profile_results.tsv
```

Do **not** commit `nn_mcts_profile_results.tsv`.

---

## In-Scope Files

Primary editable files are files directly involved in NN-MCTS search, encoding, model inference, profiling, and bot execution.

Likely files include:

```text
py/search/native/mcts.py
py/profiling/mcts_search.py
py/profiling/selfplay_mcts_nn.py
py/profiling/config.py
py/profiling/configs/mcts_search.json
py/profiling/configs/selfplay_mcts_nn.json
py/search/config.py
py/nn/encoding.py
py/nn/model.py
py/training/selfplay.py
py/bots/hybrid_nn_bot.py
```

The exact filenames may differ in this repository. If a listed file does not exist, locate the corresponding file by searching for:

```text
run_native_mcts
_evaluate_messages
encode_observation
HybridPolicyValueNet
wall_clock
profiling.mcts_search
profiling.selfplay_mcts_nn
search_batch_size
```

Secondary editable files, only when required for tests or profiling:

```text
py/tests/test_native_mcts.py
py/tests/test_*.py
```

Do not edit tests merely to hide failures. Only update tests when behavior intentionally changes and the old assertion is clearly obsolete.

---

## Out-of-Scope Files

Do not modify files unrelated to NN-MCTS search/profiling.

Do not modify:

- game rules unless a correctness test proves the current rule implementation is wrong
- data generation pipelines
- training loops
- model checkpoints
- unrelated static-evaluation code
- unrelated tournament/evaluation scripts

This program is only about NN-MCTS profiling and search-mechanics optimization.

---

## Forbidden Optimization Targets

Do not optimize this loop by:

- changing game rules
- weakening native transition/parity checks
- disabling important validation
- reducing legal action sets incorrectly
- hardcoding specific profiler seeds or payloads
- detecting the profiling script and taking special shortcuts
- skipping NN evaluation while pretending it happened
- replacing search with a trivial fixed policy
- changing metrics in the profiler to make results look better
- deleting expensive but semantically required work without proving equivalence
- lowering the profiling workload without recording the change

---

## Setup Reading

Before the first experiment, read the relevant code paths:

1. NN-MCTS entry point
2. native MCTS wrapper
3. `_evaluate_messages` or equivalent NN leaf evaluator
4. observation encoding path
5. model forward call
6. postprocessing/action-prior conversion
7. profiling harness
8. existing native-MCTS tests

Specifically understand these stages:

```text
payload/messages
→ optional belief annotation
→ encode_observation
→ stack batch
→ transfer to device
→ model forward
→ softmax/postprocess
→ MCTS expansion/backup
```

---

## Baseline First

The first run must establish the profiling baseline.

Run the smoke tests:

```powershell
$env:PYTHONPATH = "$PWD\py"
python -m pytest py/tests/test_native_mcts.py *> test.log
```

If the native extension must be built first, use the repository’s normal build command.

Then run the profiling harness using a stable config. Use the repository’s existing profiling command if documented. If unsure, use a command shaped like:

```powershell
$env:PYTHONPATH = "$PWD\py"
python -m profiling.mcts_search `
  --config py/profiling/configs/mcts_search.json `
  --evaluator nn `
  --positions 16 `
  --simulations 256 `
  --batch-size 32 `
  --top-k-actions 32 `
  --device cuda `
  --no-dirichlet `
  *> profile.log
```

If the local profiler uses different flags, inspect the parser and use the equivalent options.

Record the baseline metrics in `nn_mcts_profile_results.tsv` with status `baseline`.

---

## Required Metrics

Extract as many of the following metrics as the profiler provides:

```text
elapsed_sec
simulations
selected_paths
expanded_nodes
eval_batches
eval_positions
eval_cache_hits
eval_cache_size
average_depth
max_depth
root action counts
eval action counts
root visit entropy
timing: belief_annotate
timing: encode_observation
timing: stack_encoded_batch
timing: transfer_batch
timing: model_forward
timing: postprocess_output
timing: nn_eval.total
```

Derived metrics:

```text
eval_pos_per_sec = eval_positions / elapsed_sec
selected_paths_per_sec = selected_paths / elapsed_sec
avg_eval_batch = eval_positions / max(1, eval_batches)
cache_hit_rate = eval_cache_hits / max(1, eval_cache_hits + eval_positions)
```

If a metric is unavailable, write `NA`.

---

## Keep / Discard Rules

A candidate may be kept if:

- smoke tests pass
- no native parity/correctness checks are weakened
- profiler completes successfully
- throughput improves meaningfully, especially:
  - `eval_pos_per_sec`
  - `selected_paths_per_sec`
  - `avg_eval_batch`
  - reduced encode/transfer/postprocess overhead
- average depth does not collapse suspiciously
- root/action distributions do not collapse suspiciously
- code complexity is reasonable

A candidate must be discarded if:

- tests fail
- profiler crashes
- native extension fails to load
- generated payloads become invalid
- unsupported/approximate/invalid native transitions increase unexpectedly
- search degenerates into trivial behavior
- throughput improvement is caused by silently doing less required work
- average depth or explored paths collapse without a good reason
- metric changes are ambiguous and the code became more complex

---

## Suggested Thresholds

These thresholds are guidelines, not hard laws.

A simple change can be kept if it gives:

```text
>= 5% improvement in eval_pos_per_sec or selected_paths_per_sec
```

A more complex change should usually need:

```text
>= 10–15% improvement
```

A candidate with flat throughput can still be kept if it significantly simplifies code without changing behavior.

A candidate with faster throughput but much worse search depth/action coverage should be discarded.

---

## Experiment Loop

For each experiment:

1. Confirm clean branch:

```bash
git status --short
```

2. Choose one profiling/search-mechanics idea.

Good experiment areas:

```text
reduce encode_observation overhead
cache reusable action masks
cache stable encoded components
improve _stack_encoded batching
avoid repeated CPU tensor construction
reduce CPU↔GPU transfers
reduce unnecessary torch.cuda.synchronize calls
improve eval batch sizes
improve search_batch_size usage
batch across roots / games / leaf requests
avoid repeated belief annotation when safe
optimize postprocess softmax/masking
reduce JSON/message copying
improve eval cache key use
tune top_k_actions / max_depth / c_puct for throughput stability
```

3. Edit only in-scope files.

4. Run smoke tests:

```powershell
$env:PYTHONPATH = "$PWD\py"
python -m pytest py/tests/test_native_mcts.py *> test.log
```

5. If tests fail:
   - fix obvious syntax/compile errors
   - otherwise discard the experiment

6. Commit the candidate before profiling:

```bash
git add <changed-files>
git commit -m "try: <short profiling experiment description>"
```

7. Run the profiler:

```powershell
$env:PYTHONPATH = "$PWD\py"
python -m profiling.mcts_search `
  --config py/profiling/configs/mcts_search.json `
  --evaluator nn `
  --positions 16 `
  --simulations 256 `
  --batch-size 32 `
  --top-k-actions 32 `
  --device cuda `
  --no-dirichlet `
  *> profile.log
```

Use the local equivalent if the profiler has different flags.

8. Parse `profile.log`.

9. Append a row to `nn_mcts_profile_results.tsv`.

10. Decision:
    - if improved and safe: keep the commit
    - if failed, noisy, unsafe, or worse: reset

Discard command:

```bash
git reset --hard HEAD~1
```

---

## Logging Format

Append one row per experiment:

```tsv
commit	status	eval_pos_per_sec	selected_paths_per_sec	avg_eval_batch	cache_hit_rate	avg_depth	total_seconds	description
```

Example:

```tsv
a1b2c3d	baseline	420.5	390.2	18.7	0.12	7.4	61.3	baseline
b2c3d4e	keep	486.1	450.9	25.2	0.14	7.3	60.8	cache action-count masks in postprocess
c3d4e5f	discard	510.0	130.4	27.1	0.15	2.1	60.4	pruned too aggressively and collapsed search depth
```

---

## Recommended First Experiments

Start with low-risk profiling improvements:

1. Reduce repeated allocation in action mask construction.
2. Cache masks keyed by action counts and device.
3. Avoid repeated tensor creation inside postprocessing.
4. Improve `_stack_encoded` efficiency.
5. Batch more leaf evaluations together.
6. Increase average eval batch size without hurting depth.
7. Reduce repeated `encode_observation` work.
8. Avoid repeated belief annotation where a snapshot can be reused.
9. Reduce `.cpu().tolist()` hot-path overhead if possible.
10. Tune `search_batch_size` for actual GPU utilization.

Do **not** start by changing model architecture.

Do **not** start by changing training.

Do **not** start by adding a game-strength evaluation step.

---

## Profiling Interpretation

If `model_forward` is small but `encode_observation` and `postprocess_output` dominate, optimize CPU-side preprocessing/postprocessing.

If `transfer_batch` dominates, reduce transfer frequency, improve batching, or keep tensors on device when safe.

If `avg_eval_batch` is low, improve batching strategy before changing the model.

If `cache_hit_rate` is low but many repeated states are expected, inspect cache keys and message normalization.

If average depth is low but throughput is high, the search may be shallow or over-pruned.

If root visit entropy collapses after a change, inspect whether priors or legal action filtering became too sharp.

---

## What Not To Do

Do not make the profiler faster by:

- lowering simulations/cases without recording it
- editing the profiler metrics
- hiding expensive operations outside timed regions
- skipping model forward
- replacing values/priors with constants
- pruning legal actions incorrectly
- silently disabling belief features
- disabling native transition checks
- ignoring failed leaf encodings

---

## End Condition

Continue until manually stopped.

At the end, summarize:

```text
latest commit
number of experiments tried
kept / discarded / crashed counts
best throughput improvement
main bottleneck before
main bottleneck after
remaining bottlenecks
recommended next experiments
```

Use `nn_mcts_profile_results.tsv`, Git history, and profiler logs.
