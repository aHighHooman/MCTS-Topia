# AutoResearch: Self-Play Sims/Sec Optimization

This program is an autonomous local experiment loop for improving NN-MCTS throughput during full Java self-play games.

The goal is to increase useful simulations per second in the real self-play path while preserving correctness, valid game behavior, and replay generation.

The loop is:

```text
self-play profile -> optimize self-play/search throughput -> smoke test -> keep/reset
```

This program does **not** optimize rating, tournament strength, or training loss. It uses self-play profiling as the primary signal, because the target is end-to-end sims/sec during data generation.

---

## High-Level Goal

Improve useful NN-MCTS self-play throughput under `profiling.selfplay_mcts_nn`.

Primary optimization targets include:

- higher full-run `simulations_per_sec`
- lower `choose_action.total_ms`
- lower MCTS `total_inner_ms` per path
- lower `choose_action.encode_ms`
- lower `choose_action.policy_ms`
- lower `choose_action.search_ms`
- better eval batching during actual games
- less Java/Python process, protocol, and JSON overhead
- lower persistent-bot startup and warmup overhead
- more replay steps per wall-clock second
- no regression in native-MCTS correctness, legal actions, protocol behavior, or replay validity

This is a **self-play throughput optimization loop**, not a standalone profiler loop and not a game-strength evaluation loop.

---

## Core Principle

Each experiment should make a clear, motivated change to the real self-play path.

Prefer isolated changes because profiling attribution matters. Larger changes are allowed only when they are clearly explained and still testable.

Keep changes that improve self-play sims/sec while preserving smoke-test correctness and normal-looking game behavior.

Discard changes that improve a metric by damaging the game contract, causing invalid actions, hiding failed searches, skipping required NN evaluation, breaking replay output, or making self-play suspiciously shallow.

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
git checkout -b autoresearch/selfplay-profile-<tag>
```

Confirm the working tree is clean:

```bash
git status --short
```

Create an untracked results file:

```powershell
"commit`tstatus`tsims_per_sec`treplay_steps_per_sec`tactions`tsimulations`teval_positions_per_sec`tavg_eval_batch`tmean_action_ms`tp95_action_ms`ttotal_seconds`tdescription" | Set-Content -LiteralPath selfplay_profile_results.tsv
```

Do **not** commit `selfplay_profile_results.tsv`.

---

## In-Scope Files

Primary editable files are files directly involved in self-play orchestration, NN-MCTS bot execution, profiling, encoding, model inference, native search, and protocol overhead.

Likely files include:

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

The exact filenames may differ. If a listed file does not exist, locate the corresponding code by searching for:

```text
profile_selfplay
run_selfplay_match
HybridPolicyValueNet
hybrid_nn_bot
choose_action
mcts_nn.profile
mcts_nn.search_profile
persistent_bot
_persistent_bridge_command
run_native_mcts
encode_observation
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

Do not modify files unrelated to NN-MCTS self-play throughput.

Do not modify:

- game rules unless a correctness test proves the current rule implementation is wrong
- model checkpoints
- training objectives
- tournament/evaluation rating scripts
- unrelated static-evaluation bots
- data formats unrelated to generated self-play replay records

This program is only about making NN-MCTS self-play generate valid search-backed data faster.

---

## Forbidden Optimization Targets

Do not optimize this loop by:

- changing game rules
- weakening native transition/parity checks
- disabling important validation
- reducing legal action sets incorrectly
- hardcoding profiler seeds, maps, or payloads
- detecting the profiling script and taking special shortcuts
- skipping NN evaluation while pretending it happened
- replacing search with a trivial fixed policy
- changing profiler metrics to make results look better
- deleting expensive but semantically required work without proving equivalence
- lowering simulations, max actions, games, turn caps, or map complexity without recording it
- disabling replay writes if replay throughput is part of the measured workload
- hiding invalid actions or fallback actions

---

## Setup Reading

Before the first experiment, read the relevant code paths:

1. `profiling.selfplay_mcts_nn` argument parsing, metrics, CSV/JSON output, and log parsing
2. `training.selfplay` and Java self-play launch behavior
3. persistent bot setup and shutdown
4. `hybrid_nn_bot` action-selection path
5. native MCTS wrapper
6. `_evaluate_messages` or equivalent NN leaf evaluator
7. observation encoding path
8. model forward and policy/value postprocessing
9. replay writing path
10. native-MCTS and protocol tests

Specifically understand these stages:

```text
Java self-play game
-> external bot request
-> Python bot choose_action
-> optional root policy/value call
-> native MCTS leaf requests
-> encode_observation batch
-> transfer to device
-> model forward
-> postprocess priors/values
-> native MCTS expansion/backup
-> action response to Java
-> replay step write
```

---

## Baseline First

The first run must establish the self-play profiling baseline.

Run the smoke tests:

```powershell
$env:PYTHONPATH = "$PWD\py"
python -m pytest py/tests/test_native_mcts.py *> test.log
```

If Java changes are made or Java protocol behavior is suspected, also run the Java regression harness:

```powershell
$sources = Join-Path (Get-Location) 'sources.txt'
Get-ChildItem -Recurse src -Filter *.java | ForEach-Object { $_.FullName } | Set-Content -LiteralPath $sources
if (Test-Path out) { Remove-Item -LiteralPath out -Recurse -Force }
New-Item -ItemType Directory -Force out | Out-Null
& "$env:JAVA_HOME\bin\javac.exe" -cp "lib/json.jar" -d out @$sources
& "$env:JAVA_HOME\bin\java.exe" -cp "out;lib/json.jar" core.game.RegressionHarness
Remove-Item -LiteralPath $sources -Force
```

Then run the self-play profiling harness using a stable config:

```powershell
$env:PYTHONPATH = "$PWD\py"
python -m profiling.selfplay_mcts_nn `
  --config py/profiling/configs/selfplay_mcts_nn.json `
  *> selfplay_profile.log
```

If the local profiler uses different flags, inspect the parser and use the equivalent options. Keep the baseline workload fixed across experiments unless the experiment is explicitly a config-throughput experiment.

Record the baseline metrics in `selfplay_profile_results.tsv` with status `baseline`.

---

## Required Metrics

Extract as many of the following metrics as the profiler provides:

```text
games
elapsed_sec
simulations
simulations_per_sec
configured_simulations
actions
replay_steps
replay_steps_per_sec
fallback_or_invalid_action_count
warmup_ms
mean choose_action.total_ms
median choose_action.total_ms
p95 choose_action.total_ms
mean choose_action.encode_ms
mean choose_action.policy_ms
mean choose_action.search_ms
mean mcts.select_ms
mean mcts.eval_ms
mean mcts.expand_ms
mean mcts.total_inner_ms
eval_batches
eval_positions
average eval batch size
paths
device
persistent_bot
map_type/map_size
max_turns_capitals
max_actions_per_game
```

Derived metrics:

```text
sims_per_sec = simulations / elapsed_sec
replay_steps_per_sec = replay_steps / elapsed_sec
eval_positions_per_sec = eval_positions / elapsed_sec
avg_eval_batch = eval_positions / max(1, eval_batches)
ms_per_sim = total_search_ms / max(1, simulations)
actions_per_sec = actions / elapsed_sec
fallback_rate = fallback_or_invalid_action_count / max(1, actions)
```

If a metric is unavailable, write `NA`.

The primary score is full-run `simulations_per_sec`. Use `replay_steps_per_sec` and action timing as secondary checks because self-play data generation is the actual workload.

---

## Keep / Discard Rules

A candidate may be kept if:

- smoke tests pass
- no native parity/correctness checks are weakened
- self-play profiler completes successfully
- `simulations_per_sec` improves meaningfully
- `replay_steps_per_sec` does not regress without a good explanation
- invalid/fallback action count does not increase
- replay output remains valid
- average eval batch size, action counts, and path counts look plausible
- code complexity is reasonable

A candidate must be discarded if:

- tests fail
- profiler crashes or times out
- native extension fails to load
- Java self-play fails, hangs, or emits invalid bot protocol payloads
- invalid/fallback actions increase unexpectedly
- replay generation breaks
- self-play degenerates into very short or suspicious games
- throughput improvement is caused by silently doing less required work
- search paths, eval positions, or action counts collapse without a clear reason
- metric changes are ambiguous and the code became more complex

---

## Suggested Thresholds

These thresholds are guidelines, not hard laws.

A simple change can be kept if it gives:

```text
>= 5% improvement in full-run simulations_per_sec
```

A more complex change should usually need:

```text
>= 10-15% improvement in simulations_per_sec
```

A candidate with flat sims/sec can still be kept if it significantly improves replay steps/sec, removes process overhead, or simplifies code without changing behavior.

A candidate with faster sims/sec but more invalid actions, broken replay output, or suspiciously reduced search work should be discarded.

---

## Experiment Loop

For each experiment:

1. Confirm clean branch:

```bash
git status --short
```

2. Choose one self-play throughput idea.

Good experiment areas:

```text
reduce external bot protocol JSON copying
improve persistent bot startup and reuse
avoid repeated model/device setup per action
reduce choose_action encode overhead
batch leaf evaluations more effectively during real games
improve search_batch_size usage
avoid repeated CPU tensor construction
reduce CPU/GPU transfers
cache reusable action masks
cache stable encoded components when safe
optimize postprocess softmax/masking
reduce replay serialization overhead
reduce unnecessary logging in hot paths
reduce torch.cuda.synchronize calls
reuse buffers for encoded observations
improve native MCTS path selection/expansion overhead
keep Java/Python bridge alive reliably
tune wall_clock_per_action_seconds only if the resulting work remains comparable
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
git commit -m "try: <short selfplay profiling experiment description>"
```

7. Run the self-play profiler:

```powershell
$env:PYTHONPATH = "$PWD\py"
python -m profiling.selfplay_mcts_nn `
  --config py/profiling/configs/selfplay_mcts_nn.json `
  *> selfplay_profile.log
```

Use the local equivalent if the profiler has different flags.

8. Parse `selfplay_profile.log` and generated profiler CSV/JSON files.

9. Append a row to `selfplay_profile_results.tsv`.

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
commit	status	sims_per_sec	replay_steps_per_sec	actions	simulations	eval_positions_per_sec	avg_eval_batch	mean_action_ms	p95_action_ms	total_seconds	description
```

Example:

```tsv
a1b2c3d	baseline	812.4	5.8	96	49152	522.1	18.7	615.3	890.4	60.5	baseline
b2c3d4e	keep	930.7	6.2	98	50176	641.8	24.3	537.2	781.0	53.9	reuse encoded tensor buffers in selfplay bot
c3d4e5f	discard	1010.2	1.1	18	9216	700.0	25.1	210.4	260.8	9.1	games ended suspiciously early after action filtering change
```

---

## Recommended First Experiments

Start with low-risk self-play throughput improvements:

1. Confirm persistent bot mode is the default profiling path and remove avoidable per-game startup overhead.
2. Reduce repeated allocation in action mask construction.
3. Cache masks keyed by action counts and device.
4. Avoid repeated tensor creation inside postprocessing.
5. Improve encoded-observation stacking and buffer reuse.
6. Increase average eval batch size during real self-play without reducing actual simulations.
7. Reduce repeated `encode_observation` work.
8. Reduce external bot JSON/message copying.
9. Reduce replay write overhead without dropping required records.
10. Tune `search_batch_size` for actual GPU utilization in full games.

Do **not** start by changing model architecture.

Do **not** start by changing training.

Do **not** start by adding tournaments or rating evaluation.

---

## Profiling Interpretation

If full-run `simulations_per_sec` is low but standalone `profiling.mcts_search` is fast, the bottleneck is likely Java/Python protocol, process lifecycle, replay writing, or per-action overhead.

If `choose_action.encode_ms` dominates, optimize observation encoding, message normalization, and batch stacking.

If `choose_action.policy_ms` or MCTS `eval_ms` dominates, inspect model forward, device transfer, and NN batch size.

If `choose_action.search_ms` dominates while eval time is small, inspect native MCTS select/expand overhead and action-prior postprocessing.

If `avg_eval_batch` is low, improve batching strategy before changing the model.

If replay steps/sec is poor while sims/sec is good, inspect Java game progression, protocol round trips, and replay serialization.

If invalid/fallback actions appear after a change, inspect legal action filtering, action-id mapping, and request-scoped action ids.

If self-play games become much shorter or action counts collapse, treat the speedup as suspicious until proven valid.

---

## What Not To Do

Do not make self-play faster by:

- lowering games, simulations, max actions, or turn caps without recording it
- editing profiler metrics
- hiding expensive operations outside timed regions
- skipping model forward
- replacing values/priors with constants
- pruning legal actions incorrectly
- silently disabling belief or hidden-information features
- disabling native transition checks
- ignoring failed leaf encodings
- suppressing invalid-action failures
- dropping replay records
- switching to an easier map or tribe setup without treating it as a new baseline

---

## Optional Diagnostic: Standalone MCTS Profiler

`profiling.mcts_search` can be used as a diagnostic tool when self-play results are unclear.

Use it to isolate search/NN/encoding bottlenecks, but do not make keep/discard decisions from standalone profiler metrics alone. A change only counts for this program when it improves full self-play profiling.

Example diagnostic command:

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
  *> mcts_profile.log
```

---

## End Condition

Continue until manually stopped.

At the end, summarize:

```text
latest commit
number of experiments tried
kept / discarded / crashed counts
best full self-play sims/sec improvement
best replay steps/sec improvement
main bottleneck before
main bottleneck after
remaining bottlenecks
recommended next experiments
```

Use `selfplay_profile_results.tsv`, Git history, profiler logs, and generated profiler CSV/JSON files.
