# AutoResearch Program: Native Static MCTS Bot

This program describes an autonomous local experiment loop for improving the native static-MCTS bot.

The goal is to improve the accepted variant Elo of the static MCTS bot while preserving native forward-model correctness, evaluator safety, and reasonable simplicity.

This is not a neural-network checkpoint training loop. The optimization target is the hand-written/static MCTS decision logic, mainly the native static evaluator and nearby search code.

---

## Core Idea

Each experiment should be:

1. committed before evaluation,
2. evaluated only through the blackbox evaluator,
3. kept if the evaluator says `status: keep`, and
4. discarded by resetting Git if the evaluator says `discard` or `crash`.

Prefer small, isolated experiments. Larger changes are okay if they are clearly motivated and tested

The branch advances only through accepted commits.

Prefer simple improvements. A small gain from deleting or simplifying logic is valuable. A small gain from fragile special cases is suspicious and should usually be discarded unless the blackbox evaluator clearly accepts it.

---

## Protected Blackbox Evaluator

The evaluator is the source of truth. Do not edit it.

Protected evaluator file:

```text
py/auto/static_autoresearch_eval.py
```

Protected Elo ledger:

```text
~/.cache/static_mcts_autoresearch/static_eval_elos.csv
```

The autoresearcher must not manually open, edit, delete, rewrite, fabricate, or patch the Elo ledger. The evaluator script is the only allowed writer.

The autoresearcher must not modify the evaluator script, its constants, its CLI, its acceptance thresholds, its ledger path, its Elo rules, its wall-clock budget, its game count, or its opponent-selection policy.

The autoresearch loop should use only these evaluator CLI controls:

```text
--candidate-ref
--previous-ref
--baseline-ref
--iteration
--require-clean
```

Do not add or use flags to override Elo, ledger behavior, baseline rating, K-factor, wall-clock time, number of games, score thresholds, or write/discard behavior.

The evaluator's intended Elo model is:

- the baseline variant is fixed at Elo 1000,
- the previous accepted variant's Elo is read from the protected ledger,
- bootstrap is allowed only when `previous_ref == baseline_ref`,
- the candidate starts at the previous accepted variant's Elo,
- official candidate Elo updates only from games against the previous accepted variant,
- games against the baseline are only a divergence/regression check unless previous is also baseline,
- only accepted `keep` variants are written to the ledger,
- discarded and crashed variants are not written to the ledger.

---

## Baseline and Variant Naming

Use Git tags to identify accepted variants.

Recommended convention:

```text
static-v1   original baseline
static-v2   first accepted improvement
static-v3   second accepted improvement
...
```

The baseline tag must point at the original static-eval baseline commit.

If no baseline tag exists yet, create it from the clean baseline commit:

```bash
git tag static-v1
```

Do not move `static-v1` after creating it.

Each accepted candidate should be tagged after the evaluator accepts it:

```bash
git tag static-v<N>
```

where `<N>` is the accepted iteration number.

---

## Branch Setup

Create a dedicated local branch for the autoresearch run:

```bash
git checkout -b autoresearch/static-mcts
```

The branch should start from the latest accepted variant. For a fresh run, this is usually `static-v1`.

Before starting the loop, verify:

```bash
git status --short
git tag --list "static-v*" --sort=v:refname
```

If `static-v1` is missing, create it only if the current commit is truly the intended baseline.

---

## In-Scope Files

Primary editable files:

```text
py/search/native/native_static_eval.cpp
py/search/native/native_static_eval.hpp
py/search/native/static_mcts.py
```

Only edit these files unless there is a clear, minimal reason not to.

Secondary files are normally out of scope. Do not edit game rules, Java engine code, self-play infrastructure, replay infrastructure, test expectations, or the blackbox evaluator to make an experiment pass.

You may read relevant code for context, but do not weaken correctness or evaluation machinery.

---

## Out-of-Scope Files

Do not modify these during autoresearch:

```text
py/auto/static_autoresearch_eval.py
~/.cache/static_mcts_autoresearch/static_eval_elos.csv
py/training/replay.py
py/training/selfplay.py
py/tests/test_native_mcts.py
```

Do not modify native transition rules unless the human explicitly asks. The static evaluator should be improved without changing the game model.

---

## Setup Checklist

Before the experiment loop begins:

1. Confirm this is a Git repo.
2. Confirm `static-v1` exists and points to the intended baseline.
3. Confirm `py/auto/static_autoresearch_eval.py` exists.
4. Confirm the working tree is clean.
5. Create an untracked local results log if missing:

```bash
printf "iteration\tcommit\tstatus\tcandidate_elo\tdelta_vs_previous\tscore_prev\tscore_baseline\ttotal_seconds\tdescription\n" > results.tsv
```

Do not commit `results.tsv`.

---

## Test Gate

Before running the blackbox evaluator, run the static/native MCTS tests:

```bash
PYTHONPATH=py python -m unittest discover -s py/tests -p "test_native_mcts.py" > test.log 2>&1
```

If tests fail because of a trivial typo or compile issue caused by the current experiment, fix it and retry.

If the idea itself caused native parity errors, invalid priors, unsupported transitions, invalid leaf payloads, or tactical regressions, discard the experiment.

Do not edit the tests to pass.

---

## Evaluator Command

Run the evaluator only after committing the candidate and only with a clean working tree.

For iteration 2, previous and baseline are the same:

```bash
PYTHONPATH=py python -m auto.static_autoresearch_eval \
  --iteration static-v2 \
  --candidate-ref HEAD \
  --previous-ref static-v1 \
  --baseline-ref static-v1 \
  --require-clean \
  > eval.log 2>&1
```

For iteration N where N > 2:

```bash
PYTHONPATH=py python -m auto.static_autoresearch_eval \
  --iteration static-v<N> \
  --candidate-ref HEAD \
  --previous-ref static-v<N-1> \
  --baseline-ref static-v1 \
  --require-clean \
  > eval.log 2>&1
```

Never use `tee`. Redirect the evaluator output to `eval.log` and inspect the summary afterwards.

The evaluator prints a summary after a line containing:

```text
---
```

Key fields to inspect:

```text
status
candidate_elo
candidate_elo_delta_vs_previous
candidate_elo_delta_from_start
baseline_check_candidate_elo
candidate_group_scores
ledger_written
reasons
```

The evaluator returns a nonzero exit code when the candidate should not be accepted.

---

## Experiment Loop

LOOP until manually stopped:

1. Identify the latest accepted tag.

```bash
git tag --list "static-v*" --sort=v:refname
```

2. Let the latest accepted tag be `static-v<K-1>`. Create the next candidate iteration name `static-v<K>`.

3. Confirm the branch is currently at the latest accepted commit or reset to it:

```bash
git reset --hard static-v<K-1>
```

4. Think of one small experiment.

Good experiment types:

- tune unit value tables,
- tune capture/village/ruin priorities,
- tune end-turn penalties,
- tune city value and production terms,
- tune threat penalties,
- tune movement-to-objective heuristics,
- improve value normalization,
- remove brittle special cases,
- simplify duplicated scoring logic,
- improve root prior shaping in static MCTS.

Avoid giant rewrites unless many smaller experiments are exhausted.

5. Edit only the in-scope files.

6. Run the test gate.

```bash
PYTHONPATH=py python -m unittest discover -s py/tests -p "test_native_mcts.py" > test.log 2>&1
```

7. If tests pass, commit the candidate.

```bash
git add py/search/native/native_static_eval.cpp py/search/native/native_static_eval.hpp py/search/native/static_mcts.py
git commit -m "try: <short experiment description>"
```

If some of those files were not edited, `git add` will simply ignore unchanged files.

8. Run the blackbox evaluator.

For `static-v<K>`:

```bash
PYTHONPATH=py python -m auto.static_autoresearch_eval \
  --iteration static-v<K> \
  --candidate-ref HEAD \
  --previous-ref static-v<K-1> \
  --baseline-ref static-v1 \
  --require-clean \
  > eval.log 2>&1
```

9. Read the summary.

```bash
tail -n 80 eval.log
```

10. Record the result in `results.tsv`.

Use the candidate commit hash:

```bash
git rev-parse --short HEAD
```

Append one tab-separated row:

```text
iteration	commit	status	candidate_elo	delta_vs_previous	score_prev	score_baseline	total_seconds	description
```

11. Decision:

- If `status: keep` and `ledger_written: True`, tag the commit as accepted:

```bash
git tag static-v<K>
```

Then continue from this new accepted commit.

- If `status: discard` or `status: crash`, do not tag it. Reset back to the previous accepted variant:

```bash
git reset --hard static-v<K-1>
```

Then try a different idea.

---

## Crash Handling

If tests crash:

1. inspect `test.log`,
2. fix simple typos/import/compile errors caused by the current edit,
3. rerun tests,
4. if still broken or the idea is fundamentally bad, reset to the previous accepted tag.

If the evaluator crashes:

1. inspect `eval.log`,
2. do not edit the evaluator,
3. if the crash was caused by the candidate code, fix or discard the candidate,
4. if the crash appears to be an evaluator/infrastructure issue, stop and report the issue to the human.

Never patch the evaluator or ledger to make an experiment pass.

---

## Acceptance Rule

The only automatic acceptance condition is:

```text
evaluator summary says status: keep
and ledger_written: True
```

Do not override this judgment manually.

Do not accept a candidate because it “looks good” if the evaluator discarded it.

Do not discard a candidate that the evaluator kept unless the code is obviously malicious, corrupt, or outside the allowed scope.

---

## Simplicity Criterion

When choosing new ideas, prioritize changes with favorable strength-to-complexity ratio.

Strong candidates:

- simple coefficient changes,
- better unit/resource/city valuation,
- removing an obviously bad penalty or bonus,
- replacing duplicated logic with one clearer calculation,
- improvements that help broad classes of actions.

Weak candidates:

- large hard-coded action scripts,
- seed-specific or opponent-specific hacks,
- tricks that only beat the baseline but not the previous accepted variant,
- changes that reduce testability or make transitions harder to reason about,
- changes that require editing the evaluator or ledger.

---

## Metric Discipline

Do not optimize only against the baseline.

The candidate is evaluated against:

1. the previous accepted variant, which determines official Elo, and
2. the original baseline, which checks for divergence/regression.

The baseline has fixed Elo 1000 and is not a moving target.

The previous accepted variant has whatever Elo it earned when it was accepted.

For iteration 2, the previous accepted variant is the baseline, so the candidate gets its first real Elo by playing against the baseline.

For later iterations, the official Elo chain comes from playing against the immediately previous accepted variant.

---

## Output Files

Allowed generated files:

```text
results.tsv
test.log
eval.log
rl/static_autoresearch_eval/
```

Do not commit logs or generated tournament outputs unless the human explicitly asks.

The protected ledger is not an output file for the autoresearcher to manage. It belongs to the evaluator.

---

## Things To Try First

Start with small changes in `native_static_eval.cpp`.

Good early experiments:

1. Adjust village capture and city capture weighting.
2. Tune ruin/examine value.
3. Tune end-turn penalty so the bot does not end too early but also does not waste actions.
4. Improve movement scoring toward villages, ruins, exposed cities, and combat opportunities.
5. Tune unit value based on HP, veteran status, range, mobility, and attack/defense.
6. Tune threat penalties for fragile units.
7. Tune city production/population value.
8. Remove or reduce any obviously over-specific bonus.

Only move into `static_mcts.py` after exhausting obvious evaluator improvements.

---

## Do Not Game The Evaluator

Do not make changes designed to exploit the evaluator setup, fixed seeds, baseline behavior, or ledger rules.

Do not detect iteration names, Git refs, seeds, opponent identity, worktree names, or environment paths to alter play.

Do not add special cases for `baseline`, `previous`, `candidate`, `static-v1`, or similar strings.

The goal is a genuinely stronger static MCTS bot, not a bot that exploits the test harness.

---

## End-of-Run Report

When stopped by the human, summarize:

- latest accepted tag,
- latest accepted commit,
- latest accepted Elo,
- number of experiments attempted,
- number kept,
- number discarded,
- number crashed,
- best idea found,
- notable failed ideas,
- recommended next experiments.

Use `results.tsv`, Git history, and evaluator summaries. Do not read or edit the protected ledger directly.
