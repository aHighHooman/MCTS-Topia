# Tribes MCTS Documentation

This folder documents the current search and static-evaluation stack in this repository as of the local checkout inspected on 2026-07-03.

The goal is practical: make the current behavior readable, record what has already been tried, and reduce repeated trips into old static-eval tuning dead ends.

## Documents

- [MCTS Search](mcts-search.md) explains the native tree, NN-backed search loop, static-search loop, root pruning, batching, backup, output selection, and tree reuse.
- [Static Evaluation](static-eval.md) explains static priors, value terms, variants, weight overrides, breakdown output, and important caveats.
- [Experiment Record](experiment-record.md) summarizes the visible profiling/tuning experiments, saved payload corpus, branch comparisons, value breakdowns, and validation outcomes.
- [Analysis Tooling](analysis-tooling.md) is a compact runbook for the scripts and configs used to reproduce or extend the experiments.
- [Tournament Evaluation](tournament-evaluation.md) defines the valid bot-vs-bot objective, seat balancing, completion checks, and wall-clock calibration procedure.

## Main Source Files

- `py/search/native/mcts.py`: Python-facing NN MCTS wrapper and reusable session.
- `py/search/native/src/mcts.cpp`: native tree, PUCT, batching, static search, backup, visit output.
- `py/search/native/src/static_eval.cpp`: static action priors and static value function.
- `bots/static_mcts_bot.cpp`: standalone external bot using static eval plus native MCTS.
- `py/profiling/analysis/`: analysis and tuning tools.
- `py/profiling/configs/`: checked-in experiment defaults.
- `debug-logs/analysis/`: local generated evidence used by the experiment record.

## Accuracy Notes

These docs describe the code that exists now. They do not claim that the static evaluator is strong, only that the implementation and visible experiment evidence have been captured.

Generated logs under `debug-logs/` are local artifacts. Treat their counts and conclusions as useful history, not canonical benchmark results, unless you rerun them from clean configs.
