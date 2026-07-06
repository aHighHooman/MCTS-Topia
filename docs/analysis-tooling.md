# Analysis Tooling

This is a runbook for the currently visible profiling and static-eval experiment tools.

Run commands from the repository root with:

```powershell
$env:PYTHONPATH = "$PWD\py"
```

Build Java and the standalone static bot when needed:

```powershell
.\scripts\build_java.ps1
.\scripts\build_static_bot.ps1
```

## Static Bot

The standalone static bot is `out/native/static_mcts_bot.exe` after build.

Useful options:

```powershell
out/native/static_mcts_bot.exe `
  --search-mode primitive `
  --simulations 512 `
  --search-batch-size 64 `
  --top-k-actions 64 `
  --static-eval-variant baseline `
  --deterministic `
  --profile-json
```

Important flags:

- `--search-mode primitive|turn-macro-exp`
- `--simulations N`
- `--wall-clock-per-action-seconds S`
- `--top-k-actions N`
- `--max-actions N`
- `--search-batch-size N`
- `--c-puct X`
- `--no-dirichlet`
- `--uniform-prior`
- `--static-eval-variant baseline|experimental|experimental-2|experimental-training`
- `--static-eval-weight-overrides "term=value,term=value"`
- `--native-opponent-mode root-max|root-adversarial`

## Generate Profile Positions

Config:

```text
py/profiling/configs/generate_mcts_profile_positions.json
```

Command:

```powershell
python -m profiling.generate_mcts_profile_positions --config py/profiling/configs/generate_mcts_profile_positions.json
```

Outputs:

- selected payloads in `debug-logs/mcts-profile-payloads/`;
- raw source game/action logs in `debug-logs/mcts-profile-position-corpus/`;
- selection summary in `debug-logs/mcts-profile-payloads/mcts_profile_position_selection_summary.json`.

## Profile Search Runtime

Config:

```text
py/profiling/configs/mcts_search.json
```

Command:

```powershell
python -m profiling.mcts_search --config py/profiling/configs/mcts_search.json
```

Useful knobs:

- `evaluator`: `nn`, `static`, or `static_exe` depending on path.
- `wall_time_sec` or `simulations`.
- `batch_size`.
- `top_k_actions`.
- `no_dirichlet`.
- `native_static_search_mode`: `primitive` or `turn-macro-exp`.
- CSV output paths for profile/position/branching/action/hardware/module detail.

Use this for performance and branching-pressure questions, not for strength claims.

## Compare Branches

Config:

```text
py/profiling/configs/compare_branches.json
```

Command:

```powershell
python -m profiling.analysis.compare_branches --config py/profiling/configs/compare_branches.json
```

Outputs:

- `positions.jsonl`
- `positions_summary.csv`
- per-target action files
- `actions_wide.csv`
- `summary.json`
- `report.html`

This compares root visit distributions for multiple targets, usually `baseline` and `experimental`.

## Build Branchpoint Dataset

Config:

```text
py/profiling/configs/branchpoint_dataset_builder.json
```

Typical command:

```powershell
python -m profiling.analysis.branchpoint_dataset_builder `
  --config py/profiling/configs/branchpoint_dataset_builder.json `
  --positions-jsonl debug-logs/analysis/branch-compare/<run>/positions.jsonl
```

Outputs:

- `branchpoints.jsonl`
- `branchpoints.csv`
- `summary.json`
- `report.html`

This filters branch comparisons down to candidate disagreement positions.

## Analyze One Position

Config:

```text
py/profiling/configs/position_analyzer.json
```

Command:

```powershell
python -m profiling.analysis.position_analyzer_cli --config py/profiling/configs/position_analyzer.json
```

Targets can be:

- `baseline`
- `experimental`
- `uniform`
- `name:variant=experimental-training,weight_overrides=term=value;term=value`

The analyzer runs the static executable, captures root stats, and optionally attaches static value breakdowns.

## Value Breakdown

Config:

```text
py/profiling/configs/value_breakdown.json
```

Command:

```powershell
python -m profiling.analysis.value_breakdown --config py/profiling/configs/value_breakdown.json
```

Outputs:

- `positions.jsonl`
- `terms.csv`
- `terms_components.csv`
- `terms_aggregate.csv`
- `term_summary.csv`
- `term_comparison.csv` when comparing against another run
- `summary.json`
- `run_status.json`
- `report.html`

Use this to answer "which terms are driving this value/action preference?"

## Root Child Value Matrix

Config:

```text
py/profiling/configs/root_child_value_matrix.json
```

Command:

```powershell
python -m profiling.analysis.root_child_value_matrix --config py/profiling/configs/root_child_value_matrix.json
```

This evaluates root children and writes a matrix useful for seeing whether MCTS and one-step static values agree.

## Pairwise Action Explainer

Config:

```text
py/profiling/configs/pairwise_action_explainer.json
```

Command:

```powershell
python -m profiling.analysis.pairwise_action_explainer --config py/profiling/configs/pairwise_action_explainer.json
```

Use this when two specific actions need explanation.

## Tune Static Eval Weights

Configs:

```text
py/profiling/configs/tune_static_eval_weights.json
py/profiling/configs/tune_static_value_from_results.json
py/profiling/configs/tune_static_value_experimental_training.json
```

Commands:

```powershell
python -m profiling.analysis.tune_static_eval_weights --config py/profiling/configs/tune_static_eval_weights.json
python -m profiling.analysis.temp_variant_optimizer --config py/profiling/configs/temp_variant_optimizer.json
```

Native helpers are built with:

```powershell
.\scripts\build_static_value_tuning_tools.ps1
```

Use validation results, not fit metrics alone, to decide whether a candidate is worth keeping.

## Overlay and Counterfactuals

Overlay config:

```text
py/profiling/configs/overlay_match.json
```

Counterfactual config:

```text
py/profiling/configs/forced_root_counterfactual.json
```

Overlay is correlation over a recorded match timeline. Counterfactual replay is the better tool when the question is "what if this root action had been forced?"

## Recommended Workflow

1. Generate or reuse a payload corpus.
2. Compare `baseline` vs candidate target.
3. Build branchpoint candidates.
4. Use value breakdown, root-child matrix, or pairwise explainer on the worst disagreements.
5. Propose a small targeted override or code change.
6. Validate through balanced games on held-out seeds.
7. Record the run in `docs/experiment-record.md` if it changes the decision history.
