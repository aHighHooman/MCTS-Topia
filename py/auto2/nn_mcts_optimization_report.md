# NN-MCTS Optimization Report

Date: 2026-05-21

Focus: NN-MCTS search throughput and search mechanics, not playing strength.

Note: profiler instrumentation and workload accounting have changed during this program. This report intentionally avoids preserving older numeric results. Use fresh `profiling.mcts_search` runs and `nn_mcts_profile_results.tsv` for comparable measurements under the current profiler.

## Current Direction

The strongest optimization direction is reducing unnecessary work between native tree search and NN evaluation.

The search workload repeatedly selects the same in-flight native child states before the NN evaluator has completed them. Treating every repeated selection as a full Python payload/evaluation candidate wastes time in native transition application, payload serialization, Python object handling, and path completion. Grouping duplicates in native code lets one evaluated leaf value complete multiple reserved paths while preserving the selected-path accounting.

## Kept Threads

### Compact Search Inference

Compact MCTS encoding remains the right default for search-time NN inference. The profiler workload uses small generated-start positions with far fewer units, cities, and legal actions than the training-shape maxima. Paying for max-capacity tensors in the search hot path is wasteful.

Keep the compact path aligned with any future feature-schema changes in:

- `py/nn/encoding.py`
- `py/nn/model.py`
- `py/search/config.py`
- related encoding and native MCTS tests

### CUDA Search Batching

Larger CUDA leaf batches are still useful because they better amortize NN inference overhead. The tradeoff is search-shape quality: bigger batches can reduce the number of unique leaves evaluated for a fixed selected-path budget.

Do not treat batch size as a pure throughput knob. When changing it, inspect eval positions, root entropy, action breadth, top-visit share, and downstream self-play data quality.

### Native Compiler Optimization

The native MCTS extension should keep optimized compiler flags enabled for profiling and production search. Continue watching parity behavior when using aggressive floating-point settings.

### Native Duplicate Selection Grouping

The latest high-value native direction is grouping duplicate in-flight selections. Native search can reserve repeated paths for the same parent/action, emit one leaf payload for NN evaluation, and complete the repeated backups when Python returns the value.

This reduces:

- duplicate native transition application
- duplicate payload serialization
- duplicate Python selection records
- duplicate `expand` calls from Python

This is currently the most important native-side cleanup to preserve when further refactoring the search loop.

### Native Payload Deduplication

When grouping is not possible or not enabled, deduplicating serialized payloads by state key is still a useful fallback idea. It avoids repeatedly converting identical native states into Python dictionaries when Python will evaluate only one of them.

## Threads That Did Not Pay Off

- BF16/FP16 autocast did not improve this workload.
- CUDNN benchmark mode did not help.
- Parallel Python observation encoding was not useful for this shape of workload.
- Small action-feature caching in Python encoding was not enough to justify keeping.
- Direct token-type embedding row adds in the model were slower than the existing embedding-call path.
- Caching coordinate prior planes in the model was slower in the measured profiler run.
- Reusing pending child states without grouping duplicate selections did not materially improve the workload.

## Current Bottlenecks

After native duplicate-selection grouping, the dominant remaining work is NN evaluation:

- model forward, especially board convolution and transformer work
- Python observation encoding
- output postprocessing and CPU/GPU transfer overhead at smaller batch counts

Native selection is no longer the primary target for this profiler unless a future workload or search policy changes the duplicate-selection pattern.

## Recommended Next Experiments

1. Add current-profiler metrics for grouped native selections: number of emitted eval payloads, grouped duplicate count, and repeated backup count.
2. Try structured/native feature emission for leaf states so Python does less dictionary walking in `encode_observation`.
3. Investigate model-forward cost under current compact shapes, especially board encoder cost for small boards.
4. Compare fixed-simulation and wall-clock modes after any search-loop change; they stress batching and warmup differently.
5. Run a self-play batch-quality sweep separately from this program before adopting throughput-only knobs for training data generation.

## Bottom Line

The main lesson is not a specific old profiler number. Search-time inference should be compact, batched, and native-aware. The highest-value native work is avoiding Python payloads and Python completions for repeated in-flight leaf selections; the highest-value remaining work is reducing NN evaluation and encoding cost under the current compact search workload.
