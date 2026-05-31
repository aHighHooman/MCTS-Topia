# MCTS Hybrid Bot Performance Implementation Plan

## What Changed From The First Draft

- Removed ideas that depend on a trained neural network providing useful policy or value guidance.
- Reframed neural-network work around raw inference throughput, batching, encoding, and architecture speed.
- Replaced static-prior/NN-distillation assumptions with tests that work before training starts.
- Kept native/static and multi-action-turn search ideas because they do not require a trained network.

## Current Baseline From Profiling

- NN MCTS: about 381 sims/s, about 2,985 selected paths/s, avg depth about 1.13, max depth about 2.6.
- Static MCTS: about 1,707 sims/s, avg depth about 9.64, max depth about 25.
- NN hotspots: `encode_observation` 43%, model forward 32%, native selection 19%, about 4.85 GB CPU-to-GPU transfer.
- Static hotspot: `run_static_search_batch` 94%, so static gains likely need C++ tree/rules/search work.

## Experiment Backlog

1. Dynamic NN batch sizing by tree phase
   - Use smaller frontier batches at shallow depth and larger batches after root expansion.
   - Goal: reduce duplicate shallow selections and improve average depth.

2. Progressive widening
   - Start nodes with a small legal-action subset and unlock more actions as visits grow.
   - Goal: reduce MOVE-heavy multi-action branching while preserving important tactical options.

3. Heuristic root pruning independent of NN quality
   - Use static action-type rules, duplicate MOVE suppression, and tactical forced-keep rules before MCTS.
   - Do not use NN logits for this until training exists.

4. Native or semi-native observation encoding
   - Move repeated board/unit/city/action feature extraction out of Python dict walking.
   - Goal: reduce the largest NN cost, currently `encode_observation`.

5. Component-level encoding cache
   - Cache board, unit, city, and scalar encodings separately.
   - Rebuild only action features when possible for leaf states.

6. Faster random-init inference architecture benchmark
   - Test smaller MCTS evaluator shapes while preserving output interfaces.
   - Candidate knobs: fewer transformer layers, smaller `d_model`, fewer CNN blocks, or no transformer core.
   - This measures speed ceiling before training, not strength.

7. Static MCTS internal timing
   - Add C++ timers around selection, `apply_action_strict`, legal-action regeneration, static eval, node allocation, and backup.
   - Goal: identify the exact source of the 94% native hotspot.

8. Optimize native state copies
   - Audit `NativeGameState` copying during `apply_action`, pending child storage, expansion, and backup.
   - Replace avoidable deep copies/moves with pooled or in-place child construction where parity allows.

9. Action-type tactical shortcuts
   - Fast-path terminal, capture, kill, city capture, veteran, and clearly dominated recover/move cases in C++.
   - Goal: skip full expansion/eval when the action outcome is obvious.

10. Tune static branching policy
    - Static currently searches about 10-12 root actions from about 51.
    - Test forced tactical quotas plus duplicate MOVE pruning to improve speed and useful depth.

11. Macro-action search
    - Search composed action chunks such as move+attack, spawn+move, gather+level-up, or economy+end-turn.
    - Goal: better match the game's multi-action turn structure.

12. Turn-boundary MCTS
    - Treat a complete player turn as one ply.
    - Use an inner greedy or beam policy to assemble intra-turn actions.
    - Measure turns searched in addition to single-action depth.

13. Beam-MCTS hybrid
    - Use beam search within the current player's turn and MCTS at turn boundaries.
    - Goal: reduce action-order explosion.

14. Root-parallel static MCTS
    - Run multiple native static search workers in parallel and merge root visits.
    - Useful if static search is CPU-bound and mostly single-threaded.

15. Asynchronous CPU/GPU pipeline for NN
    - Let native selection prepare the next batch while GPU evaluates the current batch.
    - Goal: overlap native selection with model forward.

16. Canonical transposition table
    - Hash native states so equivalent states reached by different intra-turn action orders share nodes/evals.
    - Goal: reduce redundant search in multi-action turns.

17. Compact C++ state representation
    - Stop retaining Python `py::dict` observation payloads in native states except when serialization is needed.
    - Keep search state in compact C++ structs.

18. C++ leaf feature encoder
    - Instead of serializing leaf payloads back to Python for NN encoding, emit already-packed tensors or flat arrays from C++.
    - Goal: remove the Python leaf encoding bottleneck.

## Experiment Protocol

- Test ideas one at a time unless a group is required to make a meaningful measurement.
- Run each experiment on the same captured profiler payloads.
- Track sims/s, selected paths/s, avg depth, max depth, root entropy, searched root actions, action-type visit share, elapsed time, and memory transfer.
- Keep a change only if it improves performance without correctness regressions.
- If an experiment slows down or fails correctness checks, revert it cleanly before continuing.

## Validation

- For native/rules changes, run focused native parity coverage and `python -m pytest py/tests`.
- For profiler-only instrumentation, run the MCTS profiler and relevant profiler tests.
- For search-shape changes, compare selected action distributions against current static and NN baselines.
- For any change that survives profiling, run a small tournament or head-to-head evaluation before considering it broadly accepted.

## Assumptions

- The NN is intentionally untrained, so experiments must not rely on NN policy or value quality.
- Requested improvements should become default behavior if kept.
- Java rules remain authoritative.
- Native transition changes require parity validation.
