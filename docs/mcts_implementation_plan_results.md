# MCTS Implementation Plan Results

Profiler workload unless noted: `python -m profiling.mcts_search` using `py/profiling/configs/mcts_search.json`, 36 captured PlayLG Drylands/Tiny payloads, 3 seconds per position.

| # | Idea | Result | Decision |
|---|---|---:|---|
| Baseline | Starting NN profile | 319.36 sims/s, 2914.32 paths/s, avg depth 1.11, max depth 4 | Reference |
| 1 | Dynamic NN batch sizing | 314.39 sims/s, 2905.63 paths/s, avg depth 1.11 | Reverted, slower |
| 2 | Progressive widening | 341.99 sims/s, 2949.76 paths/s, avg depth 1.18 | Kept |
| 3 | Heuristic root pruning | 345.30 sims/s, 2949.49 paths/s, avg depth 1.18 | Kept |
| 4 | Semi-native normalized encoding fast path | 364.80 sims/s, 2964.55 paths/s, avg depth 1.20 | Kept |
| 5 | Component-level encoding cache | 378.10 sims/s, 2964.28 paths/s, avg depth 1.21 | Kept |
| 6 | Smaller random-init evaluator | 394.61 sims/s, 3006.72 paths/s, avg depth 1.22 | Kept |
| 7 | Static internal timing | Static timing showed `apply_action` at 73.5% and static eval at 13.6% of run time | Kept, profiler-gated |
| 8 | Native tree storage reserve | 398.30 sims/s, 3008.67 paths/s, avg depth 1.22 | Kept |
| 9 | Terminal child tactical shortcut | 396.11 sims/s, 3008.97 paths/s, avg depth 1.22 | Reverted, slower |
| 10 | Static branching policy tuning | Static profile improved from 1455.11 to 1463.43 sims/s | Kept |
| 11-13 | Greedy macro / turn-boundary / beam-width-1 prototype | 370.30 sims/s, 2983.53 paths/s, avg depth 1.20 | Reverted, slower |
| 14 | Process root-parallel static MCTS prototype | Fixed-node microbenchmark: serial 4.369s, 2-worker root-parallel 13.217s, speedup 0.331x | Reverted, slower |
| 15 | Selection-ahead NN pipeline proxy | 451.50 sims/s, 5632.94 paths/s, avg depth 1.14, max depth 5 | Kept |
| 16 | Canonical native eval key prototype | 448.84 sims/s, 5623.33 paths/s, avg depth 1.14, eval cache hits 0 | Reverted, slower/no reuse |
| 17 | Move-only native state storage prototype | 448.18 sims/s, 5628.74 paths/s, avg depth 1.14 | Reverted, slower |
| 18 | Partial C++ leaf board feature encoder | 493.30 sims/s, 5621.17 paths/s, avg depth 1.16, max depth 5 | Kept |

Correctness gates run during the retained changes:

- `python -m pytest py/tests` passed after the smaller evaluator change.
- Focused native/profiler tests passed after native C++ changes.
- C++ board tensor parity was checked against the Python encoder before keeping idea 18.
