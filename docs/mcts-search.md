# MCTS Search

This document describes the currently implemented MCTS search path. There are two main modes:

- NN-backed native MCTS, driven by `py/search/native/mcts.py` and implemented by `py/search/native/src/mcts.cpp`.
- Static-eval native MCTS, used by `bots/static_mcts_bot.cpp`, where leaf expansion evaluates directly in C++ through `evaluate_static_state`.

## Default Configuration

The default Python `SearchConfig` in `py/search/config.py` is:

| Setting | Default | Effect |
| --- | ---: | --- |
| `num_simulations` | 64 | Expanded-node budget when no wall-clock budget is used. |
| `batch_size` | 64 | Leaf-selection frontier size. |
| `c_puct` | 1.5 | Exploration constant in PUCT. |
| `dirichlet_alpha` | 0.15 | Root noise concentration. |
| `dirichlet_epsilon` | 0.2 | Root noise mixture amount. |
| `root_temperature` | 1.0 | Visit distribution temperature. |
| `top_k_actions` | 64 | Root action cap before tree construction. |
| `sample_action` | true | Sample from visits instead of argmax. |
| `min_non_end_turn_visits` | 1 | Avoid selecting `END_TURN` if no non-end action was visited. |
| `use_progressive_widening` | true | Only expose a growing prefix of child actions. |
| `reuse_tree` | false | Reusable session is off unless explicitly enabled. |

The standalone static bot has similar defaults but uses `dirichlet_alpha=0.3`, `dirichlet_epsilon=0.25`, and `seed=13`.

## Root Setup

1. The root protocol message is normalized.
2. Root actions are capped to `model_cfg.max_actions` (`512` by default).
3. Root priors and root value are produced:
   - NN mode calls the policy-value network.
   - Static bot mode calls `evaluate_static`.
4. The root action set is optionally pruned by `top_k_actions`.
5. The native `NativeMCTS` tree is constructed with:
   - parsed game state,
   - selected root action indexes,
   - normalized priors for those selected actions,
   - root value,
   - terminal flag,
   - seed,
   - max action count,
   - progressive widening flag.
6. Dirichlet noise is mixed into root priors.

The Python root pruning keeps tactical/action-critical types first, then fills remaining budget with early unique move signatures. The standalone static bot ranks non-forced fill actions by static prior instead. In both cases, `top_k_actions <= 0` means "search all root actions."

## Tree State

Each native node stores:

- `state_index`: index into the native state array.
- `priors`: local priors for the state's legal actions.
- `visits`: per-edge visit counts.
- `value_sums`: per-edge accumulated root-perspective value.
- `child_node_ids`: per-edge child node ids, or `-1` if not expanded.
- `total_visits`: sum of visits.
- `value_estimate`: value estimate at this node.
- `terminal`: terminal marker.

States are copied through the native forward model in `rules.cpp` via `apply_action_strict`.

## Selection

Selection uses a PUCT score:

```text
q = value_sum / visits, or 0 if unvisited
u = c_puct * prior * sqrt(parent_total_visits) / (1 + visits)
score = player_sign * q + u
```

`player_sign` is `-1` only when `adversarial_opponent` is enabled and the active player is not the root player. The Python NN path and standalone static bot default this to false, so search maximizes root-perspective Q at all nodes unless explicitly configured otherwise.

Progressive widening exposes:

```text
all actions if action_count <= 8
otherwise min(action_count, 8 + floor(sqrt(total_visits)))
```

This means the local action order matters: actions earlier in the legal/prior list become searchable first.

## Turn-macro plan confidence

`turn-macro-exp` enumerates maximal positive-visit trajectories from its inner primitive MCTS, so multiple complete-turn candidates may share the same first action. A trajectory's log-confidence is the sum of the log conditional visit share chosen at each inner node. Sparse trajectory leaves are completed by applying forced actions and then ending the turn; these deterministic finalization steps do not reduce confidence.

Candidates compete globally by confidence with no per-first-action quota. They are deduplicated first by exact executed trajectory and then by a material diversity key containing the first action and normalized continuation commitments. The accepted candidates' confidences are normalized with log-sum-exp and used as their outer edge priors; `max_new_edges_per_node` remains the final edge cap. Profile JSON records each root plan's log-confidence, normalized prior, selection reason, and deduplication counters.

The standalone bot preserves the selected root plan across `action_request` messages. Remaining actions are matched by stable action signatures and remapped to each request's action IDs. A continuation is abandoned when its predicted native state fingerprint no longer matches, the next action is unavailable, a forced action appears, or the turn reaches a boundary; profile JSON reports whether a request continued or replanned.

## Batched NN Search

The Python search loop repeatedly:

1. Calls native batch selection (`select_leaf_batches_evals_only` when present, otherwise older selection methods).
2. Uses virtual reservations so multiple selected paths do not all collide on the same frontier edge.
3. Deduplicates leaf evaluations by native `state_key`.
4. Encodes unevaluated leaf payloads and evaluates the NN in one batch.
5. Expands each non-terminal selected leaf with returned priors/value.
6. Completes all reserved paths by adding the leaf value to edge value sums.

Unsupported or approximate native transitions raise `NativeSearchParityError`; this prevents the NN search from silently training on forward-model states known to be approximate.

## Static Search

The standalone static bot uses `NativeMCTS.run_static_search_batch`. This stays fully in C++:

1. Select a leaf path with the same PUCT logic.
2. If the leaf is already expanded or terminal, back up the existing leaf value.
3. Otherwise apply the selected primitive action.
4. If terminal, use the terminal value.
5. Otherwise call `evaluate_static_state(child_state, actions)`.
6. Allocate the child node with static priors/value.
7. Back up the value.

The static bot can run by simulation count or wall-clock seconds. With profiling enabled it reports native timing buckets for selection, apply action, static eval, node allocation, backup, and transition substeps.

## Backup and Perspective

Values are stored on edges from the root player's perspective. If a leaf value is known to already be root-perspective, it is used as-is. Otherwise it is converted from the leaf active player's perspective using `value_to_root_perspective`.

The older exposed `backprop` method simply adds `leaf_value` directly and is not the main batched completion path.

## Visit Distribution and Action Choice

Root visit probabilities are:

```text
temperature <= 1e-6: one-hot argmax
otherwise: visits ** (1 / temperature), normalized
```

If no paths were selected, Python falls back to prior/uniform distribution depending on path.

Action selection is:

- sample from visit probabilities when `sample_action=true`;
- otherwise choose max visit probability;
- then apply the `END_TURN` guard, which rejects `END_TURN` if configured and no non-end action received any visit.

The result includes:

- selected `action_id`;
- original root `action_index`;
- sparse `visit_distribution`;
- full `visit_target` aligned to original root actions;
- root value;
- optional root action stats.

## Tree Reuse

`ReusableNativeMCTSSession` can preserve a tree across action requests when `reuse_tree=true`.

Reuse only succeeds when:

- the promoted root payload exists;
- player id matches;
- observation JSON signature matches;
- native root actions map uniquely to the current request actions by action signature.

After selecting an action, the session tries to promote the selected child as the new root. It resets on `END_TURN`, missing promotion support, ambiguous action mapping, observation mismatch, or unexpanded selected child.

## Practical Caveats

- Root action ids are request-scoped; the code maps by action signatures when reusing trees.
- Hidden-information states are player-specific observations, not omniscient game states.
- Native transition parity matters. Approximate transitions intentionally fail strict NN MCTS.
- With `adversarial_opponent=false`, opponent nodes are not minimax nodes.
- Progressive widening depends on action order. If Java action ordering changes, search behavior can shift even with identical priors.
