# Report: Training a Neural Network With MCTS Using Static Eval as Temporary Guidance

## Goal

You want to train a neural network with MCTS while using your existing static evaluator as early guidance.

The key constraint is:

> The neural network should not directly imitate the static evaluator.

So the static evaluator should help MCTS search early on, but the neural network should still learn from:

```text
value target  = final win/loss result
policy target = MCTS visit distribution
```

not from:

```text
value target  = static eval value
policy target = static eval priors
```

---

## High-Level Recommendation

Use the static evaluator as a temporary MCTS scaffold.

The training should look like this:

```text
Early training:
    Static eval strongly guides MCTS.
    NN starts weak/random/neutral.

Middle training:
    Static eval and NN are mixed together.

Late training:
    NN dominates MCTS.
    Static eval influence becomes tiny.

Final training/evaluation:
    Static eval is fully disabled.
    Pure NN-MCTS is tested.
```

This avoids reward shaping while still preventing early MCTS from being useless.

---

## Core Design

Create a hybrid evaluator:

```text
P_used = mix(P_nn, P_static)
V_used = mix(V_nn, V_static)
```

Where:

```text
P_static = priors from static evaluator
V_static = value from static evaluator

P_nn = policy head output
V_nn = value head output
```

Then MCTS uses:

```text
P_used
V_used
```

But training uses only:

```text
final game result
MCTS visit distribution
```

---

## Add Two Control Parameters

Add these parameters to your search/training config:

```text
static_policy_weight
static_value_weight
```

These control how much the static evaluator influences MCTS.

Example:

```text
static_policy_weight = 1.0 means use mostly static policy guidance.
static_policy_weight = 0.0 means use only NN policy.

static_value_weight = 1.0 means use mostly static value.
static_value_weight = 0.0 means use only NN value.
```

---

## Suggested Weight Schedule

| Phase | static_policy_weight | static_value_weight | Meaning |
|---|---:|---:|---|
| Bootstrap | 1.0 | 1.0 | Static eval heavily guides MCTS |
| Early | 0.75 | 0.5 | NN starts contributing |
| Middle | 0.4 | 0.2 | NN becomes important |
| Late | 0.1 | 0.0 | Static only weakly biases policy |
| Final | 0.0 | 0.0 | Pure NN-MCTS |

Important:

> Decay `static_value_weight` faster than `static_policy_weight`.

Reason:

Static value affects MCTS backups very strongly. If it stays active too long, the NN may never learn its own long-horizon value estimate.

---

## Policy Mixing

Best version:

```text
logits_used = logits_nn + beta * log(P_static + eps)
P_used = softmax(logits_used)
```

Where:

```text
beta = static_policy_weight
```

This treats static eval as a soft prior bias.

Simpler version:

```text
P_used = (1 - static_policy_weight) * P_nn
       + static_policy_weight * P_static
```

The simpler version is easier to implement and probably fine for a first attempt.

---

## Value Mixing

Use direct interpolation:

```text
V_used = (1 - static_value_weight) * V_nn
       + static_value_weight * V_static
```

Early on, this lets static eval guide MCTS while the NN value head is untrained.

Later, `static_value_weight` should become zero.

---

## Training Targets

Train the value head using only the final game result:

```text
win  = +1
draw =  0
loss = -1
```

Train the policy head using MCTS visit counts:

```text
policy_target = root visit distribution from MCTS
```

Do not train directly on:

```text
V_static
P_static
```

The static evaluator should influence the search, not become the supervised label.

---

## Important Warning: Policy Targets Are Still Indirectly Biased

Even if you do not directly train on static priors, the MCTS visit distribution will be affected by static eval early on.

So the policy head is still indirectly influenced by the static evaluator.

That is okay as long as:

```text
static_policy_weight -> 0
static_value_weight  -> 0
```

over time.

This makes the static evaluator a curriculum tool rather than a permanent teacher.

---

## Avoid Hard Static Pruning During Training

Be careful with using static eval to remove legal actions from the root.

Hard pruning can create blind spots.

Bad training behavior:

```text
Static eval dislikes action A.
Action A is pruned.
MCTS never explores action A.
NN never learns that action A might be good.
```

For training, prefer one of these:

```text
Use no root pruning.
```

or:

```text
Keep top-k static actions.
Always keep tactical/forced actions.
Also keep random low-prior actions.
Also keep NN-preferred actions.
```

Hard pruning is more acceptable during final play/evaluation than during learning.

---

## Recommended Training Phases

### Phase 0: Static-Assisted Bootstrap

Purpose:

```text
Generate reasonable early self-play data.
Avoid totally random garbage games.
```

Use:

```text
static_policy_weight = 1.0
static_value_weight  = 1.0
```

Train on:

```text
policy target = MCTS visits
value target  = final game result
```

Do this only long enough to bootstrap the NN.

---

### Phase 1: Mixed Static + NN MCTS

Purpose:

```text
Let the NN start influencing search.
Reduce dependence on static eval.
```

Use:

```text
static_policy_weight = 0.75 -> 0.4
static_value_weight  = 0.5  -> 0.2
```

At this point, the NN should already affect both priors and values.

---

### Phase 2: NN-Dominant MCTS

Purpose:

```text
Make the NN the main driver.
Use static eval only as a weak search hint.
```

Use:

```text
static_policy_weight = 0.2 -> 0.05
static_value_weight  = 0.0
```

At this stage, static value should be gone.

---

### Phase 3: Pure NN-MCTS

Purpose:

```text
Verify the NN can play without static help.
```

Use:

```text
static_policy_weight = 0.0
static_value_weight  = 0.0
```

This is the real test.

If the NN only performs well when static eval is still enabled, it has not actually taken over.

---

## What to Log

Log these during training:

```text
static_policy_weight
static_value_weight
policy loss
value loss
policy entropy
root visit entropy
average MCTS depth
NN-only win rate
static-assisted win rate
win rate vs static-MCTS bot
percentage of root actions pruned
```

The most important evaluation is:

```text
NN-MCTS with static weights = 0
vs
static-MCTS bot
```

That tells you whether the NN has surpassed the static evaluator.

---

## What to Avoid

Avoid reward shaping unless absolutely necessary.

Avoid:

```text
value_target = static_eval_value
policy_target = static_eval_priors
```

Avoid permanent static-value fallback.

Avoid hard static pruning during training unless there are escape routes.

Avoid only evaluating the NN with static assistance enabled.

---

## Implementation Checklist

1. Add a new hybrid MCTS path:

```text
run_native_hybrid_mcts(...)
```

2. At each root/leaf evaluation, compute:

```text
static_eval = evaluate_static(...)
nn_eval     = evaluate_nn(...)
```

3. Combine priors:

```text
P_used = mix(P_nn, P_static, static_policy_weight)
```

4. Combine values:

```text
V_used = mix(V_nn, V_static, static_value_weight)
```

5. Pass `P_used` and `V_used` into MCTS.

6. Store training examples:

```text
state
MCTS visit distribution
final game result
static_policy_weight
static_value_weight
```

7. Train the NN only on:

```text
policy target = MCTS visits
value target  = final game result
```

8. Anneal static weights over training.

9. Periodically evaluate with:

```text
static_policy_weight = 0.0
static_value_weight  = 0.0
```

---

## Final Recommendation

Do this.

The clean version is:

```text
Use static eval as a temporary MCTS scaffold.
Do not train directly on static eval labels.
Decay static value quickly.
Decay static policy more slowly.
Avoid hard static pruning during training.
Evaluate final strength with static eval fully disabled.
```

This gives you useful early search without permanently chaining the neural network to the static evaluator.
