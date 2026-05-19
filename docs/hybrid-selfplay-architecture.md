# Hybrid CNN+Transformer Self-Play Plan

This repo is already surprisingly close to supporting a serious hidden-information learning stack:

- The engine is authoritative in Java.
- Player-specific observation copies already exist.
- External Python bots already receive legal actions plus a request-scoped forward model.
- The engine already supports hidden-information observation copies and forward-model search hooks, which is enough to build observation-tree baselines here.

## What Will Matter Most

Before spending time on bigger neural models, the main throughput bottlenecks are:

1. `Game.updateAssignedGameStates()` currently copies a fresh observation `GameState` for every player after each action.
2. `GameState.copy()` deep-copies the board, actors, diplomacy state, and recomputes legal actions.
3. Search paths repeatedly call `state.copy() -> advance() -> computePlayerActions() -> refreshVisibility()`.
4. The current external-bot path adds JSON serialization and stdin/stdout round trips on top of that.

For training, this means:

- Bigger models will help only after search/data throughput is under control.
- The first real win is to keep the engine authoritative but reduce how often we copy and serialize.
- A C++ search core is useful, but the biggest long-term gain is moving hot search closer to the Java forward model or exposing a faster binary bridge.

## Recommended Architecture

### Environment and search split

- Java remains the source of truth for rules, legality, fog-of-war, and determinization support.
- Python owns representation learning, replay, checkpointing, and experiment control.
- Search is policy-guided and observation-based.
- For now, the Python bot can search using the external forward-model session.
- The C++ helper is used for the inner PUCT child-selection path so there is already a fast search kernel to grow into.

### Network design

The starting model in `py/tribes_rl/model.py` uses three streams:

1. Spatial CNN stream
   - Encodes the visible board as a stack of channels.
   - Good at local terrain, borders, roads, city fronts, and tactical shape.

2. Token transformer stream
   - Encodes units, cities, and other structured entities as tokens.
   - Lets the model reason across long-range interactions, diplomacy, and distributed fronts better than a pure CNN.

3. Memory stream
   - A recurrent belief state tracks information that is not visible in the current observation.
   - An optional episodic cache stores compressed turn-to-turn summaries.
   - This is the right default for a partial-visibility strategy game.

Outputs:

- Policy logits over the current legal action set.
- Value estimate for the current information state.
- Updated memory state.

## Why Memory Is Needed Here

This game is not just locally partially observable. It also has:

- hidden enemy positions and movement histories
- capitals and city ownership knowledge that may be stale
- diplomacy state with delayed consequences
- exploration pressure where past observations matter
- multi-step turn structure where the best current micro-action depends on earlier choices in the same turn

A memoryless AlphaZero-style encoder would be leaving a lot on the table.

## DDO-Style Research Ideas

I read this request as likely referring to `Multi-Level Discovery of Deep Options` (DDO), which is a good fit here.

Useful DDO-style ideas for this project:

1. Learn macro-options instead of only atomic actions.
   - Examples: "pressure capital", "secure frontier city", "tech-rush economy", "escort capture", "naval expansion".
   - This can reduce search depth and stabilize self-play.

2. Use option-conditioned memory.
   - The memory state should depend on the current strategic mode or option.
   - A "defend capital" option should remember threats differently than an "expand fog" option.

3. Train a manager/worker policy split.
   - Manager picks a latent option every few decision steps.
   - Worker picks the legal engine action conditioned on that option.
   - This is especially attractive in a game with long turns and mixed tactical/economic actions.

4. Add an option head as an auxiliary target.
   - Even before full hierarchical RL, you can cluster search trajectories into strategic modes and train an auxiliary option classifier.
   - That often improves representation quality for imperfect-information play.

5. Build search around macro priors.
   - Instead of searching all legal actions uniformly, use the network to propose an option first, then bias action priors toward actions consistent with that option.

My recommendation is not to start with full DDO training immediately. Start with:

- recurrent belief state
- action-level policy/value learning
- optional auxiliary option head later

Then add macro-options once replay data is flowing.

## Search Recommendation

Short term:

- Use policy-guided observation-tree search.
- Keep it close to observation-tree search behavior when hidden information matters.
- For the Python stack, treat the current forward model as an observation-tree simulator.

Medium term:

- Add determinization sampling or public-state search in Java.
- ReBeL and Student of Games are the best conceptual references for imperfect-information self-play with search.

Important caveat:

- Vanilla AlphaZero-style MCTS is not theoretically sound in imperfect-information games.
- If this becomes strongly adversarial and strategically deceptive, the stronger direction is a ReBeL or SoG-style public-state / information-state method rather than just scaling observation MCTS.

## Good Near-Term Roadmap

1. Get policy/value + memory training working end to end.
2. Use the current external protocol for slow but correct self-play.
3. Replace the slowest search inner loops first.
4. Move more of the search/determinization path into Java if throughput becomes the blocker.
5. Add option-level or belief-model auxiliary learning once baseline self-play is stable.

## Papers Worth Borrowing From

- Student of Games: a strong reference for imperfect-information search + learning.
  - https://arxiv.org/abs/2112.03178
- ReBeL / Combining Deep Reinforcement Learning and Search for Imperfect-Information Games.
  - https://arxiv.org/abs/2007.13544
- The Act of Remembering: a useful memory-focused reference for POMDP-style RL.
  - https://arxiv.org/abs/2010.01753
- Multi-Level Discovery of Deep Options.
  - https://arxiv.org/abs/1703.08294

