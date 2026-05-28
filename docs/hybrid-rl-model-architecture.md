# Hybrid RL Model Architecture

This document describes the current neural model and runtime data flow implemented in `py/nn`, `py/search`, and `py/training`.

Default configuration from `ModelConfig` is derived from named schemas in `py/nn/encoding.py`:

| Item | Default |
|---|---:|
| Board input | `82 x 16 x 16` |
| Board tokens | `256` |
| Unit tokens | up to `256` |
| City tokens | up to `64` |
| Action candidates | up to `512` |
| Action feature width | `325` |
| Scalar features | `95`, represented as `95` scalar tokens plus 1 scalar summary token |
| Unit feature width | `41` |
| City feature width | `34` |
| CNN channels | `128` |
| Board residual blocks | `4` |
| Model width | `256` |
| Transformer layers | `6` |
| Attention heads | `8` |
| Explicit belief board planes in NN input | `0` |
| Coordinate-prior empty board planes | `0` |
| Trainable parameters | `7,124,610` |

## End-to-End Data Flow

```mermaid
flowchart LR
    JavaGame["Java game engine<br/>HeadlessPlay / GameState"]
    Payload["External bot JSON payload<br/>player_id, observation, legal actions"]
    Normalize["normalize_message<br/>canonical board, units, cities, tribes, actions"]
    Belief["BeliefTracker.annotate<br/>keeps diagnostics/search evidence"]
    Encode["encode_observation<br/>known facts only"]
    Model["HybridPolicyValueNet"]
    Snapshot["BeliefSnapshot<br/>frozen evidence for search"]
    MCTS["Native MCTS<br/>policy priors + value evaluator"]
    LeafBelief["Leaf payload annotation<br/>annotate_without_update"]
    Action["Selected legal action id<br/>plus rankedActionIds"]
    Replay["Replay shard<br/>StepRecord list"]
    Train["train_round<br/>policy/value optimization"]
    Checkpoint["Checkpoint<br/>model + optimizer"]

    JavaGame --> Payload --> Normalize --> Belief --> Encode --> Model
    Belief --> Snapshot --> MCTS
    Model --> MCTS --> Action --> JavaGame
    MCTS --> LeafBelief --> Encode
    Belief --> Replay
    MCTS --> Replay
    JavaGame --> Replay
    Replay --> Train --> Checkpoint
    Checkpoint --> Model
```

The model is not used as a direct one-shot actor during self-play. `HybridRLBot.choose_action` normalizes the incoming payload, annotates it with the Python belief tracker for search/diagnostics, evaluates the root state with the network, then uses native MCTS to search over legal actions.

The current NN encoding follows a strict known-fact rule: tensors may include observed facts and deterministic facts implied by observed evidence, but not hidden authoritative state, stale projections, or probabilistic guesses. Existing replay observations may still contain `observation["belief"]`, but `encode_observation` ignores belief planes and belief opponent scalars.

## Observation Encoding

```mermaid
flowchart TB
    Msg["Normalized message<br/>belief may be present but is not NN input"]

    subgraph Board["Board tensor: B x 82 x H x W"]
        TileBase["Base channels<br/>valid, explored, visible, x, y,<br/>tile unit/city presence, road"]
        Terrain["Terrain one-hot<br/>9 types"]
        Resource["Resource one-hot<br/>9 types"]
        Building["Building one-hot<br/>13 types"]
        VisibleOwners["Visible owner buckets<br/>unit owner, city owner,<br/>territory owner"]
        VisibleFacts["Visible unit/city facts<br/>unit status, hp fraction,<br/>city level"]
    end

    subgraph Entities["Entity tensors"]
        Units["unit_features<br/>B x 256 x 41<br/>id, position, owner, hp, stats,<br/>veteran/hidden/hint/status,<br/>unit-type one-hot"]
        UnitMask["unit_mask<br/>B x 256"]
        Cities["city_features<br/>B x 64 x 34<br/>id, position, owner, level,<br/>population, production, walls,<br/>building counts"]
        CityMask["city_mask<br/>B x 64"]
    end

    subgraph Actions["Legal action tensors"]
        ActionFeatures["action_features<br/>B x 512 x 325<br/>spatial deltas, source/target summaries,<br/>tile summary, native context,<br/>action/unit/building/resource/tech/level-up one-hot"]
        ActionMask["action_mask<br/>B x 512"]
        ActionIds["action_ids<br/>original Java action ids"]
    end

    Scalars["scalar_features<br/>B x 95<br/>game/own facts, own exact techs,<br/>relationships, visible/deduced opponent tech evidence"]

    Msg --> Board
    Msg --> Entities
    Msg --> Actions
    Msg --> Scalars
```

The encoder exposes every configured board channel through `BOARD_SCHEMA`; there are no empty coordinate-prior channels in the default schema. Some channels are explicitly reserved placeholders, including `reserved_hidden_authoritative`, but they are still named schema channels and are not treated as free coordinate-prior capacity. `HybridPolicyValueNet._with_empty_board_priors` remains as a compatibility hook, but `ModelConfig.use_empty_board_coordinate_priors` defaults to `False` and `POPULATED_BOARD_CHANNELS` covers every schema channel.

Board features include visible facts only:

- valid/explored/visible flags
- normalized coordinates
- terrain/resource/building one-hots
- road flag
- tile unit/city id presence
- visible unit owner buckets: own, enemy, neutral
- visible city owner buckets: own, enemy, neutral
- visible unit status and HP fraction
- visible city level
- visible territory ownership when present

Unit and city feature schemas are split. Unit rows use `UNIT_FEATURE_SCHEMA` with width `41`; city rows use `CITY_FEATURE_SCHEMA` with width `34`. The older shared `entity_feature_dim` remains in `ModelConfig` as a unit-width compatibility alias.

Action features use `ACTION_FEATURE_SCHEMA` with width `325`:

- source/target coordinates and deltas
- action intent flags
- source and target unit summaries
- source and target city summaries
- target tile summary
- native context: capture type, target-player relationship, target player id, diplomacy flags
- typed one-hots for action type, requested unit/building/resource/tech, and level-up bonus

Scalar features use `SCALAR_FEATURE_SCHEMA` with width `95`:

- game metadata and own tribe facts
- own exact researched tech flags from the player-specific observation
- monument status counts
- visible relationship summaries and diplomacy-offer flags
- known opponent tech evidence summary

Known opponent tech evidence is derived only from visible enemy units/buildings and deterministic tech prerequisites. For example, a visible enemy `KNIGHT` marks `CHIVALRY`, `FREE_SPIRIT`, and `RIDING` as known evidence. Opponent `researched_tech_ids` are not encoded unless the same tech is visible/deducible from player-specific observation evidence.

## Belief Layer

`BeliefTracker` is still a Python-only helper used by bot flow and native search. It lives per bot episode/player and resets in `HybridRLBot.reset_episode`.

Its outputs are split by use:

| Output | Current use |
|---|---|
| Visible/deduced tech evidence | Recomputed by `encode_observation` from current visible units/cities for NN scalars |
| Possible hidden capitals, stale last-seen units, hidden cloak candidates, threat projections | Kept in `observation["belief"]` for diagnostics/search plumbing, not encoded into NN tensors |

`BeliefSnapshot` is a pure search-time view. Native MCTS receives a snapshot after root annotation. Leaf payloads are annotated with `annotate_without_update`, so hypothetical search states get the same search-side context without mutating episode evidence.

## Neural Architecture

```mermaid
flowchart TB
    BoardIn["Board<br/>B x 82 x H x W"]
    UnitIn["Units<br/>B x 256 x 41 + mask"]
    CityIn["Cities<br/>B x 64 x 34 + mask"]
    ActionIn["Actions<br/>B x 512 x 325 + mask"]
    ScalarIn["Scalars<br/>B x 95"]

    subgraph BoardEncoder["Board encoder"]
        Conv0["3x3 Conv<br/>82 -> 128"]
        GN0["GroupNorm + GELU"]
        Res1["ResidualConvBlock<br/>3x3 Conv, GN, GELU,<br/>3x3 Conv, GN, skip"]
        ResN["4 total ResidualConvBlocks<br/>configured by board_res_blocks"]
        Conv1["1x1 Conv<br/>128 -> 256 + GELU"]
        Flatten["Flatten spatial<br/>B x 256 x 256"]
    end

    UnitProj["Linear unit projection<br/>41 -> 256"]
    CityProj["Linear city projection<br/>34 -> 256"]
    ScalarProj["Scalar token encoder<br/>per-scalar Linear 1 -> 256<br/>+ learned scalar-index embedding"]
    ScalarSummary["Scalar summary token<br/>Linear 95 -> 256 -> 256"]
    ActProj["Linear action projection<br/>325 -> 256"]

    TypeEmb["Token type embeddings<br/>scalar, scalar summary, board, unit, city, CLS"]
    Cat["Token concat + dropout<br/>default B x 673 x 256"]
    Mask["Padding mask<br/>invalid unit and city tokens"]
    Core["TransformerEncoder<br/>6 layers, 8 heads,<br/>FFN width 1024, GELU"]
    Pooled["Pooled state<br/>CLS token output"]
    ActionAttn["Action cross-attention<br/>queries=legal actions<br/>keys/values=context latent"]
    ValueActionPool["Value action context<br/>mean pool + max pool + attention pool"]
    Policy["Policy head<br/>Linear 256 -> 256 -> 1<br/>masked logits B x 512"]
    Value["Value head<br/>Linear 1024 -> 256 -> 1<br/>tanh value B"]

    BoardIn --> Conv0 --> GN0 --> Res1 --> ResN --> Conv1 --> Flatten
    UnitIn --> UnitProj
    CityIn --> CityProj
    ScalarIn --> ScalarProj
    ScalarIn --> ScalarSummary
    ActionIn --> ActProj

    Flatten --> TypeEmb
    UnitProj --> TypeEmb
    CityProj --> TypeEmb
    ScalarProj --> TypeEmb
    ScalarSummary --> TypeEmb
    TypeEmb --> Cat
    Cat --> Core
    Mask --> Core
    Core --> Pooled
    Core --> ActionAttn
    ActProj --> ActionAttn
    ActionAttn --> Policy
    ActionAttn --> ValueActionPool
    Pooled --> ValueActionPool
    Pooled --> Value
    ValueActionPool --> Value
```

Token order in the Transformer context:

```text
[CLS] [scalar summary] [95 scalar tokens] [board tiles] [unit tokens] [city tokens]
```

Token type ids:

| Token type id | Meaning |
|---:|---|
| `0` | scalar |
| `1` | scalar summary |
| `3` | board |
| `4` | unit |
| `5` | city |
| `6` | CLS |

Token type id `2` is currently unused.

Default context token count:

```text
1 CLS + 1 scalar summary + 95 scalar + 256 board + 256 units + 64 cities = 673 tokens
```

The policy path uses action features as queries into the Transformer latent context, then applies a per-action MLP and masks invalid actions. The value path now also consumes legal-action context: it concatenates the pooled state with masked mean-pool, masked max-pool, and pooled-state attention over action tokens, producing a `4 * d_model` value input before the final tanh head.

## Action Selection with MCTS

```mermaid
flowchart TB
    Current["Current game payload"]
    RootBelief["BeliefTracker.annotate<br/>search/diagnostic state updated"]
    EncRoot["Encode root observation<br/>known facts only"]
    RootForward["Model forward at root<br/>policy logits, value"]
    Snapshot["BeliefSnapshot<br/>frozen evidence"]
    TopK["Keep top-k actions by prior<br/>default top_k_actions=64"]
    Tree["NativeMCTS tree<br/>PUCT c=1.5<br/>Dirichlet root noise"]
    Select["Select leaf paths<br/>batched up to search.batch_size"]
    LeafMsgs["Create leaf eval messages"]
    LeafBelief["BeliefSnapshot.annotate_without_update"]
    LeafEval["Model evaluates leaves<br/>known-fact encoder ignores uncertainty"]
    Backup["Expand and back up value"]
    Visits["Root visit distribution"]
    Sample["Sample or argmax action<br/>default sample_action=True"]
    Record["StepRecord<br/>observation, legal actions,<br/>visit_target, root_value"]

    Current --> RootBelief --> EncRoot --> RootForward --> TopK --> Tree
    RootBelief --> Snapshot
    Snapshot --> LeafBelief
    Tree --> Select --> LeafMsgs --> LeafBelief --> LeafEval --> Backup --> Tree
    Tree --> Visits --> Sample
    RootBelief --> Record
    RootForward --> Record
    Visits --> Record
    Sample --> Record
```

Current native search uses the neural model as a batched evaluator. Leaf observations are re-annotated from the frozen belief snapshot before encoding, so search does not update monotonic episode evidence. Since the encoder is known-fact only, search-side uncertainty annotations do not change NN tensors.

## Training Loop

```mermaid
flowchart LR
    SelfPlay["Self-play games<br/>Java HeadlessPlay + external bots"]
    Episodes["Episode records"]
    Returns["compute_returns<br/>terminal reward discounted by gamma"]
    ReplaySample["ReplayStore.sample<br/>random records from shards"]
    Augment["Optional symmetry augmentation<br/>spatial observation/action transforms"]
    Collate["collate_batch<br/>known-fact encoded tensors<br/>policy targets + value targets"]
    Forward["Model forward"]
    Loss["Loss<br/>policy CE against visit_target<br/>+ 0.5 value MSE"]
    Step["AdamW step<br/>grad clip 1.0"]
    Save["Save latest + iteration checkpoint<br/>append metrics.csv"]

    SelfPlay --> Episodes --> Returns --> ReplaySample --> Augment --> Collate --> Forward --> Loss --> Step --> Save
```

Loss definition in the current implementation:

```text
loss = policy_loss_weight * policy_cross_entropy
     + value_loss_weight * value_mse
```

## Analysis Points / Modification Hooks

| Area | Current behavior | Why it matters |
|---|---|---|
| Known-fact encoding | NN tensors include observed facts plus deterministic implications only | Prevents hidden authoritative state and probabilistic guesses from leaking into model inputs. |
| Belief uncertainty | Kept outside NN tensors in `observation["belief"]` | Diagnostics/search plumbing can evolve without silently changing model inputs. |
| Opponent tech | Visible enemy units/buildings imply required techs and deterministic prerequisites | Gives useful factual evidence without encoding exact hidden opponent tech trees. |
| Recurrent memory | Removed from active architecture | Checkpoints from memory-based models are not shape-compatible by design. |
| Board ownership/control | Visible owner buckets encode unit/city/territory ownership | Spatial ownership is available to the CNN only when present in observation evidence. |
| Action bottleneck | Only legal actions are action-query tokens; default cap is 512 | If Java generates more than 512 legal actions, later actions are truncated and cannot be selected. |
| Entity truncation | Units cap 256, cities cap 64 | Large or unusual maps should still be checked. |
| Dense board channels | `board_channels=82`, all channels named by schema | There are no spare coordinate-prior channels in the default model. |
| Value target | Mostly terminal discounted return, shaped reward defaults to `0.0` | Sparse value signal may be slow in long games unless search/value quality is already good. |
| Masking | Unit/city masks are applied in Transformer; action mask after policy head | Invalid entities/actions are masked, board tiles are dense. |

## Source Map

| Concern | File |
|---|---|
| Model layers and tensor routing | `py/nn/model.py` |
| Explicit belief construction | `py/nn/belief.py` |
| Observation/action encoding and schemas | `py/nn/encoding.py` |
| Bot inference, replay recording | `py/nn/bot_agent.py` |
| Native MCTS wrapper | `py/search/native/mcts.py` |
| Replay records and return targets | `py/training/replay.py` |
| Training loop and losses | `py/training/train.py` |
| Symmetry augmentation | `py/nn/augmentation.py` |
| Hyperparameters | `py/search/config.py` |
