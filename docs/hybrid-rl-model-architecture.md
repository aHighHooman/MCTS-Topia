# Hybrid RL Model Architecture

This document describes the current neural model and runtime data flow implemented in `py/tribes_rl`.

Default configuration from `ModelConfig`:

| Item | Default |
|---|---:|
| Board input | `96 x 16 x 16` |
| Board tokens | `256` |
| Unit tokens | up to `256` |
| City tokens | up to `64` |
| Action candidates | up to `512` |
| Action feature width | `85` |
| Scalar features | `88`, represented as `88` scalar tokens |
| Entity feature width | `64` |
| Model width | `160` |
| Transformer layers | `5` |
| Attention heads | `8` |
| Explicit belief board planes | `15` channels |
| Coordinate-prior empty board planes | `43` channels |
| Trainable parameters | `2,210,250` |

## End-to-End Data Flow

```mermaid
flowchart LR
    JavaGame["Java game engine<br/>HeadlessPlay / GameState"]
    Payload["External bot JSON payload<br/>player_id, observation, legal actions"]
    Normalize["normalize_message<br/>canonical board, units, cities, tribes, actions"]
    Belief["BeliefTracker.annotate<br/>adds observation.belief"]
    Encode["encode_observation"]
    Model["HybridPolicyValueNet"]
    Snapshot["BeliefSnapshot<br/>frozen evidence for search"]
    MCTS["Native MCTS<br/>policy priors + value evaluator"]
    LeafBelief["Leaf payload annotation<br/>annotate_without_update"]
    Action["Selected legal action id<br/>plus rankedActionIds"]
    Replay["Replay shard<br/>StepRecord list with observation.belief"]
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

The model is not used as a direct one-shot actor during self-play. `HybridRLBot.choose_action` normalizes the incoming payload, annotates it with explicit belief features, evaluates the root state with the network, then uses native MCTS to search over legal actions. Replay stores the annotated observation, so training consumes the same belief representation used during self-play. Older replay records without `observation["belief"]` remain valid and encode the belief inputs as zeros.

## Observation Encoding

```mermaid
flowchart TB
    Msg["Normalized, belief-annotated message"]

    subgraph Board["Board tensor: B x 96 x H x W"]
        TileBase["Base channels<br/>valid, explored, x, y,<br/>unit present, city present, road<br/>old visible/id channels left empty"]
        Terrain["Terrain one-hot<br/>9 types"]
        Resource["Resource one-hot<br/>9 types"]
        Building["Building one-hot<br/>13 types"]
        BeliefPlanes["Belief planes<br/>channels 41..55<br/>15 spatial uncertainty/control planes"]
        CoordPriors["Coordinate priors<br/>43 remaining empty channels<br/>inserted inside model before CNN"]
    end

    subgraph Entities["Entity tensors"]
        Units["unit_features<br/>B x 256 x 64<br/>id, position, owner, hp, stats,<br/>veteran/hidden/hint/status,<br/>unit-type one-hot"]
        UnitMask["unit_mask<br/>B x 256"]
        Cities["city_features<br/>B x 64 x 64<br/>id, position, owner, level,<br/>population, production, walls,<br/>building counts"]
        CityMask["city_mask<br/>B x 64"]
    end

    subgraph Actions["Legal action tensors"]
        ActionFeatures["action_features<br/>B x 512 x 85<br/>index, actor ids, target ids,<br/>target position flags,<br/>action/unit/building/resource one-hot"]
        ActionMask["action_mask<br/>B x 512"]
        ActionIds["action_ids<br/>original Java action ids"]
    end

    Scalars["scalar_features<br/>B x 88<br/>first 18 legacy scalars<br/>+ 70 opponent belief scalars"]

    Msg --> Board
    Msg --> Entities
    Msg --> Actions
    Msg --> Scalars
```

The encoder explicitly fills 53 of the configured 96 board channels: 38 pre-existing board channels plus 15 belief channels. `HybridPolicyValueNet` fills only the remaining 43 channels with deterministic coordinate-prior planes. `POPULATED_BOARD_CHANNELS` includes channels `41..55`, so coordinate priors never overwrite belief planes.

Belief board channels:

| Channel | Plane |
|---:|---|
| `41` | `own_unit_presence` |
| `42` | `enemy_unit_presence` |
| `43` | `own_city_center` |
| `44` | `enemy_city_center` |
| `45` | `own_territory` |
| `46` | `enemy_territory` |
| `47` | `unexplored` |
| `48` | `frontier_unexplored` |
| `49` | `deep_unexplored` |
| `50` | `possible_enemy_capital` |
| `51` | `hidden_cloak_hint_source` |
| `52` | `possible_hidden_cloak` |
| `53` | `known_enemy_threat` |
| `54` | `possible_hidden_cloak_threat` |
| `55` | `known_enemy_city_zone` |

Scalar features preserve the previous first 18 values exactly. The appended 70 features are 10 scalar belief features per opponent for up to 7 opponents, ordered by ascending tribe id excluding the acting player:

```text
opponent score
my score minus opponent score
visible enemy city count
visible enemy unit count
observed enemy capital flag
inferred minimum tech count
inferred military tech flag
inferred economy tech flag
inferred naval tech flag
inferred strategy/diplomacy flag
```

## Belief Layer

`BeliefTracker` is a Python-only, conservative belief feature builder. It lives per bot episode/player and resets in `HybridRLBot.reset_episode`.

Persistent evidence is deliberately narrow:

- inferred opponent tech evidence
- observed/known opponent capital evidence

The tracker does not maintain recurrent neural memory and does not keep long-term hidden-unit position guesses. Hidden cloak/dinghy uncertainty is recomputed from current hint payloads.

`BeliefSnapshot` is a pure search-time view. Native MCTS receives a snapshot after root annotation. Leaf payloads are annotated with `annotate_without_update`, so hypothetical search states get the same belief context without mutating episode evidence.

## Neural Architecture

```mermaid
flowchart TB
    BoardIn["Board<br/>B x 96 x H x W"]
    UnitIn["Units<br/>B x 256 x 64 + mask"]
    CityIn["Cities<br/>B x 64 x 64 + mask"]
    ActionIn["Actions<br/>B x 512 x 85 + mask"]
    ScalarIn["Scalars<br/>B x 88"]

    subgraph BoardEncoder["Board encoder"]
        Priors["Fill empty channels only<br/>coordinate-prior planes"]
        Conv0["3x3 Conv<br/>96 -> 96"]
        GN0["GroupNorm + GELU"]
        Res1["ResidualConvBlock<br/>3x3 Conv, GN, GELU,<br/>3x3 Conv, GN, skip"]
        Res2["ResidualConvBlock"]
        Conv1["1x1 Conv<br/>96 -> 160 + GELU"]
        Flatten["Flatten spatial<br/>B x 256 x 160"]
    end

    UnitProj["Linear unit projection<br/>64 -> 160"]
    CityProj["Linear city projection<br/>64 -> 160"]
    ScalarProj["Scalar token encoder<br/>per-scalar Linear 1 -> 160<br/>+ learned scalar-index embedding"]
    ActProj["Linear action projection<br/>85 -> 160"]

    TypeEmb["Token type embeddings<br/>scalar, board, unit, city, CLS"]
    Cat["Token concat + dropout<br/>default B x 665 x 160"]
    Mask["Padding mask<br/>invalid unit and city tokens"]
    Core["TransformerEncoder<br/>5 layers, 8 heads,<br/>FFN width 640, GELU"]
    Pooled["Pooled state<br/>CLS token output"]
    ActionAttn["Action cross-attention<br/>queries=legal actions<br/>keys/values=context latent"]
    Policy["Policy head<br/>Linear 160 -> 160 -> 1<br/>masked logits B x 512"]
    Value["Value head<br/>Linear 160 -> 160 -> 1<br/>tanh value B"]

    BoardIn --> Priors --> Conv0 --> GN0 --> Res1 --> Res2 --> Conv1 --> Flatten
    UnitIn --> UnitProj
    CityIn --> CityProj
    ScalarIn --> ScalarProj
    ActionIn --> ActProj

    Flatten --> TypeEmb
    UnitProj --> TypeEmb
    CityProj --> TypeEmb
    ScalarProj --> TypeEmb
    TypeEmb --> Cat
    Cat --> Core
    Mask --> Core
    Core --> Pooled
    Pooled --> Value
    Core --> ActionAttn
    ActProj --> ActionAttn
    ActionAttn --> Policy
```

Token order in the Transformer context:

```text
[CLS] [88 scalar tokens] [board tiles] [unit tokens] [city tokens]
```

Token type ids:

| Token type id | Meaning |
|---:|---|
| `0` | scalar |
| `3` | board |
| `4` | unit |
| `5` | city |
| `6` | CLS |

Token type ids `1` and `2` are currently unused.

Default context token count:

```text
1 CLS + 88 scalar + 256 board + 256 units + 64 cities = 665 tokens
```

## Action Selection with MCTS

```mermaid
flowchart TB
    Current["Current game payload"]
    RootBelief["BeliefTracker.annotate<br/>persistent evidence updated"]
    EncRoot["Encode root observation<br/>belief planes + scalars"]
    RootForward["Model forward at root<br/>policy logits, value"]
    Snapshot["BeliefSnapshot<br/>frozen evidence"]
    TopK["Keep top-k actions by prior<br/>default top_k_actions=128"]
    Tree["NativeMCTS tree<br/>PUCT c=1.5<br/>Dirichlet root noise"]
    Select["Select leaf paths<br/>batched up to search.batch_size"]
    LeafMsgs["Create leaf eval messages"]
    LeafBelief["BeliefSnapshot.annotate_without_update"]
    LeafEval["Model evaluates annotated leaves<br/>priors + value"]
    Backup["Expand and back up value"]
    Visits["Root visit distribution"]
    Sample["Sample or argmax action<br/>default sample_action=True"]
    Record["StepRecord<br/>observation with belief,<br/>legal actions, visit_target, root_value"]

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

Current native search uses the neural model as a batched evaluator. Leaf observations are re-annotated from the frozen belief snapshot before encoding, so search does not update monotonic episode evidence. The C++ transition support determines how much of the state advances for each action; unsupported transitions become terminal leaf payloads.

## Training Loop

```mermaid
flowchart LR
    SelfPlay["Self-play games<br/>Java HeadlessPlay + external bots"]
    Episodes["Episode records<br/>observations include belief"]
    Returns["compute_returns<br/>terminal reward discounted by gamma"]
    ReplaySample["ReplayStore.sample<br/>random records from shards"]
    Augment["Optional symmetry augmentation<br/>belief planes transform spatially<br/>opponent scalars unchanged"]
    Collate["collate_batch<br/>encoded tensors<br/>policy targets + value targets"]
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
| Explicit belief | Python builder writes `observation["belief"]` | Representation captures uncertainty without Java protocol changes. |
| Recurrent memory | Removed from active architecture | Checkpoints from memory-based models are not shape-compatible by design. |
| Board ownership/control | Dense belief planes encode own/enemy units, city centers, territory, city zones | Spatial ownership is now available to the CNN instead of only entity tokens. |
| Hidden units | Current cloak hints produce local candidate/threat planes | No long-term last-seen normal-unit tracking is implemented. |
| Opponent economy/tech | 70 scalar belief features appended after legacy scalars | Evidence is monotonic per episode and ordered by tribe id. |
| Search semantics | Leaf payloads are annotated without mutating tracker state | Avoids search hallucinations changing real episode evidence. |
| Action bottleneck | Only legal actions are action-query tokens; default cap is 512 | If Java generates more than 512 legal actions, later actions are truncated and cannot be selected. |
| Entity truncation | Units cap 256, cities cap 64 | Large or unusual maps should still be checked. |
| Dense board channels | `board_channels=96`, with 43 coordinate-prior empty channels | Spare channels are useful spatial priors and do not overwrite populated or belief channels. |
| Value target | Mostly terminal discounted return, shaped reward defaults to `0.0` | Sparse value signal may be slow in long games unless search/value quality is already good. |
| Masking | Unit/city masks are applied in Transformer; action mask after policy head | Invalid entities/actions are masked, board tiles are dense. |

## Source Map

| Concern | File |
|---|---|
| Model layers and tensor routing | `py/tribes_rl/model.py` |
| Explicit belief construction | `py/tribes_rl/belief.py` |
| Observation/action encoding | `py/tribes_rl/encoding.py` |
| Bot inference, replay recording | `py/tribes_rl/bot_agent.py` |
| Native MCTS wrapper | `py/tribes_rl/native/mcts.py` |
| Replay records and return targets | `py/tribes_rl/replay.py` |
| Training loop and losses | `py/tribes_rl/train.py` |
| Symmetry augmentation | `py/tribes_rl/augmentation.py` |
| Hyperparameters | `py/tribes_rl/config.py` |
