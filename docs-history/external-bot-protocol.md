# External Bot Protocol v1

This repository now supports an external-bot interface for non-Java agents such as Python bots.

## Goal

Keep the Java engine authoritative while letting external processes choose moves and, when needed,
query request-scoped forward-model simulations.

- The engine still computes legal actions.
- The bot receives an observation snapshot and a list of legal action ids.
- The bot replies with one chosen action id.
- Bots that want tree search can also ask the engine to expand simulated successor states built from
  the same observed `GameState` copy Java agents receive.

## Transport

- The Java engine starts the bot as a child process.
- Messages are sent over `stdin`/`stdout`.
- Each message is one JSON object per line.

## Incoming Message Types

### `action_request`

Sent every time the bot must choose one action.

Top-level fields:

- `type`: `"action_request"`
- `protocol_version`: currently `1`
- `request_id`: turn-local sequence number for this bot process
- `player_id`: the tribe id controlled by this bot
- `time_remaining_ms`: remaining decision time reported by the engine
- `observation`: visible game snapshot for this bot
- `actions`: legal actions for the current decision point
- `forward_model`: optional descriptor for request-scoped simulation support

Each action includes:

- `id`: stable for this request only, for example `A0`
- `type`: engine action enum name such as `MOVE` or `END_TURN`
- `description`: human-readable action string from the engine
- extra structured fields when available, such as `unit_id`, `city_id`, `destination`, `technology`, and so on

If `forward_model.enabled` is `true`, the request also includes:

- `root_state_id`: request-scoped state id for the current decision root, usually `root`
- `request_scoped`: successor states live only until the bot answers this `action_request`
- `non_mutating`: simulation calls never mutate an existing state id in place
- `commands`: currently `inspect`, `step`, and `release`

### `game_over`

Sent once at the end of the game.

- `type`: `"game_over"`
- `protocol_version`: currently `1`
- `player_id`
- `reward`
- `final_state`

## Bot Response

For each `action_request`, the bot should print one JSON line:

```json
{"actionId":"A3"}
```

The engine also accepts:

```json
{"actionIndex":3}
```

but `actionId` is the preferred form.

Bots may also stay in the same stdin/stdout exchange loop and send forward-model commands before
eventually replying with an action choice.

## Forward-Model Command Loop

When a bot wants to search, it can send a JSON line with `type: "forward_model"` instead of an
immediate move choice.

### `inspect`

Request:

```json
{"type":"forward_model","command":"inspect","state_id":"root"}
```

Response:

```json
{
  "type":"forward_model_result",
  "command":"inspect",
  "player_id":0,
  "state":{
    "state_id":"root",
    "observation":{ "...": "..." },
    "actions":[{ "...": "..." }],
    "is_terminal":false,
    "active_player_id":0
  }
}
```

### `step`

`step` clones the source state, applies one legal action, and returns a new derived state id.
The source state remains unchanged, so bots can branch freely for tree search.

Request:

```json
{"type":"forward_model","command":"step","state_id":"root","action_id":"A3"}
```

Response:

```json
{
  "type":"forward_model_result",
  "command":"step",
  "player_id":0,
  "source_state_id":"root",
  "action_id":"A3",
  "action_index":3,
  "state":{
    "state_id":"s1",
    "observation":{ "...": "..." },
    "actions":[{ "...": "..." }],
    "is_terminal":false,
    "active_player_id":0
  }
}
```

### `release`

Bots can optionally free derived states they no longer need:

```json
{"type":"forward_model","command":"release","state_ids":["s1","s2"]}
```

The engine replies with `released_state_ids`. Releasing the root state is ignored.

## Observation Notes

The observation and forward-model states are produced from the player-specific `GameState` copy already
used by in-process Java agents. That means hidden information remains hidden before serialization, and
tree search runs over the bot's observed model of the world rather than hidden authoritative state.

The current v1 observation includes:

- turn metadata such as `tick`, `game_mode`, `active_player_id`
- visible tribe summaries
- visible cities
- visible units
- board tiles with `explored` and `visible` flags
- visible relationships
- ranking ids

## Example `play.json`

```json
{
  "Run Mode": "PlayLG",
  "Game Mode": "Capitals",
  "Players": ["External", "External"],
  "External Commands": [
    ["python", "py/bots/random_bot.py"],
    null
  ],
  "Tribes": ["Xin Xi", "Imperius"],
  "Verbose": false,
  "Rollouts": false,
  "Force End": false,
  "Population Size": 1,
  "Progressive Bias": true,
  "Pruning": true,
  "K init mult": 0.5,
  "T mult": 2.0,
  "A mult": 1.5,
  "B": 1.3,
  "Game Seed": "-1",
  "Agents Seed": "-1",
  "Level Seed": "-1"
}
```

## Sample Bot

A minimal Python example lives at [random_bot.py](/C:/Users/Umair/OneDrive/Desktop/Work/Self_Projects/Poly/Tribes/py/bots/random_bot.py:1).

A more strategy-shaped Python example is available at [simple_bot.py](/C:/Users/Umair/OneDrive/Desktop/Work/Self_Projects/Tribes_MCTS/py/bots/simple_bot.py:1).

