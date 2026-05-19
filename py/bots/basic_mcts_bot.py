import json
import math
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from strong_external_bot_v2 import ObservationView, action_score, tie_break_key


RNG = random.Random(29)
ROOT_ACTION_LIMIT = 12
ROLLOUT_ACTION_LIMIT = 4
ROLLOUT_DEPTH = 5
MIN_ITERATIONS = 24
MAX_ITERATIONS = 96
EXPLORATION = 1.35


def message_for_state(player_id: int, state_payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "player_id": player_id,
        "observation": state_payload["observation"],
    }


def static_state_score(player_id: int, state_payload: Dict[str, Any]) -> float:
    view = ObservationView(message_for_state(player_id, state_payload))
    my_tribe = view.my_tribe()
    enemy_scores = [
        float(tribe.get("score", 0))
        for tribe_id, tribe in view.tribes.items()
        if int(tribe_id) != view.player_id
    ]
    material = sum(view.unit_value(unit) for unit in view.my_units) - sum(view.unit_value(unit) for unit in view.enemy_units)
    city_balance = 7.5 * (len(view.my_cities) - len(view.enemy_cities))
    economy = 0.45 * float(my_tribe.get("stars", 0)) + 0.14 * float(my_tribe.get("score", 0))
    map_control = 0.18 * len(view.visible_tiles) + 2.1 * len(view.villages) + 2.4 * len(view.ruins)
    capital_safety = 0.0
    capital = view.my_capital()
    if capital:
        capital_safety -= min(7.0, view.city_threat(capital)) * 1.2
    city_threat = sum(min(5.5, view.city_threat(city)) for city in view.my_cities)
    scoreboard = 0.12 * (float(my_tribe.get("score", 0)) - (max(enemy_scores) if enemy_scores else 0.0))
    return material + city_balance + economy + map_control + scoreboard + capital_safety - city_threat


def rank_actions_for_player(state_payload: Dict[str, Any], player_id: int, limit: int) -> List[Tuple[float, Dict[str, Any]]]:
    message = message_for_state(player_id, state_payload)
    view = ObservationView(message)
    mode = view.strategic_mode()
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for action in state_payload.get("actions", []):
        score = action_score(action, view, mode)
        scored.append((score, action))
    scored.sort(key=lambda item: tie_break_key(item[1], item[0]), reverse=True)
    return scored[:limit]


def choose_greedy(message: Dict[str, Any]) -> Dict[str, Optional[str]]:
    player_id = int(message["player_id"])
    root_state = {
        "observation": message["observation"],
        "actions": message.get("actions", []),
    }
    ranked = rank_actions_for_player(root_state, player_id, len(root_state["actions"]))
    if not ranked:
        return {"actionId": None}
    return {"actionId": str(ranked[0][1]["id"])}


class ForwardModelClient:
    def command(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        outgoing = dict(payload)
        outgoing["type"] = "forward_model"
        sys.stdout.write(json.dumps(outgoing) + "\n")
        sys.stdout.flush()

        while True:
            raw_line = sys.stdin.readline()
            if not raw_line:
                return None
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                return json.loads(raw_line)
            except json.JSONDecodeError:
                continue

    def step(self, state_id: str, action_id: str) -> Optional[Dict[str, Any]]:
        response = self.command({
            "command": "step",
            "state_id": state_id,
            "action_id": action_id,
        })
        if not response or response.get("type") != "forward_model_result":
            return None
        return response.get("state")

    def release(self, state_ids: List[str]) -> None:
        if not state_ids:
            return
        self.command({
            "command": "release",
            "state_ids": state_ids,
        })


@dataclass
class SearchNode:
    state_id: str
    state_payload: Dict[str, Any]
    root_action_id: Optional[str]
    parent: Optional["SearchNode"] = None
    action_from_parent: Optional[str] = None
    children: List["SearchNode"] = field(default_factory=list)
    visits: int = 0
    total_value: float = 0.0
    unexpanded_actions: List[Dict[str, Any]] = field(default_factory=list)

    def mean_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_value / float(self.visits)

    def is_terminal(self) -> bool:
        return bool(self.state_payload.get("is_terminal")) or not self.state_payload.get("actions")

    def active_player_id(self, fallback_player_id: int) -> int:
        return int(self.state_payload.get("active_player_id", fallback_player_id))


def build_root_node(message: Dict[str, Any], player_id: int) -> SearchNode:
    root_payload = {
        "state_id": str((message.get("forward_model") or {}).get("root_state_id", "root")),
        "observation": message["observation"],
        "actions": list(message.get("actions", [])),
        "is_terminal": not message.get("actions"),
        "active_player_id": int(message["observation"].get("active_player_id", player_id)),
    }
    ranked_root = rank_actions_for_player(root_payload, player_id, ROOT_ACTION_LIMIT)
    return SearchNode(
        state_id=str(root_payload["state_id"]),
        state_payload=root_payload,
        root_action_id=None,
        unexpanded_actions=[action for _, action in ranked_root],
    )


def iteration_budget(time_remaining_ms: int) -> int:
    if time_remaining_ms <= 0:
        return MIN_ITERATIONS
    scaled = time_remaining_ms // 14
    return max(MIN_ITERATIONS, min(MAX_ITERATIONS, int(scaled)))


def rollout_policy_action(state_payload: Dict[str, Any], root_player_id: int) -> Optional[Dict[str, Any]]:
    active_player = int(state_payload.get("active_player_id", root_player_id))
    ranked = rank_actions_for_player(state_payload, active_player, ROLLOUT_ACTION_LIMIT)
    if not ranked:
        return None

    if active_player == root_player_id:
        top_score = ranked[0][0]
        shortlist = [action for score, action in ranked if score >= top_score - 1.2]
    else:
        top_score = ranked[0][0]
        shortlist = [action for score, action in ranked if score >= top_score - 0.8]

    return RNG.choice(shortlist) if shortlist else ranked[0][1]


def rollout_value(client: ForwardModelClient, node: SearchNode, root_player_id: int,
                  allocated_state_ids: List[str]) -> float:
    current_state = node.state_payload
    depth = 0
    while depth < ROLLOUT_DEPTH and not current_state.get("is_terminal"):
        action = rollout_policy_action(current_state, root_player_id)
        if action is None:
            break
        next_state = client.step(str(current_state["state_id"]), str(action["id"]))
        if not next_state:
            break
        allocated_state_ids.append(str(next_state["state_id"]))
        current_state = next_state
        depth += 1

    return static_state_score(root_player_id, current_state)


def select_child(node: SearchNode, root_player_id: int) -> SearchNode:
    parent_log = math.log(max(1, node.visits))
    active_player = node.active_player_id(root_player_id)
    maximizing = active_player == root_player_id

    best_child = node.children[0]
    best_value = float("-inf")
    for child in node.children:
        if child.visits == 0:
            uct_value = float("inf")
        else:
            exploitation = child.mean_value()
            if not maximizing:
                exploitation = -exploitation
            exploration = EXPLORATION * math.sqrt(parent_log / float(child.visits))
            uct_value = exploitation + exploration

        if uct_value > best_value:
            best_value = uct_value
            best_child = child
    return best_child


def expand_node(client: ForwardModelClient, node: SearchNode, root_player_id: int,
                allocated_state_ids: List[str]) -> SearchNode:
    if not node.unexpanded_actions:
        return node

    action = node.unexpanded_actions.pop(0)
    child_state = client.step(node.state_id, str(action["id"]))
    if not child_state:
        return node

    allocated_state_ids.append(str(child_state["state_id"]))
    ranked_child = rank_actions_for_player(child_state, child_state.get("active_player_id", root_player_id), ROOT_ACTION_LIMIT)
    child = SearchNode(
        state_id=str(child_state["state_id"]),
        state_payload=child_state,
        root_action_id=node.root_action_id or str(action["id"]),
        parent=node,
        action_from_parent=str(action["id"]),
        unexpanded_actions=[candidate for _, candidate in ranked_child],
    )
    node.children.append(child)
    return child


def run_mcts(message: Dict[str, Any]) -> Dict[str, Optional[str]]:
    fm_info = message.get("forward_model") or {}
    if not fm_info.get("enabled"):
        return choose_greedy(message)

    player_id = int(message["player_id"])
    root = build_root_node(message, player_id)
    if not root.unexpanded_actions and not root.state_payload.get("actions"):
        return {"actionId": None}

    client = ForwardModelClient()
    allocated_state_ids: List[str] = []
    iterations = iteration_budget(int(message.get("time_remaining_ms", 0)))

    try:
        for _ in range(iterations):
            node = root
            while not node.is_terminal() and not node.unexpanded_actions and node.children:
                node = select_child(node, player_id)

            if not node.is_terminal() and node.unexpanded_actions:
                node = expand_node(client, node, player_id, allocated_state_ids)

            value = rollout_value(client, node, player_id, allocated_state_ids)

            current: Optional[SearchNode] = node
            while current is not None:
                current.visits += 1
                current.total_value += value
                current = current.parent
    finally:
        client.release(allocated_state_ids)

    root_children = [child for child in root.children if child.root_action_id is not None]
    if not root_children:
        return choose_greedy(message)

    best_child = max(
        root_children,
        key=lambda child: (child.visits, child.mean_value()),
    )
    return {"actionId": best_child.root_action_id}


def main() -> None:
    while True:
        raw_line = sys.stdin.readline()
        if not raw_line:
            break
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        try:
            message = json.loads(raw_line)
        except json.JSONDecodeError:
            continue

        if message.get("type") == "action_request":
            response = run_mcts(message)
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
        elif message.get("type") == "game_over":
            break


if __name__ == "__main__":
    main()
