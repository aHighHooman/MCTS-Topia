import json
import sys
from typing import Any, Dict, List, Optional, Tuple

from strong_external_bot_v2 import ObservationView, action_score, tie_break_key


ROOT_BRANCH_LIMIT = 6
DISCOUNT = 0.65


def message_for_state(player_id: int, state_payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "player_id": player_id,
        "observation": state_payload["observation"],
    }


def rank_actions(actions: List[Dict[str, Any]], view: ObservationView, mode: str, limit: int) -> List[Tuple[float, Dict[str, Any]]]:
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for action in actions:
        score = action_score(action, view, mode)
        scored.append((score, action))
    scored.sort(key=lambda item: tie_break_key(item[1], item[0]), reverse=True)
    return scored[:limit]


def static_state_score(view: ObservationView) -> float:
    my_tribe = view.my_tribe()
    enemy_scores = [
        float(tribe.get("score", 0))
        for tribe_id, tribe in view.tribes.items()
        if int(tribe_id) != view.player_id
    ]
    material = sum(view.unit_value(unit) for unit in view.my_units) - sum(view.unit_value(unit) for unit in view.enemy_units)
    city_balance = 7.0 * (len(view.my_cities) - len(view.enemy_cities))
    star_balance = 0.4 * float(my_tribe.get("stars", 0))
    map_control = 0.2 * len(view.visible_tiles) + 2.0 * len(view.villages) + 2.2 * len(view.ruins)
    threat_penalty = sum(min(6.0, view.city_threat(city)) for city in view.my_cities)
    scoreboard = 0.12 * (float(my_tribe.get("score", 0)) - (max(enemy_scores) if enemy_scores else 0.0))
    return material + city_balance + star_balance + map_control + scoreboard - threat_penalty


class ForwardModelClient:
    def __init__(self, player_id: int):
        self.player_id = player_id

    def command(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        payload = dict(payload)
        payload["type"] = "forward_model"
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()

        while True:
            raw_line = sys.stdin.readline()
            if not raw_line:
                return None
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                response = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            return response

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


def choose_greedy(message: Dict[str, Any]) -> Dict[str, Optional[str]]:
    view = ObservationView(message)
    mode = view.strategic_mode()
    actions = message.get("actions", [])
    if not actions:
        return {"actionId": None}

    ranked = rank_actions(actions, view, mode, len(actions))
    return {"actionId": ranked[0][1]["id"] if ranked else None}


def choose_with_forward_model(message: Dict[str, Any]) -> Dict[str, Optional[str]]:
    fm_info = message.get("forward_model") or {}
    if not fm_info.get("enabled"):
        return choose_greedy(message)

    root_state_id = str(fm_info.get("root_state_id", "root"))
    player_id = int(message["player_id"])
    root_view = ObservationView(message)
    root_mode = root_view.strategic_mode()
    root_actions = message.get("actions", [])
    if not root_actions:
        return {"actionId": None}

    client = ForwardModelClient(player_id)
    candidates = rank_actions(root_actions, root_view, root_mode, ROOT_BRANCH_LIMIT)

    best_action_id: Optional[str] = None
    best_value = float("-inf")
    release_ids: List[str] = []
    try:
        for action_value, action in candidates:
            child_state = client.step(root_state_id, str(action["id"]))
            if not child_state:
                continue

            child_view = ObservationView(message_for_state(player_id, child_state))
            state_value = action_value + DISCOUNT * static_state_score(child_view)
            release_ids.append(str(child_state["state_id"]))
            if state_value > best_value:
                best_value = state_value
                best_action_id = str(action["id"])
    finally:
        client.release(release_ids)

    if best_action_id is None:
        return choose_greedy(message)
    return {"actionId": best_action_id}


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
            response = choose_with_forward_model(message)
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
        elif message.get("type") == "game_over":
            break


if __name__ == "__main__":
    main()
