import json
import random
import sys


rng = random.Random(7)


def pick_action(message):
    actions = message.get("actions", [])
    if not actions:
        return {"actionId": None}

    non_end_turn = [action for action in actions if action.get("type") != "END_TURN"]
    pool = non_end_turn or actions
    choice = rng.choice(pool)
    return {"actionId": choice["id"]}


for raw_line in sys.stdin:
    raw_line = raw_line.strip()
    if not raw_line:
        continue

    message = json.loads(raw_line)
    message_type = message.get("type")

    if message_type == "action_request":
        response = pick_action(message)
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()
    elif message_type == "game_over":
        break
