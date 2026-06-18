from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from profiling.analysis.actions import action_fingerprint, action_id


def _matches(action: dict[str, Any], selector: str) -> bool:
    return selector in {action_id(action), action_fingerprint(action)}


def _delegate(delegate: subprocess.Popen[str], message: dict[str, Any]) -> dict[str, Any]:
    assert delegate.stdin is not None
    assert delegate.stdout is not None
    delegate.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
    delegate.stdin.flush()
    while True:
        line = delegate.stdout.readline()
        if not line:
            raise RuntimeError("delegate bot exited without a response")
        response = json.loads(line)
        if response.get("type") == "forward_model":
            print(json.dumps(response, separators=(",", ":")), flush=True)
            fm_response = json.loads(sys.stdin.readline())
            delegate.stdin.write(json.dumps(fm_response, separators=(",", ":")) + "\n")
            delegate.stdin.flush()
            continue
        return response


def main() -> int:
    parser = argparse.ArgumentParser(description="Force one matching action once, then proxy to a delegate bot.")
    parser.add_argument("--force-action", required=True, help="Action id or action fingerprint to force once.")
    parser.add_argument("--force-request-id", type=int, default=None, help="Optional request id that must match before forcing.")
    parser.add_argument("delegate", nargs=argparse.REMAINDER, help="Delegate command after --.")
    args = parser.parse_args()
    delegate_command = list(args.delegate)
    if delegate_command and delegate_command[0] == "--":
        delegate_command = delegate_command[1:]
    if not delegate_command:
        raise RuntimeError("force_once_bot requires a delegate command after --")
    delegate = subprocess.Popen(
        delegate_command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
        bufsize=1,
    )
    forced = False
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            message = json.loads(line)
            if message.get("type") == "game_over":
                if delegate.stdin is not None:
                    delegate.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
                    delegate.stdin.flush()
                break
            if message.get("type") == "action_request" and not forced:
                request_id = message.get("request_id")
                request_matches = args.force_request_id is None or int(request_id) == int(args.force_request_id)
                if request_matches:
                    actions = list(message.get("actions", []))
                    for index, action in enumerate(actions):
                        if isinstance(action, dict) and _matches(action, args.force_action):
                            forced = True
                            print(json.dumps({"i": index, "actionId": action_id(action)}, separators=(",", ":")), flush=True)
                            break
                    if forced:
                        continue
                    raise RuntimeError(f"forced action was not legal in request {request_id}: {args.force_action}")
            response = _delegate(delegate, message)
            print(json.dumps(response, separators=(",", ":")), flush=True)
    finally:
        if delegate.stdin is not None:
            delegate.stdin.close()
        delegate.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
