import json
import subprocess
import sys
from pathlib import Path

PY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PY))
REPO = Path(__file__).resolve().parents[3]

from nn.encoding import normalize_message
from search.native.cpp_extension import load_native_mcts_extension

cp = f"{REPO / 'out'};{REPO / 'lib' / 'json.jar'}"
out = subprocess.check_output(["java", "-cp", cp, "core.game.NativeParityOracle", "--fixture", "smoke"], text=True)
oracle = json.loads(out)
player_id = oracle["player_id"]
root = normalize_message({"player_id": player_id, **oracle["root"]})
ext = load_native_mcts_extension()
for child in oracle["children"]:
    if child["action_id"] != "A6":
        continue
    idx = child["action_index"]
    tree = ext.NativeMCTS(root, [idx], [1.0], 0.0, False, 7, 256)
    sel = dict(tree.select_leaf(1.0))
    cpp = normalize_message({"player_id": player_id, **dict(sel["leaf_payload"])})
    java = normalize_message({"player_id": player_id, **dict(child["state"])})
    print("java actions", len(java["actions"]), [a.get("type") for a in java["actions"]])
    print("cpp actions", len(cpp["actions"]), [a.get("type") for a in cpp["actions"][:10]])
    print("java active", java["active_player_id"], "cpp active", cpp["active_player_id"])
    print("java t1 stars", java["observation"]["tribes"][1]["stars"], "cpp", cpp["observation"]["tribes"][1]["stars"])
