import sys
from pathlib import Path

PY = Path(__file__).resolve().parents[2]
if str(PY) not in sys.path:
    sys.path.insert(0, str(PY))

from torch.utils.cpp_extension import load
from pathlib import Path

source = Path(__file__).with_name("native_mcts.cpp")
rules = Path(__file__).with_name("native_rules.cpp")
static_eval = Path(__file__).with_name("native_static_eval.cpp")
try:
    load(
        name="tribes_rl_native_mcts",
        sources=[str(source), str(rules), str(static_eval)],
        extra_cflags=["/O2", "/std:c++17"],
        build_directory=str(source.parent / ".build"),
        verbose=True,
    )
    print("loaded")
except Exception as exc:
    print("failed:", exc)
