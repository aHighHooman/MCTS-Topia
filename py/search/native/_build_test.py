import sys
from pathlib import Path

PY = Path(__file__).resolve().parents[2]
if str(PY) not in sys.path:
    sys.path.insert(0, str(PY))

from torch.utils.cpp_extension import load
from pathlib import Path

source = Path(__file__).with_name("src") / "mcts.cpp"
rules = source.with_name("rules.cpp")
static_eval = source.with_name("static_eval.cpp")
try:
    load(
        name="tribes_rl_native_mcts",
        sources=[str(source), str(rules), str(static_eval)],
        extra_cflags=["/O2", "/std:c++17"],
        extra_include_paths=[str(source.parent), str(Path(__file__).parent)],
        build_directory=str(Path(__file__).with_name(".build")),
        verbose=True,
    )
    print("loaded")
except Exception as exc:
    print("failed:", exc)
