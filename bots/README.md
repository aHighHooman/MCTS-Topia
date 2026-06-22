# Supported Bots

This directory is the visible entrypoint surface for supported bots.

## Hybrid NN Bot

Run the Python NN-guided MCTS bot through:

```powershell
$env:PYTHONPATH = "$PWD\py"
python bots\hybrid_nn_bot.py --checkpoint rl\checkpoints\latest.pt --replay-dir rl\replay
```

The implementation lives in `py/bots/hybrid_nn_bot.py` and `py/nn/bot_agent.py`.

## Native Static MCTS Bot

The static MCTS bot is the standalone C++ executable built from:

```text
bots/static_mcts_bot.cpp
```

Build it with:

```powershell
.\scripts\build_static_bot.ps1
```

The output is:

```text
out/native/static_mcts_bot.exe
```

`py/bots/simple_bot.py` is retained as an adjudication/test utility, not a supported primary bot.
