from __future__ import annotations

import torch


def require_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Tribes RL self-play/training, but torch.cuda.is_available() is false.")
    return torch.device("cuda")


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)
