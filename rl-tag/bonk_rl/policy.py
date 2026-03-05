from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.distributions import Categorical


class PolicyValueNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden = hidden

        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, action_dim)
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.body(obs)
        return self.pi(h), self.v(h).squeeze(-1)

    def sample_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action, logp, value

    def act_greedy(self, obs: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(obs)
        return torch.argmax(logits, dim=-1)


@dataclass
class PPOBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    old_logp: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    values: torch.Tensor


def save_checkpoint(
    out_path: Path,
    policy: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    step: int,
    metadata: Dict,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": step,
            "metadata": metadata,
            "arch": {
                "obs_dim": policy.obs_dim,
                "action_dim": policy.action_dim,
                "hidden": policy.hidden,
            },
        },
        out_path,
    )


def load_checkpoint(path: Path, map_location: str = "cpu") -> Dict:
    return torch.load(path, map_location=map_location)
