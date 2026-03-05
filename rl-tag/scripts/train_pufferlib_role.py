#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import typer

from bonk_rl.physics import TagPhysics, build_obs
from bonk_rl.policy import PolicyValueNet, load_checkpoint

app = typer.Typer(add_completion=False)


class RoleGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, role: str, max_steps: int = 900, domain_randomization: bool = True):
        assert role in {"catcher", "evader"}
        self.role = role
        self.env = TagPhysics(max_steps=max_steps, domain_randomization=domain_randomization)
        self.observation_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(16,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(9)

    def reset(self, seed=None, options=None):
        state = self.env.reset(seed=seed)
        obs = np.asarray(build_obs(state, self.env.world, 0 if self.role == "catcher" else 1), dtype=np.float32)
        return obs, {}

    def _heuristic_action(self, state) -> int:
        me = state.players[0 if self.role == "evader" else 1]
        other = state.players[1 if self.role == "evader" else 0]

        if self.role == "evader":
            if me.x < other.x:
                horiz = 1
            else:
                horiz = 2
        else:
            if me.x < other.x:
                horiz = 2
            else:
                horiz = 1

        jump = 3 if me.y < other.y - 0.1 else 0
        if jump == 3 and horiz == 1:
            return 5
        if jump == 3 and horiz == 2:
            return 6
        return horiz

    def step(self, action):
        opponent_action = self._heuristic_action(self.env.state)
        if self.role == "catcher":
            state, info = self.env.step(int(action), int(opponent_action))
        else:
            state, info = self.env.step(int(opponent_action), int(action))

        done = bool(self.env.state.done)
        caught = bool(info.get("caught", False))
        timeout = bool(info.get("timeout", False))

        dx = state.players[1].x - state.players[0].x
        dy = state.players[1].y - state.players[0].y
        dist = float(np.sqrt(dx * dx + dy * dy))

        if self.role == "catcher":
            reward = 1.0 if caught else (-0.001 - 0.0003 * dist)
            if timeout:
                reward -= 0.05
            idx = 0
        else:
            reward = -1.0 if caught else (0.001 + 0.0003 * dist)
            if timeout:
                reward += 0.25
            idx = 1

        obs = np.asarray(build_obs(state, self.env.world, idx), dtype=np.float32)
        return obs, reward, done, False, info


class Policy(torch.nn.Module):
    def __init__(self, env):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(env.single_observation_space.shape[0], 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 256),
            torch.nn.Tanh(),
        )
        self.action_head = torch.nn.Linear(256, env.single_action_space.n)
        self.value_head = torch.nn.Linear(256, 1)

    def forward(self, observations, state=None):
        hidden = self.net(observations)
        logits = self.action_head(hidden)
        values = self.value_head(hidden)
        return logits, values

    def forward_eval(self, observations, state=None):
        return self.forward(observations, state)


@app.command()
def main(
    role: str = typer.Option("evader"),
    total_timesteps: int = typer.Option(20_000_000),
    num_envs: int = typer.Option(32),
    num_workers: int = typer.Option(8),
    env_batch_size: int = typer.Option(4),
):
    import pufferlib.vector
    from pufferlib import pufferl
    import pufferlib.emulation

    def env_creator(**kwargs: Any):
        return pufferlib.emulation.GymnasiumPufferEnv(RoleGymEnv(role=role, **kwargs))

    vecenv = pufferlib.vector.make(
        env_creator,
        num_envs=num_envs,
        num_workers=num_workers,
        batch_size=env_batch_size,
        backend=pufferlib.vector.Multiprocessing,
        env_kwargs={"max_steps": 900, "domain_randomization": True},
    )

    args = pufferl.load_config("default")
    args["train"]["env"] = f"bonk_tag_{role}"
    args["train"]["total_timesteps"] = total_timesteps
    args["train"]["learning_rate"] = 3e-4
    args["train"]["gamma"] = 0.995
    args["train"]["gae_lambda"] = 0.95
    args["train"]["ent_coef"] = 0.01

    policy = Policy(vecenv.driver_env).cuda() if torch.cuda.is_available() else Policy(vecenv.driver_env)
    trainer = pufferl.PuffeRL(args["train"], vecenv, policy)

    while trainer.epoch < trainer.total_epochs:
        trainer.evaluate()
        trainer.train()

    trainer.print_dashboard()
    trainer.close()


if __name__ == "__main__":
    app()
