#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import typer
from rich.console import Console
from rich.table import Table

from bonk_rl.physics import TagPhysics, build_obs
from bonk_rl.policy import PolicyValueNet, load_checkpoint

app = typer.Typer(add_completion=False)
console = Console()


def load_policy(ckpt_path: Path, device: torch.device) -> PolicyValueNet:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    arch = ckpt["arch"]
    model = PolicyValueNet(arch["obs_dim"], arch["action_dim"], hidden=arch["hidden"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@app.command()
def main(
    catcher_ckpt: Path = typer.Option(..., exists=True),
    evader_ckpt: Path = typer.Option(..., exists=True),
    episodes: int = typer.Option(1000),
    max_steps: int = typer.Option(900),
    domain_randomization: bool = typer.Option(False),
    seed: int = typer.Option(42),
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    catcher = load_policy(catcher_ckpt, device)
    evader = load_policy(evader_ckpt, device)

    catches = 0
    timeouts = 0
    catch_steps = []

    for _ in range(episodes):
        env = TagPhysics(max_steps=max_steps, domain_randomization=domain_randomization)
        state = env.reset()
        for step in range(max_steps):
            obs_c = torch.as_tensor([build_obs(state, env.world, 0)], dtype=torch.float32, device=device)
            obs_e = torch.as_tensor([build_obs(state, env.world, 1)], dtype=torch.float32, device=device)
            with torch.no_grad():
                a_c = int(catcher.act_greedy(obs_c).item())
                a_e = int(evader.act_greedy(obs_e).item())
            state, info = env.step(a_c, a_e)
            if info.get("caught", False):
                catches += 1
                catch_steps.append(step + 1)
                break
            if info.get("timeout", False):
                timeouts += 1
                break

    table = Table(title="Match Evaluation")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Episodes", str(episodes))
    table.add_row("Catch rate", f"{catches / episodes:.3f}")
    table.add_row("Timeout rate", f"{timeouts / episodes:.3f}")
    table.add_row(
        "Mean catch step",
        f"{float(np.mean(catch_steps)):.1f}" if catch_steps else "n/a",
    )
    console.print(table)


if __name__ == "__main__":
    app()
