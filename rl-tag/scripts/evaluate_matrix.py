#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

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


def extract_step(path: Path) -> int:
    m = re.search(r"_(\d+)\.pt$", path.name)
    if m:
        return int(m.group(1))
    return -1


def evaluate_pair(
    catcher: PolicyValueNet,
    evader: PolicyValueNet,
    episodes: int,
    max_steps: int,
    domain_randomization: bool,
) -> Dict[str, float]:
    catches = 0
    timeouts = 0
    catch_steps: List[int] = []
    for _ in range(episodes):
        env = TagPhysics(max_steps=max_steps, domain_randomization=domain_randomization)
        state = env.reset()
        for step in range(max_steps):
            obs_c = torch.as_tensor([build_obs(state, env.world, 0)], dtype=torch.float32, device=next(catcher.parameters()).device)
            obs_e = torch.as_tensor([build_obs(state, env.world, 1)], dtype=torch.float32, device=next(evader.parameters()).device)
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
    catch_rate = catches / max(episodes, 1)
    timeout_rate = timeouts / max(episodes, 1)
    mean_catch_step = float(np.mean(catch_steps)) if catch_steps else float("nan")
    return {
        "catch_rate": catch_rate,
        "timeout_rate": timeout_rate,
        "mean_catch_step": mean_catch_step,
    }


def select_paths(paths: List[Path], stride: int, limit: int) -> List[Path]:
    selected = paths[:: max(1, stride)]
    if limit > 0:
        selected = selected[-limit:]
    return selected


@app.command()
def main(
    catchers_glob: str = typer.Option("runs/self_play/catcher_*.pt"),
    evaders_glob: str = typer.Option("runs/self_play/evader_*.pt"),
    stride: int = typer.Option(1, help="Take every Nth checkpoint."),
    limit: int = typer.Option(0, help="If >0, keep only the newest N checkpoints per role."),
    episodes: int = typer.Option(100),
    max_steps: int = typer.Option(900),
    domain_randomization: bool = typer.Option(False),
    csv_out: Path | None = typer.Option(None, help="Optional output CSV path."),
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    catcher_paths = sorted(Path().glob(catchers_glob), key=extract_step)
    evader_paths = sorted(Path().glob(evaders_glob), key=extract_step)
    if not catcher_paths or not evader_paths:
        raise typer.BadParameter("No checkpoints found for one or both globs.")

    catcher_paths = select_paths(catcher_paths, stride, limit)
    evader_paths = select_paths(evader_paths, stride, limit)

    catchers = [(extract_step(p), load_policy(p, device), p) for p in catcher_paths]
    evaders = [(extract_step(p), load_policy(p, device), p) for p in evader_paths]

    matrix: List[Tuple[int, int, float, float, float]] = []
    for c_step, c_model, _ in catchers:
        for e_step, e_model, _ in evaders:
            metrics = evaluate_pair(
                catcher=c_model,
                evader=e_model,
                episodes=episodes,
                max_steps=max_steps,
                domain_randomization=domain_randomization,
            )
            matrix.append(
                (c_step, e_step, metrics["catch_rate"], metrics["timeout_rate"], metrics["mean_catch_step"])
            )

    latest_c_step, latest_c_model, _ = catchers[-1]
    latest_e_step, latest_e_model, _ = evaders[-1]

    latest_c_vs_old_e = []
    for e_step, e_model, _ in evaders[:-1]:
        m = evaluate_pair(latest_c_model, e_model, episodes, max_steps, domain_randomization)
        latest_c_vs_old_e.append((e_step, m["catch_rate"]))

    old_c_vs_latest_e = []
    for c_step, c_model, _ in catchers[:-1]:
        m = evaluate_pair(c_model, latest_e_model, episodes, max_steps, domain_randomization)
        old_c_vs_latest_e.append((c_step, m["catch_rate"]))

    table = Table(title="Cross-Play Matrix (Catch Rate)")
    table.add_column("Catcher step")
    table.add_column("Evader step")
    table.add_column("Catch rate")
    table.add_column("Timeout rate")
    table.add_column("Mean catch step")
    for c_step, e_step, catch_rate, timeout_rate, mean_catch_step in matrix:
        mean_str = f"{mean_catch_step:.1f}" if np.isfinite(mean_catch_step) else "n/a"
        table.add_row(
            str(c_step),
            str(e_step),
            f"{catch_rate:.3f}",
            f"{timeout_rate:.3f}",
            mean_str,
        )
    console.print(table)

    summary = Table(title="Latest-vs-History Summary")
    summary.add_column("Metric")
    summary.add_column("Value")
    summary.add_row("Latest catcher step", str(latest_c_step))
    summary.add_row("Latest evader step", str(latest_e_step))
    if latest_c_vs_old_e:
        summary.add_row(
            "Latest catcher mean catch rate vs old evaders",
            f"{float(np.mean([v for _, v in latest_c_vs_old_e])):.3f}",
        )
    else:
        summary.add_row("Latest catcher mean catch rate vs old evaders", "n/a")
    if old_c_vs_latest_e:
        summary.add_row(
            "Old catchers mean catch rate vs latest evader",
            f"{float(np.mean([v for _, v in old_c_vs_latest_e])):.3f}",
        )
    else:
        summary.add_row("Old catchers mean catch rate vs latest evader", "n/a")
    console.print(summary)

    if csv_out is not None:
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        with csv_out.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["catcher_step", "evader_step", "catch_rate", "timeout_rate", "mean_catch_step"])
            for row in matrix:
                writer.writerow(row)
        console.print(f"[green]Wrote:[/green] {csv_out}")


if __name__ == "__main__":
    app()
