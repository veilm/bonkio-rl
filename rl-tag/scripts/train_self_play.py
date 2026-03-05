#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import typer
from rich.console import Console
from rich.table import Table
from tqdm import trange

from bonk_rl.physics import TagPhysics, build_obs
from bonk_rl.policy import PPOBatch, PolicyValueNet, save_checkpoint

app = typer.Typer(add_completion=False)
console = Console()


@dataclass
class Rollout:
    obs: np.ndarray
    actions: np.ndarray
    logp: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    next_values: np.ndarray


class OpponentPool:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden: int,
        max_size: int,
        latest_prob: float,
        device: torch.device,
        seed: int,
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden = hidden
        self.max_size = max(1, max_size)
        self.latest_prob = float(np.clip(latest_prob, 0.0, 1.0))
        self.device = device
        self.rng = np.random.default_rng(seed)
        self._pool: List[PolicyValueNet] = []
        self._steps: List[int] = []

    def _clone_policy(self, policy: PolicyValueNet) -> PolicyValueNet:
        clone = PolicyValueNet(self.obs_dim, self.action_dim, hidden=self.hidden).to(self.device)
        clone.load_state_dict(policy.state_dict())
        clone.eval()
        for param in clone.parameters():
            param.requires_grad_(False)
        return clone

    def add(self, policy: PolicyValueNet, step: int) -> None:
        self._pool.append(self._clone_policy(policy))
        self._steps.append(step)
        if len(self._pool) > self.max_size:
            self._pool.pop(0)
            self._steps.pop(0)

    def sample(self, latest: PolicyValueNet) -> Tuple[PolicyValueNet, int]:
        if not self._pool:
            return latest, -1
        if self.rng.random() < self.latest_prob:
            return self._pool[-1], self._steps[-1]
        idx = int(self.rng.integers(0, len(self._pool)))
        return self._pool[idx], self._steps[idx]


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    t_steps, n_envs = rewards.shape
    advantages = np.zeros_like(rewards, dtype=np.float32)
    lastgaelam = np.zeros(n_envs, dtype=np.float32)
    for t in reversed(range(t_steps)):
        if t == t_steps - 1:
            next_nonterminal = 1.0 - dones[t]
            next_vals = next_values
        else:
            next_nonterminal = 1.0 - dones[t + 1]
            next_vals = values[t + 1]
        delta = rewards[t] + gamma * next_vals * next_nonterminal - values[t]
        lastgaelam = delta + gamma * gae_lambda * next_nonterminal * lastgaelam
        advantages[t] = lastgaelam
    returns = advantages + values
    return advantages, returns


def ppo_update(
    policy: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    batch: PPOBatch,
    clip_coef: float,
    vf_coef: float,
    ent_coef: float,
    max_grad_norm: float,
    update_epochs: int,
    minibatch_size: int,
) -> Dict[str, float]:
    n = batch.obs.shape[0]
    idx = np.arange(n)
    clipfracs: List[float] = []

    policy_losses: List[float] = []
    value_losses: List[float] = []
    entropy_losses: List[float] = []
    approx_kls: List[float] = []

    for _ in range(update_epochs):
        np.random.shuffle(idx)
        for start in range(0, n, minibatch_size):
            mb = idx[start : start + minibatch_size]
            logits, new_values = policy(batch.obs[mb])
            dist = torch.distributions.Categorical(logits=logits)
            new_logp = dist.log_prob(batch.actions[mb])
            entropy = dist.entropy().mean()

            logratio = new_logp - batch.old_logp[mb]
            ratio = logratio.exp()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - logratio).mean().item()
                clipfracs.append(((ratio - 1.0).abs() > clip_coef).float().mean().item())

            mb_adv = batch.advantages[mb]
            mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
            pg_loss1 = -mb_adv * ratio
            pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
            policy_loss = torch.max(pg_loss1, pg_loss2).mean()

            value_loss = 0.5 * ((new_values - batch.returns[mb]) ** 2).mean()

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropy_losses.append(float(entropy.item()))
            approx_kls.append(float(approx_kl))

    return {
        "policy_loss": float(np.mean(policy_losses)),
        "value_loss": float(np.mean(value_losses)),
        "entropy": float(np.mean(entropy_losses)),
        "approx_kl": float(np.mean(approx_kls)),
        "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
    }


def init_env_states(envs: List[TagPhysics]) -> Tuple[np.ndarray, np.ndarray]:
    obs_c = []
    obs_e = []
    for env in envs:
        state = env.reset()
        obs_c.append(build_obs(state, env.world, 0))
        obs_e.append(build_obs(state, env.world, 1))
    return np.asarray(obs_c, dtype=np.float32), np.asarray(obs_e, dtype=np.float32)


def init_role_obs(envs: List[TagPhysics], role_index: int) -> np.ndarray:
    obs = []
    for env in envs:
        state = env.reset()
        obs.append(build_obs(state, env.world, role_index))
    return np.asarray(obs, dtype=np.float32)


def collect_role_rollout(
    envs: List[TagPhysics],
    learner_policy: PolicyValueNet,
    opponent_policy: PolicyValueNet,
    learner_role: int,
    obs_learner: np.ndarray,
    horizon: int,
    device: torch.device,
) -> Tuple[Rollout, np.ndarray, Dict[str, float]]:
    n_envs = len(envs)
    obs_dim = obs_learner.shape[1]

    r_obs = np.zeros((horizon, n_envs, obs_dim), dtype=np.float32)
    r_act = np.zeros((horizon, n_envs), dtype=np.int64)
    r_logp = np.zeros((horizon, n_envs), dtype=np.float32)
    r_val = np.zeros((horizon, n_envs), dtype=np.float32)
    r_rew = np.zeros((horizon, n_envs), dtype=np.float32)
    r_done = np.zeros((horizon, n_envs), dtype=np.float32)

    catches = 0
    timeouts = 0
    mean_dist = 0.0

    for t in range(horizon):
        r_obs[t] = obs_learner

        with torch.no_grad():
            t_learner = torch.as_tensor(obs_learner, device=device)
            act_learner, logp_learner, val_learner = learner_policy.sample_action(t_learner)

        opp_obs = []
        for env in envs:
            opp_obs.append(build_obs(env.state, env.world, 1 - learner_role))
        opp_obs_t = torch.as_tensor(np.asarray(opp_obs, dtype=np.float32), device=device)
        with torch.no_grad():
            act_opp, _, _ = opponent_policy.sample_action(opp_obs_t)

        a_learner = act_learner.cpu().numpy()
        a_opp = act_opp.cpu().numpy()
        r_act[t] = a_learner
        r_logp[t] = logp_learner.cpu().numpy()
        r_val[t] = val_learner.cpu().numpy()

        next_obs = np.zeros_like(obs_learner)

        for i, env in enumerate(envs):
            if learner_role == 0:
                catcher_action = int(a_learner[i])
                evader_action = int(a_opp[i])
            else:
                catcher_action = int(a_opp[i])
                evader_action = int(a_learner[i])
            state, info = env.step(catcher_action, evader_action)

            dx = state.players[1].x - state.players[0].x
            dy = state.players[1].y - state.players[0].y
            dist = float(np.sqrt(dx * dx + dy * dy))
            mean_dist += dist

            caught = bool(info.get("caught", False))
            timeout = bool(info.get("timeout", False))
            done = env.state.done

            if learner_role == 0:
                reward = 1.0 if caught else (-0.001 - 0.0003 * dist)
                if timeout:
                    reward -= 0.05
            else:
                reward = -1.0 if caught else (0.001 + 0.0003 * dist)
                if timeout:
                    reward += 0.25
            r_rew[t, i] = reward

            r_done[t, i] = 1.0 if done else 0.0

            if caught:
                catches += 1
            if timeout:
                timeouts += 1

            if done:
                state = env.reset()

            next_obs[i] = np.asarray(build_obs(state, env.world, learner_role), dtype=np.float32)

        obs_learner = next_obs

    with torch.no_grad():
        obs_t = torch.as_tensor(obs_learner, device=device)
        _, next_val = learner_policy.forward(obs_t)

    rollout = Rollout(
        obs=r_obs,
        actions=r_act,
        logp=r_logp,
        values=r_val,
        rewards=r_rew,
        dones=r_done,
        next_values=next_val.cpu().numpy(),
    )
    stats = {
        "catch_rate": catches / max(horizon * n_envs, 1),
        "timeout_rate": timeouts / max(horizon * n_envs, 1),
        "mean_dist": mean_dist / max(horizon * n_envs, 1),
    }
    return rollout, obs_learner, stats


def flatten_rollout(
    rollout: Rollout,
    gamma: float,
    gae_lambda: float,
    device: torch.device,
) -> PPOBatch:
    advantages, returns = compute_gae(
        rewards=rollout.rewards,
        values=rollout.values,
        dones=rollout.dones,
        next_values=rollout.next_values,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    return PPOBatch(
        obs=torch.as_tensor(rollout.obs.reshape(-1, rollout.obs.shape[-1]), device=device),
        actions=torch.as_tensor(rollout.actions.reshape(-1), dtype=torch.long, device=device),
        old_logp=torch.as_tensor(rollout.logp.reshape(-1), device=device),
        returns=torch.as_tensor(returns.reshape(-1), device=device),
        advantages=torch.as_tensor(advantages.reshape(-1), device=device),
        values=torch.as_tensor(rollout.values.reshape(-1), device=device),
    )


@app.command()
def main(
    total_updates: int = typer.Option(5000),
    num_envs: int = typer.Option(256),
    horizon: int = typer.Option(128),
    max_steps: int = typer.Option(900),
    learning_rate: float = typer.Option(3e-4),
    gamma: float = typer.Option(0.995),
    gae_lambda: float = typer.Option(0.95),
    clip_coef: float = typer.Option(0.2),
    vf_coef: float = typer.Option(0.5),
    ent_coef: float = typer.Option(0.01),
    update_epochs: int = typer.Option(4),
    minibatch_size: int = typer.Option(4096),
    max_grad_norm: float = typer.Option(0.5),
    hidden: int = typer.Option(256),
    seed: int = typer.Option(7),
    checkpoint_every: int = typer.Option(50),
    opponent_pool: bool = typer.Option(True),
    pool_size: int = typer.Option(12),
    pool_add_every: int = typer.Option(10),
    latest_opponent_prob: float = typer.Option(0.5),
    out_dir: Path = typer.Option(Path("runs/self_play")),
    domain_randomization: bool = typer.Option(True),
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim = 16
    action_dim = 9

    catcher = PolicyValueNet(obs_dim, action_dim, hidden=hidden).to(device)
    evader = PolicyValueNet(obs_dim, action_dim, hidden=hidden).to(device)

    opt_c = torch.optim.Adam(catcher.parameters(), lr=learning_rate, eps=1e-5)
    opt_e = torch.optim.Adam(evader.parameters(), lr=learning_rate, eps=1e-5)

    envs_c: List[TagPhysics] = [
        TagPhysics(max_steps=max_steps, domain_randomization=domain_randomization) for _ in range(num_envs)
    ]
    envs_e: List[TagPhysics] = [
        TagPhysics(max_steps=max_steps, domain_randomization=domain_randomization) for _ in range(num_envs)
    ]
    obs_c = init_role_obs(envs_c, role_index=0)
    obs_e = init_role_obs(envs_e, role_index=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    catcher_pool = OpponentPool(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden=hidden,
        max_size=pool_size,
        latest_prob=latest_opponent_prob,
        device=device,
        seed=seed + 101,
    )
    evader_pool = OpponentPool(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden=hidden,
        max_size=pool_size,
        latest_prob=latest_opponent_prob,
        device=device,
        seed=seed + 202,
    )
    catcher_pool.add(catcher, step=0)
    evader_pool.add(evader, step=0)

    progress = trange(1, total_updates + 1, desc="updates", dynamic_ncols=True)
    for update in progress:
        sampled_evader, sampled_evader_step = evader_pool.sample(evader)
        if not opponent_pool:
            sampled_evader = evader
            sampled_evader_step = update
        rollout_c, obs_c, stats_c = collect_role_rollout(
            envs_c,
            learner_policy=catcher,
            opponent_policy=sampled_evader,
            learner_role=0,
            obs_learner=obs_c,
            horizon=horizon,
            device=device,
        )
        batch_c = flatten_rollout(rollout_c, gamma, gae_lambda, device)

        logs_c = ppo_update(
            catcher,
            opt_c,
            batch_c,
            clip_coef,
            vf_coef,
            ent_coef,
            max_grad_norm,
            update_epochs,
            minibatch_size,
        )

        sampled_catcher, sampled_catcher_step = catcher_pool.sample(catcher)
        if not opponent_pool:
            sampled_catcher = catcher
            sampled_catcher_step = update
        rollout_e, obs_e, stats_e = collect_role_rollout(
            envs_e,
            learner_policy=evader,
            opponent_policy=sampled_catcher,
            learner_role=1,
            obs_learner=obs_e,
            horizon=horizon,
            device=device,
        )
        batch_e = flatten_rollout(rollout_e, gamma, gae_lambda, device)
        logs_e = ppo_update(
            evader,
            opt_e,
            batch_e,
            clip_coef,
            vf_coef,
            ent_coef,
            max_grad_norm,
            update_epochs,
            minibatch_size,
        )

        if opponent_pool and (update % pool_add_every == 0):
            catcher_pool.add(catcher, step=update)
            evader_pool.add(evader, step=update)

        progress.set_postfix(
            c_catch=f"{stats_c['catch_rate']:.3f}",
            e_timeout=f"{stats_e['timeout_rate']:.3f}",
            c_ent=f"{logs_c['entropy']:.3f}",
            e_ent=f"{logs_e['entropy']:.3f}",
            opp_e=str(sampled_evader_step),
            opp_c=str(sampled_catcher_step),
        )

        if update % checkpoint_every == 0 or update == total_updates:
            save_checkpoint(
                out_dir / f"catcher_{update:06d}.pt",
                catcher,
                opt_c,
                update,
                {
                    "role": "catcher",
                    "stats": stats_c,
                    "logs": logs_c,
                    "sampled_evader_step": sampled_evader_step,
                },
            )
            save_checkpoint(
                out_dir / f"evader_{update:06d}.pt",
                evader,
                opt_e,
                update,
                {
                    "role": "evader",
                    "stats": stats_e,
                    "logs": logs_e,
                    "sampled_catcher_step": sampled_catcher_step,
                },
            )

    table = Table(title="Training Complete")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Device", str(device))
    table.add_row("Updates", str(total_updates))
    table.add_row("Output", str(out_dir))
    console.print(table)


if __name__ == "__main__":
    app()
