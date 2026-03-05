# Bonk Tag RL (Self-Play)

This folder contains a full self-play PPO project for the Bonk-style tag game, using the same physics/collision rules as `physics-extractor/demo/sandbox.js`.

## What this trains

- `catcher` model: optimized to tag quickly.
- `evader` model: optimized to survive.
- Both are trained simultaneously in self-play.
- Domain randomization is enabled by default (map jitter + physics jitter).

## Project layout

- `bonk_rl/physics.py`: core simulation and observation builder.
- `bonk_rl/policy.py`: policy/value network + checkpoint helpers.
- `scripts/train_self_play.py`: main training script.
- `scripts/evaluate.py`: benchmark script.
- `scripts/evaluate_matrix.py`: cross-play checkpoint matrix + latest-vs-history summary.
- `scripts/export_policy_json.py`: export checkpoint to browser JSON.
- `scripts/train_pufferlib_role.py`: optional PufferLib role-specific trainer scaffold.

## Setup with uv

From repo root:

```bash
cd rl-tag
uv sync
```

## Train

```bash
cd rl-tag
uv run python scripts/train_self_play.py \
  --total-updates 5000 \
  --num-envs 256 \
  --horizon 128 \
  --opponent-pool True \
  --pool-size 12 \
  --pool-add-every 10 \
  --latest-opponent-prob 0.5 \
  --checkpoint-every 50 \
  --out-dir runs/self_play
```

This writes checkpoints like:

- `runs/self_play/catcher_000050.pt`
- `runs/self_play/evader_000050.pt`

## Evaluate

```bash
cd rl-tag
uv run python scripts/evaluate.py \
  --catcher-ckpt runs/self_play/catcher_005000.pt \
  --evader-ckpt runs/self_play/evader_005000.pt \
  --episodes 1000
```

Cross-play checkpoints (proves newest vs older):

```bash
cd rl-tag
uv run python scripts/evaluate_matrix.py \
  --catchers-glob "runs/self_play/catcher_*.pt" \
  --evaders-glob "runs/self_play/evader_*.pt" \
  --episodes 200 \
  --stride 2 \
  --csv-out results/cross_play.csv
```

## Export for browser play

```bash
cd rl-tag
uv run python scripts/export_policy_json.py \
  --checkpoint runs/self_play/evader_005000.pt \
  --out-json ../physics-extractor/models/evader_latest.json

uv run python scripts/export_policy_json.py \
  --checkpoint runs/self_play/catcher_005000.pt \
  --out-json ../physics-extractor/models/catcher_latest.json
```

## Human vs model in browser

Open:

- `physics-extractor/demo/intag.html`

You can choose bot role and policy JSON path.

- If bot role is `evader`: human catches with Arrow keys.
- If bot role is `catcher`: human evades with WASD.

## Running on legion with shc

Sync code to remote:

```bash
cd /home/oboro/src/bonkio-rl
shc -r ./ legion:~/bonkio-rl
```

Run remote training:

```bash
shc legion "cd ~/bonkio-rl/rl-tag && uv sync && uv run python scripts/train_self_play.py --total-updates 5000 --num-envs 256 --horizon 128 --out-dir runs/self_play"
```

Run remote eval:

```bash
shc legion "cd ~/bonkio-rl/rl-tag && uv run python scripts/evaluate.py --catcher-ckpt runs/self_play/catcher_005000.pt --evader-ckpt runs/self_play/evader_005000.pt --episodes 2000"
```

Pull results back:

```bash
shc -r legion:~/bonkio-rl/rl-tag/runs ./rl-tag/
```

## Notes

- Rewards and PPO hyperparameters are intentionally simple to start; tune as needed.
- Increase `--total-updates` significantly for superhuman performance.
- For faster + larger-scale runs, switch to a native PufferLib env implementation later.
