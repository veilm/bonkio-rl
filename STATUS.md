# STATUS

Last updated: 2026-02-20

## Objective
Train superhuman Bonk.io-style tag agents using extracted physics with:
- Self-play between catcher and evader
- Domain randomization
- Browser deployment path (`physics-extractor/demo/intag.html`)
- GPU training on remote `legion-remote`

## What Is Implemented

### RL pipeline (`rl-tag/`)
- `bonk_rl/physics.py`: two-player tag sim with map/physics randomization.
- `bonk_rl/policy.py`: MLP actor-critic + checkpoint IO.
- `scripts/train_self_play.py`: PPO self-play with **opponent checkpoint pools**.
  - Learner vs sampled historical opponent (per role)
  - Pool controls: `--opponent-pool`, `--pool-size`, `--pool-add-every`, `--latest-opponent-prob`
- `scripts/evaluate.py`: greedy checkpoint-vs-checkpoint evaluation.
- `scripts/evaluate_matrix.py`: cross-play checkpoint matrix + latest-vs-history summary.
- `scripts/export_policy_json.py`: PyTorch checkpoint -> browser JSON policy.

### Browser integration
- `physics-extractor/demo/sandbox.js`: external intent-provider hook.
- `physics-extractor/demo/rl_policy.js`: JS inference for exported MLP.
- `physics-extractor/demo/intag.html`: human-vs-bot page.

## Remote Training Status

### Connectivity
- `shc legion` is still unreachable from this machine.
- `shc legion-remote` works and reaches host `LEGION`.

### Remote environment facts
- OS/shell target: Windows PowerShell.
- GPU: NVIDIA GeForce RTX 5070 Ti (driver 591.44).
- Python available: 3.14.
- `git` and `uv` are not installed on that target.

### Workarounds that succeeded
- Pulled repo to remote via GitHub ZIP (no git available).
- Pushed local modified files directly to remote using base64 transfer.
- Created venv + pip install of dependencies manually.
- `torch+cu126` failed on RTX 5070 Ti (`sm_120` kernel image error).
- Reinstalled `torch==2.10.0+cu128`; CUDA ops now work.

## Validation Performed

### Local
- `python -m py_compile rl-tag/bonk_rl/*.py rl-tag/scripts/*.py` passes.
- Opponent-pool smoke train/eval executed.

### Remote GPU run (completed)
Train command:
```bash
python -m scripts.train_self_play \
  --total-updates 200 \
  --num-envs 128 \
  --horizon 96 \
  --checkpoint-every 20 \
  --pool-size 16 \
  --pool-add-every 10 \
  --latest-opponent-prob 0.35 \
  --out-dir runs/pool_remote_200
```
Result:
- Device: `cuda`
- Updates: `200`

Latest-vs-latest eval:
```bash
python -m scripts.evaluate \
  --catcher-ckpt runs/pool_remote_200/catcher_000200.pt \
  --evader-ckpt runs/pool_remote_200/evader_000200.pt \
  --episodes 300
```
- Catch rate: `1.000`
- Timeout rate: `0.000`
- Mean catch step: `363.0`

Cross-play matrix (last 6 checkpoints each role):
```bash
python -m scripts.evaluate_matrix \
  --limit 6 \
  --episodes 30 \
  --max-steps 600 \
  --csv-out results/pool_remote_200_matrix.csv
```
Latest-vs-history summary:
- Latest catcher mean catch rate vs old evaders: `0.667`
- Old catchers mean catch rate vs latest evader: `0.687`

Interpretation:
- Models are clearly learning non-trivial behaviors.
- Latest-vs-latest is strong for catcher in this run.
- Cross-play still shows non-transitive dynamics; robust “current always clowns old opposite-role checkpoints” is not solved yet.

## Known Gaps / Risks
- Meta-game cycling is still present despite opponent pool.
- No Elo/rating system yet for checkpoint selection.
- No automated remote artifact sync-back implemented.
- Browser inference path works, but no latency/perf optimization pass yet.

## Recommended Next Steps
1. Add rating-based opponent sampling (Elo/TrueSkill-style) instead of uniform/random pool sampling.
2. Add periodic league-style evaluation and keep only promoted checkpoints.
3. Increase remote training horizon (e.g., 2k+ updates) and run hyperparameter sweeps.
4. Export best-rated checkpoints and validate human-vs-bot in `intag.html`.
