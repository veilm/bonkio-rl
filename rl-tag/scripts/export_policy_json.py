#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import torch
import typer

from bonk_rl.policy import PolicyValueNet, load_checkpoint

app = typer.Typer(add_completion=False)


def tensor_to_list(t: torch.Tensor):
    return t.detach().cpu().numpy().tolist()


@app.command()
def main(
    checkpoint: Path = typer.Option(..., exists=True),
    out_json: Path = typer.Option(...),
):
    ckpt = load_checkpoint(checkpoint)
    arch = ckpt["arch"]
    model = PolicyValueNet(arch["obs_dim"], arch["action_dim"], hidden=arch["hidden"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    state = model.state_dict()
    payload = {
        "arch": arch,
        "state_dict": {k: tensor_to_list(v) for k, v in state.items()},
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    app()
