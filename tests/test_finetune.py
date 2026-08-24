from pathlib import Path

import numpy as np
import pandas as pd
import torch

from semdepth.finetune import fit_affine, run_finetune
from semdepth.model import UnetTimm


def test_fit_affine_recovers_known_map():
    rng = np.random.default_rng(0)
    pred = rng.uniform(90, 130, size=200)
    avg = -50.0 + 1.5 * pred + rng.normal(0, 0.01, size=200)
    a, b = fit_affine(pred, avg)
    assert abs(a - (-50.0)) < 0.1 and abs(b - 1.5) < 0.01


def _cfg(synth_root: Path, tmp_path: Path, init_ckpt: Path) -> dict:
    return {
        "seed": 42,
        "init_ckpt": str(init_ckpt),
        "data": {
            "sim_sem_dir": str(synth_root / "simulation_data" / "SEM"),
            "sim_depth_dir": str(synth_root / "simulation_data" / "Depth"),
            "real_sem_root": str(synth_root / "train" / "SEM"),
            "real_csv": str(synth_root / "train" / "average_depth.csv"),
            "val_fraction": 0.34,
        },
        "model": {"encoder": "resnet18"},
        "train": {
            "epochs": 1, "batch_size": 4, "real_batch": 4, "lr": 1.0e-3,
            "weight_decay": 0.0, "num_workers": 0, "amp": False, "augment": False,
            "device": "cpu", "lambda_real": 1.0, "refit_sample": 9,
        },
        "out": {
            "run_name": "ft_smoke",
            "runs_dir": str(tmp_path / "runs"),
            "results_csv": str(tmp_path / "results.csv"),
        },
    }


def test_run_finetune_end_to_end(synth_root, tmp_path):
    torch.manual_seed(0)
    init = UnetTimm("resnet18", pretrained=False)
    init_ckpt = tmp_path / "init.pt"
    torch.save(init.state_dict(), init_ckpt)

    row = run_finetune(_cfg(synth_root, tmp_path, init_ckpt))

    ckpt_path = tmp_path / "runs" / "ft_smoke" / "best.pt"
    assert ckpt_path.exists()
    df = pd.read_csv(tmp_path / "results.csv")
    assert df.loc[0, "run_name"] == "ft_smoke"
    assert np.isfinite(row["best_cal_proxy_sample"])
    assert np.isfinite(row["best_val_rmse255"])
    # weights actually moved away from the init checkpoint
    trained = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    deltas = [
        (trained[k].float() - v.float()).abs().max().item()
        for k, v in init.state_dict().items()
        if v.dtype.is_floating_point
    ]
    assert max(deltas) > 0.0
