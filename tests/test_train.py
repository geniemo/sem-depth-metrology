from pathlib import Path

import pandas as pd
import pytest

from semdepth.train import run_training


def _tiny_cfg(synth_root: Path, tmp_path: Path) -> dict:
    return {
        "seed": 42,
        "data": {
            "sim_sem_dir": str(synth_root / "simulation_data" / "SEM"),
            "sim_depth_dir": str(synth_root / "simulation_data" / "Depth"),
            "val_fraction": 0.34,
        },
        "model": {"encoder": "resnet18", "pretrained": False},
        "train": {
            "epochs": 2, "batch_size": 4, "lr": 1.0e-3, "weight_decay": 0.0,
            "num_workers": 0, "amp": False, "augment": True, "device": "cpu",
        },
        "out": {
            "run_name": "smoke",
            "runs_dir": str(tmp_path / "runs"),
            "results_csv": str(tmp_path / "results.csv"),
        },
    }


def test_run_training_end_to_end(synth_root, tmp_path):
    row = run_training(_tiny_cfg(synth_root, tmp_path))
    assert (tmp_path / "runs" / "smoke" / "best.pt").exists()
    df = pd.read_csv(tmp_path / "results.csv")
    assert df.loc[0, "run_name"] == "smoke"
    assert 0.0 < row["best_val_rmse255"] < 255.0


def test_run_training_rejects_empty_train_loader(synth_root, tmp_path):
    cfg = _tiny_cfg(synth_root, tmp_path)
    cfg["train"]["batch_size"] = 999  # larger than the whole synthetic train set
    with pytest.raises(ValueError, match="no training batches"):
        run_training(cfg)


def test_make_loss_variants():
    import torch
    from semdepth.train import make_loss
    p, t = torch.tensor([0.0, 1.0]), torch.tensor([1.0, 1.0])
    assert abs(make_loss("l1")(p, t).item() - 0.5) < 1e-6
    assert abs(make_loss("l2")(p, t).item() - 0.5) < 1e-6
    assert abs(make_loss("l1l2")(p, t).item() - 0.5) < 1e-6
    p2 = torch.tensor([0.5, 1.0])
    assert abs(make_loss("l1l2")(p2, t).item() - (0.5 * 0.25 + 0.5 * 0.125)) < 1e-6
    import pytest
    with pytest.raises(ValueError, match="unknown loss"):
        make_loss("huber")
