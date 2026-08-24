"""Evaluate a trained checkpoint against the real train domain (submission-free).

Reports two numbers and appends them to experiments/results.csv:
  raw_proxy_rmse   — RMSE of mean(pred)*255 vs average_depth as-is. The two live in
                     different label encodings (EDA: sim pixel smaller=deeper vs
                     avg_depth larger=deeper), so this is tracked for continuity,
                     not interpreted as an absolute error.
  cal_proxy_rmse   — residual RMSE after fitting avg_depth ~= a + b*mean(pred)*255
                     on real train. Scale/direction-free: measures how much of the
                     per-site depth ordering the model recovers across the domain gap.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from semdepth.data import RealDataset
from semdepth.model import UnetTimm
from semdepth.results import append_result


@torch.no_grad()
def image_mean_preds(model, ds, device: str, batch_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    dl = DataLoader(ds, batch_size=batch_size, num_workers=8)
    preds, avgs = [], []
    for batch in dl:
        x = batch["image"].to(device)
        preds.append((model(x).mean(dim=(1, 2, 3)) * 255.0).cpu().numpy())
        avgs.append(batch["avg_depth"].float().numpy())
    return np.concatenate(preds), np.concatenate(avgs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--real-sem-root", default="data/raw/train/SEM")
    ap.add_argument("--real-csv", default="data/raw/train/average_depth.csv")
    ap.add_argument("--results-csv", default="experiments/results.csv")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model = UnetTimm(cfg["model"]["encoder"], pretrained=False)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=True))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    ds = RealDataset(Path(args.real_sem_root), Path(args.real_csv))
    pred, avg = image_mean_preds(model, ds, device)
    raw = float(np.sqrt(np.mean((pred - avg) ** 2)))
    b, a = np.polyfit(pred, avg, 1)
    resid = avg - (a + b * pred)
    cal = float(np.sqrt(np.mean(resid ** 2)))
    corr = float(np.corrcoef(pred, avg)[0, 1])
    print(f"n={len(ds)} raw_proxy_rmse={raw:.4f} cal_proxy_rmse={cal:.4f} "
          f"corr={corr:.4f} fit: avg_depth ~= {a:.2f} + {b:.4f}*mean_pred255")
    append_result(Path(args.results_csv), {
        "run_name": Path(args.ckpt).parent.name + "_proxy",
        "raw_proxy_rmse": round(raw, 4),
        "cal_proxy_rmse": round(cal, 4),
        "proxy_corr": round(corr, 4),
        "cal_a": round(float(a), 3),
        "cal_b": round(float(b), 4),
    })


if __name__ == "__main__":
    main()
