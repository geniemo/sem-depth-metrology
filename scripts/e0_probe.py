"""E0 probe: inference-time GLOBAL affine input alignment (real -> sim intensity).

No retraining. Compares the baseline checkpoint's real-domain proxy metrics with
and without mapping real inputs into the sim intensity distribution via one
global affine x' = (x - mu_r)/sigma_r * sigma_s + mu_s (order-preserving across
images — per-image normalization is forbidden because absolute brightness IS the
depth signal, r=-0.977).
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from semdepth.data import RealDataset, load_image01
from semdepth.model import UnetTimm
from semdepth.results import append_result


def domain_stats(root: Path, n: int = 3000, seed: int = 0) -> tuple[float, float]:
    paths = [p for p in sorted(Path(root).rglob("*.png"))
             if not any(part.startswith(".") for part in p.parts)]
    rng = random.Random(seed)
    pick = paths if len(paths) <= n else rng.sample(paths, n)
    arr = np.stack([load_image01(p) * 255.0 for p in pick])
    return float(arr.mean()), float(arr.std())


@torch.no_grad()
def proxy_metrics(model, ds, device, transform=None, batch_size=512):
    dl = DataLoader(ds, batch_size=batch_size, num_workers=8)
    preds, avgs = [], []
    for batch in dl:
        x = batch["image"].to(device)
        if transform is not None:
            x = transform(x)
        preds.append((model(x).mean(dim=(1, 2, 3)) * 255.0).cpu().numpy())
        avgs.append(batch["avg_depth"].float().numpy())
    pred, avg = np.concatenate(preds), np.concatenate(avgs)
    b, a = np.polyfit(pred, avg, 1)
    cal = float(np.sqrt(np.mean((avg - (a + b * pred)) ** 2)))
    corr = float(np.corrcoef(pred, avg)[0, 1])
    return cal, corr, float(a), float(b), float(pred.min()), float(pred.max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/baseline.yaml")
    ap.add_argument("--ckpt", default="experiments/runs/s1_baseline_r34/best.pt")
    ap.add_argument("--root", default="data/raw")
    args = ap.parse_args()
    root = Path(args.root)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model = UnetTimm(cfg["model"]["encoder"], pretrained=False)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=True))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    mu_s, sd_s = domain_stats(root / "simulation_data" / "SEM")
    mu_r, sd_r = domain_stats(root / "train" / "SEM")
    print(f"[stats] sim ({mu_s:.1f},{sd_s:.1f})  real ({mu_r:.1f},{sd_r:.1f})")

    def align(x: torch.Tensor) -> torch.Tensor:  # x in [0,1]
        x255 = x * 255.0
        return (((x255 - mu_r) / sd_r * sd_s + mu_s) / 255.0).clamp(0, 1)

    ds = RealDataset(root / "train" / "SEM", root / "train" / "average_depth.csv")
    for name, tf in [("raw", None), ("aligned", align)]:
        cal, corr, a, b, lo, hi = proxy_metrics(model, ds, device, tf)
        print(f"[{name}] cal_proxy={cal:.4f} corr={corr:.4f} "
              f"fit a={a:.2f} b={b:.4f} pred-mean range {lo:.1f}~{hi:.1f}")
        append_result(Path("experiments/results.csv"), {
            "run_name": f"e0_probe_{name}",
            "cal_proxy_rmse": round(cal, 4),
            "proxy_corr": round(corr, 4),
            "cal_a": round(a, 3),
            "cal_b": round(b, 4),
        })


if __name__ == "__main__":
    main()
