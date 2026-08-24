"""Weak-supervision fine-tuning: sim L1 + real per-site average-depth constraint.

Real labels (avg_depth: larger = deeper, physical scale) and the sim pixel
encoding (smaller = deeper) are linked by an unknown affine map. EM-style loop:
  (1) refit (alpha, beta) in closed form on the current model's real-train mean
      predictions (least squares of avg ~= alpha + beta * mean_pred255),
  (2) train with loss = L1_sim + lambda * |alpha + beta*mean_pred255 - avg| / 255,
so predictions spread the way the real labels demand without freezing a wrong
encoding guess. Sim L1 anchors pixel-level shape; the real term fixes the range.
"""
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from semdepth.data import (
    RealDataset,
    SimDataset,
    list_pseudo_pairs,
    list_sim_pairs,
    split_pairs,
)
from semdepth.model import UnetTimm
from semdepth.results import append_result
from semdepth.train import evaluate, seed_all


def fit_affine(pred255: np.ndarray, avg: np.ndarray) -> tuple[float, float]:
    """Closed-form least squares avg ~= a + b * pred255."""
    b, a = np.polyfit(pred255, avg, 1)
    return float(a), float(b)


@torch.no_grad()
def mean_preds(model, dataset, indices, device: str, batch_size: int = 512):
    """Per-image mean predictions (0-255) and labels for a fixed index subset."""
    dl = DataLoader(Subset(dataset, indices), batch_size=batch_size, num_workers=0)
    model.eval()
    preds, avgs = [], []
    for batch in dl:
        x = batch["image"].to(device)
        preds.append((model(x).mean(dim=(1, 2, 3)) * 255.0).cpu().numpy())
        avgs.append(batch["avg_depth"].float().numpy())
    return np.concatenate(preds), np.concatenate(avgs)


def run_finetune(cfg: dict) -> dict:
    seed_all(cfg["seed"])
    d, tr, m, out = cfg["data"], cfg["train"], cfg["model"], cfg["out"]
    device = tr.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    pairs = list_sim_pairs(Path(d["sim_sem_dir"]), Path(d["sim_depth_dir"]))
    train_pairs, val_pairs = split_pairs(pairs, d["val_fraction"], cfg["seed"])
    if d.get("pseudo_sem_root"):
        pseudo = list_pseudo_pairs(Path(d["pseudo_sem_root"]), Path(d["pseudo_depth_root"]))
        print(f"mixing {len(pseudo)} pseudo pairs into {len(train_pairs)} sim train pairs")
        train_pairs = train_pairs + pseudo  # sim hold-out stays pure sim
    sim_ds = SimDataset(train_pairs, augment=tr["augment"], appearance=tr.get("appearance"))
    val_ds = SimDataset(val_pairs, augment=False)
    sim_dl = DataLoader(
        sim_ds, batch_size=tr["batch_size"], shuffle=True,
        num_workers=tr["num_workers"], pin_memory=(device == "cuda"), drop_last=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=tr["batch_size"], shuffle=False,
        num_workers=tr["num_workers"], pin_memory=(device == "cuda"),
    )
    if len(sim_dl) == 0:
        raise ValueError(f"no sim batches: {len(sim_ds)} images < batch_size {tr['batch_size']}")

    real_ds = RealDataset(Path(d["real_sem_root"]), Path(d["real_csv"]))
    real_dl = DataLoader(
        real_ds, batch_size=tr.get("real_batch", tr["batch_size"]), shuffle=True,
        num_workers=tr["num_workers"], pin_memory=(device == "cuda"), drop_last=True,
    )
    n_refit = min(int(tr.get("refit_sample", 8192)), len(real_ds))
    refit_idx = random.Random(cfg["seed"]).sample(range(len(real_ds)), n_refit)

    model = UnetTimm(m["encoder"], pretrained=False).to(device)
    model.load_state_dict(torch.load(cfg["init_ckpt"], map_location="cpu", weights_only=True))
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tr["epochs"] * len(sim_dl))
    lam = float(tr.get("lambda_real", 1.0))

    run_dir = Path(out["runs_dir"]) / out["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))
    best_cal, best_val, t0 = float("inf"), float("inf"), time.time()
    a, b = None, None

    for epoch in range(tr["epochs"]):
        pred255, avg = mean_preds(model, real_ds, refit_idx, device)
        a_new, b_new = fit_affine(pred255, avg)
        if a is None or abs(b_new) > 0.1:  # degenerate-slope guard
            a, b = a_new, b_new
        cal = float(np.sqrt(np.mean((avg - (a + b * pred255)) ** 2)))
        writer.add_scalar("real/cal_proxy_sample", cal, epoch)
        writer.add_scalar("real/fit_b", b, epoch)
        print(f"epoch {epoch}: refit a={a:.2f} b={b:.4f} cal_proxy(sample)={cal:.4f}")
        # epoch 0's refit scores the INIT weights — track but never checkpoint them
        if epoch > 0 and cal < best_cal:
            best_cal = cal
            torch.save(model.state_dict(), run_dir / "best.pt")

        model.train()
        real_iter = iter(real_dl)
        for step, batch in enumerate(tqdm(sim_dl, desc=f"epoch {epoch}", leave=False)):
            try:
                rbatch = next(real_iter)
            except StopIteration:
                real_iter = iter(real_dl)
                rbatch = next(real_iter)
            xs = batch["image"].to(device, non_blocking=True)
            ts = batch["target"].to(device, non_blocking=True)
            xr = rbatch["image"].to(device, non_blocking=True)
            ar = rbatch["avg_depth"].to(device, non_blocking=True).float()
            with torch.autocast(device, dtype=torch.bfloat16, enabled=tr["amp"]):
                l1 = torch.nn.functional.l1_loss(model(xs), ts)
                mean255 = model(xr).mean(dim=(1, 2, 3)) * 255.0
                lreal = (a + b * mean255 - ar).abs().mean() / 255.0
                loss = l1 + lam * lreal
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if step % 50 == 0:
                writer.add_scalar("train/l1", l1.item(), epoch * len(sim_dl) + step)
                writer.add_scalar("train/lreal", lreal.item(), epoch * len(sim_dl) + step)
        val_rmse = evaluate(model, val_dl, device)
        best_val = min(best_val, val_rmse)
        writer.add_scalar("val/rmse255", val_rmse, epoch)
        print(f"epoch {epoch}: sim val rmse255 = {val_rmse:.4f}")

    # final refit + score on the sample with the LAST weights (best.pt may be earlier)
    pred255, avg = mean_preds(model, real_ds, refit_idx, device)
    a_f, b_f = fit_affine(pred255, avg)
    cal_f = float(np.sqrt(np.mean((avg - (a_f + b_f * pred255)) ** 2)))
    if cal_f < best_cal:
        best_cal = cal_f
        torch.save(model.state_dict(), run_dir / "best.pt")

    row = {
        "run_name": out["run_name"],
        "encoder": m["encoder"],
        "epochs": tr["epochs"],
        "lr": tr["lr"],
        "lambda_real": lam,
        "seed": cfg["seed"],
        "best_cal_proxy_sample": round(best_cal, 4),
        "best_val_rmse255": round(best_val, 4),
        "cal_a": round(a_f, 3),
        "cal_b": round(b_f, 4),
        "wall_min": round((time.time() - t0) / 60, 1),
    }
    append_result(Path(out["results_csv"]), row)
    writer.close()
    return row
