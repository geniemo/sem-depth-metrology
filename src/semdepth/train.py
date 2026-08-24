import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from semdepth.data import SimDataset, list_sim_pairs, split_pairs
from semdepth.model import UnetTimm
from semdepth.results import append_result


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    se_sum, n = 0.0, 0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        t = batch["target"].to(device, non_blocking=True)
        p = model(x)
        se_sum += ((p - t) ** 2).sum().item()
        n += t.numel()
    return 255.0 * (se_sum / n) ** 0.5


def run_training(cfg: dict) -> dict:
    seed_all(cfg["seed"])
    d, tr, m, out = cfg["data"], cfg["train"], cfg["model"], cfg["out"]
    device = tr.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    pairs = list_sim_pairs(Path(d["sim_sem_dir"]), Path(d["sim_depth_dir"]))
    train_pairs, val_pairs = split_pairs(pairs, d["val_fraction"], cfg["seed"])
    train_ds = SimDataset(train_pairs, augment=tr["augment"])
    val_ds = SimDataset(val_pairs, augment=False)
    train_dl = DataLoader(
        train_ds, batch_size=tr["batch_size"], shuffle=True,
        num_workers=tr["num_workers"], pin_memory=(device == "cuda"), drop_last=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=tr["batch_size"], shuffle=False,
        num_workers=tr["num_workers"], pin_memory=(device == "cuda"),
    )

    model = UnetTimm(m["encoder"], pretrained=m["pretrained"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=tr["epochs"] * max(1, len(train_dl))
    )

    run_dir = Path(out["runs_dir"]) / out["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))
    best, t0 = float("inf"), time.time()

    for epoch in range(tr["epochs"]):
        model.train()
        for step, batch in enumerate(tqdm(train_dl, desc=f"epoch {epoch}", leave=False)):
            x = batch["image"].to(device, non_blocking=True)
            t = batch["target"].to(device, non_blocking=True)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=tr["amp"]):
                loss = torch.nn.functional.l1_loss(model(x), t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if step % 50 == 0:
                writer.add_scalar("train/l1", loss.item(), epoch * len(train_dl) + step)
        val_rmse = evaluate(model, val_dl, device)
        writer.add_scalar("val/rmse255", val_rmse, epoch)
        print(f"epoch {epoch}: val rmse255 = {val_rmse:.4f}")
        if val_rmse < best:
            best = val_rmse
            torch.save(model.state_dict(), run_dir / "best.pt")

    row = {
        "run_name": out["run_name"],
        "encoder": m["encoder"],
        "epochs": tr["epochs"],
        "batch_size": tr["batch_size"],
        "lr": tr["lr"],
        "augment": tr["augment"],
        "seed": cfg["seed"],
        "n_train_imgs": len(train_ds),
        "n_val_imgs": len(val_ds),
        "best_val_rmse255": round(best, 4),
        "wall_min": round((time.time() - t0) / 60, 1),
    }
    append_result(Path(out["results_csv"]), row)
    writer.close()
    return row
