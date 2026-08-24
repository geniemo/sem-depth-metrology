"""Extend Case_1 WGAN-GP training from a checkpoint (published run used 300 epochs)."""
import argparse
import time
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from semdepth.classify import list_real_labeled
from semdepth.cyclegan import CycleWGanGP, RealGanDataset, SimGanDataset
from semdepth.data import list_sim_pairs
from semdepth.train import seed_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/cyclegan.yaml")
    ap.add_argument("--resume", required=True)
    ap.add_argument("--start-epoch", type=int, required=True)
    ap.add_argument("--until", type=int, default=300)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    seed_all(cfg["seed"] + args.start_epoch)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr, root = cfg["train"], Path(cfg["data"]["root"])
    out_dir = Path(cfg["out"]["dir"])

    pairs = [p for p in list_sim_pairs(root / "simulation_data" / "SEM",
                                       root / "simulation_data" / "Depth")
             if p.case == "Case_1"]
    real = [p for p, bi, _ in list_real_labeled(root / "train" / "SEM") if bi == 0]

    trainer = CycleWGanGP(device, dim=tr["dim"], n_res=tr["n_res"], dropout=tr["dropout"],
                          lr=tr["lr"], lambda_cycle=tr["lambda_cycle"],
                          lambda_idt=tr["lambda_idt"], lambda_gp=tr["lambda_gp"],
                          critic_iters=tr["critic_iters"])
    trainer.load_generators(torch.load(args.resume, map_location="cpu", weights_only=True))
    print(f"resumed from {args.resume}")
    sim_dl = DataLoader(SimGanDataset(pairs), batch_size=tr["batch_size"], shuffle=True,
                        num_workers=tr["num_workers"], drop_last=True)
    real_dl = DataLoader(RealGanDataset(real), batch_size=tr["batch_size"], shuffle=True,
                         num_workers=tr["num_workers"], drop_last=True)
    t0 = time.time()
    for epoch in range(args.start_epoch, args.until):
        log = trainer.train_epoch(sim_dl, real_dl)
        if epoch % 5 == 0 or epoch == args.until - 1:
            print(f"Case_1 epoch {epoch}: g={log.get('g', float('nan')):.3f} "
                  f"cyc={log.get('cyc', float('nan')):.3f} ({(time.time()-t0)/60:.1f} min)",
                  flush=True)
        if (epoch + 1) % tr["save_every"] == 0 or epoch == args.until - 1:
            torch.save(trainer.state(), out_dir / f"Case_1_ep{epoch + 1:03d}.pt")
    print("Case_1 extension done", flush=True)


if __name__ == "__main__":
    main()
