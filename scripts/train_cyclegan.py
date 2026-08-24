"""Train the per-case WGAN-GP cycle translator (case 1 long, cases 2-4 warm-started)."""
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
from semdepth.retrieval import BUCKET_TO_CASE
from semdepth.train import seed_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    seed_all(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr, root = cfg["train"], Path(cfg["data"]["root"])
    out_dir = Path(cfg["out"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = list_sim_pairs(root / "simulation_data" / "SEM", root / "simulation_data" / "Depth")
    by_case = defaultdict(list)
    for p in pairs:
        by_case[p.case].append(p)
    real_by_case = defaultdict(list)
    case_of_bucket = {b: c for b, c in BUCKET_TO_CASE.items()}
    for path, bi, site in list_real_labeled(root / "train" / "SEM"):
        bucket = ("110", "120", "130", "140")[bi]
        real_by_case[case_of_bucket[bucket]].append(path)

    prev_state = None
    for case in ("Case_1", "Case_2", "Case_3", "Case_4"):
        epochs = tr["epochs_first"] if prev_state is None else tr["epochs_warm"]
        trainer = CycleWGanGP(device, dim=tr["dim"], n_res=tr["n_res"],
                              dropout=tr["dropout"], lr=tr["lr"],
                              lambda_cycle=tr["lambda_cycle"], lambda_idt=tr["lambda_idt"],
                              lambda_gp=tr["lambda_gp"], critic_iters=tr["critic_iters"])
        if prev_state is not None:
            trainer.load_generators(prev_state)
            print(f"{case}: warm-started from previous case")
        sim_dl = DataLoader(SimGanDataset(by_case[case]), batch_size=tr["batch_size"],
                            shuffle=True, num_workers=tr["num_workers"], drop_last=True)
        real_dl = DataLoader(RealGanDataset(real_by_case[case]), batch_size=tr["batch_size"],
                             shuffle=True, num_workers=tr["num_workers"], drop_last=True)
        t0 = time.time()
        for epoch in range(epochs):
            log = trainer.train_epoch(sim_dl, real_dl)
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"{case} epoch {epoch}: g={log.get('g', float('nan')):.3f} "
                      f"cyc={log.get('cyc', float('nan')):.3f} "
                      f"d_t={log.get('d_t', float('nan')):.3f} "
                      f"({(time.time() - t0) / 60:.1f} min)", flush=True)
            if (epoch + 1) % tr["save_every"] == 0 or epoch == epochs - 1:
                torch.save(trainer.state(), out_dir / f"{case}_ep{epoch + 1:03d}.pt")
        prev_state = trainer.state()
        print(f"{case}: done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
