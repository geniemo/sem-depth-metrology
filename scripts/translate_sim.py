"""Translate the whole sim library to real style with trained per-case generators."""
import argparse
from collections import defaultdict
from pathlib import Path

import torch

from semdepth.cyclegan import Generator, translate_pairs
from semdepth.data import list_sim_pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--ckpt-dir", default="experiments/runs/cyclegan")
    ap.add_argument("--epochs", required=True,
                    help="per-case checkpoint epochs, comma-separated (e.g. 120,60,60,60)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--n-res", type=int, default=2)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pairs = list_sim_pairs(Path(args.root) / "simulation_data" / "SEM",
                           Path(args.root) / "simulation_data" / "Depth")
    by_case = defaultdict(list)
    for p in pairs:
        by_case[p.case].append(p)
    total = 0
    epochs = [int(x) for x in args.epochs.split(",")]
    for case, ep in zip(("Case_1", "Case_2", "Case_3", "Case_4"), epochs):
        state = torch.load(Path(args.ckpt_dir) / f"{case}_ep{ep:03d}.pt",
                           map_location="cpu", weights_only=True)
        g = Generator(args.dim, args.n_res)
        g.load_state_dict(state["G_st"])
        n = translate_pairs(g, by_case[case], Path(args.out), device,
                            batch_size=512, num_workers=8)
        print(f"{case}: translated {n}")
        total += n
    print(f"total {total}")


if __name__ == "__main__":
    main()
