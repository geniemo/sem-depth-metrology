"""Train the cross-domain structure-identity embedding; monitor the ranking oracle."""
import argparse
import random
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from semdepth.classify import list_real_labeled
from semdepth.data import list_sim_pairs
from semdepth.embed import (
    EmbedNet,
    PairViewDataset,
    nt_xent,
    real_site_identities,
    sim_identities,
    site_consistency_oracle,
)
from semdepth.results import append_result
from semdepth.train import seed_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    seed_all(cfg["seed"])
    tr = cfg["train"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(cfg["data"]["root"])

    pairs = list_sim_pairs(root / "simulation_data" / "SEM", root / "simulation_data" / "Depth")
    translated = cfg["data"].get("translated_root")
    sims = sim_identities(pairs, translated_root=Path(translated) if translated else None)
    records = list_real_labeled(root / "train" / "SEM")
    # oracle uses bucket-110 hold-out sites vs Case_1 keys
    site_ids = sorted({site for _, bi, site in records if bi == 0})
    holdout = set(random.Random(cfg["seed"]).sample(site_ids, cfg["data"]["holdout_sites"]))
    reals = real_site_identities([r for r in records if r[2] not in holdout])
    holdout_sites = real_site_identities([r for r in records if r[2] in holdout])
    key_paths = [p.sem_paths[0] for p in pairs if p.case == "Case_1"]
    print(f"identities: sim {len(sims)} + real {len(reals)}x{tr.get('real_oversample',1)} | holdout {len(holdout_sites)}")

    reals_over = reals * int(tr.get("real_oversample", 1))
    ds = PairViewDataset(sims + reals_over, max_shift=tr["max_shift"], seed=cfg["seed"])
    dl = DataLoader(ds, batch_size=tr["batch_size"], shuffle=True,
                    num_workers=tr["num_workers"], drop_last=True)
    model = EmbedNet(tr["encoder"], dim=tr["dim"], pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tr["epochs"] * len(dl))

    out_dir = Path(cfg["out"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    best, t0 = 0.0, time.time()
    for epoch in range(tr["epochs"]):
        model.train()
        for batch in dl:
            za = model(batch["a"].to(device, non_blocking=True))
            zb = model(batch["b"].to(device, non_blocking=True))
            loss = nt_xent(za, zb, temperature=tr["temperature"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        o = site_consistency_oracle(model, key_paths, holdout_sites, device)
        print(f"epoch {epoch}: loss={loss.item():.4f} "
              f"consistency={o['consistency']:.4f} diversity={o['diversity']:.3f} "
              f"score={o['score']:.4f} ({(time.time()-t0)/60:.1f} min)", flush=True)
        if o["score"] > best:
            best = o["score"]
            torch.save(model.state_dict(), out_dir / "best.pt")
    append_result(Path(cfg["out"]["results_csv"]), {
        "run_name": cfg["out"]["run_name"], "encoder": tr["encoder"],
        "epochs": tr["epochs"], "lr": tr["lr"],
        "best_oracle_score": round(best, 4),
        "wall_min": round((time.time() - t0) / 60, 1),
    })
    print({"best_oracle_score": best})


if __name__ == "__main__":
    main()
