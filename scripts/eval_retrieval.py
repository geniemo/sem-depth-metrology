"""P1 gate: evaluate no-GAN retrieval on REAL TRAIN (bucket -> case is known).

Per case: build the sim key library (itr-mean + blur sigma* + standardize),
retrieve top-k for a site-sampled set of real images, and report
  (a) avg-depth RMSE/corr: hole_mean_depth(top-1 GT) vs the site's avg_depth
  (b) site retrieval consistency: images of one site matching the site's modal key
  (c) top-1 similarity distribution.
"""
import argparse
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from semdepth.data import list_sim_pairs, load_image01
from semdepth.retrieval import (
    BUCKET_TO_CASE,
    build_keys,
    hole_mean_depth,
    retrieve_batch,
    standardize,
)

_SITE_RE = re.compile(r"^depth_(?P<bucket>\d+)_site_(?P<site>\d+)$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--sites-per-bucket", type=int, default=60)
    ap.add_argument("--blur", type=float, default=0.7)
    ap.add_argument("--shift", type=int, default=2)
    ap.add_argument("--flips", action="store_true")
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()
    root = Path(args.root)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pairs = list_sim_pairs(root / "simulation_data" / "SEM", root / "simulation_data" / "Depth")
    by_case = defaultdict(list)
    for p in pairs:
        by_case[p.case].append(p)

    df = pd.read_csv(root / "train" / "average_depth.csv")
    site_avg = {str(k): float(v) for k, v in zip(df.iloc[:, 0], df.iloc[:, 1])}
    sites_by_bucket = defaultdict(list)
    for key in site_avg:
        m = _SITE_RE.match(key)
        sites_by_bucket[m["bucket"]].append((key, m["site"]))

    rng = random.Random(3)
    all_err, all_sims = [], []
    for bucket in ("110", "120", "130", "140"):
        case = BUCKET_TO_CASE[bucket]
        cpairs = by_case[case]
        keys, depths = build_keys(cpairs, blur_sigma=args.blur, device=device)

        picked = rng.sample(sites_by_bucket[bucket], args.sites_per_bucket)
        site_hits, errs, sims_c = [], [], []
        for key_name, site_id in picked:
            site_dir = root / "train" / "SEM" / f"Depth_{bucket}" / f"site_{site_id}"
            imgs = np.stack([standardize(load_image01(p))
                             for p in sorted(site_dir.glob("*.png"))])
            idx, sims = retrieve_batch(imgs, keys, device, shift=args.shift,
                                       flips=args.flips, topk=1)
            top1 = idx[:, 0]
            modal = Counter(top1.tolist()).most_common(1)[0]
            site_hits.append(modal[1] / len(top1))
            sims_c.extend(sims[:, 0].tolist())
            # avg-depth check with the site's modal retrieved structure
            gt = (load_image01(depths[modal[0]]) * 255.0)
            errs.append(hole_mean_depth(gt, case) - site_avg[key_name])
        errs = np.array(errs)
        print(f"[{case} / Depth_{bucket}] avg-depth RMSE={np.sqrt((errs**2).mean()):.3f} "
              f"bias={errs.mean():+.3f} | site-consistency={np.mean(site_hits):.3f} "
              f"| top1 sim mean={np.mean(sims_c):.4f} p10={np.percentile(sims_c, 10):.4f}")
        all_err.extend(errs.tolist())
        all_sims.extend(sims_c)
        del keys
        torch.cuda.empty_cache() if device == "cuda" else None

    all_err = np.array(all_err)
    print(f"[ALL] avg-depth RMSE={np.sqrt((all_err**2).mean()):.3f} "
          f"corr-target: beat regression cal-proxy 1.84 | "
          f"top1 sim mean={np.mean(all_sims):.4f}")


if __name__ == "__main__":
    main()
