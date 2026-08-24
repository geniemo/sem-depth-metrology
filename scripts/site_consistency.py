"""Within-site prediction disagreement: a label-free real-domain stability probe.

Images inside one site are repeated acquisitions of (nearly) the same structure
(within-site image diffs match the sim itr noise level), so a model's
predictions inside a site should agree. Disagreement = per-pixel std across the
site's predictions, averaged, in 0-255 units. Stability is necessary but not
sufficient for accuracy (a self-trained model can be confidently wrong), so
this is a diagnostic, not a gate.
"""
import argparse
import random
import re
from pathlib import Path

import numpy as np
import torch
import yaml

from semdepth.data import load_image01
from semdepth.model import UnetTimm

_SITE_RE = re.compile(r"^depth_(?P<bucket>\d+)_site_(?P<site>\d+)$")


@torch.no_grad()
def site_disagreement(model, site_dirs, device: str) -> float:
    vals = []
    for d in site_dirs:
        imgs = [load_image01(p) for p in sorted(d.glob("*.png"))]
        x = torch.from_numpy(np.stack(imgs)).unsqueeze(1).to(device)
        pred = model(x).squeeze(1).cpu().numpy() * 255.0
        vals.append(float(pred.std(axis=0).mean()))
    return float(np.mean(vals))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", action="append", required=True,
                    help="label:config.yaml:checkpoint.pt (repeatable)")
    ap.add_argument("--real-sem-root", default="data/raw/train/SEM")
    ap.add_argument("--n-sites", type=int, default=300)
    args = ap.parse_args()
    root = Path(args.real_sem_root)
    sites = [d for bucket in sorted(root.iterdir()) if bucket.is_dir()
             for d in sorted(bucket.iterdir()) if d.is_dir()]
    picked = random.Random(11).sample(sites, min(args.n_sites, len(sites)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for spec in args.member:
        label, cfg_path, ckpt = spec.split(":")
        cfg = yaml.safe_load(open(cfg_path))
        model = UnetTimm(cfg["model"]["encoder"], pretrained=False)
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        model = model.to(device).eval()
        d = site_disagreement(model, picked, device)
        print(f"{label}: within-site prediction std = {d:.3f} (0-255, {len(picked)} sites)")


if __name__ == "__main__":
    main()
