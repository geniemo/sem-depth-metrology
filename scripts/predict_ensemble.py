"""Ensemble inference: average predictions of several (config, ckpt) members.

--member is repeatable: --member configs/a.yaml:experiments/runs/a/best.pt
"""
import argparse
from pathlib import Path

import torch
import yaml

from semdepth.infer import make_submission_zip, predict_dir
from semdepth.model import UnetTimm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", action="append", required=True,
                    help="config.yaml:checkpoint.pt (repeatable)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--zip", default=None)
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()
    models = []
    for spec in args.member:
        cfg_path, ckpt = spec.split(":")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        m = UnetTimm(cfg["model"]["encoder"], pretrained=False)
        m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        models.append(m)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = predict_dir(models, Path(args.input), Path(args.out), device,
                    batch_size=512, flip_tta=args.tta, num_workers=8)
    print(f"wrote {n} depth maps ({len(models)} members, tta={args.tta}) to {args.out}")
    if args.zip:
        nz = make_submission_zip(Path(args.out), Path(args.zip))
        print(f"zipped {nz} files -> {args.zip}")


if __name__ == "__main__":
    main()
