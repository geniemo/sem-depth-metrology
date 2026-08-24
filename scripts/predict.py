"""Predict depth maps for a directory of SEM PNGs; optionally build submission zip."""
import argparse
from pathlib import Path

import torch
import yaml

from semdepth.infer import make_submission_zip, predict_dir
from semdepth.model import UnetTimm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--zip", default=None)
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model = UnetTimm(cfg["model"]["encoder"], pretrained=False)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=True))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = predict_dir(model, Path(args.input), Path(args.out), device, flip_tta=args.tta)
    print(f"wrote {n} depth maps to {args.out}")
    if args.zip:
        nz = make_submission_zip(Path(args.out), Path(args.zip))
        print(f"zipped {nz} files -> {args.zip}")


if __name__ == "__main__":
    main()
