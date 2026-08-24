"""Generate pseudo depth maps for the real train set with a trained checkpoint."""
import argparse
from pathlib import Path

import torch
import yaml

from semdepth.infer import predict_dir
from semdepth.model import UnetTimm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--real-sem-root", default="data/raw/train/SEM")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model = UnetTimm(cfg["model"]["encoder"], pretrained=False)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=True))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = predict_dir(model, Path(args.real_sem_root), Path(args.out), device,
                    batch_size=512, num_workers=8, recursive=True)
    print(f"wrote {n} pseudo depth maps to {args.out}")


if __name__ == "__main__":
    main()
