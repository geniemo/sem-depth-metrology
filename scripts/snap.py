"""Regression-guided exemplar snapping over the test set.

The regression model translates a real image into depth space; the prediction
is then snapped to the nearest library GT map (L2, cyclic-shift search) and the
matched GT is emitted re-aligned to the query frame (pure exemplar, --out-a).
Optionally an existing regression-prediction directory is averaged in 50/50
(--mix-pred-dir, --out-b) — exemplar boundary-band errors and regression blur
errors are complementary, so the mix beats both in the in-domain harness.
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from semdepth.classify import BucketClassifier, predict_buckets
from semdepth.data import list_sim_pairs, load_image01
from semdepth.infer import make_submission_zip
from semdepth.model import UnetTimm
from semdepth.retrieval import (
    align_key_to_query,
    build_gt_keys,
    snap_batch,
    variant_specs,
)

_CASES = ("Case_1", "Case_2", "Case_3", "Case_4")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--clf-ckpt", default="experiments/runs/clf_bucket/best.pt")
    ap.add_argument("--reg-config", default="configs/e12d_cns_l1l2_weak.yaml")
    ap.add_argument("--reg-ckpt", default="experiments/runs/e12d_cns_l1l2_weak/best.pt")
    ap.add_argument("--shift", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--out-a", required=True, help="pure exemplar output dir")
    ap.add_argument("--zip-a", default=None)
    ap.add_argument("--mix-pred-dir", default=None,
                    help="existing regression PNG dir to 50/50-mix with the exemplar")
    ap.add_argument("--out-b", default=None)
    ap.add_argument("--zip-b", default=None)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.root)
    out_a = Path(args.out_a)
    out_a.mkdir(parents=True, exist_ok=True)
    out_b = Path(args.out_b) if args.out_b else None
    if out_b:
        out_b.mkdir(parents=True, exist_ok=True)

    test_paths = sorted((root / "test" / "SEM").glob("*.png"))
    clf = BucketClassifier(pretrained=False)
    clf.load_state_dict(torch.load(args.clf_ckpt, map_location="cpu", weights_only=True))
    buckets = predict_buckets(clf, test_paths, device)
    print(f"test case distribution: {Counter((buckets + 1).tolist())}")

    with open(args.reg_config) as f:
        rcfg = yaml.safe_load(f)
    reg = UnetTimm(rcfg["model"]["encoder"], pretrained=False)
    reg.load_state_dict(torch.load(args.reg_ckpt, map_location="cpu", weights_only=True))
    reg = reg.to(device).eval()

    pairs = list_sim_pairs(root / "simulation_data" / "SEM", root / "simulation_data" / "Depth")
    by_case = defaultdict(list)
    for p in pairs:
        by_case[p.case].append(p)
    specs = variant_specs(args.shift, False)

    n = 0
    for ci, case in enumerate(_CASES):
        cpaths = [p for p, b in zip(test_paths, buckets) if b == ci]
        if not cpaths:
            continue
        depths = [p.depth_path for p in by_case[case]]
        keys, sq = build_gt_keys(depths, device)
        for s in range(0, len(cpaths), args.chunk):
            chunk = cpaths[s:s + args.chunk]
            x = torch.stack([torch.from_numpy(load_image01(p)).unsqueeze(0) for p in chunk])
            with torch.no_grad():
                preds = reg(x.to(device)).squeeze(1).cpu().numpy()
            idx, _vals, var = snap_batch(preds, keys, sq, device, shift=args.shift, topk=1)
            for j, p in enumerate(chunk):
                gt = load_image01(depths[int(idx[j, 0])]) * 255.0
                ex = align_key_to_query(gt, specs[int(var[j, 0])])
                ex_u8 = np.clip(np.round(ex), 0, 255).astype(np.uint8)
                Image.fromarray(ex_u8, mode="L").save(out_a / p.name)
                if out_b:
                    rp = np.asarray(Image.open(Path(args.mix_pred_dir) / p.name),
                                    dtype=np.float32)
                    mix = np.clip(np.round(0.5 * ex + 0.5 * rp), 0, 255).astype(np.uint8)
                    Image.fromarray(mix, mode="L").save(out_b / p.name)
                n += 1
        del keys, sq
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"{case}: {len(cpaths)} done")
    print(f"wrote {n} exemplar maps")
    if args.zip_a:
        print(f"zipped {make_submission_zip(out_a, Path(args.zip_a))} -> {args.zip_a}")
    if args.zip_b and out_b:
        print(f"zipped {make_submission_zip(out_b, Path(args.zip_b))} -> {args.zip_b}")


if __name__ == "__main__":
    main()
