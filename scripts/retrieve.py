"""Full retrieval inference over the test set.

Pipeline: bucket/case classifier -> per-case key library (translated or
itr-mean+blur) -> shift(-and-flip) cosine retrieval -> top-k GT depth blend ->
depth PNGs + submission zip.
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from semdepth.classify import BucketClassifier, predict_buckets
from semdepth.data import list_sim_pairs, load_image01
from semdepth.infer import make_submission_zip
from semdepth.retrieval import (blend_depths_aligned, build_keys, retrieve_batch,
                                standardize, variant_specs)

_CASES = ("Case_1", "Case_2", "Case_3", "Case_4")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--clf-ckpt", default="experiments/runs/clf_bucket/best.pt")
    ap.add_argument("--translated-root", default=None)
    ap.add_argument("--blur", type=float, default=0.7)
    ap.add_argument("--shift", type=int, default=2)
    ap.add_argument("--flips", action="store_true")
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--out", required=True)
    ap.add_argument("--zip", default=None)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_paths = sorted((root / "test" / "SEM").glob("*.png"))
    clf = BucketClassifier(pretrained=False)
    clf.load_state_dict(torch.load(args.clf_ckpt, map_location="cpu", weights_only=True))
    buckets = predict_buckets(clf, test_paths, device)
    print(f"test case distribution: {Counter((buckets + 1).tolist())}")

    pairs = list_sim_pairs(root / "simulation_data" / "SEM", root / "simulation_data" / "Depth")
    by_case = defaultdict(list)
    for p in pairs:
        by_case[p.case].append(p)

    n_written = 0
    sims_all = []
    for ci, case in enumerate(_CASES):
        cpaths = [p for p, b in zip(test_paths, buckets) if b == ci]
        if not cpaths:
            continue
        keys, depths = build_keys(by_case[case], blur_sigma=args.blur, device=device,
                                  translated_root=args.translated_root)
        for s in range(0, len(cpaths), args.chunk):
            chunk = cpaths[s:s + args.chunk]
            q = np.stack([standardize(load_image01(p)) for p in chunk])
            idx, sims, var = retrieve_batch(q, keys, device, shift=args.shift,
                                            flips=args.flips, topk=args.topk)
            specs = variant_specs(args.shift, args.flips)
            for j, p in enumerate(chunk):
                pred = blend_depths_aligned(depths, idx[j], sims[j], var[j], specs)
                Image.fromarray(pred, mode="L").save(out_dir / p.name)
                n_written += 1
            sims_all.extend(sims[:, 0].tolist())
        del keys
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"{case}: {len(cpaths)} images done")
    print(f"wrote {n_written} depth maps | top1 sim mean={np.mean(sims_all):.4f} "
          f"p10={np.percentile(sims_all, 10):.4f}")
    if args.zip:
        nz = make_submission_zip(out_dir, Path(args.zip))
        print(f"zipped {nz} -> {args.zip}")


if __name__ == "__main__":
    main()
