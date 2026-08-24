"""Two-stage retrieval inference: embedding shortlist -> pixel shift-refined exemplar.

Stage 1: the contrastive identity embedding ranks the case's sim library and
takes a top-K shortlist per query (this is what fixes the cross-domain ranking
that plain pixel cosine got wrong). Stage 2: pixel cosine with cyclic-shift
search WITHIN the shortlist picks the final exemplar and its alignment, and the
matched GT depth map is emitted re-aligned to the query frame.
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from semdepth.classify import BucketClassifier, predict_buckets
from semdepth.data import list_sim_pairs, load_image01
from semdepth.embed import EmbedNet, embed_paths
from semdepth.infer import make_submission_zip
from semdepth.retrieval import (
    align_key_to_query,
    build_keys,
    refine_shortlist,
    standardize,
    variant_specs,
)

_CASES = ("Case_1", "Case_2", "Case_3", "Case_4")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--clf-ckpt", default="experiments/runs/clf_bucket/best.pt")
    ap.add_argument("--embed-ckpt", required=True)
    ap.add_argument("--embed-encoder", default="resnet18")
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--blur", type=float, default=0.7)
    ap.add_argument("--shift", type=int, default=2)
    ap.add_argument("--shortlist", type=int, default=50)
    ap.add_argument("--chunk", type=int, default=2048)
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

    net = EmbedNet(args.embed_encoder, dim=args.embed_dim, pretrained=False)
    net.load_state_dict(torch.load(args.embed_ckpt, map_location="cpu", weights_only=True))

    pairs = list_sim_pairs(root / "simulation_data" / "SEM", root / "simulation_data" / "Depth")
    by_case = defaultdict(list)
    for p in pairs:
        by_case[p.case].append(p)

    specs = variant_specs(args.shift, False)
    n_written = 0
    sims_all = []
    for ci, case in enumerate(_CASES):
        cpaths = [p for p, b in zip(test_paths, buckets) if b == ci]
        if not cpaths:
            continue
        cpairs = by_case[case]
        keys_pix, depths = build_keys(cpairs, blur_sigma=args.blur, device=device)
        keys_emb = torch.from_numpy(
            embed_paths(net, [p.sem_paths[0] for p in cpairs], device)).to(device)
        for s in range(0, len(cpaths), args.chunk):
            chunk = cpaths[s:s + args.chunk]
            qs = np.stack([standardize(load_image01(p)) for p in chunk])
            q_emb = torch.from_numpy(
                embed_paths(net, chunk, device)).to(device)
            cand = torch.topk(q_emb @ keys_emb.T, k=args.shortlist, dim=1).indices.cpu().numpy()
            idx, sims, var = refine_shortlist(qs, cand, keys_pix, device, shift=args.shift)
            for j, p in enumerate(chunk):
                gt = load_image01(depths[int(idx[j])]) * 255.0
                pred = np.clip(np.round(align_key_to_query(gt, specs[int(var[j])])),
                               0, 255).astype(np.uint8)
                Image.fromarray(pred, mode="L").save(out_dir / p.name)
                n_written += 1
            sims_all.extend(sims.tolist())
        del keys_pix, keys_emb
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"{case}: {len(cpaths)} images done")
    print(f"wrote {n_written} depth maps | refine sim mean={np.mean(sims_all):.4f}")
    if args.zip:
        nz = make_submission_zip(out_dir, Path(args.zip))
        print(f"zipped {nz} -> {args.zip}")


if __name__ == "__main__":
    main()
