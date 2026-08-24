"""Cross-domain structure retrieval (re-identification) against the sim library.

The test/real images are re-acquisitions of the same structure population as the
simulation library (established by reverse-engineering the published 1st-place
solution and by our data probes: per-case constant depth background 140/150/160/170,
hole floor 30, real Depth_<110..140> buckets = sim Case_<1..4>). Retrieval:

  key   = per-case sim library image: mean(itr0, itr1) -> Gaussian blur sigma*
          -> per-image standardization -> flattened, L2-normalized
  query = real/test image -> per-image standardization -> flattened
  score = max cosine similarity over cyclic +-shift (and optional flip) variants
  pred  = softmax-weighted blend of the top-k matched GT depth maps

Per-image standardization removes the brightness/contrast domain gap for the
MATCHING step only (depth then comes from the retrieved GT, so the r=-0.977
brightness cue is not needed here); itr-mean + calibrated blur (E6/E9) close the
texture gap without a GAN.
"""
from pathlib import Path

import numpy as np
import torch

from semdepth.data import SimPair, _gaussian_blur, load_image01

# verified constants (data probes 2026-08-24): depth background per case, hole floor
CASE_BG = {"Case_1": 140, "Case_2": 150, "Case_3": 160, "Case_4": 170}
HOLE_FLOOR = 30
BUCKET_TO_CASE = {"110": "Case_1", "120": "Case_2", "130": "Case_3", "140": "Case_4"}


def standardize(img: np.ndarray) -> np.ndarray:
    """Per-image zero-mean unit-std (matching space; kills brightness/contrast)."""
    return ((img - img.mean()) / (img.std() + 1e-8)).astype(np.float32)


def build_keys(
    pairs: list[SimPair], blur_sigma: float = 0.7, device: str = "cpu"
) -> tuple[torch.Tensor, list[Path]]:
    """L2-normalized key matrix (N, H*W) fp16 and the aligned GT depth paths."""
    rows, depths = [], []
    for p in pairs:
        img = np.mean([load_image01(q) for q in p.sem_paths], axis=0, dtype=np.float32)
        if blur_sigma > 0.05:
            img = _gaussian_blur(img, blur_sigma)
        rows.append(standardize(img).ravel())
        depths.append(p.depth_path)
    keys = torch.from_numpy(np.stack(rows)).to(device)
    keys = torch.nn.functional.normalize(keys, dim=1).half()
    return keys, depths


def _query_variants(q: torch.Tensor, shift: int, flips: bool) -> list[torch.Tensor]:
    """Cyclic-shift (and optional flip) variants of a query batch (B, H, W)."""
    bases = [q]
    if flips:
        bases += [torch.flip(q, [2]), torch.flip(q, [1]), torch.flip(q, [1, 2])]
    out = []
    for b in bases:
        for dh in range(-shift, shift + 1):
            for dw in range(-shift, shift + 1):
                out.append(torch.roll(b, shifts=(dh, dw), dims=(1, 2)))
    return out


@torch.no_grad()
def retrieve_batch(
    queries: np.ndarray,  # (B, H, W) float32, already standardized
    keys: torch.Tensor,  # (N, D) L2-normalized fp16
    device: str,
    shift: int = 2,
    flips: bool = False,
    topk: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k key indices and cosine similarities per query (max over variants)."""
    q = torch.from_numpy(queries).to(device)
    best = None
    for variant in _query_variants(q, shift, flips):
        v = torch.nn.functional.normalize(variant.reshape(variant.shape[0], -1), dim=1).half()
        sim = v @ keys.T  # (B, N)
        best = sim if best is None else torch.maximum(best, sim)
    sims, idx = torch.topk(best.float(), k=topk, dim=1)
    return idx.cpu().numpy(), sims.cpu().numpy()


def blend_depths(depth_paths: list[Path], idx: np.ndarray, sims: np.ndarray) -> np.ndarray:
    """Softmax-weighted blend of the top-k GT depth maps -> uint8 (H, W)."""
    w = np.exp(sims - sims.max())
    w = w / w.sum()
    acc = None
    for i, wi in zip(idx, w):
        d = load_image01(depth_paths[int(i)]) * 255.0
        acc = wi * d if acc is None else acc + wi * d
    return np.clip(np.round(acc), 0, 255).astype(np.uint8)


def hole_mean_depth(depth_map255: np.ndarray, case: str) -> float:
    """Average depth below the surface over hole pixels (the avg_depth semantics)."""
    bg = CASE_BG[case]
    hole = depth_map255 < bg
    if not hole.any():
        return 0.0
    return float((bg - depth_map255[hole]).mean())
