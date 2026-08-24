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
    pairs: list[SimPair],
    blur_sigma: float = 0.7,
    device: str = "cpu",
    translated_root: Path | None = None,
) -> tuple[torch.Tensor, list[Path]]:
    """L2-normalized key matrix (N, H*W) fp16 and the aligned GT depth paths.

    Default keys are itr-mean + blur (no-GAN alignment). With translated_root,
    keys are the GAN-translated images stored under the depth-map relative path
    (<root>/<Case_X>/<bucket>/<base>.png).
    """
    rows, depths = [], []
    for p in pairs:
        if translated_root is not None:
            rel = p.depth_path.relative_to(p.depth_path.parents[2])
            img = load_image01(Path(translated_root) / rel)
        else:
            img = np.mean([load_image01(q) for q in p.sem_paths], axis=0, dtype=np.float32)
            if blur_sigma > 0.05:
                img = _gaussian_blur(img, blur_sigma)
        rows.append(standardize(img).ravel())
        depths.append(p.depth_path)
    keys = torch.from_numpy(np.stack(rows)).to(device)
    keys = torch.nn.functional.normalize(keys, dim=1).half()
    return keys, depths


def variant_specs(shift: int, flips: bool) -> list[tuple[int, int, int]]:
    """(flip_code, dh, dw) for every searched variant; flip_code 0=id,1=h,2=v,3=hv."""
    codes = [0, 1, 2, 3] if flips else [0]
    return [(f, dh, dw) for f in codes
            for dh in range(-shift, shift + 1)
            for dw in range(-shift, shift + 1)]


def _apply_variant(q: torch.Tensor, spec: tuple[int, int, int]) -> torch.Tensor:
    f, dh, dw = spec
    if f == 1:
        q = torch.flip(q, [2])
    elif f == 2:
        q = torch.flip(q, [1])
    elif f == 3:
        q = torch.flip(q, [1, 2])
    return torch.roll(q, shifts=(dh, dw), dims=(1, 2))


def align_key_to_query(key_img: np.ndarray, spec: tuple[int, int, int]) -> np.ndarray:
    """Inverse-transform a matched key/GT image into the ORIGINAL query frame.

    The search shifts/flips the query to fit the key, so the key maps back with
    the inverse roll first, then the (self-inverse) flip.
    """
    f, dh, dw = spec
    out = np.roll(key_img, shift=(-dh, -dw), axis=(0, 1))
    if f == 1:
        out = out[:, ::-1]
    elif f == 2:
        out = out[::-1, :]
    elif f == 3:
        out = out[::-1, ::-1]
    return out.copy()


@torch.no_grad()
def retrieve_batch(
    queries: np.ndarray,  # (B, H, W) float32, already standardized
    keys: torch.Tensor,  # (N, D) L2-normalized fp16
    device: str,
    shift: int = 2,
    flips: bool = False,
    topk: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-k (indices, similarities, variant ids) per query; max over variants.

    variant ids index into variant_specs(shift, flips) and record WHICH
    shift/flip of the query produced each match (for output re-alignment).
    """
    q = torch.from_numpy(queries).to(device)
    specs = variant_specs(shift, flips)
    best = None
    best_var = None
    for vi, spec in enumerate(specs):
        variant = _apply_variant(q, spec)
        v = torch.nn.functional.normalize(variant.reshape(variant.shape[0], -1), dim=1).half()
        sim = v @ keys.T  # (B, N)
        if best is None:
            best = sim
            best_var = torch.zeros_like(sim, dtype=torch.int16)
        else:
            better = sim > best
            best = torch.where(better, sim, best)
            best_var = torch.where(better, torch.tensor(vi, dtype=torch.int16, device=sim.device), best_var)
    sims, idx = torch.topk(best.float(), k=topk, dim=1)
    var = torch.gather(best_var, 1, idx).cpu().numpy()
    return idx.cpu().numpy(), sims.cpu().numpy(), var


def blend_depths(depth_paths: list[Path], idx: np.ndarray, sims: np.ndarray) -> np.ndarray:
    """Softmax-weighted blend of the top-k GT depth maps -> uint8 (H, W)."""
    w = np.exp(sims - sims.max())
    w = w / w.sum()
    acc = None
    for i, wi in zip(idx, w):
        d = load_image01(depth_paths[int(i)]) * 255.0
        acc = wi * d if acc is None else acc + wi * d
    return np.clip(np.round(acc), 0, 255).astype(np.uint8)


@torch.no_grad()
def refine_shortlist(
    queries: np.ndarray,  # (B, H, W) standardized
    cand_idx: np.ndarray,  # (B, K) candidate key indices from the embedding stage
    keys: torch.Tensor,  # (N, D) L2-normalized fp16 pixel keys
    device: str,
    shift: int = 2,
    flips: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel shift-search cosine within each query's candidate shortlist.

    Returns (global key index, similarity, variant id) of the best candidate.
    """
    q = torch.from_numpy(queries).to(device)
    cand = torch.from_numpy(cand_idx.astype(np.int64)).to(device)
    kp = keys[cand]  # (B, K, D)
    specs = variant_specs(shift, flips)
    best = None
    best_var = None
    for vi, spec in enumerate(specs):
        v = torch.nn.functional.normalize(
            _apply_variant(q, spec).reshape(q.shape[0], -1), dim=1).half()
        sim = torch.einsum("bd,bkd->bk", v, kp)
        if best is None:
            best = sim
            best_var = torch.zeros_like(sim, dtype=torch.int16)
        else:
            better = sim > best
            best = torch.where(better, sim, best)
            best_var = torch.where(
                better, torch.tensor(vi, dtype=torch.int16, device=sim.device), best_var)
    j = best.float().argmax(dim=1)
    rows = torch.arange(q.shape[0], device=device)
    idx = cand[rows, j]
    return (idx.cpu().numpy(),
            best.float()[rows, j].cpu().numpy(),
            best_var[rows, j].cpu().numpy())


def blend_depths_aligned(
    depth_paths: list[Path],
    idx: np.ndarray,
    sims: np.ndarray,
    var_ids: np.ndarray,
    specs: list[tuple[int, int, int]],
) -> np.ndarray:
    """Softmax-weighted blend of top-k GT maps, each re-aligned to the query frame."""
    w = np.exp(sims - sims.max())
    w = w / w.sum()
    acc = None
    for i, wi, vi in zip(idx, w, var_ids):
        d = align_key_to_query(load_image01(depth_paths[int(i)]) * 255.0, specs[int(vi)])
        acc = wi * d if acc is None else acc + wi * d
    return np.clip(np.round(acc), 0, 255).astype(np.uint8)


def hole_mean_depth(depth_map255: np.ndarray, case: str) -> float:
    """Average depth below the surface over hole pixels (the avg_depth semantics)."""
    bg = CASE_BG[case]
    hole = depth_map255 < bg
    if not hole.any():
        return 0.0
    return float((bg - depth_map255[hole]).mean())
