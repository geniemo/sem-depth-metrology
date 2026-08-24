import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

_ITR_RE = re.compile(r"^(?P<base>.+)_itr(?P<k>\d+)$")
_SITE_RE = re.compile(r"^depth_(?P<bucket>\d+)_site_(?P<site>\d+)$")


@dataclass(frozen=True)
class SimPair:
    group_id: str  # structure identity: depth-map stem, shared across cases
    case: str
    bucket: str
    sem_paths: tuple[Path, ...]
    depth_path: Path


def load_image01(path: Path) -> np.ndarray:
    """Load 8-bit grayscale PNG as float32 [H,W] in [0,1]."""
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def list_sim_pairs(sem_root: Path, depth_root: Path) -> list[SimPair]:
    """Pair SEM <case>/<bucket>/<base>_itr<k>.png with Depth <case>/<bucket>/<base>.png."""
    sem_root, depth_root = Path(sem_root), Path(depth_root)
    by_rel: dict[Path, list[Path]] = {}
    for p in sorted(sem_root.rglob("*.png")):
        m = _ITR_RE.match(p.stem)
        if m is None:
            raise ValueError(f"unexpected SEM name (no _itr suffix): {p}")
        rel = p.relative_to(sem_root).parent / f"{m['base']}.png"
        by_rel.setdefault(rel, []).append(p)
    n_depth = sum(1 for _ in depth_root.rglob("*.png"))
    if n_depth != len(by_rel):
        raise ValueError(f"{n_depth} depth maps but {len(by_rel)} SEM groups")
    pairs = []
    for rel, sems in sorted(by_rel.items()):
        depth = depth_root / rel
        if not depth.exists():
            raise ValueError(f"depth map missing: {depth}")
        parts = rel.parts
        case = parts[0] if len(parts) > 1 else ""
        bucket = parts[1] if len(parts) > 2 else ""
        pairs.append(SimPair(rel.stem, case, bucket, tuple(sorted(sems)), depth))
    return pairs


def split_pairs(
    pairs: list[SimPair], val_fraction: float, seed: int
) -> tuple[list[SimPair], list[SimPair]]:
    """Structure-level split: a group_id never appears on both sides.

    The same structure is simulated under every case (and twice per case via
    itr0/itr1); splitting by anything finer would leak it across the boundary.
    """
    groups = sorted({p.group_id for p in pairs})
    random.Random(seed).shuffle(groups)
    n_val = max(1, round(len(groups) * val_fraction))
    val_groups = set(groups[:n_val])
    train = [p for p in pairs if p.group_id not in val_groups]
    val = [p for p in pairs if p.group_id in val_groups]
    return train, val


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.copy()).unsqueeze(0)


def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur via shifted-slice accumulation (edge padding)."""
    radius = max(1, int(3.0 * sigma + 0.5))
    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(xs ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    out = image
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (radius, radius)
        padded = np.pad(out, pad, mode="edge")
        acc = np.zeros_like(out)
        for i, w in enumerate(kernel):
            sl = [slice(None), slice(None)]
            sl[axis] = slice(i, i + out.shape[axis])
            acc += w * padded[tuple(sl)]
        out = acc
    return out.astype(np.float32)


def _apply_appearance(image: np.ndarray, spec: dict) -> np.ndarray:
    """Input-only appearance jitter (brightness/contrast/blur/noise), [0,1] domain.

    Randomizes the sim appearance toward the real domain's wider band. Applied
    to the SEM image only — never to the depth target.
    """
    lo, hi = spec.get("contrast", (1.0, 1.0))
    c = random.uniform(lo, hi)
    lo, hi = spec.get("brightness", (0.0, 0.0))
    b = random.uniform(lo, hi)
    image = c * (image - 0.5) + 0.5 + b
    lo, hi = spec.get("blur_sigma", (0.0, 0.0))
    s = random.uniform(lo, hi)
    if s > 0.05:
        image = _gaussian_blur(image, s)
    lo, hi = spec.get("noise_std", (0.0, 0.0))
    n = random.uniform(lo, hi)
    if n > 0:
        image = image + np.random.normal(0.0, n, image.shape).astype(np.float32)
    return np.clip(image, 0.0, 1.0)


class SimDataset(Dataset):
    """(SEM image, depth map) pairs.

    input_mode="single": one item per SEM realization (itr) — the default.
    input_mode="itr_mean": one item per pair; the input is the pixel mean of the
    pair's realizations (mimics the frame-averaged look of real acquisitions
    while preserving absolute brightness, the primary depth cue).
    """

    def __init__(
        self,
        pairs: list[SimPair],
        augment: bool = False,
        appearance: dict | None = None,
        input_mode: str = "single",
    ):
        if input_mode not in ("single", "itr_mean"):
            raise ValueError(f"unknown input_mode: {input_mode}")
        if input_mode == "itr_mean":
            self.items = [(p.sem_paths, p.depth_path) for p in pairs]
        else:
            self.items = [
                ((p.sem_paths[k],), p.depth_path)
                for p in pairs
                for k in range(len(p.sem_paths))
            ]
        self.augment = augment
        self.appearance = appearance

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        sem_paths, depth_path = self.items[i]
        imgs = [load_image01(p) for p in sem_paths]
        image = imgs[0] if len(imgs) == 1 else np.mean(imgs, axis=0, dtype=np.float32)
        target = load_image01(depth_path)
        if self.augment:
            if random.random() < 0.5:
                image, target = image[:, ::-1], target[:, ::-1]
            if random.random() < 0.5:
                image, target = image[::-1, :], target[::-1, :]
        if self.appearance:
            image = _apply_appearance(image, self.appearance)
        return {"image": _to_tensor(image), "target": _to_tensor(target)}


class RealDataset(Dataset):
    """Real-domain SEM images; the average-depth label is shared per site."""

    def __init__(self, sem_root: Path, csv_path: Path):
        df = pd.read_csv(csv_path)
        self.records: list[tuple[Path, float]] = []
        for key, avg in zip(df.iloc[:, 0], df.iloc[:, 1]):
            m = _SITE_RE.match(str(key))
            if m is None:
                raise ValueError(f"unexpected site key in csv: {key}")
            site_dir = Path(sem_root) / f"Depth_{m['bucket']}" / f"site_{m['site']}"
            pngs = sorted(site_dir.glob("*.png"))
            if not pngs:
                raise ValueError(f"no images for site {key}: {site_dir}")
            self.records += [(p, float(avg)) for p in pngs]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        path, avg = self.records[i]
        return {"image": _to_tensor(load_image01(path)), "avg_depth": avg, "name": path.name}


def list_pseudo_pairs(sem_root: Path, depth_root: Path) -> list[SimPair]:
    """Pair real SEM images with model-generated pseudo depth maps (same rel path)."""
    sem_root, depth_root = Path(sem_root), Path(depth_root)
    pairs = []
    for p in sorted(sem_root.rglob("*.png")):
        if any(part.startswith(".") for part in p.parts):
            continue
        rel = p.relative_to(sem_root)
        depth = depth_root / rel
        if not depth.exists():
            raise ValueError(f"pseudo depth missing: {depth}")
        bucket = rel.parts[0] if len(rel.parts) > 1 else ""
        pairs.append(SimPair(f"pseudo/{rel.as_posix()}", "real", bucket, (p,), depth))
    return pairs


class ImageDirDataset(Dataset):
    """PNGs in a directory, sorted by name (inference input).

    recursive=True walks subdirectories (hidden dirs excluded) and reports names
    as POSIX relative paths so callers can mirror the tree on output.
    """

    def __init__(self, sem_dir: Path, recursive: bool = False):
        root = Path(sem_dir)
        if recursive:
            paths = [p for p in root.rglob("*.png")
                     if not any(part.startswith(".") for part in p.parts)]
            self.entries = sorted((p.relative_to(root).as_posix(), p) for p in paths)
        else:
            self.entries = sorted((p.name, p) for p in root.glob("*.png"))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> dict:
        name, p = self.entries[i]
        return {"image": _to_tensor(load_image01(p)), "name": name}
