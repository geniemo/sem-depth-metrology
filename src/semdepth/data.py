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


class SimDataset(Dataset):
    """(SEM image, depth map) pairs; one item per SEM realization (itr)."""

    def __init__(self, pairs: list[SimPair], augment: bool = False):
        self.items = [(p.sem_paths[k], p.depth_path) for p in pairs for k in range(len(p.sem_paths))]
        self.augment = augment

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        sem_path, depth_path = self.items[i]
        image, target = load_image01(sem_path), load_image01(depth_path)
        if self.augment:
            if random.random() < 0.5:
                image, target = image[:, ::-1], target[:, ::-1]
            if random.random() < 0.5:
                image, target = image[::-1, :], target[::-1, :]
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


class ImageDirDataset(Dataset):
    """All PNGs in a directory, sorted by name (inference input)."""

    def __init__(self, sem_dir: Path):
        self.paths = sorted(Path(sem_dir).glob("*.png"))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> dict:
        p = self.paths[i]
        return {"image": _to_tensor(load_image01(p)), "name": p.name}
