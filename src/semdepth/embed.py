"""Cross-domain structure-identity embedding (contrastive matcher).

The R1 submission showed pixel-cosine ranking is near-random across the domain
gap. This module learns a shared embedding where acquisition noise, itr, CASE
condition, and domain are invariances, using supervision the published solutions
ignored:
  sim identities  = one structure -> its 4 cases x 2 itr = 8 views
  real identities = one site      -> its ~29 repeated acquisitions
NT-Xent over mixed batches (both domains share the encoder and the batch), with
cyclic-shift augmentation so small stage offsets become an invariance too.
Ranking quality is monitored by the true oracle: hold-out-site consistency.
"""
import random
from pathlib import Path

import numpy as np
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from semdepth.data import SimPair, load_image01
from semdepth.retrieval import standardize


def _rand_shift(img: np.ndarray, rng: random.Random, max_shift: int) -> np.ndarray:
    if max_shift <= 0:
        return img
    dh = rng.randint(-max_shift, max_shift)
    dw = rng.randint(-max_shift, max_shift)
    return np.roll(img, shift=(dh, dw), axis=(0, 1))


class PairViewDataset(Dataset):
    """Yields two standardized views of one identity per item.

    identities: list of view-path lists (>=2 views each; an identity with a
    single view is duplicated so the pair becomes an augmentation pair).
    """

    def __init__(self, identities: list[list[Path]], max_shift: int = 2, seed: int = 0):
        self.identities = identities
        self.max_shift = max_shift
        self.seed = seed

    def __len__(self) -> int:
        return len(self.identities)

    def __getitem__(self, i: int) -> dict:
        rng = random.Random((self.seed, i, random.random()).__hash__())
        views = self.identities[i]
        if len(views) >= 2:
            a, b = rng.sample(views, 2)
        else:
            a = b = views[0]
        va = _rand_shift(standardize(load_image01(a)), rng, self.max_shift)
        vb = _rand_shift(standardize(load_image01(b)), rng, self.max_shift)
        return {"a": torch.from_numpy(va.copy()).unsqueeze(0),
                "b": torch.from_numpy(vb.copy()).unsqueeze(0)}


def sim_identities(pairs: list[SimPair]) -> list[list[Path]]:
    """Group ALL sem views (every case, every itr) of one structure."""
    by_group: dict[str, list[Path]] = {}
    for p in pairs:
        by_group.setdefault(p.group_id, []).extend(p.sem_paths)
    return [sorted(v) for _, v in sorted(by_group.items())]


def real_site_identities(records: list[tuple[Path, int, str]]) -> list[list[Path]]:
    """Group real train images by site id (records from classify.list_real_labeled)."""
    by_site: dict[str, list[Path]] = {}
    for path, _bi, site in records:
        by_site.setdefault(site, []).append(path)
    return [sorted(v) for _, v in sorted(by_site.items())]


class EmbedNet(nn.Module):
    def __init__(self, encoder: str = "resnet18", dim: int = 128, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(encoder, pretrained=pretrained,
                                          num_classes=0, in_chans=1)
        self.head = nn.Linear(self.backbone.num_features, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.head(self.backbone(x))
        return nn.functional.normalize(z, dim=1)


def nt_xent(za: torch.Tensor, zb: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Normalized-temperature cross entropy over a batch of (a, b) positive pairs."""
    n = za.size(0)
    z = torch.cat([za, zb], dim=0)  # (2n, d), already L2-normalized
    sim = z @ z.T / temperature
    sim.fill_diagonal_(float("-inf"))
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return nn.functional.cross_entropy(sim, targets)


@torch.no_grad()
def embed_paths(model: nn.Module, paths: list[Path], device: str,
                batch_size: int = 1024) -> np.ndarray:
    model = model.to(device).eval()
    out = []
    for i in range(0, len(paths), batch_size):
        x = torch.stack([torch.from_numpy(standardize(load_image01(p)).copy()).unsqueeze(0)
                         for p in paths[i:i + batch_size]])
        out.append(model(x.to(device)).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def site_consistency_oracle(
    model: nn.Module,
    key_paths: list[Path],
    holdout_sites: list[list[Path]],
    device: str,
) -> float:
    """Mean fraction of a hold-out site's images agreeing on the top-1 key."""
    from collections import Counter

    keys = torch.from_numpy(embed_paths(model, key_paths, device)).to(device)
    hits = []
    for site in holdout_sites:
        q = torch.from_numpy(embed_paths(model, site, device)).to(device)
        top1 = (q @ keys.T).argmax(dim=1).cpu().tolist()
        hits.append(Counter(top1).most_common(1)[0][1] / len(top1))
    return float(np.mean(hits))
