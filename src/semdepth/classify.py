"""4-way case classifier for real/test SEM images.

Real depth buckets (Depth_110..140) correspond 1:1 to sim cases (data probe:
depth-map background minus hole floor = 110/120/130/140), so the classifier is
trained directly on real train images with folder-derived labels and applied to
the test set within the same domain. Inputs stay raw [0,1] — absolute brightness
is the discriminative cue here, so no standardization.
"""
import random
import re
from pathlib import Path

import numpy as np
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from semdepth.data import load_image01

_BUCKETS = ("110", "120", "130", "140")
_SITE_DIR_RE = re.compile(r"^site_\d+$")


def list_real_labeled(sem_root: Path) -> list[tuple[Path, int, str]]:
    """(image path, bucket index 0..3, site id) for every real train image."""
    sem_root = Path(sem_root)
    out = []
    for bi, bucket in enumerate(_BUCKETS):
        bdir = sem_root / f"Depth_{bucket}"
        if not bdir.is_dir():
            continue
        for site_dir in sorted(d for d in bdir.iterdir()
                               if d.is_dir() and _SITE_DIR_RE.match(d.name)):
            for p in sorted(site_dir.glob("*.png")):
                out.append((p, bi, f"{bucket}/{site_dir.name}"))
    return out


def split_by_site(
    records: list[tuple[Path, int, str]], val_fraction: float, seed: int
) -> tuple[list, list]:
    sites = sorted({r[2] for r in records})
    random.Random(seed).shuffle(sites)
    n_val = max(1, round(len(sites) * val_fraction))
    val_sites = set(sites[:n_val])
    train = [r for r in records if r[2] not in val_sites]
    val = [r for r in records if r[2] in val_sites]
    return train, val


class BucketDataset(Dataset):
    def __init__(self, records: list[tuple[Path, int, str]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        path, label, _site = self.records[i]
        img = torch.from_numpy(load_image01(path)).unsqueeze(0)
        return {"image": img, "label": label}


class BucketClassifier(nn.Module):
    def __init__(self, encoder: str = "efficientnet_b0", pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model(encoder, pretrained=pretrained,
                                       num_classes=4, in_chans=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # logits


def train_classifier(cfg: dict) -> dict:
    from semdepth.train import seed_all

    seed_all(cfg["seed"])
    tr = cfg["train"]
    device = tr.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    records = list_real_labeled(Path(cfg["data"]["real_sem_root"]))
    train_recs, val_recs = split_by_site(records, cfg["data"]["val_fraction"], cfg["seed"])
    train_dl = DataLoader(BucketDataset(train_recs), batch_size=tr["batch_size"],
                          shuffle=True, num_workers=tr["num_workers"], drop_last=True)
    val_dl = DataLoader(BucketDataset(val_recs), batch_size=tr["batch_size"],
                        shuffle=False, num_workers=tr["num_workers"])
    if len(train_dl) == 0:
        raise ValueError("no training batches for classifier")

    model = BucketClassifier(cfg["model"]["encoder"], cfg["model"]["pretrained"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"])
    best_acc, ckpt = 0.0, Path(cfg["out"]["ckpt"])
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(tr["epochs"]):
        model.train()
        for batch in train_dl:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            loss = nn.functional.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        correct = n = 0
        with torch.no_grad():
            for batch in val_dl:
                pred = model(batch["image"].to(device)).argmax(dim=1).cpu()
                correct += int((pred == batch["label"]).sum())
                n += len(pred)
        acc = correct / n
        print(f"epoch {epoch}: val site-holdout acc = {acc:.5f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), ckpt)
    return {"best_val_acc": best_acc, "n_train": len(train_recs), "n_val": len(val_recs)}


@torch.no_grad()
def predict_buckets(
    model: nn.Module, image_paths: list[Path], device: str, batch_size: int = 512
) -> np.ndarray:
    """Bucket index (0..3) per image path."""
    model = model.to(device).eval()
    out = []
    for i in range(0, len(image_paths), batch_size):
        chunk = image_paths[i:i + batch_size]
        x = torch.stack([torch.from_numpy(load_image01(p)).unsqueeze(0) for p in chunk])
        out.append(model(x.to(device)).argmax(dim=1).cpu().numpy())
    return np.concatenate(out)
