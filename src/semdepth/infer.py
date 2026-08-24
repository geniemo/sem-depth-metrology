import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from semdepth.data import ImageDirDataset


@torch.no_grad()
def predict_dir(
    model: torch.nn.Module | list[torch.nn.Module],
    sem_dir: Path,
    out_dir: Path,
    device: str,
    batch_size: int = 256,
    flip_tta: bool = False,
    num_workers: int = 0,
    recursive: bool = False,
) -> int:
    """Write uint8 depth PNGs for every input PNG; a model list is ensembled by mean."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = model if isinstance(model, (list, tuple)) else [model]
    models = [m.to(device).eval() for m in models]
    ds = ImageDirDataset(Path(sem_dir), recursive=recursive)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
    n = 0
    for batch in tqdm(dl, desc="predict"):
        x = batch["image"].to(device)
        preds = []
        for m in models:
            p = m(x)
            if flip_tta:
                p = (p + torch.flip(m(torch.flip(x, [-1])), [-1])) / 2
            preds.append(p)
        pred = torch.stack(preds).mean(dim=0)
        arr = (pred.clamp(0, 1) * 255).round().byte().cpu().numpy()
        for name, a in zip(batch["name"], arr):
            dest = out_dir / name  # name is a relative path in recursive mode
            dest.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(a[0].astype(np.uint8), mode="L").save(dest)
            n += 1
    return n


def make_submission_zip(pred_dir: Path, zip_path: Path) -> int:
    paths = sorted(Path(pred_dir).glob("*.png"))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for p in paths:
            zf.write(p, arcname=p.name)
    return len(paths)
