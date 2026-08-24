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
    model: torch.nn.Module,
    sem_dir: Path,
    out_dir: Path,
    device: str,
    batch_size: int = 256,
    flip_tta: bool = False,
) -> int:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device).eval()
    ds = ImageDirDataset(Path(sem_dir))
    dl = DataLoader(ds, batch_size=batch_size, num_workers=4)
    n = 0
    for batch in tqdm(dl, desc="predict"):
        x = batch["image"].to(device)
        pred = model(x)
        if flip_tta:
            pred = (pred + torch.flip(model(torch.flip(x, [-1])), [-1])) / 2
        arr = (pred.clamp(0, 1) * 255).round().byte().cpu().numpy()
        for name, a in zip(batch["name"], arr):
            Image.fromarray(a[0].astype(np.uint8), mode="L").save(out_dir / name)
            n += 1
    return n


def make_submission_zip(pred_dir: Path, zip_path: Path) -> int:
    paths = sorted(Path(pred_dir).glob("*.png"))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for p in paths:
            zf.write(p, arcname=p.name)
    return len(paths)
