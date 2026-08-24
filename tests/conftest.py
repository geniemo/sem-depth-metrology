from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

HW = (72, 48)  # (rows, cols) — matches the real data (PIL reports size 48x72)


def _save_gray(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), mode="L").save(path)


def _depth_pattern(rng: np.random.Generator) -> np.ndarray:
    """Smooth radial 'hole' pattern with random depth scale — learnable signal."""
    h, w = HW
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2 + rng.uniform(-5, 5), w / 2 + rng.uniform(-8, 8)
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    depth = 255 - rng.uniform(100, 220) * np.exp(-(r ** 2) / rng.uniform(80, 300))
    return np.clip(depth, 0, 255)


def make_synth_data(root: Path) -> Path:
    rng = np.random.default_rng(0)
    # simulation: same structure names (and identical depth maps) repeated across cases
    for bucket in ["80", "81"]:
        for j in range(3):
            name = f"struct_{bucket}_{j}"
            depth = _depth_pattern(rng)
            for case in ["Case_1", "Case_2"]:
                _save_gray(
                    root / "simulation_data" / "Depth" / case / bucket / f"{name}.png", depth
                )
                for k in range(2):  # two noise realizations of the same structure
                    sem = np.clip(depth + rng.normal(0, 12, HW), 0, 255)
                    _save_gray(
                        root / "simulation_data" / "SEM" / case / bucket / f"{name}_itr{k}.png",
                        sem,
                    )
    # real train: 3 sites x 3 images, one average-depth label per SITE
    rows, n = [], 0
    for bucket, site in [("110", "site_00000"), ("110", "site_00001"), ("120", "site_00002")]:
        depth = _depth_pattern(rng)
        for _ in range(3):
            sem = np.clip(depth + rng.normal(0, 20, HW), 0, 255)
            _save_gray(
                root / "train" / "SEM" / f"Depth_{bucket}" / site / f"SEM_{n:06d}.png", sem
            )
            n += 1
        rows.append({"0": f"depth_{bucket}_{site}", "1": float(depth.mean())})
    pd.DataFrame(rows).to_csv(root / "train" / "average_depth.csv", index=False)
    for i in range(4):
        sem = np.clip(_depth_pattern(rng) + rng.normal(0, 20, HW), 0, 255)
        _save_gray(root / "test" / "SEM" / f"{i:06d}.png", sem)
    return root


@pytest.fixture()
def synth_root(tmp_path: Path) -> Path:
    return make_synth_data(tmp_path / "data")
