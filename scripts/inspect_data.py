"""Inventory the raw competition data and verify the assumptions the code relies on.

Checks (each printed as CHECK <name>: OK/FAIL):
  counts       — file counts per root vs the numbers recorded in docs/data-notes.md
                 (train contains exactly one stray Jupyter-checkpoint PNG shipped by
                 the organizers; labeled images are 60,664)
  pairing      — list_sim_pairs runs without error over the full simulation data
  group_ids    — per-case unique stems == per-case depth count (no cross-bucket collision)
  site_cover   — RealDataset(csv) covers every non-hidden train PNG exactly once
  image_format — sampled PNGs are 48x72 (WxH); sim is mode L, real is a mix of
                 L and RGB (RGB channels verified identical, so convert("L") is lossless)
"""
import argparse
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from semdepth.data import RealDataset, list_sim_pairs


def sample_stats(root: Path, n: int = 300, seed: int = 0) -> tuple[Counter, Counter, dict]:
    paths = sorted(root.rglob("*.png"))
    rng = random.Random(seed)
    pick = paths if len(paths) <= n else rng.sample(paths, n)
    sizes, modes = Counter(), Counter()
    vals = []
    for p in pick:
        with Image.open(p) as im:
            sizes[im.size] += 1
            modes[im.mode] += 1
            vals.append(np.asarray(im.convert("L"), dtype=np.float32))
    arr = np.stack(vals)
    stats = {"mean": float(arr.mean()), "std": float(arr.std()),
             "min": float(arr.min()), "max": float(arr.max())}
    return sizes, modes, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw")
    args = ap.parse_args()
    root = Path(args.root)
    sim_sem, sim_depth = root / "simulation_data" / "SEM", root / "simulation_data" / "Depth"
    train_sem, train_csv = root / "train" / "SEM", root / "train" / "average_depth.csv"
    test_sem = root / "test" / "SEM"

    print("== counts ==")
    counts = {}
    for name, d in [("sim_sem", sim_sem), ("sim_depth", sim_depth),
                    ("train_sem", train_sem), ("test_sem", test_sem)]:
        counts[name] = sum(1 for _ in d.rglob("*.png"))
        print(f"  {name}: {counts[name]}")
    hidden = [p for p in train_sem.rglob("*.png")
              if any(part.startswith(".") for part in p.parts)]
    counts["train_visible"] = counts["train_sem"] - len(hidden)
    print(f"  train hidden-dir strays: {[str(p.relative_to(train_sem)) for p in hidden]}")
    ok = (counts["sim_sem"], counts["sim_depth"], counts["train_visible"],
          counts["test_sem"]) == (173304, 86652, 60664, 25988) and len(hidden) == 1
    print(f"CHECK counts: {'OK' if ok else 'FAIL'}")

    print("== pairing ==")
    pairs = list_sim_pairs(sim_sem, sim_depth)
    n_itr = Counter(len(p.sem_paths) for p in pairs)
    print(f"  pairs: {len(pairs)}, sem-per-pair histogram: {dict(n_itr)}")
    print(f"CHECK pairing: {'OK' if len(pairs) == 86652 and n_itr == Counter({2: 86652}) else 'FAIL'}")

    print("== group_ids (cross-bucket collision) ==")
    per_case_stems: dict[str, set] = {}
    per_case_depths: Counter = Counter()
    for p in pairs:
        per_case_stems.setdefault(p.case, set()).add(p.group_id)
        per_case_depths[p.case] += 1
    collision_free = all(len(s) == per_case_depths[c] for c, s in per_case_stems.items())
    for c in sorted(per_case_stems):
        print(f"  {c}: {per_case_depths[c]} depths, {len(per_case_stems[c])} unique stems")
    groups = {p.group_id for p in pairs}
    cases_per_group = Counter(Counter(p.group_id for p in pairs).values())
    print(f"  unique groups: {len(groups)}, pairs-per-group histogram: {dict(cases_per_group)}")
    print(f"CHECK group_ids: {'OK' if collision_free else 'FAIL'}")

    print("== site coverage ==")
    ds = RealDataset(train_sem, train_csv)
    df = pd.read_csv(train_csv)
    print(f"  csv rows (sites): {len(df)}, RealDataset images: {len(ds)}")
    print(f"  avg_depth: min={df.iloc[:, 1].min():.3f} max={df.iloc[:, 1].max():.3f}")
    print(f"CHECK site_cover: {'OK' if len(ds) == counts['train_visible'] == 60664 else 'FAIL'}")

    print("== image format (sampled) ==")
    fmt_ok = True
    allowed = {"sim_sem": {"L"}, "sim_depth": {"L"},
               "train_sem": {"L", "RGB"}, "test_sem": {"L", "RGB"}}
    for name, d in [("sim_sem", sim_sem), ("sim_depth", sim_depth),
                    ("train_sem", train_sem), ("test_sem", test_sem)]:
        sizes, modes, stats = sample_stats(d)
        print(f"  {name}: sizes={dict(sizes)} modes={dict(modes)} "
              f"mean={stats['mean']:.1f} std={stats['std']:.1f} "
              f"min={stats['min']:.0f} max={stats['max']:.0f}")
        fmt_ok &= sizes == Counter({(48, 72): sum(sizes.values())})
        fmt_ok &= set(modes) <= allowed[name]
    print(f"CHECK image_format: {'OK' if fmt_ok else 'FAIL'}")

    print("== rgb channel identity (sampled) ==")
    rng = random.Random(1)
    rgb_ok = True
    for name, d in [("train_sem", train_sem), ("test_sem", test_sem)]:
        paths = sorted(d.rglob("*.png"))
        n_rgb, n_bad = 0, 0
        for p in rng.sample(paths, 100):
            with Image.open(p) as im:
                if im.mode == "RGB":
                    n_rgb += 1
                    a = np.asarray(im)
                    if not (np.array_equal(a[..., 0], a[..., 1])
                            and np.array_equal(a[..., 1], a[..., 2])):
                        n_bad += 1
        print(f"  {name}: sampled 100, RGB={n_rgb}, non-identical={n_bad}")
        rgb_ok &= n_bad == 0
    print(f"CHECK rgb_identity: {'OK' if rgb_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
