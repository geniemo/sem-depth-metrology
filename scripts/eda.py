"""EDA: quantify the sim-to-real domain gap and produce the report figures.

Outputs PNG figures to report/figures/ and prints the headline numbers used
in report/eda.md. All sampling is seeded and hidden files are excluded.
"""
import argparse
import random
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from semdepth.data import list_sim_pairs, load_image01  # noqa: E402

_SITE_RE = re.compile(r"^depth_(?P<bucket>\d+)_site_(?P<site>\d+)$")


def sample_paths(root: Path, n: int, seed: int = 0) -> list[Path]:
    paths = [p for p in sorted(Path(root).rglob("*.png"))
             if not any(part.startswith(".") for part in p.parts)]
    rng = random.Random(seed)
    return paths if len(paths) <= n else rng.sample(paths, n)


def load_stack(paths: list[Path]) -> np.ndarray:
    return np.stack([load_image01(p) * 255.0 for p in paths])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--n-sample", type=int, default=3000)
    args = ap.parse_args()
    root = Path(args.root)
    sim_sem_dir = root / "simulation_data" / "SEM"
    sim_depth_dir = root / "simulation_data" / "Depth"
    train_sem_dir = root / "train" / "SEM"
    train_csv = root / "train" / "average_depth.csv"
    test_sem_dir = root / "test" / "SEM"
    out = Path("report/figures")
    out.mkdir(parents=True, exist_ok=True)

    sim = load_stack(sample_paths(sim_sem_dir, args.n_sample))
    real = load_stack(sample_paths(train_sem_dir, args.n_sample))
    test = load_stack(sample_paths(test_sem_dir, args.n_sample))

    # 1. pixel intensity histograms: the appearance gap
    plt.figure(figsize=(7, 4))
    for arr, label in [(sim, "sim SEM"), (real, "real train SEM"), (test, "real test SEM")]:
        plt.hist(arr.ravel(), bins=64, range=(0, 255), density=True, alpha=0.5, label=label)
    plt.legend()
    plt.title("Pixel intensity distribution by domain")
    plt.savefig(out / "01_intensity_hist.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[stats] pixel mean/std — sim {sim.mean():.1f}/{sim.std():.1f}, "
          f"real train {real.mean():.1f}/{real.std():.1f}, "
          f"real test {test.mean():.1f}/{test.std():.1f}")

    # 2. per-image mean brightness by domain
    plt.figure(figsize=(7, 4))
    for arr, label in [(sim, "sim"), (real, "real train"), (test, "real test")]:
        plt.hist(arr.mean(axis=(1, 2)), bins=50, density=True, alpha=0.5, label=label)
    plt.legend()
    plt.title("Per-image mean brightness by domain")
    plt.savefig(out / "02_mean_brightness.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. label distributions: sim depth-map means per bucket vs real avg_depth per bucket
    pairs = list_sim_pairs(sim_sem_dir, sim_depth_dir)
    rng = random.Random(0)
    by_bucket_means: dict[str, list[float]] = defaultdict(list)
    for p in rng.sample(pairs, 4000):
        by_bucket_means[p.bucket].append(float(load_image01(p.depth_path).mean() * 255.0))
    df = pd.read_csv(train_csv)
    real_by_bucket: dict[str, list[float]] = defaultdict(list)
    for key, avg in zip(df.iloc[:, 0], df.iloc[:, 1]):
        real_by_bucket[_SITE_RE.match(str(key))["bucket"]].append(float(avg))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for b in sorted(by_bucket_means):
        ax1.hist(by_bucket_means[b], bins=40, alpha=0.6, label=f"sim bucket {b}")
    ax1.legend()
    ax1.set_title("Sim depth-map per-image mean, by folder bucket")
    for b in sorted(real_by_bucket):
        ax2.hist(real_by_bucket[b], bins=40, alpha=0.6, label=f"real Depth_{b}")
    ax2.legend()
    ax2.set_title("Real per-site average depth, by folder bucket")
    fig.savefig(out / "03_label_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    sim_bucket_stats = {b: (float(np.mean(v)), float(np.std(v)))
                        for b, v in sorted(by_bucket_means.items())}
    real_bucket_stats = {b: (float(np.mean(v)), float(np.std(v)))
                         for b, v in sorted(real_by_bucket.items())}
    print(f"[stats] sim depth-map mean by bucket: "
          f"{ {b: f'{m:.1f}±{s:.1f}' for b, (m, s) in sim_bucket_stats.items()} }")
    print(f"[stats] real avg_depth by bucket: "
          f"{ {b: f'{m:.1f}±{s:.1f}' for b, (m, s) in real_bucket_stats.items()} }")

    # 4. physics hook: site mean brightness vs site average depth
    site_imgs: dict[str, list[Path]] = {}
    for key in df.iloc[:, 0]:
        m = _SITE_RE.match(str(key))
        site_dir = train_sem_dir / f"Depth_{m['bucket']}" / f"site_{m['site']}"
        site_imgs[str(key)] = sorted(site_dir.glob("*.png"))
    keys = rng.sample(list(site_imgs), 400)
    xs = [float(np.mean([load_image01(p).mean() * 255.0 for p in site_imgs[k]])) for k in keys]
    ys = [float(df[df.iloc[:, 0] == k].iloc[0, 1]) for k in keys]
    r = float(np.corrcoef(xs, ys)[0, 1])
    plt.figure(figsize=(5, 5))
    plt.scatter(xs, ys, s=8, alpha=0.5)
    plt.xlabel("site mean SEM brightness")
    plt.ylabel("site average depth")
    plt.title(f"Brightness vs average depth per site (r={r:.3f})")
    plt.savefig(out / "04_brightness_vs_depth.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[stats] site brightness↔avg_depth correlation r={r:.3f} (n=400 sites)")

    # 5. itr0 vs itr1: noise level between realizations of the same structure
    diffs = []
    for p in rng.sample(pairs, 500):
        a, b = (load_image01(q) * 255.0 for q in p.sem_paths[:2])
        diffs.append(float(np.abs(a - b).mean()))
    plt.figure(figsize=(7, 4))
    plt.hist(diffs, bins=50)
    plt.title("Mean |itr0 - itr1| per structure (sim noise level)")
    plt.savefig(out / "05_itr_noise.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[stats] mean |itr0-itr1| = {np.mean(diffs):.2f} ± {np.std(diffs):.2f} (0-255 scale)")

    # 6. example grid: sim SEM / sim Depth / real train / real test
    fig, axes = plt.subplots(4, 6, figsize=(11, 9))
    show_pairs = rng.sample(pairs, 6)
    for j, p in enumerate(show_pairs):
        axes[0, j].imshow(load_image01(p.sem_paths[0]), cmap="gray", vmin=0, vmax=1)
        axes[1, j].imshow(load_image01(p.depth_path), cmap="viridis", vmin=0, vmax=1)
    for j, p in enumerate(sample_paths(train_sem_dir, 6, seed=2)):
        axes[2, j].imshow(load_image01(p), cmap="gray", vmin=0, vmax=1)
    for j, p in enumerate(sample_paths(test_sem_dir, 6, seed=3)):
        axes[3, j].imshow(load_image01(p), cmap="gray", vmin=0, vmax=1)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    for i, label in enumerate(["sim SEM", "sim Depth", "real train", "real test"]):
        axes[i, 0].set_ylabel(label, fontsize=10)
    fig.savefig(out / "06_examples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
