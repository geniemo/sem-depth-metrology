import pytest
import torch

from semdepth.data import (
    ImageDirDataset,
    RealDataset,
    SimDataset,
    list_sim_pairs,
    split_pairs,
)


def _pairs(synth_root):
    return list_sim_pairs(
        synth_root / "simulation_data" / "SEM", synth_root / "simulation_data" / "Depth"
    )


def test_pairing(synth_root):
    pairs = _pairs(synth_root)
    assert len(pairs) == 12  # 2 cases x 2 buckets x 3 structs
    assert all(len(p.sem_paths) == 2 for p in pairs)
    assert all(p.depth_path.stem == p.group_id for p in pairs)
    by_group: dict[str, set] = {}
    for p in pairs:
        by_group.setdefault(p.group_id, set()).add(p.case)
    assert len(by_group) == 6  # same structure appears once per case
    assert all(cases == {"Case_1", "Case_2"} for cases in by_group.values())


def test_pairing_rejects_orphan_sem(synth_root):
    orphan = synth_root / "simulation_data" / "SEM" / "Case_1" / "80" / "orphan_itr0.png"
    orphan.write_bytes((synth_root / "test" / "SEM" / "000000.png").read_bytes())
    with pytest.raises(ValueError):
        _pairs(synth_root)


def test_pairing_rejects_bad_sem_name(synth_root):
    bad = synth_root / "simulation_data" / "SEM" / "Case_1" / "80" / "noitr.png"
    bad.write_bytes((synth_root / "test" / "SEM" / "000000.png").read_bytes())
    with pytest.raises(ValueError, match="no _itr suffix"):
        _pairs(synth_root)


def test_pairing_rejects_misplaced_depth(synth_root):
    depth_dir = synth_root / "simulation_data" / "Depth" / "Case_1" / "80"
    victim = sorted(depth_dir.glob("*.png"))[0]
    victim.rename(depth_dir / "renamed.png")  # count unchanged, per-item lookup fails
    with pytest.raises(ValueError, match="depth map missing"):
        _pairs(synth_root)


def test_augment_flips_image_and_target_together(synth_root, monkeypatch):
    aug_ds = SimDataset(_pairs(synth_root), augment=True)
    base_ds = SimDataset(_pairs(synth_root), augment=False)
    monkeypatch.setattr("semdepth.data.random.random", lambda: 0.0)  # force both flips
    aug, orig = aug_ds[0], base_ds[0]
    assert torch.equal(aug["image"], torch.flip(orig["image"], [1, 2]))
    assert torch.equal(aug["target"], torch.flip(orig["target"], [1, 2]))


def test_group_split_no_leak(synth_root):
    pairs = _pairs(synth_root)
    tr, va = split_pairs(pairs, val_fraction=0.34, seed=7)
    assert len(tr) + len(va) == 12
    tr_groups = {p.group_id for p in tr}
    va_groups = {p.group_id for p in va}
    assert tr_groups.isdisjoint(va_groups)  # a structure never crosses the boundary
    assert len(va_groups) == 2 and len(va) == 4  # 2 groups x 2 cases
    tr2, va2 = split_pairs(pairs, val_fraction=0.34, seed=7)
    assert sorted(p.depth_path for p in va2) == sorted(p.depth_path for p in va)


def test_sim_dataset_items(synth_root):
    ds = SimDataset(_pairs(synth_root))
    assert len(ds) == 24
    item = ds[0]
    assert item["image"].shape == (1, 72, 48) and item["image"].dtype == torch.float32
    assert item["target"].shape == (1, 72, 48)
    assert 0.0 <= item["image"].min() and item["image"].max() <= 1.0


def test_real_dataset_site_labels(synth_root):
    ds = RealDataset(synth_root / "train" / "SEM", synth_root / "train" / "average_depth.csv")
    assert len(ds) == 9  # every image of every site
    items = [ds[i] for i in range(len(ds))]
    assert all(it["image"].shape == (1, 72, 48) for it in items)
    assert all(0.0 <= it["avg_depth"] <= 255.0 for it in items)
    assert len({round(it["avg_depth"], 6) for it in items}) == 3  # one label per site
    assert all(it["name"].endswith(".png") for it in items)


def test_image_dir_dataset_sorted(synth_root):
    ds = ImageDirDataset(synth_root / "test" / "SEM")
    names = [ds[i]["name"] for i in range(len(ds))]
    assert names == sorted(names) and len(names) == 4


def test_appearance_augment_image_only(synth_root):
    spec = {"brightness": [0.1, 0.1], "contrast": [1.2, 1.2],
            "blur_sigma": [0.0, 0.0], "noise_std": [0.0, 0.0]}
    ds_aug = SimDataset(_pairs(synth_root), appearance=spec)
    ds_raw = SimDataset(_pairs(synth_root))
    a, r = ds_aug[0], ds_raw[0]
    assert torch.equal(a["target"], r["target"])  # target untouched
    assert not torch.equal(a["image"], r["image"])  # image jittered
    expected = (1.2 * (r["image"] - 0.5) + 0.5 + 0.1).clamp(0, 1)
    assert torch.allclose(a["image"], expected, atol=1e-6)


def test_appearance_blur_and_noise_stay_in_range(synth_root):
    spec = {"brightness": [0.2, 0.2], "contrast": [1.3, 1.3],
            "blur_sigma": [1.0, 1.0], "noise_std": [0.04, 0.04]}
    ds = SimDataset(_pairs(synth_root), appearance=spec)
    for i in range(4):
        img = ds[i]["image"]
        assert 0.0 <= img.min() and img.max() <= 1.0
        assert img.dtype == torch.float32
