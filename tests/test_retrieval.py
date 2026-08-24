import numpy as np
import torch

from semdepth.retrieval import (
    BUCKET_TO_CASE,
    CASE_BG,
    blend_depths,
    build_keys,
    hole_mean_depth,
    retrieve_batch,
    standardize,
)


def _rand_img(rng, h=16, w=12):
    return rng.random((h, w)).astype(np.float32)


def test_standardize_removes_brightness_contrast():
    rng = np.random.default_rng(0)
    img = _rand_img(rng)
    jittered = 1.3 * img + 0.2
    assert np.allclose(standardize(img), standardize(jittered), atol=1e-4)


def test_retrieve_recovers_planted_key_under_shift_and_jitter():
    rng = np.random.default_rng(1)
    keys_raw = [_rand_img(rng) for _ in range(20)]
    keys = torch.nn.functional.normalize(
        torch.from_numpy(np.stack([standardize(k).ravel() for k in keys_raw])), dim=1
    ).half()
    target = 7
    q = np.roll(keys_raw[target], shift=(1, -2), axis=(0, 1))  # shifted acquisition
    q = 0.8 * q + 0.1 + rng.normal(0, 0.01, q.shape).astype(np.float32)  # jitter+noise
    idx, sims = retrieve_batch(standardize(q)[None], keys, device="cpu", shift=2, topk=3)
    assert idx[0, 0] == target
    assert sims[0, 0] > 0.95
    # without shift search the same query must NOT match as well
    _, sims_ns = retrieve_batch(standardize(q)[None], keys, device="cpu", shift=0, topk=1)
    assert sims_ns[0, 0] < sims[0, 0]


def test_retrieve_flip_variant_recovers_flipped_query():
    rng = np.random.default_rng(2)
    keys_raw = [_rand_img(rng) for _ in range(10)]
    keys = torch.nn.functional.normalize(
        torch.from_numpy(np.stack([standardize(k).ravel() for k in keys_raw])), dim=1
    ).half()
    q = standardize(keys_raw[4][:, ::-1].copy())  # horizontally mirrored acquisition
    idx_no, sims_no = retrieve_batch(q[None], keys, device="cpu", shift=0, flips=False, topk=1)
    idx_fl, sims_fl = retrieve_batch(q[None], keys, device="cpu", shift=0, flips=True, topk=1)
    assert sims_fl[0, 0] > 0.99 and idx_fl[0, 0] == 4
    assert sims_fl[0, 0] > sims_no[0, 0]


def test_build_keys_and_blend(synth_root):
    from semdepth.data import list_sim_pairs

    pairs = list_sim_pairs(
        synth_root / "simulation_data" / "SEM", synth_root / "simulation_data" / "Depth"
    )
    keys, depths = build_keys(pairs, blur_sigma=0.0, device="cpu")
    assert keys.shape == (12, 72 * 48) and len(depths) == 12
    assert torch.allclose(keys.float().norm(dim=1), torch.ones(12), atol=1e-2)
    pred = blend_depths(depths, np.array([0, 1]), np.array([5.0, 5.0]))
    from PIL import Image

    a = np.asarray(Image.open(depths[0]).convert("L"), dtype=np.float32)
    b = np.asarray(Image.open(depths[1]).convert("L"), dtype=np.float32)
    assert np.abs(pred.astype(np.float32) - np.round((a + b) / 2)).max() <= 1.0


def test_hole_mean_depth_semantics():
    d = np.full((10, 10), CASE_BG["Case_2"], dtype=np.uint8)  # all background
    d[2:5, 2:5] = 30  # hole at the floor
    got = hole_mean_depth(d, "Case_2")
    assert abs(got - (150 - 30)) < 1e-6
    assert hole_mean_depth(np.full((4, 4), 150, dtype=np.uint8), "Case_2") == 0.0


def test_bucket_case_mapping_constants():
    assert BUCKET_TO_CASE == {"110": "Case_1", "120": "Case_2", "130": "Case_3", "140": "Case_4"}
    assert [CASE_BG[BUCKET_TO_CASE[b]] - 30 for b in ("110", "120", "130", "140")] == [110, 120, 130, 140]
