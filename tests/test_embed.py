import numpy as np
import torch

from semdepth.classify import list_real_labeled
from semdepth.data import list_sim_pairs
from semdepth.embed import (
    EmbedNet,
    PairViewDataset,
    nt_xent,
    real_site_identities,
    sim_identities,
    site_consistency_oracle,
)


def test_identity_grouping(synth_root):
    pairs = list_sim_pairs(
        synth_root / "simulation_data" / "SEM", synth_root / "simulation_data" / "Depth"
    )
    sims = sim_identities(pairs)
    assert len(sims) == 6 and all(len(v) == 4 for v in sims)  # 2 cases x 2 itr views
    sims_t = sim_identities(pairs, translated_root=synth_root / "simulation_data" / "Depth")
    assert all(len(v) == 6 for v in sims_t)  # + one 'translated' path per case
    reals = real_site_identities(list_real_labeled(synth_root / "train" / "SEM"))
    assert len(reals) == 3 and all(len(v) == 3 for v in reals)


def test_pair_view_dataset_shapes(synth_root):
    pairs = list_sim_pairs(
        synth_root / "simulation_data" / "SEM", synth_root / "simulation_data" / "Depth"
    )
    ds = PairViewDataset(sim_identities(pairs), max_shift=2)
    item = ds[0]
    assert item["a"].shape == (1, 72, 48) and item["b"].shape == (1, 72, 48)
    assert abs(float(item["a"].mean())) < 0.2  # standardized-ish


def test_nt_xent_prefers_matched_pairs():
    torch.manual_seed(0)
    z = torch.nn.functional.normalize(torch.randn(8, 16), dim=1)
    loss_matched = nt_xent(z, z.clone())  # perfect positives
    loss_shuffled = nt_xent(z, z[torch.randperm(8)])
    assert loss_matched < loss_shuffled


def test_embednet_and_oracle_run(synth_root):
    model = EmbedNet(pretrained=False, dim=32).eval()
    pairs = list_sim_pairs(
        synth_root / "simulation_data" / "SEM", synth_root / "simulation_data" / "Depth"
    )
    keys = [p.sem_paths[0] for p in pairs]
    sites = real_site_identities(list_real_labeled(synth_root / "train" / "SEM"))
    o = site_consistency_oracle(model, keys, sites, device="cpu")
    assert 0.0 <= o["consistency"] <= 1.0
    assert 0.0 < o["diversity"] <= 1.0
    assert 0.0 <= o["score"] <= 1.0
