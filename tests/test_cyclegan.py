import numpy as np
import torch

from semdepth.cyclegan import (
    Critic,
    CycleWGanGP,
    Generator,
    RealGanDataset,
    SimGanDataset,
    from_gan_space,
    gradient_penalty,
    to_gan_space,
    translate_pairs,
)
from semdepth.data import list_sim_pairs


def test_gan_space_roundtrip():
    img = np.random.default_rng(0).random((72, 48)).astype(np.float32) * 0.8
    x = torch.from_numpy(to_gan_space(img)).unsqueeze(0).unsqueeze(0)
    back = from_gan_space(x).squeeze().numpy()
    assert np.allclose(back, img, atol=1e-5)


def test_generator_critic_shapes():
    g, d = Generator(dim=8, n_res=1).eval(), Critic(dim=8).eval()
    x = torch.rand(2, 1, 72, 48) * 2 - 1
    with torch.no_grad():
        y = g(x)
        s = d(x)
    assert y.shape == (2, 1, 72, 48) and y.min() >= -1.0 and y.max() <= 1.0
    assert s.shape == (2, 1, 1, 1)


def test_gradient_penalty_finite():
    d = Critic(dim=8)
    real, fake = torch.rand(3, 1, 72, 48), torch.rand(3, 1, 72, 48)
    gp = gradient_penalty(d, real, fake)
    assert torch.isfinite(gp) and gp.item() >= 0.0


def test_train_epoch_updates_weights(synth_root):
    torch.manual_seed(0)
    pairs = list_sim_pairs(
        synth_root / "simulation_data" / "SEM", synth_root / "simulation_data" / "Depth"
    )
    real_paths = sorted((synth_root / "train" / "SEM").rglob("*.png"))
    trainer = CycleWGanGP(device="cpu", dim=8, n_res=1, critic_iters=2)
    before = trainer.G_st.net[0].weight.detach().clone()
    sim_dl = torch.utils.data.DataLoader(SimGanDataset(pairs), batch_size=4, shuffle=True)
    real_dl = torch.utils.data.DataLoader(RealGanDataset(real_paths), batch_size=4, shuffle=True)
    log = trainer.train_epoch(sim_dl, real_dl)
    assert "g" in log and "d_t" in log
    assert not torch.equal(before, trainer.G_st.net[0].weight.detach())
    # warm-start roundtrip
    trainer2 = CycleWGanGP(device="cpu", dim=8, n_res=1)
    trainer2.load_generators(trainer.state())
    assert torch.equal(trainer2.G_st.net[0].weight, trainer.G_st.net[0].weight)


def test_translate_pairs_mirrors_depth_tree(synth_root, tmp_path):
    pairs = list_sim_pairs(
        synth_root / "simulation_data" / "SEM", synth_root / "simulation_data" / "Depth"
    )
    g = Generator(dim=8, n_res=1).eval()
    n = translate_pairs(g, pairs, tmp_path / "translated", device="cpu", batch_size=5)
    assert n == 12
    outs = sorted(p.relative_to(tmp_path / "translated").as_posix()
                  for p in (tmp_path / "translated").rglob("*.png"))
    ins = sorted(p.depth_path.relative_to(synth_root / "simulation_data" / "Depth").as_posix()
                 for p in pairs)
    assert outs == ins
