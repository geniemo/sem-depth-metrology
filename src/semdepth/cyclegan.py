"""Sim->real domain translator: WGAN-GP CycleGAN (per case, warm-startable).

Reimplementation of the architecture published in the 1st-place solution of this
competition (lastdefiance20/2022-Samsung-AI-Challenge-3D-Metrology-1st-place-
Solution: WGAN-GP adversarial + L1 cycle(lambda=10) + L1 identity(lambda=0.5),
critic 5:1, tiny 2-resblock generator, GroupNorm critic, inputs (itr0+itr1)/212-1),
written from scratch for this codebase. Our addition: checkpoints are selected by
the retrieval gate (site-consistency / avg-depth check) instead of by eye.
"""
import itertools
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from semdepth.data import SimPair, load_image01

NORM = 212.0  # published normalization constant for this dataset


def to_gan_space(img01: np.ndarray) -> np.ndarray:
    """[0,1] image -> roughly [-1,1] using the dataset's 212 convention."""
    return (img01 * 255.0) / NORM * 2.0 - 1.0


def from_gan_space(x: torch.Tensor) -> torch.Tensor:
    """Generator output -> [0,1] image space (clamped)."""
    return (((x + 1.0) * NORM / 2.0) / 255.0).clamp(0.0, 1.0)


class _ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, padding_mode="replicate"),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1, padding_mode="replicate"),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class Generator(nn.Module):
    def __init__(self, dim: int = 64, n_res: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, dim, 7, 1, 3, padding_mode="replicate"),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, 2 * dim, 4, 2, 1),
            nn.BatchNorm2d(2 * dim),
            nn.ReLU(inplace=True),
            *[_ResBlock(2 * dim) for _ in range(n_res)],
            nn.ConvTranspose2d(2 * dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, 1, 7, 1, 3, padding_mode="replicate"),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, dim: int = 64, dropout: float = 0.3):
        super().__init__()

        def block(cin, cout, k, s, p):
            return [nn.Conv2d(cin, cout, k, s, p), nn.GroupNorm(4, cout),
                    nn.LeakyReLU(0.2, inplace=True), nn.Dropout(dropout)]

        self.net = nn.Sequential(
            nn.Conv2d(1, dim, 7, 1, 3, padding_mode="replicate"),
            nn.LeakyReLU(0.2, inplace=True),
            *block(dim, dim, 4, 2, 1),        # 36x24
            *block(dim, dim, 3, 1, 1),
            *block(dim, 2 * dim, 4, 2, 1),    # 18x12
            *block(2 * dim, 2 * dim, 3, 1, 1),
            *block(2 * dim, 4 * dim, 4, 2, 1),  # 9x6
            *block(4 * dim, 4 * dim, 3, 1, 1),
            nn.Conv2d(4 * dim, 1, (9, 6), 1, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimGanDataset(Dataset):
    """(itr0+itr1)/2 sim images in GAN space, one item per pair."""

    def __init__(self, pairs: list[SimPair]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> torch.Tensor:
        p = self.pairs[i]
        img = np.mean([load_image01(q) for q in p.sem_paths], axis=0, dtype=np.float32)
        return torch.from_numpy(to_gan_space(img)).unsqueeze(0)


class RealGanDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        return torch.from_numpy(to_gan_space(load_image01(self.paths[i]))).unsqueeze(0)


def gradient_penalty(critic: nn.Module, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    alpha = torch.rand(real.size(0), 1, 1, 1, device=real.device)
    inter = (alpha * real.detach() + (1 - alpha) * fake.detach()).requires_grad_(True)
    score = critic(inter)
    grads = torch.autograd.grad(score, inter, grad_outputs=torch.ones_like(score),
                                create_graph=True)[0].reshape(real.size(0), -1)
    return ((grads.norm(2, dim=1) - 1.0) ** 2).mean()


class CycleWGanGP:
    """Two-generator, two-critic WGAN-GP cycle trainer for one case."""

    def __init__(self, device: str, dim: int = 64, n_res: int = 2, dropout: float = 0.3,
                 lr: float = 2e-4, lambda_cycle: float = 10.0, lambda_idt: float = 0.5,
                 lambda_gp: float = 10.0, critic_iters: int = 5):
        self.device = device
        self.G_st = Generator(dim, n_res).to(device)  # sim -> real style
        self.G_ts = Generator(dim, n_res).to(device)  # real -> sim style
        self.D_t = Critic(dim, dropout).to(device)    # judges real-style images
        self.D_s = Critic(dim, dropout).to(device)    # judges sim-style images
        self.opt_g = torch.optim.Adam(
            itertools.chain(self.G_st.parameters(), self.G_ts.parameters()),
            lr=lr, betas=(0.5, 0.999))
        self.opt_dt = torch.optim.Adam(self.D_t.parameters(), lr=lr, betas=(0.5, 0.999))
        self.opt_ds = torch.optim.Adam(self.D_s.parameters(), lr=lr, betas=(0.5, 0.999))
        self.l_cycle, self.l_idt, self.l_gp = lambda_cycle, lambda_idt, lambda_gp
        self.critic_iters = critic_iters
        self.step = 0

    def load_generators(self, ckpt: dict) -> None:
        self.G_st.load_state_dict(ckpt["G_st"])
        self.G_ts.load_state_dict(ckpt["G_ts"])
        if "D_t" in ckpt:
            self.D_t.load_state_dict(ckpt["D_t"])
            self.D_s.load_state_dict(ckpt["D_s"])

    def state(self) -> dict:
        return {"G_st": self.G_st.state_dict(), "G_ts": self.G_ts.state_dict(),
                "D_t": self.D_t.state_dict(), "D_s": self.D_s.state_dict()}

    def _critic_step(self, sim: torch.Tensor, real: torch.Tensor) -> dict:
        with torch.no_grad():
            fake_t = self.G_st(sim)
            fake_s = self.G_ts(real)
        d_t_loss = (self.D_t(fake_t).mean() - self.D_t(real).mean()
                    + self.l_gp * gradient_penalty(self.D_t, real, fake_t))
        self.opt_dt.zero_grad(set_to_none=True)
        d_t_loss.backward()
        self.opt_dt.step()
        d_s_loss = (self.D_s(fake_s).mean() - self.D_s(sim).mean()
                    + self.l_gp * gradient_penalty(self.D_s, sim, fake_s))
        self.opt_ds.zero_grad(set_to_none=True)
        d_s_loss.backward()
        self.opt_ds.step()
        return {"d_t": float(d_t_loss.detach()), "d_s": float(d_s_loss.detach())}

    def _gen_step(self, sim: torch.Tensor, real: torch.Tensor) -> dict:
        fake_t = self.G_st(sim)
        fake_s = self.G_ts(real)
        idt = (nn.functional.l1_loss(fake_t, sim) + nn.functional.l1_loss(fake_s, real)) * self.l_idt
        cyc = (nn.functional.l1_loss(self.G_ts(fake_t), sim)
               + nn.functional.l1_loss(self.G_st(fake_s), real)) * self.l_cycle
        adv = -self.D_t(fake_t).mean() - self.D_s(fake_s).mean()
        loss = idt + cyc + adv
        self.opt_g.zero_grad(set_to_none=True)
        loss.backward()
        self.opt_g.step()
        return {"g": float(loss.detach()), "cyc": float(cyc.detach()), "idt": float(idt.detach())}

    def train_epoch(self, sim_dl: DataLoader, real_dl: DataLoader) -> dict:
        real_iter = iter(real_dl)
        last = {}
        for sim in sim_dl:
            try:
                real = next(real_iter)
            except StopIteration:
                real_iter = iter(real_dl)
                real = next(real_iter)
            n = min(sim.size(0), real.size(0))
            sim = sim[:n].to(self.device, non_blocking=True)
            real = real[:n].to(self.device, non_blocking=True)
            self.step += 1
            last.update(self._critic_step(sim, real))
            if self.step % self.critic_iters == 0:
                last.update(self._gen_step(sim, real))
        return last


@torch.no_grad()
def translate_pairs(
    generator: nn.Module, pairs: list[SimPair], out_root: Path,
    device: str, batch_size: int = 512, num_workers: int = 0,
) -> int:
    """G_st over itr-mean sim images; outputs mirror the DEPTH rel path (per pair)."""
    from PIL import Image

    generator = generator.to(device).eval()
    ds = SimGanDataset(pairs)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
    rels = [p.depth_path.relative_to(p.depth_path.parents[2]) for p in pairs]
    n = 0
    for bi, batch in enumerate(dl):
        out = from_gan_space(generator(batch.to(device)))
        arr = (out * 255.0).round().byte().cpu().numpy()
        for j in range(arr.shape[0]):
            rel = rels[bi * batch_size + j]
            dest = Path(out_root) / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(arr[j, 0], mode="L").save(dest)
            n += 1
    return n
