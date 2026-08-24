import torch

from semdepth.data import RealDataset
from semdepth.proxy import real_proxy_rmse


class _ConstModel(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x, self.value)


def test_proxy_exact_for_constant_model(synth_root):
    ds = RealDataset(synth_root / "train" / "SEM", synth_root / "train" / "average_depth.csv")
    model = _ConstModel(0.5)  # predicts mean depth 127.5 everywhere
    got = real_proxy_rmse(model, ds, device="cpu")
    avgs = torch.tensor([ds[i]["avg_depth"] for i in range(len(ds))])
    expected = torch.sqrt(torch.mean((127.5 - avgs) ** 2)).item()
    assert abs(got - expected) < 1e-3
