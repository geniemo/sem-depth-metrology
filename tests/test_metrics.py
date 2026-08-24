import torch

from semdepth.metrics import rmse, rmse_255


def test_rmse_known_value():
    pred = torch.tensor([0.0, 0.0])
    target = torch.tensor([3.0, 4.0])
    assert abs(rmse(pred, target) - (12.5 ** 0.5)) < 1e-6


def test_rmse_zero_for_identical():
    x = torch.rand(2, 1, 8, 8)
    assert rmse(x, x) == 0.0


def test_rmse_255_scales_unit_interval():
    pred = torch.zeros(1, 1, 4, 4)
    target = torch.full((1, 1, 4, 4), 0.5)
    assert abs(rmse_255(pred, target) - 127.5) < 1e-4
