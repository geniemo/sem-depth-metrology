import torch


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """RMSE between two tensors of identical shape/scale."""
    return torch.sqrt(torch.mean((pred.float() - target.float()) ** 2)).item()


def rmse_255(pred01: torch.Tensor, target01: torch.Tensor) -> float:
    """RMSE in 0-255 units for tensors normalized to [0,1] (leaderboard scale)."""
    return 255.0 * rmse(pred01, target01)
