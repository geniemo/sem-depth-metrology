import torch
from torch.utils.data import DataLoader

from semdepth.data import RealDataset


@torch.no_grad()
def real_proxy_rmse(
    model: torch.nn.Module, dataset: RealDataset, device: str, batch_size: int = 256
) -> float:
    """RMSE between predicted-map means (0-255) and given average depths.

    Submission-free proxy that quantifies the sim-to-real domain gap.
    """
    model = model.to(device).eval()
    dl = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    se_sum, n = 0.0, 0
    for batch in dl:
        x = batch["image"].to(device)
        pred_mean = model(x).mean(dim=(1, 2, 3)) * 255.0
        avg = batch["avg_depth"].to(device).float()
        se_sum += ((pred_mean - avg) ** 2).sum().item()
        n += len(avg)
    return (se_sum / n) ** 0.5
