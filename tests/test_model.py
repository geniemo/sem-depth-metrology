import torch

from semdepth.model import UnetTimm


def _model():
    return UnetTimm(encoder_name="resnet18", pretrained=False)


def test_output_shape_and_range():
    m = _model().eval()
    for h, w in [(48, 72), (64, 64), (40, 56)]:
        x = torch.rand(2, 1, h, w)
        with torch.no_grad():
            y = m(x)
        assert y.shape == (2, 1, h, w)
        assert 0.0 <= y.min() and y.max() <= 1.0


def test_overfits_one_batch():
    torch.manual_seed(0)
    m = _model().train()
    x = torch.rand(4, 1, 48, 72)
    t = torch.rand(4, 1, 48, 72) * 0.5 + 0.25
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    losses = []
    for _ in range(40):
        opt.zero_grad()
        loss = torch.nn.functional.l1_loss(m(x), t)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0], f"no learning: {losses[0]:.4f} -> {losses[-1]:.4f}"
