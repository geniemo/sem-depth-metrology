import timm
import torch
import torch.nn.functional as F
from torch import nn


class _DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UnetTimm(nn.Module):
    """U-Net with a timm backbone encoder; grayscale in, [0,1] depth map out."""

    def __init__(self, encoder_name: str = "resnet18", pretrained: bool = True):
        super().__init__()
        self.encoder = timm.create_model(
            encoder_name, features_only=True, pretrained=pretrained, in_chans=1
        )
        enc_chs = self.encoder.feature_info.channels()  # shallow -> deep
        dec_chs = [256, 128, 64, 32, 16][-len(enc_chs) :]
        blocks, in_ch = [], enc_chs[-1]
        skips = enc_chs[:-1][::-1] + [0]  # deepest skip first, last block has none
        for skip_ch, out_ch in zip(skips, dec_chs):
            blocks.append(_DecoderBlock(in_ch, skip_ch, out_ch))
            in_ch = out_ch
        self.decoder = nn.ModuleList(blocks)
        self.head = nn.Conv2d(in_ch, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        ph, pw = (-h) % 32, (-w) % 32
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        feats = self.encoder(x)
        out, skips = feats[-1], feats[:-1][::-1] + [None]
        for block, skip in zip(self.decoder, skips):
            out = block(out, skip)
        if out.shape[-2:] != x.shape[-2:]:  # encoders whose first feature is stride>2
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        out = torch.sigmoid(self.head(out))
        return out[..., :h, :w]
