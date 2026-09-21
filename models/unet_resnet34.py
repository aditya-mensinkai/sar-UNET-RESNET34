"""UNet + ResNet34 binary segmentation (genuine UNet decoder, NOT LinkNet).

Encoder: torchvision ResNet34 stages (stem, layer1..layer4) with spatial
hierarchy preserved for skips. Decoder: upsample -> concat skip -> DoubleConv
at 4 stages, then 1x1 conv to 1 channel of RAW LOGITS (no sigmoid, no threshold).

2-channel SAR handling (documented, deterministic):
  conv1 is rebuilt as nn.Conv2d(2, 64, 7x7, stride 2). With pretrained weights,
  the new kernel = mean over the 3 pretrained input channels, repeated to 2
  channels (i.e. each SAR band starts from the average RGB filter). Random init
  (Kaiming) when pretrained=False.
Weight policy: pretrained=True requires a LOCAL weights file (weights_path);
  automatic download is disabled unless allow_download=True is passed
  explicitly. Missing local weights -> clear error, never silent fallback.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torchvision


@dataclass(frozen=True)
class UNetResNet34Config:
    in_channels: int = 2
    out_channels: int = 1
    encoder_name: str = "resnet34"
    pretrained: bool = False
    weights_path: Optional[str] = None
    allow_download: bool = False
    decoder_channels: List[int] = field(default_factory=lambda: [256, 128, 64, 32])


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    """Upsample -> concat encoder skip -> DoubleConv (UNet, not LinkNet)."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            raise RuntimeError(f"skip/feature spatial mismatch {tuple(x.shape)} vs {tuple(skip.shape)}")
        return self.conv(torch.cat([x, skip], dim=1))


class UNetResNet34(nn.Module):
    """ResNet34 encoder + UNet concat decoder. Forward returns raw logits."""

    # (stage_attr, channels, spatial scale relative to 512 input)
    SKIPS = [("s256", 64), ("l1", 64), ("l2", 128), ("l3", 256)]

    def __init__(self, cfg=None, **kw):
        super().__init__()
        self.cfg = cfg or UNetResNet34Config(**kw)
        if self.cfg.encoder_name != "resnet34":
            raise ValueError(f"only resnet34 encoder supported, got {self.cfg.encoder_name}")
        base = torchvision.models.resnet34(weights=None)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.l1, self.l2, self.l3, self.l4 = base.layer1, base.layer2, base.layer3, base.layer4
        self._adapt_first_conv(base.conv1)
        dch = list(self.cfg.decoder_channels)
        if len(dch) != 4:
            raise ValueError("decoder_channels must hold 4 stage widths")
        self.dec4 = DecoderBlock(512, 256, dch[0])   # 16->32 + l3
        self.dec3 = DecoderBlock(dch[0], 128, dch[1])  # 32->64 + l2
        self.dec2 = DecoderBlock(dch[1], 64, dch[2])   # 64->128 + l1
        self.dec1 = DecoderBlock(dch[2], 64, dch[3])   # 128->256 + stem
        self.final_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.final_conv = nn.Conv2d(dch[3], self.cfg.out_channels, kernel_size=1)
        # NOTE: no Sigmoid, no threshold - raw logits out.

    def _adapt_first_conv(self, orig_conv1):
        c = self.cfg.in_channels
        new = nn.Conv2d(c, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if self.cfg.pretrained:
            w = self._pretrained_conv1_weights()
            with torch.no_grad():
                new.weight.copy_(w.mean(dim=1, keepdim=True).repeat(1, c, 1, 1))
        else:
            nn.init.kaiming_normal_(new.weight, mode="fan_out", nonlinearity="relu")
        self.stem[0] = new
        self.adaptation = (f"conv1 rebuilt 3->64 as {c}->64; " +
                           ("init=mean-over-RGB pretrained kernel, repeated to 2ch"
                            if self.cfg.pretrained else "init=Kaiming (random, pretrained=False)"))

    def _pretrained_conv1_weights(self):
        if self.cfg.weights_path:
            import os
            if not os.path.isfile(self.cfg.weights_path):
                raise FileNotFoundError(
                    f"pretrained weights not found at {self.cfg.weights_path}; "
                    f"automatic download is disabled.")
            return torch.load(self.cfg.weights_path, map_location="cpu")["conv1.weight"]
        if self.cfg.allow_download:
            from torchvision.models import ResNet34_Weights
            return ResNet34_Weights.IMAGENET1K_V1.get_state_dict(progress=True)["conv1.weight"]
        raise FileNotFoundError(
            "Pretrained ResNet34 weights requested but no local weights were found. "
            "Automatic download is disabled. Provide a local weights path or set "
            "pretrained=False.")

    def forward(self, x):
        if x.shape[1] != self.cfg.in_channels:
            raise ValueError(f"expected {self.cfg.in_channels} input channels, got {x.shape[1]}")
        # stem stages with 256px tap BEFORE maxpool
        e0 = self.stem[0](x)
        e0 = self.stem[1](e0)
        e0 = self.stem[2](e0)      # 256px, 64ch -> skip
        e0p = self.stem[3](e0)     # 128px
        l1 = self.l1(e0p)          # 128px, 64ch -> skip
        l2 = self.l2(l1)           # 64px, 128ch -> skip
        l3 = self.l3(l2)           # 32px, 256ch -> skip
        b = self.l4(l3)            # 16px, 512ch bottleneck
        d = self.dec4(b, l3)
        d = self.dec3(d, l2)
        d = self.dec2(d, l1)
        d = self.dec1(d, e0)
        return self.final_conv(self.final_up(d))  # 512px, 1ch logits
