import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """
    Light UNet for SIOR (CPU-safe)
    """
    def __init__(self, in_channels, num_classes, base_c=32):
        super().__init__()

        self.d1 = DoubleConv(in_channels, base_c)
        self.d2 = DoubleConv(base_c, base_c * 2)
        self.d3 = DoubleConv(base_c * 2, base_c * 4)

        self.pool = nn.MaxPool2d(2)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.u2 = DoubleConv(base_c * 4 + base_c * 2, base_c * 2)
        self.u1 = DoubleConv(base_c * 2 + base_c, base_c)

        self.out = nn.Conv2d(base_c, num_classes, 1)

    def forward(self, x):
        x1 = self.d1(x)              # [B, 32, H, W]
        x2 = self.d2(self.pool(x1))  # [B, 64, H/2, W/2]
        x3 = self.d3(self.pool(x2))  # [B, 128, H/4, W/4]

        y = self.up(x3)
        y = self.u2(torch.cat([y, x2], dim=1))

        y = self.up(y)
        y = self.u1(torch.cat([y, x1], dim=1))

        return self.out(y)
