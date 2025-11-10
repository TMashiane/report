import torch
import torch.nn as nn


class SEBlock3D(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, max(ch // r, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(ch // r, 4), ch),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, d, h, w = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


class SpatialAtt3D(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        padding = k // 2
        self.conv = nn.Conv3d(2, 1, kernel_size=k, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        y = self.sigmoid(y)
        return x * y


class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class ResAttUNet3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=32):
        super().__init__()
        self.enc1 = nn.Sequential(ResBlock3D(in_ch, base_ch), SEBlock3D(base_ch), SpatialAtt3D())
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = nn.Sequential(ResBlock3D(base_ch, base_ch * 2), SEBlock3D(base_ch * 2), SpatialAtt3D())
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = nn.Sequential(ResBlock3D(base_ch * 2, base_ch * 4), SEBlock3D(base_ch * 4), SpatialAtt3D())
        self.pool3 = nn.MaxPool3d(2)

        self.bottleneck = nn.Sequential(ResBlock3D(base_ch * 4, base_ch * 8), SEBlock3D(base_ch * 8), SpatialAtt3D())

        self.up1 = nn.ConvTranspose3d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec1 = nn.Sequential(ResBlock3D(base_ch * 8, base_ch * 4), SEBlock3D(base_ch * 4), SpatialAtt3D())
        self.up2 = nn.ConvTranspose3d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = nn.Sequential(ResBlock3D(base_ch * 4, base_ch * 2), SEBlock3D(base_ch * 2), SpatialAtt3D())
        self.up3 = nn.ConvTranspose3d(base_ch * 2, base_ch, 2, stride=2)
        self.dec3 = nn.Sequential(ResBlock3D(base_ch * 2, base_ch), SEBlock3D(base_ch), SpatialAtt3D())

        self.out = nn.Conv3d(base_ch, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d = self.up1(b)
        d = self.dec1(torch.cat([d, e3], dim=1))
        d = self.up2(d)
        d = self.dec2(torch.cat([d, e2], dim=1))
        d = self.up3(d)
        d = self.dec3(torch.cat([d, e1], dim=1))
        return self.out(d)

