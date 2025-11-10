import math
from typing import Tuple, Optional
import torch
import torch.nn.functional as F


def mse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x, y, reduction='mean')


def rmse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(mse(x, y) + 1e-12)


def psnr(x: torch.Tensor, y: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    m = mse(x, y)
    return 20 * torch.log10(torch.tensor(max_val, device=x.device)) - 10 * torch.log10(m + 1e-12)


def _gaussian_kernel_3d(win_size: int, sigma: float, device) -> torch.Tensor:
    coords = torch.arange(win_size, device=device) - (win_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    g3 = g[:, None, None] * g[None, :, None] * g[None, None, :]
    g3 = g3 / g3.sum()
    return g3


def ssim3d(x: torch.Tensor, y: torch.Tensor, win_size: int = 7, sigma: float = 1.5, K: Tuple[float, float] = (0.01, 0.03),
           max_val: float = 1.0) -> torch.Tensor:
    # x,y: [N,1,D,H,W] in [0,1]
    device = x.device
    window = _gaussian_kernel_3d(win_size, sigma, device)
    window = window.view(1, 1, win_size, win_size, win_size)

    def filt(z):
        return F.conv3d(z, window, padding=win_size // 2, groups=1)

    mu_x = filt(x)
    mu_y = filt(y)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = filt(x * x) - mu_x2
    sigma_y2 = filt(y * y) - mu_y2
    sigma_xy = filt(x * y) - mu_xy

    C1 = (K[0] * max_val) ** 2
    C2 = (K[1] * max_val) ** 2

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = num / (den + 1e-12)
    return ssim_map.mean()


def ms_ssim3d(x: torch.Tensor, y: torch.Tensor, levels: int = 4, win_size: int = 7, sigma: float = 1.5,
              K: Tuple[float, float] = (0.01, 0.03), max_val: float = 1.0,
              weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    # Multi-scale SSIM in 3D; downsample by average pooling
    if weights is None:
        weights = torch.tensor([0.1, 0.3, 0.3, 0.3], device=x.device)  # sum to 1
    weights = weights / weights.sum()

    mssim_list = []
    mcs_list = []
    C1 = (K[0] * max_val) ** 2
    C2 = (K[1] * max_val) ** 2

    device = x.device
    window = _gaussian_kernel_3d(win_size, sigma, device)
    window = window.view(1, 1, win_size, win_size, win_size)

    def filt(z):
        return F.conv3d(z, window, padding=win_size // 2, groups=1)

    x_scale = x
    y_scale = y
    for l in range(levels):
        mu_x = filt(x_scale)
        mu_y = filt(y_scale)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = filt(x_scale * x_scale) - mu_x2
        sigma_y2 = filt(y_scale * y_scale) - mu_y2
        sigma_xy = filt(x_scale * y_scale) - mu_xy

        num_ssim = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        den_ssim = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
        ssim_map = num_ssim / (den_ssim + 1e-12)
        cs_map = (2 * sigma_xy + C2) / (sigma_x2 + sigma_y2 + C2)

        mssim_list.append(ssim_map.mean())
        mcs_list.append(cs_map.mean())

        if l < levels - 1:
            # Downsample
            x_scale = F.avg_pool3d(x_scale, kernel_size=2, stride=2, ceil_mode=True)
            y_scale = F.avg_pool3d(y_scale, kernel_size=2, stride=2, ceil_mode=True)

    mssim_list = torch.stack(mssim_list)
    mcs_list = torch.stack(mcs_list)

    # MS-SSIM = prod(cs^{w}) * ssim^{w_last}
    pow1 = weights[:-1]
    pow2 = weights[-1]
    ms_ssim = torch.prod(mcs_list[:-1] ** pow1) * (mssim_list[-1] ** pow2)
    return ms_ssim

