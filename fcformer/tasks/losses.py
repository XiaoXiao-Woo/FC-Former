"""Eq. (11) of the paper, the loss of the supervised tasks (MHIF and pansharpening):

    L = L1(f(I), GT) + lambda * L_SSIM(f(I), GT),   lambda = 0.1,   L_SSIM = 1 - SSIM.

SSIM uses an 11 x 11 Gaussian window (sigma 1.5) per channel, with C1 = 0.01^2 and C2 = 0.03^2 for data
in [0, 1].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window(size, sigma, channels):
    x = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :]).expand(channels, 1, size, size).contiguous()


def ssim(x, y, window_size=11, sigma=1.5):
    """Mean SSIM of two [B, C, H, W] tensors in [0, 1]."""
    c = x.size(1)
    window = _gaussian_window(window_size, sigma, c).to(x)
    conv = lambda t: F.conv2d(t, window, padding=window_size // 2, groups=c)
    mu_x, mu_y = conv(x), conv(y)
    var_x = conv(x * x) - mu_x ** 2
    var_y = conv(y * y) - mu_y ** 2
    cov = conv(x * y) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))
    return ssim_map.mean()


class SupervisedLoss(nn.Module):
    """Eq. (11). Takes the output as a tensor or as {"sr": tensor}; returns the total under "loss" and both terms."""

    def __init__(self, ssim_weight=0.1):
        super().__init__()
        self.ssim_weight = ssim_weight

    def forward(self, outputs, gt):
        sr = outputs["sr"] if isinstance(outputs, dict) else outputs
        l1 = F.l1_loss(sr, gt)
        l_ssim = 1 - ssim(sr, gt)
        return {"loss": l1 + self.ssim_weight * l_ssim, "l1_loss": l1, "ssim_loss": l_ssim}
