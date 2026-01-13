import torch.nn as nn

from .metrics import metric_mrae, metric_sam, metric_spectral_gradient


class MRAELoss(nn.Module):
    def __init__(self, eps=1e-6, mask_zero=False):
        super().__init__()
        self.eps = eps
        self.mask_zero = mask_zero

    def forward(self, pred, gt):
        return metric_mrae(pred, gt, eps=self.eps, mask_zero=self.mask_zero)


class SAMLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, gt):
        return metric_sam(pred, gt, eps=self.eps)


class SpectralGradientLoss(nn.Module):
    def forward(self, pred, gt):
        return metric_spectral_gradient(pred, gt)
