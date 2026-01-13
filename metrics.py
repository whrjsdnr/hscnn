import torch
import torch.nn.functional as F


def metric_mrae(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6, mask_zero: bool = False) -> torch.Tensor:
    """
    MRAE = mean( |pred-gt| / |gt| )
    - eps: numerical stability
    - mask_zero: if True, ignore pixels where |gt| <= eps
    """
    denom = gt.abs()
    if mask_zero:
        mask = denom > eps
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        return (torch.abs(pred - gt)[mask] / (denom[mask] + eps)).mean()
    return (torch.abs(pred - gt) / (denom + eps)).mean()


def metric_rmse(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(F.mse_loss(pred, gt, reduction="mean"))


def metric_sam(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    SAM (radians), mean over pixels.
    pred, gt: (B,C,H,W)
    """
    B, C, H, W = pred.shape
    p = pred.permute(0, 2, 3, 1).reshape(B, H * W, C)
    g = gt.permute(0, 2, 3, 1).reshape(B, H * W, C)

    dot = (p * g).sum(dim=-1)
    pn = torch.sqrt((p * p).sum(dim=-1) + eps)
    gn = torch.sqrt((g * g).sum(dim=-1) + eps)

    cos = dot / (pn * gn + eps)
    cos = torch.clamp(cos, -1.0, 1.0)
    ang = torch.acos(cos)
    return ang.mean()


def metric_spectral_gradient(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    L1 between adjacent band gradients.
    """
    dp = pred[:, 1:, :, :] - pred[:, :-1, :, :]
    dg = gt[:, 1:, :, :] - gt[:, :-1, :, :]
    return F.l1_loss(dp, dg, reduction="mean")
