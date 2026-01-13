import torch.nn as nn


def init_kaiming(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def poly_lr(base_lr, epoch, max_epoch, power=1.5):
    t = epoch / max_epoch
    return base_lr * ((1.0 - t) ** power)


def exp_gamma_for_target(start_lr: float, end_lr: float, epochs: int) -> float:
    if epochs <= 1:
        return 1.0
    return (end_lr / start_lr) ** (1.0 / (epochs - 1))
