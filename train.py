import os
import argparse
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hscnn.dataset import RGBMatFolderDataset
from hscnn.models import build_model, rgb_to_spectral_interp
from hscnn.metrics import metric_mrae, metric_rmse, metric_sam, metric_spectral_gradient
from hscnn.losses import MRAELoss, SAMLoss, SpectralGradientLoss
from hscnn.utils import init_kaiming, poly_lr, exp_gamma_for_target


def forward_model(mode: str, model: nn.Module, rgb: torch.Tensor, out_bands: int):
    if mode == "hscnn":
        upspec = rgb_to_spectral_interp(rgb, out_bands=out_bands)
        return model(upspec)
    return model(rgb)


def train_only(model, mode, train_loader, device, optimizer, loss_fn, out_bands, epochs,
               use_poly_lr: bool, base_lr: float, poly_power: float,
               scheduler=None):
    model.train()
    for epoch in range(1, epochs + 1):
        if use_poly_lr:
            lr = poly_lr(base_lr, epoch - 1, epochs, power=poly_power)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
        else:
            lr = optimizer.param_groups[0]["lr"]

        total = 0.0
        n = 0
        for rgb, hsi, _ in train_loader:
            rgb = rgb.to(device)
            hsi = hsi.to(device)

            pred = forward_model(mode, model, rgb, out_bands)
            loss = loss_fn(pred, hsi)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total += float(loss.item())
            n += 1

        if scheduler is not None:
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]

        print(f"  [Epoch {epoch:04d}/{epochs}] lr={lr:.3e} train_loss={total/max(n,1):.6f}")


@torch.no_grad()
def test_once(model, mode, valid_loader, device, out_bands, mrae_eps: float, mrae_mask_zero: bool):
    model.eval()
    agg = {"mrae": 0.0, "rmse": 0.0, "sam": 0.0, "sgrad": 0.0}
    n = 0
    for rgb, hsi, _ in valid_loader:
        rgb = rgb.to(device)
        hsi = hsi.to(device)
        pred = forward_model(mode, model, rgb, out_bands)

        agg["mrae"] += float(metric_mrae(pred, hsi, eps=mrae_eps, mask_zero=mrae_mask_zero).item())
        agg["rmse"] += float(metric_rmse(pred, hsi).item())
        agg["sam"]  += float(metric_sam(pred, hsi).item())
        agg["sgrad"] += float(metric_spectral_gradient(pred, hsi).item())
        n += 1

    for k in agg:
        agg[k] /= max(n, 1)
    return agg


def main():
    p = argparse.ArgumentParser()

    # paths
    p.add_argument("--train_rgb", type=str, required=True)
    p.add_argument("--train_hsi", type=str, required=True)
    p.add_argument("--valid_rgb", type=str, required=True)
    p.add_argument("--valid_hsi", type=str, required=True)

    # mat
    p.add_argument("--mat_key", type=str, default=None)
    p.add_argument("--mat_norm", type=str, default="auto_divmax", choices=["auto_divmax", "fixed_div", "none"])
    p.add_argument("--mat_fixed_div", type=float, default=65535.0)

    # training common
    p.add_argument("--out_bands", type=int, default=31)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--crop", type=int, default=50)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)

    # arch
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--r_blocks", type=int, default=16)
    p.add_argument("--d_blocks", type=int, default=38)

    # loss
    p.add_argument("--train_loss", type=str, default="mrae",
                   choices=["mrae", "mse", "sam", "sgrad", "mrae+sgrad", "mrae+sam"])
    p.add_argument("--mrae_eps", type=float, default=1e-6)
    p.add_argument("--mrae_mask_zero", action="store_true")

    # R
    p.add_argument("--epochs_r", type=int, default=1000)
    p.add_argument("--lr_r", type=float, default=2e-4)
    p.add_argument("--poly_power", type=float, default=1.5)

    # D
    p.add_argument("--epochs_d", type=int, default=300)
    p.add_argument("--lr_d", type=float, default=1e-3)
    p.add_argument("--lr_d_end", type=float, default=1e-4)
    p.add_argument("--wd_d", type=float, default=1e-4)

    # other
    p.add_argument("--epochs_other", type=int, default=300)
    p.add_argument("--lr_other", type=float, default=2e-4)

    p.add_argument("--save_dir", type=str, default="checkpoints_hscnn")
    p.add_argument("--bad_log", type=str, default="bad_mat.log")
    p.add_argument("--max_skip_tries", type=int, default=50)

    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    train_ds = RGBMatFolderDataset(
        args.train_rgb, args.train_hsi,
        train=True, crop_size=args.crop, seed=args.seed,
        mat_key=args.mat_key, out_bands=args.out_bands,
        bad_log_path=args.bad_log, max_skip_tries=args.max_skip_tries,
        mat_norm=args.mat_norm, mat_fixed_div=args.mat_fixed_div
    )
    valid_ds = RGBMatFolderDataset(
        args.valid_rgb, args.valid_hsi,
        train=False, crop_size=None, seed=args.seed,
        mat_key=args.mat_key, out_bands=args.out_bands,
        bad_log_path=args.bad_log, max_skip_tries=args.max_skip_tries,
        mat_norm=args.mat_norm, mat_fixed_div=args.mat_fixed_div
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    base_losses = {
        "mrae": MRAELoss(eps=args.mrae_eps, mask_zero=args.mrae_mask_zero),
        "mse": nn.MSELoss(),
        "sam": SAMLoss(),
        "sgrad": SpectralGradientLoss(),
    }

    def make_train_loss(name: str):
        if name in base_losses:
            return base_losses[name]
        if name == "mrae+sgrad":
            return lambda pred, gt: base_losses["mrae"](pred, gt) + 0.1 * base_losses["sgrad"](pred, gt)
        if name == "mrae+sam":
            return lambda pred, gt: base_losses["mrae"](pred, gt) + 0.1 * base_losses["sam"](pred, gt)
        raise ValueError(name)

    train_loss_fn = make_train_loss(args.train_loss)

    modes = ["hscnn", "hscnn_u", "hscnn_r", "hscnn_d"]
    results: Dict[str, Dict[str, float]] = {}

    for mode in modes:
        print(f"\n====================\nMODEL = {mode}\n====================")
        model = build_model(mode, args.out_bands, args.depth, args.width, args.r_blocks, args.d_blocks).to(device)
        model.apply(init_kaiming)

        scheduler = None
        use_poly = False

        if mode == "hscnn_r":
            epochs = args.epochs_r
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_r, betas=(0.9, 0.999), eps=1e-8)
            use_poly = True
            base_lr = args.lr_r
        elif mode == "hscnn_d":
            epochs = args.epochs_d
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_d, betas=(0.9, 0.999), eps=1e-8,
                                          weight_decay=args.wd_d)
            gamma = exp_gamma_for_target(args.lr_d, args.lr_d_end, epochs)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
            base_lr = args.lr_d
        else:
            epochs = args.epochs_other
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_other, betas=(0.9, 0.999), eps=1e-8)
            base_lr = args.lr_other

        train_only(model, mode, train_loader, device, optimizer, train_loss_fn,
                   out_bands=args.out_bands, epochs=epochs,
                   use_poly_lr=use_poly, base_lr=base_lr, poly_power=args.poly_power,
                   scheduler=scheduler)

        ckpt_path = os.path.join(args.save_dir, f"{mode}_final.pt")
        torch.save({"mode": mode, "state_dict": model.state_dict(), "args": vars(args)}, ckpt_path)

        test_metrics = test_once(model, mode, valid_loader, device, args.out_bands,
                                 mrae_eps=args.mrae_eps, mrae_mask_zero=args.mrae_mask_zero)
        results[mode] = test_metrics

        print(f"[TEST ONCE] {mode}")
        print(f"  MRAE  : {test_metrics['mrae']:.6f}")
        print(f"  RMSE  : {test_metrics['rmse']:.6f}")
        print(f"  SAM   : {test_metrics['sam']:.6f} (radians)")
        print(f"  SGrad : {test_metrics['sgrad']:.6f}")
        print(f"  saved : {ckpt_path}")

    print("\n====================\nFINAL SUMMARY\n====================")
    for mode in modes:
        m = results[mode]
        print(f"{mode:8s} | MRAE {m['mrae']:.6f} | RMSE {m['rmse']:.6f} | SAM {m['sam']:.6f} | SGrad {m['sgrad']:.6f}")


if __name__ == "__main__":
    main()
