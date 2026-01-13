import torch
import torch.nn as nn


def rgb_to_spectral_interp(rgb: torch.Tensor, out_bands: int = 31,
                          wl_min: float = 400.0, wl_max: float = 700.0,
                          wl_rgb=(450.0, 550.0, 650.0)) -> torch.Tensor:
    """
    Baseline 'hscnn' only: simple piecewise-linear interpolation RGB->spectral.
    """
    B, C, H, W = rgb.shape
    assert C == 3
    wls = torch.linspace(wl_min, wl_max, out_bands, device=rgb.device)
    wl_b, wl_g, wl_r = wl_rgb

    b = rgb[:, 0:1]
    g = rgb[:, 1:2]
    r = rgb[:, 2:3]

    out = []
    for wl in wls:
        if wl <= wl_g:
            t = (wl - wl_b) / (wl_g - wl_b + 1e-12)
            t = torch.clamp(t, 0.0, 1.0)
            band = (1 - t) * b + t * g
        else:
            t = (wl - wl_g) / (wl_r - wl_g + 1e-12)
            t = torch.clamp(t, 0.0, 1.0)
            band = (1 - t) * g + t * r
        out.append(band)
    return torch.cat(out, dim=1)


class HSCNN_Original(nn.Module):
    def __init__(self, out_bands=31, depth=5, width=64):
        super().__init__()
        assert depth >= 3
        self.head = nn.Sequential(nn.Conv2d(out_bands, width, 3, padding=1), nn.ReLU(inplace=True))
        body = []
        for _ in range(depth - 2):
            body += [nn.Conv2d(width, width, 3, padding=1), nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*body)
        self.tail = nn.Conv2d(width, out_bands, 3, padding=1)

    def forward(self, upspec):
        x = self.head(upspec)
        x = self.body(x)
        res = self.tail(x)
        return upspec + res


class HSCNN_u(nn.Module):
    def __init__(self, out_bands=31, depth=5, width=64):
        super().__init__()
        self.spec_up = nn.Conv2d(3, out_bands, 1, bias=False)
        self.core = HSCNN_Original(out_bands=out_bands, depth=depth, width=width)

    def forward(self, rgb):
        upspec = self.spec_up(rgb)
        return self.core(upspec)


class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.r = nn.ReLU(inplace=True)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.c2(self.r(self.c1(x))) + x


class HSCNN_R(nn.Module):
    def __init__(self, out_bands=31, n_blocks=16, width=64):
        super().__init__()
        self.spec_up = nn.Conv2d(3, out_bands, 1, bias=False)
        self.head = nn.Sequential(nn.Conv2d(out_bands, width, 3, padding=1), nn.ReLU(inplace=True))
        self.body = nn.Sequential(*[ResidualBlock(width) for _ in range(n_blocks)])
        self.tail = nn.Conv2d(width, out_bands, 3, padding=1)

    def forward(self, rgb):
        base = self.spec_up(rgb)
        x = self.head(base)
        x = self.body(x)
        res = self.tail(x)
        return base + res


class PWFDenseBlock(nn.Module):
    """
    Paper-like PWF:
    - 3x3 paths: 16 and 16
    - 1x1 paths: 8 and 8
    - concat -> 1x1 -> 16
    """
    def __init__(self, in_ch: int, g3: int = 16, g1: int = 8, fuse_out: int = 16):
        super().__init__()
        self.p3a = nn.Sequential(nn.Conv2d(in_ch, g3, 3, padding=1), nn.ReLU(inplace=True))
        self.p3b = nn.Sequential(nn.Conv2d(in_ch, g3, 3, padding=1), nn.ReLU(inplace=True))
        self.p1a = nn.Sequential(nn.Conv2d(in_ch, g1, 1), nn.ReLU(inplace=True))
        self.p1b = nn.Sequential(nn.Conv2d(in_ch, g1, 1), nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(nn.Conv2d(g3 + g3 + g1 + g1, fuse_out, 1), nn.ReLU(inplace=True))

    def forward(self, x):
        a = self.p3a(x); b = self.p3b(x)
        c = self.p1a(x); d = self.p1b(x)
        return self.fuse(torch.cat([a, b, c, d], dim=1))


class HSCNN_D_PaperLike(nn.Module):
    def __init__(self, out_bands=31, n_blocks=38, stem_width=64, fuse_out=16, g3=16, g1=8):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, stem_width, 3, padding=1), nn.ReLU(inplace=True))

        self.blocks = nn.ModuleList()
        in_ch = stem_width
        for _ in range(n_blocks):
            self.blocks.append(PWFDenseBlock(in_ch, g3=g3, g1=g1, fuse_out=fuse_out))
            in_ch += fuse_out

        self.compress = nn.Sequential(nn.Conv2d(in_ch, 64, 1), nn.ReLU(inplace=True))
        self.recon = nn.Conv2d(64, out_bands, 1)

        # paper: last layer init N(0, 0.001)
        nn.init.normal_(self.recon.weight, mean=0.0, std=0.001)
        if self.recon.bias is not None:
            nn.init.zeros_(self.recon.bias)

    def forward(self, rgb):
        f0 = self.stem(rgb)
        feats = [f0]
        xcat = f0
        for blk in self.blocks:
            nf = blk(xcat)
            feats.append(nf)
            xcat = torch.cat(feats, dim=1)
        x = self.compress(xcat)
        return self.recon(x)


def build_model(mode: str, out_bands: int, depth: int, width: int, r_blocks: int, d_blocks: int):
    if mode == "hscnn":
        return HSCNN_Original(out_bands=out_bands, depth=depth, width=width)
    if mode == "hscnn_u":
        return HSCNN_u(out_bands=out_bands, depth=depth, width=width)
    if mode == "hscnn_r":
        return HSCNN_R(out_bands=out_bands, n_blocks=r_blocks, width=width)
    if mode == "hscnn_d":
        return HSCNN_D_PaperLike(out_bands=out_bands, n_blocks=d_blocks, stem_width=64)
    raise ValueError(f"Unknown mode: {mode}")
