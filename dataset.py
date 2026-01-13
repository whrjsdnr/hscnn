import os
import glob
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from .matio import read_mat_any


class RGBMatFolderDataset(Dataset):
    """
    rgb_dir: jpg/png
    hsi_dir: mat (legacy or v7.3)
    pair by basename
    Robust: skips broken MAT files and logs them.
    """

    def __init__(
        self,
        rgb_dir: str,
        hsi_dir: str,
        train: bool,
        crop_size: Optional[int] = 50,
        seed: int = 0,
        mat_key: Optional[str] = None,
        out_bands: int = 31,
        bad_log_path: str = "bad_mat.log",
        max_skip_tries: int = 50,
        mat_norm: str = "auto_divmax",     # auto_divmax | fixed_div | none
        mat_fixed_div: float = 65535.0
    ):
        self.rgb_dir = rgb_dir
        self.hsi_dir = hsi_dir
        self.train = train
        self.crop_size = crop_size if train else None
        self.rng = np.random.default_rng(seed)
        self.mat_key = mat_key
        self.out_bands = out_bands

        self.bad_log_path = bad_log_path
        self.max_skip_tries = max_skip_tries
        self.mat_norm = mat_norm
        self.mat_fixed_div = mat_fixed_div

        assert os.path.isdir(rgb_dir), f"RGB dir not found: {rgb_dir}"
        assert os.path.isdir(hsi_dir), f"HSI dir not found: {hsi_dir}"

        rgb_files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            rgb_files += glob.glob(os.path.join(rgb_dir, ext))
        mat_files = glob.glob(os.path.join(hsi_dir, "*.mat"))

        rgb_map = {os.path.splitext(os.path.basename(f))[0]: f for f in sorted(rgb_files)}
        mat_map = {os.path.splitext(os.path.basename(f))[0]: f for f in sorted(mat_files)}

        keys = sorted(list(set(rgb_map.keys()).intersection(set(mat_map.keys()))))
        if len(keys) == 0:
            raise RuntimeError("No paired files found. Make sure basenames match between RGB and MAT folders.")

        self.pairs = [(rgb_map[k], mat_map[k], k) for k in keys)

    @staticmethod
    def _load_rgb(path: str) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img).astype(np.float32) / 255.0  # HWC
        arr = np.transpose(arr, (2, 0, 1))  # CHW
        return arr

    def _load_mat(self, path: str) -> np.ndarray:
        cube = read_mat_any(path, mat_key=self.mat_key)
        cube = np.array(cube, dtype=np.float32)

        if self.mat_norm == "none":
            return cube
        if self.mat_norm == "fixed_div":
            div = float(self.mat_fixed_div) if self.mat_fixed_div > 0 else 1.0
            return cube / div
        if self.mat_norm == "auto_divmax":
            if cube.max() > 1.5:
                denom = float(cube.max()) if cube.max() > 0 else 1.0
                cube = cube / denom
            return cube

        raise ValueError(f"Unknown mat_norm: {self.mat_norm}")

    @staticmethod
    def _fix_hsi_axes(cube: np.ndarray, rgb_hw: Tuple[int, int], out_bands: int) -> np.ndarray:
        """
        Align cube to (C,H,W):
          - band axis has size == out_bands
          - remaining two axes match RGB (H,W) or swapped (W,H)
        """
        if cube.ndim != 3:
            raise ValueError(f"HSI cube must be 3D, got {cube.shape}")

        H, W = rgb_hw
        shp = cube.shape

        band_axes = [i for i, s in enumerate(shp) if s == out_bands]
        if len(band_axes) == 0:
            raise ValueError(f"Cannot find band axis == {out_bands} in cube shape={shp}")
        band_axis = band_axes[0]

        spatial_axes = [i for i in range(3) if i != band_axis]
        a0, a1 = spatial_axes
        s0, s1 = shp[a0], shp[a1]

        if (s0, s1) == (H, W):
            order = (band_axis, a0, a1)
        elif (s0, s1) == (W, H):
            order = (band_axis, a1, a0)
        else:
            raise ValueError(f"Cannot align cube shape={shp} to RGB spatial {(H, W)} with out_bands={out_bands}")

        return np.transpose(cube, order)  # (C,H,W)

    def _random_crop(self, rgb: np.ndarray, hsi: np.ndarray, size: int):
        _, H, W = rgb.shape
        if H < size or W < size:
            raise ValueError(f"Crop size {size} bigger than image {H}x{W}")
        top = self.rng.integers(0, H - size + 1)
        left = self.rng.integers(0, W - size + 1)
        return (rgb[:, top:top+size, left:left+size],
                hsi[:, top:top+size, left:left+size])

    def _log_bad(self, mat_path: str, err: Exception):
        try:
            with open(self.bad_log_path, "a", encoding="utf-8") as f:
                f.write(f"{mat_path}\n  {repr(err)}\n")
        except Exception:
            pass

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        tries = 0
        cur = idx

        while tries < self.max_skip_tries:
            rgb_path, mat_path, key = self.pairs[cur]
            rgb = self._load_rgb(rgb_path)

            try:
                hsi_raw = self._load_mat(mat_path)
                H, W = rgb.shape[1], rgb.shape[2]
                hsi = self._fix_hsi_axes(hsi_raw, (H, W), out_bands=self.out_bands)
            except (OSError, ValueError, KeyError) as e:
                self._log_bad(mat_path, e)
                tries += 1
                cur = (cur + 1) % len(self.pairs)
                continue

            if self.train and self.crop_size is not None:
                rgb, hsi = self._random_crop(rgb, hsi, self.crop_size)
                if self.rng.random() < 0.5:
                    rgb = rgb[:, :, ::-1].copy()
                    hsi = hsi[:, :, ::-1].copy()
                if self.rng.random() < 0.5:
                    rgb = rgb[:, ::-1, :].copy()
                    hsi = hsi[:, ::-1, :].copy()

            return torch.from_numpy(rgb), torch.from_numpy(hsi), key

        raise RuntimeError(
            f"Too many bad samples encountered ({self.max_skip_tries}). "
            f"Check '{self.bad_log_path}' for broken MAT files."
        )
