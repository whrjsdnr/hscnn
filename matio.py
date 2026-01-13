from typing import Optional
import numpy as np
from scipy.io import loadmat
import h5py


def _h5_first_3d_dataset(f: h5py.File) -> Optional[str]:
    found = None

    def visitor(name, obj):
        nonlocal found
        if found is not None:
            return
        if isinstance(obj, h5py.Dataset) and obj.ndim == 3:
            found = name

    f.visititems(visitor)
    return found


def read_mat_any(path: str, mat_key: Optional[str] = None) -> np.ndarray:
    """
    Read 3D cube from .mat
    - try scipy loadmat (legacy)
    - if v7.3 => h5py
    """
    try:
        m = loadmat(path)
        if mat_key is not None:
            if mat_key not in m:
                raise KeyError(f"mat_key='{mat_key}' not in legacy MAT. Keys={list(m.keys())}")
            cube = m[mat_key]
        else:
            cube = None
            for k, v in m.items():
                if k.startswith("__"):
                    continue
                if isinstance(v, np.ndarray) and v.ndim == 3 and np.issubdtype(v.dtype, np.number):
                    cube = v
                    break
            if cube is None:
                raise ValueError("No 3D numeric array found in legacy MAT.")
        return cube
    except NotImplementedError:
        pass  # v7.3

    with h5py.File(path, "r") as f:
        if mat_key is not None:
            if mat_key in f:
                cube = f[mat_key][()]
            else:
                raise KeyError(f"mat_key='{mat_key}' not found in v7.3(HDF5) MAT.")
        else:
            key = _h5_first_3d_dataset(f)
            if key is None:
                raise ValueError("No 3D dataset found in v7.3(HDF5) MAT. Provide --mat_key.")
            cube = f[key][()]
        return cube
