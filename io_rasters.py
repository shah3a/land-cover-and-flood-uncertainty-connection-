import os
import numpy as np
import rasterio

def read_raster(path: str):
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        profile = ds.profile
    return arr, profile

def write_raster(path: str, arr: np.ndarray, profile: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    prof = profile.copy()
    prof.update(count=1, dtype=str(arr.dtype))
    with rasterio.open(path, "w", **prof) as ds:
        ds.write(arr, 1)

def write_stack(path: str, stack: np.ndarray, profile: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    prof = profile.copy()
    prof.update(count=stack.shape[0], dtype=str(stack.dtype))
    with rasterio.open(path, "w", **prof) as ds:
        ds.write(stack)
