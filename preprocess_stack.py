import json
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from src.io_rasters import read_raster, write_stack

def _match_to_template(src_path: str, template_profile: dict) -> np.ndarray:
    with rasterio.open(src_path) as src:
        src_arr = src.read(1)
        src_transform = src.transform
        src_crs = src.crs
        dst_h = template_profile["height"]
        dst_w = template_profile["width"]
        dst_arr = np.empty((dst_h, dst_w), dtype=np.float32)
        reproject(
            source=src_arr,
            destination=dst_arr,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=template_profile["transform"],
            dst_crs=template_profile["crs"],
            resampling=Resampling.bilinear
        )
    return dst_arr

def build_stack(config_path: str) -> str:
    cfg = json.load(open(config_path, "r", encoding="utf-8"))
    raster_paths = cfg["raster_paths"]
    stack_path = cfg["stack_path"]

    _, base_profile = read_raster(raster_paths[0])
    base_profile = base_profile.copy()
    base_profile.update(dtype="float32", count=1)

    layers = []
    for p in raster_paths:
        arr = _match_to_template(p, base_profile).astype(np.float32)
        layers.append(arr)

    stack = np.stack(layers, axis=0)
    write_stack(stack_path, stack, base_profile)
    return stack_path
