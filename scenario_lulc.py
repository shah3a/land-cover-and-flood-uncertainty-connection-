import os
import json
import numpy as np
from src.io_rasters import read_raster, write_raster

def apply_lulc_scenario(config_path: str, scenario: dict) -> str:
    cfg = json.load(open(config_path, "r", encoding="utf-8"))
    lulc_path = cfg["raster_paths"][scenario["lulc_layer_index"]]
    arr, prof = read_raster(lulc_path)
    a = arr.copy()

    from_cls = scenario["from_class"]
    to_cls = scenario["to_class"]
    frac = float(scenario["fraction"])

    idx = np.where(a == from_cls)
    n = idx[0].shape[0]
    if n == 0:
        out_path = os.path.join("data/scenarios", f"{scenario['name']}_lulc.tif")
        write_raster(out_path, a, prof)
        return out_path

    k = int(np.floor(n * frac))
    sel = np.random.choice(np.arange(n), size=max(k, 1), replace=False)
    rr = idx[0][sel]
    cc = idx[1][sel]
    a[rr, cc] = to_cls

    out_path = os.path.join("data/scenarios", f"{scenario['name']}_lulc.tif")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_raster(out_path, a, prof)
    return out_path
