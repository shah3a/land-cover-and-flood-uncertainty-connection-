import os
import json
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd

def _extract_from_points(stack_path: str, points_df: pd.DataFrame, x_col: str, y_col: str) -> np.ndarray:
    with rasterio.open(stack_path) as ds:
        coords = list(zip(points_df[x_col].values, points_df[y_col].values))
        vals = np.array([v for v in ds.sample(coords)], dtype=np.float32)
    return vals

def _load_points(cfg: dict) -> pd.DataFrame:
    return pd.read_csv(cfg["label_points_csv"])

def _load_polygons_as_points(cfg: dict) -> pd.DataFrame:
    gdf = gpd.read_file(cfg["label_polygons_path"])
    cent = gdf.geometry.centroid
    df = pd.DataFrame({cfg["x_col"]: cent.x.values, cfg["y_col"]: cent.y.values})
    df[cfg["target_column"]] = gdf[cfg["target_column"]].values
    return df

def build_samples(config_path: str) -> str:
    cfg = json.load(open(config_path, "r", encoding="utf-8"))
    stack_path = cfg["stack_path"]
    samples_path = cfg["samples_path"]
    x_col = cfg["x_col"]
    y_col = cfg["y_col"]
    target = cfg["target_column"]

    if cfg["label_polygons_path"]:
        df = _load_polygons_as_points(cfg)
    else:
        df = _load_points(cfg)

    X = _extract_from_points(stack_path, df, x_col, y_col)
    y = df[target].values

    out = pd.DataFrame(X)
    out["y"] = y
    os.makedirs(os.path.dirname(samples_path), exist_ok=True)
    out.to_parquet(samples_path, index=False)
    return samples_path
