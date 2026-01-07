import os
import json
import numpy as np
import rasterio
import torch
from src.model_cnn1d import CNN1D
from src.io_rasters import write_raster

def predict_probability_map(config_path: str, model_path: str, stack_path: str, out_path: str) -> str:
    cfg = json.load(open(config_path, "r", encoding="utf-8"))
    tile = int(cfg["prediction"]["tile_size"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with rasterio.open(stack_path) as ds:
        prof = ds.profile.copy()
        prof.update(count=1, dtype="float32")
        bands = ds.count
        H = ds.height
        W = ds.width

        model = CNN1D(n_features=bands, hidden_channels=cfg["model"]["hidden_channels"], kernel_size=cfg["model"]["kernel_size"], dropout=cfg["model"]["dropout"]).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        out = np.full((H, W), np.nan, dtype=np.float32)

        for r0 in range(0, H, tile):
            for c0 in range(0, W, tile):
                r1 = min(r0 + tile, H)
                c1 = min(c0 + tile, W)
                block = ds.read(window=((r0, r1), (c0, c1))).astype(np.float32)
                x = np.moveaxis(block, 0, -1).reshape(-1, bands)
                x_t = torch.from_numpy(x).to(device)
                with torch.no_grad():
                    p = torch.sigmoid(model(x_t)).cpu().numpy()
                out[r0:r1, c0:c1] = p.reshape((r1 - r0, c1 - c0)).astype(np.float32)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_raster(out_path, out, prof)
    return out_path
