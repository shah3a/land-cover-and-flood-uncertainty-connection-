import os
import json
from copy import deepcopy
from src.preprocess_stack import build_stack
from src.samples import build_samples
from src.train import train_model
from src.evaluate import evaluate_model
from src.sensitivity_leave_one_out import leave_one_out
from src.scenario_lulc import apply_lulc_scenario
from src.predict_map import predict_probability_map

def main():
    config_path = "configs/config.json"
    cfg = json.load(open(config_path, "r", encoding="utf-8"))

    os.makedirs("data/processed/stack", exist_ok=True)
    os.makedirs("data/processed/samples", exist_ok=True)
    os.makedirs(cfg["outputs"]["models_dir"], exist_ok=True)
    os.makedirs(cfg["outputs"]["metrics_dir"], exist_ok=True)
    os.makedirs(cfg["outputs"]["maps_dir"], exist_ok=True)
    os.makedirs(cfg["outputs"]["scenario_dir"], exist_ok=True)

    stack_path = build_stack(config_path)
    build_samples(config_path)
    model_path = train_model(config_path)
    evaluate_model(config_path, model_path)

    base_map = os.path.join(cfg["outputs"]["maps_dir"], "flood_susceptibility.tif")
    predict_probability_map(config_path, model_path, stack_path, base_map)

    leave_one_out(config_path)

    for sc in cfg.get("scenarios", []):
        scen_lulc = apply_lulc_scenario(config_path, sc)

        tmp_cfg = deepcopy(cfg)
        tmp_cfg["raster_paths"] = tmp_cfg["raster_paths"].copy()
        tmp_cfg["raster_paths"][sc["lulc_layer_index"]] = scen_lulc
        tmp_cfg_path = os.path.join("configs", f"config_{sc['name']}.json")
        json.dump(tmp_cfg, open(tmp_cfg_path, "w", encoding="utf-8"), indent=2)

        scen_stack = build_stack(tmp_cfg_path)
        scen_map = os.path.join(cfg["outputs"]["scenario_dir"], f"{sc['name']}.tif")
        predict_probability_map(tmp_cfg_path, model_path, scen_stack, scen_map)

if __name__ == "__main__":
    main()
