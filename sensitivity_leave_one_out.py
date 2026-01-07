import os
import json
import pandas as pd
from copy import deepcopy
from src.train import train_model
from src.evaluate import evaluate_model

def leave_one_out(config_path: str) -> str:
    cfg = json.load(open(config_path, "r", encoding="utf-8"))
    df = pd.read_parquet(cfg["samples_path"])
    feature_cols = [c for c in df.columns if c != "y"]

    out_csv = os.path.join(cfg["outputs"]["metrics_dir"], "sensitivity_leave_one_out.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    base_model_path = train_model(config_path)
    base_metrics_path = evaluate_model(config_path, base_model_path)
    base_metrics = json.load(open(base_metrics_path, "r", encoding="utf-8"))

    rows = [{"dropped_feature": "none", "roc_auc": base_metrics.get("roc_auc")}]

    for i, feat in enumerate(feature_cols):
        tmp = df.drop(columns=[feat]).copy()
        tmp_path = cfg["samples_path"].replace(".parquet", f"_drop_{i}.parquet")
        tmp.to_parquet(tmp_path, index=False)

        tmp_cfg = deepcopy(cfg)
        tmp_cfg["samples_path"] = tmp_path
        tmp_cfg_path = config_path.replace(".json", f"_drop_{i}.json")
        json.dump(tmp_cfg, open(tmp_cfg_path, "w", encoding="utf-8"), indent=2)

        mp = train_model(tmp_cfg_path)
        metp = evaluate_model(tmp_cfg_path, mp)
        met = json.load(open(metp, "r", encoding="utf-8"))
        rows.append({"dropped_feature": feat, "roc_auc": met.get("roc_auc")})

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv
