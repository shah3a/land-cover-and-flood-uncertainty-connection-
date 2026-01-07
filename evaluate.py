import os
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.model_cnn1d import CNN1D

def evaluate_model(config_path: str, model_path: str) -> str:
    cfg = json.load(open(config_path, "r", encoding="utf-8"))
    out_dirs = cfg["outputs"]
    metrics_dir = out_dirs["metrics_dir"]
    os.makedirs(metrics_dir, exist_ok=True)

    df = pd.read_parquet(cfg["samples_path"])
    X = df.drop(columns=["y"]).values.astype(np.float32)
    y = df["y"].values.astype(int)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN1D(n_features=X.shape[1], hidden_channels=cfg["model"]["hidden_channels"], kernel_size=cfg["model"]["kernel_size"], dropout=cfg["model"]["dropout"]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(device))
        prob = torch.sigmoid(logits).cpu().numpy()

    pred = (prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else None
    }

    out_path = os.path.join(metrics_dir, "metrics.json")
    json.dump(metrics, open(out_path, "w", encoding="utf-8"), indent=2)
    return out_path
