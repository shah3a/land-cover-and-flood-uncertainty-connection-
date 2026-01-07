import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from src.model_cnn1d import CNN1D

def _split(X, y, seed, train_frac, val_frac, test_frac):
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=(1-train_frac), random_state=seed, stratify=y)
    rel_val = val_frac / (val_frac + test_frac)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=(1-rel_val), random_state=seed, stratify=y_tmp)
    return X_train, y_train, X_val, y_val, X_test, y_test

def _metrics(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else None
    return {"roc_auc": auc}

def train_model(config_path: str) -> str:
    cfg = json.load(open(config_path, "r", encoding="utf-8"))
    mcfg = cfg["model"]
    samples_path = cfg["samples_path"]
    out_dirs = cfg["outputs"]
    models_dir = out_dirs["models_dir"]
    metrics_dir = out_dirs["metrics_dir"]

    df = pd.read_parquet(samples_path)
    X = df.drop(columns=["y"]).values.astype(np.float32)
    y = df["y"].values.astype(np.float32)

    X_train, y_train, X_val, y_val, X_test, y_test = _split(
        X, y, cfg["random_seed"], cfg["train_fraction"], cfg["val_fraction"], cfg["test_fraction"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN1D(n_features=X.shape[1], hidden_channels=mcfg["hidden_channels"], kernel_size=mcfg["kernel_size"], dropout=mcfg["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=mcfg["lr"], weight_decay=mcfg["weight_decay"])
    loss_fn = torch.nn.BCEWithLogitsLoss()

    tr_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    va_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    tr_dl = DataLoader(tr_ds, batch_size=mcfg["batch_size"], shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=mcfg["batch_size"], shuffle=False)

    best_auc = -1
    best_path = None
    patience = 0

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    for epoch in range(mcfg["epochs"]):
        model.train()
        for xb, yb in tr_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        probs = []
        ys = []
        with torch.no_grad():
            for xb, yb in va_dl:
                xb = xb.to(device)
                logits = model(xb)
                p = torch.sigmoid(logits).cpu().numpy()
                probs.append(p)
                ys.append(yb.numpy())
        probs = np.concatenate(probs)
        ys = np.concatenate(ys)
        met = _metrics(ys, probs)
        auc = met["roc_auc"] if met["roc_auc"] is not None else -1

        if auc > best_auc:
            best_auc = auc
            patience = 0
            best_path = os.path.join(models_dir, "best_model.pt")
            torch.save(model.state_dict(), best_path)
        else:
            patience += 1
            if patience >= mcfg["patience"]:
                break

    test_metrics_path = os.path.join(metrics_dir, "train_split.json")
    json.dump(
        {
            "best_val_roc_auc": best_auc,
            "model_path": best_path
        },
        open(test_metrics_path, "w", encoding="utf-8"),
        indent=2
    )
    return best_path
