# training.py (COMPLETE, copy-paste)
# Fixes:
# ✅ EXACT XGB target: df["logk"] = -df["logk"]; keep 0 < logk < 6
# ✅ TRUE scaffold split on filtered df
# ✅ Uses a small INTERNAL val split from TRAIN ONLY for early stopping (test untouched)
# ✅ Reports only TRAIN + TEST metrics (as you want)
# ✅ Adds target sanity-check: verifies Create_Dataset targets match df["logk"]
# ✅ Saves/loads best epoch by val loss (prevents overfit disaster)

import os
import random
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict, Counter

from GNN_photodegradation.featurizer import Create_Dataset, collate_fn
from GNN_photodegradation.models.gat_model import GNNModel
from GNN_photodegradation.evaluations import collect_predictions
from GNN_photodegradation.config import DATA_path, NUM_epochs
from GNN_photodegradation.get_logger import get_logger

logger = get_logger()

# ----------------------- Reproducibility -----------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# ---------------------------------------------------------------

OUT_PREFIX = "GCN"

NUM_FEATS = ["Intensity","Wavelength","Temp","Dosage","InitialC","Humid","Reactor"]

def _safe_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MSE": float(mse), "RMSE": rmse, "MAE": float(mae), "r2": float(r2)}

# -------------------------
# TRUE scaffold split (your exact code)
# -------------------------
def _scaffold_or_none(smiles_str: str):
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        return None
    scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    return scaf if scaf else None

def scaffold_train_test_split(X, y, smiles, test_size=0.3, random_state=0, n_tries=2000):
    scaffold_to_idx = defaultdict(list)
    for i, smi in enumerate(smiles):
        scaffold_to_idx[_scaffold_or_none(smi)].append(i)

    scaffolds = list(scaffold_to_idx.keys())
    target = int(len(smiles) * test_size)

    best_test_idx = None
    best_gap = float("inf")

    rng = np.random.default_rng(random_state)
    for _ in range(n_tries):
        rng.shuffle(scaffolds)
        test_idx = []
        for scaf in scaffolds:
            test_idx.extend(scaffold_to_idx[scaf])
            if len(test_idx) >= target:
                break
        gap = abs(len(test_idx) - target)
        if gap < best_gap:
            best_gap = gap
            best_test_idx = test_idx.copy()
            if best_gap == 0:
                break

    test_set = set(best_test_idx)
    train_idx = [i for i in range(len(smiles)) if i not in test_set]

    return X[train_idx], X[list(test_set)], y[train_idx], y[list(test_set)], train_idx, list(test_set)

def main():
    t0_total = time.time()

    # ----------------------- Load dataset -----------------------
    if not os.path.exists(DATA_path):
        raise FileNotFoundError(f"DATA_path not found: {DATA_path}")

    df = pd.read_excel(DATA_path)
    logger.info(f"Loaded {len(df)} rows.")

    # ----------------------- Clean numeric -----------------------
    df["logk"] = pd.to_numeric(df["logk"], errors="coerce")
    for c in NUM_FEATS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Smile","logk"] + NUM_FEATS).copy()
    df[NUM_FEATS] = df[NUM_FEATS].astype(np.float32)
    df["logk"] = df["logk"].astype(np.float32)

    # ----------------------- EXACT XGB TARGET + FILTER -----------------------
    # XGB: Y_all = -logk ; mask = (0<Y_all<6)
    df["logk"] = (-df["logk"]).astype(np.float32)
    mask = (df["logk"].values > 0.0) & (df["logk"].values < 6.0)
    df = df.loc[mask].copy().reset_index(drop=True)

    print("After XGB-style filter shape:", df.shape)
    print("Target (logk) min/max:", float(df["logk"].min()), float(df["logk"].max()))

    # ----------------------- Scaffold split (filtered universe) -----------------------
    smiles_all = df["Smile"].values
    y_all = df["logk"].values
    X_idx = np.arange(len(df))

    _, _, _, _, train_idx, test_idx = scaffold_train_test_split(
        X_idx, y_all, smiles_all, test_size=0.30, random_state=SEED, n_tries=2000
    )

    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df  = df.iloc[test_idx].copy().reset_index(drop=True)

    print("Train/Test shapes:", train_df.shape, test_df.shape)

    train_scaff = [_scaffold_or_none(s) for s in train_df["Smile"].values]
    test_scaff  = [_scaffold_or_none(s) for s in test_df["Smile"].values]
    print("Scaffold overlap (None excluded):", len((set(train_scaff) & set(test_scaff)) - {None}))
    print("Unique scaffolds train/test:", len(set(train_scaff)), len(set(test_scaff)))

    print("Train y mean/std:", float(train_df["logk"].mean()), float(train_df["logk"].std()))
    print("Test  y mean/std:", float(test_df["logk"].mean()),  float(test_df["logk"].std()))

    # -----------------------
    # INTERNAL val split from TRAIN ONLY (for early stopping)
    # -----------------------
    tr_df, val_df = train_test_split(train_df, test_size=0.15, random_state=SEED)
    tr_df = tr_df.copy().reset_index(drop=True)
    val_df = val_df.copy().reset_index(drop=True)

    # ----------------------- Create datasets -----------------------
    # NOTE: Create_Dataset reads df["logk"] as target.
    tr_dataset  = Create_Dataset(tr_df,  NUM_FEATS)
    scaler = tr_dataset.scaler
    val_dataset = Create_Dataset(val_df, NUM_FEATS, scaler=scaler)
    te_dataset  = Create_Dataset(test_df, NUM_FEATS, scaler=scaler)

    # ----------------------- TARGET SANITY CHECK -----------------------
    # We must ensure Create_Dataset targets match df["logk"] (negated + filtered).
    # This catches hidden transforms in Create_Dataset.
    try:
        # try common attribute names
        if hasattr(tr_dataset, "targets"):
            ds_t = np.asarray(tr_dataset.targets).reshape(-1)
        elif hasattr(tr_dataset, "y"):
            ds_t = np.asarray(tr_dataset.y).reshape(-1)
        else:
            ds_t = None

        if ds_t is not None:
            raw_t = tr_df["logk"].values.reshape(-1)
            print("Sanity check targets (first 5):")
            print("df['logk']:", raw_t[:5])
            print("dataset  :", ds_t[:5])
            print("Target abs diff mean:", float(np.mean(np.abs(raw_t - ds_t))))
        else:
            print("NOTE: Could not auto-read dataset targets attribute; skipping sanity check.")
    except Exception as e:
        print("Sanity check failed:", e)

    # ----------------------- DataLoaders -----------------------
    tr_loader  = DataLoader(tr_dataset,  batch_size=16, shuffle=True,  collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    te_loader  = DataLoader(te_dataset,  batch_size=16, shuffle=False, collate_fn=collate_fn)

    # ----------------------- Model -----------------------
    experimental_input_dim = tr_dataset.experimental_feats.shape[1]
    model = GNNModel(22, experimental_input_dim=experimental_input_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # ----------------------- Train with early stopping (val from train only) -----------------------
    best_val = float("inf")
    best_path = "best_gnn.pth"
    patience = 30
    bad = 0

    t_fit = time.time()
    for epoch in range(1, NUM_epochs + 1):
        model.train()
        tr_losses = []

        for graphs, exp_feats, targets in tr_loader:
            graphs = graphs.to(device)
            exp_feats = exp_feats.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            out, _, _ = model(graphs, exp_feats)

            out = out.view(-1)
            targets = targets.view(-1)

            loss = criterion(out, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            tr_losses.append(loss.item())

        tr_loss = float(np.mean(tr_losses)) if tr_losses else float("nan")

        # val loss
        model.eval()
        v_losses = []
        with torch.no_grad():
            for graphs, exp_feats, targets in val_loader:
                graphs = graphs.to(device)
                exp_feats = exp_feats.to(device)
                targets = targets.to(device)

                out, _, _ = model(graphs, exp_feats)
                out = out.view(-1)
                targets = targets.view(-1)
                v_losses.append(criterion(out, targets).item())

        v_loss = float(np.mean(v_losses)) if v_losses else float("nan")
        logger.info(f"Epoch {epoch}/{NUM_epochs} | train_loss={tr_loss:.4f} | val_loss={v_loss:.4f}")

        if v_loss < best_val:
            best_val = v_loss
            torch.save(model.state_dict(), best_path)
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                logger.info("Early stopping.")
                break

    fit_seconds = time.time() - t_fit

    # ----------------------- Evaluate using BEST epoch -----------------------
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    tr_pred, tr_tgt, *_ = collect_predictions(tr_loader, model, device, criterion)
    te_pred, te_tgt, *_ = collect_predictions(te_loader, model, device, criterion)

    tr_pred = np.asarray(tr_pred).reshape(-1)
    tr_tgt  = np.asarray(tr_tgt).reshape(-1)
    te_pred = np.asarray(te_pred).reshape(-1)
    te_tgt  = np.asarray(te_tgt).reshape(-1)

    # extra debug prints (very important)
    print("Pred stats:")
    print("Train pred mean/std:", float(tr_pred.mean()), float(tr_pred.std()))
    print("Test  pred mean/std:", float(te_pred.mean()), float(te_pred.std()))
    print("Train tgt  mean/std:", float(tr_tgt.mean()),  float(tr_tgt.std()))
    print("Test  tgt  mean/std:", float(te_tgt.mean()),  float(te_tgt.std()))

    train_metrics = _safe_metrics(tr_tgt, tr_pred)
    test_metrics  = _safe_metrics(te_tgt, te_pred)

    metrics_df = pd.DataFrame([
        {"split": "Training", **train_metrics},
        {"split": "Test",     **test_metrics},
    ])
    metrics_path = f"{OUT_PREFIX}_gnn_metrics_train_test.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print("Saved:", metrics_path)
    print(metrics_df)

    # runtime csv
    total_seconds = time.time() - t0_total
    rt = pd.DataFrame([
        {"model": "GNN", "fit_seconds": fit_seconds, "total_seconds": total_seconds, "best_val_loss": best_val}
    ])
    rt_path = f"{OUT_PREFIX}_gnn_runtime_with_total.csv"
    rt.to_csv(rt_path, index=False)
    print("Saved:", rt_path)

if __name__ == "__main__":
    main()
