# training.py (COMPLETE, scaffold train/val/test, best-epoch, stronger regularization)
# ✅ EXACT XGB target: df["logk"] = -df["logk"]; keep 0<logk<6
# ✅ TRUE scaffold split for TRAIN/VAL/TEST (NO overlap)
# ✅ Early stopping on SCAFFOLD-VAL (meaningful for OOD)
# ✅ Stronger regularization (dropout+weight_decay+smaller dims)
# ✅ Reports TRAIN + VAL + TEST metrics (so you can see if val matches test)

import os
import random
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict, Counter

from torch.optim.lr_scheduler import ReduceLROnPlateau

from GNN_photodegradation.featurizer import Create_Dataset, collate_fn
from GNN_photodegradation.models.gat_model import GNNModel
from GNN_photodegradation.evaluations import collect_predictions
from GNN_photodegradation.config import DATA_path, NUM_epochs
from GNN_photodegradation.get_logger import get_logger

logger = get_logger()

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
# your scaffold helper
# -------------------------
def _scaffold_or_none(smiles_str: str):
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        return None
    scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    return scaf if scaf else None

def scaffold_split_indices(smiles, frac_train=0.70, frac_val=0.15, frac_test=0.15, random_state=42):
    """
    Deterministic scaffold split into train/val/test with NO overlap.
    """
    assert abs(frac_train + frac_val + frac_test - 1.0) < 1e-6

    scaffold_to_idx = defaultdict(list)
    for i, smi in enumerate(smiles):
        scaffold_to_idx[_scaffold_or_none(smi)].append(i)

    scaffolds = list(scaffold_to_idx.keys())
    rng = np.random.default_rng(random_state)
    rng.shuffle(scaffolds)

    n = len(smiles)
    n_test_target = int(frac_test * n)
    n_val_target  = int(frac_val * n)

    test_idx, val_idx, train_idx = [], [], []

    # fill test
    for scaf in scaffolds:
        if len(test_idx) < n_test_target:
            test_idx.extend(scaffold_to_idx[scaf])
        else:
            break

    remaining = [s for s in scaffolds if s not in set([_scaffold_or_none(smiles[i]) for i in test_idx])]

    # fill val from remaining scaffolds
    for scaf in remaining:
        if len(val_idx) < n_val_target:
            val_idx.extend(scaffold_to_idx[scaf])
        else:
            break

    used = set(test_idx) | set(val_idx)
    train_idx = [i for i in range(n) if i not in used]

    return train_idx, val_idx, test_idx

def main():
    t0 = time.time()

    # ----------------------- Load -----------------------
    if not os.path.exists(DATA_path):
        raise FileNotFoundError(f"DATA_path not found: {DATA_path}")

    df = pd.read_excel(DATA_path)
    print("Loaded:", df.shape)

    # ----------------------- Clean -----------------------
    df["logk"] = pd.to_numeric(df["logk"], errors="coerce")
    for c in NUM_FEATS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Smile","logk"] + NUM_FEATS).copy()
    df[NUM_FEATS] = df[NUM_FEATS].astype(np.float32)
    df["logk"] = df["logk"].astype(np.float32)

    # ----------------------- EXACT XGB target + filter -----------------------
    df["logk"] = (-df["logk"]).astype(np.float32)
    df = df[(df["logk"] > 0.0) & (df["logk"] < 6.0)].copy().reset_index(drop=True)

    print("After filter:", df.shape)
    print("Target min/max:", float(df["logk"].min()), float(df["logk"].max()))

    smiles = df["Smile"].values

    # ----------------------- Scaffold train/val/test split -----------------------
    train_idx, val_idx, test_idx = scaffold_split_indices(
        smiles, frac_train=0.70, frac_val=0.15, frac_test=0.15, random_state=SEED
    )

    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    val_df   = df.iloc[val_idx].copy().reset_index(drop=True)
    test_df  = df.iloc[test_idx].copy().reset_index(drop=True)

    print("Split sizes:", len(train_df), len(val_df), len(test_df))

    # overlap checks
    train_sc = set(_scaffold_or_none(s) for s in train_df["Smile"].values)
    val_sc   = set(_scaffold_or_none(s) for s in val_df["Smile"].values)
    test_sc  = set(_scaffold_or_none(s) for s in test_df["Smile"].values)
    print("Scaffold overlaps:",
          "train∩val", len((train_sc & val_sc) - {None}),
          "train∩test", len((train_sc & test_sc) - {None}),
          "val∩test", len((val_sc & test_sc) - {None}))

    # ----------------------- Datasets -----------------------
    train_dataset = Create_Dataset(train_df, NUM_FEATS)
    scaler = train_dataset.scaler
    val_dataset   = Create_Dataset(val_df, NUM_FEATS, scaler=scaler)
    test_dataset  = Create_Dataset(test_df, NUM_FEATS, scaler=scaler)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # ----------------------- Model -----------------------
    experimental_input_dim = train_dataset.experimental_feats.shape[1]

    # SMALLER model + rely on dropout inside your model
    model = GNNModel(22, experimental_input_dim=experimental_input_dim).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    device = next(model.parameters()).device

    criterion = nn.MSELoss()

    # lower LR + weight decay helps generalization
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)

    # ----------------------- Train with early stopping on scaffold-val -----------------------
    best_path = "best_gnn_scaffold.pth"
    best_val = float("inf")
    patience = 25
    bad = 0

    for epoch in range(1, NUM_epochs + 1):
        model.train()
        tr_losses = []

        for graphs, exp_feats, targets in train_loader:
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
        scheduler.step(v_loss)

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

    # ----------------------- Evaluate BEST -----------------------
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    tr_pred, tr_tgt, *_ = collect_predictions(train_loader, model, device, criterion)
    va_pred, va_tgt, *_ = collect_predictions(val_loader, model, device, criterion)
    te_pred, te_tgt, *_ = collect_predictions(test_loader, model, device, criterion)

    tr_pred = np.asarray(tr_pred).reshape(-1); tr_tgt = np.asarray(tr_tgt).reshape(-1)
    va_pred = np.asarray(va_pred).reshape(-1); va_tgt = np.asarray(va_tgt).reshape(-1)
    te_pred = np.asarray(te_pred).reshape(-1); te_tgt = np.asarray(te_tgt).reshape(-1)

    metrics = pd.DataFrame([
        {"split":"Training", **_safe_metrics(tr_tgt, tr_pred)},
        {"split":"Val",      **_safe_metrics(va_tgt, va_pred)},
        {"split":"Test",     **_safe_metrics(te_tgt, te_pred)},
    ])
    out_path = f"{OUT_PREFIX}_gnn_metrics_scaffold_train_val_test.csv"
    metrics.to_csv(out_path, index=False)
    print("Saved:", out_path)
    print(metrics)

    print("Runtime seconds:", float(time.time() - t0))

if __name__ == "__main__":
    main()
