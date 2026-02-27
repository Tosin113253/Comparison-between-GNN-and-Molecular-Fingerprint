# training.py (COMPLETE, train/test only, XGB-style)
# ✅ NO validation
# ✅ EXACT XGB target: logk := -logk ; keep 0<logk<6
# ✅ TRUE scaffold split (no overlap) with safety for EMPTY/INVALID scaffolds
# ✅ Saves metrics + runtime to CSV

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
from collections import defaultdict

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
# TRUE scaffold split (your code) + small safety for EMPTY scaffolds
# -------------------------
def _scaffold_or_none(smiles_str: str):
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        return "INVALID_SMILES"
    scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    return scaf if scaf else "EMPTY_SCAFFOLD"

def scaffold_train_test_split(X, y, smiles, test_size=0.3, random_state=0, n_tries=2000):
    scaffold_to_idx = defaultdict(list)
    for i, smi in enumerate(smiles):
        scaffold_to_idx[_scaffold_or_none(smi)].append(i)

    # IMPORTANT: keep EMPTY/INVALID scaffolds in TRAIN (prevents weird leakage + tiny test)
    forced_train_scaffolds = {"EMPTY_SCAFFOLD", "INVALID_SMILES"}
    normal_scaffolds = [s for s in scaffold_to_idx.keys() if s not in forced_train_scaffolds]

    target = int(len(smiles) * test_size)
    best_test_idx = None
    best_gap = float("inf")

    rng = np.random.default_rng(random_state)
    for _ in range(n_tries):
        rng.shuffle(normal_scaffolds)
        test_idx = []
        for scaf in normal_scaffolds:
            test_idx.extend(scaffold_to_idx[scaf])
            if len(test_idx) >= target:
                break

        gap = abs(len(test_idx) - target)
        if gap < best_gap and len(test_idx) > 0:
            best_gap = gap
            best_test_idx = test_idx.copy()
            if best_gap == 0:
                break

    if best_test_idx is None or len(best_test_idx) == 0:
        raise ValueError("Could not form a non-empty test split from non-empty scaffolds.")

    test_set = set(best_test_idx)

    # train is everything else + forced scaffolds
    train_idx = [i for i in range(len(smiles)) if i not in test_set]

    # final guard
    if len(train_idx) == 0:
        raise ValueError("Train split ended empty. Too few samples/scaffolds for this test_size.")
    if len(test_set) == 0:
        raise ValueError("Test split ended empty. Too few samples/scaffolds for this test_size.")

    return X[train_idx], X[list(test_set)], y[train_idx], y[list(test_set)], train_idx, list(test_set)

def main():
    t0_total = time.time()

    if not os.path.exists(DATA_path):
        raise FileNotFoundError(f"DATA_path not found: {DATA_path}")

    df = pd.read_excel(DATA_path)
    print("Loaded:", df.shape)

    # numeric coercion
    df["logk"] = pd.to_numeric(df["logk"], errors="coerce")
    for c in NUM_FEATS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Smile","logk"] + NUM_FEATS).copy().reset_index(drop=True)
    df[NUM_FEATS] = df[NUM_FEATS].astype(np.float32)
    df["logk"] = df["logk"].astype(np.float32)

    # EXACT XGB target + filter
    df["logk"] = (-df["logk"]).astype(np.float32)
    df = df[(df["logk"] > 0.0) & (df["logk"] < 6.0)].copy().reset_index(drop=True)

    print("After filter:", df.shape)
    if len(df) < 20:
        raise ValueError("Too few samples after filter. Scaffold split will be unstable.")

    smiles = df["Smile"].values
    Y = df["logk"].values
    X = np.arange(len(df))  # just indices, we split indices then slice df

    # scaffold split train/test only
    _, _, _, _, train_idx, test_idx = scaffold_train_test_split(
        X, Y, smiles, test_size=0.30, random_state=SEED, n_tries=2000
    )

    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df  = df.iloc[test_idx].copy().reset_index(drop=True)

    print("Train/Test sizes:", len(train_df), len(test_df))

    # no overlap check
    train_scaff = set(_scaffold_or_none(s) for s in train_df["Smile"].values)
    test_scaff  = set(_scaffold_or_none(s) for s in test_df["Smile"].values)
    overlap = len(train_scaff & test_scaff)
    print("Scaffold overlap (should be 0):", overlap)

    # datasets
    train_dataset = Create_Dataset(train_df, NUM_FEATS)
    scaler = train_dataset.scaler
    test_dataset = Create_Dataset(test_df, NUM_FEATS, scaler=scaler)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # model
    experimental_input_dim = train_dataset.experimental_feats.shape[1]
    model = GNNModel(22, experimental_input_dim=experimental_input_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-4)

    # train
    t_fit = time.time()
    for epoch in range(1, NUM_epochs + 1):
        model.train()
        losses = []
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
            losses.append(loss.item())

        logger.info(f"Epoch {epoch}/{NUM_epochs} train_loss={float(np.mean(losses)):.4f}")

    fit_seconds = time.time() - t_fit

    # eval
    model.eval()
    train_pred, train_tgt, *_ = collect_predictions(train_loader, model, device, criterion)
    test_pred, test_tgt, *_   = collect_predictions(test_loader, model, device, criterion)

    train_pred = np.asarray(train_pred).reshape(-1)
    train_tgt  = np.asarray(train_tgt).reshape(-1)
    test_pred  = np.asarray(test_pred).reshape(-1)
    test_tgt   = np.asarray(test_tgt).reshape(-1)

    metrics_df = pd.DataFrame([
        {"split":"Training", **_safe_metrics(train_tgt, train_pred)},
        {"split":"Test",     **_safe_metrics(test_tgt,  test_pred)},
    ])

    print(metrics_df)

    # save outputs like your XGB style
    metrics_path = f"{OUT_PREFIX}_train_test_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    runtime_df = pd.DataFrame([{
        "runtime/final_fit_seconds": float(fit_seconds),
        "runtime/total_seconds": float(time.time() - t0_total),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_total_filtered": int(len(df)),
        "scaffold_overlap": int(overlap),
    }])
    runtime_path = f"{OUT_PREFIX}_runtime_train_test.csv"
    runtime_df.to_csv(runtime_path, index=False)

    print("Saved:", metrics_path)
    print("Saved:", runtime_path)

if __name__ == "__main__":
    main()
