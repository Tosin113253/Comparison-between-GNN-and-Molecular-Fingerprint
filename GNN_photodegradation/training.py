# training.py (corrected, copy-paste)
# FIXES:
# ✅ scaffold split unchanged
# ✅ train/test only
# ✅ baseline tabular + SHAP unchanged
# ✅ GNN training fixed: shuffle=True + shape-safe loss + gradient clipping
# ✅ added sanity checks for y distribution + scaffold distribution
# ✅ runtime CSV for GNN (final_fit_seconds + total_seconds)

import os
import random
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

import shap
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict, Counter

from GNN_photodegradation.featurizer import Create_Dataset, collate_fn
from GNN_photodegradation.models.gat_model import GNNModel
from GNN_photodegradation.evaluations import collect_predictions, compute_regression_stats
from GNN_photodegradation.plots import (
    plot_calculated_vs_experimental,
    plot_pca,
    plot_umap,
    plot_williams,
)
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

out_prefix = "GCN"


def _safe_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MSE": float(mse), "RMSE": rmse, "MAE": float(mae), "r2": float(r2)}


def run_experimental_baselines_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    out_prefix: str,
    make_shap: bool = True,
):
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

    results_metrics = []
    results_runtime_rows = []

    # Ridge
    ridge = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=SEED)),
        ]
    )
    t_fit = time.time()
    ridge.fit(X_train, y_train)
    ridge_fit_seconds = time.time() - t_fit

    results_metrics.append({"model": "Ridge", "split": "train", **_safe_metrics(y_train, ridge.predict(X_train))})
    results_metrics.append({"model": "Ridge", "split": "test",  **_safe_metrics(y_test,  ridge.predict(X_test))})

    results_runtime_rows.append({
        "model": "Ridge",
        "trial_runtime_seconds": float(ridge_fit_seconds),
        "train_r2": float(ridge.score(X_train, y_train)),
        "test_r2": float(ridge.score(X_test, y_test)),
    })

    # RandomForest
    rf = RandomForestRegressor(
        n_estimators=500,
        random_state=SEED,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    t_fit = time.time()
    rf.fit(X_train, y_train)
    rf_fit_seconds = time.time() - t_fit

    results_metrics.append({"model": "RandomForest", "split": "train", **_safe_metrics(y_train, rf.predict(X_train))})
    results_metrics.append({"model": "RandomForest", "split": "test",  **_safe_metrics(y_test,  rf.predict(X_test))})

    results_runtime_rows.append({
        "model": "RandomForest",
        "trial_runtime_seconds": float(rf_fit_seconds),
        "train_r2": float(rf.score(X_train, y_train)),
        "test_r2": float(rf.score(X_test, y_test)),
    })

    metrics_df = pd.DataFrame(results_metrics)
    metrics_path = f"{out_prefix}_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    fi = pd.DataFrame(
        {"Feature": feature_cols, "Importance": rf.feature_importances_.astype(float)}
    ).sort_values("Importance", ascending=False)
    fi_path = f"{out_prefix}_feature_importance.csv"
    fi.to_csv(fi_path, index=False)

    shap_path = None
    if make_shap:
        try:
            n_plot = min(500, X_train.shape[0])
            rng = np.random.default_rng(SEED)
            plot_idx = rng.choice(X_train.shape[0], size=n_plot, replace=False)

            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X_train[plot_idx])

            plt.figure()
            shap.summary_plot(
                shap_values,
                X_train[plot_idx],
                feature_names=feature_cols,
                show=False,
                max_display=len(feature_cols),
            )
            shap_path = f"{out_prefix}_shap_beeswarm.png"
            plt.tight_layout()
            plt.savefig(shap_path, dpi=300)
            plt.close()
        except Exception as e:
            logger.warning(f"SHAP plot failed for baseline: {e}")

    res_df = pd.DataFrame(results_runtime_rows)
    final_row = {k: np.nan for k in res_df.columns}
    final_row["model"] = "FINAL"
    final_row["trial_runtime_seconds"] = np.nan
    final_row["runtime/total_seconds"] = float(ridge_fit_seconds + rf_fit_seconds)
    out_df = pd.concat([res_df, pd.DataFrame([final_row])], ignore_index=True)
    out_path = f"{out_prefix}_results_with_runtime.csv"
    out_df.to_csv(out_path, index=False)

    return metrics_path, fi_path, shap_path, out_path


# -------------------------
# TRUE scaffold split (exactly your version)
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

    dataset_path = DATA_path
    num_epochs = NUM_epochs

    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file not found at {dataset_path}")
        return

    try:
        df = pd.read_excel(dataset_path)
        logger.info(f"Dataset loaded successfully with {len(df)} records.")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    required_columns = {
        "Smile","logk","Intensity","Wavelength","Temp","Dosage","InitialC","Humid","Reactor"
    }
    if not required_columns.issubset(df.columns):
        missing = sorted(list(required_columns - set(df.columns)))
        logger.error(f"Dataset missing required columns: {missing}")
        logger.error(f"Columns found: {list(df.columns)}")
        return

    df["logk"] = pd.to_numeric(df["logk"], errors="coerce")

    numerical_features = ["Intensity","Wavelength","Temp","Dosage","InitialC","Humid","Reactor"]
    for col in numerical_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["Smile", "logk"] + numerical_features).copy()
    after = len(df)
    if after < before:
        logger.info(f"Dropped {before - after} rows due to NaNs in required columns.")

    df[numerical_features] = df[numerical_features].astype(np.float32)
    df["logk"] = df["logk"].astype(np.float32)

    # -----------------------
    # Scaffold split (train/test only)
    # -----------------------
    smiles_all = df["Smile"].values
    Y = df["logk"].values
    X = np.arange(len(df))

    _, _, _, _, train_idx, test_idx = scaffold_train_test_split(
        X, Y, smiles_all, test_size=0.30, random_state=SEED, n_tries=2000
    )

    train_df = df.iloc[train_idx].copy()
    test_df  = df.iloc[test_idx].copy()

    print("Shapes:", train_df.shape, test_df.shape)
    train_scaff = [_scaffold_or_none(smiles_all[i]) for i in train_idx]
    test_scaff  = [_scaffold_or_none(smiles_all[i]) for i in test_idx]
    print("Scaffold overlap (None excluded):", len((set(train_scaff) & set(test_scaff)) - {None}))
    print("Train scaffold counts (top 5):", Counter(train_scaff).most_common(5))
    print("Test  scaffold counts (top 5):", Counter(test_scaff).most_common(5))

    # --- sanity on target distribution ---
    print("Train y mean/std:", float(train_df["logk"].mean()), float(train_df["logk"].std()))
    print("Test  y mean/std:", float(test_df["logk"].mean()),  float(test_df["logk"].std()))

    # -----------------------
    # Leakage-safe TE for baseline only
    # -----------------------
    global_mean = float(train_df["logk"].mean())
    te_map = train_df.groupby("Smile")["logk"].mean().to_dict()
    for dfx in (train_df, test_df):
        dfx["OrganicContaminant_TE"] = dfx["Smile"].map(te_map).fillna(global_mean).astype(np.float32)

    # ------------------- Baseline -------------------
    baseline_feature_cols = numerical_features + ["OrganicContaminant_TE"]
    metrics_path, fi_path, shap_path, baseline_runtime_csv = run_experimental_baselines_train_test(
        train_df=train_df,
        test_df=test_df,
        feature_cols=baseline_feature_cols,
        target_col="logk",
        out_prefix="baseline_tabular",
        make_shap=True,
    )
    logger.info(f"Baseline metrics saved: {metrics_path}")
    logger.info(f"Baseline FI saved: {fi_path}")
    logger.info(f"Baseline runtime CSV saved: {baseline_runtime_csv}")
    if shap_path:
        logger.info(f"Baseline SHAP saved: {shap_path}")

    # ------------------- GNN datasets -------------------
    train_dataset = Create_Dataset(train_df, numerical_features)
    scaler = train_dataset.scaler
    test_dataset = Create_Dataset(test_df, numerical_features, scaler=scaler)

    # IMPORTANT FIX: shuffle=True for training
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=8, shuffle=False, collate_fn=collate_fn)

    experimental_input_dim = train_dataset.experimental_feats.shape[1]
    model = GNNModel(22, experimental_input_dim=experimental_input_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # 1e-4 can be too slow; 1e-3 is standard

    # ------------------- Training -------------------
    t_fit = time.time()
    for epoch in range(1, num_epochs + 1):
        model.train()
        losses = []

        for graphs, exp_feats, targets in train_loader:
            graphs = graphs.to(device)
            exp_feats = exp_feats.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs, _, _ = model(graphs, exp_feats)

            # IMPORTANT FIX: shape-safe loss (prevents silent broadcasting)
            outputs = outputs.view(-1)
            targets = targets.view(-1)

            loss = criterion(outputs, targets)
            loss.backward()

            # helpful stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            losses.append(loss.item())

        logger.info(f"Epoch {epoch}/{num_epochs} - Train Loss: {float(np.mean(losses)):.4f}")

    final_fit_seconds = time.time() - t_fit

    # ------------------- Evaluation -------------------
    model.eval()
    train_pred, train_tgt, train_feats, train_graph_feats, _ = collect_predictions(train_loader, model, device, criterion)
    test_pred,  test_tgt,  test_feats,  test_graph_feats,  _ = collect_predictions(test_loader,  model, device, criterion)

    # Plots + metrics
    results = []
    dsname = []
    train_labels = np.array(train_idx) + 1
    test_labels  = np.array(test_idx) + 1

    for pred, tgt, name, label in zip(
        [train_pred, test_pred],
        [train_tgt, test_tgt],
        ["Training", "Test"],
        [train_labels, test_labels],
    ):
        slope, intercept, slope_sd, intercept_sd, result = compute_regression_stats(tgt, pred)
        results.append(result)
        dsname.append(name)
        plot_calculated_vs_experimental(pred.flatten(), tgt.flatten(), name, label, slope, intercept)

    pd.DataFrame(results, index=dsname).to_excel(f"{out_prefix}_Regression_results.xlsx")

    # PCA/UMAP combined (train+test)
    combined_exp_feats = np.vstack((train_feats, test_feats))
    combined_graph_feats = np.vstack((train_graph_feats, test_graph_feats))
    combined_targets = np.vstack((train_tgt, test_tgt))

    plot_pca(combined_exp_feats, combined_graph_feats, combined_targets.flatten(), "Combined", "2D PCA Plot", dimensions=2)
    plot_pca(combined_exp_feats, combined_graph_feats, combined_targets.flatten(), "Combined", "3D PCA Plot", dimensions=3)
    plot_umap(combined_exp_feats, combined_graph_feats, combined_targets.flatten(), "Combined", title="2D UMAP Plot", dimensions=2)
    plot_umap(combined_exp_feats, combined_graph_feats, combined_targets.flatten(), "Combined", title="3D UMAP Plot", dimensions=3)

    # Williams plot often expects 3 splits; skip safely if needed
    try:
        dataset_dict = {
            "train": np.hstack((train_feats, train_graph_feats)),
            "test":  np.hstack((test_feats,  test_graph_feats)),
        }
        # if your plot_williams strictly needs 3, it will error -> caught below
        plot_williams(
            dataset_dict["train"],
            dataset_dict["train"],  # placeholder
            dataset_dict["test"],
            train_pred,
            train_pred,             # placeholder
            test_pred,
            train_tgt,
            train_tgt,              # placeholder
            test_tgt,
            train_labels,
            train_labels,           # placeholder
            test_labels,
        )
    except Exception as e:
        logger.warning(f"Skipping Williams plot (needs 3 splits): {e}")

    # ------------------- Runtime CSV (like your format) -------------------
    total_seconds = time.time() - t0_total

    gnn_rows = [
        {"model": "GNN", "trial_runtime_seconds": float(final_fit_seconds), "split": "train+test"}
    ]
    gnn_df = pd.DataFrame(gnn_rows)

    final_row = {k: np.nan for k in gnn_df.columns}
    final_row["model"] = "FINAL"
    final_row["runtime/final_fit_seconds"] = float(final_fit_seconds)
    final_row["runtime/total_seconds"] = float(total_seconds)

    gnn_out_df = pd.concat([gnn_df, pd.DataFrame([final_row])], ignore_index=True)
    gnn_out_path = f"{out_prefix}_gnn_runtime_with_total.csv"
    gnn_out_df.to_csv(gnn_out_path, index=False)
    logger.info(f"GNN runtime CSV saved: {gnn_out_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
