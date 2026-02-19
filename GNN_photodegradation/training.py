# training.py (complete, copy-paste) — MINIMAL CHANGES ONLY
# ONLY changes:
# 1) Random split -> Scaffold split
# 2) Runtime reporting added
#
# Everything else is exactly your original logic:
# - Same features, same baseline + SHAP, same GNN training pipeline, same plots, same dataloaders, same early stopping.

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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

import shap
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import ReduceLROnPlateau

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

# ----- NEW (for scaffold split only) -----
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
# ----------------------------------------

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

out_prefix = "GCN"  # used in filenames


def _safe_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MSE": float(mse), "RMSE": rmse, "MAE": float(mae), "r2": float(r2)}


def run_experimental_baselines(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    out_prefix: str,
    make_shap: bool = True,
):
    """
    Baseline on tabular features ONLY (experimental + OrganicContaminant_TE),
    so SHAP shows only meaningful features (no Graph_*).
    Produces:
      - {out_prefix}_metrics.csv
      - {out_prefix}_feature_importance.csv (RF)
      - {out_prefix}_shap_beeswarm.png (optional)
    """
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_val = val_df[feature_cols].values
    y_val = val_df[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

    results = []

    # Ridge (scaled)
    ridge = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=SEED)),
        ]
    )
    ridge.fit(X_train, y_train)
    results.append({"model": "Ridge", "split": "train", **_safe_metrics(y_train, ridge.predict(X_train))})
    results.append({"model": "Ridge", "split": "val", **_safe_metrics(y_val, ridge.predict(X_val))})
    results.append({"model": "Ridge", "split": "test", **_safe_metrics(y_test, ridge.predict(X_test))})

    # RandomForest (no scaling needed)
    rf = RandomForestRegressor(
        n_estimators=500,
        random_state=SEED,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    rf.fit(X_train, y_train)
    results.append({"model": "RandomForest", "split": "train", **_safe_metrics(y_train, rf.predict(X_train))})
    results.append({"model": "RandomForest", "split": "val", **_safe_metrics(y_val, rf.predict(X_val))})
    results.append({"model": "RandomForest", "split": "test", **_safe_metrics(y_test, rf.predict(X_test))})

    metrics_df = pd.DataFrame(results)
    metrics_path = f"{out_prefix}_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # Feature importance (RF)
    fi = pd.DataFrame(
        {"Feature": feature_cols, "Importance": rf.feature_importances_.astype(float)}
    ).sort_values("Importance", ascending=False)
    fi_path = f"{out_prefix}_feature_importance.csv"
    fi.to_csv(fi_path, index=False)

    # SHAP beeswarm (RF)
    shap_path = None
    if make_shap:
        try:
            # sample for speed
            n_bg = min(300, X_train.shape[0])
            n_plot = min(500, X_train.shape[0])
            rng = np.random.default_rng(SEED)
            bg_idx = rng.choice(X_train.shape[0], size=n_bg, replace=False)
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

    return metrics_path, fi_path, shap_path


# ----------------------- NEW: Scaffold split ONLY -----------------------
def _murcko_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def scaffold_split_df(df: pd.DataFrame, smiles_col: str = "Smile", frac_train=0.75, frac_val=0.125, frac_test=0.125):
    """
    Bemis–Murcko scaffold split.
    This function ONLY affects splitting (as requested).
    """
    df = df.copy()
    df["_scaffold"] = df[smiles_col].apply(_murcko_scaffold)

    # drop invalid SMILES
    df = df[df["_scaffold"] != ""].copy()

    # group indices by scaffold
    scaffold_to_indices = {}
    for idx, scaf in zip(df.index.tolist(), df["_scaffold"].tolist()):
        scaffold_to_indices.setdefault(scaf, []).append(idx)

    # sort scaffold groups by size (desc)
    groups = sorted(scaffold_to_indices.values(), key=lambda g: len(g), reverse=True)

    n_total = len(df)
    n_train = int(round(frac_train * n_total))
    n_val = int(round(frac_val * n_total))

    train_idx, val_idx, test_idx = [], [], []
    for g in groups:
        if len(train_idx) + len(g) <= n_train:
            train_idx.extend(g)
        elif len(val_idx) + len(g) <= n_val:
            val_idx.extend(g)
        else:
            test_idx.extend(g)

    train_df = df.loc[train_idx].drop(columns=["_scaffold"])
    val_df = df.loc[val_idx].drop(columns=["_scaffold"])
    test_df = df.loc[test_idx].drop(columns=["_scaffold"])

    return train_df, val_df, test_df
# ----------------------------------------------------------------------


def main():
    dataset_path = DATA_path
    num_epochs = NUM_epochs

    # ----------------------- Load dataset -----------------------
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file not found at {dataset_path}")
        return

    try:
        df = pd.read_excel(dataset_path)
        logger.info(f"Dataset loaded successfully with {len(df)} records.")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # ----------------------- Column checks ----------------------
    required_columns = {
        "Smile",
        "logk",
        "Intensity",
        "Wavelength",
        "Temp",
        "Dosage",
        "InitialC",
        "Humid",
        "Reactor",
    }
    if not required_columns.issubset(df.columns):
        missing = sorted(list(required_columns - set(df.columns)))
        logger.error(f"Dataset missing required columns: {missing}")
        logger.error(f"Columns found: {list(df.columns)}")
        return

    # ----------------------- Coerce numeric ----------------------
    df["logk"] = pd.to_numeric(df["logk"], errors="coerce")

    numerical_features = [
        "Intensity",
        "Wavelength",
        "Temp",
        "Dosage",
        "InitialC",
        "Humid",
        "Reactor",
    ]
    for col in numerical_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with NaNs
    before = len(df)
    df = df.dropna(subset=["Smile", "logk"] + numerical_features).copy()
    after = len(df)
    if after < before:
        logger.info(f"Dropped {before - after} rows due to NaNs in required columns.")

    # Ensure float32
    df[numerical_features] = df[numerical_features].astype(np.float32)
    df["logk"] = df["logk"].astype(np.float32)

    # ----------------------- Split dataset (CHANGED to scaffold) -----------------------
    t_split0 = time.perf_counter()
    train_df, val_df, test_df = scaffold_split_df(df, smiles_col="Smile", frac_train=0.75, frac_val=0.125, frac_test=0.125)
    t_split1 = time.perf_counter()

    # For plot labels (1-based)
    train_idx = train_df.index.to_numpy() + 1
    val_idx = val_df.index.to_numpy() + 1
    test_idx = test_df.index.to_numpy() + 1

    logger.info(f"Scaffold split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    logger.info(f"Split runtime (sec): {t_split1 - t_split0:.2f}")

    # ----------------------- Organic Contaminant TE -----------------------
    # Leakage-safe target encoding: learned on TRAIN only
    global_mean = float(train_df["logk"].mean())
    te_map = train_df.groupby("Smile")["logk"].mean().to_dict()

    for dfx in (train_df, val_df, test_df):
        dfx["OrganicContaminant_TE"] = (
            dfx["Smile"].map(te_map).fillna(global_mean).astype(np.float32)
        )

    logger.info("Created OrganicContaminant_TE via leakage-safe target encoding.")
    # ---------------------------------------------------------------------

    # ------------------- Baseline (tabular) + SHAP -----------------------
    baseline_feature_cols = numerical_features + ["OrganicContaminant_TE"]

    t_base0 = time.perf_counter()
    metrics_path, fi_path, shap_path = run_experimental_baselines(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=baseline_feature_cols,
        target_col="logk",
        out_prefix="baseline_tabular",
        make_shap=True,
    )
    t_base1 = time.perf_counter()

    logger.info(f"Baseline metrics saved: {metrics_path}")
    logger.info(f"Baseline feature importance saved: {fi_path}")
    if shap_path:
        logger.info(f"Baseline SHAP beeswarm saved: {shap_path}")
    logger.info(f"Baseline runtime (sec): {t_base1 - t_base0:.2f}")
    # ---------------------------------------------------------------------

    # ----------------------- Create GNN datasets -------------------------
    # IMPORTANT: GNN pipeline remains unchanged (graphs + standardized experimental feats)
    train_dataset = Create_Dataset(train_df, numerical_features)
    scaler = train_dataset.scaler
    val_dataset = Create_Dataset(val_df, numerical_features, scaler=scaler)
    test_dataset = Create_Dataset(test_df, numerical_features, scaler=scaler)
    logger.info("Datasets created and features standardized (GNN).")

    # ----------------------- DataLoaders -------------------------
    # (unchanged) kept your shuffle=False exactly
    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn
    )
    logger.info("Data loaders initialized (GNN).")

    # ----------------------- Model init --------------------------
    experimental_input_dim = train_dataset.experimental_feats.shape[1]
    model = GNNModel(22, experimental_input_dim=experimental_input_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info(f"Model initialized and moved to {device}.")

    # ----------------------- Train setup -------------------------
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)
    logger.info("Loss function, optimizer, and scheduler defined.")

    # ----------------------- Training loop -----------------------
    PATIENCE = 50
    best_val_loss = float("inf")
    patience_counter = 0

    t_train0 = time.perf_counter()

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_losses = []

        for graphs, exp_feats, targets in train_loader:
            graphs = graphs.to(device)
            exp_feats = exp_feats.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs, _, _ = model(graphs, exp_feats)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        model.eval()
        val_losses = []
        with torch.no_grad():
            for graphs, exp_feats, targets in val_loader:
                graphs = graphs.to(device)
                exp_feats = exp_feats.to(device)
                targets = targets.to(device)

                outputs, _, _ = model(graphs, exp_feats)
                loss = criterion(outputs, targets)
                val_losses.append(loss.item())

        avg_val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

        scheduler.step(avg_val_loss)
        last_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else None

        logger.info(
            f"Epoch {epoch}/{num_epochs} - LR: {last_lr} "
            f"- Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    t_train1 = time.perf_counter()
    logger.info(f"GNN training runtime (sec): {t_train1 - t_train0:.2f}")

    # ----------------------- Evaluation --------------------------
    t_eval0 = time.perf_counter()

    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()
    logger.info("Best model loaded for evaluation.")

    train_pred, train_tgt, train_feats, train_graph_feats, _ = collect_predictions(
        train_loader, model, device, criterion
    )
    val_pred, val_tgt, val_feats, val_graph_feats, _ = collect_predictions(
        val_loader, model, device, criterion
    )
    test_pred, test_tgt, test_feats, test_graph_feats, _ = collect_predictions(
        test_loader, model, device, criterion
    )

    # ----------------------- GNN Plots + metrics ---------------------
    results = []
    dsname = []

    for pred, tgt, name, label in zip(
        [train_pred, val_pred, test_pred],
        [train_tgt, val_tgt, test_tgt],
        ["Training", "Validation", "Test"],
        [train_idx, val_idx, test_idx],
    ):
        slope, intercept, slope_sd, intercept_sd, result = compute_regression_stats(tgt, pred)
        results.append(result)
        dsname.append(name)
        plot_calculated_vs_experimental(pred.flatten(), tgt.flatten(), name, label, slope, intercept)

    results_df = pd.DataFrame(results, index=dsname)
    results_df.to_excel(f"{out_prefix}_Regression_results.xlsx")

    dataset_dict = {}
    for name, exp_feats, graph_feats in zip(
        ["train", "val", "test"],
        [train_feats, val_feats, test_feats],
        [train_graph_feats, val_graph_feats, test_graph_feats],
    ):
        dataset_dict[name] = np.hstack((exp_feats, graph_feats))
        if dataset_dict[name].ndim == 1:
            dataset_dict[name] = dataset_dict[name].reshape(-1, 1)

    plot_williams(
        dataset_dict["train"],
        dataset_dict["val"],
        dataset_dict["test"],
        train_pred,
        val_pred,
        test_pred,
        train_tgt,
        val_tgt,
        test_tgt,
        train_idx,
        val_idx,
        test_idx,
    )

    combined_exp_feats = np.vstack((train_feats, val_feats, test_feats))
    combined_graph_feats = np.vstack((train_graph_feats, val_graph_feats, test_graph_feats))
    combined_targets = np.vstack((train_tgt, val_tgt, test_tgt))

    plot_pca(
        combined_exp_feats, combined_graph_feats, combined_targets.flatten(),
        "Combined", "2D PCA Plot", dimensions=2
    )
    plot_pca(
        combined_exp_feats, combined_graph_feats, combined_targets.flatten(),
        "Combined", "3D PCA Plot", dimensions=3
    )

    plot_umap(
        combined_exp_feats, combined_graph_feats, combined_targets.flatten(),
        "Combined", title="2D UMAP Plot", dimensions=2
    )
    plot_umap(
        combined_exp_feats, combined_graph_feats, combined_targets.flatten(),
        "Combined", title="3D UMAP Plot", dimensions=3
    )

    t_eval1 = time.perf_counter()
    logger.info(f"GNN eval+plots runtime (sec): {t_eval1 - t_eval0:.2f}")

    logger.info("All plots have been generated and saved.")

    # ----------------------- Runtime summary file (NEW) -----------------------
    runtime_df = pd.DataFrame([{
        "split_time_sec": float(t_split1 - t_split0),
        "baseline_time_sec": float(t_base1 - t_base0),
        "gnn_train_time_sec": float(t_train1 - t_train0),
        "gnn_eval_time_sec": float(t_eval1 - t_eval0),
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "test_size": int(len(test_df)),
    }])
    runtime_df.to_csv(f"{out_prefix}_runtime_summary.csv", index=False)
    logger.info(f"Runtime summary saved to: {out_prefix}_runtime_summary.csv")
    # ------------------------------------------------------------------------


if __name__ == "__main__":
    main()
