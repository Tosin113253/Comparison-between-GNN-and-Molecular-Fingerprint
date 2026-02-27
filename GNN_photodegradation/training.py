# training.py (complete, copy-paste)
# - Uses your dataset columns: Smile, logk, Intensity, Wavelength, Temp, Dosage, InitialC, Humid, Reactor
# - Runs GNN training (your existing pipeline)
# - Adds a baseline (experimental-only) + SHAP beeswarm INCLUDING Organic Contaminant via leakage-safe Target Encoding
# - Avoids "Graph_*" features in SHAP by using baseline model on tabular features only
# - No outlier removal
# - TRAIN/TEST only (no validation split)
# - Adds training time logging

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


def _format_seconds(sec: float) -> str:
    sec = float(sec)
    if sec < 60:
        return f"{sec:.2f}s"
    if sec < 3600:
        return f"{sec/60.0:.2f} min"
    return f"{sec/3600.0:.2f} hr"


def run_experimental_baselines_train_test(
    train_df: pd.DataFrame,
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
    t0 = time.perf_counter()

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
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

    elapsed = time.perf_counter() - t0
    time_path = f"{out_prefix}_timing.txt"
    with open(time_path, "w", encoding="utf-8") as f:
        f.write(f"baseline_wall_time_seconds,{elapsed:.6f}\n")
        f.write(f"baseline_wall_time_human,{_format_seconds(elapsed)}\n")

    return metrics_path, fi_path, shap_path, time_path


def main():
    total_t0 = time.perf_counter()

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

    # ----------------------- Split dataset (TRAIN/TEST only) -----------------------
    train_df, test_df, train_idx, test_idx = train_test_split(
        df, df.index, test_size=0.25, random_state=SEED
    )

    # For plot labels (1-based)
    train_idx = train_idx + 1
    test_idx = test_idx + 1

    # ----------------------- Organic Contaminant TE -----------------------
    # Leakage-safe target encoding: learned on TRAIN only
    global_mean = float(train_df["logk"].mean())
    te_map = train_df.groupby("Smile")["logk"].mean().to_dict()

    for dfx in (train_df, test_df):
        dfx["OrganicContaminant_TE"] = (
            dfx["Smile"].map(te_map).fillna(global_mean).astype(np.float32)
        )

    logger.info("Created OrganicContaminant_TE via leakage-safe target encoding.")
    # ---------------------------------------------------------------------

    # ------------------- Baseline (tabular) + SHAP -----------------------
    baseline_feature_cols = numerical_features + ["OrganicContaminant_TE"]
    b_metrics_path, b_fi_path, b_shap_path, b_time_path = run_experimental_baselines_train_test(
        train_df=train_df,
        test_df=test_df,
        feature_cols=baseline_feature_cols,
        target_col="logk",
        out_prefix="baseline_tabular",
        make_shap=True,
    )
    logger.info(f"Baseline metrics saved: {b_metrics_path}")
    logger.info(f"Baseline feature importance saved: {b_fi_path}")
    logger.info(f"Baseline timing saved: {b_time_path}")
    if b_shap_path:
        logger.info(f"Baseline SHAP beeswarm saved: {b_shap_path}")
    # ---------------------------------------------------------------------

    # ----------------------- Create GNN datasets -------------------------
    # IMPORTANT: GNN pipeline remains unchanged (graphs + standardized experimental feats)
    train_dataset = Create_Dataset(train_df, numerical_features)
    scaler = train_dataset.scaler
    test_dataset = Create_Dataset(test_df, numerical_features, scaler=scaler)
    logger.info("Datasets created and features standardized (GNN).")

    # ----------------------- DataLoaders -------------------------
    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn
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

    # No validation set: scheduler monitors TRAIN loss only (does NOT touch test)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)
    logger.info("Loss function, optimizer, and scheduler defined.")

    # ----------------------- Training loop -----------------------
    # With no validation set, early stopping uses TRAIN loss only (safe from test leakage)
    PATIENCE = 50
    best_train_loss = float("inf")
    patience_counter = 0

    gnn_t0 = time.perf_counter()

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

        scheduler.step(avg_train_loss)
        last_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else None

        logger.info(
            f"Epoch {epoch}/{num_epochs} - LR: {last_lr} "
            f"- Train Loss: {avg_train_loss:.4f}"
        )

        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            torch.save(model.state_dict(), "best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered (train-loss based).")
                break

    gnn_elapsed = time.perf_counter() - gnn_t0
    logger.info(f"GNN training wall time: {_format_seconds(gnn_elapsed)} ({gnn_elapsed:.2f}s)")

    with open(f"{out_prefix}_gnn_timing.txt", "w", encoding="utf-8") as f:
        f.write(f"gnn_wall_time_seconds,{gnn_elapsed:.6f}\n")
        f.write(f"gnn_wall_time_human,{_format_seconds(gnn_elapsed)}\n")
    # ---------------------------------------------------------------------

    # ----------------------- Evaluation (TRAIN/TEST) --------------------------
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()
    logger.info("Best model loaded for evaluation.")

    train_pred, train_tgt, train_feats, train_graph_feats, _ = collect_predictions(
        train_loader, model, device, criterion
    )
    test_pred, test_tgt, test_feats, test_graph_feats, _ = collect_predictions(
        test_loader, model, device, criterion
    )

    # ----------------------- GNN Plots + metrics (TRAIN/TEST only) ---------------------
    results = []
    dsname = []

    for pred, tgt, name, label in zip(
        [train_pred, test_pred],
        [train_tgt, test_tgt],
        ["Training", "Test"],
        [train_idx, test_idx],
    ):
        slope, intercept, slope_sd, intercept_sd, result = compute_regression_stats(tgt, pred)
        results.append(result)
        dsname.append(name)
        plot_calculated_vs_experimental(pred.flatten(), tgt.flatten(), name, label, slope, intercept)

    results_df = pd.DataFrame(results, index=dsname)
    results_df.to_excel(f"{out_prefix}_Regression_results.xlsx")

    # Some plotting utilities expect train/val/test. We will try them safely; if they require 3 splits, skip.
    try:
        dataset_dict = {
            "train": np.hstack((train_feats, train_graph_feats)),
            "test": np.hstack((test_feats, test_graph_feats)),
        }

        # If plot_williams strictly requires 3 splits, this will throw; we catch and skip.
        plot_williams(
            dataset_dict["train"],
            dataset_dict["test"],   # used as "val" placeholder internally if needed
            dataset_dict["test"],   # used as "test" placeholder internally if needed
            train_pred,
            test_pred,
            test_pred,
            train_tgt,
            test_tgt,
            test_tgt,
            train_idx,
            test_idx,
            test_idx,
        )
    except Exception as e:
        logger.warning(f"Skipping Williams plot (requires 3 splits): {e}")

    combined_exp_feats = np.vstack((train_feats, test_feats))
    combined_graph_feats = np.vstack((train_graph_feats, test_graph_feats))
    combined_targets = np.vstack((train_tgt, test_tgt))

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

    total_elapsed = time.perf_counter() - total_t0
    logger.info(f"TOTAL script wall time: {_format_seconds(total_elapsed)} ({total_elapsed:.2f}s)")
    with open(f"{out_prefix}_total_timing.txt", "w", encoding="utf-8") as f:
        f.write(f"total_wall_time_seconds,{total_elapsed:.6f}\n")
        f.write(f"total_wall_time_human,{_format_seconds(total_elapsed)}\n")

    logger.info("All plots have been generated and saved (train/test only).")


if __name__ == "__main__":
    main()
