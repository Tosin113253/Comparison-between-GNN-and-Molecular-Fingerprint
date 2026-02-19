import pandas as pd
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from GNN_photodegradation.get_logger import get_logger
logger = get_logger()

def collect_predictions(loader, model, device, criterion):
    """
    Collects predictions, targets, and features from a data loader.
    """
    losses = []
    predictions = []
    targets_list = []
    experimental_feats = []
    graph_feats = []  # To store the graph features output from the model
    combined_feats = []
    
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for graphs, exp_feats, tgt in loader:
            graphs = graphs.to(device)
            exp_feats = exp_feats.to(device)
            tgt = tgt.to(device)

            outputs, graph_feat, combined = model(graphs, exp_feats)  # Assuming model outputs a tuple
            loss = criterion(outputs, tgt)
            losses.append(loss.item())
            predictions.append(outputs.cpu().numpy())
            targets_list.append(tgt.cpu().numpy())
            experimental_feats.append(exp_feats.cpu().numpy())
            graph_feats.append(graph_feat.cpu().numpy())  # Save the graph features from the model
            combined_feats.append(combined.numpy())

    avg_loss = np.mean(losses)
    predictions = np.vstack(predictions)
    targets = np.vstack(targets_list)
    experimental_feats = np.vstack(experimental_feats)
    graph_feats = np.vstack(graph_feats)  # Stack graph features from all batches
    r2 = r2_score(targets, predictions)
    combined_feats = np.vstack(combined_feats)
    #logger.info(f" - Loss: {avg_loss:.4f} - R² Score: {r2:.4f}")
    return predictions, targets, experimental_feats, graph_feats, combined_feats
    logger.info(f" - Loss: {avg_loss:.4f} - R² Score: {r2:.4f}")  

# ----------------------------- Regression Statistics ----------------------------- #
def compute_regression_stats(actual, predicted, num_iterations=1000):
    """
    Computes the slope and intercept statistics using bootstrapping.
    """
    slopes = []
    intercepts = []
    for _ in range(num_iterations):
        indices = np.random.randint(0, len(actual), len(actual))
        sample_actual = actual[indices].reshape(-1, 1)
        sample_predicted = predicted[indices].reshape(-1, 1)
        reg = LinearRegression().fit(sample_actual, sample_predicted)
        slopes.append(reg.coef_[0][0])
        intercepts.append(reg.intercept_[0])
    slope_mean = np.mean(slopes)
    intercept_mean = np.mean(intercepts)
    slope_sd = np.std(slopes)
    intercept_sd = np.std(intercepts)
    
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    prediction_results = {'MSE': mse,
                          'RMSE': rmse,
                          'MAE': mae,
                          'r2': r2}
    return slope_mean, intercept_mean, slope_sd, intercept_sd, prediction_results
