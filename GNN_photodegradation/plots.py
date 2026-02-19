import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from scipy import stats


def plot_calculated_vs_experimental(predicted, actual, dataset_name, label, slope, intercept):
    """
    Generates and displays a scatter plot comparing calculated and experimental mlogk values
    with regression line and SD of slope and intercept.
    """
    # Linear regression for confidence intervals
    reg = LinearRegression().fit(actual.reshape(-1, 1), predicted)
    y_fit = reg.predict(actual.reshape(-1, 1)).ravel()  # Flatten to 1D
    
    residuals = predicted - actual
    std_res = np.std(residuals)
    n = len(actual)
    standard_error = std_res * np.sqrt(1 + 1/n + (actual - np.mean(actual))**2 / np.sum((actual - np.mean(actual))**2))
    t_val = stats.t.ppf(0.975, n - 2)  # 95% confidence
    # Create upper and lower bounds for the confidence interval
    upper_bound = y_fit + t_val * standard_error  # This should be 1D
    lower_bound = y_fit - t_val * standard_error  # This should be 1D
    
    # Plotting predicted vs true values
    plt.figure(figsize=(4.5, 4.5))
    plt.rcParams['font.family'] = 'Arial'
    sns.scatterplot(x=actual, y=predicted, edgecolor='k', alpha=0.7, s =40)
    plt.plot(actual, y_fit, color='red', linewidth=2, label=f'Fit: y={slope:.2f}x + {intercept:.2f}')
    # for i, (x, y, lb, ub) in enumerate(zip(actual, predicted, lower_bound, upper_bound)): 
    #     if y>ub or y<lb: 
    #         plt.annotate(label[i], (x, y), textcoords="offset points", xytext=(5,5), ha='center', fontsize = 10)
     # Ensure lower_bound and upper_bound are 1-dimensional
    plt.fill_between(actual, lower_bound, upper_bound, color='red', alpha=0.4, label='95% Confidence Interval')
    
    plt.title(f'{dataset_name} set', fontsize = 14)
    plt.xlabel('Experimental logk', fontsize = 12)
    plt.ylabel('Predicted logk', fontsize = 12)
    plt.legend(fontsize = 10, loc = 'lower right', frameon= False)
    # plt.grid(True)
   
    # Determine the global minimum and maximum limits from both axes
    global_min = min(min(actual), min(predicted))  # Smallest value across both
    global_max = max(max(actual), max(predicted))  # Largest value across both

    # Force both axes to use the same range
    plt.xlim(global_min-0.5, global_max+0.5)
    plt.ylim(global_min-0.5, global_max+0.5)
    
    plt.tight_layout()
    plt.savefig(f'calculated_vs_experimental_{dataset_name.lower()}.png', dpi=600,bbox_inches='tight')
    plt.show()
   

def plot_pca(numerical_features, graph_features, targets, dataset_name, title, dimensions=3):
    """
    Plot PCA with combined numerical and graph features
    """
    
    combined_features = np.hstack((numerical_features, graph_features))

    # Standardize the combined features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(combined_features)
    
    # Apply PCA
    pca = PCA(n_components=dimensions)
    pca_result = pca.fit_transform(features_scaled)
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # Create plot
    if dimensions == 2:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1],
                            c=targets, cmap='viridis', alpha=0.7, s = 60)
        plt.xlabel(f'PC1 ({explained_variance[0]:.2%} variance)',fontsize = 14)
        plt.ylabel(f'PC2 ({explained_variance[1]:.2%} variance)', fontsize = 14)
    else:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2],
                           c=targets, cmap='viridis', alpha=0.7, s = 60)
        ax.set_xlabel(f'PC1 ({explained_variance[0]:.2%} variance)', fontsize = 14)
        ax.set_ylabel(f'PC2 ({explained_variance[1]:.2%} variance)', fontsize = 14)
        ax.set_zlabel(f'PC3 ({explained_variance[2]:.2%} variance)', fontsize = 14)
    
    plt.title(f'{title} - {dataset_name}\nTotal Explained Variance: {cumulative_variance[-1]:.2%}', fontsize = 16)
    plt.colorbar(scatter, label='Target Values')
    plt.tight_layout()
    plt.savefig(f'pca_{dimensions}d_{dataset_name.lower()}.png', dpi=900)
    plt.show()

def plot_umap(numerical_features, graph_features, targets, dataset_name, title, dimensions=3):
    """
    Plot UMAP with combined numerical and graph features
    """
    
    combined_features = np.hstack((numerical_features, graph_features))
    # Standardize the combined features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(combined_features)
    
    # Apply UMAP
    reducer = umap.UMAP(n_components=dimensions, random_state=42,
                       n_neighbors=15, min_dist=0.1)
    umap_result = reducer.fit_transform(features_scaled)
    
    # Create plot
    if dimensions == 2:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(umap_result[:, 0], umap_result[:, 1],
                            c=targets, cmap='viridis', alpha=0.7, s = 60)
        plt.xlabel('UMAP1', fontsize = 14)
        plt.ylabel('UMAP2', fontsize = 14)
    else:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(umap_result[:, 0], umap_result[:, 1], umap_result[:, 2],
                           c=targets, cmap='viridis', alpha=0.7, s = 60)

        ax.set_xlabel('UMAP1',fontsize = 14)
        ax.set_ylabel('UMAP2', fontsize = 14)
        ax.set_zlabel('UMAP3', fontsize = 14)
   
    plt.title(f'{title} - {dataset_name}', fontsize = 16)
    plt.colorbar(scatter, label='Target Values')
    plt.tight_layout()
    plt.savefig(f'umap_{dimensions}d_{dataset_name.lower()}.png', dpi=900)
    plt.show()    
  
def plot_williams(train, val, test, train_pred, val_pred, test_pred, train_nom, val_nom, test_nom, train_idx, val_idx, test_idx):
    """
    plot all three data sets on one williams plot
    """

    def notation(h, h_crit, std_residuals, idx):
        texts = []  # collect text objects for optional adjustment
        for i in range(len(h)):
            if h[i] > h_crit or std_residuals[i] > 3 or std_residuals[i] < -3:
                x = h[i]
                y = std_residuals[i]
                
                text = plt.annotate(
                    str(idx[i]),
                    (x, y),
                    textcoords="offset points",
                    xytext=(6, 6),
                    ha='left',
                    fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.2', edgecolor='none', facecolor='white', alpha=0.6)
                )
                texts.append(text)
        return texts

    # Standardize features
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    val_scaled = scaler.fit_transform(val)
    test_scaled = scaler.fit_transform(test)
    
    pca = PCA(n_components=0.98, svd_solver='full')
    pca.fit(train_scaled)
    train_scores = pca.fit_transform(train_scaled)
    val_scores = pca.transform(val_scaled)
    test_scores = pca.transform(test_scaled)
    #Calculate leverage values
    try:
        scores_dict = {
            'train' : train_scores,
            'val' : val_scores,
            'test' : test_scores
            }
        train_inv = np.linalg.inv(train_scores.T@train_scores)
        leverage_dict = {}
        for dset, scores in scores_dict.items():
            leverage_dict[dset] = np.diag(scores@train_inv@scores.T)
            
            
        train_lev = leverage_dict['train']
        val_lev = leverage_dict['val']
        test_lev = leverage_dict['test']
        
    except Exception as e:
        print(f'the error in leverage calculation is: {e}')
    
    #Calculate std residuals
    try:
        pred_dict = {
            'train':{'pred': train_pred, 'nom': train_nom},
            'val':{'pred':val_pred, 'nom': val_nom},
            'test':{'pred':test_pred, 'nom': test_nom}
            }

        std_residual_dict = {}  
        for name, data in pred_dict.items():
            residual = data['pred'] - data['nom']
            std_residual_dict[name] = (residual - np.mean(residual))/np.std(residual)
            
        train_std_residuals = std_residual_dict['train']
        val_std_residuals = std_residual_dict['val']
        test_std_residuals = std_residual_dict['test']
    except Exception as e:
        print(f"The error in std_residual calculation is: {e}")
        
    # Critical leverage threshold
    p = train_scores.shape[1]  # Number of variables
    n = len(train_pred)
    h_crit = 3 * p / n
    
    plt.figure(figsize=(3.25,3))
    plt.rcParams['font.family'] = 'Arial'
    colors = {
        'train': '#1f78b4',      # Navy blue
        'val': '#e66100',        # Vermilion
        'test': '#5ab4ac'        # Teal
    }
    plt.scatter(train_lev, train_std_residuals, label= 'Train', edgecolors='k', color=colors['train'], s = 20, marker='o', linewidths=0.5)
    plt.scatter(val_lev, val_std_residuals, label= 'Validation', edgecolors= 'k', color =colors['val'], s = 20, marker='s', linewidths=0.5)
    plt.scatter(test_lev, test_std_residuals, label = 'Test',edgecolors='k', color =colors['test'], s = 20, marker='D', linewidths=0.5)
    notation(train_lev, h_crit, train_std_residuals, train_idx)
    notation(val_lev, h_crit, val_std_residuals, val_idx)
    notation(test_lev, h_crit, test_std_residuals, test_idx)
    plt.axhline(y=3, color='r' , linestyle = '--', linewidth=0.8)
    plt.axhline(y=-3, color= 'r', linestyle = '--', linewidth=0.8)
    plt.axvline(x = h_crit, color= 'g', linestyle = '--', linewidth=0.8)
    plt.annotate(f' $h^*$ = {h_crit:0.2f}', (h_crit + 0.005, min(np.min(train_std_residuals), -3.5 )), fontsize = 9, color = 'black')
    plt.xlabel('Leverage', fontsize = 10)
    plt.ylabel('Standardized Residuals', fontsize = 10)
    plt.legend(fontsize = 6, loc = 'upper right', frameon = False)
    # plt.title('Williams plot', fontsize = 16)
    plt.tight_layout()
    plt.savefig('Williams_Plot_Improved.png', dpi=600, bbox_inches='tight')
    plt.show()
