import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool

class GNNModel(nn.Module):
    def __init__(self, GAT_input_dim, experimental_input_dim, GAT_hidden_dim=64, edge_dim=7, experimental_hidden_dim=64, combined_hidden_dim=128):
        super(GNNModel, self).__init__()
        # GAT branch
        self.conv1 = GATConv(GAT_input_dim, GAT_hidden_dim, heads=4, concat=True, edge_dim=edge_dim)
        self.bn1 = nn.BatchNorm1d(GAT_hidden_dim * 4)
        self.conv2 = GATConv(GAT_hidden_dim*4, GAT_hidden_dim, heads=4, concat=True, edge_dim=edge_dim)
        self.bn2 = nn.BatchNorm1d(GAT_hidden_dim*4)
        self.conv3 = GATConv(GAT_hidden_dim*4, GAT_hidden_dim, heads=1, concat=False, edge_dim=edge_dim)
        self.bn3 = nn.BatchNorm1d(GAT_hidden_dim)

        # Experimental features branch
        self.experiment_fc1 = nn.Linear(experimental_input_dim, experimental_hidden_dim)
        self.experiment_fc2 = nn.Linear(experimental_hidden_dim, experimental_hidden_dim)
        # Combined layers
        self.fc1 = nn.Linear(GAT_hidden_dim + experimental_hidden_dim, combined_hidden_dim)
        self.fc2 = nn.Linear(combined_hidden_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, data, experimental_feat):
        # GAT branch
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr = edge_attr)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x, edge_index, edge_attr = edge_attr)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x, edge_index, edge_attr = edge_attr)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.dropout(x)
        
        x = global_mean_pool(x, data.batch)  # Global pooling
        x = self.dropout(x)
        # Experimental features branch
        e = self.relu(self.experiment_fc1(experimental_feat))

        e = self.relu(self.experiment_fc2(e))
        e = self.dropout(e)
        # Combine
        combined = torch.cat([x, e], dim=1)
        combined = self.relu(self.fc1(combined))
        combined = self.dropout(combined)
        out = self.fc2(combined)
        return out, x , combined # Return both the output and the combined features