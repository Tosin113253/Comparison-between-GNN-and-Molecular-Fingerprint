import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class GNNModel(nn.Module):
    def __init__(self, input_dim, experimental_input_dim, gcn_hidden_dim=64, experimental_hidden_dim=64, combined_hidden_dim=128):
        super(GNNModel, self).__init__()
        # GCN branch
        self.conv1 = GCNConv(input_dim, gcn_hidden_dim)
        self.bn1 = nn.BatchNorm1d(gcn_hidden_dim)
        self.conv2 = GCNConv(gcn_hidden_dim, gcn_hidden_dim)
        self.bn2 = nn.BatchNorm1d(gcn_hidden_dim)
        self.conv3 = GCNConv(gcn_hidden_dim, gcn_hidden_dim)
        self.bn3 = nn.BatchNorm1d(gcn_hidden_dim)

        # Experimental features branch
        self.experiment_fc1 = nn.Linear(experimental_input_dim, experimental_hidden_dim)
        self.experiment_fc2 = nn.Linear(experimental_hidden_dim, experimental_hidden_dim)

        # Combined layers
        self.fc1 = nn.Linear(gcn_hidden_dim + experimental_hidden_dim, combined_hidden_dim)
        self.fc2 = nn.Linear(combined_hidden_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, data, experimental_feat):
        # GCN branch
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x, edge_index)
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
