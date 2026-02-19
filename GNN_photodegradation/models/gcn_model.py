# gcn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

class StrongGCN(nn.Module):
    def __init__(self, node_input_dim, experimental_input_dim, hidden_dim=128, dropout=0.2):
        super().__init__()

        self.conv1 = GCNConv(node_input_dim, hidden_dim, normalize=True)
        self.conv2 = GCNConv(hidden_dim, hidden_dim, normalize=True)
        self.conv3 = GCNConv(hidden_dim, hidden_dim, normalize=True)
        self.conv4 = GCNConv(hidden_dim, hidden_dim, normalize=True)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)

        self.exp_mlp = nn.Sequential(
            nn.Linear(experimental_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        self.dropout = dropout

    def forward(self, data, experimental_feat):
        x, edge_index = data.x.float(), data.edge_index

        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, self.dropout, self.training)

        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, self.dropout, self.training)

        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x = F.dropout(x, self.dropout, self.training)

        x = F.relu(self.bn4(self.conv4(x, edge_index)))
        x = F.dropout(x, self.dropout, self.training)

        graph_emb = torch.cat(
            [global_mean_pool(x, data.batch), global_max_pool(x, data.batch)],
            dim=1,
        )

        exp_emb = self.exp_mlp(experimental_feat.float())

        out = self.fc(torch.cat([graph_emb, exp_emb], dim=1))
        return out, graph_emb, exp_emb
