import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool


class StrongGNNModel(nn.Module):
    """
    Stronger (scaffold-split friendly) molecular GNN:
    - Edge-aware message passing (GINEConv)
    - Residual connections
    - BatchNorm + Dropout
    - Better readout: concat(mean_pool, max_pool)
    - Slightly deeper (default 5 layers)
    - Robust to edge_attr being either:
        (A) categorical (LongTensor with shape [E,2] = bond_type, bond_dir)
        (B) continuous float edge features (FloatTensor with shape [E, d])
        (C) missing edge_attr (will auto-create zeros)

    Forward signature matches your pipeline:
        out, graph_emb, combined = model(data, experimental_feat)
    """

    def __init__(
        self,
        node_input_dim: int,
        experimental_input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 5,
        experimental_hidden_dim: int = 128,
        combined_hidden_dim: int = 256,
        dropout: float = 0.2,
        # If you have categorical bond_type and bond_dir:
        num_bond_type: int = 6,       # safe default (e.g., single/double/triple/aromatic/self-loop/other)
        num_bond_dir: int = 4,        # none/endup/enddown/other
        edge_emb_dim: int = 64,       # edge embedding dim if categorical
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # --- Node projection (handles any numeric node features you already have) ---
        self.node_proj = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # --- Edge handling: categorical or continuous ---
        self.num_bond_type = num_bond_type
        self.num_bond_dir = num_bond_dir
        self.edge_emb_dim = edge_emb_dim

        # categorical edge embeddings (used only when edge_attr is Long [E,2])
        self.edge_type_emb = nn.Embedding(num_bond_type, edge_emb_dim)
        self.edge_dir_emb = nn.Embedding(num_bond_dir, edge_emb_dim)

        # continuous edge projection (used only when edge_attr is Float [E,d])
        self.edge_cont_proj = nn.Linear(edge_emb_dim, edge_emb_dim)

        # --- GINEConv stack (edge-aware) ---
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            conv = GINEConv(nn=mlp, edge_dim=edge_emb_dim)
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # --- Readout: mean + max pooling ---
        graph_readout_dim = hidden_dim * 2  # mean + max

        # --- Experimental branch ---
        self.exp_mlp = nn.Sequential(
            nn.Linear(experimental_input_dim, experimental_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(experimental_hidden_dim, experimental_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # --- Combined head ---
        self.head = nn.Sequential(
            nn.Linear(graph_readout_dim + experimental_hidden_dim, combined_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(combined_hidden_dim, combined_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(combined_hidden_dim // 2, 1),
        )

        # init
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def _build_edge_features(self, data):
        """
        Returns edge_attr_emb: FloatTensor [E, edge_emb_dim]
        Handles:
          - categorical edges: Long [E,2]
          - continuous edges: Float [E,d]
          - missing edges: None -> zeros
        """
        edge_index = data.edge_index
        E = edge_index.size(1)
        device = edge_index.device

        # Case 1: no edge_attr
        if not hasattr(data, "edge_attr") or data.edge_attr is None:
            return torch.zeros((E, self.edge_emb_dim), device=device, dtype=torch.float)

        edge_attr = data.edge_attr

        # Case 2: categorical edge_attr [E,2] long -> embeddings
        if edge_attr.dtype in (torch.int64, torch.int32, torch.int16, torch.int8) and edge_attr.dim() == 2 and edge_attr.size(1) >= 2:
            bond_type = edge_attr[:, 0].clamp(min=0, max=self.num_bond_type - 1)
            bond_dir = edge_attr[:, 1].clamp(min=0, max=self.num_bond_dir - 1)
            emb = self.edge_type_emb(bond_type) + self.edge_dir_emb(bond_dir)
            return emb.float()

        # Case 3: continuous edge_attr -> project to edge_emb_dim
        # If its dimension isn't edge_emb_dim, we pad/truncate then project.
        edge_attr = edge_attr.float()
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)

        d = edge_attr.size(1)
        if d < self.edge_emb_dim:
            pad = torch.zeros((edge_attr.size(0), self.edge_emb_dim - d), device=edge_attr.device, dtype=edge_attr.dtype)
            edge_attr = torch.cat([edge_attr, pad], dim=1)
        elif d > self.edge_emb_dim:
            edge_attr = edge_attr[:, : self.edge_emb_dim]

        return self.edge_cont_proj(edge_attr)

    def forward(self, data, experimental_feat):
        # --- Node features ---
        x = data.x
        if x.dtype != torch.float:
            x = x.float()
        h = self.node_proj(x)

        # --- Edge features for GINE ---
        edge_attr_emb = self._build_edge_features(data)

        # --- Message passing with residuals ---
        for conv, bn in zip(self.convs, self.bns):
            h_in = h
            h = conv(h, data.edge_index, edge_attr_emb)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + h_in  # residual

        # --- Readout (mean + max) ---
        h_mean = global_mean_pool(h, data.batch)
        h_max = global_max_pool(h, data.batch)
        graph_emb = torch.cat([h_mean, h_max], dim=1)  # [B, 2*hidden_dim]

        # --- Experimental branch ---
        exp = experimental_feat
        if exp.dtype != torch.float:
            exp = exp.float()
        exp_emb = self.exp_mlp(exp)

        # --- Combine and predict ---
        combined = torch.cat([graph_emb, exp_emb], dim=1)
        out = self.head(combined)

        return out, graph_emb, combined
