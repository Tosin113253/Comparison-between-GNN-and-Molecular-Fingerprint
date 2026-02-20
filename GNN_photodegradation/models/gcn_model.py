import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool


class StrongGNNModel(nn.Module):
    """
    Scaffold-split friendly GNN with robust input handling:
    - If data.x is categorical ([N,2] long): uses atom/chirality embeddings (GOOD).
    - If data.x is float ([N,F] float): uses an MLP projection (also OK).
    - Edge-aware message passing (GINEConv)
    - Residual connections + BatchNorm + Dropout
    - Readout: concat(mean_pool, max_pool)
    - Forward returns: out, graph_emb, combined
    """

    def __init__(
        self,
        node_input_dim: int,                 # used only if node features are float
        experimental_input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 5,
        experimental_hidden_dim: int = 256,
        combined_hidden_dim: int = 512,
        dropout: float = 0.25,

        # If categorical node features exist (common in molecular graphs)
        num_atom_type: int = 120,            # safe default
        num_chirality_tag: int = 4,          # safe default

        # If categorical edge features exist (bond type/dir)
        num_bond_type: int = 6,
        num_bond_dir: int = 4,
        edge_emb_dim: int = 128,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # ---- Node encoders (two modes) ----
        # Mode A: categorical atom_type + chirality (data.x is LongTensor [N,2])
        self.atom_emb = nn.Embedding(num_atom_type, hidden_dim)
        self.chiral_emb = nn.Embedding(num_chirality_tag, hidden_dim)

        # Mode B: float node features (data.x is FloatTensor [N,F])
        self.node_float_proj = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ---- Edge encoders (two modes) ----
        self.edge_emb_dim = edge_emb_dim

        # categorical edges: LongTensor [E,2] (bond_type, bond_dir)
        self.edge_type_emb = nn.Embedding(num_bond_type, edge_emb_dim)
        self.edge_dir_emb = nn.Embedding(num_bond_dir, edge_emb_dim)

        # continuous edges: FloatTensor [E,d] -> project to edge_emb_dim
        self.edge_cont_proj = nn.Sequential(
            nn.Linear(edge_emb_dim, edge_emb_dim),
            nn.ReLU(),
            nn.Linear(edge_emb_dim, edge_emb_dim),
        )

        # ---- GINE blocks ----
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(nn=mlp, edge_dim=edge_emb_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # ---- Experimental branch ----
        self.exp_mlp = nn.Sequential(
            nn.Linear(experimental_input_dim, experimental_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(experimental_hidden_dim, experimental_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ---- Head ----
        graph_readout_dim = hidden_dim * 2  # mean + max
        self.head = nn.Sequential(
            nn.Linear(graph_readout_dim + experimental_hidden_dim, combined_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(combined_hidden_dim, combined_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(combined_hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def _encode_nodes(self, x):
        """
        Returns node embeddings h: FloatTensor [N, hidden_dim]
        Handles:
          - categorical x: Long [N,2] -> atom/chiral embeddings
          - float x: Float [N,F] -> MLP projection
        """
        # categorical case
        if x.dtype in (torch.int64, torch.int32, torch.int16, torch.int8) and x.dim() == 2 and x.size(1) >= 2:
            atom_type = x[:, 0].clamp(min=0, max=self.atom_emb.num_embeddings - 1)
            chiral = x[:, 1].clamp(min=0, max=self.chiral_emb.num_embeddings - 1)
            return self.atom_emb(atom_type) + self.chiral_emb(chiral)

        # float case
        if x.dtype != torch.float:
            x = x.float()
        return self.node_float_proj(x)

    def _encode_edges(self, data):
        """
        Returns edge_attr_emb: FloatTensor [E, edge_emb_dim]
        Handles:
          - categorical edges: Long [E,2]
          - continuous edges: Float [E,d]
          - missing edge_attr: zeros
        """
        edge_index = data.edge_index
        E = edge_index.size(1)
        device = edge_index.device

        if not hasattr(data, "edge_attr") or data.edge_attr is None:
            return torch.zeros((E, self.edge_emb_dim), device=device, dtype=torch.float)

        edge_attr = data.edge_attr

        # categorical edges
        if edge_attr.dtype in (torch.int64, torch.int32, torch.int16, torch.int8) and edge_attr.dim() == 2 and edge_attr.size(1) >= 2:
            bond_type = edge_attr[:, 0].clamp(min=0, max=self.edge_type_emb.num_embeddings - 1)
            bond_dir = edge_attr[:, 1].clamp(min=0, max=self.edge_dir_emb.num_embeddings - 1)
            return (self.edge_type_emb(bond_type) + self.edge_dir_emb(bond_dir)).float()

        # continuous edges -> pad/truncate to edge_emb_dim then project
        edge_attr = edge_attr.float()
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)

        d = edge_attr.size(1)
        if d < self.edge_emb_dim:
            pad = torch.zeros((edge_attr.size(0), self.edge_emb_dim - d), device=device, dtype=edge_attr.dtype)
            edge_attr = torch.cat([edge_attr, pad], dim=1)
        elif d > self.edge_emb_dim:
            edge_attr = edge_attr[:, : self.edge_emb_dim]

        return self.edge_cont_proj(edge_attr)

    def forward(self, data, experimental_feat):
        # nodes
        h = self._encode_nodes(data.x)

        # edges
        edge_attr_emb = self._encode_edges(data)

        # message passing (residual)
        for conv, bn in zip(self.convs, self.bns):
            h_in = h
            h = conv(h, data.edge_index, edge_attr_emb)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + h_in

        # pooling
        h_mean = global_mean_pool(h, data.batch)
        h_max = global_max_pool(h, data.batch)
        graph_emb = torch.cat([h_mean, h_max], dim=1)

        # experimental branch
        exp = experimental_feat.float()
        exp_emb = self.exp_mlp(exp)

        # combine + predict
        combined = torch.cat([graph_emb, exp_emb], dim=1)
        out = self.head(combined)

        return out, graph_emb, combined
