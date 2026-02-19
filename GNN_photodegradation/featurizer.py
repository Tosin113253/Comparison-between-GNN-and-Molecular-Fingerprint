import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import Data, Batch
from GNN_photodegradation.data_utils import smiles_to_graph
from GNN_photodegradation.get_logger import get_logger
logger = get_logger()

class Create_Dataset(Dataset):
    def __init__(self, dataframe, numerical_features, scaler=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.smiles = self.dataframe['Smile']
        self.experimental_feats = self.dataframe[numerical_features].values.astype(np.float32)
        self.targets = self.dataframe['logk'].values.astype(np.float32)
        # Standardize numerical features
        if scaler is None:
            self.scaler = StandardScaler()
            self.experimental_feats = self.scaler.fit_transform(self.experimental_feats)
        else:
            self.scaler = scaler
            self.experimental_feats = self.scaler.transform(self.experimental_feats)
        
        # Convert to tensors
        self.experimental_feats = torch.tensor(self.experimental_feats, dtype=torch.float)
        self.targets = torch.tensor(self.targets, dtype=torch.float).unsqueeze(1)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        smiles = self.smiles.iloc[idx]
        graph = smiles_to_graph(smiles, logger)
        if graph is None:
            # Return a dummy graph if conversion fails
            graph = Data(x=torch.zeros((1, 22)), edge_index=torch.empty((2, 0), dtype=torch.long), edge_attr=torch.zeros((0, 6)))
        # Ensure the graph has a 'batch' attribute for pooling
        # Note: 'batch' is handled during batching, not individual samples
        experimental_feat = self.experimental_feats[idx]
        target = self.targets[idx]
        return graph, experimental_feat, target 
# ----------------------------- Collate Function ----------------------------- #
def collate_fn(batch):
    """
    Custom collate function to handle batches of (graph, experimental_feat, target).
    """
    graphs, experimental_feats, targets = zip(*batch)
    batch_graph = Batch.from_data_list(graphs)
    experimental_feats = torch.stack(experimental_feats)
    targets = torch.stack(targets)
    return batch_graph, experimental_feats, targets
