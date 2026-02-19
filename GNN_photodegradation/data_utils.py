from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import torch
from torch_geometric.data import Data

def get_atom_features(mol):
    try:      
        atom_features = []
        for atom in mol.GetAtoms():    
            features = [
                atom.GetAtomicNum(),
                atom.GetMass(),
                atom.GetDegree(),
                int(atom.GetIsAromatic()),
                int(atom.IsInRing()),
                atom.GetTotalValence(),
                atom.GetNumRadicalElectrons(),
                atom.GetTotalNumHs()
                ]
            
            #One-hot encoding for Formal Charge
            formal_charge = [0]*7
            charge = atom.GetFormalCharge()
            charge_map = [-3, -2, -1, 0, 1, 2, 3]
            if charge in charge_map:
                formal_charge[charge_map[charge]]=1
                
            #One-hot encoding for Hybridization   
            hybridization = [0]*7
            hybrid = atom.GetHybridization()
            hybrid_map = {
                Chem.rdchem.HybridizationType.S:0,
                Chem.rdchem.HybridizationType.SP:1,
                Chem.rdchem.HybridizationType.SP2:2,
                Chem.rdchem.HybridizationType.SP3:3,
                Chem.rdchem.HybridizationType.SP2D:4,
                Chem.rdchem.HybridizationType.SP3D:5,
                Chem.rdchem.HybridizationType.SP3D2:6
                }
            if hybrid in hybrid_map:
                hybridization[hybrid_map[hybrid]] = 1
            atom_features.append(features + formal_charge + hybridization)
        
    except Exception as e:
        print(f"the error is :{e}")
        
    return atom_features

def compute_bond_legth(mol):
    try:
        conf = mol.GetConformer()
        bond_lengths = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            pos_i = np.array(conf.GetAtomPosition(i))
            pos_j = np.array(conf.GetAtomPosition(j))
            bond_length = np.linalg.norm(pos_i - pos_j)
            bond_lengths.append(bond_length)
    except Exception as e:
        print(f' error in computing bond legth: {e}')
        
    return bond_lengths
    

def get_bond_features(bond, bond_length):
    try:
        bond_type = bond.GetBondType()
        if bond_type == Chem.rdchem.BondType.SINGLE:
            bond_type_feat = [1, 0, 0]
        elif bond_type == Chem.rdchem.BondType.DOUBLE:
            bond_type_feat = [0, 1, 0]
        elif bond_type == Chem.rdchem.BondType.TRIPLE:
            bond_type_feat = [0, 0, 1]
        else:
            bond_type_feat = [0, 0, 0]
        # Bond stereochemistry (cis/trans) 
        stereo = bond.GetStereo()
        stereo_feat = [1, 0] if stereo == Chem.rdchem.BondStereo.STEREOZ else [0, 1] if stereo == Chem.rdchem.BondStereo.STEREOE else [0, 0] 
        # Bond conjugation
        conjugation_feat = [1] if bond.GetIsConjugated() else [0]
                
        #Combine all features
        bond_features = bond_type_feat + stereo_feat + conjugation_feat + [bond_length]
        
        return bond_features
    except Exception as e:
        print(f'Error in getting bond features : {e}')

def smiles_to_graph(smiles, logger=None):
    """
    Converts a SMILES string to a PyTorch Geometric Data object.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
            
        mol = Chem.AddHs(mol)
        # Compute 3D coordinates
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)
        
        # Node features: Atomic number
        atom_features = get_atom_features(mol)
        # node_feature = atom_features(mol)
        x = torch.tensor(atom_features, dtype=torch.float)
        
        # Edge indices and edge attributes
        edge_index = []
        edge_attr = []
        bond_lengths = compute_bond_legth(mol)
        for bond , bond_length in zip(mol.GetBonds(), bond_lengths):
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index += [[i, j], [j, i]]
            bond_features = get_bond_features(bond, bond_length)
            edge_attr += [bond_features, bond_features]
        
        if len(edge_attr) == 0:
            # Handle molecules with no bonds
            edge_attr = torch.zeros((0, 3), dtype=torch.float)
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # Create Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data
    except Exception as e:
        if logger:
            logger.error(f"Error converting SMILES to graph: {e}")
