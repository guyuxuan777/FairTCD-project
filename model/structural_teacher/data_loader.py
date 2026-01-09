import os
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  

def load_structural_adversarial_data(adj_dir: str,
                                     label_dir: str,
                                     device: torch.device):

    # 1) list and sort
    adj_files   = sorted(f for f in os.listdir(adj_dir)   if f.endswith('.npy'))
    label_files = sorted(f for f in os.listdir(label_dir) if f.endswith('.npy'))
    assert len(adj_files) == len(label_files), "adj and label not same"

    first_A = np.load(os.path.join(adj_dir, adj_files[0]))
    D = first_A.shape[0]

    graphs = []
    label_list = []

    # 3) load adj + label
    for adj_fn, lbl_fn in zip(adj_files, label_files):
        # 3.1) read adjacency matrix
        A = np.load(os.path.join(adj_dir, adj_fn))
        assert A.shape[0] == D, f"{adj_fn} shape {A.shape} no equal {D}"
        ei, ew = dense_to_sparse(torch.from_numpy(A).cpu())
        ei, ew = ei.to(device), ew.to(device)

        # 3.2) read labels
        y = torch.from_numpy(np.load(os.path.join(label_dir, lbl_fn))).long().to(device)
        if y.numel() < D:
            pad = torch.full((D - y.numel(),), -1, dtype=y.dtype, device=device)
            y = torch.cat([y, pad], dim=0)
        label_list.append(y)

        # 3.3) one hot
        x = torch.eye(D, device=device)

        graphs.append(Data(x=x, edge_index=ei, edge_weight=ew))

    # 4) stack labels
    labels = torch.stack(label_list, dim=0)
    return graphs, labels



if __name__ == '__main__':
    from torch import device

    data_root    = PROJECT_ROOT / 'Movielens_1M'
    adj_folder   = data_root / 'snapshots'
    label_folder = data_root / 'structural_groups'
    device       = device('mps' if torch.cuda.is_available() else 'cpu')

    graphs, labels = load_structural_adversarial_data(adj_folder,
                                                      label_folder,
                                                      device)

    print(f'Loaded {len(graphs)} snapshots')
    print(graphs[0].x[:5])  
    print('Example Data[0]:', graphs[0].edge_index)
    print('Labels shape:', labels.shape)