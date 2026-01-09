#!/usr/bin/env python3
import os
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2] 

def build_head_tail_labels_from_adj(adj: np.ndarray, top_ratio: float = 0.2) -> np.ndarray:

    N = adj.shape[0]
    # calculate degree
    degrees = adj.sum(axis=1)
    # find（degree>0）
    interacted_mask = degrees > 0
    interacted_nodes = np.where(interacted_mask)[0]
    M = len(interacted_nodes)
    print(M)
    
    if M == 0:
        return -np.ones(N, dtype=np.int64)
    
    # sort
    interacted_degrees = degrees[interacted_nodes]

    sorted_indices = np.argsort(interacted_degrees)
    
    # 4) calculate head
    k = max(int(np.ceil(M * top_ratio)), 1)
    # top k nodes as head
    head_indices_within_interacted = sorted_indices[-k:]
  
    head_nodes = interacted_nodes[head_indices_within_interacted]
    
    # 5) remain as tail
    tail_mask = np.ones(M, dtype=bool)
    tail_mask[head_indices_within_interacted] = False
    tail_indices_within_interacted = np.where(tail_mask)[0]
    tail_nodes = interacted_nodes[tail_indices_within_interacted]
    

    labels = -np.ones(N, dtype=np.int64)  
    labels[head_nodes] = 0  
    labels[tail_nodes] = 1  
    
    return labels

if __name__ == "__main__":

    data_root = PROJECT_ROOT / "Electronics"
    snap_dir  = data_root / "snapshots"
    out_dir   = data_root / "structural_groups_ratio_0.4"
    top_ratio = 0.4

    os.makedirs(out_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(snap_dir)
                   if f.startswith("snapshot_") and f.endswith(".npy"))

    for fname in files:
        adj = np.load(os.path.join(snap_dir, fname))
        assert adj.shape[0] == adj.shape[1], f"{fname}: error"

        labels = build_head_tail_labels_from_adj(adj, top_ratio)

        idx = fname.split("_")[-1].split(".")[0]
        save_path = os.path.join(out_dir, f"snapshot_{idx}_structural_groups.npy")
        np.save(save_path, labels)

        total      = labels.size
        n_interact = (labels != -1).sum()
        n_head     = (labels == 0).sum()
        n_tail     = (labels == 1).sum()
        n_none     = (labels == -1).sum()
        print(
            f"{fname}: node total number={total}, interact={n_interact}, "
            f"head={n_head}, tail={n_tail}, no interact={n_none}"
        )