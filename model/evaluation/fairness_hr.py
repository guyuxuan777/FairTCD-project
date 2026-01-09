import numpy as np
import torch
from collections import defaultdict

def calculate_hr_at_k_fair(
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    graph_data,
    target_user_ids,
    k: int = 10
) -> float:

    # 1) calculate hr@k
    ue = user_embeddings.cpu().numpy()   
    ie = item_embeddings.cpu().numpy()  
    num_users = ue.shape[0]
    num_items = ie.shape[0]

    edge = graph_data.edge_index.cpu().numpy()
    user_true = defaultdict(set)  
    for u_glob, i_glob in zip(edge[0], edge[1]):
        if u_glob < num_users and u_glob in target_user_ids:
            local_i = int(i_glob) - num_users
            if 0 <= local_i < num_items:
                user_true[int(u_glob)].add(local_i)

    # 3) rating for each user
    hr_count = 0
    valid = 0
    for u in target_user_ids:
        true_items = user_true.get(u, None)
        if not true_items:
            continue
        valid += 1

        sims = ue[u].dot(ie.T) / (
            np.linalg.norm(ue[u]) * np.linalg.norm(ie, axis=1)
        )
        topk = np.argsort(-sims)[:k]
        if any(j in true_items for j in topk):
            hr_count += 1

    return hr_count / valid if valid > 0 else 0.0


def load_and_split_active_users(
    snapshot_path: str,
    graph_data,
    user_embeddings: torch.Tensor
):

    num_users = user_embeddings.shape[0]
    edge = graph_data.edge_index.cpu().numpy()

    active = [int(u) for u in np.unique(edge[0]) if u < num_users]

    groups = np.load(snapshot_path, allow_pickle=True)
    label0, label1 = [], []
    for u in active:
        if u < len(groups):
            if groups[u] == 0:
                label0.append(u)
            elif groups[u] == 1:
                label1.append(u)

    return active, label0, label1
