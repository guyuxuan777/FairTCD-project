import numpy as np
import torch
from collections import defaultdict

def calculate_ndcg_at_k(
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    graph_data,
    k: int = 10
) -> float:

    ue = user_embeddings.cpu().numpy()    
    ie = item_embeddings.cpu().numpy()    
    num_users = ue.shape[0]
    num_items = ie.shape[0]

    # 2) item ID
    edge = graph_data.edge_index.cpu().numpy()
    user_true = defaultdict(set)
    for u_glob, i_glob in zip(edge[0], edge[1]):
        if u_glob < num_users:
            local_i = i_glob - num_users
            if 0 <= local_i < num_items:
                user_true[u_glob].add(local_i)

    # 3) NDCG
    ndcgs = []
    for u, true_items in user_true.items():
        if len(true_items) == 0:
            continue
        # rate and sort
        scores = ue[u].dot(ie.T)           
        ranking = np.argsort(-scores)      

        # DCG@K
        dcg = 0.0
        for rank_idx, item in enumerate(ranking[:k]):
            if item in true_items:
                dcg += 1.0 / np.log2(rank_idx + 2)

        # IDCG@K
        max_rel = min(len(true_items), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(max_rel))

        ndcgs.append(dcg / idcg)

    return float(np.mean(ndcgs)) if ndcgs else 0.0
