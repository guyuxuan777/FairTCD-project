import numpy as np
import torch
from collections import defaultdict

def calculate_hr_at_k(user_embeddings: torch.Tensor,
                      item_embeddings: torch.Tensor,
                      graph_data,
                      k: int = 10,
                      return_details: bool = False):


    ue = user_embeddings.cpu().numpy()      
    ie = item_embeddings.cpu().numpy()      
    num_users = ue.shape[0]
    num_items = ie.shape[0]


    edge = graph_data.edge_index.cpu().numpy()
    user_true = defaultdict(set)
    for u_glob, i_glob in zip(edge[0], edge[1]):
        if u_glob < num_users:
            local_i = i_glob - num_users
            if 0 <= local_i < num_items:
                user_true[u_glob].add(local_i)


    hr_count = 0
    details = []
    for u in range(num_users):
        true_items = user_true.get(u)
        if not true_items:
            details.append({
                'user_id': u,
                'hit': False,
                'top_k': [],
                'true': [],
                'hit_items': []
            })
            continue

        # cosione similarity 

        scores = ue[u].dot(ie.T)
        # Top-K 
        topk = np.argsort(-scores)[:k].tolist()
        #hit items
        hit_items = [i for i in topk if i in true_items]
        hit = len(hit_items) > 0
        if hit:
            hr_count += 1

        details.append({
            'user_id': u,
            'hit': hit,
            'top_k': topk,
            'true': list(true_items),
            'hit_items': hit_items
        })

    # 4) average HR
    denom = len(user_true)
    hr_at_k = hr_count / denom if denom > 0 else 0.0

    if return_details:
        return hr_at_k, details
    else:
        return hr_at_k
