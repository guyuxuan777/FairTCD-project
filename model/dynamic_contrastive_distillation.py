# dynamic_contrastive_distillation.py

import pickle
from typing import List, Tuple

import torch
import torch.nn.functional as F

def load_contrastive_pairs(path: str) -> Tuple[List[Tuple[int,int]], List[Tuple[int,int]]]:
    """
    load contrastive pairs from a pickle file.
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['pos'], data['neg']


def dynamic_contrastive_loss(
    emb_s: torch.Tensor,
    emb_a: torch.Tensor,
    pos_pairs: List[Tuple[int,int]],
    neg_pairs: List[Tuple[int,int]],
    temperature: float = 0.1
) -> torch.Tensor:
    """
    Contrastive distillation between the structural teacher embeddings emb_s and the attribute teacher embeddings emb_a.
    emb_s and emb_a both have shape (T, N, F)

    pos_pairs: [(u, u), ...]
    neg_pairs: [(u, v), ...]

    Contrast is computed only for user nodes u that appear in pos_pairs/neg_pairs.
    """
    T, N, Fdim = emb_s.shape
    device = emb_s.device

    emb_s = F.normalize(emb_s, dim=-1)
    emb_a = F.normalize(emb_a, dim=-1)

    total_loss = 0.0
    count = 0

    for t in range(T):
        h_s = emb_s[t] 
        h_a = emb_a[t]

        for (u_pos, v_pos) in pos_pairs:

            if u_pos != v_pos:
                continue
            pos_sim = (h_s[u_pos] * h_a[u_pos]).sum() / temperature


            neg_logits = []
            for (u_neg, v_neg) in neg_pairs:
                if u_neg != u_pos:
                    continue

                neg_logits.append((h_s[u_pos] @ h_a[v_neg]) / temperature)
                neg_logits.append((h_a[u_pos] @ h_s[v_neg]) / temperature)

            if not neg_logits:
                continue

            logits = torch.stack([pos_sim] + neg_logits, dim=0).unsqueeze(0)  
            labels = torch.zeros(1, dtype=torch.long, device=device)         
            total_loss += F.cross_entropy(logits, labels)
            count += 1

    if count == 0:

        return torch.tensor(0.0, device=device)
    return total_loss / count