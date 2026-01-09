import torch


def temporal_decay_aggregate(all_embs: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Aggregate temporal embeddings with an exponential decay factor.

    Parameters
    ----------
    all_embs : torch.Tensor, shape (T, N, F)
        Node embeddings from T snapshots.
    gamma : float
        Decay factor, 0 < gamma <= 1.

    Returns
    -------
    torch.Tensor, shape (N, F)
        Aggregated node features.
    """
    T, N, F = all_embs.shape

    # gamma^(T-1-t)
    exponents = torch.arange(T-1, -1, -1, device=all_embs.device, dtype=all_embs.dtype)
    weights = gamma ** exponents
    weights = weights / weights.sum()

    agg = (all_embs * weights.view(T, 1, 1)).sum(dim=0)
    return agg
