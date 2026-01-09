import torch
import torch.nn as nn

class MLPDiscriminator(nn.Module):
    """
    Discriminator for predicting sensitive attributes of the graph (e.g., gender, race).
    Input: graph representation from GNN [batch_size, feat_dim]
    Output: classification logits for the sensitive attribute [batch_size, num_classes]
    """
    def __init__(self, in_dim: int, hidden_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, graph_repr: torch.Tensor) -> torch.Tensor:
        # graph_repr: [B, F]
        return self.net(graph_repr)
    