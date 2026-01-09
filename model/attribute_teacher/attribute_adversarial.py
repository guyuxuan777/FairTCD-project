# model/attribute_adversarial.py
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from model.backbone.gcn_backbone import GNNBackbone
from model.discriminator.mlp_discriminator import MLPDiscriminator
from typing import List, Union, Optional

class AttributeAdversarialModel(nn.Module):

    def __init__(
        self,
        # GNNBackbone parameter
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
        residual: bool = True,
        # Discriminator parameter
        disc_hidden_dim: int = 64,
        disc_num_classes: int = 2,
    ):
        super().__init__()
        # Backbone
        self.backbone = GNNBackbone(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            residual=residual,
            readout_type='none',      
            concat_readout=False
        )
        # Discriminator
        self.discriminator = MLPDiscriminator(
            in_dim=out_channels,
            hidden_dim=disc_hidden_dim,
            num_classes=disc_num_classes
        )

    def forward(
        self,
        graph_seq: List[Union[Data, Batch]],
        edge_weight: Optional[torch.Tensor] = None
    ):

        # 1) GNN node embedding
        node_emb = self.backbone(graph_seq, edge_weight)  

        # 2) Discriminator logits
        logits = self.discriminator(node_emb)             

        return node_emb, logits