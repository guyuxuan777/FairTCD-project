import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.nn.conv import LGConv
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data, Batch
from typing import List, Union, Optional
from model.structural_teacher.data_loader import load_structural_adversarial_data
import pandas as pd

class LightgcnBackbone(nn.Module):
    """
    Dynamic graph backbone based on LightGCN (LGConv). Processes a sequence of graph snapshots and outputs node-level or graph-level features.

    readout_type supports:
    - 'none': node-level output, returns [T, N, F]
    - 'mean', 'max', 'add', 'attention': graph-level output, returns [B, T, F]
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        use_batch_norm: bool = False,
        residual: bool = False,
        readout_type: str = "mean",
        concat_readout: bool = False,
    ):
        super().__init__()
        self.dropout = dropout
        self.residual = residual
        self.concat_readout = concat_readout
        self.readout_type = readout_type
        self.num_layers = num_layers

        # LGConv + linear layer 
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            out_dim = out_channels if i == num_layers - 1 else hidden_channels
            self.conv_layers.append(nn.ModuleDict({
                'agg': LGConv(),
                'lin': nn.Linear(in_dim, out_dim, bias=False)
            }))
            if use_batch_norm and i < num_layers - 1:
                self.batch_norms.append(nn.BatchNorm1d(out_dim))

        # Attention readout 
        self.gate_input_dim = out_channels
        if concat_readout:
            dims = [hidden_channels] * (num_layers - 1) + [out_channels]
            self.gate_input_dim = sum(dims)
        if readout_type == 'attention':
            self.readout = pyg_nn.aggr.AttentionalAggregation(
                gate_nn=nn.Sequential(
                    nn.Linear(self.gate_input_dim, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, 1)
                )
            )
        else:
            self.readout = None
        if concat_readout:
            self.projection = nn.Linear(self.gate_input_dim, out_channels)

    def forward(
        self,
        graph_seq: List[Union[Data, Batch]],
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        if self.readout_type == 'none':
            node_embs = []
            for graph in graph_seq:
                x, edge_index = graph.x, graph.edge_index
               
                edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
                h = x
                for i, layer in enumerate(self.conv_layers):
                    h_in = h
                    h = layer['agg'](h, edge_index)
                    h = layer['lin'](h)
                    if i < self.num_layers - 1:
                        if self.batch_norms is not None:
                            h = self.batch_norms[i](h)
                        h = nn.functional.relu(h)
                        h = nn.functional.dropout(h, p=self.dropout, training=self.training)
                    if self.residual and h.shape == h_in.shape:
                        h = h + h_in
                node_embs.append(h)
            return torch.stack(node_embs, dim=0)


        reps = []
        for graph in graph_seq:
            x, edge_index = graph.x, graph.edge_index
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
            h = x
            layer_outs = []
            for i, layer in enumerate(self.conv_layers):
                h_in = h
                h = layer['agg'](h, edge_index)
                h = layer['lin'](h)
                if i < self.num_layers - 1:
                    if self.batch_norms is not None:
                        h = self.batch_norms[i](h)
                    h = nn.functional.relu(h)
                    h = nn.functional.dropout(h, p=self.dropout, training=self.training)
                if self.residual and h.shape == h_in.shape:
                    h = h + h_in
                layer_outs.append(h)
            if self.concat_readout:
                cat_h = torch.cat(layer_outs, dim=1)
                batch_idx = getattr(graph, 'batch', torch.zeros(cat_h.size(0), dtype=torch.long, device=cat_h.device))
                g_repr = self.readout(cat_h, index=batch_idx) if self.readout_type=='attention' else self._readout(cat_h, batch_idx)
                g_repr = self.projection(g_repr)
            else:
                batch_idx = getattr(graph, 'batch', torch.zeros(h.size(0), dtype=torch.long, device=h.device))
                g_repr = self.readout(h, index=batch_idx) if self.readout_type=='attention' else self._readout(h, batch_idx)
            reps.append(g_repr)
        return torch.stack(reps, dim=1)

    def _readout(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.readout_type == 'mean':
            return pyg_nn.global_mean_pool(x, batch)
        if self.readout_type == 'max':
            return pyg_nn.global_max_pool(x, batch)
        if self.readout_type == 'add':
            return pyg_nn.global_add_pool(x, batch)
        raise ValueError(f"no support: {self.readout_type}")

class NodeHeadTailClassifier(nn.Module):
    def __init__(self, backbone: LightgcnBackbone, emb_dim: int, num_classes: int = 2):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, graph_seq: List[Data]) -> torch.Tensor:
        node_embs = self.backbone(graph_seq)
        last_emb = node_embs[-1]
        logits = self.classifier(last_emb)
        return last_emb, logits



