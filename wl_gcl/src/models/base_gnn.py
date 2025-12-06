# src/models/base_gnn.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv


class BaseGCN(nn.Module):
    """
    Simple 2-layer GCN encoder for node embeddings.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        """
        Compute node embeddings.

        Args:
            x: (N, in_dim)
            edge_index: (2, E)

        Returns:
            z: (N, out_dim)
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.conv2(x, edge_index)

        # Normalize embeddings for cosine similarity stability
        z = F.normalize(x, p=2, dim=-1)

        return z