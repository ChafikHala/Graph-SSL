import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv

class GINEncoder(nn.Module):
    def __init__(self,input_dim, hidden_dim, output_dim, num_layers=3):
        super(GINEncoder, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            dim_in = input_dim if i == 0 else hidden_dim
            dim_out = output_dim if i == num_layers - 1 else hidden_dim
            
            mlp = nn.Sequential(
                nn.Linear(dim_in, dim_out),
                nn.BatchNorm1d(dim_out),
                nn.ReLU(),
                nn.Linear(dim_out, dim_out)
            )
            self.layers.append(GINConv(mlp))
            if i < num_layers - 1:
                self.batch_norms.append(nn.BatchNorm1d(dim_out))
    
    def forward(self,x,edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = self.batch_norms[i](x)
                x = F.relu(x)
        return x