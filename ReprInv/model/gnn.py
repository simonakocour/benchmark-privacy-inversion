import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool

from ReprInv.utils.positional_encoding import get_2d_positional_encoding_normalized


class SparsePixelGraphEncoder(nn.Module):
    
    def __init__(self, graph_dim: int = 64, latent_dim: int = 1024, num_heads: int = 4):
        super().__init__()
        self.graph_dim = graph_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        
        self.gnn = GATConv(graph_dim, graph_dim, heads=num_heads)
        self.output_proj = nn.Linear(graph_dim * num_heads, latent_dim)
    
    def forward(self, pixel_coords: torch.Tensor, edge_index: torch.Tensor, n_bins: int = 4) -> torch.Tensor:
        x_bins = torch.clamp((pixel_coords[:, 0] * n_bins).long(), 0, n_bins - 1)
        y_bins = torch.clamp((pixel_coords[:, 1] * n_bins).long(), 0, n_bins - 1)
        bin_indices = x_bins * n_bins + y_bins
        
        coord_emb = get_2d_positional_encoding_normalized(pixel_coords, self.graph_dim)
        h = self.gnn(coord_emb, edge_index)

        graph_emb = global_mean_pool(h, bin_indices)
        
        num_bins_sq = n_bins ** 2
        if graph_emb.shape[0] < num_bins_sq:
            output = torch.zeros(num_bins_sq, self.latent_dim, device=h.device, dtype=h.dtype)
            output[:graph_emb.shape[0]] = self.output_proj(graph_emb)
            return output
        
        return self.output_proj(graph_emb)
