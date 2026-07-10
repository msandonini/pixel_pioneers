import torch
from torch import nn


class FusionMLP(nn.Module):
    def __init__(self,
        input_dims,
        fused_dim: int = 256,
        hidden_dim: int = 512,
        out_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, fused_dim),
                nn.LayerNorm(fused_dim),
                nn.ReLU()
            ) for dim in input_dims
        ])

        fusion_in_dims = fused_dim * len(input_dims)

        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dims, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, embeddings):
        # embeddings = [ [B, d1], [B, d2], ... ]
        projected = [ proj(e) for proj, e in zip(self.projections, embeddings) ]

        x = torch.cat(projected, dim=-1)

        return self.fusion(x)
