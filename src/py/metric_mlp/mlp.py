import torch
from torch import nn


class MetricMLP(nn.Module):
    def __init__(self,
        embed_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.2
    ):
        super().__init__()

        in_dim = embed_dim * 4

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, pred_emb: torch.Tensor, target_emb) -> torch.Tensor:
        x = torch.cat([
            pred_emb,
            target_emb,
            torch.abs(pred_emb - target_emb),
            pred_emb * target_emb,
        ], dim=-1)

        score = self.net(x)

        return torch.sigmoid(score) # [0, 1] = Similarity -> higher is better
