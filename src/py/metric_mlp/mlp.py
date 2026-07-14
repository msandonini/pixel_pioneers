import torch
from torch import nn
import torch.nn.functional as F

from typing import Optional, Tuple


class MetricMLP(nn.Module):
    def __init__(self,
        embed_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.2,
        normalize_embeddings: bool = True,
        output_range: Optional[Tuple[float, float]] = None
    ):
        super().__init__()

        self.normalize_embeddings = normalize_embeddings
        self.output_range = output_range

        in_dim = embed_dim * 4

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, 1),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, pred_emb: torch.Tensor, target_emb) -> torch.Tensor:
        if self.normalize_embeddings:
            pred_emb = F.normalize(pred_emb, dim=-1)
            target_emb = F.normalize(target_emb, dim=-1)

        x = torch.cat([
            pred_emb,
            target_emb,
            torch.abs(pred_emb - target_emb),
            pred_emb * target_emb,
        ], dim=-1)

        score = self.net(x).squeeze(-1) # (batch, )

        if self.output_range is not None:
            low, high = self.output_range
            score = low + (high - low) * torch.sigmoid(score)

        return score # Similarity -> higher is better
